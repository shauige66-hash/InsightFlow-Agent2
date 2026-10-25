"""Host-owned immutable profiles and transactional activation/rollback history.

This is a local trusted-control API, never a model tool or a public HTTP route.
Candidates contain text only and have no registry/evaluator write capability.
"""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import time
from uuid import uuid4

from improvement.gates import PromotionPolicy, checked_rows, compare, digest
from improvement.profiles import CandidateProfile


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def load_active_profile(root: str | Path) -> CandidateProfile:
    """Read an explicitly configured registry without initializing or mutating it."""
    path = Path(root).resolve() / "versions.sqlite3"
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=30) as db:
        row = db.execute("SELECT p.id,p.payload FROM active a JOIN profiles p ON p.id=a.id WHERE a.singleton=1").fetchone()
    if row is None:
        raise ValueError("Registry has no accepted active profile")
    profile = CandidateProfile.from_dict(json.loads(row[1]))
    if profile.profile_id != row[0]:
        raise ValueError("Active profile integrity failure")
    return profile


class VersionRegistry:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "versions.sqlite3"
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS profiles(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS active(singleton INTEGER PRIMARY KEY CHECK(singleton=1), id TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS campaigns(id TEXT PRIMARY KEY, base_id TEXT NOT NULL,
                  candidate_id TEXT, status TEXT NOT NULL, holdout_used INTEGER NOT NULL DEFAULT 0,
                  policy TEXT NOT NULL, result TEXT);
                CREATE TABLE IF NOT EXISTS reports(campaign TEXT NOT NULL, role TEXT NOT NULL,
                  payload TEXT NOT NULL, hash TEXT NOT NULL, PRIMARY KEY(campaign,role));
                CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT,
                  at REAL NOT NULL, kind TEXT NOT NULL, from_id TEXT, to_id TEXT, evidence TEXT NOT NULL);
            """)

    def _connect(self):
        return sqlite3.connect(self.path, timeout=30)

    def register(self, profile: CandidateProfile):
        profile = CandidateProfile.from_dict(profile.to_dict())
        with self._connect() as db:
            db.execute("INSERT OR IGNORE INTO profiles VALUES (?,?)", (profile.profile_id, _json(profile.to_dict())))
        return profile.profile_id

    def get(self, profile_id: str) -> CandidateProfile:
        with self._connect() as db:
            row = db.execute("SELECT payload FROM profiles WHERE id=?", (profile_id,)).fetchone()
        if row is None:
            raise ValueError("Unknown version")
        profile = CandidateProfile.from_dict(json.loads(row[0]))
        if profile.profile_id != profile_id:
            raise ValueError("Stored profile integrity failure")
        return profile

    def initialize(self, profile: CandidateProfile | None = None) -> CandidateProfile:
        profile = profile or CandidateProfile()
        self.register(profile)
        with self._connect() as db:
            db.execute("INSERT OR IGNORE INTO active VALUES (1,?)", (profile.profile_id,))
        return self.current()

    def current(self) -> CandidateProfile:
        with self._connect() as db:
            row = db.execute("SELECT id FROM active WHERE singleton=1").fetchone()
        if row is None:
            raise ValueError("Registry is not initialized")
        return self.get(row[0])

    def start_campaign(self, policy: PromotionPolicy) -> str:
        base = self.current()
        campaign_id = uuid4().hex
        with self._connect() as db:
            db.execute("INSERT INTO campaigns(id,base_id,status,policy) VALUES (?,?,?,?)",
                       (campaign_id, base.profile_id, "searching", _json(policy.to_dict())))
        return campaign_id

    def record(self, campaign_id: str, role: str, report: dict):
        checked_rows(report)
        # The host controller records reports; model proposals cannot submit them.
        with self._connect() as db:
            row = db.execute("SELECT status FROM campaigns WHERE id=?", (campaign_id,)).fetchone()
            if row is None or row[0] not in {"searching", "testing"}:
                raise ValueError("Campaign is not accepting evaluations")
            db.execute("INSERT INTO reports VALUES (?,?,?,?)",
                       (campaign_id, role, _json(report), digest(report)))

    def lock_for_holdout(self, campaign_id: str, profile: CandidateProfile):
        self.register(profile)
        with self._connect() as db:
            changed = db.execute("UPDATE campaigns SET candidate_id=?,status='testing',holdout_used=1 "
                                 "WHERE id=? AND status='searching' AND holdout_used=0",
                                 (profile.profile_id, campaign_id)).rowcount
            if changed != 1:
                raise ValueError("Final candidate is already locked or holdout was consumed")

    def finish(self, campaign_id: str) -> dict:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT base_id,candidate_id,status,policy FROM campaigns WHERE id=?",
                             (campaign_id,)).fetchone()
            if row is None or row[2] != "testing":
                raise ValueError("Campaign has no locked final candidate")
            reports = {}
            for role, payload, fingerprint in db.execute("SELECT role,payload,hash FROM reports WHERE campaign=?", (campaign_id,)):
                value = json.loads(payload)
                if digest(value) != fingerprint:
                    raise ValueError("Evaluation receipt integrity failure")
                reports[role] = value
            policy = PromotionPolicy(**json.loads(row[3]))
            for role, pid, split in (("baseline_validation", row[0], "validation"),
                                     ("candidate_validation", row[1], "validation"),
                                     ("baseline_holdout", row[0], "holdout"),
                                     ("candidate_holdout", row[1], "holdout")):
                if role not in reports:
                    raise ValueError(f"Missing independent evaluation: {role}")
                checked_rows(reports[role], profile_id=pid, split=split)
            validation = compare(reports["baseline_validation"], reports["candidate_validation"],
                                 policy=policy, require_improvement=True)
            holdout = compare(reports["baseline_holdout"], reports["candidate_holdout"],
                              policy=policy, require_improvement=False)
            contracts = {(r["scorer_fingerprint"], r["config_fingerprint"]) for r in reports.values()}
            accepted = validation["accepted"] and holdout["accepted"] and len(contracts) == 1
            result = {"campaign_id": campaign_id, "accepted": accepted, "base_id": row[0],
                      "candidate_id": row[1], "validation": validation, "holdout": holdout,
                      "receipt_hashes": {k: digest(v) for k, v in reports.items()}}
            if len(contracts) != 1:
                result["reason"] = "Scorer/model/budget changed within the campaign"
            if accepted:
                changed = db.execute("UPDATE active SET id=? WHERE singleton=1 AND id=?", (row[1], row[0])).rowcount
                if changed != 1:
                    raise ValueError("Active version changed concurrently; refusing stale promotion")
                db.execute("INSERT INTO events(at,kind,from_id,to_id,evidence) VALUES (?,?,?,?,?)",
                           (time.time(), "promotion", row[0], row[1], _json(result)))
            db.execute("UPDATE campaigns SET status=?,result=? WHERE id=?",
                       ("promoted" if accepted else "rejected", _json(result), campaign_id))
        return result

    def close_search(self, campaign_id: str, reason: str):
        with self._connect() as db:
            db.execute("UPDATE campaigns SET status='rejected',result=? WHERE id=? AND status='searching'",
                       (_json({"accepted": False, "reason": reason}), campaign_id))

    def abort(self, campaign_id: str, error_type: str):
        with self._connect() as db:
            db.execute("UPDATE campaigns SET status='failed',result=? WHERE id=? AND status IN ('searching','testing')",
                       (_json({"accepted": False, "error_type": error_type}), campaign_id))

    def rollback(self, target_id: str) -> CandidateProfile:
        target = self.get(target_id)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT id FROM active WHERE singleton=1").fetchone()[0]
            known = db.execute("SELECT 1 FROM events WHERE kind='promotion' AND (from_id=? OR to_id=?)",
                               (target_id, target_id)).fetchone()
            if known is None:
                raise ValueError("Rollback target has no accepted version history")
            db.execute("UPDATE active SET id=? WHERE singleton=1", (target_id,))
            db.execute("INSERT INTO events(at,kind,from_id,to_id,evidence) VALUES (?,?,?,?,?)",
                       (time.time(), "rollback", old, target_id, "{}"))
        return target

    def status(self) -> dict:
        with self._connect() as db:
            events = [{"seq": r[0], "kind": r[1], "from_id": r[2], "to_id": r[3]}
                      for r in db.execute("SELECT seq,kind,from_id,to_id FROM events ORDER BY seq")]
            campaigns = [{"id": r[0], "status": r[1], "holdout_used": bool(r[2])}
                         for r in db.execute("SELECT id,status,holdout_used FROM campaigns ORDER BY rowid")]
        return {"active_profile_id": self.current().profile_id, "events": events, "campaigns": campaigns}
