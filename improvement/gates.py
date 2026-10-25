"""Trusted promotion rules. Candidate Critic verdicts are never inputs here."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class PromotionPolicy:
    max_call_ratio: float = 1.25
    min_call_saving: int = 1

    def __post_init__(self):
        if not math.isfinite(self.max_call_ratio) or self.max_call_ratio < 1:
            raise ValueError("max_call_ratio must be finite and >= 1")
        if type(self.min_call_saving) is not int or self.min_call_saving < 1:
            raise ValueError("min_call_saving must be a positive integer")

    def to_dict(self):
        return asdict(self)


def checked_rows(report: dict, *, profile_id: str | None = None, split: str | None = None) -> list[dict]:
    if profile_id is not None and report.get("profile_id") != profile_id:
        raise ValueError("Evaluation belongs to a different candidate")
    if split is not None and report.get("split") != split:
        raise ValueError("Evaluation belongs to a different split")
    for key in ("scorer_fingerprint", "dataset_fingerprint", "config_fingerprint", "profile_id"):
        if not isinstance(report.get(key), str) or not report[key]:
            raise ValueError(f"Missing trusted evaluation identity: {key}")
    rows = report.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("An empty evaluation cannot approve a version")
    keys, trial_ids = set(), set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("task_id"), str) or not row["task_id"]:
            raise ValueError("Invalid task identity")
        index = row.get("trial_index")
        if type(index) is not int or index < 0:
            raise ValueError("Invalid trial index")
        key = row["task_id"], index
        if key in keys or not row.get("trial_id") or row["trial_id"] in trial_ids:
            raise ValueError("Duplicate or missing trial")
        keys.add(key)
        trial_ids.add(row["trial_id"])
        if row.get("profile_id") != report["profile_id"]:
            raise ValueError("Mixed candidate evaluations")
        if type(row.get("passed")) is not bool or type(row.get("completed")) is not bool:
            raise ValueError("Pass/completion must be independently graded booleans")
        if row["passed"] and (not row["completed"] or row.get("error")):
            raise ValueError("Failed execution cannot be a pass")
        for field in ("llm_calls", "tool_calls"):
            if type(row.get(field)) is not int or row[field] < 0:
                raise ValueError(f"Invalid accounting: {field}")
        seconds = row.get("seconds")
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Invalid runtime accounting")
    counts = defaultdict(set)
    for row in rows:
        counts[row["task_id"]].add(row["trial_index"])
    repeat_sets = list(counts.values())
    if any(value != set(range(len(value))) or value != repeat_sets[0] for value in repeat_sets):
        raise ValueError("Missing repeated trials")
    return rows


def summary(report: dict) -> dict:
    rows = checked_rows(report)
    task_passes = defaultdict(list)
    for row in rows:
        task_passes[row["task_id"]].append(row["passed"])
    passed = sum(row["passed"] for row in rows)
    return {"total": len(rows), "passed": passed,
            "correctness_rate": passed / len(rows),
            "completion_rate": sum(row["completed"] for row in rows) / len(rows),
            "all_trials_pass_rate": sum(all(v) for v in task_passes.values()) / len(task_passes),
            "errors": sum(bool(row.get("error")) for row in rows),
            "llm_calls": sum(row["llm_calls"] for row in rows),
            "tool_calls": sum(row["tool_calls"] for row in rows),
            "seconds": sum(row["seconds"] for row in rows)}


def compare(baseline: dict, candidate: dict, *, policy: PromotionPolicy,
            require_improvement: bool) -> dict:
    """Same inputs/model/budget; no per-task regression, then quality/cost gain."""
    left, right = checked_rows(baseline), checked_rows(candidate)
    reasons = []
    for key in ("split", "scorer_fingerprint", "dataset_fingerprint", "config_fingerprint"):
        if baseline.get(key) != candidate.get(key):
            reasons.append(f"Evaluation contract changed: {key}")
    if {(r["task_id"], r["trial_index"]) for r in left} != {(r["task_id"], r["trial_index"]) for r in right}:
        reasons.append("Task/trial coverage changed")
    per_task = defaultdict(lambda: [0, 0])
    for side, rows in enumerate((left, right)):
        for row in rows:
            per_task[row["task_id"]][side] += row["passed"]
    for task, (old, new) in per_task.items():
        if new < old:
            reasons.append(f"Task regressed: {task}")
    before, after = summary(baseline), summary(candidate)
    if after["errors"] > before["errors"]:
        reasons.append("Execution errors increased")
    if after["llm_calls"] > before["llm_calls"] * policy.max_call_ratio:
        reasons.append("Model-call cost guard exceeded")
    quality_gain = after["passed"] > before["passed"]
    efficiency_gain = (after["passed"] == before["passed"] and
                       before["llm_calls"] - after["llm_calls"] >= policy.min_call_saving)
    if require_improvement and not (quality_gain or efficiency_gain):
        reasons.append("No independently measured improvement")
    return {"accepted": not reasons, "reasons": reasons,
            "baseline": before, "candidate": after, "policy": policy.to_dict()}


def development_feedback(report: dict) -> list[dict]:
    """Allowlisted diagnostics; no references, expected values or held-out rows."""
    if report.get("split") != "development":
        raise ValueError("Only development diagnostics may reach the improver")
    result = []
    for row in checked_rows(report):
        raw = row.get("feedback") or {}
        result.append({"task_id": row["task_id"], "trial_id": row["trial_id"],
                       "passed": row["passed"], "error": str(row.get("error") or "")[:400],
                       "feedback": {key: str(raw[key])[:1600] for key in
                                    ("reason", "error_kind", "critic_feedback", "report_excerpt") if key in raw}})
    return result
