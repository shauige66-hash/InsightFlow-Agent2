"""PostgreSQL owner-scoped Run registry and fenced execution leases."""

from __future__ import annotations

import os
import time
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    select,
    text,
    or_,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert

from services.run_store import (
    STALE_HEARTBEAT_SECONDS,
    RUN_STATUSES,
    RunRecord,
    _process_alive,
)

from services.run_identity import RunConflict, LeaseLost, current_owner, fingerprint, lease_token_for

_metadata = MetaData()

runs = Table(
    "runs",
    _metadata,
    Column("run_id", String(128), primary_key=True),
    Column("owner", Text, nullable=False, server_default="local-demo"),
    Column("request_hash", Text),
    Column("request_payload", JSON, nullable=False, server_default="{}"),
    Column("lease_token", Text),
    Column("lease_expires_at", Float),
    Column("status", String(32), nullable=False),
    Column("data_sources", JSON, nullable=False, default=list),
    Column("analysis_focus", Text),
    Column("created_at", Float, nullable=False, default=0.0),
    Column("updated_at", Float, nullable=False, default=0.0),
    Column("error", Text),
    Column("pending_approval", JSON),
    Column("approvals", JSON, nullable=False, default=list),
    Column("owner_pid", Integer),
    Column("heartbeat_at", Float),
    Column("sql_backend", String(32), nullable=False, default="direct"),
    Column("require_plan_approval", Boolean, nullable=False, default=False),
    Column("require_report_approval", Boolean, nullable=False, default=False),
)


def ensure_schema(url: str) -> None:
    import storage

    engine = storage.get_engine(url)
    with engine.begin() as connection:
        connection.execute(text("SELECT pg_advisory_xact_lock(7300161)"))
        _metadata.create_all(connection, checkfirst=True)
        for ddl in ("owner TEXT NOT NULL DEFAULT 'local-demo'", "request_hash TEXT", "request_payload JSON NOT NULL DEFAULT '{}'", "lease_token TEXT", "lease_expires_at DOUBLE PRECISION"):
            connection.execute(text("ALTER TABLE runs ADD COLUMN IF NOT EXISTS " + ddl))


class PostgresRunStore:
    """Owner-scoped insert-only Run registry with shared-database leases."""
    def __init__(self, url):
        import storage
        self._url = url
        ensure_schema(url)
        self._engine = storage.get_engine(url)

    def _session(self):
        import storage
        return storage.session_scope(self._url)

    def _lease_now(self):
        with self._engine.connect() as connection:
            return float(connection.scalar(text("SELECT EXTRACT(EPOCH FROM clock_timestamp())")))

    @staticmethod
    def _normalize_backend(value):
        from core.runtime import SqlBackend
        return SqlBackend(value).value

    @staticmethod
    def _row_to_record(row):
        values = dict(row._mapping)
        values["data_sources"] = list(values.get("data_sources") or [])
        values["approvals"] = list(values.get("approvals") or [])
        values["request_payload"] = dict(values.get("request_payload") or {})
        return RunRecord(**values)

    def create(self, run_id, *, data_sources, analysis_focus=None, status="pending", sql_backend="direct",
               require_plan_approval=False, require_report_approval=False, owner=None, request_payload=None):
        owner = current_owner(owner)
        payload = request_payload if request_payload is not None else {
            "data_sources": data_sources,"analysis_focus": analysis_focus,"sql_backend": self._normalize_backend(sql_backend),
            "require_plan_approval": bool(require_plan_approval),"require_report_approval": bool(require_report_approval),
        }
        digest = fingerprint(payload)
        now = self._lease_now()
        with self._session() as session:
            session.execute(pg_insert(runs).values(run_id=run_id,owner=owner,status=status,data_sources=list(data_sources),
                analysis_focus=analysis_focus,created_at=now,updated_at=now,approvals=[],sql_backend=self._normalize_backend(sql_backend),
                require_plan_approval=require_plan_approval,require_report_approval=require_report_approval,
                request_hash=digest,request_payload=payload).on_conflict_do_nothing(index_elements=[runs.c.run_id]))
            row = session.execute(select(runs).where(runs.c.run_id == run_id)).first()
            if row._mapping["owner"] != owner or row._mapping["request_hash"] != digest:
                raise RunConflict("run_id 已存在且不属于同一请求")
            return self._row_to_record(row)

    def get(self, run_id, *, owner=None):
        with self._session() as session:
            row = session.execute(select(runs).where(runs.c.run_id==run_id,runs.c.owner==current_owner(owner))).first()
        return self._row_to_record(row) if row else None

    def list(self, *, limit=50, owner=None):
        with self._session() as session:
            rows = session.execute(select(runs).where(runs.c.owner==current_owner(owner)).order_by(runs.c.updated_at.desc()).limit(limit)).all()
        return [self._row_to_record(row) for row in rows]

    def get_sql_backend(self, run_id):
        record = self.get(run_id)
        return record.sql_backend if record else "direct"

    def set_sql_backend(self, run_id, sql_backend):
        with self._session() as session:
            session.execute(runs.update().where(runs.c.run_id==run_id,runs.c.owner==current_owner()).values(sql_backend=self._normalize_backend(sql_backend),updated_at=time.time()))

    def claim(self, run_id, token, *, owner=None, resume=False, ttl=STALE_HEARTBEAT_SECONDS):
        now = self._lease_now()
        allowed = ("waiting_approval","interrupted","failed","running","queued") if resume else ("pending","queued")
        with self._session() as session:
            result = session.execute(runs.update().where(runs.c.run_id==run_id,runs.c.owner==current_owner(owner),runs.c.status.in_(allowed),
                or_(runs.c.lease_token.is_(None),runs.c.lease_expires_at <= now)).values(status="running",lease_token=token,lease_expires_at=now+ttl,heartbeat_at=now,owner_pid=os.getpid(),updated_at=now))
            return result.rowcount == 1

    def assert_lease(self, run_id, token, *, owner=None):
        record = self.get(run_id,owner=owner)
        if record is None or record.lease_token != token or not record.lease_expires_at or record.lease_expires_at <= self._lease_now():
            raise LeaseLost("运行执行权已失效，拒绝继续提交")

    def touch(self, run_id, *, token=None, owner=None):
        token = token or lease_token_for(run_id)
        now = self._lease_now()
        conditions = [runs.c.run_id==run_id,runs.c.owner==current_owner(owner)]
        values = dict(heartbeat_at=now,updated_at=now)
        if token:
            conditions += [runs.c.lease_token==token,runs.c.lease_expires_at>now]
            values["lease_expires_at"] = now+STALE_HEARTBEAT_SECONDS
        else:
            conditions += [runs.c.lease_token.is_(None)]
        with self._session() as session:
            result = session.execute(runs.update().where(*conditions).values(**values))
            if token and result.rowcount != 1:
                raise LeaseLost("运行租约续期失败")

    def update_status(self, run_id, status, *, error=None,pending_approval=None,owner=None,token=None):
        if status not in RUN_STATUSES:
            raise ValueError(f"未知的运行状态：{status}")
        token = token or lease_token_for(run_id)
        now = self._lease_now()
        conditions=[runs.c.run_id==run_id,runs.c.owner==current_owner(owner)]
        conditions += [runs.c.lease_token==token,runs.c.lease_expires_at>now] if token else [or_(runs.c.lease_token.is_(None),runs.c.lease_expires_at<=now)]
        values=dict(status=status,updated_at=now,error=error,pending_approval=pending_approval)
        if status != "running":
            values.update(lease_token=None,lease_expires_at=None)
        elif not token:
            values.update(owner_pid=os.getpid(),heartbeat_at=now)
        with self._session() as session:
            result = session.execute(runs.update().where(*conditions).values(**values))
            if token and result.rowcount != 1:
                raise LeaseLost("旧执行者不得修改运行状态")

    def is_stale_running(self, run_id):
        record=self.get(run_id)
        if record is None or record.status != "running":
            return False
        if record.lease_token:
            return not record.lease_expires_at or record.lease_expires_at <= self._lease_now()
        return record.heartbeat_at is None or self._lease_now()-record.heartbeat_at > STALE_HEARTBEAT_SECONDS

    def record_approval(self, run_id, approval):
        with self._session() as session:
            row=session.execute(select(runs).where(runs.c.run_id==run_id,runs.c.owner==current_owner()).with_for_update()).first()
            if row is None:
                raise KeyError("未找到运行")
            approvals=list(row._mapping["approvals"] or [])
            existing=next((item for item in approvals if item.get("approval_id")==approval.get("approval_id")),None)
            if existing and existing.get("decision") != approval.get("decision"):
                raise RunConflict("同一审批已经提交不同决定")
            if existing is None:
                approvals.append(approval)
            session.execute(runs.update().where(runs.c.run_id==run_id,runs.c.owner==current_owner()).values(approvals=approvals,updated_at=time.time()))
            return existing or approval

__all__=["PostgresRunStore","ensure_schema"]
