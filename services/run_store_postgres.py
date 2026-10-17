"""PostgreSQL 实现的 Run 登记。

与 SQLite 版 `RunStore` 保持同一组方法与语义，差别只在存储：

- `create` 是 upsert（`ON CONFLICT (run_id) DO UPDATE`），重复登记同一 Run
  不会失败，也不会把 `created_at` 改掉；
- `update_status` 与 `touch` 仍然是"最后写入者生效"，不引入额外的锁；
- `is_stale_running` 的进程判断只在同一台主机上有意义——`owner_pid` 是
  本机 PID，跨主机部署时应以心跳为准，接口返回的仍是启发式结论。

运行时状态只有一处权威：Run 的执行状态以这里（或 SQLite 版）为准，
会话库里的 `conversation_turns.status` 只是展示用的同步值。
"""

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
)
from sqlalchemy.dialects.postgresql import insert as pg_insert

from services.run_store import (
    STALE_HEARTBEAT_SECONDS,
    RUN_STATUSES,
    RunRecord,
    _process_alive,
)

_metadata = MetaData()

runs = Table(
    "runs",
    _metadata,
    Column("run_id", String(128), primary_key=True),
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
    _metadata.create_all(engine, checkfirst=True)


class PostgresRunStore:
    """与 `services.run_store.RunStore` 接口一致的 PostgreSQL 版本。"""

    def __init__(self, url: str) -> None:
        import storage

        self._url = url
        ensure_schema(url)
        self._engine = storage.get_engine(url)

    # -- 内部 -------------------------------------------------------------

    def _session(self):
        import storage

        return storage.session_scope(self._url)

    @staticmethod
    def _normalize_backend(sql_backend: str) -> str:
        from core.runtime import SqlBackend

        try:
            return SqlBackend(sql_backend).value
        except ValueError as error:
            raise ValueError(f"不支持的 SQL 后端模式：{sql_backend}") from error

    @staticmethod
    def _row_to_record(row: Any) -> RunRecord:
        mapping = row._mapping
        return RunRecord(
            run_id=mapping["run_id"],
            status=mapping["status"],
            data_sources=list(mapping["data_sources"] or []),
            analysis_focus=mapping["analysis_focus"],
            created_at=float(mapping["created_at"] or 0.0),
            updated_at=float(mapping["updated_at"] or 0.0),
            error=mapping["error"],
            pending_approval=mapping["pending_approval"],
            approvals=list(mapping["approvals"] or []),
            owner_pid=mapping["owner_pid"],
            heartbeat_at=mapping["heartbeat_at"],
            sql_backend=mapping["sql_backend"] or "direct",
            require_plan_approval=bool(mapping["require_plan_approval"]),
            require_report_approval=bool(mapping["require_report_approval"]),
        )

    # -- 写 ---------------------------------------------------------------

    def create(
        self,
        run_id: str,
        *,
        data_sources: list[str],
        analysis_focus: str | None = None,
        status: str = "pending",
        sql_backend: str = "direct",
        require_plan_approval: bool = False,
        require_report_approval: bool = False,
    ) -> RunRecord:
        now = time.time()
        backend = self._normalize_backend(sql_backend)

        with self._session() as session:
            statement = pg_insert(runs).values(
                run_id=run_id,
                status=status,
                data_sources=list(data_sources),
                analysis_focus=analysis_focus,
                created_at=now,
                updated_at=now,
                error=None,
                pending_approval=None,
                approvals=[],
                sql_backend=backend,
                require_plan_approval=require_plan_approval,
                require_report_approval=require_report_approval,
            )
            # 重复登记同一 Run：刷新可改字段，保留 created_at
            session.execute(
                statement.on_conflict_do_update(
                    index_elements=[runs.c.run_id],
                    set_={
                        "status": statement.excluded.status,
                        "data_sources": statement.excluded.data_sources,
                        "analysis_focus": statement.excluded.analysis_focus,
                        "updated_at": statement.excluded.updated_at,
                        "error": None,
                        "pending_approval": None,
                        "approvals": [],
                        "sql_backend": statement.excluded.sql_backend,
                        "require_plan_approval": (
                            statement.excluded.require_plan_approval
                        ),
                        "require_report_approval": (
                            statement.excluded.require_report_approval
                        ),
                    },
                )
            )
        return self.get(run_id)

    def set_sql_backend(self, run_id: str, sql_backend: str) -> None:
        backend = self._normalize_backend(sql_backend)
        with self._session() as session:
            session.execute(
                runs.update()
                .where(runs.c.run_id == run_id)
                .values(sql_backend=backend, updated_at=time.time())
            )

    def update_status(
        self,
        run_id: str,
        status: str,
        *,
        error: str | None = None,
        pending_approval: dict[str, Any] | None = None,
    ) -> None:
        if status not in RUN_STATUSES:
            raise ValueError(f"未知的运行状态：{status}")

        owner = os.getpid() if status == "running" else None
        heartbeat = time.time() if status == "running" else None

        with self._session() as session:
            session.execute(
                runs.update()
                .where(runs.c.run_id == run_id)
                .values(
                    status=status,
                    updated_at=time.time(),
                    error=error,
                    pending_approval=pending_approval,
                    owner_pid=owner if owner is not None else runs.c.owner_pid,
                    heartbeat_at=(
                        heartbeat if heartbeat is not None else runs.c.heartbeat_at
                    ),
                )
            )

    def touch(self, run_id: str) -> None:
        now = time.time()
        with self._session() as session:
            session.execute(
                runs.update()
                .where(runs.c.run_id == run_id)
                .values(heartbeat_at=now, updated_at=now, owner_pid=os.getpid())
            )

    def record_approval(self, run_id: str, approval: dict[str, Any]) -> None:
        record = self.get(run_id)
        if record is None:
            raise KeyError(f"未登记的 run：{run_id}")

        approvals = list(record.approvals)
        approvals.append(approval)

        with self._session() as session:
            session.execute(
                runs.update()
                .where(runs.c.run_id == run_id)
                .values(
                    approvals=approvals,
                    updated_at=time.time(),
                    pending_approval=None,
                )
            )

    # -- 读 ---------------------------------------------------------------

    def get(self, run_id: str) -> RunRecord | None:
        with self._session() as session:
            row = session.execute(
                select(runs).where(runs.c.run_id == run_id)
            ).first()
        return self._row_to_record(row) if row is not None else None

    def list(self, *, limit: int = 50) -> list[RunRecord]:
        with self._session() as session:
            rows = session.execute(
                select(runs).order_by(runs.c.updated_at.desc()).limit(limit)
            ).all()
        return [self._row_to_record(row) for row in rows]

    def get_sql_backend(self, run_id: str) -> str:
        record = self.get(run_id)
        return record.sql_backend if record is not None else "direct"

    def is_stale_running(self, run_id: str) -> bool:
        record = self.get(run_id)
        if record is None or record.status != "running":
            return False
        if not _process_alive(record.owner_pid):
            return True
        if record.heartbeat_at is None:
            return True
        return (time.time() - record.heartbeat_at) > STALE_HEARTBEAT_SECONDS


__all__ = ["PostgresRunStore", "ensure_schema"]
