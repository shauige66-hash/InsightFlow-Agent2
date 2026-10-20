"""PostgreSQL 作业队列：让 Run 的执行与 API 进程解耦。

为什么只支持 postgres：`SELECT ... FOR UPDATE SKIP LOCKED` 是这里唯一可靠的
抢占手段，SQLite 没有它。local 模式继续用进程内线程池——那本来就不是给多进程
部署用的，硬凑一个假队列只会让人以为重启不丢任务。

设计取舍：

- **用数据库而不是 Redis 做队列**。Redis 已经用于检索缓存，但队列要求"任务不能
  因为缓存被清空而消失"，这是持久化语义；再引入一套 Redis 队列等于两个真相来源。
- **payload 原样存 JSON**：入队时不解释它，worker 端按同一份参数启动 Run，
  避免 API 与 worker 对字段理解不一致。
- **不自动重试**。抢占失败（worker 崩溃）由运维决定：`requeue()` 显式放回队列，
  而不是偷偷重跑一次可能已经产生副作用的任务。
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import text

import storage

_SCHEMA = """
CREATE TABLE IF NOT EXISTS run_jobs (
    run_id     TEXT PRIMARY KEY,
    payload    JSONB NOT NULL,
    status     TEXT NOT NULL DEFAULT 'queued',
    attempts   INTEGER NOT NULL DEFAULT 0,
    claimed_by TEXT,
    claimed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    error      TEXT
);
CREATE INDEX IF NOT EXISTS run_jobs_status_created
    ON run_jobs (status, created_at);
"""


class QueueUnavailable(RuntimeError):
    """当前存储后端不支持队列（只有 postgres 支持）。"""


def _require_postgres() -> None:
    if not storage.current_config().is_postgres:
        raise QueueUnavailable(
            "作业队列需要 INSIGHTFLOW_STORAGE=postgres；local 模式请使用进程内执行"
        )


def ensure_schema() -> None:
    """建表（幂等）。"""
    _require_postgres()
    with storage.session_scope(storage.database_url()) as session:
        for statement in _SCHEMA.strip().split(";"):
            if statement.strip():
                session.execute(text(statement))
        session.commit()


def enqueue(run_id: str, payload: dict[str, Any]) -> None:
    """把一个 Run 放进队列。重复入队是幂等的。"""
    _require_postgres()
    with storage.session_scope(storage.database_url()) as session:
        session.execute(
            text(
                """
                INSERT INTO run_jobs (run_id, payload)
                VALUES (:run_id, CAST(:payload AS JSONB))
                ON CONFLICT (run_id) DO NOTHING
                """
            ),
            {"run_id": run_id, "payload": json.dumps(payload, ensure_ascii=False)},
        )
        session.commit()


def claim(worker_id: str) -> tuple[str, dict[str, Any]] | None:
    """抢占一个待执行任务；没有任务时返回 None。

    `FOR UPDATE SKIP LOCKED` 保证两个 worker 不会拿到同一条，也不会互相阻塞。
    """
    _require_postgres()
    with storage.session_scope(storage.database_url()) as session:
        row = session.execute(
            text(
                """
                UPDATE run_jobs
                   SET status = 'claimed',
                       claimed_by = :worker,
                       claimed_at = now(),
                       attempts = attempts + 1
                 WHERE run_id = (
                       SELECT run_id FROM run_jobs
                        WHERE status = 'queued'
                        ORDER BY created_at
                        FOR UPDATE SKIP LOCKED
                        LIMIT 1
                 )
             RETURNING run_id, payload
                """
            ),
            {"worker": worker_id},
        ).first()
        session.commit()

    if row is None:
        return None
    payload = row.payload
    if isinstance(payload, str):
        payload = json.loads(payload)
    return row.run_id, dict(payload)


def finish(run_id: str, *, error: str | None = None) -> None:
    """标记任务结束（成功或失败）。"""
    _require_postgres()
    with storage.session_scope(storage.database_url()) as session:
        session.execute(
            text(
                """
                UPDATE run_jobs
                   SET status = :status, error = :error
                 WHERE run_id = :run_id
                """
            ),
            {
                "run_id": run_id,
                "status": "failed" if error else "done",
                "error": error,
            },
        )
        session.commit()


def requeue(run_id: str) -> None:
    """把一条 claimed 的任务放回队列（人工决定重试时用）。"""
    _require_postgres()
    with storage.session_scope(storage.database_url()) as session:
        session.execute(
            text(
                """
                UPDATE run_jobs
                   SET status = 'queued', claimed_by = NULL, claimed_at = NULL
                 WHERE run_id = :run_id AND status = 'claimed'
                """
            ),
            {"run_id": run_id},
        )
        session.commit()


def depth() -> dict[str, int]:
    """队列水位，给健康检查与监控用。"""
    _require_postgres()
    with storage.session_scope(storage.database_url()) as session:
        rows = session.execute(
            text("SELECT status, COUNT(*) AS n FROM run_jobs GROUP BY status")
        ).all()
    return {row.status: int(row.n) for row in rows}


def status_of(run_id: str) -> str | None:
    _require_postgres()
    with storage.session_scope(storage.database_url()) as session:
        row = session.execute(
            text("SELECT status FROM run_jobs WHERE run_id = :run_id"),
            {"run_id": run_id},
        ).first()
    return row.status if row else None


def attempts(run_id: str) -> int | None:
    """这个任务被抢占过几次——用来确认"重试"是可见事实而不是猜测。"""
    _require_postgres()
    with storage.session_scope(storage.database_url()) as session:
        row = session.execute(
            text("SELECT attempts FROM run_jobs WHERE run_id = :run_id"),
            {"run_id": run_id},
        ).first()
    return int(row.attempts) if row else None


def last_error(run_id: str) -> str | None:
    _require_postgres()
    with storage.session_scope(storage.database_url()) as session:
        row = session.execute(
            text("SELECT error FROM run_jobs WHERE run_id = :run_id"),
            {"run_id": run_id},
        ).first()
    return row.error if row else None
