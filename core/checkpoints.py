"""Checkpoint 存储。

两种后端，接口都是 LangGraph 的 `BaseCheckpointSaver`：

- **SQLite**（`local` 模式）：单文件、零外部服务，适合单机与测试；
- **PostgreSQL**（`postgres` 模式）：用官方 `PostgresSaver`，不自己实现协议。

SQLite 上的旧 Checkpoint **不会**自动迁移到 PostgreSQL：两种后端是不同的
表结构与不同的写入方。切到 `postgres` 模式后，旧 Run 应当继续用 `local`
模式读取，或者在明确知情的前提下重新开始——这里不会把一个不存在的 thread
静默当成"没有进度"，`/runs/{id}` 会返回空状态，调用方能看出区别。

副作用语义不变：每个 super-step 提交一次，进程被强杀时丢失最后一次提交之后
的进度，因此写操作仍需幂等键（见 utils/idempotency.py）。
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver

# 与 run_logger 的 outputs 约定一致，整体已在 .gitignore 中
DEFAULT_CHECKPOINT_PATH = Path("outputs") / "checkpoints" / "insightflow.sqlite"


def checkpoint_path(root: Path | str = "outputs") -> Path:
    return Path(root) / "checkpoints" / "insightflow.sqlite"


def _postgres_url() -> str | None:
    """当前配置需要 PostgreSQL Checkpoint 时返回 libpq 连接串。"""
    import storage

    config = storage.current_config()
    if not config.is_postgres:
        return None
    # LangGraph 的适配器用 psycopg 直连，不接受 SQLAlchemy 的 +psycopg 方言前缀
    return storage.database_url().replace("postgresql+psycopg://", "postgresql://", 1)


@contextmanager
def open_checkpointer(
    path: Path | str | None = None,
    *,
    url: str | None = None,
) -> Iterator[BaseCheckpointSaver]:
    """打开 Checkpointer，退出时关闭连接。

    每次调用新建连接，因此"进程重启后重新打开同一个存储"就是真实的恢复路径，
    而不是依赖进程内的内存状态。
    """
    database_url = url or _postgres_url()

    if database_url is not None:
        from langgraph.checkpoint.postgres import PostgresSaver

        with PostgresSaver.from_conn_string(database_url) as saver:
            # 建表是幂等的；放在这里保证第一次使用时 schema 就绪
            saver.setup()
            yield saver
        return

    database = Path(path) if path is not None else DEFAULT_CHECKPOINT_PATH
    database.parent.mkdir(parents=True, exist_ok=True)

    with SqliteSaver.from_conn_string(str(database)) as saver:
        yield saver


def list_checkpoint_threads(path: Path | str | None = None) -> list[str]:
    """列出已有 checkpoint 的 thread_id（运维排查用）。"""
    database_url = _postgres_url()
    if database_url is not None:
        import psycopg

        with psycopg.connect(database_url) as connection:
            rows = connection.execute(
                "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
            ).fetchall()
        return [row[0] for row in rows]

    import sqlite3

    database = Path(path) if path is not None else DEFAULT_CHECKPOINT_PATH
    if not database.exists():
        return []

    with sqlite3.connect(str(database)) as connection:
        rows = connection.execute(
            "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
        ).fetchall()
    return [row[0] for row in rows]


def checkpoint_backend() -> str:
    """当前 Checkpoint 后端名，供健康检查与报告使用。"""
    return "postgres" if _postgres_url() is not None else "sqlite"
