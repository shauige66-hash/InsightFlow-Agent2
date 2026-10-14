"""Checkpoint 存储。

使用 LangGraph 自带的 SQLite Checkpointer：本地文件、零外部服务，
适合单机部署；接口按 `BaseCheckpointSaver` 编程，将来换 PostgreSQL
只需要替换这里的构造函数。

持久化模式说明：
- `SqliteSaver` 默认 `WRITE` 模式，每个 super-step 提交一次，
  进程被强杀时最后一次提交之后的进度会丢失，恢复时从最后提交点继续。
  这是"至少一次"而非"恰好一次"，因此副作用必须配幂等键（见 utils/idempotency.py）。
- 数据库文件放在 V2 自己的 `outputs/` 下，不碰 ModelGate 的 PostgreSQL，
  也不引入新的数据库服务。
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


@contextmanager
def open_checkpointer(
    path: Path | str | None = None,
) -> Iterator[BaseCheckpointSaver]:
    """打开持久化 Checkpointer，退出时关闭连接。

    每次调用新建一个连接，因此"进程重启后重新打开同一个文件"就是真实的恢复路径，
    而不是依赖进程内的内存状态。
    """
    database = Path(path) if path is not None else DEFAULT_CHECKPOINT_PATH
    database.parent.mkdir(parents=True, exist_ok=True)

    with SqliteSaver.from_conn_string(str(database)) as saver:
        yield saver


def list_checkpoint_threads(path: Path | str | None = None) -> list[str]:
    """列出已有 checkpoint 的 thread_id。

    用于运维排查：确认某个 run 是否有可恢复的状态。
    """
    import sqlite3

    database = Path(path) if path is not None else DEFAULT_CHECKPOINT_PATH
    if not database.exists():
        return []

    with sqlite3.connect(str(database)) as connection:
        rows = connection.execute(
            "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
        ).fetchall()
    return [row[0] for row in rows]
