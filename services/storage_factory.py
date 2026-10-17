"""按当前存储模式构造 Run 登记与 Checkpoint 后端。

服务与 API 只调用这里，不自己判断"现在该用哪个数据库"：模式判断集中在一处，
启动、查询、恢复、审批才会用同一个后端。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import storage
from services.run_store import RunStore


def run_store(output_root: Path | str = "outputs") -> Any:
    """Run 登记：local 用 SQLite，postgres 用 PostgreSQL。"""
    config = storage.current_config()
    if config.is_postgres:
        from services.run_store_postgres import PostgresRunStore

        return PostgresRunStore(storage.database_url())

    root = Path(output_root)
    return RunStore(root / "runs" / "runs.sqlite")


def checkpoint_path(output_root: Path | str = "outputs") -> Path:
    return Path(output_root) / "checkpoints" / "insightflow.sqlite"


def describe() -> dict[str, Any]:
    """存储后端摘要（健康检查与报告用，不含凭据）。"""
    config = storage.current_config()
    from core.checkpoints import checkpoint_backend

    return {
        "mode": config.mode,
        "run_store": "postgres" if config.is_postgres else "sqlite",
        "checkpointer": checkpoint_backend(),
        "knowledge_index": "pgvector" if config.is_postgres else "inmemory",
        "cache": "redis" if config.redis_enabled else "disabled",
        "owner": config.owner,
    }


__all__ = ["checkpoint_path", "describe", "run_store"]
