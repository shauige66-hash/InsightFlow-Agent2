"""持久化与缓存的运行配置。

两种模式，语义不同：

- `local`：SQLite RunStore + LangGraph SQLite Checkpointer + 进程内知识索引。
  轻量单机、测试与演示用，不需要外部服务。
- `postgres`：会话、长期记忆、知识索引进 PostgreSQL（pgvector），
  RunStore 与 LangGraph Checkpoint 也切到 PostgreSQL。

模式只由宿主环境决定（`INSIGHTFLOW_STORAGE`），不从请求体读取：
存储后端属于运行期配置，让调用方按请求切换会破坏状态一致性。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

LOCAL = "local"
POSTGRES = "postgres"

DEFAULT_LOCAL_DB = "outputs/conversations.sqlite"
DEFAULT_OWNER = "local-demo"


class StorageConfigError(ValueError):
    """存储配置不可用。"""


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as error:
        raise StorageConfigError(f"{name} 必须是整数，收到 {raw!r}") from error


@dataclass(frozen=True)
class StorageConfig:
    mode: str = LOCAL
    database_url: str = ""
    local_db_path: str = DEFAULT_LOCAL_DB
    pool_size: int = 5
    owner: str = DEFAULT_OWNER
    redis_enabled: bool = False
    redis_url: str = ""
    redis_ttl_seconds: int = 600

    @property
    def is_postgres(self) -> bool:
        return self.mode == POSTGRES

    def describe(self) -> dict[str, object]:
        """给健康检查用的描述，不含凭据。"""
        return {
            "mode": self.mode,
            "database": _redact(self.database_url or self.local_db_path),
            "cache": "redis" if self.redis_enabled else "disabled",
            "owner": self.owner,
        }


def _redact(url: str) -> str:
    """去掉 URL 里的口令，健康检查不能把密码回显出去。"""
    if "://" not in url:
        return url
    scheme, _, rest = url.partition("://")
    if "@" not in rest:
        return url
    credentials, _, host = rest.rpartition("@")
    user = credentials.split(":", 1)[0]
    return f"{scheme}://{user}:***@{host}"


def load_storage_config() -> StorageConfig:
    """从环境变量读出配置。"""
    mode = (os.getenv("INSIGHTFLOW_STORAGE") or LOCAL).strip().lower()
    if mode not in (LOCAL, POSTGRES):
        raise StorageConfigError(
            f"INSIGHTFLOW_STORAGE 只能是 {LOCAL} 或 {POSTGRES}，收到 {mode!r}"
        )

    database_url = (os.getenv("INSIGHTFLOW_DATABASE_URL") or "").strip()
    if mode == POSTGRES and not database_url:
        raise StorageConfigError(
            "postgres 模式必须设置 INSIGHTFLOW_DATABASE_URL"
        )

    redis_enabled = _flag("INSIGHTFLOW_REDIS_ENABLED", False)
    redis_url = (os.getenv("INSIGHTFLOW_REDIS_URL") or "").strip()
    if redis_enabled and not redis_url:
        raise StorageConfigError(
            "启用 Redis 缓存时必须设置 INSIGHTFLOW_REDIS_URL"
        )

    return StorageConfig(
        mode=mode,
        database_url=database_url,
        local_db_path=os.getenv("INSIGHTFLOW_LOCAL_DB") or DEFAULT_LOCAL_DB,
        pool_size=_int("INSIGHTFLOW_DB_POOL_SIZE", 5),
        owner=(os.getenv("INSIGHTFLOW_OWNER") or DEFAULT_OWNER),
        redis_enabled=redis_enabled,
        redis_url=redis_url,
        redis_ttl_seconds=_int("INSIGHTFLOW_REDIS_TTL_SECONDS", 600),
    )


def resolve_database_url(config: StorageConfig, *, base_dir: Path | None = None) -> str:
    """会话/记忆/知识库使用的数据库 URL。"""
    if config.is_postgres:
        return config.database_url

    path = Path(config.local_db_path)
    if not path.is_absolute():
        root = base_dir or Path.cwd()
        path = root / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite+pysqlite:///{path}"


__all__ = [
    "DEFAULT_LOCAL_DB",
    "DEFAULT_OWNER",
    "LOCAL",
    "POSTGRES",
    "StorageConfig",
    "StorageConfigError",
    "load_storage_config",
    "resolve_database_url",
]
