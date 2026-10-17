"""持久化层：会话、长期记忆、知识库与存储运行配置。

对外只暴露"当前存储"这一件事：配置来自环境变量，数据库 URL 与 schema
在第一次使用时解析并迁移一次。服务层拿到的是 session/repository，不直接
接触 Engine。
"""

from __future__ import annotations

from pathlib import Path

from storage.config import (
    LOCAL,
    POSTGRES,
    StorageConfig,
    StorageConfigError,
    load_storage_config,
    resolve_database_url,
)
from storage.engine import (
    dispose_all,
    get_engine,
    get_session_factory,
    healthcheck,
    migrate,
    session_scope,
)

_config: StorageConfig | None = None
_url: str | None = None
_base_dir: Path | None = None


def configure(
    config: StorageConfig | None = None,
    *,
    base_dir: Path | None = None,
) -> StorageConfig:
    """显式设置存储配置（测试与启动脚本用）。"""
    global _config, _url, _base_dir
    _config = config or load_storage_config()
    _base_dir = base_dir
    _url = None
    return _config


def reset() -> None:
    """清空进程内的存储状态（测试用）。"""
    global _config, _url, _base_dir
    _config = None
    _url = None
    _base_dir = None
    dispose_all()


def current_config() -> StorageConfig:
    global _config
    if _config is None:
        _config = load_storage_config()
    return _config


def database_url() -> str:
    """当前存储的数据库 URL，并确保 schema 已迁移。"""
    global _url
    if _url is None:
        config = current_config()
        _url = resolve_database_url(config, base_dir=_base_dir)
        migrate(
            _url,
            knowledge=config.is_postgres,
            pool_size=config.pool_size,
        )
    return _url


def has_knowledge_index() -> bool:
    """持久化向量索引只在 PostgreSQL 模式下可用。"""
    return current_config().is_postgres


__all__ = [
    "LOCAL",
    "POSTGRES",
    "StorageConfig",
    "StorageConfigError",
    "configure",
    "current_config",
    "database_url",
    "dispose_all",
    "get_engine",
    "get_session_factory",
    "has_knowledge_index",
    "healthcheck",
    "migrate",
    "reset",
    "resolve_database_url",
    "session_scope",
]
