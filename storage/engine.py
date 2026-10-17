"""数据库连接、迁移与健康检查。

一个 URL 对应一个 Engine，进程内复用连接池。迁移是**幂等、有序、可重复执行**的：
每一步只做"还没有就补上"，并在 `schema_migrations` 里记一笔，因此重复调用
不会重建表，也不会重复建索引。
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import Engine, create_engine, event, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from storage.config import StorageConfig, resolve_database_url
from storage.models import Base, SchemaVersion

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "001_init"

_ENGINES: dict[str, Engine] = {}
_SESSIONS: dict[str, sessionmaker[Session]] = {}


def _is_postgres(engine: Engine) -> bool:
    return engine.dialect.name == "postgresql"


def _configure(engine: Engine) -> None:
    if engine.dialect.name == "sqlite":
        # SQLite 默认不校验外键；删除会话时依赖 ondelete 的行为必须打开它
        @event.listens_for(engine, "connect")
        def _fk_on(dbapi_connection, _record):  # noqa: ANN001
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()


def get_engine(url: str, *, pool_size: int = 5) -> Engine:
    """按 URL 取（或创建）Engine。"""
    engine = _ENGINES.get(url)
    if engine is not None:
        return engine

    options: dict[str, Any] = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        options["connect_args"] = {"check_same_thread": False}
    else:
        options["pool_size"] = max(1, pool_size)
        options["max_overflow"] = max(1, pool_size)

    engine = create_engine(url, **options)
    _configure(engine)
    _ENGINES[url] = engine
    return engine


def get_session_factory(url: str, *, pool_size: int = 5) -> sessionmaker[Session]:
    factory = _SESSIONS.get(url)
    if factory is None:
        factory = sessionmaker(
            bind=get_engine(url, pool_size=pool_size),
            expire_on_commit=False,
        )
        _SESSIONS[url] = factory
    return factory


@contextmanager
def session_scope(url: str, *, pool_size: int = 5) -> Iterator[Session]:
    """事务边界：正常结束提交，异常回滚。"""
    factory = get_session_factory(url, pool_size=pool_size)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def migrate(
    url: str,
    *,
    knowledge: bool,
    pool_size: int = 5,
) -> dict[str, Any]:
    """建表并记录 schema 版本。

    `knowledge=True` 时额外创建 pgvector 扩展、向量表与索引；只有 PostgreSQL
    支持它，SQLite 模式继续使用进程内索引。
    """
    engine = get_engine(url, pool_size=pool_size)
    applied: list[str] = []
    before = set(inspect(engine).get_table_names())

    if _is_postgres(engine) and knowledge:
        # 扩展必须先于向量列存在
        with engine.begin() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    Base.metadata.create_all(engine, checkfirst=True)
    if set(inspect(engine).get_table_names()) - before:
        applied.append("core_tables")

    if _is_postgres(engine) and knowledge:
        from storage.knowledge_models import KnowledgeBase

        KnowledgeBase.metadata.create_all(engine, checkfirst=True)
        with engine.begin() as connection:
            # hnsw + cosine：与检索时使用的距离算子保持一致
            connection.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_knowledge_chunks_embedding "
                    "ON knowledge_chunks USING hnsw (embedding vector_cosine_ops)"
                )
            )
        if "knowledge_chunks" not in before:
            applied.append("knowledge_tables")

    with session_scope(url, pool_size=pool_size) as session:
        if session.get(SchemaVersion, SCHEMA_VERSION) is None:
            session.add(SchemaVersion(version=SCHEMA_VERSION))
            applied.append("schema_version")

    return {"database": engine.dialect.name, "applied": applied}


def healthcheck(url: str) -> dict[str, Any]:
    """连通性与表是否就绪。不返回凭据。"""
    engine = get_engine(url)
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
        tables = set(inspect(connection).get_table_names())
    return {
        "dialect": engine.dialect.name,
        "reachable": True,
        "tables_ready": "conversation_sessions" in tables,
        "knowledge_ready": "knowledge_chunks" in tables,
    }


def dispose_all() -> None:
    """关闭并丢弃所有 Engine（测试与进程退出用）。"""
    for engine in _ENGINES.values():
        engine.dispose()
    _ENGINES.clear()
    _SESSIONS.clear()


def storage_from_config(config: StorageConfig, *, base_dir=None) -> str:
    """把配置解析成数据库 URL，并确保 schema 就绪。"""
    url = resolve_database_url(config, base_dir=base_dir)
    migrate(url, knowledge=config.is_postgres, pool_size=config.pool_size)
    return url


__all__ = [
    "SCHEMA_VERSION",
    "dispose_all",
    "get_engine",
    "get_session_factory",
    "healthcheck",
    "migrate",
    "session_scope",
    "storage_from_config",
]
