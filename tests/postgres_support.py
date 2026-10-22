"""PostgreSQL / Redis 测试支持：连通性探测与可重复的清理。

测试连真实容器（`deploy/compose.yaml`），不用 SQLite 或假客户端替代：
"向量索引在真实 PostgreSQL 里可复用"这件事，只有真跑才能验证。

必须显式指定专用测试数据库和清理许可才会连接。缺少许可或服务不可达时
测试会被 skip；默认不会连接或重置应用数据库。
"""

from __future__ import annotations

import os
from urllib.parse import urlsplit

import pytest

DEFAULT_DATABASE = (
    "postgresql+psycopg://insightflow:insightflow@127.0.0.1:5433/insightflow_test"
)
DEFAULT_REDIS = "redis://127.0.0.1:6380/15"

TABLES = (
    "knowledge_chunks",
    "knowledge_documents",
    "session_run_links",
    "conversation_messages",
    "conversation_turns",
    "conversation_sessions",
    "long_term_memories",
    "runs",
)


def database_url() -> str:
    return os.getenv("INSIGHTFLOW_TEST_DATABASE_URL", DEFAULT_DATABASE)


def redis_url() -> str:
    return os.getenv("INSIGHTFLOW_TEST_REDIS_URL", DEFAULT_REDIS)


def libpq_url(url: str | None = None) -> str:
    return (url or database_url()).replace("postgresql+psycopg://", "postgresql://", 1)


def require_isolated_postgres() -> None:
    """Destructive fixtures require explicit opt-in and a dedicated test DB."""
    name = urlsplit(libpq_url()).path.strip('/')
    if (os.getenv('INSIGHTFLOW_ALLOW_TEST_DB_RESET') != '1'
            or not os.getenv('INSIGHTFLOW_TEST_DATABASE_URL')
            or not (name.endswith('_test') or name.startswith('test_'))):
        raise RuntimeError('PostgreSQL fixture reset requires an explicit dedicated *_test database and INSIGHTFLOW_ALLOW_TEST_DB_RESET=1')


def require_isolated_redis() -> None:
    parsed = urlsplit(redis_url())
    if (os.getenv('INSIGHTFLOW_ALLOW_TEST_REDIS_RESET') != '1'
            or not os.getenv('INSIGHTFLOW_TEST_REDIS_URL')
            or not parsed.path.strip('/').isdigit()
            or int(parsed.path.strip('/')) == 0):
        raise RuntimeError('Redis fixture reset requires an explicit nonzero test DB and INSIGHTFLOW_ALLOW_TEST_REDIS_RESET=1')


def postgres_available() -> bool:
    try:
        require_isolated_postgres()
        import psycopg

        with psycopg.connect(libpq_url(), connect_timeout=2) as connection:
            connection.execute("SELECT 1")
        return True
    except Exception:  # noqa: BLE001 - 探测失败即视为不可用
        return False


def redis_available() -> bool:
    try:
        require_isolated_redis()
        import redis

        client = redis.Redis.from_url(redis_url(), socket_connect_timeout=2)
        return bool(client.ping())
    except Exception:  # noqa: BLE001
        return False


requires_postgres = pytest.mark.skipif(
    not postgres_available(),
    reason="需要真实 PostgreSQL（docker compose -f deploy/compose.yaml up -d）",
)

requires_redis = pytest.mark.skipif(
    not redis_available(),
    reason="需要真实 Redis（docker compose -f deploy/compose.yaml up -d）",
)


def drop_tables() -> None:
    """删掉本轮涉及的表，让下一次迁移从零开始。

    只删这些表，不碰 LangGraph Checkpoint 表：不同用例的 thread_id 不同，
    不需要清理。
    """
    require_isolated_postgres()
    import psycopg

    with psycopg.connect(libpq_url()) as connection:
        connection.execute(
            "DROP TABLE IF EXISTS " + ", ".join(TABLES) + " CASCADE"
        )


def clear_redis() -> None:
    require_isolated_redis()
    import redis

    redis.Redis.from_url(redis_url()).flushdb()


@pytest.fixture
def postgres(monkeypatch: pytest.MonkeyPatch):
    """切到 postgres 模式，并保证从一个干净的 schema 开始。"""
    if not postgres_available():
        pytest.skip("需要真实 PostgreSQL")

    import storage

    monkeypatch.setenv("INSIGHTFLOW_STORAGE", "postgres")
    monkeypatch.setenv("INSIGHTFLOW_DATABASE_URL", database_url())
    monkeypatch.setenv("INSIGHTFLOW_OWNER", "pg-test")
    monkeypatch.setenv("INSIGHTFLOW_REDIS_ENABLED", "false")

    storage.reset()
    drop_tables()
    storage.configure()
    yield storage
    storage.reset()


@pytest.fixture
def redis_cache():
    """真实 Redis；每个用例前清空，避免互相污染。"""
    if not redis_available():
        pytest.skip("需要真实 Redis")

    from rag.cache import KnowledgeCache

    clear_redis()
    cache = KnowledgeCache(redis_url(), enabled=True, ttl_seconds=600)
    yield cache
    clear_redis()
