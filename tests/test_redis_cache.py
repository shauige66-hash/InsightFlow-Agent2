"""Redis 缓存：命中、隔离、TTL、失效、并发与降级。

用例跑真实 Redis。Redis 不是权威存储，因此这里同时验证"缓存丢了不影响正确性"。
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

import storage
from rag.cache import KnowledgeCache
from rag.retrieval import get_retriever, reset_retriever
from tests.postgres_support import (  # noqa: F401
    clear_redis,
    redis_available,
    redis_cache,
    redis_url,
    requires_redis,
)

VIP_DOC = "VIP 客户指年消费金额超过 10000 元的客户。\n"


@pytest.fixture(autouse=True)
def _local_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("INSIGHTFLOW_STORAGE", "local")
    monkeypatch.setenv("INSIGHTFLOW_LOCAL_DB", str(tmp_path / "local.sqlite"))
    monkeypatch.setenv("INSIGHTFLOW_OWNER", "cache-test")
    monkeypatch.setenv("INSIGHTFLOW_REDIS_ENABLED", "true")
    monkeypatch.setenv("INSIGHTFLOW_REDIS_URL", redis_url())
    storage.reset()
    storage.configure()
    reset_retriever()
    if redis_available():
        clear_redis()
    yield
    storage.reset()
    reset_retriever()


@pytest.fixture
def doc(tmp_path: Path) -> Path:
    path = tmp_path / "vip.md"
    path.write_text(VIP_DOC, encoding="utf-8")
    return path


class TestKeyDesign:
    def test_key_includes_every_factor_that_changes_the_result(self) -> None:
        base = dict(
            namespace="kb:a",
            knowledge_version="v1",
            query_hash="q1",
            embedding_model_version="m@384",
            top_k=3,
            min_score=0.35,
            backend="inmemory",
        )
        key = KnowledgeCache.retrieval_key(**base)

        for field_name, value in base.items():
            if field_name == "query_hash":
                continue
            assert str(value) in key, field_name

        different = dict(base)
        different["top_k"] = 1
        assert KnowledgeCache.retrieval_key(**different) != key

        other_namespace = dict(base)
        other_namespace["namespace"] = "kb:b"
        assert KnowledgeCache.retrieval_key(**other_namespace) != key

    def test_embedding_key_is_separate_from_retrieval_key(self) -> None:
        retrieval = KnowledgeCache.retrieval_key(
            namespace="kb:a",
            knowledge_version="v1",
            query_hash="q",
            embedding_model_version="m",
            top_k=3,
            min_score=0.35,
            backend="inmemory",
        )
        embedding = KnowledgeCache.embedding_key(
            query_hash="q", embedding_model_version="m"
        )
        assert retrieval != embedding


class TestHitMiss:
    @requires_redis
    def test_first_miss_then_hit(self, redis_cache) -> None:
        assert redis_cache.get_json("k1") is None
        redis_cache.set_json("k1", {"value": 1})
        assert redis_cache.get_json("k1") == {"value": 1}
        assert redis_cache.stats.misses == 1
        assert redis_cache.stats.hits == 1
        assert redis_cache.stats.hit_rate == 0.5

    @requires_redis
    def test_ttl_expiry(self, redis_cache) -> None:
        cache = KnowledgeCache(redis_url(), enabled=True, ttl_seconds=1)
        cache.set_json("short", [1, 2])
        assert cache.get_json("short") == [1, 2]
        time.sleep(1.2)
        assert cache.get_json("short") is None

    @requires_redis
    def test_disabled_cache_never_hits(self) -> None:
        cache = KnowledgeCache(None, enabled=False)
        assert cache.get_json("anything") is None
        cache.set_json("anything", 1)
        assert cache.get_json("anything") is None
        assert cache.stats.hits == 0


class TestRetrieverIntegration:
    @requires_redis
    def test_second_identical_query_is_served_from_cache(self, doc: Path) -> None:
        retriever = get_retriever()
        cold = retriever.retrieve(knowledge_sources=[str(doc)], query="VIP 客户定义")
        warm = retriever.retrieve(knowledge_sources=[str(doc)], query="VIP 客户定义")

        assert cold.cache_hit is False
        assert warm.cache_hit is True
        assert warm.evidence[0].chunk_id == cold.evidence[0].chunk_id
        assert warm.latency_ms < cold.latency_ms

    @requires_redis
    def test_a_different_knowledge_base_does_not_hit_the_same_entry(
        self, doc: Path, tmp_path: Path
    ) -> None:
        other = tmp_path / "other.md"
        other.write_text("配送时效为三个工作日。\n", encoding="utf-8")

        first = get_retriever().retrieve(
            knowledge_sources=[str(doc)], query="VIP 客户定义"
        )
        second = get_retriever().retrieve(
            knowledge_sources=[str(other)], query="VIP 客户定义"
        )

        assert first.cache_hit is False
        assert second.cache_hit is False
        assert second.namespace != first.namespace

    @requires_redis
    def test_document_update_is_not_served_from_an_old_entry(self, doc: Path) -> None:
        retriever = get_retriever()
        before = retriever.retrieve(knowledge_sources=[str(doc)], query="VIP 年消费门槛")

        doc.write_text("VIP 客户指年消费金额超过 50000 元的客户。\n", encoding="utf-8")
        after = retriever.retrieve(knowledge_sources=[str(doc)], query="VIP 年消费门槛")

        assert after.cache_hit is False
        assert after.knowledge_version != before.knowledge_version
        assert "50000" in after.evidence[0].content

    @requires_redis
    def test_deleting_the_document_stops_returning_it(self, doc: Path) -> None:
        retriever = get_retriever()
        assert retriever.retrieve(knowledge_sources=[str(doc)], query="VIP 定义").evidence

        doc.unlink()
        with pytest.raises(FileNotFoundError):
            retriever.retrieve(knowledge_sources=[str(doc)], query="VIP 定义")

    @requires_redis
    def test_concurrent_identical_queries_stay_consistent(self, doc: Path) -> None:
        """并发请求同一个问句：结果一致，不因击穿而失败。"""
        results: list[list[str]] = []
        errors: list[BaseException] = []
        barrier = threading.Barrier(6)

        def worker() -> None:
            try:
                barrier.wait(timeout=30)
                outcome = get_retriever().retrieve(
                    knowledge_sources=[str(doc)], query="VIP 客户定义"
                )
                results.append([item.chunk_id for item in outcome.evidence])
            except BaseException as error:  # noqa: BLE001
                errors.append(error)

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)

        assert not errors, errors
        assert len(results) == 6
        assert len({tuple(item) for item in results}) == 1, results


class TestDegradation:
    def test_redis_down_falls_back_to_the_index(
        self, doc: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("INSIGHTFLOW_REDIS_URL", "redis://127.0.0.1:6399/0")
        storage.reset()
        storage.configure()
        reset_retriever()

        outcome = get_retriever().retrieve(
            knowledge_sources=[str(doc)], query="VIP 客户定义"
        )

        assert outcome.cache_hit is False
        assert [item.source for item in outcome.evidence] == ["vip.md"]

    def test_missing_source_still_fails_with_cache_down(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """降级不能绕过来源/权限校验。"""
        monkeypatch.setenv("INSIGHTFLOW_REDIS_URL", "redis://127.0.0.1:6399/0")
        storage.reset()
        storage.configure()
        reset_retriever()

        with pytest.raises(FileNotFoundError):
            get_retriever().retrieve(
                knowledge_sources=[str(tmp_path / "missing.md")], query="任意"
            )

    @requires_redis
    def test_cache_stats_expose_hit_rate_and_latency(self, doc: Path) -> None:
        retriever = get_retriever()
        retriever.retrieve(knowledge_sources=[str(doc)], query="VIP 客户定义")
        retriever.retrieve(knowledge_sources=[str(doc)], query="VIP 客户定义")

        stats = retriever.stats(knowledge_sources=[str(doc)])["cache"]

        assert stats["enabled"] is True
        assert stats["reachable"] is True
        assert stats["hits"] >= 1
        assert 0.0 < stats["hit_rate"] <= 1.0
        assert stats["ttl_seconds"] == 600
