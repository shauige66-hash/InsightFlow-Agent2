"""RAG 查询缓存（Redis）。

Redis 只缓存**可以重算**的东西：查询向量与检索结果。它不是会话、记忆或
Checkpoint 的权威存储，清空它只会让下一次查询变慢。

缓存键必须包含所有影响结果的因素，否则不同知识库或不同版本会互相命中：

    ns:知识库  ver:内容版本  model:embedding 版本  k:top_k  min:阈值  q:问句哈希

失效策略靠版本而不是靠手工清理：文档一变，`knowledge_version` 就变，旧键
自然不再被读到，再由 TTL 回收。Redis 不可用时检索回退到持久化索引——
但权限校验失败、知识源缺失这类问题不能被回退掩盖，因此在进入缓存之前
就抛错。
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

CACHE_PREFIX = "insightflow:rag"


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    errors: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return round(self.hits / total, 4) if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "errors": self.errors,
            "hit_rate": self.hit_rate,
        }


class KnowledgeCache:
    """薄封装的 Redis 缓存；不可用时自动降级为空缓存。"""

    def __init__(
        self,
        url: str | None,
        *,
        enabled: bool,
        ttl_seconds: int = 600,
        client: Any | None = None,
    ) -> None:
        self._ttl = max(1, ttl_seconds)
        self._enabled = bool(enabled and (client is not None or url))
        self.stats = CacheStats()
        self._client = client
        self._url = url

    # -- 连接 -------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled and self._client is not None

    def _connection(self) -> Any | None:
        if not self._enabled:
            return None
        if self._client is None:
            import redis

            self._client = redis.Redis.from_url(
                self._url, decode_responses=True, socket_timeout=1.0,
                socket_connect_timeout=1.0,
            )
        return self._client

    def ping(self) -> bool:
        client = self._connection()
        if client is None:
            return False
        try:
            return bool(client.ping())
        except Exception:  # noqa: BLE001 - 探测失败即视为不可用
            self.stats.errors += 1
            return False

    # -- 键 ---------------------------------------------------------------

    @staticmethod
    def retrieval_key(
        *,
        namespace: str,
        knowledge_version: str,
        query_hash: str,
        embedding_model_version: str,
        top_k: int,
        min_score: float,
        backend: str,
    ) -> str:
        return (
            f"{CACHE_PREFIX}:retrieval:{namespace}:{knowledge_version}:"
            f"{embedding_model_version}:{backend}:k{top_k}:"
            f"min{min_score:.3f}:q{query_hash}"
        )

    @staticmethod
    def embedding_key(*, query_hash: str, embedding_model_version: str) -> str:
        return f"{CACHE_PREFIX}:embedding:{embedding_model_version}:q{query_hash}"

    # -- 读写 -------------------------------------------------------------

    def get_json(self, key: str) -> Any | None:
        client = self._connection()
        if client is None:
            self.stats.misses += 1
            return None
        try:
            raw = client.get(key)
        except Exception:  # noqa: BLE001 - 缓存故障不该让检索失败
            self.stats.errors += 1
            logger.warning("redis get failed, falling back to index", exc_info=True)
            return None
        if raw is None:
            self.stats.misses += 1
            return None
        self.stats.hits += 1
        return json.loads(raw)

    def set_json(self, key: str, value: Any) -> None:
        client = self._connection()
        if client is None:
            return
        try:
            client.set(key, json.dumps(value, ensure_ascii=False), ex=self._ttl)
        except Exception:  # noqa: BLE001
            self.stats.errors += 1
            logger.warning("redis set failed", exc_info=True)

    def get_embedding(self, key: str) -> list[float] | None:
        payload = self.get_json(key)
        if not isinstance(payload, list):
            return None
        return [float(item) for item in payload]

    def set_embedding(self, key: str, embedding: list[float]) -> None:
        self.set_json(key, embedding)

    def delete_prefix(self, prefix: str) -> int:
        """按前缀清理（运维/测试用）。版本失效不依赖它。"""
        client = self._connection()
        if client is None:
            return 0
        removed = 0
        try:
            for key in client.scan_iter(match=f"{prefix}*", count=500):
                removed += int(client.delete(key))
        except Exception:  # noqa: BLE001
            self.stats.errors += 1
        return removed

    def flush_namespace(self, namespace: str) -> int:
        return self.delete_prefix(f"{CACHE_PREFIX}:retrieval:{namespace}")

    def measured(self) -> dict[str, Any]:
        started = time.perf_counter()
        reachable = self.ping()
        return {
            **self.stats.to_dict(),
            "enabled": self._enabled,
            "reachable": reachable,
            "ping_ms": round((time.perf_counter() - started) * 1000, 2),
            "ttl_seconds": self._ttl,
        }


__all__ = ["CACHE_PREFIX", "CacheStats", "KnowledgeCache"]
