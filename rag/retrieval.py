"""任务级知识检索：进程内索引与持久化索引共用一套证据结构。

两件事刻意分开：

- **选什么**：`build_task_query` 决定问句，一个任务一个问句；
- **从哪查**：local 模式查进程内向量库，postgres 模式查 pgvector 表。

无论哪种后端，返回的都是 `RetrievalEvidence`，分数方向统一为"越大越相关"：
进程内索引返回余弦相似度，pgvector 的余弦距离在这里换算成 `1 - distance`。
两套阈值因此不能互换，也不该在同一个缓存键里混用。
"""

from __future__ import annotations

import logging
import time
from threading import RLock
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from langchain_core.documents import Document

from rag.cache import KnowledgeCache
from rag.chunker import split_documents
from rag.config import DEFAULT_TOP_K, MIN_RELATIVE_SCORE, MIN_SIMILARITY_SCORE
from rag.document_loader import load_knowledge_document
from rag.evidence import RetrievalEvidence
from rag.versioning import (
    EMBEDDING_MODEL_VERSION,
    file_content_hash,
    knowledge_namespace,
    knowledge_version,
    query_fingerprint,
    source_identity,
    KnowledgeSourceChanged,
)
from rag.vector_store import get_embedding_model

logger = logging.getLogger(__name__)

LOCAL_BACKEND = "inmemory"
POSTGRES_BACKEND = "pgvector"


@dataclass
class RetrievalOutcome:
    """一次检索的结果与观测数据。"""

    evidence: list[RetrievalEvidence] = field(default_factory=list)
    namespace: str = ""
    knowledge_version: str = ""
    query: str = ""
    backend: str = LOCAL_BACKEND
    cache_hit: bool = False
    embedding_cache_hit: bool = False
    latency_ms: float = 0.0
    indexed_documents: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence": [item.to_dict() for item in self.evidence],
            "namespace": self.namespace,
            "knowledge_version": self.knowledge_version,
            "query": self.query,
            "backend": self.backend,
            "cache_hit": self.cache_hit,
            "embedding_cache_hit": self.embedding_cache_hit,
            "latency_ms": self.latency_ms,
            "indexed_documents": self.indexed_documents,
        }


class KnowledgeRetriever:
    """按任务检索知识，并把可重算的部分放进缓存。"""

    def __init__(
        self,
        *,
        owner: str,
        cache: KnowledgeCache,
        persistent: bool,
        session_factory=None,
        embedding_model=None,
    ) -> None:
        self._owner = owner
        self._cache = cache
        self._persistent = persistent
        self._session_factory = session_factory
        self._embedding_model = embedding_model

    # -- 对外入口 ---------------------------------------------------------

    def retrieve(
        self,
        *,
        knowledge_sources: Sequence[str],
        query: str,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = MIN_SIMILARITY_SCORE,
    ) -> RetrievalOutcome:
        started = time.perf_counter()
        sources = [str(item) for item in knowledge_sources]
        if not sources or not query.strip():
            return RetrievalOutcome(
                query=query,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )

        # 文档缺失会在这里抛错：不能靠缓存或回退把授权/来源问题掩盖过去
        namespace = knowledge_namespace(self._owner, sources)
        version = knowledge_version(sources)
        backend = POSTGRES_BACKEND if self._persistent else LOCAL_BACKEND

        key = self._cache.retrieval_key(
            namespace=namespace,
            knowledge_version=version,
            query_hash=query_fingerprint(query),
            embedding_model_version=EMBEDDING_MODEL_VERSION,
            top_k=top_k,
            min_score=min_score,
            backend=backend,
        )

        cached = self._cache.get_json(key)
        if isinstance(cached, list):
            return RetrievalOutcome(
                evidence=[RetrievalEvidence.from_dict(item) for item in cached],
                namespace=namespace,
                knowledge_version=version,
                query=query,
                backend=backend,
                cache_hit=True,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )

        embedding, embedding_hit = self._embed_query(query)
        indexed = 0
        if self._persistent:
            indexed, evidence = self._search_persistent(
                namespace=namespace,
                version=version,
                sources=sources,
                query=query,
                embedding=embedding,
                top_k=top_k,
                min_score=min_score,
            )
        else:
            evidence = self._search_local(
                namespace=namespace,
                version=version,
                sources=sources,
                query=query,
                embedding=embedding,
                top_k=top_k,
                min_score=min_score,
            )

        if knowledge_version(sources) != version:
            raise KnowledgeSourceChanged("知识源在检索期间已变化，请新建Run并重新取证")
        payload = [item.to_dict() for item in evidence]
        self._cache.set_json(key, payload)

        return RetrievalOutcome(
            evidence=evidence,
            namespace=namespace,
            knowledge_version=version,
            query=query,
            backend=backend,
            embedding_cache_hit=embedding_hit,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            indexed_documents=indexed,
        )

    def index(
        self,
        *,
        knowledge_sources: Sequence[str],
    ) -> dict[str, Any]:
        """显式索引知识库（postgres 模式）；local 模式返回说明。"""
        sources = [str(item) for item in knowledge_sources]
        if not sources:
            return {"indexed": 0, "backend": LOCAL_BACKEND}
        if not self._persistent:
            return {"indexed": 0, "backend": LOCAL_BACKEND, "note": "local 模式不持久化索引"}

        namespace = knowledge_namespace(self._owner, sources)
        version = knowledge_version(sources)
        indexed = self._ensure_indexed(namespace=namespace, version=version, sources=sources)
        return {"indexed": indexed, "namespace": namespace, "backend": POSTGRES_BACKEND}

    def delete(self, *, knowledge_sources: Sequence[str]) -> dict[str, Any]:
        """删除知识库：文档与向量一起移除，缓存按命名空间清理。"""
        sources = [str(item) for item in knowledge_sources]
        if not sources:
            return {"documents": 0}
        namespace = knowledge_namespace(self._owner, sources)
        if not self._persistent:
            return {"documents": 0, "namespace": namespace, "backend": LOCAL_BACKEND}

        factory = self._session_factory
        assert factory is not None
        from storage.knowledge import PostgresKnowledgeIndex

        with factory() as session:
            index = PostgresKnowledgeIndex(session)
            removed = index.delete_namespace(namespace)
            session.commit()
        self._cache.flush_namespace(namespace)
        return {"documents": removed, "namespace": namespace, "backend": POSTGRES_BACKEND}

    def stats(self, *, knowledge_sources: Sequence[str]) -> dict[str, Any]:
        sources = [str(item) for item in knowledge_sources]
        namespace = knowledge_namespace(self._owner, sources) if sources else ""
        payload: dict[str, Any] = {
            "namespace": namespace,
            "cache": self._cache.measured(),
            "backend": POSTGRES_BACKEND if self._persistent else LOCAL_BACKEND,
        }
        if sources:
            payload["knowledge_version"] = knowledge_version(sources)
        if self._persistent and namespace:
            from storage.knowledge import PostgresKnowledgeIndex

            factory = self._session_factory
            assert factory is not None
            with factory() as session:
                payload["index"] = PostgresKnowledgeIndex(session).stats(namespace)
        return payload

    # -- 内部 -------------------------------------------------------------

    def _embedding(self):
        if self._embedding_model is None:
            self._embedding_model = get_embedding_model()
        return self._embedding_model

    def _embed_query(self, query: str) -> tuple[list[float], bool]:
        key = self._cache.embedding_key(
            query_hash=query_fingerprint(query),
            embedding_model_version=EMBEDDING_MODEL_VERSION,
        )
        cached = self._cache.get_embedding(key)
        if cached is not None:
            return cached, True
        embedding = [float(value) for value in self._embedding().embed_query(query)]
        self._cache.set_embedding(key, embedding)
        return embedding, False

    def _search_local(
        self,
        *,
        namespace: str,
        version: str,
        sources: list[str],
        query: str,
        embedding: list[float],
        top_k: int,
        min_score: float,
    ) -> list[RetrievalEvidence]:
        from rag.vector_store import (
            create_and_cache_vector_store,
            get_cached_vector_store,
        )

        cache_key = f"{namespace}:{version}"
        vector_store = get_cached_vector_store(cache_key)

        if vector_store is None:
            documents = _load_and_chunk(namespace=namespace, version=version, sources=sources)
            vector_store = create_and_cache_vector_store(
                documents=documents, cache_key=cache_key, embeddings=self._embedding()
            )

        results = vector_store.similarity_search_with_score_by_vector(embedding=embedding, k=top_k)
        best = max((float(score) for _, score in results), default=0.0)
        cutoff = max(min_score, best * MIN_RELATIVE_SCORE)

        evidence: list[RetrievalEvidence] = []
        for document, score in results:
            if float(score) < cutoff:
                continue
            evidence.append(
                RetrievalEvidence(
                    chunk_id=document.metadata["chunk_id"],
                    document_id=document.metadata["document_id"],
                    source=str(document.metadata.get("source") or ""),
                    page=document.metadata.get("page"),
                    content=document.page_content.strip(),
                    score=float(score),
                    knowledge_version=version,
                    embedding_model_version=EMBEDDING_MODEL_VERSION,
                    namespace=namespace,
                    query=query,
                )
            )
        return evidence

    def _search_persistent(
        self,
        *,
        namespace: str,
        version: str,
        sources: list[str],
        query: str,
        embedding: list[float],
        top_k: int,
        min_score: float,
    ) -> tuple[int, list[RetrievalEvidence]]:
        from storage.knowledge import PostgresKnowledgeIndex

        factory = self._session_factory
        assert factory is not None

        with factory() as session:
            index = PostgresKnowledgeIndex(session)
            index.lock_namespace(namespace)
            indexed = self._index_into(index, namespace=namespace, version=version, sources=sources)

            hits = index.search(
                namespaces=[namespace],
                embedding=embedding,
                top_k=top_k,
                min_score=min_score,
                embedding_model_version=EMBEDDING_MODEL_VERSION,
                knowledge_version=version,
            )
            session.commit()

        best = max((hit.score for hit in hits), default=0.0)
        cutoff = max(min_score, best * MIN_RELATIVE_SCORE)

        evidence = [
            RetrievalEvidence(
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                source=hit.source,
                page=hit.page,
                content=hit.content,
                score=hit.score,
                knowledge_version=version,
                embedding_model_version=EMBEDDING_MODEL_VERSION,
                namespace=namespace,
                query=query,
            )
            for hit in hits
            if hit.score >= cutoff
        ]
        return indexed, evidence

    def _ensure_indexed(
        self, *, namespace: str, version: str, sources: list[str]
    ) -> int:
        from storage.knowledge import PostgresKnowledgeIndex

        factory = self._session_factory
        assert factory is not None
        with factory() as session:
            index = PostgresKnowledgeIndex(session)
            index.lock_namespace(namespace)
            indexed = self._index_into(
                index,
                namespace=namespace,
                version=version,
                sources=sources,
            )
            session.commit()
        return indexed

    def _index_into(
        self,
        index,
        *,
        namespace: str,
        version: str,
        sources: list[str],
    ) -> int:
        from storage.knowledge import IndexedChunk

        indexed = 0
        for source in sources:
            path = Path(source)
            content_hash = file_content_hash(path)
            if index.has_document(
                namespace=namespace, source=source_identity(path), content_hash=content_hash,
                knowledge_version=version, embedding_model_version=EMBEDDING_MODEL_VERSION,
            ):
                continue

            documents = load_knowledge_document(source)
            chunks = split_documents(documents)
            if file_content_hash(path) != content_hash:
                raise KnowledgeSourceChanged("知识源在加载期间已变化，请新建Run并重新取证")
            texts = [chunk.page_content for chunk in chunks]
            vectors = self._embedding().embed_documents(texts) if texts else []
            index.put_document(
                namespace=namespace,
                owner=self._owner,
                source=source_identity(path),
                content_hash=content_hash,
                knowledge_version=version,
                embedding_model_version=EMBEDDING_MODEL_VERSION,
                byte_size=path.stat().st_size,
                chunks=[
                    IndexedChunk(
                        index=position,
                        content=chunk.page_content,
                        embedding=vector,
                        page=chunk.metadata.get("page"),
                    )
                    for position, (chunk, vector) in enumerate(zip(chunks, vectors))
                ],
            )
            indexed += 1
        return indexed


def _load_and_chunk(
    *, namespace: str, version: str, sources: list[str]
) -> list[Document]:
    """加载并切分，同时给每个片段分配与持久化后端一致的稳定 id。"""
    import hashlib

    documents: list[Document] = []

    for source in sources:
        path = Path(source)
        content_hash = file_content_hash(path)
        document_id = hashlib.sha256(
            f"{namespace}|{source_identity(path)}|{content_hash}".encode("utf-8")
        ).hexdigest()[:32]
        loaded = split_documents(load_knowledge_document(source))
        if file_content_hash(path) != content_hash:
            raise KnowledgeSourceChanged("知识源在加载期间已变化，请新建Run并重新取证")
        for position, chunk in enumerate(loaded):
            chunk_id = hashlib.sha256(
                f"{document_id}|{position}".encode("utf-8")
            ).hexdigest()[:32]
            chunk.metadata.update(chunk_id=chunk_id, document_id=document_id)
        documents.extend(loaded)

    return documents


_retriever: KnowledgeRetriever | None = None
_RETRIEVER_LOCK = RLock()


def get_retriever() -> KnowledgeRetriever:
    """复用当前配置/owner实例，不改变其他请求正在使用的实例。"""
    global _retriever
    import storage

    config = storage.current_config()
    owner = storage.current_owner()
    # Each returned object captures one trusted owner; swapping the cached
    # object never mutates an in-flight retriever used by another request.
    with _RETRIEVER_LOCK:
        if _retriever is not None and _retriever._owner == owner and getattr(_retriever, "_config", None) == config:
            return _retriever
    cache = KnowledgeCache(
        config.redis_url if config.redis_enabled else None,
        enabled=config.redis_enabled,
        ttl_seconds=config.redis_ttl_seconds,
    )
    session_factory = None
    if config.is_postgres:
        session_factory = storage.get_session_factory(
            storage.database_url(), pool_size=config.pool_size
        )
    retriever = KnowledgeRetriever(
        owner=owner,
        cache=cache,
        persistent=config.is_postgres,
        session_factory=session_factory,
    )
    retriever._config = config
    with _RETRIEVER_LOCK:
        _retriever = retriever
    return retriever


def reset_retriever() -> None:
    """测试用：丢弃检索器与包含片段元数据的有界向量缓存。"""
    global _retriever
    _retriever = None
    from rag.vector_store import _VECTOR_STORE_CACHE, _CACHE_LOCK
    with _CACHE_LOCK:
        _VECTOR_STORE_CACHE.clear()


__all__ = [
    "LOCAL_BACKEND",
    "POSTGRES_BACKEND",
    "KnowledgeRetriever",
    "RetrievalOutcome",
    "get_retriever",
    "reset_retriever",
]
