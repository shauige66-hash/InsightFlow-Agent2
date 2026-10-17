"""持久化知识索引（PostgreSQL + pgvector）。

与进程内索引的区别只有一点，但很关键：向量保存在数据库里，服务重启后
仍然可查，不需要重新加载文档、重新切分、重新 embedding。

身份与版本：

- 文档身份 = `(namespace, source, content_hash)`，同一份内容重复索引是幂等的；
- 同名文件内容变化 → 插入新文档、把旧的置为非活跃，旧向量自然退出检索；
- `namespace_version` 是所有活跃文档内容指纹的哈希，供缓存键与失效判断使用。

每次查询都按 `namespace` 过滤：不同知识库之间不会互相命中。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import Session

from storage.knowledge_models import EMBEDDING_DIM, KnowledgeChunk, KnowledgeDocument


def document_id_for(namespace: str, source: str, content_hash: str) -> str:
    payload = f"{namespace}|{source}|{content_hash}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


def chunk_id_for(document_id: str, index: int) -> str:
    return hashlib.sha256(f"{document_id}|{index}".encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True)
class IndexedChunk:
    """写入索引的单个片段。"""

    index: int
    content: str
    embedding: Sequence[float]
    page: int | None = None


@dataclass(frozen=True)
class ChunkHit:
    chunk_id: str
    document_id: str
    source: str
    page: int | None
    content: str
    score: float
    namespace: str


class PostgresKnowledgeIndex:
    """把知识文档与向量存进 PostgreSQL。"""

    def __init__(self, session: Session) -> None:
        self._session = session

    # -- 写入 -------------------------------------------------------------

    def put_document(
        self,
        *,
        namespace: str,
        owner: str,
        source: str,
        content_hash: str,
        knowledge_version: str,
        embedding_model_version: str,
        chunks: Iterable[IndexedChunk],
        byte_size: int = 0,
    ) -> tuple[str, bool]:
        """写入一份文档；返回 `(document_id, 是否复用已有索引)`。"""
        doc_id = document_id_for(namespace, source, content_hash)

        existing = self._session.get(KnowledgeDocument, doc_id)
        if existing is not None and existing.is_active:
            return doc_id, True

        chunks = list(chunks)
        for chunk in chunks:
            if len(chunk.embedding) != EMBEDDING_DIM:
                raise ValueError(
                    f"embedding 维度 {len(chunk.embedding)} 与索引维度 "
                    f"{EMBEDDING_DIM} 不一致；更换模型需要一次 schema 变更"
                )

        # 同名的旧版本退出检索（不物理删除，保留可追溯性）
        self._session.execute(
            update(KnowledgeDocument)
            .where(
                KnowledgeDocument.namespace == namespace,
                KnowledgeDocument.source == source,
                KnowledgeDocument.id != doc_id,
            )
            .values(is_active=False)
        )

        document = existing or KnowledgeDocument(id=doc_id)
        document.namespace = namespace
        document.owner = owner
        document.source = source
        document.content_hash = content_hash
        document.knowledge_version = knowledge_version
        document.byte_size = byte_size
        document.chunk_count = len(chunks)
        document.is_active = True
        self._session.add(document)

        # 重新索引同一 id 时先清掉旧片段，避免 (document_id, index) 冲突
        self._session.execute(
            delete(KnowledgeChunk).where(KnowledgeChunk.document_id == doc_id)
        )
        for chunk in chunks:
            self._session.add(
                KnowledgeChunk(
                    id=chunk_id_for(doc_id, chunk.index),
                    document_id=doc_id,
                    namespace=namespace,
                    chunk_index=chunk.index,
                    page=chunk.page,
                    content=chunk.content,
                    embedding=list(chunk.embedding),
                    embedding_model_version=embedding_model_version,
                )
            )
        self._session.flush()
        return doc_id, False

    def has_document(
        self, *, namespace: str, source: str, content_hash: str
    ) -> bool:
        """同样的内容是否已经索引过（跳过加载与 embedding 的依据）。"""
        doc_id = document_id_for(namespace, source, content_hash)
        record = self._session.get(KnowledgeDocument, doc_id)
        return record is not None and record.is_active and record.chunk_count > 0

    def deactivate_source(self, *, namespace: str, source: str) -> int:
        """删除某个来源：文档置为非活跃，片段随之不再被检索。"""
        result = self._session.execute(
            update(KnowledgeDocument)
            .where(
                KnowledgeDocument.namespace == namespace,
                KnowledgeDocument.source == source,
                KnowledgeDocument.is_active.is_(True),
            )
            .values(is_active=False)
        )
        return int(result.rowcount or 0)

    def delete_namespace(self, namespace: str) -> int:
        """删除整个知识库：先删片段，再删文档（外键级联之外显式清理）。"""
        document_ids = list(
            self._session.scalars(
                select(KnowledgeDocument.id).where(
                    KnowledgeDocument.namespace == namespace
                )
            )
        )
        if document_ids:
            self._session.execute(
                delete(KnowledgeChunk).where(
                    KnowledgeChunk.document_id.in_(document_ids)
                )
            )
        result = self._session.execute(
            delete(KnowledgeDocument).where(KnowledgeDocument.namespace == namespace)
        )
        return int(result.rowcount or 0)

    # -- 读取 -------------------------------------------------------------

    def namespace_version(self, namespace: str) -> str:
        """活跃文档内容指纹的哈希；文档更新或删除都会改变它。"""
        rows = self._session.execute(
            select(KnowledgeDocument.source, KnowledgeDocument.content_hash)
            .where(
                KnowledgeDocument.namespace == namespace,
                KnowledgeDocument.is_active.is_(True),
            )
            .order_by(KnowledgeDocument.source)
        )
        payload = "|".join(f"{source}:{content_hash}" for source, content_hash in rows)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def stats(self, namespace: str) -> dict[str, Any]:
        documents = self._session.scalar(
            select(func.count())
            .select_from(KnowledgeDocument)
            .where(
                KnowledgeDocument.namespace == namespace,
                KnowledgeDocument.is_active.is_(True),
            )
        )
        chunks = self._session.scalar(
            select(func.count())
            .select_from(KnowledgeChunk)
            .where(KnowledgeChunk.namespace == namespace)
        )
        return {"documents": int(documents or 0), "chunks": int(chunks or 0)}

    def search(
        self,
        *,
        namespaces: Sequence[str],
        embedding: Sequence[float],
        top_k: int,
        min_score: float,
        embedding_model_version: str | None = None,
    ) -> list[ChunkHit]:
        """按余弦相似度检索，返回相似度（越大越相关）。

        pgvector 的 `<=>` 是余弦**距离**，这里换算成 `1 - distance`，与进程内
        索引的分数方向保持一致——两者不能混用阈值。
        """
        if not namespaces:
            return []
        if len(embedding) != EMBEDDING_DIM:
            raise ValueError(
                f"查询向量维度 {len(embedding)} 与索引维度 {EMBEDDING_DIM} 不一致"
            )

        distance = KnowledgeChunk.embedding.cosine_distance(list(embedding))
        statement = (
            select(
                KnowledgeChunk,
                KnowledgeDocument.source,
                distance.label("distance"),
            )
            .join(KnowledgeDocument, KnowledgeDocument.id == KnowledgeChunk.document_id)
            .where(
                KnowledgeChunk.namespace.in_(list(namespaces)),
                KnowledgeDocument.is_active.is_(True),
            )
        )
        if embedding_model_version is not None:
            statement = statement.where(
                KnowledgeChunk.embedding_model_version == embedding_model_version
            )
        statement = statement.order_by(distance).limit(max(1, top_k))

        hits: list[ChunkHit] = []
        for chunk, source, distance_value in self._session.execute(statement):
            score = 1.0 - float(distance_value)
            if score < min_score:
                continue
            hits.append(
                ChunkHit(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    source=source,
                    page=chunk.page,
                    content=chunk.content,
                    score=score,
                    namespace=chunk.namespace,
                )
            )
        return hits


__all__ = [
    "ChunkHit",
    "IndexedChunk",
    "PostgresKnowledgeIndex",
    "chunk_id_for",
    "document_id_for",
]
