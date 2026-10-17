"""知识库的持久化向量表（仅 PostgreSQL + pgvector）。

文档身份由 `(namespace, source, content_hash)` 确定：同一份内容重复索引是
幂等的；同名文件内容变化会插入新文档并把旧版本置为非活跃，旧向量因此不再
被检索到——删除与更新都不需要手工清理缓存。

维度固定为 `EMBEDDING_DIM`。换用不同维度的 embedding 模型需要一次真实的
schema 变更，不能靠"先跑起来再说"：维度不符时索引会直接拒绝写入。
"""

from __future__ import annotations

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

#: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 的输出维度。
EMBEDDING_DIM = 384


class KnowledgeBase(DeclarativeBase):
    pass


class KnowledgeDocument(KnowledgeBase):
    __tablename__ = "knowledge_documents"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    namespace: Mapped[str] = mapped_column(String(128), nullable=False)
    owner: Mapped[str] = mapped_column(String(128), nullable=False)
    source: Mapped[str] = mapped_column(String(512), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    knowledge_version: Mapped[str] = mapped_column(String(64), nullable=False)
    byte_size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint(
            "namespace", "source", "content_hash", name="uq_knowledge_document_identity"
        ),
        Index("ix_knowledge_documents_active", "namespace", "is_active"),
    )


class KnowledgeChunk(KnowledgeBase):
    __tablename__ = "knowledge_chunks"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("knowledge_documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    namespace: Mapped[str] = mapped_column(String(128), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    page: Mapped[int | None] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)
    embedding_model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_chunk_document_index"),
        Index("ix_knowledge_chunks_namespace", "namespace"),
    )


__all__ = [
    "EMBEDDING_DIM",
    "KnowledgeBase",
    "KnowledgeChunk",
    "KnowledgeDocument",
]
