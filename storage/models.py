"""会话、消息、长期记忆的持久化模型。

这些表在 SQLite 与 PostgreSQL 上都能建：本地模式用同一个 ORM，避免为了
"能跑起来"再写一套只存在于测试里的存储。

知识库的向量表依赖 pgvector，单独放在 `storage.knowledge_models`，只在
PostgreSQL 模式下创建——SQLite 模式继续使用进程内索引。

权威状态划分：Run 的执行状态以 RunStore / Checkpoint 为准，这里的
`conversation_turns.status` 只是给会话界面用的展示状态，由服务层在
Run 状态变化时同步，不反过来当执行依据。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
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


class Base(DeclarativeBase):
    pass


class ConversationSession(Base):
    __tablename__ = "conversation_sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner: Mapped[str] = mapped_column(String(128), nullable=False)
    title: Mapped[str | None] = mapped_column(String(200))
    # 会话绑定的数据集与知识源：只存可序列化的身份，真实文件仍在文件系统
    dataset_refs: Mapped[list] = mapped_column(JSON, default=list)
    knowledge_refs: Mapped[list] = mapped_column(JSON, default=list)
    # 软删除：历史消息与记忆要能被追溯，不做物理级联删除
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_sessions_owner_created", "owner", "created_at"),
    )


class ConversationTurn(Base):
    """一轮用户请求：一个 Turn 通常对应一个分析 Run。"""

    __tablename__ = "conversation_turns"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("conversation_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    user_message_id: Mapped[str | None] = mapped_column(String(64))
    run_id: Mapped[str | None] = mapped_column(String(128))
    # pending / running / waiting_approval / completed / failed
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    status_reason: Mapped[str | None] = mapped_column(Text)
    # 该轮的结论与产物引用（不含完整数据行）
    result: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint("session_id", "seq", name="uq_turns_session_seq"),
        Index("ix_turns_session_seq", "session_id", "seq"),
    )


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("conversation_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    turn_id: Mapped[str | None] = mapped_column(String(64))
    # user / assistant / system
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    run_id: Mapped[str | None] = mapped_column(String(128))
    # 客户端重试的幂等键：同一会话内唯一
    client_message_id: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint(
            "session_id", "client_message_id", name="uq_messages_session_client"
        ),
        Index("ix_messages_session_created", "session_id", "created_at"),
    )


class SessionRunLink(Base):
    """Turn 与 Run 的关联，便于按 Run 反查会话。"""

    __tablename__ = "session_run_links"

    run_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("conversation_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    turn_id: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_links_session", "session_id"),
    )


class LongTermMemory(Base):
    """长期记忆：经用户确认、可跨会话复用的偏好与定义。

    `status` 区分候选与已确认：候选由服务层或模型提出，只有确认后才会被
    注入上下文。`provenance` 记录来源（哪个会话、哪个 Turn、谁确认的），
    便于回答"这条偏好是哪来的"。
    """

    __tablename__ = "long_term_memories"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner: Mapped[str] = mapped_column(String(128), nullable=False)
    # global / dataset:<name>
    scope: Mapped[str] = mapped_column(String(128), nullable=False, default="global")
    # preference / metric_definition / report_format
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    key: Mapped[str] = mapped_column(String(128), nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    # candidate / confirmed
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="candidate")
    provenance: Mapped[dict] = mapped_column(JSON, default=dict)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint(
            "owner", "scope", "kind", "key", name="uq_memories_identity"
        ),
        Index("ix_memories_owner_scope", "owner", "scope"),
    )


class SchemaVersion(Base):
    __tablename__ = "schema_migrations"

    version: Mapped[str] = mapped_column(String(32), primary_key=True)
    applied_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


__all__ = [
    "Base",
    "ConversationMessage",
    "ConversationSession",
    "ConversationTurn",
    "LongTermMemory",
    "SchemaVersion",
    "SessionRunLink",
]
