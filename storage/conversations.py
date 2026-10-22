"""会话、轮次与消息的持久化。

三条不变量：

1. **Turn 与 Run 分离**：一条新消息创建新的 Turn 和新的 Run；`resume` 只用于
   继续某个 Run 的 HITL / Checkpoint，不能把新消息当成旧 Run 的恢复。
2. **幂等**：同一会话内 `client_message_id` 唯一，客户端重试不会创建第二个
   Turn 或第二个 Run。
3. **轮次序号唯一**：`(session_id, seq)` 唯一约束保证并发提交不会出现两个
   "第 3 轮"。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from storage.models import (
    ConversationMessage,
    ConversationSession,
    ConversationTurn,
    SessionRunLink,
)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ConversationRepository:
    """会话读写。所有方法只接收一个 Session，事务边界由调用方决定。"""

    def __init__(self, session: Session, *, owner: str) -> None:
        self._session = session
        self._owner = owner

    # -- 会话 -------------------------------------------------------------

    def create_session(
        self,
        *,
        title: str | None = None,
        dataset_refs: list[str] | None = None,
        knowledge_refs: list[str] | None = None,
    ) -> ConversationSession:
        record = ConversationSession(
            id=new_id("sess"),
            owner=self._owner,
            title=title,
            dataset_refs=list(dataset_refs or []),
            knowledge_refs=list(knowledge_refs or []),
        )
        self._session.add(record)
        self._session.flush()
        return record

    def get_session(self, session_id: str) -> ConversationSession | None:
        record = self._session.get(ConversationSession, session_id)
        if record is None or record.owner != self._owner or record.deleted_at:
            return None
        return record

    def lock_session(self, session_id: str) -> ConversationSession | None:
        """Serialize sequence allocation and client-message deduplication."""
        if self._session.get_bind().dialect.name == "sqlite":
            self._session.connection().exec_driver_sql("BEGIN IMMEDIATE")
        statement = select(ConversationSession).where(
            ConversationSession.id == session_id,
            ConversationSession.owner == self._owner,
            ConversationSession.deleted_at.is_(None),
        ).with_for_update()
        return self._session.scalars(statement).first()

    def list_sessions(self, *, limit: int = 50) -> list[ConversationSession]:
        statement = (
            select(ConversationSession)
            .where(
                ConversationSession.owner == self._owner,
                ConversationSession.deleted_at.is_(None),
            )
            .order_by(ConversationSession.created_at.desc())
            .limit(limit)
        )
        return list(self._session.scalars(statement))

    def delete_session(self, session_id: str) -> bool:
        """软删除：消息与记忆保留可追溯，但会话不再出现在列表里。"""
        record = self.get_session(session_id)
        if record is None:
            return False
        record.deleted_at = _now()
        return True

    # -- 消息 -------------------------------------------------------------

    def add_message(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        turn_id: str | None = None,
        run_id: str | None = None,
        client_message_id: str | None = None,
    ) -> ConversationMessage:
        if self.get_session(session_id) is None:
            raise KeyError("会话不存在")
        if role == "assistant" and turn_id:
            client_message_id = f"assistant:{turn_id}"
            existing = self.find_message_by_client_id(session_id, client_message_id)
            if existing is not None:
                return existing
        record = ConversationMessage(
            id=new_id("msg"),
            session_id=session_id,
            turn_id=turn_id,
            role=role,
            content=content,
            run_id=run_id,
            client_message_id=client_message_id,
            created_at=_now(),
        )
        if role == "assistant":
            try:
                with self._session.begin_nested():
                    self._session.add(record)
                    self._session.flush()
            except IntegrityError:
                existing = self.find_message_by_client_id(session_id, client_message_id)
                if existing is None:
                    raise
                return existing
        else:
            self._session.add(record)
            self._session.flush()
        return record

    def find_message_by_client_id(
        self, session_id: str, client_message_id: str
    ) -> ConversationMessage | None:
        if self.get_session(session_id) is None:
            return None
        statement = select(ConversationMessage).where(
            ConversationMessage.session_id == session_id,
            ConversationMessage.client_message_id == client_message_id,
        )
        return self._session.scalars(statement).first()

    def list_messages(self, session_id: str, *, limit: int = 200) -> list[ConversationMessage]:
        if self.get_session(session_id) is None:
            return []
        statement = (
            select(ConversationMessage)
            .where(ConversationMessage.session_id == session_id)
            .order_by(ConversationMessage.created_at.asc(), ConversationMessage.id.asc())
            .limit(limit)
        )
        return list(self._session.scalars(statement))

    def recent_messages(self, session_id: str, *, limit: int = 6) -> list[ConversationMessage]:
        if self.get_session(session_id) is None:
            return []
        statement = (select(ConversationMessage)
            .where(ConversationMessage.session_id == session_id)
            .order_by(ConversationMessage.created_at.desc(), ConversationMessage.id.desc())
            .limit(limit))
        return list(reversed(list(self._session.scalars(statement))))

    def assistant_for_turn(self, turn_id: str) -> ConversationMessage | None:
        if self.get_turn(turn_id) is None:
            return None
        statement = select(ConversationMessage).where(
            ConversationMessage.turn_id == turn_id,
            ConversationMessage.role == "assistant",
        )
        return self._session.scalars(statement).first()

    # -- 轮次 -------------------------------------------------------------

    def next_turn_seq(self, session_id: str) -> int:
        statement = select(func.max(ConversationTurn.seq)).where(
            ConversationTurn.session_id == session_id
        )
        current = self._session.scalar(statement)
        return int(current or 0) + 1

    def create_turn(
        self,
        *,
        session_id: str,
        run_id: str,
        user_message_id: str | None = None,
        status: str = "pending",
    ) -> ConversationTurn:
        if self.get_session(session_id) is None:
            raise KeyError("会话不存在")
        record = ConversationTurn(
            id=new_id("turn"),
            session_id=session_id,
            seq=self.next_turn_seq(session_id),
            user_message_id=user_message_id,
            run_id=run_id,
            status=status,
            result={},
        )
        self._session.add(record)
        self._session.flush()
        return record

    def get_turn(self, turn_id: str) -> ConversationTurn | None:
        record = self._session.get(ConversationTurn, turn_id)
        if record is None or self.get_session(record.session_id) is None:
            return None
        return record

    def list_turns(self, session_id: str, *, limit: int = 50) -> list[ConversationTurn]:
        if self.get_session(session_id) is None:
            return []
        statement = (
            select(ConversationTurn)
            .where(ConversationTurn.session_id == session_id)
            .order_by(ConversationTurn.seq.asc())
            .limit(limit)
        )
        return list(self._session.scalars(statement))

    def turn_for_run(self, run_id: str) -> ConversationTurn | None:
        statement = select(ConversationTurn).where(ConversationTurn.run_id == run_id)
        record = self._session.scalars(statement).first()
        return self.get_turn(record.id) if record is not None else None

    def update_turn(
        self,
        turn_id: str,
        *,
        status: str | None = None,
        status_reason: str | None = None,
        result: dict[str, Any] | None = None,
    ) -> ConversationTurn | None:
        record = self.get_turn(turn_id)
        if record is None:
            return None
        if status is not None:
            record.status = status
        if status_reason is not None:
            record.status_reason = status_reason
        if result is not None:
            record.result = result
        record.updated_at = _now()
        return record

    # -- Run 关联 ---------------------------------------------------------

    def link_run(self, *, session_id: str, turn_id: str, run_id: str) -> SessionRunLink:
        if self.get_session(session_id) is None or self.get_turn(turn_id) is None:
            raise KeyError("会话或轮次不存在")
        existing = self._session.get(SessionRunLink, run_id)
        if existing is not None:
            return existing
        link = SessionRunLink(
            run_id=run_id, session_id=session_id, turn_id=turn_id
        )
        self._session.add(link)
        self._session.flush()
        return link

    def session_for_run(self, run_id: str) -> ConversationSession | None:
        link = self._session.get(SessionRunLink, run_id)
        if link is None:
            return None
        return self.get_session(link.session_id)

    def recent_turns(self, session_id: str, *, limit: int = 3) -> list[ConversationTurn]:
        """最近若干轮（倒序取，再按时间正序返回）。"""
        if self.get_session(session_id) is None:
            return []
        statement = (
            select(ConversationTurn)
            .where(ConversationTurn.session_id == session_id)
            .order_by(ConversationTurn.seq.desc())
            .limit(limit)
        )
        return list(reversed(list(self._session.scalars(statement))))


__all__ = ["ConversationRepository", "new_id"]
