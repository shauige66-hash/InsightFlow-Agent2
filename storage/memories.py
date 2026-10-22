"""长期记忆的持久化与检索。

长期记忆只保存**经确认、可跨会话复用**的内容：用户偏好、指标定义、报告格式。
它既不是聊天记录，也不是知识库文档。

两条硬规则：

- 候选记忆（`status="candidate"`）不会被注入上下文，只有确认后才生效；
- 模型可以提出候选，但不能自行把推断写成用户事实——写入接口把 `source`
  记进 `provenance`，来源可追溯。

检索先用结构化字段与词元重合度，不引入向量：记忆条目少，可解释比"语义相似"
更重要，也避免为一个小表维护第二套 embedding。
"""

from __future__ import annotations

import re
import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from storage.models import LongTermMemory

CONFIRMED = "confirmed"
CANDIDATE = "candidate"

_TOKEN = re.compile(r"[0-9A-Za-z_]+|[\u4e00-\u9fff]+")
_CJK = re.compile(r"[\u4e00-\u9fff]+")


def _tokens(text: str) -> set[str]:
    """粗粒度词元：英文按词，中文保留原段与相邻二元组。

    中文没有空格，整段汉字会被当成一个词元，导致"销售额用什么货币"与
    "今后销售额统一用人民币元表示"零重合。用二元组才能让同一话题的表述
    真正匹配上。刻意保持简单：这里的目标是"不相关记忆不要注入"，不是追求
    检索指标。
    """
    result: set[str] = set()
    for match in _TOKEN.finditer(text or ""):
        piece = match.group(0).lower()
        if _CJK.fullmatch(piece):
            result.add(piece)
            for index in range(len(piece) - 1):
                result.add(piece[index : index + 2])
        else:
            result.add(piece)
    return result


def _now() -> datetime:
    return datetime.now(timezone.utc)


class MemoryRepository:
    def __init__(self, session: Session, *, owner: str) -> None:
        self._session = session
        self._owner = owner

    def add(
        self,
        *,
        kind: str,
        key: str,
        value: str,
        scope: str = "global",
        status: str = CANDIDATE,
        provenance: dict[str, Any] | None = None,
    ) -> LongTermMemory:
        """写入或更新一条记忆（按 owner+scope+kind+key 幂等）。"""
        existing = self._find(scope=scope, kind=kind, key=key)
        payload = dict(provenance or {})

        # A proposal is a pending revision, never an implicit revocation of a
        # confirmed setting. Repeated identical proposals remain idempotent.
        if existing is not None and existing.status == CONFIRMED and status == CANDIDATE:
            digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]
            payload["replaces_memory_id"] = existing.id
            key = f"pending:{key[:80]}:{digest}"
            existing = self._find(scope=scope, kind=kind, key=key)

        if existing is not None:
            existing.value = value
            existing.status = status
            existing.provenance = payload
            existing.deleted_at = None
            existing.updated_at = _now()
            return existing

        record = LongTermMemory(
            id=f"mem_{uuid.uuid4().hex}",
            owner=self._owner,
            scope=scope,
            kind=kind,
            key=key,
            value=value,
            status=status,
            provenance=payload,
        )
        self._session.add(record)
        self._session.flush()
        return record

    def _find(self, *, scope: str, kind: str, key: str) -> LongTermMemory | None:
        statement = select(LongTermMemory).where(
            LongTermMemory.owner == self._owner,
            LongTermMemory.scope == scope,
            LongTermMemory.kind == kind,
            LongTermMemory.key == key,
        )
        return self._session.scalars(statement).first()

    def get(self, memory_id: str) -> LongTermMemory | None:
        record = self._session.get(LongTermMemory, memory_id)
        if record is None or record.owner != self._owner or record.deleted_at:
            return None
        return record

    def list(
        self,
        *,
        scope: str | None = None,
        include_candidates: bool = False,
        limit: int = 100,
    ) -> list[LongTermMemory]:
        statement = select(LongTermMemory).where(
            LongTermMemory.owner == self._owner,
            LongTermMemory.deleted_at.is_(None),
        )
        if scope is not None:
            statement = statement.where(LongTermMemory.scope == scope)
        if not include_candidates:
            statement = statement.where(LongTermMemory.status == CONFIRMED)
        statement = statement.order_by(LongTermMemory.updated_at.desc()).limit(limit)
        return list(self._session.scalars(statement))

    def confirm(self, memory_id: str) -> LongTermMemory | None:
        record = self.get(memory_id)
        if record is None:
            return None
        replacement = (record.provenance or {}).get("replaces_memory_id")
        if replacement:
            target = self.get(str(replacement))
            if target is not None and target.scope == record.scope and target.kind == record.kind:
                target.value = record.value
                target.provenance = {**(record.provenance or {}), "confirmed_revision_id": record.id}
                target.status = CONFIRMED
                target.updated_at = _now()
                record.deleted_at = _now()
                record.status = CANDIDATE
                return target
        record.status = CONFIRMED
        record.updated_at = _now()
        return record

    def update(
        self,
        memory_id: str,
        *,
        value: str | None = None,
        scope: str | None = None,
        status: str | None = None,
    ) -> LongTermMemory | None:
        record = self.get(memory_id)
        if record is None:
            return None
        if value is not None:
            record.value = value
        if scope is not None:
            record.scope = scope
        if status is not None:
            if status not in (CANDIDATE, CONFIRMED):
                raise ValueError("未知的记忆状态")
            record.status = status
        record.updated_at = _now()
        return record

    def delete(self, memory_id: str) -> bool:
        record = self.get(memory_id)
        if record is None:
            return False
        record.deleted_at = _now()
        record.status = CANDIDATE
        return True

    def retrieve(
        self,
        query: str,
        *,
        scopes: Iterable[str] = ("global",),
        limit: int = 5,
        min_overlap: int = 1,
    ) -> list[tuple[LongTermMemory, int]]:
        """按词元重合度取少量相关记忆；返回 (记录, 重合度)。"""
        wanted = _tokens(query)
        allowed = set(scopes)
        if not wanted or not allowed:
            return []

        candidates = select(LongTermMemory).where(
            LongTermMemory.owner == self._owner,
            LongTermMemory.deleted_at.is_(None),
            LongTermMemory.status == CONFIRMED,
            LongTermMemory.scope.in_(allowed),
        )
        scored: list[tuple[LongTermMemory, int]] = []
        for record in self._session.scalars(candidates):
            overlap = len(wanted & _tokens(f"{record.key} {record.value}"))
            # Format preferences apply to their consumer even when the user
            # simply says "analyse sales" without repeating "report format".
            if overlap >= min_overlap or record.kind == "report_format":
                scored.append((record, overlap))

        # 先按重合度、再按新旧排序：同等相关时新的偏好覆盖旧偏好
        scored.sort(key=lambda item: (item[1], item[0].updated_at), reverse=True)
        return scored[:limit]


__all__ = ["CANDIDATE", "CONFIRMED", "MemoryRepository"]
