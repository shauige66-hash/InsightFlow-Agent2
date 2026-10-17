"""长期记忆服务。

只保存**经确认、可跨会话复用**的偏好与定义。写入有两条约束：

1. 默认写入是候选（`candidate`），不会进入上下文；确认后才生效；
2. 明显敏感或属于某次分析临时数据的内容直接拒绝，而不是"先存着再说"。

候选提取用显式措辞（"今后"、"以后"、"统一用"、"优先用"）而不是让模型自由
发挥：没有真实语义模型时，宁可少提几条，也不把猜测写成用户事实。
"""

from __future__ import annotations

import logging
import re
from typing import Any

import storage
from storage.memories import CANDIDATE, CONFIRMED, MemoryRepository

logger = logging.getLogger(__name__)

ALLOWED_KINDS = {"preference", "metric_definition", "report_format"}
MAX_KEY_LENGTH = 120
MAX_VALUE_LENGTH = 500

#: 明显不该进长期记忆的内容
_SENSITIVE = (
    re.compile(r"pass(word|wd)?\s*[:=]", re.IGNORECASE),
    re.compile(r"api[\s_-]?key\s*[:=]", re.IGNORECASE),
    re.compile(r"secret\s*[:=]", re.IGNORECASE),
    re.compile(r"sk-[A-Za-z0-9]{16,}"),
    re.compile(r"postgres(ql)?://[^\s]+", re.IGNORECASE),
)

#: 显式偏好措辞：只有用户明确表达"今后要这样"才提候选
_PREFERENCE_MARKERS = ("今后", "以后", "之后统一", "统一用", "一律用", "优先用", "默认用")


class MemoryValidationError(ValueError):
    """记忆内容不可接受。"""


class MemoryService:
    def __init__(self, *, owner: str | None = None) -> None:
        self._owner = owner or storage.current_config().owner

    def _repository(self, session) -> MemoryRepository:
        return MemoryRepository(session, owner=self._owner)

    # -- 写 ---------------------------------------------------------------

    def add(
        self,
        *,
        kind: str,
        key: str,
        value: str,
        scope: str = "global",
        status: str = CANDIDATE,
        provenance: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._validate(kind=kind, key=key, value=value)
        if status not in (CANDIDATE, CONFIRMED):
            raise MemoryValidationError(f"未知的记忆状态：{status}")

        with storage.session_scope(storage.database_url()) as session:
            record = self._repository(session).add(
                kind=kind,
                key=key.strip(),
                value=value.strip(),
                scope=scope,
                status=status,
                provenance=provenance or {},
            )
            return _memory_dict(record)

    def confirm(self, memory_id: str) -> dict[str, Any] | None:
        with storage.session_scope(storage.database_url()) as session:
            record = self._repository(session).confirm(memory_id)
            return _memory_dict(record) if record is not None else None

    def update(
        self,
        memory_id: str,
        *,
        value: str | None = None,
        scope: str | None = None,
        status: str | None = None,
    ) -> dict[str, Any] | None:
        if value is not None:
            self._validate(kind="preference", key="x", value=value)
        with storage.session_scope(storage.database_url()) as session:
            record = self._repository(session).update(
                memory_id, value=value, scope=scope, status=status
            )
            return _memory_dict(record) if record is not None else None

    def delete(self, memory_id: str) -> bool:
        with storage.session_scope(storage.database_url()) as session:
            return self._repository(session).delete(memory_id)

    # -- 读 ---------------------------------------------------------------

    def list(
        self, *, scope: str | None = None, include_candidates: bool = False
    ) -> list[dict[str, Any]]:
        with storage.session_scope(storage.database_url()) as session:
            return [
                _memory_dict(record)
                for record in self._repository(session).list(
                    scope=scope, include_candidates=include_candidates
                )
            ]

    def get(self, memory_id: str) -> dict[str, Any] | None:
        with storage.session_scope(storage.database_url()) as session:
            record = self._repository(session).get(memory_id)
            return _memory_dict(record) if record is not None else None

    def retrieve(
        self, query: str, *, scopes: list[str] | None = None, limit: int = 5
    ) -> list[dict[str, Any]]:
        """按当前任务取少量已确认记忆。"""
        with storage.session_scope(storage.database_url()) as session:
            hits = self._repository(session).retrieve(
                query, scopes=scopes or ["global"], limit=limit
            )
            return [
                {**_memory_dict(record), "relevance": overlap}
                for record, overlap in hits
            ]

    def scopes_for(self, dataset_refs: list[str]) -> list[str]:
        """当前会话适用的作用域：全局 + 该数据集的专属记忆。"""
        scopes = ["global"]
        for ref in dataset_refs:
            from pathlib import Path

            scopes.append(f"dataset:{Path(ref).stem}")
        return scopes

    # -- 候选 -------------------------------------------------------------

    def propose_candidates(
        self,
        text: str,
        *,
        provenance: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """从用户消息里提出候选记忆。只记录，不确认。"""
        content = (text or "").strip()
        if not content or len(content) > MAX_VALUE_LENGTH:
            return []
        if not any(marker in content for marker in _PREFERENCE_MARKERS):
            return []

        candidates = [
            {
                "kind": "preference",
                "key": "explicit_preference",
                "value": content,
                "scope": "global",
                "status": CANDIDATE,
                "provenance": {**(provenance or {}), "source": "user_message_heuristic"},
            }
        ]
        return candidates

    # -- 校验 -------------------------------------------------------------

    @staticmethod
    def _validate(*, kind: str, key: str, value: str) -> None:
        if kind not in ALLOWED_KINDS:
            raise MemoryValidationError(
                f"不支持的记忆类型：{kind}；只允许 {sorted(ALLOWED_KINDS)}"
            )
        if not key or len(key) > MAX_KEY_LENGTH:
            raise MemoryValidationError("记忆 key 为空或过长")
        if not value or not value.strip():
            raise MemoryValidationError("记忆内容为空")
        if len(value) > MAX_VALUE_LENGTH:
            raise MemoryValidationError(
                f"记忆内容超过 {MAX_VALUE_LENGTH} 字符，长度更像一次分析结果而不是偏好"
            )
        for pattern in _SENSITIVE:
            if pattern.search(value):
                raise MemoryValidationError("记忆内容看起来包含凭据，拒绝写入")


def _memory_dict(record) -> dict[str, Any]:
    return {
        "memory_id": record.id,
        "owner": record.owner,
        "scope": record.scope,
        "kind": record.kind,
        "key": record.key,
        "value": record.value,
        "status": record.status,
        "provenance": record.provenance or {},
        "created_at": record.created_at.isoformat() if record.created_at else None,
        "updated_at": record.updated_at.isoformat() if record.updated_at else None,
    }


__all__ = [
    "ALLOWED_KINDS",
    "MemoryService",
    "MemoryValidationError",
]
