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

#: 作用域。写入前必须先分类，而不是一律 global——否则"这次的报告用红色"会被
#: 当成永久偏好，污染以后所有会话。
SCOPE_GLOBAL = "global"
SCOPE_DATASET_PREFIX = "dataset:"
SCOPE_SESSION_PREFIX = "session:"

#: 指向"当前这次"的措辞：结论只对本次分析成立
_SESSION_MARKERS = (
    "这次",
    "本次",
    "这一轮",
    "这一版",
    "这个报告",
    "这份报告",
    "此报告",
    "当前会话",
)

#: 指向"某份数据"的措辞：结论对该数据集成立，但对别的数据不一定
_DATASET_MARKERS = (
    "这张表",
    "这个表",
    "该表",
    "这个数据集",
    "该数据集",
    "这份数据",
    "这批数据",
    "这个库",
)


class MemoryValidationError(ValueError):
    """记忆内容不可接受。"""


class MemoryService:
    def __init__(self, *, owner: str | None = None) -> None:
        self._explicit_owner = owner

    @property
    def _owner(self) -> str:
        """显式传入的优先；否则现取当前请求的归属者（不在构造时缓存）。"""
        return self._explicit_owner or storage.current_owner()

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
        scopes = [SCOPE_GLOBAL]
        for ref in dataset_refs:
            from pathlib import Path

            scopes.append(f"{SCOPE_DATASET_PREFIX}{Path(ref).stem}")
        return scopes

    # -- 写入前的作用域分类 -----------------------------------------------

    @staticmethod
    def classify_scope(
        text: str,
        *,
        dataset_refs: list[str] | None = None,
        session_id: str | None = None,
    ) -> tuple[str, str]:
        """判断一条记忆该落在哪个作用域，返回 `(scope, reason)`。

        规则是确定性的、可解释的，且**保守**：拿不准就 global，因为 global 的
        记忆会被所有会话读到，错放一条的代价比漏放一条大得多。

        优先级：
        1. 文中点名了本次的某个数据集 → `dataset:<stem>`（跨会话，但只对该数据）
        2. 出现"这次/本次/这份报告"这类措辞 → `session:<id>`（会话内有效）
        3. 出现"这张表/这个数据集"但没有点名 → 只有一个数据集时归它，否则 global
        4. 其余视为稳定偏好 → `global`
        """
        content = (text or "").strip()
        refs = [ref for ref in (dataset_refs or []) if ref]
        stems = []
        for ref in refs:
            from pathlib import Path

            stems.append(Path(ref).stem)

        lowered = content.lower()
        for stem in stems:
            if stem and stem.lower() in lowered:
                return f"{SCOPE_DATASET_PREFIX}{stem}", f"点名了数据集 {stem}"

        if any(marker in content for marker in _SESSION_MARKERS):
            if session_id:
                return f"{SCOPE_SESSION_PREFIX}{session_id}", "措辞指向本次分析"
            if len(stems) == 1:
                return f"{SCOPE_DATASET_PREFIX}{stems[0]}", "措辞指向本次分析，且只有一个数据集"
            return SCOPE_GLOBAL, "措辞指向本次分析，但缺少会话与数据集上下文"

        if any(marker in content for marker in _DATASET_MARKERS):
            if len(stems) == 1:
                return f"{SCOPE_DATASET_PREFIX}{stems[0]}", "措辞指向数据集，且只有一个数据集"
            return SCOPE_GLOBAL, "措辞指向数据集，但本次有多个数据集"

        return SCOPE_GLOBAL, "稳定偏好"

    # -- 候选 -------------------------------------------------------------

    def propose_candidates(
        self,
        text: str,
        *,
        provenance: dict[str, Any] | None = None,
        dataset_refs: list[str] | None = None,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """从用户消息里提出候选记忆。只记录，不确认。

        作用域在**写入前**分类：默认 global 会让"这次报告用红色"变成永久偏好。
        """
        content = (text or "").strip()
        if not content or len(content) > MAX_VALUE_LENGTH:
            return []
        if not any(marker in content for marker in _PREFERENCE_MARKERS):
            return []

        scope, reason = self.classify_scope(
            content, dataset_refs=dataset_refs, session_id=session_id
        )
        candidates = [
            {
                "kind": "preference",
                "key": "explicit_preference",
                "value": content,
                "scope": scope,
                "status": CANDIDATE,
                "provenance": {
                    **(provenance or {}),
                    "source": "user_message_heuristic",
                    "scope_reason": reason,
                },
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
