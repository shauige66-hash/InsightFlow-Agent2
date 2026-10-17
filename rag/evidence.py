"""检索证据：可序列化、可追溯，能跟着分析结果一起进报告。

这里只描述"哪一段文字被当作什么依据"，不携带任何执行指令。业务知识只能
影响业务事实与指标定义，不能改变工具权限、预算或系统行为——它的来源是用户
上传的文档，属于不受信任内容。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class RetrievalEvidence:
    chunk_id: str
    document_id: str
    source: str
    content: str
    score: float
    knowledge_version: str
    embedding_model_version: str
    namespace: str
    page: int | None = None
    query: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RetrievalEvidence":
        return cls(
            chunk_id=str(payload.get("chunk_id") or ""),
            document_id=str(payload.get("document_id") or ""),
            source=str(payload.get("source") or ""),
            content=str(payload.get("content") or ""),
            score=float(payload.get("score") or 0.0),
            knowledge_version=str(payload.get("knowledge_version") or ""),
            embedding_model_version=str(payload.get("embedding_model_version") or ""),
            namespace=str(payload.get("namespace") or ""),
            page=payload.get("page"),
            query=payload.get("query"),
        )


def contents(evidence: Iterable[dict[str, Any] | RetrievalEvidence]) -> list[str]:
    """给 Prompt 用的正文列表。"""
    result: list[str] = []
    for item in evidence:
        text = item.content if isinstance(item, RetrievalEvidence) else item.get("content")
        if text:
            result.append(str(text))
    return result


def citations(evidence: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """去重后的引用：同一片段只出现一次，按来源与页码排序。"""
    seen: dict[str, dict[str, Any]] = {}
    for item in evidence:
        chunk_id = str(item.get("chunk_id") or "")
        if not chunk_id or chunk_id in seen:
            continue
        seen[chunk_id] = {
            "chunk_id": chunk_id,
            "source": item.get("source"),
            "page": item.get("page"),
            "score": round(float(item.get("score") or 0.0), 4),
            "knowledge_version": item.get("knowledge_version"),
        }
    return sorted(
        seen.values(), key=lambda entry: (entry["source"] or "", entry["page"] or 0)
    )


__all__ = ["RetrievalEvidence", "citations", "contents"]
