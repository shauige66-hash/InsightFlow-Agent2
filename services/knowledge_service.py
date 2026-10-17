"""知识检索的服务入口。

节点只调用这里的函数，不直接碰 Redis、pgvector 或文件加载：

- `retrieve_global`：规划前的全局业务知识，让 Planner 理解业务术语；
- `retrieve_task`：某个任务的知识证据，供执行与校验使用；
- `evidence_from_state`：同一任务的多次尝试复用已取得的证据。

任务级检索的结果按 `task_id` 存进 state，因此 Executor 生成的代码进入修复或
精修时不会重新检索；知识版本变化会让旧证据失效（版本号参与缓存键）。
"""

from __future__ import annotations

from typing import Any

from rag.evidence import contents
from rag.retrieval import RetrievalOutcome, get_retriever
from rag.task_query import build_task_query

DEFAULT_GLOBAL_QUERY = "与当前数据分析相关的业务规则、指标定义和术语"


def _previous_user_messages(state: dict[str, Any]) -> list[str]:
    context = state.get("conversation_context") or {}
    messages = context.get("recent_messages") or []
    return [
        str(item.get("content"))
        for item in messages
        if item.get("role") == "user" and item.get("content")
    ]


def retrieve_global(state: dict[str, Any]) -> dict[str, Any]:
    """规划前的全局知识检索。"""
    knowledge_sources = [str(item) for item in (state.get("knowledge_sources") or [])]
    if not knowledge_sources:
        return {"retrieved_knowledge": [], "retrieved_knowledge_details": []}

    query = (
        state.get("analysis_focus")
        or DEFAULT_GLOBAL_QUERY
    )
    outcome = get_retriever().retrieve(
        knowledge_sources=knowledge_sources,
        query=query,
    )
    return {
        "retrieved_knowledge": contents(outcome.evidence),
        "retrieved_knowledge_details": [item.to_dict() for item in outcome.evidence],
        "knowledge_version": outcome.knowledge_version,
    }


def retrieve_task(
    state: dict[str, Any], task: dict[str, Any], *, focus: str | None = None
) -> RetrievalOutcome:
    """按任务检索知识；`focus` 是补充检索的额外依据。"""
    knowledge_sources = [str(item) for item in (state.get("knowledge_sources") or [])]
    query = build_task_query(
        task,
        analysis_focus=state.get("analysis_focus"),
        previous_context=_previous_user_messages(state),
        extra_terms=focus,
    )
    if not knowledge_sources:
        return RetrievalOutcome(query=query)

    return get_retriever().retrieve(
        knowledge_sources=knowledge_sources,
        query=query,
    )


def evidence_from_state(
    state: dict[str, Any], task_id: str
) -> list[dict[str, Any]]:
    """已为该任务取得的证据；没有则空列表。"""
    stored = state.get("task_knowledge") or {}
    return list(stored.get(str(task_id)) or [])


def _current_knowledge_version(state: dict[str, Any]) -> str | None:
    """当前知识源的内容版本；没有知识源时返回 None。"""
    sources = [str(item) for item in (state.get("knowledge_sources") or [])]
    if not sources:
        return None
    from rag.versioning import knowledge_version

    return knowledge_version(sources)


def task_knowledge_updates(
    state: dict[str, Any], task: dict[str, Any], *, focus: str | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """取任务证据，必要时检索；返回 `(证据, state 增量)`。

    已经在 state 里的任务不再检索：修复与精修复用第一次尝试的证据。
    两种例外：

    - `focus` 非空：动态决策要求补充检索，用新问句再查一次并按 chunk_id 合并；
    - 知识库版本变了：旧证据必须失效，否则会拿更新前的内容继续分析。
    """
    task_id = str(task.get("task_id"))
    existing = evidence_from_state(state, task_id)

    current_version = _current_knowledge_version(state)
    stored_version = state.get("knowledge_version")
    if current_version and stored_version and current_version != stored_version:
        existing = []

    if existing and not focus:
        return existing, {}

    outcome = retrieve_task(state, task, focus=focus)
    fresh = [item.to_dict() for item in outcome.evidence]
    merged = list(existing)
    seen = {str(item.get("chunk_id")) for item in existing}
    for item in fresh:
        if str(item.get("chunk_id")) not in seen:
            merged.append(item)
            seen.add(str(item.get("chunk_id")))

    updates: dict[str, Any] = {
        "task_knowledge": {
            **(state.get("task_knowledge") or {}),
            task_id: merged,
        },
    }
    if outcome.knowledge_version:
        updates["knowledge_version"] = outcome.knowledge_version
    updates["knowledge_cache_hit"] = outcome.cache_hit
    return merged, updates


def knowledge_contents(state: dict[str, Any], task_id: str) -> list[str]:
    """给 Prompt 用的证据正文；没有证据时为空列表。"""
    return contents(evidence_from_state(state, task_id))


__all__ = [
    "DEFAULT_GLOBAL_QUERY",
    "evidence_from_state",
    "knowledge_contents",
    "retrieve_global",
    "retrieve_task",
    "task_knowledge_updates",
]
