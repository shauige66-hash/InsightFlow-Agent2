# 统一组织业务知识文档的加载、切分、向量化和检索流程。

from typing import Any
from rag.config import DEFAULT_TOP_K


def build_and_retrieve_knowledge(
    knowledge_sources: list[str],
    query: str,
    top_k: int = DEFAULT_TOP_K
) -> list[dict[str, Any]]:

    if not knowledge_sources:
        return []

    # Compatibility facade: owner, versions, scores and caching all follow the
    # same implementation as the production workflow.
    from rag.retrieval import get_retriever

    outcome = get_retriever().retrieve(
        knowledge_sources=knowledge_sources, query=query, top_k=top_k,
    )
    return [item.to_dict() for item in outcome.evidence]
