"""节点获取运行期依赖的统一入口。

各 Agent 需要"本 Agent 的 LLM 实例"，表相关节点需要真实 DataFrame。
两者都由调用方按 run 构造，并通过 ContextVar 绑定（见 core/runtime.py），
不放进 state：

- state 会写进 Checkpoint，模型客户端与 DataFrame 都无法序列化，
  跨进程恢复后也没有意义；
- ContextVar 按上下文隔离，并发 Run 之间不会串值。

模型实例按 run_id + agent_name 缓存，因此一次运行内同一 Agent 复用同一连接池。
"""

from __future__ import annotations

from typing import Any

from core.llm import ModelFactory
from core.runtime import current_runtime
from ingestion.models import TableResource
from state import InsightFlowState


def model_for(agent_name: str) -> Any:
    """取该 Agent 本次运行要用的模型实例。"""
    return current_runtime().model_factory.for_agent(agent_name)


def resolve_tables(state: InsightFlowState) -> dict[str, TableResource]:
    """按快照把表资源读回来。

    不直接用 state 里的快照构造 TableResource，而是经 resolver 校验来源内容：
    数据在快照之后被改动时明确失败，避免用新数据得出与历史 Run 不一致的结论。
    """
    from ingestion.resources import snapshots_from_state

    snapshots = snapshots_from_state(state.get("table_snapshot"))

    if not snapshots:
        raise RuntimeError(
            "state 中缺少 table_snapshot，无法解析表资源；"
            "请确认 loader 节点已运行"
        )

    return current_runtime().resolver.resolve_all(snapshots)


def factory_for_current_run() -> ModelFactory:
    return current_runtime().model_factory


def resolve_snapshots(snapshot_state: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """给 loader 节点用：用一份临时快照试解析，失败即在此暴露。"""
    from ingestion.resources import snapshots_from_state

    snapshots = snapshots_from_state(snapshot_state)
    return current_runtime().resolver.resolve_all(snapshots)
