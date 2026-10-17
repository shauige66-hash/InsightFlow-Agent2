"""Agent Runtime Harness：模型调用、工具派发与执行预算的分层治理。

LangGraph 负责拓扑、路由、Checkpoint 与 HITL 中断；Harness 负责一次模型
调用或工具调用怎么被治理。两层之间只通过 `get_harness()` 的少数入口交互，
节点不再直接接触账本、预算计数或工具治理的上下文管理。

    harness.policy           哪些额度、什么时候判定、什么时候计数
    harness.llm_gateway      模型调用的唯一出口（幂等 + 账本 + 预算 + 追踪）
    harness.tool_dispatcher  Python / SQL / Plot 的统一派发
    harness.runtime          组装点与按执行上下文隔离的 state 绑定

底层仍是既有模块：`utils.llm_ledger`、`utils.idempotency`、`utils.llm_retry`、
`utils.agent_loop`、`utils.middleware`、`tools.tool_registry`。Harness 不重新
实现它们，只把调用顺序与边界固定下来。
"""

from harness.runtime import (
    AgentRuntimeHarness,
    StateNotBound,
    bind_state,
    current_state,
    get_harness,
)

__all__ = [
    "AgentRuntimeHarness",
    "StateNotBound",
    "bind_state",
    "current_state",
    "get_harness",
]
