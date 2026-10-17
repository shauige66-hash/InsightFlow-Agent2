"""Harness 的组装点与运行上下文。

职责边界：

- LangGraph 负责拓扑、路由、Checkpoint 与 HITL 中断；
- Harness 负责一次模型调用或工具调用怎么被执行治理。

Harness 不持有预算本身，也不持有 Run 的运行期依赖：预算在
`execution_budget`（随 Checkpoint 持久化），模型工厂、解析器、输出目录在
`core.runtime.RunRuntime`（ContextVar，不持久化）。这里只做三件事：

1. 把当前节点的 state 绑进按执行上下文隔离的 `ContextVar`；
2. 组装 `LLMGateway` 与 `ToolDispatcher`，并把请求钩子装到 `utils.agent_loop`；
3. 给节点一组稳定入口，使它们不必再直接碰账本、预算计数与工具治理的上下文管理。

进程内只有一个 Harness 实例。它不随 Run 创建：实例本身无可变状态，
每次调用的差异全部来自 `ContextVar` 里的 state 与 `RunRuntime`，
因此并发 Run 天然隔离。
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Any, Callable, Iterator

from harness import policy
from harness.llm_gateway import LLMGateway, StateNotBound
from harness.tool_dispatcher import ToolDispatcher, ToolOutcome
from utils.context_budget import context_report_of

#: 当前节点正在处理的 state。
#:
#: 节点函数签名不能带额外参数（LangGraph 会把参数名当成状态通道），
#: 所以用 ContextVar 传递，避免改动全部节点签名。图会为每个节点调用一次
#: 包装器，进入时 set、退出时 reset，因此并发 Run 各看各的。
CURRENT_STATE: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "insightflow_harness_state", default=None
)


def current_state() -> dict[str, Any] | None:
    """当前执行上下文里的 state；不在节点内部时返回 None。"""
    return CURRENT_STATE.get()


@contextmanager
def bind_state(state: dict[str, Any]) -> Iterator[None]:
    """在 `with` 块内把 state 绑定到当前执行上下文。"""
    token = CURRENT_STATE.set(state)
    try:
        yield
    finally:
        CURRENT_STATE.reset(token)


class AgentRuntimeHarness:
    """节点可用的运行治理入口。

    节点只做四件事：读任务、构造输入、调这里、解释结果。预算怎么判、
    账本怎么写、工具治理的上下文怎么进出，都不再出现在节点里。
    """

    def __init__(self) -> None:
        self.policy = policy
        self.llm = LLMGateway(state_reader=current_state)
        self.tools = ToolDispatcher()
        # 钩子在 LangGraph 开始执行之前就要就位：Agent 通过 utils.llm_retry
        # 的回调进入它们，而那时节点已经在跑。
        self.llm.install_request_hooks()

    # -- 模型调用 ---------------------------------------------------------

    def model_call(
        self,
        agent: str,
        subject: str,
        invoke: Callable[[], Any],
        *,
        attempt: int = 1,
    ) -> tuple[Any, dict, dict]:
        """执行一次 Agent 逻辑调用（幂等 + 账本 + 预算 + 追踪）。"""
        return self.llm.call_agent(agent, subject, invoke, attempt=attempt)

    # -- 工具调用 ---------------------------------------------------------

    def run_tool(
        self,
        state: dict[str, Any],
        *,
        node: str,
        tool: str,
        arguments: dict[str, Any],
        snapshots: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> ToolOutcome:
        """派发一次工具调用；预算耗尽时抛 `BudgetExceeded`。"""
        return self.tools.dispatch(
            state,
            node=node,
            tool=tool,
            arguments=arguments,
            snapshots=snapshots,
            context=context,
        )

    # -- 预算与轮次 -------------------------------------------------------

    def initialize_budget(self, state: dict[str, Any]) -> dict[str, Any]:
        """补齐预算快照；已有计数不重置。"""
        return policy.initialize(state)

    def note_round(self, state: dict[str, Any]) -> dict[str, Any]:
        """记一次轮次，返回要并入节点增量的预算片段。"""
        return policy.count_round(state)

    def context_report(self, agent: str) -> dict[str, Any]:
        """取该 Agent 本次的上下文裁剪度量。

        度量按执行上下文隔离，并发 Run 不会互相串写；取不到返回空字典。
        """
        return context_report_of(agent)


_HARNESS: AgentRuntimeHarness | None = None


def get_harness() -> AgentRuntimeHarness:
    """取进程内的 Harness 实例，首次调用时创建。"""
    global _HARNESS
    if _HARNESS is None:
        _HARNESS = AgentRuntimeHarness()
    return _HARNESS


__all__ = [
    "AgentRuntimeHarness",
    "CURRENT_STATE",
    "StateNotBound",
    "ToolOutcome",
    "bind_state",
    "current_state",
    "get_harness",
]
