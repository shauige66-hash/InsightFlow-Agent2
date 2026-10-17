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
from harness.context_sources import (
    inject_context_sources,
    knowledge_chunk_count,
)
from harness.context_manager import (
    TableSelection,
    build_context as build_agent_context,
    ensure_known_resources,
    select_history,
    select_tables,
)
from harness.llm_gateway import LLMGateway, StateNotBound
from harness.tool_dispatcher import ToolDispatcher, ToolOutcome
from utils.context_budget import context_report_of, publish_context_report

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

    # -- 上下文 -----------------------------------------------------------

    def task_context(
        self,
        state: dict[str, Any],
        target_tables: list[str] | tuple[str, ...],
    ) -> TableSelection:
        """按任务选出相关表的上下文，不存在的资源会被拒绝。"""
        return select_tables(state, target_tables)

    def ensure_resources(
        self,
        state: dict[str, Any],
        target_tables: list[str] | tuple[str, ...],
    ) -> tuple[str, ...]:
        """只校验资源名，不做选择。给路由这类"还没构造上下文"的步骤用。"""
        return ensure_known_resources(state, target_tables)

    def history_for_task(
        self,
        state: dict[str, Any],
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """当前任务相关的历史执行摘要（有界、确定性）。"""
        if limit is None:
            return select_history(state)
        return select_history(state, limit=limit)

    def build_context(
        self,
        agent: str,
        fields: dict[str, Any],
        *,
        data_context: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
        task_id: str | None = None,
        budget: int | None = None,
    ) -> tuple[dict[str, Any], Any]:
        """压缩一次 Agent 的输入，并发布这次构造的度量。

        `data_context` 是节点按任务选好的表上下文；这里用它推出"保留/排除"
        写进报告。排除集合来自当前 Run 的授权上下文，因此报告能说明
        "这次只看了哪几张表"，而不是只有总字符数。
        """
        selection = self._selection_for(data_context)

        # 记忆与多轮历史在这里合并进输入字段：Agent 只看到已格式化的文本，
        # 不接触存储，也不知道它们来自哪张表。
        fields = dict(fields)
        state = current_state() or {}
        injected = inject_context_sources(agent, fields, state)

        bounded, report = build_agent_context(
            agent,
            fields,
            selection=selection,
            history=history,
            task_id=task_id,
            budget=budget,
            knowledge_chunks=knowledge_chunk_count(fields),
            memory_entries=injected["memory_entries"],
            knowledge_cache_hit=bool(state.get("knowledge_cache_hit")),
        )
        publish_context_report(agent, report.to_dict())
        return bounded, report

    @staticmethod
    def _selection_for(data_context: dict[str, Any] | None) -> TableSelection | None:
        if data_context is None:
            return None
        kept = tuple(data_context.keys())
        authorized = (current_state() or {}).get("data_context") or {}
        return TableSelection(
            tables=dict(data_context),
            kept=kept,
            excluded=tuple(sorted(set(authorized) - set(kept))),
        )


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
