"""Python / SQL / Plot 的统一派发入口。

三种工具的实际执行都收敛到这里，节点只需要给出"要跑什么"。派发顺序是：

    预算与策略检查 → 授权资源解析 → 实际派发 → 错误分类 → 时间与次数结算 → 追踪

预算检查与计数由 `utils.middleware.governed_tool_call` 完成，这里不重复计一次：
两处同时计数会让工具额度按一半的速度消耗。本模块负责的是它上面一层——
把参数、授权资源、错误分类与节点语义接起来。

授权不是靠校验模型给的参数，而是靠 `resolve_tables(state)`：它只解析本次
Run 快照里的表，因此模型即使编造表名也拿不到额外数据。表名参数只能缩小范围。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.inference import resolve_tables
from tools.tool_registry import invoke_tool
from utils.agent_loop import BUDGET_ERROR_KIND, BudgetExceeded
from utils.middleware import governed_tool_call
from utils.run_logger import log_event


@dataclass(frozen=True)
class ToolOutcome:
    """一次工具派发的结果。

    `execution` 是 `invoke_tool` 归一化后的字典（success / result / error /
    error_kind），`budget_updates` 是必须由调用方写回 state 的预算增量。
    """

    execution: dict[str, Any]
    budget_updates: dict[str, Any]


class ToolDispatcher:
    """把一次工具调用交给治理层执行。"""

    def dispatch(
        self,
        state: dict[str, Any],
        *,
        node: str,
        tool: str,
        arguments: dict[str, Any],
        snapshots: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> ToolOutcome:
        """执行一次工具调用。

        预算耗尽时记一条 `budget_exceeded` 并**向上抛出**：失败该长成什么样
        （任务级失败还是图表级失败、要不要进修复循环）是节点的语义，
        不在派发层决定。
        """
        try:
            with governed_tool_call(state, node=node, tool=tool) as gate:
                execution = invoke_tool(
                    tool,
                    resolve_tables(state),
                    arguments,
                    snapshots=(
                        snapshots if snapshots is not None
                        else state.get("table_snapshot") or {}
                    ),
                )
                gate["error_kind"] = execution.get("error_kind")
            # 时间结算发生在 governed_tool_call 的 finally 里，
            # 因此必须在 with 退出之后再读，块内取到的是结算前的值。
            budget_updates = gate.get("budget_updates") or {}
        except BudgetExceeded as exceeded:
            log_event(
                run_id=state.get("run_id") or "unknown",
                node=node,
                event="budget_exceeded",
                status="failed",
                details={
                    "scope": exceeded.scope,
                    "used": exceeded.used,
                    "limit": exceeded.limit,
                    "tool": tool,
                    "error_kind": BUDGET_ERROR_KIND,
                    **(context or {}),
                },
            )
            raise

        return ToolOutcome(
            execution=execution,
            budget_updates=budget_updates,
        )


__all__ = ["ToolDispatcher", "ToolOutcome"]
