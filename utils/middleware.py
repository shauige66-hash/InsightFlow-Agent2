"""P3-B：统一治理层（Middleware）。

## 为什么需要它，以及为什么它必须很薄

现有代码里，同一类治理逻辑散落在多个节点：

- 「记一次结构化日志」在每个节点里各写一遍 `log_event(...)`；
- 「工具失败转成结果字典」在 `invoke_tool` 里，但**没有**统一计时；
- 「模型调用预算」不存在，只有 `MAX_RETRIES` 这类局部上限；
- 「错误分类」在 `sandbox.protocol.classify_exception` 与
  `tool_registry._kind_of` 两处，语义正确但没有统一的观察点。

因此这里只做一件事：**把这些横切关注点收成少数几个函数**，
并复用既有的日志、账本、错误分类与 Tool Registry。
不新建幂等系统、不新建重试系统、不接管节点路由。

## 入口

- `governed_tool_call()`：包住一次工具调用，判预算 + 计时 + 计数 + 记录。
- `record_round()`：记一次 Reason→Act→Observe→Decide 轮次。

两者都在预算耗尽时抛 `BudgetExceeded`，由节点转成正常的失败记录。

**模型调用不在这里。** 它必须与幂等登记、账本的 `started`/`completed`、
以及"未确认调用需人工裁决"的拦截协同，那些逻辑集中在
`utils/idempotency.run_agent_call`；预算闸门以 `before_request` 回调的形式
接进那条路径（见 `nodes._governed_agent_call` 与
`utils/agent_loop.before_model_request`）。

早期这里还有一个 `governed_llm_call()`，但没有任何代码调用它，而真正生效的
是上面那条路径——两套并行实现会让人误以为它参与了预算执行，已删除。
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator

from utils.agent_loop import (
    BUDGET_ERROR_KIND,
    BUDGET_FIELD,
    BudgetExceeded,
    check,
    consume_round,
    consume_tool_call,
    settle_active_time,
    snapshot,
)
from utils.run_logger import log_event


def _log(
    state: dict[str, Any],
    *,
    node: str,
    event: str,
    status: str,
    details: dict[str, Any] | None = None,
    duration_ms: float | None = None,
) -> None:
    """统一的结构化日志出口。

    刻意直接复用 `utils/run_logger`：它已经写了 jsonl 并把 output_root
    接好了，再包一层只会制造第二套日志格式。
    """
    log_event(
        run_id=state.get("run_id") or "unknown",
        node=node,
        event=event,
        status=status,
        details={**(details or {}), "budget": snapshot(state)},
        duration_ms=duration_ms,
    )


def record_round(state: dict[str, Any], *, node: str) -> dict[str, Any]:
    """记一次循环轮次，并返回要写回 state 的增量。

    调用方必须把返回值合并进自己的 state 增量，否则计数不会落盘，
    恢复后预算就归零了。
    """
    updates = consume_round(state)
    _log(state, node=node, event="loop_round", status="ok")
    return updates


@contextmanager
def governed_tool_call(
    state: dict[str, Any],
    *,
    node: str,
    tool: str,
) -> Iterator[dict[str, Any]]:
    """包住一次工具调用：先判预算，再计时与记录。

    用法：

        with governed_tool_call(state, node="tool_execution", tool="sql") as gate:
            outcome = invoke_tool(...)
            gate["error_kind"] = outcome.get("error_kind")

    预算耗尽可能发生在进入之前（`check`）——那时会抛 `BudgetExceeded`。
    工具自身抛异常不受影响：异常照常向上传播，计时与日志在 finally 里完成。

    调用结束时会把执行时钟**结算**到当前时刻（`settle_active_time`）：
    这样"距上次操作"的那段尾巴被一次算清，之后停下来等待人工审批的时间
    不会继续被计入执行预算。返回值在 `budget_updates` 里，调用方必须写回
    state，否则时钟推进不会落盘——结算发生在 `finally`，因此要在 `with`
    **退出之后**再取，块内取到的是结算前的值。
    """
    check(state)

    counter = consume_tool_call(state)
    started = time.perf_counter()
    results: dict[str, Any] = {"budget_updates": counter, "duration_ms": None}

    try:
        yield results
    finally:
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        results["duration_ms"] = duration_ms
        # 结算执行时钟（与计数合并成同一份增量）
        merged = results.get("budget_updates") or {}
        settled = settle_active_time({**state, **merged})
        merged = {
            **merged,
            BUDGET_FIELD: settled[BUDGET_FIELD],
        }
        results["budget_updates"] = merged
        _log(
            {**state, **merged},
            node=node,
            event="tool_call",
            status=str(results.get("error_kind") or "ok"),
            details={"tool": tool, "error_kind": results.get("error_kind")},
            duration_ms=duration_ms,
        )


def merge_budget(
    *budget_updates: dict[str, Any] | None,
) -> dict[str, Any]:
    """把若干次预算增量合并成一份 state 增量。

    后写的覆盖先写的：同一字段在一轮里被多次递增时，最后一次的计数值
    才是最新的（`consume_*` 每次都基于传入 state 重算，不会丢计数）。
    """
    merged: dict[str, Any] = {}
    for item in budget_updates:
        if item:
            merged.update(item)
    return merged


__all__ = [
    "BUDGET_ERROR_KIND",
    "BudgetExceeded",
    "governed_tool_call",
    "merge_budget",
    "record_round",
]
