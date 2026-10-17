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
  预算耗尽时抛 `BudgetExceeded`，由节点转成正常的失败记录。

**模型调用不在这里。** 它必须与幂等登记、账本的 `started`/`completed`、
以及"未确认调用需人工裁决"的拦截协同，那些逻辑集中在
`utils/idempotency.run_agent_call`；预算闸门以 `before_request` 回调的形式
接进那条路径（见 `harness.llm_gateway` 与
`utils/agent_loop.before_model_request`）。

早期这里还有 `governed_llm_call()`、`record_round()` 与 `merge_budget()`，
都没有生产调用方：前两个与真正生效的路径重复（轮次计数走
`harness.policy.count_round`），第三个只有测试在用。
两套并行实现会让人误以为它们参与了治理，已删除。
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator

from sandbox.protocol import classify_exception
from utils.agent_loop import (
    BUDGET_ERROR_KIND,
    BUDGET_FIELD,
    BudgetExceeded,
    check,
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
    调用体自己抛异常时，异常照常向上传播，但这次调用会按错误分类记一条
    `failed`，不会被当成 `ok`；计时与时钟结算仍在 finally 里完成。

    调用结束时会把执行时钟**结算**到当前时刻（`settle_active_time`）：
    这样"距上次操作"的那段尾巴被一次算清，之后停下来等待人工审批的时间
    不会继续被计入执行预算。返回值在 `budget_updates` 里，调用方必须写回
    state，否则时钟推进不会落盘——结算发生在 `finally`，因此要在 `with`
    **退出之后**再取，块内取到的是结算前的值。

    `finally` 里同时就地写入传入的 `state`。工具调用一旦开始就已经消耗了
    额度，调用方若在写回之前抛异常（例如 `resolve_tables` 失败），只有就地
    写入才能让这次消耗留在 state 上，不会因为一次崩溃白拿一次额度。这与
    模型请求计数钩子（`harness.llm_gateway`）的做法一致。
    """
    check(state)

    counter = consume_tool_call(state)
    started = time.perf_counter()
    results: dict[str, Any] = {"budget_updates": counter, "duration_ms": None}

    try:
        yield results
    except Exception as error:
        # 调用体自己抛异常时（`resolve_tables` 失败、工具层没归一化的错误），
        # 这次调用依然消耗了额度、也依然花了时间，但不能记成 ok——否则日志
        # 与真实结果相反。分类后继续向上抛，调用方按原语义处理。
        results["error_kind"] = classify_exception(error)
        results["error_type"] = type(error).__name__
        results["error"] = str(error)[:300]
        raise
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
        state[BUDGET_FIELD] = merged[BUDGET_FIELD]
        _log(
            {**state, **merged},
            node=node,
            event="tool_call",
            status=str(results.get("error_kind") or "ok"),
            details={
                "tool": tool,
                "error_kind": results.get("error_kind"),
                "error_type": results.get("error_type"),
                "error": results.get("error"),
            },
            duration_ms=duration_ms,
        )


__all__ = [
    "BUDGET_ERROR_KIND",
    "BudgetExceeded",
    "governed_tool_call",
]
