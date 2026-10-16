"""P3 定点修复：模型请求计数作用域的并发隔离。

`run_agent_call` 用 `_request_scope()` 划出"本次逻辑调用"的边界，`utils.agent_loop._requests_in_flight` 记录这个边界内
已经发出的真实模型请求数，模型预算的判定读它。

早期实现把 `begin_operation_requests()` 返回的 token 存进模块级全局
`_request_scope`。`RunExecutor` 默认用 `ThreadPoolExecutor(max_workers=2)`
并发执行 Run，两个 Run 同时进入这个区间时后进入者会覆盖前者的 token：

- 先进入者结算时拿到的是**别人上下文**里的 token，`ContextVar.reset` 直接抛
  `ValueError`（实测跨线程 reset 不可用）；
- 或者全局已被清空，先进入者的 token 永远不被 reset，`_requests_in_flight`
  泄漏到后续逻辑调用，让预算判定提前失败。

下面用 `threading.Barrier` 把两个线程的区间**确定性**地串在一起，
不依赖调度时序。
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from core.runtime import RunRuntime, use_runtime
from utils import idempotency
from utils.agent_loop import (
    BUDGET_FIELD,
    AgentBudget,
    begin_operation_requests,
    end_operation_requests,
    initialize,
    new_usage,
    requests_in_flight,
)

#: 制造区间交错时的等待上限。超时即视为死锁，用例直接失败而不是挂住。
BARRIER_TIMEOUT = 10


def fresh_state(run_id: str, *, max_llm_calls: int = 5) -> dict[str, Any]:
    state: dict[str, Any] = {
        "run_id": run_id,
        "completed_operations": {},
        "approvals": [],
        "pending_llm_calls": [],
        "llm_retry_granted": None,
    }
    state.update(initialize(state))
    limits = state[BUDGET_FIELD]["limits"]
    limits["max_llm_calls"] = max_llm_calls
    return state


@pytest.fixture(autouse=True)
def _isolated_ledger(tmp_path: Path):
    """把账本落到 tmp_path，避免测试写入仓库的 outputs/。"""
    runtime = RunRuntime(
        run_id="scope-isolation",
        model_factory=None,  # type: ignore[arg-type]
        resolver=None,  # type: ignore[arg-type]
        output_root=tmp_path / "outputs",
    )
    with use_runtime(runtime):
        yield


class TestRequestScopeTokenIsContextLocal:
    """回归：token 不能放在模块级全局里。"""

    def test_concurrent_scopes_do_not_reset_across_contexts(self) -> None:
        barrier = threading.Barrier(2)
        errors: list[BaseException] = []
        leaked: dict[str, int] = {}

        def worker(run_id: str) -> None:
            state = fresh_state(run_id)

            def call() -> dict:
                # 两个 Run 的区间在这里确定性地重叠
                barrier.wait(timeout=BARRIER_TIMEOUT)
                return {"value": run_id}

            try:
                idempotency.run_agent_call(
                    state, node="planner", subject=f"{run_id}#0", call=call
                )
                leaked[run_id] = requests_in_flight()
            except BaseException as error:  # noqa: BLE001
                errors.append(error)
                try:
                    barrier.abort()
                except Exception:  # noqa: BLE001
                    pass

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(worker, f"scope-{index}") for index in (1, 2)]
            for future in futures:
                future.result(timeout=BARRIER_TIMEOUT)

        assert not errors, (
            "并发 Run 的请求计数作用域互相踩踏："
            + "; ".join(f"{type(e).__name__}: {e}" for e in errors)
        )
        assert leaked == {"scope-1": 0, "scope-2": 0}, (
            f"作用域退出后仍有未结算请求：{leaked}"
        )

    def test_scope_is_balanced_after_a_normal_call(self) -> None:
        state = fresh_state("scope-balanced")

        payload, _ops, extra = idempotency.run_agent_call(
            state, node="planner", subject="plan#0", call=lambda: {"value": 1}
        )

        assert payload == {"value": 1}
        assert extra.get("__upstream_invoked__") is True
        assert requests_in_flight() == 0, "作用域退出后计数没有归零"

    def test_scope_is_cleaned_up_when_the_call_raises(self) -> None:
        """异常路径同样要结算作用域，否则计数会泄漏到下一次调用。"""

        def boom() -> dict:
            raise RuntimeError("注入的上游失败")

        state = fresh_state("scope-error")
        with pytest.raises(RuntimeError, match="注入的上游失败"):
            idempotency.run_agent_call(
                state, node="planner", subject="plan#0", call=boom
            )

        assert requests_in_flight() == 0, "异常路径没有清理请求作用域"

    def test_scope_is_cleaned_up_when_the_budget_refuses_the_call(self) -> None:
        """预算拦下请求时也要结算作用域。

        闸门在账本写入之前抛 `BudgetExceeded`，早期实现里它位于
        `try/finally` 之外，被拦下的调用会把 `_requests_in_flight` 留在原地。
        """
        from utils.agent_loop import BudgetExceeded

        state = fresh_state("scope-refused", max_llm_calls=0)

        def guard() -> None:
            raise BudgetExceeded("llm_calls", 0, 0)

        with pytest.raises(BudgetExceeded):
            idempotency.run_agent_call(
                state,
                node="planner",
                subject="plan#0",
                call=lambda: {"value": 1},
                before_request=guard,
            )

        assert requests_in_flight() == 0, "预算拒绝路径没有清理请求作用域"


class TestRequestScopeDoesNotLeakIntoOtherRuns:
    """两个 Run 的请求计数必须各算各的。"""

    def test_counts_stay_independent_across_threads(self) -> None:
        barrier = threading.Barrier(2)
        counted: dict[str, int] = {}
        errors: list[BaseException] = []

        def worker(run_id: str, requests: int) -> None:
            state = fresh_state(run_id, max_llm_calls=10)

            def call() -> dict:
                barrier.wait(timeout=BARRIER_TIMEOUT)
                assert requests_in_flight() == 0, "区间开始时计数应为 0"
                for _ in range(requests):
                    # 与 utils.llm_retry 的 on_request 钩子同一口径
                    from utils.agent_loop import note_request_in_flight

                    note_request_in_flight()
                return {"count": requests_in_flight()}

            try:
                payload, _ops, _extra = idempotency.run_agent_call(
                    state, node="planner", subject=f"{run_id}#0", call=call
                )
                counted[run_id] = int(payload["count"])
            except BaseException as error:  # noqa: BLE001
                errors.append(error)
                try:
                    barrier.abort()
                except Exception:  # noqa: BLE001
                    pass

        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(worker, "scope-a", 1)
            second = pool.submit(worker, "scope-b", 3)
            for future in (first, second):
                future.result(timeout=BARRIER_TIMEOUT)

        assert not errors, errors
        assert counted == {"scope-a": 1, "scope-b": 3}, (
            f"两个 Run 的请求计数串了：{counted}"
        )


class TestRequestScopeHelpersStayPaired:
    """直接对底层帮助函数做一次配对性检查，便于定位将来的回归。"""

    def test_begin_and_end_are_context_local(self) -> None:
        token = begin_operation_requests()
        from utils.agent_loop import note_request_in_flight

        note_request_in_flight()
        assert requests_in_flight() == 1

        end_operation_requests(token)
        assert requests_in_flight() == 0

    def test_other_thread_does_not_see_this_context(self) -> None:
        begin_operation_requests()
        from utils.agent_loop import note_request_in_flight

        note_request_in_flight()
        seen: dict[str, int] = {}

        def worker() -> None:
            seen["count"] = requests_in_flight()

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=BARRIER_TIMEOUT)
        assert seen["count"] == 0, "请求计数泄漏到了另一个执行上下文"
