"""模型请求预算的判别性验证。

覆盖四种情形，全部走生产路径：

    run_agent_call → utils.llm_retry 的 before_request / on_request
                   → harness.llm_gateway 安装的钩子
                   → agent_loop.check_llm_budget / consume_llm_call

- 一次逻辑调用内 2 次请求：不提前耗尽、用量等于真实请求数；
- 一次逻辑调用内 3 次请求：同上；
- 模型持续返回坏结构：第 3 次请求必须在发出前被拦下；
- 从非零用量恢复：预算不重置、剩余额度正确；
- 两个 Run 并发：请求计数互不串扰。

计数口径：`usage.llm_calls` 计**真实上游请求**（含解析失败后的重发），每次
请求只累加一次，判定与计数读的是同一个数。
"""

from __future__ import annotations

import contextvars
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

import harness
from core.runtime import RunRuntime, use_runtime
from utils import agent_loop as al
from utils.agent_loop import BUDGET_FIELD, BudgetExceeded, initialize
from utils.idempotency import run_agent_call
from utils.llm_retry import invoke_json_with_retry


class CountingChain:
    """前 `bad` 次返回坏结构，之后返回合法结构。

    `barrier` 用于并发用例：第 `rendezvous_at` 次请求前在此等待，
    让两个 Run 的请求确定性地重叠。
    """

    def __init__(
        self,
        bad: int = 0,
        *,
        always_bad: bool = False,
        barrier: threading.Barrier | None = None,
        rendezvous_at: int = 1,
    ) -> None:
        self.bad = bad
        self.always_bad = always_bad
        self.barrier = barrier
        self.rendezvous_at = rendezvous_at
        self.requests: list[int] = []

    def invoke(self, _inputs: Any) -> Any:
        self.requests.append(len(self.requests) + 1)
        if self.barrier is not None and len(self.requests) == self.rendezvous_at:
            self.barrier.wait(timeout=10)
        if self.always_bad or len(self.requests) <= self.bad:
            return []  # 不是 dict → 解析失败
        return {"ok": True}


def validate(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("ok") is not True:
        raise ValueError("missing ok")
    return payload


def budget_state(cap: int, *, used: int = 0, run_id: str = "llm-budget") -> dict[str, Any]:
    state: dict[str, Any] = {
        "run_id": run_id,
        "completed_operations": {},
        "approvals": [],
        "pending_llm_calls": [],
        "llm_retry_granted": None,
    }
    state.update(initialize(state))
    state[BUDGET_FIELD]["limits"]["max_llm_calls"] = cap
    state[BUDGET_FIELD]["usage"]["llm_calls"] = used
    return state


def governed_call(state: dict[str, Any], chain: Any, *, attempt: int = 1):
    """走一次完整的生产逻辑调用（含预算闸门与解析重试）。"""
    import nodes

    def call() -> Any:
        return invoke_json_with_retry(
            chain,
            {},
            validate=validate,
            max_attempts=3,
            before_request=al.before_model_request,
            on_request=al.after_model_request,
        )

    return run_agent_call(
        state,
        node="executor",
        subject=f"task_1#r{attempt}",
        call=call,
        attempt=attempt,
        before_request=harness.get_harness().llm.budget_guard,
    )


@pytest.fixture
def node_state(monkeypatch: pytest.MonkeyPatch):
    """让 `harness.get_harness().llm.budget_guard` 读到本用例的 state。

    生产里这份 state 由 `nodes._with_state` 放进 ContextVar；这里替换同一个
    ContextVar，从而在不启动图的前提下驱动真实钩子。
    """
    import nodes

    var: contextvars.ContextVar = contextvars.ContextVar("test_current_state")
    monkeypatch.setattr(harness.runtime, "CURRENT_STATE", var)
    return var


@pytest.fixture(autouse=True)
def isolated_ledger(tmp_path: Path):
    """账本与产物写到 tmp_path，不碰仓库的 outputs/。"""
    runtime = RunRuntime(
        run_id="llm-budget",
        model_factory=None,  # type: ignore[arg-type]
        resolver=None,  # type: ignore[arg-type]
        output_root=tmp_path / "outputs",
    )
    with use_runtime(runtime):
        yield


class TestTwoRequestsInOneLogicalCall:
    def test_second_request_is_allowed_and_counted_once(self, node_state) -> None:
        state = budget_state(cap=2)
        token = node_state.set(state)
        try:
            chain = CountingChain(bad=1)
            payload, _ops, _extra = governed_call(state, chain)
        finally:
            node_state.reset(token)

        assert payload == {"ok": True}
        assert len(chain.requests) == 2, f"实际请求数：{chain.requests}"
        assert al.usage_of(state)["llm_calls"] == 2, "用量与真实请求数不一致"


class TestThreeRequestsInOneLogicalCall:
    def test_third_request_is_allowed_and_counted_once(self, node_state) -> None:
        state = budget_state(cap=3)
        token = node_state.set(state)
        try:
            chain = CountingChain(bad=2)
            payload, _ops, _extra = governed_call(state, chain)
        finally:
            node_state.reset(token)

        assert payload == {"ok": True}
        assert len(chain.requests) == 3, f"实际请求数：{chain.requests}"
        assert al.usage_of(state)["llm_calls"] == 3


class TestBudgetStopsTheRetryLoop:
    def test_third_request_is_blocked_before_it_is_sent(self, node_state) -> None:
        """上限 2、模型一直返回坏结构：只允许发出 2 次请求。"""
        state = budget_state(cap=2)
        token = node_state.set(state)
        try:
            chain = CountingChain(always_bad=True)
            with pytest.raises(BudgetExceeded) as excinfo:
                governed_call(state, chain)
        finally:
            node_state.reset(token)

        assert excinfo.value.scope == "llm_calls"
        assert len(chain.requests) == 2, (
            f"上限 2 却发出了 {len(chain.requests)} 次请求"
        )
        assert al.usage_of(state)["llm_calls"] == 2


class TestResumeKeepsTheBudget:
    def test_existing_usage_is_not_reset_and_remaining_is_exact(
        self, node_state
    ) -> None:
        """从已消耗 2 次的状态恢复：只剩 1 个额度。"""
        state = budget_state(cap=3, used=2)
        token = node_state.set(state)
        try:
            chain = CountingChain()
            payload, _ops, _extra = governed_call(state, chain)
            assert payload == {"ok": True}
            assert len(chain.requests) == 1
            assert al.usage_of(state)["llm_calls"] == 3

            # 额度已尽：下一次逻辑调用不得再发请求
            chain2 = CountingChain()
            with pytest.raises(BudgetExceeded):
                governed_call(state, chain2, attempt=2)
            assert chain2.requests == [], "额度已尽却仍发出了请求"
        finally:
            node_state.reset(token)

        assert state[BUDGET_FIELD]["limits"]["max_llm_calls"] == 3, "上限被改动"

    def test_reinitialise_does_not_reset_usage(self) -> None:
        state = budget_state(cap=3, used=2)
        for _ in range(3):
            state.update(initialize(state))
        assert al.usage_of(state)["llm_calls"] == 2
        assert state[BUDGET_FIELD]["limits"]["max_llm_calls"] == 3


class TestConcurrentRunsKeepSeparateCounts:
    def test_two_threads_do_not_share_request_counts(self) -> None:
        """两个 Run 各发 2 次请求（含一次重发），计数互不影响。"""
        import nodes

        barrier = threading.Barrier(2)
        results: dict[str, int] = {}
        errors: list[BaseException] = []

        def worker(name: str) -> None:
            state = budget_state(cap=5, run_id=f"llm-budget-{name}")
            chain = CountingChain(bad=1, barrier=barrier)

            def call() -> Any:
                return invoke_json_with_retry(
                    chain,
                    {},
                    validate=validate,
                    max_attempts=3,
                    before_request=al.before_model_request,
                    on_request=al.after_model_request,
                )

            token = harness.runtime.CURRENT_STATE.set(state)
            try:
                run_agent_call(
                    state,
                    node="executor",
                    subject="task_1#r1",
                    call=call,
                    attempt=1,
                    before_request=harness.get_harness().llm.budget_guard,
                )
                results[name] = al.usage_of(state)["llm_calls"]
            except BaseException as error:  # noqa: BLE001
                errors.append(error)
                try:
                    barrier.abort()
                except Exception:  # noqa: BLE001
                    pass
            finally:
                harness.runtime.CURRENT_STATE.reset(token)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(worker, name) for name in ("a", "b")]
            for future in futures:
                future.result(timeout=15)

        assert not errors, errors
        assert results == {"a": 2, "b": 2}, (
            f"并发 Run 的请求计数串了：{results}"
        )
