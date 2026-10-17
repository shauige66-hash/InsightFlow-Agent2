"""模型请求预算在并发 Run 之间的隔离。

`RunExecutor` 用 `ThreadPoolExecutor(max_workers=2)` 并发执行 Run，因此
"预算记在哪个 state 上"必须按执行上下文区分。这里用 `threading.Barrier`
让两个 Run 的请求确定性地重叠，验证：

- 各自的 `llm_calls` 只累加自己的请求；
- 一个 Run 的额度耗尽不影响另一个；
- 预算闸门读取的是本 Run 的 state，不会串到别人。

历史背景：这里曾经用模块级全局保存请求计数作用域的 token，两个 Run 区间
重叠时会互相覆盖，先进入者结算时拿到别人上下文的 token，`ContextVar.reset`
直接抛 `ValueError`。现在请求计数只存在于 state 上，没有跨上下文的 token。
"""

from __future__ import annotations

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
    state[BUDGET_FIELD]["limits"]["max_llm_calls"] = max_llm_calls
    return state


@pytest.fixture(autouse=True)
def isolated_ledger(tmp_path: Path):
    """账本写到 tmp_path，避免测试写入仓库的 outputs/。"""
    runtime = RunRuntime(
        run_id="scope-isolation",
        model_factory=None,  # type: ignore[arg-type]
        resolver=None,  # type: ignore[arg-type]
        output_root=tmp_path / "outputs",
    )
    with use_runtime(runtime):
        yield


class TestConcurrentRunsDoNotShareBudgets:
    def test_each_run_accumulates_only_its_own_requests(self) -> None:
        import nodes

        barrier = threading.Barrier(2)
        counted: dict[str, int] = {}
        errors: list[BaseException] = []

        def worker(run_id: str, requests: int) -> None:
            state = fresh_state(run_id, max_llm_calls=10)

            def call() -> dict:
                barrier.wait(timeout=BARRIER_TIMEOUT)
                for _ in range(requests):
                    updates = al.consume_llm_call(state)
                    state[BUDGET_FIELD] = updates[BUDGET_FIELD]
                return {"count": al.usage_of(state)["llm_calls"]}

            token = harness.runtime.CURRENT_STATE.set(state)
            try:
                payload, _ops, _extra = run_agent_call(
                    state,
                    node="planner",
                    subject=f"{run_id}#0",
                    call=call,
                    before_request=harness.get_harness().llm.budget_guard,
                )
                counted[run_id] = int(payload["count"])
            except BaseException as error:  # noqa: BLE001
                errors.append(error)
                try:
                    barrier.abort()
                except Exception:  # noqa: BLE001
                    pass
            finally:
                harness.runtime.CURRENT_STATE.reset(token)

        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(worker, "scope-a", 1)
            second = pool.submit(worker, "scope-b", 3)
            for future in (first, second):
                future.result(timeout=15)

        assert not errors, errors
        assert counted == {"scope-a": 1, "scope-b": 3}, (
            f"两个 Run 的请求计数串了：{counted}"
        )

    def test_an_exhausted_run_does_not_block_another(self) -> None:
        """一个 Run 被预算拒绝，不影响另一个 Run 正常执行。"""
        import nodes

        barrier = threading.Barrier(2)
        outcomes: dict[str, str] = {}
        errors: list[BaseException] = []

        def worker(run_id: str, cap: int) -> None:
            state = fresh_state(run_id, max_llm_calls=cap)
            # 两个 Run 在进入闸门之前对齐，保证判定是并发发生的
            barrier.wait(timeout=BARRIER_TIMEOUT)

            def call() -> dict:
                updates = al.consume_llm_call(state)
                state[BUDGET_FIELD] = updates[BUDGET_FIELD]
                return {"ok": True}

            token = harness.runtime.CURRENT_STATE.set(state)
            try:
                run_agent_call(
                    state,
                    node="planner",
                    subject=f"{run_id}#0",
                    call=call,
                    before_request=harness.get_harness().llm.budget_guard,
                )
                outcomes[run_id] = "ran"
            except BudgetExceeded:
                outcomes[run_id] = "refused"
            except BaseException as error:  # noqa: BLE001
                errors.append(error)
            finally:
                harness.runtime.CURRENT_STATE.reset(token)

        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(worker, "scope-zero", 0)
            second = pool.submit(worker, "scope-open", 3)
            for future in (first, second):
                future.result(timeout=15)

        assert not errors, errors
        assert outcomes == {"scope-zero": "refused", "scope-open": "ran"}, outcomes


class TestBudgetBelongsToItsOwnState:
    def test_counts_do_not_leak_across_contexts(self) -> None:
        """一个执行上下文里的累计不会出现在另一个上下文。"""
        state = fresh_state("scope-local")
        updates = al.consume_llm_call(state)
        state[BUDGET_FIELD] = updates[BUDGET_FIELD]
        assert al.usage_of(state)["llm_calls"] == 1

        seen: dict[str, int] = {}

        def worker() -> None:
            other = fresh_state("scope-other")
            seen["own"] = al.usage_of(other)["llm_calls"]

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=BARRIER_TIMEOUT)

        assert seen["own"] == 0, "另一个 Run 看到了别人的请求计数"

    def test_guard_refuses_without_touching_the_ledger(self, tmp_path: Path) -> None:
        """额度为 0 时闸门直接拒绝，且不留下 started 记录。"""
        state = fresh_state("scope-refused", max_llm_calls=0)

        def guard() -> None:
            al.check_llm_budget(state)

        with pytest.raises(BudgetExceeded):
            run_agent_call(
                state,
                node="planner",
                subject="plan#0",
                call=lambda: {"value": 1},
                before_request=guard,
            )

        ledger_dir = tmp_path / "outputs" / "llm_ledger"
        lines: list[str] = []
        if ledger_dir.exists():
            for path in ledger_dir.glob("*.jsonl"):
                lines.extend(path.read_text(encoding="utf-8").splitlines())
        assert lines == [], f"被拒绝的调用留下了账本记录：{lines}"

    def test_scope_is_balanced_after_a_normal_call(self) -> None:
        state = fresh_state("scope-balanced")

        payload, _ops, extra = run_agent_call(
            state, node="planner", subject="plan#0", call=lambda: {"value": 1}
        )

        assert payload == {"value": 1}
        assert extra.get("__upstream_invoked__") is True
