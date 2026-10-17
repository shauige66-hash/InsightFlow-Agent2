"""工具调用的额度消耗必须留在 state 上。

一次工具调用只要进入执行段就已经消耗了额度（`consume_tool_call`），因此这次
消耗不能只存在于返回给调用方的局部增量里。调用方在写回之前抛异常时（例如
`resolve_tables` 失败、节点被取消），只有直接写进 state 的计数才会留下，
否则一次崩溃就能白拿一次工具额度，重复崩溃可以无限启动容器。

这与模型请求计数钩子 `harness.llm_gateway.LLMGateway.install_request_hooks` 的做法一致：计数先落到
当前 state 上，正式的 Checkpoint 提交仍由节点返回的增量完成。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from utils.agent_loop import BUDGET_FIELD, BudgetExceeded, initialize, usage_of
from utils.middleware import governed_tool_call


def fresh_state(run_id: str = "tool-delta", *, max_tool_calls: int = 10) -> dict:
    state: dict[str, Any] = {"run_id": run_id}
    state.update(initialize(state))
    state[BUDGET_FIELD]["limits"]["max_tool_calls"] = max_tool_calls
    return state


class TestTheCounterLandsOnState:
    def test_a_normal_call_writes_the_counter_to_state(self) -> None:
        state = fresh_state()

        with governed_tool_call(state, node="tool_execution", tool="python") as gate:
            gate["error_kind"] = None

        assert usage_of(state)["tool_calls"] == 1, "计数没有写回 state"
        # 调用方仍然拿得到同一份增量
        updates = gate.get("budget_updates") or {}
        assert (updates.get(BUDGET_FIELD) or {}).get("usage", {})["tool_calls"] == 1

    def test_the_clock_is_settled_on_state_after_the_block(self) -> None:
        state = fresh_state()

        with governed_tool_call(state, node="tool_execution", tool="sql") as gate:
            pass

        usage = usage_of(state)
        assert usage["clock_open"] is False, "时钟没有结算到 state 上"
        assert usage["tool_calls"] == 1
        updates = (gate.get("budget_updates") or {}).get(BUDGET_FIELD, {}).get("usage", {})
        assert updates["clock_open"] is False

    def test_an_exception_from_the_body_still_leaves_the_counter(self) -> None:
        """调用体内抛异常（调用方没机会写回）时计数仍然留下。"""
        state = fresh_state()

        with pytest.raises(RuntimeError, match="resolve 失败"):
            with governed_tool_call(state, node="tool_execution", tool="python"):
                raise RuntimeError("resolve 失败")

        usage = usage_of(state)
        assert usage["tool_calls"] == 1, "崩溃路径把这次工具额度漏掉了"
        assert usage["clock_open"] is False, "崩溃路径没有结算时钟"

    def test_repeated_crashes_keep_consuming_the_budget(self) -> None:
        """反复崩溃不能被当成"没用过额度"。"""
        state = fresh_state(max_tool_calls=3)

        for _ in range(3):
            with pytest.raises(RuntimeError):
                with governed_tool_call(state, node="tool_execution", tool="python"):
                    raise RuntimeError("boom")

        assert usage_of(state)["tool_calls"] == 3
        with pytest.raises(BudgetExceeded) as excinfo:
            with governed_tool_call(state, node="tool_execution", tool="python"):
                pass
        assert excinfo.value.scope == "tool_calls"

    def test_the_budget_check_still_refuses_before_any_counter(self) -> None:
        """预算已尽时 `check` 在计数之前拒绝，不产生虚假消耗。"""
        state = fresh_state(max_tool_calls=0)

        with pytest.raises(BudgetExceeded):
            with governed_tool_call(state, node="tool_execution", tool="python"):
                pytest.fail("预算为 0 时不应进入调用体")

        assert usage_of(state)["tool_calls"] == 0, "被拒绝的调用却记了消耗"

    def test_state_and_returned_delta_agree(self) -> None:
        """就地写入的值与调用方写回的值必须一致，否则预算会双算或漏算。"""
        state = fresh_state()

        with governed_tool_call(state, node="tool_execution", tool="python") as gate:
            pass

        returned = ((gate.get("budget_updates") or {}).get(BUDGET_FIELD) or {}).get(
            "usage"
        ) or {}
        assert returned["tool_calls"] == usage_of(state)["tool_calls"]
        assert returned["active_seconds"] == usage_of(state)["active_seconds"]
        assert returned["clock_open"] == usage_of(state)["clock_open"]
