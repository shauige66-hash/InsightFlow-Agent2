"""Harness 三个模块各自的契约。

这里测的是分层本身的语义，不是底层功能：`utils.agent_loop` 的预算计算、
`utils.idempotency` 的幂等、`utils.middleware` 的工具治理各自已有测试。
这些用例回答的是"Harness 这一层决定了什么"。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import harness
from core.runtime import RunRuntime, use_runtime
from harness import policy
from harness import tool_dispatcher
from harness.tool_dispatcher import ToolDispatcher
from utils.agent_loop import (
    BUDGET_FIELD,
    BudgetExceeded,
    after_model_request,
    before_model_request,
    elapsed_seconds,
    initialize,
    usage_of,
)


def make_state(run_id: str = "harness-unit", **limits: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "run_id": run_id,
        "completed_operations": {},
        "approvals": [],
        "pending_llm_calls": [],
        "llm_retry_granted": None,
    }
    state.update(initialize(state))
    state[BUDGET_FIELD]["limits"].update(limits)
    return state


@pytest.fixture(autouse=True)
def isolated_trace(tmp_path: Path):
    runtime = RunRuntime(
        run_id="harness-unit",
        model_factory=None,  # type: ignore[arg-type]
        resolver=None,  # type: ignore[arg-type]
        output_root=tmp_path / "outputs",
    )
    with use_runtime(runtime):
        yield


class TestPolicyScopes:
    def test_model_request_is_judged_on_llm_calls_only(self) -> None:
        """工具额度用尽不该拒绝一次合法的模型请求。"""
        state = make_state(max_tool_calls=1)
        state[BUDGET_FIELD]["usage"]["tool_calls"] = 1

        policy.ensure_model_request_allowed(state)  # 不抛

        with pytest.raises(BudgetExceeded):
            policy.ensure_tool_allowed(state)

    def test_tool_call_is_judged_on_every_scope(self) -> None:
        for scope, limit_key in (
            ("tool_calls", "max_tool_calls"),
            ("rounds", "max_rounds"),
            ("llm_calls", "max_llm_calls"),
        ):
            state = make_state()
            limit = state[BUDGET_FIELD]["limits"][limit_key]
            state[BUDGET_FIELD]["usage"][scope] = limit
            with pytest.raises(BudgetExceeded) as excinfo:
                policy.ensure_tool_allowed(state)
            assert excinfo.value.scope == scope

    def test_counting_a_request_writes_it_into_the_state(self) -> None:
        """同一次逻辑调用内部的重试要能看到已经发过几次。"""
        state = make_state()

        policy.count_model_request(state)
        policy.count_model_request(state)

        assert usage_of(state)["llm_calls"] == 2

    def test_settling_closes_the_clock(self) -> None:
        state = make_state()
        policy.count_model_request(state)  # 打开时钟
        assert usage_of(state)["clock_open"] is True

        settled = policy.settle(state)
        assert settled[BUDGET_FIELD]["usage"]["clock_open"] is False

        # 时钟停住后，再等多久都不增加执行时间
        before = elapsed_seconds({**state, **settled})
        assert elapsed_seconds({**state, **settled}) == pytest.approx(before)


class TestGatewayIsTheOnlyModelPath:
    def test_a_completed_logical_call_is_reused_without_new_requests(self) -> None:
        from utils.llm_retry import invoke_json_with_retry

        state = make_state()
        runtime = harness.get_harness()
        calls = {"n": 0}

        class Chain:
            def invoke(self, _inputs: Any) -> Any:
                calls["n"] += 1
                return {"answer": 1}

        def invoke() -> dict:
            # 与九个 Agent 一致：经 llm_retry 驱动 Harness 安装的请求钩子
            return invoke_json_with_retry(
                Chain(),
                {},
                before_request=before_model_request,
                on_request=after_model_request,
            )

        with harness.bind_state(state):
            payload, operations, extra = runtime.model_call("planner", "plan#0", invoke)
            # 真正发过请求：用量增加，并带回结算后的预算增量
            assert usage_of(state)["llm_calls"] == 1
            assert BUDGET_FIELD in extra
            state["completed_operations"] = operations

            reused, _ops2, extra2 = runtime.model_call("planner", "plan#0", invoke)

        assert payload == reused == {"answer": 1}
        assert calls["n"] == 1, "幂等复用时又发了请求"
        assert usage_of(state)["llm_calls"] == 1, "复用不该增加请求数"
        assert extra2 == {}, "复用路径不应报告发生了上游请求"

    def test_the_guard_refuses_before_the_request(self) -> None:
        state = make_state(max_llm_calls=0)
        invoked: list[int] = []

        with harness.bind_state(state):
            with pytest.raises(BudgetExceeded):
                harness.get_harness().model_call(
                    "planner", "plan#0", lambda: invoked.append(1)
                )

        assert invoked == [], "被拒绝的调用仍执行了请求体"


class TestDispatcherContract:
    def _patch_tool(self, monkeypatch: pytest.MonkeyPatch, execution: dict) -> list:
        invoked: list[tuple] = []

        def fake(tool_name, tables, arguments, *, snapshots=None):
            invoked.append((tool_name, arguments))
            return execution

        monkeypatch.setattr(tool_dispatcher, "invoke_tool", fake)
        monkeypatch.setattr(tool_dispatcher, "resolve_tables", lambda state: {})
        return invoked

    def test_a_dispatch_returns_the_execution_and_the_budget_delta(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state = make_state()
        self._patch_tool(
            monkeypatch,
            {"success": True, "result": [1], "error": None, "error_kind": None},
        )

        outcome = harness.get_harness().run_tool(
            state,
            node="tool_execution",
            tool="sql",
            arguments={"query": "select 1"},
        )

        assert outcome.execution["success"] is True
        # 增量里必须带结算后的时钟，否则调用方写回的是结算前的值
        usage = (outcome.budget_updates.get(BUDGET_FIELD) or {}).get("usage") or {}
        assert usage["tool_calls"] == 1
        assert usage["clock_open"] is False

    def test_an_exhausted_budget_never_reaches_the_tool(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state = make_state(max_tool_calls=0)
        invoked = self._patch_tool(
            monkeypatch,
            {"success": True, "result": None, "error": None, "error_kind": None},
        )

        with pytest.raises(BudgetExceeded):
            harness.get_harness().run_tool(
                state,
                node="tool_execution",
                tool="python",
                arguments={"code": "pass"},
            )

        assert invoked == [], "预算为 0 时仍调用了工具"

    def test_the_dispatcher_logs_the_refusal_and_reraises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import json

        state = make_state(run_id="dispatch-refuse", max_tool_calls=0)
        self._patch_tool(
            monkeypatch,
            {"success": True, "result": None, "error": None, "error_kind": None},
        )

        with pytest.raises(BudgetExceeded):
            ToolDispatcher().dispatch(
                state,
                node="plot_execution",
                tool="plot",
                arguments={"chart_id": "chart_1"},
                context={"chart_id": "chart_1"},
            )

        log_path = tmp_path / "outputs" / "logs" / "dispatch-refuse.jsonl"
        assert log_path.is_file(), "拒绝没有留下追踪记录"
        events = [
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        budget_events = [item for item in events if item["event"] == "budget_exceeded"]
        assert budget_events, events
        details = budget_events[-1]["details"]
        assert details["scope"] == "tool_calls"
        assert details["tool"] == "plot"
        assert details["chart_id"] == "chart_1"
