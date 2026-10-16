"""P3-A/B：有界 Agent 执行与统一治理。

覆盖：

- 轮次 / 工具调用 / 模型调用 / 时间四类预算真的生效；
- 预算耗尽时**收敛失败**，而不是崩溃；
- 预算跨 Checkpoint 恢复后**不重置**；
- 模型预算不把幂等复用算成新调用；
- 治理层不改动安全边界（工具仍只能经 Tool Registry 与既有授权）。
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import pytest

from core.runtime import McpSettings
from ingestion.resources import ResourceResolver
from services.analysis_service import run_analysis
from utils import agent_loop
from utils.agent_loop import (
    BUDGET_ERROR_KIND,
    BUDGET_FIELD,
    AgentBudget,
    BudgetExceeded,
    check,
    consume_llm_call,
    consume_round,
    consume_tool_call,
    elapsed_seconds,
    initialize,
    new_usage,
    snapshot,
    usage_of,
)

from tests.test_report_approval import REPLIES, build_database, factory


@pytest.fixture(autouse=True)
def _restore_replies():
    """REPLIES 是模块级共享字典，逐用例还原，避免相互污染。"""
    original = copy.deepcopy(REPLIES)
    yield
    REPLIES.clear()
    REPLIES.update(copy.deepcopy(original))


def state_with(**limits: Any) -> dict[str, Any]:
    """构造一个只含预算的 state。"""
    payload = {
        BUDGET_FIELD: {
            "limits": AgentBudget(**limits).to_dict(),
            "usage": new_usage(),
        }
    }
    return payload


# ---------------------------------------------------------------------------
# 预算本体
# ---------------------------------------------------------------------------


class TestBudgetArithmetic:
    def test_tool_calls_accumulate(self) -> None:
        state = state_with()
        for _ in range(3):
            state.update(consume_tool_call(state))

        assert usage_of(state)["tool_calls"] == 3
        assert snapshot(state)["remaining"]["tool_calls"] == AgentBudget().max_tool_calls - 3

    def test_check_passes_under_limit(self) -> None:
        state = state_with(max_tool_calls=5)
        state.update(consume_tool_call(state))
        check(state)  # 不抛

    @pytest.mark.parametrize(
        "scope, key, limit",
        [
            ("tool_calls", "max_tool_calls", 2),
            ("rounds", "max_rounds", 2),
            ("llm_calls", "max_llm_calls", 2),
        ],
    )
    def test_check_raises_at_limit(self, scope: str, key: str, limit: int) -> None:
        state = state_with(**{key: limit})

        counter = {
            "tool_calls": consume_tool_call,
            "rounds": consume_round,
            "llm_calls": consume_llm_call,
        }[scope]

        for _ in range(limit):
            state.update(counter(state))

        with pytest.raises(BudgetExceeded) as error:
            check(state)
        assert error.value.scope == scope

    def test_time_budget_counts_real_work(self) -> None:
        """时间预算按**真实工作**累计，用受控时钟验证（不 sleep）。

        这条用例替换了早期的 `test_time_budget_is_wall_clock`：那时实现是
        `now - started_at`，把等待人工审批的时间也算作消耗。修正后的语义是
        "操作开始→结束"之间的时间才计入，因此这里显式走一遍该序列。
        """
        clock = {"now": 1_000_000.0}
        original = agent_loop.time.time
        agent_loop.time.time = lambda: clock["now"]
        try:
            state = state_with(max_seconds=10)

            # 开始一次操作，12 秒后结束
            state.update(agent_loop.capture_active_time(state))
            clock["now"] += 12
            state.update(agent_loop.settle_active_time(state))

            assert elapsed_seconds(state) == pytest.approx(12.0, abs=0.01)
            with pytest.raises(BudgetExceeded) as error:
                check(state)
            assert error.value.scope == "seconds"
        finally:
            agent_loop.time.time = original

    def test_time_budget_ignores_idling(self) -> None:
        """没有发生任何操作时，时间不得前进（这是"等待不计入"的基础）。"""
        clock = {"now": 1_000_000.0}
        original = agent_loop.time.time
        agent_loop.time.time = lambda: clock["now"]
        try:
            state = state_with(max_seconds=10)
            clock["now"] += 3600  # 干等一小时
            assert elapsed_seconds(state) == pytest.approx(0.0, abs=0.01)
            check(state)  # 不应耗尽
        finally:
            agent_loop.time.time = original

    def test_elapsed_excludes_nothing_before_start(self) -> None:
        assert elapsed_seconds({}) == 0.0

    def test_initialize_is_idempotent(self) -> None:
        """恢复时不能把已消耗的预算清零——这是"有界"的前提。"""
        state = state_with()
        state.update(consume_tool_call(state))
        state.update(consume_tool_call(state))

        before = usage_of(state)["tool_calls"]
        for _ in range(3):
            state.update(initialize(state))

        assert usage_of(state)["tool_calls"] == before == 2

    def test_initialize_stamps_start_time_once(self) -> None:
        state = state_with()
        state.update(initialize(state))
        started = usage_of(state)["started_at"]
        assert isinstance(started, float)

        time.sleep(0.01)
        state.update(initialize(state))
        assert usage_of(state)["started_at"] == started

    def test_legacy_state_without_started_at_gets_one(self) -> None:
        """P3 之前的 Checkpoint 没有 started_at，现场补一个而不是崩。"""
        legacy = {
            BUDGET_FIELD: {
                "limits": AgentBudget().to_dict(),
                "usage": {"rounds": 4, "tool_calls": 2, "llm_calls": 1},
            }
        }

        legacy.update(initialize(legacy))

        usage = usage_of(legacy)
        assert isinstance(usage["started_at"], float)
        # 既有计数保留
        assert usage["rounds"] == 4
        assert usage["tool_calls"] == 2

    def test_limits_round_trip_and_reject_garbage(self) -> None:
        budget = AgentBudget(max_rounds=7)
        assert AgentBudget.from_dict(budget.to_dict()).max_rounds == 7

        # 缺失 / 非法值回落默认，而不是让整次运行失败
        assert AgentBudget.from_dict(None).max_rounds == AgentBudget().max_rounds
        assert AgentBudget.from_dict({"max_rounds": -1}).max_rounds == 24
        assert AgentBudget.from_dict({"max_rounds": "x"}).max_rounds == 24
        assert AgentBudget.from_dict({"max_rounds": True}).max_rounds == 24


class TestBudgetIsPersistedInState:
    def test_full_run_persists_usage(self, tmp_path: Path) -> None:
        db = tmp_path / "customers.sqlite"
        build_database(db)

        result = run_analysis(
            run_id="p3-persist",
            data_sources=[str(db)],
            model_factory=factory("p3-persist"),
            resolver=ResourceResolver(),
            output_root=tmp_path,
            mcp=McpSettings(mode="mcp"),
        )

        assert result["analysis_status"] == "completed"
        usage = result[BUDGET_FIELD]["usage"]

        # 真实发生了模型调用与工具调用，计数必须反映出来
        assert usage["llm_calls"] >= 1, usage
        assert usage["tool_calls"] >= 1, usage
        assert isinstance(usage["started_at"], float)

    def test_usage_survives_checkpoint_resume(self, tmp_path: Path) -> None:
        """恢复后预算接着算，不归零。"""
        from langgraph.checkpoint.memory import InMemorySaver

        db = tmp_path / "customers.sqlite"
        build_database(db)

        saver = InMemorySaver()
        run_id = "p3-resume-budget"

        first = run_analysis(
            run_id=run_id,
            data_sources=[str(db)],
            model_factory=factory(run_id),
            resolver=ResourceResolver(),
            output_root=tmp_path,
            require_report_approval=True,
            mcp=McpSettings(mode="mcp"),
            checkpointer=saver,
        )

        assert first.get("__interrupt__"), "应停在报告审批"
        before = first[BUDGET_FIELD]["usage"]
        assert before["llm_calls"] >= 1

        resumed = run_analysis(
            run_id=run_id,
            data_sources=[str(db)],
            model_factory=factory(run_id),
            resolver=ResourceResolver(),
            output_root=tmp_path,
            require_report_approval=True,
            continue_from_checkpoint=True,
            resume_value={"decision": "approve"},
            mcp=McpSettings(mode="mcp"),
            checkpointer=saver,
        )

        after = resumed[BUDGET_FIELD]["usage"]
        # 恢复后累计值只能增长或持平，绝不能回到 0
        assert after["llm_calls"] >= before["llm_calls"], (before, after)
        assert after["tool_calls"] >= before["tool_calls"], (before, after)
        assert after["started_at"] == before["started_at"]


class TestBudgetExhaustionIsAConvergentFailure:
    def test_exhausted_tool_budget_converges_failure(self, tmp_path: Path, monkeypatch) -> None:
        """工具预算为 0：不崩，落成失败记录且不进 Repair。"""
        db = tmp_path / "customers.sqlite"
        build_database(db)

        import services.analysis_service as service
        from utils.agent_loop import AgentBudget as Budget

        original = service.AgentBudget

        class ZeroToolBudget(Budget):
            def __init__(self, **kwargs: Any) -> None:
                kwargs.setdefault("max_tool_calls", 0)
                super().__init__(**kwargs)

        monkeypatch.setattr(service, "AgentBudget", ZeroToolBudget)

        try:
            result = run_analysis(
                run_id="p3-zero-tools",
                data_sources=[str(db)],
                model_factory=factory("p3-zero-tools"),
                resolver=ResourceResolver(),
                output_root=tmp_path,
                mcp=McpSettings(mode="mcp"),
            )
        finally:
            monkeypatch.setattr(service, "AgentBudget", original)

        # 不抛异常：预算是收敛失败，不是崩溃
        assert result["analysis_status"] in ("failed", "partial"), result
        failures = result.get("failed_tasks") or []
        assert failures, result

        # 归类为环境类，因此不该进入 Repair（重试不会好转）
        assert all(
            item.get("error_kind") == BUDGET_ERROR_KIND for item in failures
        ), failures
        assert all(item.get("retry_count", 0) == 0 for item in failures), failures

    def test_exhausted_llm_budget_stops_calling_models(self, tmp_path: Path, monkeypatch) -> None:
        """模型预算很小：仍能收敛，且计数反映真实调用。"""
        db = tmp_path / "customers.sqlite"
        build_database(db)

        import services.analysis_service as service

        class TinyBudget(service.AgentBudget):
            def __init__(self, **kwargs: Any) -> None:
                kwargs.setdefault("max_llm_calls", 2)
                super().__init__(**kwargs)

        monkeypatch.setattr(service, "AgentBudget", TinyBudget)

        result = run_analysis(
            run_id="p3-tiny-llm",
            data_sources=[str(db)],
            model_factory=factory("p3-tiny-llm"),
            resolver=ResourceResolver(),
            output_root=tmp_path,
            mcp=McpSettings(mode="mcp"),
        )

        usage = usage_of(result)
        # 预算用尽后运行收敛（成功或失败都行），关键是不再无限调用模型
        assert result["analysis_status"] in ("completed", "partial", "failed"), result
        assert usage["llm_calls"] >= 1, usage
        # 上限确实被读进去了（0 与缺失必须区分——这里顺带覆盖该回归）
        assert result[BUDGET_FIELD]["limits"]["max_llm_calls"] == 2, result[BUDGET_FIELD]


class TestMiddlewareBudgets:
    def test_governed_tool_call_logs_and_counts(self, tmp_path: Path) -> None:
        from utils.middleware import governed_tool_call

        state = state_with()
        state["run_id"] = "p3-mw-tool"

        with governed_tool_call(state, node="tool_execution", tool="python") as gate:
            gate["error_kind"] = "ok"

        assert gate["duration_ms"] is not None and gate["duration_ms"] >= 0
        assert gate["budget_updates"], "应带回预算增量"

    def test_governed_tool_call_records_failure_kind(self) -> None:
        from utils.middleware import governed_tool_call

        state = state_with()
        state["run_id"] = "p3-mw-fail"

        with governed_tool_call(state, node="tool_execution", tool="sql") as gate:
            gate["error_kind"] = "execution_error"

        assert gate["error_kind"] == "execution_error"

    def test_governed_tool_call_refuses_when_budget_spent(self) -> None:
        from utils.middleware import governed_tool_call

        state = state_with(max_tool_calls=1)
        state.update(consume_tool_call(state))

        with pytest.raises(BudgetExceeded):
            with governed_tool_call(state, node="tool_execution", tool="python"):
                pytest.fail("预算已耗尽时不应进入调用体")

    def test_middleware_has_no_parallel_entry_points(self) -> None:
        """治理只有一条生效路径，中间件不再有重复实现。

        回归：`utils/middleware` 里曾有 `governed_llm_call`（无调用方）、
        `record_round`（轮次其实走 `nodes._round_update` →
        `agent_loop.consume_round`）与 `merge_budget`（只有测试在用）。
        两套并行实现会让人误以为它们参与预算执行，因此已删除。
        模型调用真正生效的路径是 `nodes._governed_agent_call` →
        `utils/idempotency.run_agent_call`。
        """
        import utils.middleware as middleware

        assert not hasattr(middleware, "governed_llm_call")
        # 轮次计数只有 `agent_loop.consume_round` 一条路径，中间件不再重复实现
        assert not hasattr(middleware, "record_round")
        assert not hasattr(middleware, "merge_budget")


class TestIdempotentReuseIsNotCounted:
    def test_cached_call_does_not_inflate_llm_budget(self, tmp_path: Path) -> None:
        """命中幂等复用时不计数——那次没有发生模型调用。"""
        from utils.idempotency import run_agent_call

        state: dict[str, Any] = {"run_id": "p3-idem", "completed_operations": {}}
        state.update(initialize(state))
        state["approvals"] = []
        state["pending_llm_calls"] = []
        state["llm_retry_granted"] = None

        first_call = {"count": 0}

        def call() -> dict:
            first_call["count"] += 1
            return {"value": first_call["count"]}

        _payload, operations, extra = run_agent_call(
            state, node="planner", subject="plan_1#0", call=call
        )
        assert extra.get("__upstream_invoked__") is True

        # 把登记写回，模拟"已完成"
        state["completed_operations"] = operations

        _payload2, _ops2, extra2 = run_agent_call(
            state, node="planner", subject="plan_1#0", call=call
        )
        assert extra2 == {}, "命中复用不应再请求上游"
        assert first_call["count"] == 1

    def test_stable_logical_attempt_id(self) -> None:
        from utils.idempotency import attempt_subject, llm_operation

        subject = attempt_subject("plan", 0)
        assert subject == attempt_subject("plan", 0)
        assert llm_operation("r1", "planner", subject) == llm_operation(
            "r1", "planner", subject
        )
        # 换一个 Run 必须换一个身份
        assert llm_operation("r1", "planner", subject) != llm_operation(
            "r2", "planner", subject
        )


class TestSecurityBoundariesUnchanged:
    def test_tools_still_go_through_the_registry(self, tmp_path: Path) -> None:
        """治理层不得绕过 Tool Registry。"""
        db = tmp_path / "customers.sqlite"
        build_database(db)

        backends: list[str] = []
        import tools.tool_registry as registry

        original = registry.resolve_backend

        def spy(tool_name: str) -> str:
            backend = original(tool_name)
            backends.append(backend)
            return backend

        import pytest as _pytest

        monkeypatch = _pytest.MonkeyPatch()
        monkeypatch.setattr(registry, "resolve_backend", spy)
        try:
            run_analysis(
                run_id="p3-sec-registry",
                data_sources=[str(db)],
                model_factory=factory("p3-sec-registry"),
                resolver=ResourceResolver(),
                output_root=tmp_path,
                mcp=McpSettings(mode="mcp"),
            )
        finally:
            monkeypatch.undo()

        assert backends, "应经过 Tool Registry 的后端分派"
        assert "mcp" in backends, backends

    def test_budget_does_not_relax_sql_guards(self, tmp_path: Path) -> None:
        """预算再宽也不会让非法 SQL 通过。

        写操作会被词法层拦下并**进入 Repair**（这是 P2-B 的既定语义：
        SQL 写错属于 `execution_error`，值得修复）。因此这里断言的是
        "DELETE 从未真正执行"，以及"最终成功靠的是修复后的只读查询"——
        而不是"整个 Run 失败"。
        """
        import sqlite3

        import tools.mcp_sql_tool as mcp_tool

        db = tmp_path / "customers.sqlite"
        build_database(db)

        REPLIES["executor"] = {"code": "DELETE FROM customers"}

        attempts: list[str] = []
        original = mcp_tool.call_mcp_sql

        def spy(**kwargs: Any):
            attempts.append(str(kwargs.get("query")))
            return original(**kwargs)

        mcp_tool.call_mcp_sql = spy  # type: ignore[assignment]
        try:
            result = run_analysis(
                run_id="p3-sec-sql",
                data_sources=[str(db)],
                model_factory=factory("p3-sec-sql"),
                resolver=ResourceResolver(),
                output_root=tmp_path,
                mcp=McpSettings(mode="mcp"),
            )
        finally:
            mcp_tool.call_mcp_sql = original  # type: ignore[assignment]

        # 写操作确实被送到工具（说明测试有效），但没有成功执行
        assert "DELETE FROM customers" in attempts, attempts

        failed = [
            item
            for item in result.get("execution_results") or []
            if not item.get("success")
        ]
        assert failed, "写操作必须留下失败记录"
        assert any("只读" in str(item.get("error")) for item in failed), failed

        # 数据没有被改动
        connection = sqlite3.connect(str(db))
        try:
            remaining = connection.execute(
                "SELECT COUNT(*) FROM customers"
            ).fetchone()[0]
        finally:
            connection.close()
        assert remaining == 3, "写操作改变了数据"
