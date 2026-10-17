"""架构约束：LangGraph 编排与 Harness 执行治理的分界。

这些用例不看 Harness 内部函数能否单独工作，而是看**真实的图执行**有没有走
Harness：模型调用与工具派发都在 Harness 的方法上打点，然后跑一次真实工作流
（Mock 模型 + 替身工具，不启动容器），断言打点被命中。

同时用源码断言固定分界：节点层不再出现账本、预算计数、工具治理等底层原语；
Harness 不另行实现 Ledger、Sandbox 或 Checkpoint。
"""

from __future__ import annotations

import copy
import inspect
import threading
from pathlib import Path
from typing import Any

import pytest

import harness
from harness.llm_gateway import LLMGateway
from harness.tool_dispatcher import ToolDispatcher
from services.analysis_service import run_analysis

from tests.test_e2e_business_scenarios import PLOT_CODE, build_csv, charts_of, plan_of
from tests.test_e2e_business_scenarios import ScriptedFactory
from tests.test_report_approval import REPLIES

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _restore_replies():
    original = copy.deepcopy(REPLIES)
    yield
    REPLIES.clear()
    REPLIES.update(copy.deepcopy(original))


def fake_tool():
    """替身工具：形状与 `invoke_tool` 的归一化结果一致，不启动容器。"""

    def _invoke(
        tool_name: str,
        tables: dict[str, Any],
        arguments: dict[str, Any],
        *,
        snapshots: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del tables, snapshots
        if tool_name == "plot":
            return {
                "success": True,
                "result": "chart.png",
                "error": None,
                "error_kind": None,
            }
        return {
            "success": True,
            "result": [{"region": "east", "sales": 10}],
            "error": None,
            "error_kind": None,
        }

    return _invoke


def run_chart_workflow(
    tmp_path: Path,
    *,
    run_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    from harness import tool_dispatcher

    monkeypatch.setattr(tool_dispatcher, "invoke_tool", fake_tool())
    source = build_csv(tmp_path / "sales.csv")
    replies = {
        "planner": plan_of(["sales"]),
        "executor": {"code": "result = df.copy()\n"},
        "repair": {"code": "result = df.copy()\n"},
        "refine": {"code": "result = df.copy()\n"},
        "critic": {"validation_passed": True, "feedback": "ok"},
        "visualizer": charts_of(["sales"]),
        "plotter": {"code": PLOT_CODE},
        "plot_repair": {"code": PLOT_CODE},
        "reporter": "# Analysis Report\n",
    }
    recorder = ScriptedFactory(run_id, replies)
    return run_analysis(
        run_id=run_id,
        data_sources=[str(source)],
        model_factory=recorder.build(),
        output_root=tmp_path,
    )


class TestPackageShape:
    def test_the_package_has_the_declared_modules(self) -> None:
        for name in (
            "__init__.py",
            "runtime.py",
            "llm_gateway.py",
            "tool_dispatcher.py",
            "policy.py",
        ):
            assert (PROJECT_ROOT / "harness" / name).is_file(), f"缺少 harness/{name}"

    def test_get_harness_returns_a_process_singleton(self) -> None:
        assert harness.get_harness() is harness.get_harness()

    def test_the_harness_exposes_the_three_capabilities(self) -> None:
        runtime = harness.get_harness()
        assert isinstance(runtime.llm, LLMGateway)
        assert isinstance(runtime.tools, ToolDispatcher)
        # policy 决策函数带着真实语义，不是空包装
        assert runtime.policy.ensure_model_request_allowed is not None


class TestNodeLayerNoLongerOwnsGovernance:
    """节点层不得再出现账本、预算计数或工具治理的底层原语。"""

    #: 这些名字各自代表一条治理路径；出现在节点层就说明节点又绕过 Harness 了。
    FORBIDDEN = (
        "invoke_tool(",
        "run_agent_call(",
        "governed_tool_call(",
        "check_llm_budget(",
        "settle_active_time(",
        "consume_llm_call(",
        "consume_tool_call(",
        "llm_ledger",
        "_CURRENT_STATE",
    )

    def test_nodes_do_not_call_governance_primitives(self) -> None:
        source = (PROJECT_ROOT / "nodes.py").read_text(encoding="utf-8")
        found = [name for name in self.FORBIDDEN if name in source]
        assert not found, f"节点层仍在直接使用：{found}"

    def test_every_agent_drives_the_request_hooks(self) -> None:
        """计数发生在请求钩子里，因此 Agent 必须驱动它。

        八个 Agent 经 `utils.llm_retry` 的重试辅助函数发出请求；plotter 自带
        重试循环，直接调用同一对钩子。两种方式都要在请求前后经过钩子，
        否则预算判定与计数都会被绕过。
        """
        agents_dir = PROJECT_ROOT / "agents"
        model_agents: list[str] = []
        for path in sorted(agents_dir.glob("*.py")):
            source = path.read_text(encoding="utf-8")
            makes_request = any(
                marker in source
                for marker in (
                    "invoke_json_with_retry(",
                    "invoke_report_with_retry(",
                    "chain.invoke(",
                )
            )
            if not makes_request:
                continue
            model_agents.append(path.stem)
            assert "before_model_request" in source, (
                f"{path.name} 发出请求前没有判预算"
            )
            assert "after_model_request" in source, (
                f"{path.name} 发出请求后没有计数"
            )

        assert sorted(model_agents) == sorted(
            [
                "planner",
                "executor",
                "repair",
                "refine",
                "critic",
                "visualizer",
                "plotter",
                "plot_repair",
                "reporter",
            ]
        ), model_agents

    def test_model_nodes_are_wrapped_and_tool_nodes_pass_state_explicitly(
        self,
    ) -> None:
        """两条入口的绑定方式不同，各自要满足自己的前提。

        模型调用靠 `_with_state` 把 state 放进执行上下文（Agent 的请求钩子只有
        这条路能拿到它）；工具调用直接把 state 传给派发器，因此不需要包装。
        任一侧退回到"没有 state"的状态，预算判定就读不到当前 Run。
        """
        import nodes

        model_nodes: list[str] = []
        tool_nodes: list[str] = []

        for name, node in vars(nodes).items():
            if not name.endswith("_node") or not callable(node):
                continue
            source = inspect.getsource(node)
            uses_model = "HARNESS.model_call(" in source
            uses_tool = "HARNESS.run_tool(" in source
            if uses_model:
                model_nodes.append(name)
                assert hasattr(node, "__wrapped__"), (
                    f"{name} 调用了模型却没有被 _with_state 包装，"
                    "请求钩子读不到当前 Run 的 state"
                )
            if uses_tool:
                tool_nodes.append(name)
                assert "HARNESS.run_tool(\n            state," in source, (
                    f"{name} 没有把 state 显式交给派发器"
                )

        assert sorted(model_nodes) == sorted(
            [
                "planner_node",
                "execution_node",
                "repair_node",
                "critic_node",
                "refine_node",
                "report_node",
                "visualization_planner_node",
                "plot_generation_node",
                "plot_repair_node",
                # H5：观察后的决策与运行期修订同样是一次受治理的模型调用
                "decide_node",
                "runtime_replan_node",
            ]
        ), model_nodes
        assert sorted(tool_nodes) == ["plot_execution_node", "tool_execution_node"], (
            tool_nodes
        )


class TestTheGraphActuallyGoesThroughTheHarness:
    def test_every_model_call_enters_the_gateway(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agents: list[str] = []
        real_call = LLMGateway.call_agent

        def spy_call(self, agent, subject, invoke, *, attempt=1):
            agents.append(agent)
            return real_call(self, agent, subject, invoke, attempt=attempt)

        monkeypatch.setattr(LLMGateway, "call_agent", spy_call)
        result = run_chart_workflow(
            tmp_path, run_id="arch-gateway", monkeypatch=monkeypatch
        )

        assert result["analysis_status"] == "completed", result.get("status_reason")
        # 这张图会走到的模型节点
        for expected in ("planner", "executor", "critic", "visualizer", "reporter"):
            assert expected in agents, f"{expected} 没有经过 LLM Gateway：{agents}"

    def test_every_tool_execution_enters_the_dispatcher(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dispatched: list[tuple[str, str]] = []
        real_dispatch = ToolDispatcher.dispatch

        def spy_dispatch(self, state, *, node, tool, **kwargs):
            dispatched.append((node, tool))
            return real_dispatch(self, state, node=node, tool=tool, **kwargs)

        monkeypatch.setattr(ToolDispatcher, "dispatch", spy_dispatch)
        result = run_chart_workflow(
            tmp_path, run_id="arch-dispatcher", monkeypatch=monkeypatch
        )

        assert result["analysis_status"] == "completed", result.get("status_reason")
        assert ("tool_execution", "python") in dispatched, dispatched
        assert ("plot_execution", "plot") in dispatched, dispatched

    def test_tool_calls_are_counted_once_per_dispatch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """派发层与节点层各自计数会让额度按一半速度消耗。"""
        result = run_chart_workflow(
            tmp_path, run_id="arch-count", monkeypatch=monkeypatch
        )
        usage = (result.get("execution_budget") or {}).get("usage") or {}
        # 一次 python 分析 + 一次绘图
        assert usage.get("tool_calls") == 2, usage


class TestTheHarnessDoesNotReimplementThePrimitives:
    def test_no_ledger_or_sandbox_reimplementation(self) -> None:
        """Harness 只编排，不另写账本、沙箱或 Checkpoint。"""
        forbidden = ("record_started", "record_completed", "class .*Sandbox", "sqlite3")
        for module in ("policy", "llm_gateway", "tool_dispatcher", "runtime"):
            source = (PROJECT_ROOT / "harness" / f"{module}.py").read_text(
                encoding="utf-8"
            )
            for marker in forbidden:
                assert marker not in source, f"harness/{module}.py 出现 {marker!r}"

    def test_ledger_and_idempotency_are_reused(self) -> None:
        import utils.idempotency
        import utils.llm_ledger

        gateway_source = (PROJECT_ROOT / "harness" / "llm_gateway.py").read_text(
            encoding="utf-8"
        )
        assert "utils.idempotency" in gateway_source
        # 底层模块仍在，且被真实驱动
        assert utils.idempotency.run_agent_call is not None
        assert utils.llm_ledger.record_started is not None


class TestPolicyIsARealEntry:
    def test_the_guard_refuses_when_the_llm_budget_is_spent(self) -> None:
        from utils.agent_loop import BUDGET_FIELD, AgentBudget, initialize

        state: dict[str, Any] = {"run_id": "policy-guard"}
        state.update(initialize(state))
        state[BUDGET_FIELD]["limits"]["max_llm_calls"] = 0

        with pytest.raises(Exception) as excinfo:
            harness.policy.ensure_model_request_allowed(state)
        assert "llm_calls" in str(excinfo.value)

    def test_note_round_counts_once(self) -> None:
        from utils.agent_loop import BUDGET_FIELD

        state: dict[str, Any] = {"run_id": "policy-round"}
        state.update(harness.policy.initialize(state))

        # 计数以增量返回，由节点并入自己的 state 更新，因此看返回值而不是 state
        first = harness.policy.count_round(state)
        assert first[BUDGET_FIELD]["usage"]["rounds"] == 1

        state.update(first)
        second = harness.policy.count_round(state)
        assert second[BUDGET_FIELD]["usage"]["rounds"] == 2


class TestOfflineDemo:
    """离线演示入口必须真的展示预算与追踪行为，而不是只打印一段文字。"""

    def test_the_demo_counts_retries_and_refuses_at_the_cap(
        self, tmp_path: Path
    ) -> None:
        from scripts.manual.harness_demo import run_demo

        result = run_demo(max_llm_calls=2, trace_dir=tmp_path / "trace")

        first, second = result["events"]

        # 一次逻辑调用内部重发：两次真实请求，额度用满
        assert first["step"] == "retrying-call"
        assert first["requests"] == 2
        assert first["llm_calls_used"] == 2
        assert first["result"] == {"ok": True}

        # 下一次调用在发出请求之前被拒绝，且没有增加用量
        assert second["step"] == "refused"
        assert second["scope"] == "llm_calls"
        assert second["used"] == second["limit"] == 2
        assert second["llm_calls_used"] == 2

    def test_the_demo_settles_the_clock_and_leaves_a_paired_ledger_entry(
        self, tmp_path: Path
    ) -> None:
        from utils.idempotency import llm_operation

        from scripts.manual.harness_demo import RUN_ID, run_demo

        result = run_demo(max_llm_calls=2, trace_dir=tmp_path / "trace")

        assert result["usage"]["clock_open"] is False
        assert result["remaining"]["llm_calls"] == 0

        # 只有成功的那次逻辑调用留下了账本记录，且 started/completed 成对
        assert result["ledger"]["started"] == 1
        assert result["ledger"]["completed"] == 1
        assert result["ledger"]["operation_ids"] == [
            llm_operation(RUN_ID, "executor", "demo#0")
        ]

    def test_the_demo_trace_is_written_under_the_trace_dir(
        self, tmp_path: Path
    ) -> None:
        from scripts.manual.harness_demo import run_demo

        trace_dir = tmp_path / "trace"
        result = run_demo(max_llm_calls=2, trace_dir=trace_dir)

        ledger_file = Path(result["ledger_path"])
        assert ledger_file.is_file(), "演示没有留下账本"
        assert ledger_file.is_relative_to(trace_dir)


class TestConcurrentRunsAreIsolated:
    def test_state_binding_is_per_execution_context(self) -> None:
        seen: dict[str, str] = {}
        errors: list[BaseException] = []

        def worker(run_id: str) -> None:
            try:
                with harness.bind_state({"run_id": run_id}):
                    seen[run_id] = harness.current_state()["run_id"]
            except BaseException as error:  # noqa: BLE001
                errors.append(error)

        threads = [
            threading.Thread(target=worker, args=(f"run-{index}",)) for index in (1, 2)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        assert not errors, errors
        assert seen == {"run-1": "run-1", "run-2": "run-2"}

    def test_binding_is_cleared_after_the_block(self) -> None:
        with harness.bind_state({"run_id": "scoped"}):
            assert harness.current_state()["run_id"] == "scoped"
        assert harness.current_state() is None
