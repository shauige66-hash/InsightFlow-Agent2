"""P3 定点修复：工具调用结束时必须把结算后的执行时钟写回 state。

`governed_tool_call` 在 `finally` 里调用 `settle_active_time` 把时钟停住，
并把结果放回 `gate["budget_updates"]`。调用方若在 `with` **块内**读取这个
字段，拿到的就是结算前的值——`clock_open` 留在 True、`active_seconds` 少算
本次工具耗时。时钟不关，恢复后每次操作都会把等待人工审批的时间（单次封顶
`MAX_ACTIVE_GAP_SECONDS`）算进执行预算。

`tool_execution` 节点同时承载 python 与 sql 两条路径（同一个治理入口），
绘图走 `plot_execution`。
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from core.runtime import McpSettings
from ingestion.resources import ResourceResolver
from services.analysis_service import run_analysis
from utils import agent_loop as al
from utils.middleware import governed_tool_call

from tests.test_e2e_business_scenarios import (
    PLOT_CODE,
    build_csv,
    charts_of,
    plan_of,
)
from tests.test_e2e_business_scenarios import ScriptedFactory
from tests.test_report_approval import REPLIES

#: 每一步操作经历的"墙上时间"。远小于 MAX_ACTIVE_GAP_SECONDS（60），
#: 因此每一步都应当被完整计入 active_seconds。
STEP_SECONDS = 1.0


@pytest.fixture(autouse=True)
def _restore_replies():
    original = copy.deepcopy(REPLIES)
    yield
    REPLIES.clear()
    REPLIES.update(copy.deepcopy(original))


def step_clock(monkeypatch: pytest.MonkeyPatch) -> dict[str, float]:
    """把 `time.time` 换成一个每次调用前进 STEP_SECONDS 的单调假时钟。

    执行时钟只读 `utils.agent_loop` 里的 `time`，因此在这里替换即可；
    `perf_counter`（只用于耗时日志）保持真实。
    """
    clock = {"now": 1_000_000.0}

    def fake_time() -> float:
        clock["now"] += STEP_SECONDS
        return clock["now"]

    monkeypatch.setattr(al.time, "time", fake_time)
    return clock


def fake_tool(records: list[dict[str, Any]] | None = None):
    """替身工具执行器：不启动任何容器，返回正常结果。"""
    payload = records if records is not None else [{"region": "east", "sales": 10}]

    def _invoke(
        tool_name: str,
        tables: dict[str, Any],
        arguments: dict[str, Any],
        *,
        snapshots: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del tables, snapshots
        if tool_name == "plot":
            # 绘图结果是一个宿主路径，这里给一个不存在的路径即可
            return {
                "success": True,
                "result": str(Path(arguments.get("chart_id", "chart")) / "chart.png"),
                "error": None,
                "error_kind": None,
            }
        return {
            "success": True,
            "result": payload,
            "error": None,
            "error_kind": None,
        }

    return _invoke


def run_settlement_workflow(
    tmp_path: Path,
    *,
    run_id: str,
    with_charts: bool,
    data_sources: list[Path],
    target_tables: list[str],
) -> dict[str, Any]:
    replies: dict[str, Any] = {
        "planner": plan_of(target_tables),
        "executor": {"code": "result = df.copy()\n"},
        "repair": {"code": "result = df.copy()\n"},
        "refine": {"code": "result = df.copy()\n"},
        "critic": {"validation_passed": True, "feedback": "ok"},
        "visualizer": charts_of(target_tables) if with_charts else {"charts": []},
        "plotter": {"code": PLOT_CODE},
        "plot_repair": {"code": PLOT_CODE},
        "reporter": "# Analysis Report\n",
    }
    recorder = ScriptedFactory(run_id, replies)
    return run_analysis(
        run_id=run_id,
        data_sources=[str(item) for item in data_sources],
        model_factory=recorder.build(),
        resolver=ResourceResolver(),
        output_root=tmp_path,
        mcp=McpSettings(mode="direct"),
    )


def final_usage(result: dict[str, Any]) -> dict[str, Any]:
    return ((result.get("execution_budget") or {}).get("usage")) or {}


class TestGovernedToolCallSettlement:
    def test_settlement_is_visible_only_after_the_block_exits(self) -> None:
        """块内取到的是结算前的值，块外才是结算后的值。"""
        from utils.agent_loop import initialize

        state: dict[str, Any] = {"run_id": "settle-gate"}
        state.update(initialize(state))

        inside: dict[str, Any] = {}

        with governed_tool_call(state, node="tool_execution", tool="python") as gate:
            inside = gate.get("budget_updates") or {}

        after = gate.get("budget_updates") or {}

        inside_usage = (inside.get(al.BUDGET_FIELD) or {}).get("usage", {})
        after_usage = (after.get(al.BUDGET_FIELD) or {}).get("usage", {})

        assert inside_usage.get("clock_open") is True, "块内不应看到结算后的时钟"
        assert after_usage.get("clock_open") is False, "块外时钟应当已停"


class TestCallSiteReadsTheSettledValue:
    """在真实工作流里捕获调用点读到的那个值。"""

    def test_tool_execution_captures_the_post_settlement_updates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import nodes

        step_clock(monkeypatch)
        real = nodes.governed_tool_call
        captured: dict[str, Any] = {}

        @contextmanager
        def spy(*args: Any, **kwargs: Any):
            with real(*args, **kwargs) as gate:
                yield gate
                captured["inside"] = copy.deepcopy(gate.get("budget_updates") or {})
            # 出块之后 gate 里才是结算后的那份
            captured["after"] = copy.deepcopy(gate.get("budget_updates") or {})

        monkeypatch.setattr(nodes, "governed_tool_call", spy)

        source = build_csv(tmp_path / "sales.csv")
        result = run_settlement_workflow(
            tmp_path,
            run_id="settle-callsite",
            with_charts=False,
            data_sources=[source],
            target_tables=["sales"],
        )

        inside = ((captured["inside"].get(al.BUDGET_FIELD) or {}).get("usage")) or {}
        after = ((captured["after"].get(al.BUDGET_FIELD) or {}).get("usage")) or {}

        assert inside.get("clock_open") is True, "块内本应是结算前的值"
        assert after.get("clock_open") is False, "块外应当已结算"

        persisted = final_usage(result)
        assert persisted.get("clock_open") is False, (
            f"最终落盘的是块内那份：clock_open={persisted.get('clock_open')}"
        )

class TestToolPathsCloseTheClock:
    @pytest.mark.parametrize(
        ("tool", "code"),
        [("python", "result = df.copy()\n"), ("sql", "SELECT * FROM sales")],
    )
    def test_tool_node_return_value_carries_the_settled_clock(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        tool: str,
        code: str,
    ) -> None:
        """节点返回的增量里 clock_open 必须是结算后的 False。

        端到端受影响被后续的模型调用掩盖（`_governed_agent_call` 会重新结算
        一次），因此这里直接检查节点交给 LangGraph 的那份增量。
        """
        import nodes
        from core.runtime import RunRuntime, use_runtime
        from ingestion.resources import build_snapshots, snapshots_to_state
        from utils.agent_loop import initialize as init_budget

        step_clock(monkeypatch)
        monkeypatch.setattr(nodes, "invoke_tool", fake_tool())

        source = build_csv(tmp_path / "sales.csv")
        state: dict[str, Any] = {
            "run_id": "settle-node",
            "current_task_index": 0,
            "current_tool": tool,
            "current_code": code,
            "analysis_plan": plan_of(["sales"])["tasks"],
            "data_sources": [str(source)],
            "table_snapshot": snapshots_to_state(build_snapshots([str(source)])),
            "retry_count": 0,
            "refine_count": 0,
            "execution_results": [],
        }
        state.update(init_budget(state))

        runtime = RunRuntime(
            run_id="settle-node",
            model_factory=None,  # type: ignore[arg-type]
            resolver=ResourceResolver(),
            output_root=tmp_path / "outputs",
        )
        with use_runtime(runtime):
            updates = nodes.tool_execution_node(state)

        usage = ((updates.get(al.BUDGET_FIELD) or {}).get("usage")) or {}
        assert usage.get("tool_calls") == 1, "工具调用没有被计数"
        assert usage.get("clock_open") is False, (
            f"节点返回的增量里时钟仍在走：clock_open={usage.get('clock_open')}"
        )

    @pytest.mark.parametrize("with_charts", [False, True])
    def test_python_sql_and_plot_leave_the_clock_closed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_charts: bool
    ) -> None:
        """三条工具路径跑完后 clock_open 必须为 False，且耗时被累计。"""
        import nodes

        step_clock(monkeypatch)
        monkeypatch.setattr(nodes, "invoke_tool", fake_tool())

        source = build_csv(tmp_path / "sales.csv")
        result = run_settlement_workflow(
            tmp_path,
            run_id=f"settle-tools-{with_charts}",
            with_charts=with_charts,
            data_sources=[source],
            target_tables=["sales"],
        )

        usage = final_usage(result)
        assert usage.get("clock_open") is False, (
            f"工具调用后执行时钟没有停住：clock_open={usage.get('clock_open')}，"
            f"active_seconds={usage.get('active_seconds')}"
        )
        assert usage.get("tool_calls", 0) >= 1, "工具调用没有被计数"
        assert usage.get("active_seconds", 0.0) > 0.0, (
            "工具耗时没有被累计进 active_seconds"
        )

        # 时钟已停：再造一段人工审批等待，执行时间不得增长。
        # 用固定时刻替换假时钟，使断言不依赖"第几次调用"。
        elapsed = al.elapsed_seconds(result)
        monkeypatch.setattr(al.time, "time", lambda: 1_000_000.0 + 3 * 3600)
        assert al.elapsed_seconds(result) == pytest.approx(elapsed), (
            "审批等待时间被计入了执行预算"
        )
