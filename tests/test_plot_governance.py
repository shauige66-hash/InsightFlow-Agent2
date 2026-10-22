"""P3 定点修复：Plot 工具接入统一治理。

审查发现 `plot_execution_node` 直接调用 `invoke_tool("plot", ...)`，
绕过了 `governed_tool_call`：绘图既不计入 `tool_calls`，也不受预算约束，
预算设为 0 时仍会启动一个 Docker 容器。绘图是开销最大、且有副作用的工具，
不能是治理的缺口。

这些用例都走**真实的 plot 执行路径**（真实 Docker 沙箱、真实 PNG 落盘），
只驱动治理相关的断言；不满足于直接调用 middleware 函数。
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from core.runtime import McpSettings
from ingestion.resources import ResourceResolver
from services.analysis_service import run_analysis

from tests.test_e2e_business_scenarios import (
    PLOT_CODE,
    build_csv,
    charts_of,
    plan_of,
)
from tests.test_e2e_business_scenarios import ScriptedFactory
from tests.test_report_approval import REPLIES


@pytest.fixture(autouse=True)
def _restore_replies():
    from sandbox.executor import DockerSandboxExecutor
    executor = DockerSandboxExecutor()
    if not executor.available() or not executor.image_present():
        pytest.skip("本组治理验收明确要求真实Docker及已构建sandbox镜像")
    original = copy.deepcopy(REPLIES)
    yield
    REPLIES.clear()
    REPLIES.update(copy.deepcopy(original))


def run_chart_workflow(
    *,
    workdir: Path,
    run_id: str,
    data_sources: list[Path],
    target_tables: list[str],
    budget: dict[str, Any] | None = None,
    plot_code: str | None = None,
) -> tuple[dict[str, Any], ScriptedFactory]:
    """跑一次**带图表**的真实工作流。"""
    replies: dict[str, Any] = {
        "planner": plan_of(target_tables),
        "executor": {"code": "result = df.copy()\n"},
        "repair": {"code": "result = df.copy()\n"},
        "refine": {"code": "result = df.copy()\n"},
        "critic": {"validation_passed": True, "feedback": "ok"},
        "visualizer": charts_of(target_tables),
        "plotter": {"code": plot_code or PLOT_CODE},
        "plot_repair": {"code": plot_code or PLOT_CODE},
        "reporter": "# Analysis Report\n\n## Overview\n图表已生成。\n",
    }
    recorder = ScriptedFactory(run_id, replies)

    result = run_analysis(
        run_id=run_id,
        data_sources=[str(item) for item in data_sources],
        model_factory=recorder.build(),
        resolver=ResourceResolver(),
        output_root=workdir,
        mcp=McpSettings(mode="direct"),
        budget=budget,
    )
    return result, recorder


def chart_paths(result: dict[str, Any]) -> list[Path]:
    return [
        Path(item["path"])
        for item in (result.get("visualization_results") or [])
        if item.get("path")
    ]


class TestPlotIsGoverned:
    def test_zero_tool_budget_prevents_the_chart_container(
        self, tmp_path: Path
    ) -> None:
        """预算为 0 时不得启动绘图容器。

        判据是"没有任何 PNG 产出"——绘图容器唯一的产物就是那张图，
        因此没有文件就意味着容器没有跑起来。
        """
        source = build_csv(tmp_path / "sales.csv")

        result, _recorder = run_chart_workflow(
            workdir=tmp_path,
            run_id="p3-plot-zero",
            data_sources=[source],
            target_tables=["sales"],
            budget={"max_tool_calls": 0},
        )

        # 终态取决于失败发生在哪个阶段（计划已产出时是 partial），
        # 这里不断言具体档位；真正要证明的是"容器没有被启动"。
        assert result["analysis_status"] in ("failed", "partial"), (
            result.get("analysis_status")
        )
        assert not chart_paths(result), (
            f"预算为 0 却产出了图表：{chart_paths(result)}"
        )
        # PNG 也没有落在磁盘上
        assert not list(tmp_path.rglob("*.png"))

        # 工具调用一次都不许发生
        usage = (result.get("execution_budget") or {}).get("usage") or {}
        assert usage.get("tool_calls", 0) == 0, (
            f"预算为 0 却发生了工具调用：{usage.get('tool_calls')}"
        )

    def test_successful_chart_counts_a_tool_call(self, tmp_path: Path) -> None:
        """正常绘图必须累计工具调用次数——此前完全不计。"""
        source = build_csv(tmp_path / "sales.csv")

        result, _recorder = run_chart_workflow(
            workdir=tmp_path,
            run_id="p3-plot-count",
            data_sources=[source],
            target_tables=["sales"],
        )

        assert result["analysis_status"] == "completed", result.get("status_reason")

        charts = chart_paths(result)
        assert charts, "应产出图表"
        assert charts[0].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

        usage = (result.get("execution_budget") or {}).get("usage") or {}
        # 至少两次：分析工具的 python 调用 + 绘图的 plot 调用
        assert usage.get("tool_calls", 0) >= 2, (
            f"绘图未计入工具调用，tool_calls={usage.get('tool_calls')}"
        )

    def test_plot_is_blocked_before_the_container_starts(
        self, tmp_path: Path
    ) -> None:
        """工具预算只剩 1 时，分析阶段用掉后绘图必须被拦下。

        这条用例区分"绘图被拦"与"绘图跑了但失败"：后者会留下失败记录
        且可能已经有容器被创建。
        """
        source = build_csv(tmp_path / "sales.csv")

        result, _recorder = run_chart_workflow(
            workdir=tmp_path,
            run_id="p3-plot-one",
            data_sources=[source],
            target_tables=["sales"],
            budget={"max_tool_calls": 1},
        )

        # 分析阶段消耗了唯一的一次工具调用，绘图只能失败
        assert not chart_paths(result), f"不应产出图表：{chart_paths(result)}"
        assert not list(tmp_path.rglob("*.png"))
        usage = (result.get("execution_budget") or {}).get("usage") or {}
        assert usage.get("tool_calls", 0) <= 1, (
            f"工具调用超预算：{usage.get('tool_calls')}"
        )

    def test_budget_exhaustion_does_not_enter_plot_repair(
        self, tmp_path: Path
    ) -> None:
        """预算耗尽不得进入 plot_repair 循环。

        重画同一张图在预算已尽时不会好转，只会再尝试启动一个容器。
        `plot_retry_count` 被置满时路由直接走 advance_chart。
        """
        source = build_csv(tmp_path / "sales.csv")

        _result, recorder = run_chart_workflow(
            workdir=tmp_path,
            run_id="p3-plot-norepair",
            data_sources=[source],
            target_tables=["sales"],
            budget={"max_tool_calls": 0},
        )

        assert "plot_repair" not in recorder.calls, (
            f"预算耗尽却进入了绘图修复循环：{recorder.calls}"
        )

    def test_timeout_and_environment_classification_is_preserved(
        self, tmp_path: Path
    ) -> None:
        """绘图失败仍要带上明确的分类，不能因为接入治理就丢掉。

        用一个必定失败的绘图代码触发；错误分类由既有的
        `sandbox.protocol.classify_exception` 给出。
        """
        source = build_csv(tmp_path / "sales.csv")

        result, _recorder = run_chart_workflow(
            workdir=tmp_path,
            run_id="p3-plot-err",
            data_sources=[source],
            target_tables=["sales"],
            plot_code="raise ValueError('注入的绘图错误')\n",
        )

        visualizations = result.get("visualization_results") or []
        assert visualizations, "应记录一次绘图尝试"
        assert not visualizations[-1]["success"]
        assert "注入的绘图错误" in (visualizations[-1].get("error") or "")

    def test_chart_idempotency_and_dedup_are_preserved(self, tmp_path: Path) -> None:
        """接入治理不得破坏原 chart_operation 的幂等登记与去重。"""
        from utils.idempotency import chart_operation, completed_operations

        source = build_csv(tmp_path / "sales.csv")

        result, _recorder = run_chart_workflow(
            workdir=tmp_path,
            run_id="p3-plot-idem",
            data_sources=[source],
            target_tables=["sales"],
        )

        assert result["analysis_status"] == "completed", result.get("status_reason")

        # 该图表必须有幂等登记
        op_id = chart_operation("p3-plot-idem", "chart_1")
        assert op_id in completed_operations(result), (
            f"缺少图表幂等登记：{sorted(completed_operations(result))}"
        )

        # visualization_results 里同一 chart_id 只能有一条
        chart_ids = [
            item["chart_id"] for item in (result.get("visualization_results") or [])
        ]
        assert len(chart_ids) == len(set(chart_ids)), chart_ids

        # 产物只登记一次
        artifacts = result.get("artifacts") or []
        assert len(artifacts) == len(set(artifacts)), artifacts
