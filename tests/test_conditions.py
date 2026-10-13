"""条件路由与循环边界。

这些函数决定工作流走哪条边，是"不会无限循环"这一保证的实际落点，
因此边界值必须逐个钉住。
"""

from __future__ import annotations

import pytest

from conditions import (
    MAX_PLOT_RETRIES,
    MAX_REFINES,
    MAX_RETRIES,
    route_after_advance,
    route_after_chart_advance,
    route_after_critic,
    route_after_execution,
    route_after_planner,
    route_after_plot_execution,
    route_after_visualization_plan,
)

TASKS = [{"task_id": "task_1"}, {"task_id": "task_2"}]
CHARTS = [{"chart_id": "chart_1"}, {"chart_id": "chart_2"}]


class TestRouteAfterPlanner:
    def test_plan_with_tasks_enters_router(self) -> None:
        assert route_after_planner({"analysis_plan": TASKS}) == "router"

    def test_empty_plan_skips_to_completion(self) -> None:
        """空计划不能进入 router：router 会取 analysis_plan[0] 并越界崩溃。"""
        assert route_after_planner({"analysis_plan": []}) == "complete"

    def test_missing_plan_skips_to_completion(self) -> None:
        assert route_after_planner({}) == "complete"


class TestRouteAfterExecution:
    def test_success_goes_to_critic(self) -> None:
        assert route_after_execution({"execution_error": None}) == "critic"

    def test_retry_below_limit_goes_to_repair(self) -> None:
        for retry_count in range(MAX_RETRIES):
            state = {"execution_error": "boom", "retry_count": retry_count}
            assert route_after_execution(state) == "repair"

    def test_retry_at_limit_gives_up(self) -> None:
        """达到上限必须转 failed，否则会无限修复。"""
        state = {"execution_error": "boom", "retry_count": MAX_RETRIES}
        assert route_after_execution(state) == "failed"

    def test_missing_retry_count_defaults_to_zero(self) -> None:
        assert route_after_execution({"execution_error": "boom"}) == "repair"


class TestRouteAfterCritic:
    def test_passed_advances(self) -> None:
        assert route_after_critic({"validation_passed": True}) == "advance"

    def test_failed_below_limit_refines(self) -> None:
        for refine_count in range(MAX_REFINES):
            state = {"validation_passed": False, "refine_count": refine_count}
            assert route_after_critic(state) == "refine"

    def test_failed_at_limit_gives_up(self) -> None:
        state = {"validation_passed": False, "refine_count": MAX_REFINES}
        assert route_after_critic(state) == "failed"

    def test_missing_validation_result_raises(self) -> None:
        """验证结果缺失属于程序错误，不能静默当作通过。"""
        with pytest.raises(ValueError, match="Validation result is missing"):
            route_after_critic({})


class TestRouteAfterAdvance:
    def test_more_tasks_remain(self) -> None:
        state = {"current_task_index": 1, "analysis_plan": TASKS}
        assert route_after_advance(state) == "router"

    def test_all_tasks_done(self) -> None:
        state = {"current_task_index": len(TASKS), "analysis_plan": TASKS}
        assert route_after_advance(state) == "complete"

    def test_empty_plan_completes_immediately(self) -> None:
        state = {"current_task_index": 0, "analysis_plan": []}
        assert route_after_advance(state) == "complete"


class TestRouteAfterPlotExecution:
    def test_success_advances(self) -> None:
        assert route_after_plot_execution({"plot_error": None}) == "advance_chart"

    def test_retry_below_limit_repairs(self) -> None:
        for retry_count in range(MAX_PLOT_RETRIES):
            state = {"plot_error": "boom", "plot_retry_count": retry_count}
            assert route_after_plot_execution(state) == "plot_repair"

    def test_retry_at_limit_advances_anyway(self) -> None:
        """绘图失败不阻塞整体流程，达到上限后继续下一个图表。"""
        state = {"plot_error": "boom", "plot_retry_count": MAX_PLOT_RETRIES}
        assert route_after_plot_execution(state) == "advance_chart"


class TestRouteAfterChartAdvance:
    def test_more_charts_remain(self) -> None:
        state = {"current_chart_index": 1, "visualization_plan": CHARTS}
        assert route_after_chart_advance(state) == "plot_generation"

    def test_all_charts_done(self) -> None:
        state = {"current_chart_index": len(CHARTS), "visualization_plan": CHARTS}
        assert route_after_chart_advance(state) == "report"


class TestRouteAfterVisualizationPlan:
    def test_plan_present_generates_plots(self) -> None:
        assert route_after_visualization_plan({"visualization_plan": CHARTS}) == (
            "plot_generation"
        )

    def test_empty_plan_skips_to_report(self) -> None:
        assert route_after_visualization_plan({"visualization_plan": []}) == "report"

    def test_missing_plan_skips_to_report(self) -> None:
        assert route_after_visualization_plan({}) == "report"


def test_all_loops_are_bounded() -> None:
    """三个循环上限都必须是有限正整数，这是"不会无限执行"的前提。"""
    for limit in (MAX_RETRIES, MAX_REFINES, MAX_PLOT_RETRIES):
        assert isinstance(limit, int)
        assert limit > 0
