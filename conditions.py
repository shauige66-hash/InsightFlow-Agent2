# 定义 LangGraph 工作流中的条件分支判断逻辑。

from typing import Literal

from sandbox.protocol import ENVIRONMENT_ERROR_KINDS
from state import InsightFlowState


MAX_RETRIES = 3
MAX_REFINES = 3
MAX_PLOT_RETRIES = 3

def route_after_planner(
    state: InsightFlowState
) -> Literal["router", "complete"]:
    """计划为空时直接收尾。

    Planner 可能返回空任务列表（模型判断数据不足以支撑分析）。
    此时若照常进入 router，会因为 analysis_plan[0] 越界而抛 IndexError，
    把一次正常的"无可分析"变成工作流崩溃。
    """
    return "router" if state.get("analysis_plan") else "complete"


def route_to_approval(
    state: InsightFlowState
) -> Literal["approval_gate", "router", "complete"]:
    """决定计划产出后是进入人工审批、直接执行，还是收尾。

    空计划优先：没有任务可分析时不需要打扰审批人。
    `require_plan_approval` 让 HITL 成为显式选择，未开启时保持原有
    全自动路径，避免所有调用方都被迫处理中断。
    """
    if not state.get("analysis_plan"):
        return "complete"

    if state.get("require_plan_approval"):
        return "approval_gate"

    return "router"


def route_after_approval(
    state: InsightFlowState
) -> Literal["router", "complete"]:
    """审批被拒绝时不再进入执行阶段。

    该分支只在 resume 之后才会被求值：中断期间图停在审批节点，
    不会走到这里。因此"拒绝"能真正阻止后续的执行与报告生成。
    """
    decision = state.get("plan_approval") or {}

    if decision.get("decision") == "approve":
        return "router"

    return "complete"


def route_to_report_approval(
    state: InsightFlowState
) -> Literal["report_publish_gate", "artifact"]:
    """报告已经生成，决定是发布前人工确认还是直接落盘。

    与 `route_to_approval` 对称：`require_report_approval` 是显式开关，
    未开启时保持既有的全自动路径。

    放在 report 之后、artifact 之前：此时报告**内容**已在 state 里，
    但文件尚未写出，因此中断与恢复都不会留下半成品产物。
    """
    if state.get("require_report_approval"):
        return "report_publish_gate"

    return "artifact"


def route_after_report_approval(
    state: InsightFlowState
) -> Literal["artifact", "complete"]:
    """报告确认被拒绝时不发布。

    拒绝意味着"这份报告不交付"：不写文件、也不给出产物列表，
    与计划被拒的行为保持一致（被拒的分析不该产出交付物）。
    """
    decision = state.get("report_approval") or {}

    if decision.get("decision") == "approve":
        return "artifact"

    return "complete"


def route_after_execution(
    state: InsightFlowState
) -> Literal["repair", "critic", "failed"]:

    error = state.get("execution_error")
    retry_count = state.get("retry_count", 0)

    if error is None:
        return "critic"

    # 环境类失败（沙箱不可用、超时、OOM）重发同样的代码不会好转：
    # 让模型去"修"一个根本没跑起来的沙箱只会浪费调用并掩盖真实原因。
    # 这类失败直接进入失败记录，不占用有界修复循环的额度。
    if state.get("execution_error_kind") in ENVIRONMENT_ERROR_KINDS:
        return "failed"

    if retry_count < MAX_RETRIES:
        return "repair"

    return "failed"



def route_after_advance(
    state: InsightFlowState
) -> Literal["router", "complete"]:

    index = state["current_task_index"]

    total_tasks = len(
        state["analysis_plan"]
    )

    if index < total_tasks:
        return "router"

    return "complete"






def route_after_critic(
    state: InsightFlowState
) -> Literal["advance", "refine", "failed"]:

    passed = state.get("validation_passed")
    refine_count = state.get(
        "refine_count",
        0
    )

    if passed is True:
        return "advance"

    if passed is False:
        if refine_count < MAX_REFINES:
            return "refine"

        return "failed"

    raise ValueError(
        "Validation result is missing."
    )



def route_after_chart_advance(
    state: InsightFlowState
) -> Literal["plot_generation", "report"]:

    index = state["current_chart_index"]

    total_charts = len(
        state["visualization_plan"]
    )

    if index < total_charts:
        return "plot_generation"

    return "report"



def route_after_plot_execution(
    state: InsightFlowState
) -> Literal["plot_repair", "advance_chart"]:

    error = state.get("plot_error")

    retry_count = state.get(
        "plot_retry_count",
        0
    )

    if error is None:
        return "advance_chart"

    if retry_count < MAX_PLOT_RETRIES:
        return "plot_repair"

    return "advance_chart"



def route_after_visualization_plan(
    state: InsightFlowState
) -> Literal["plot_generation", "report"]:

    visualization_plan = state.get(
        "visualization_plan",
        []
    )

    if visualization_plan:
        return "plot_generation"

    return "report"