# 负责连接各个 Agent 和工具，定义整个分析工作流。

from langgraph.graph import StateGraph, START, END
from state import InsightFlowState
from nodes import (
    loader_node,
    knowledge_retrieval_node,
    context_node,
    planner_node,
    approval_gate_node,
    report_publish_gate_node,
    router_node,
    execution_node,
    tool_execution_node,
    repair_node,
    critic_node,
    refine_node,
    failed_task_node,
    advance_task_node,
    decide_node,
    runtime_replan_node,
    report_node,
    artifact_node,

    visualization_planner_node,
    plot_generation_node,
    plot_execution_node,
    plot_repair_node,
    advance_chart_node,
    budget_stop_node,
)

from conditions import (
    route_after_execution,
    route_after_critic,
    route_after_advance,
    route_after_approval,
    route_after_report_approval,
    route_to_approval,
    route_to_report_approval,
    route_after_plot_execution,
    route_after_chart_advance,
    route_after_visualization_plan,
    route_after_tool_execution,
    route_after_critic_node,
    route_after_failed_task,
    route_after_decide,
    route_after_runtime_replan,
    route_after_budget_stop,
)


def create_workflow(checkpointer=None):

    graph = StateGraph(
        InsightFlowState
    )
    graph.add_node("budget_stop", budget_stop_node)
    graph.add_conditional_edges("budget_stop", route_after_budget_stop, {"report_publish_gate": "report_publish_gate", "artifact": "artifact"})

    graph.add_node(
        "loader",
        loader_node
    )

    graph.add_node(
        "context",
        context_node
    )

    graph.add_node(
    "knowledge_retrieval",
    knowledge_retrieval_node
    )

    graph.add_node(
        "planner",
        planner_node
    )

    graph.add_node(
        "approval_gate",
        approval_gate_node
    )

    graph.add_node(
        "router",
        router_node
    )

    graph.add_node(
        "execution",
        execution_node
    )

    graph.add_node(
        "tool_execution",
        tool_execution_node
    )

    graph.add_node(
        "repair",
        repair_node
    )

    graph.add_node(
        "critic",
        critic_node
    )

    graph.add_node(
        "refine",
        refine_node
    )

    graph.add_node(
        "failed_task",
        failed_task_node
    )

    graph.add_node(
        "advance_task",
        advance_task_node
    )

    # 观察 -> 决策 -> 校验 -> 执行。关闭开关时不会被路由到。
    graph.add_node(
        "decide",
        decide_node
    )

    graph.add_node(
        "runtime_replan",
        runtime_replan_node
    )

    graph.add_node(
        "report",
        report_node
    )

    graph.add_node(
        "report_publish_gate",
        report_publish_gate_node
    )

    graph.add_node(
    "artifact",
    artifact_node
    )


    graph.add_node(
    "visualization_planner",
    visualization_planner_node
)

    graph.add_node(
        "plot_generation",
        plot_generation_node
    )

    graph.add_node(
        "plot_execution",
        plot_execution_node
    )

    graph.add_node(
        "plot_repair",
        plot_repair_node
    )

    graph.add_node(
        "advance_chart",
        advance_chart_node
    )



    graph.add_edge(
        START,
        "loader"
    )

    graph.add_edge(
        "loader",
        "context"
    )

  
    graph.add_edge(
    "context",
    "knowledge_retrieval"
)

    graph.add_edge(
        "knowledge_retrieval",
        "planner"
    )

    graph.add_conditional_edges(
        "planner",
        route_to_approval,
        {
            "approval_gate": "approval_gate",
            "router": "router",
            "complete": "visualization_planner",
            "budget_stop": "budget_stop",
        }
    )

    # 审批关口在 router 之前：此时尚无任何副作用，中断可安全重放。
    # 拒绝时直接结束，不再生成报告——被拒绝的分析不该产出交付物，
    # 否则"拒绝"只阻止了执行却仍然留下一份报告。
    graph.add_conditional_edges(
        "approval_gate",
        route_after_approval,
        {
            "router": "router",
            "complete": END,
        }
    )

    graph.add_edge(
        "router",
        "execution"
    )

    graph.add_edge(
        "execution",
        "tool_execution"
    )

    # 开启动态循环时，执行结果先交给决策节点；关闭时仍是原来的三条固定分支。
    graph.add_conditional_edges(
        "tool_execution",
        route_after_tool_execution,
        {
            "decide": "decide",
            "repair": "repair",
            "critic": "critic",
            "failed": "failed_task",
            "budget_stop": "budget_stop",
        }
    )

    graph.add_edge(
        "repair",
        "tool_execution"
    )

    graph.add_conditional_edges(
        "critic",
        route_after_critic_node,
        {
            "decide": "decide",
            "advance": "advance_task",
            "refine": "refine",
            "failed": "failed_task",
            "budget_stop": "budget_stop",
        }
    )

    graph.add_edge(
        "refine",
        "tool_execution"
    )

    graph.add_conditional_edges(
        "failed_task",
        route_after_failed_task,
        {
            "decide": "decide",
            "advance": "advance_task",
        }
    )

    graph.add_conditional_edges(
        "decide",
        route_after_decide,
        {
            "advance": "advance_task",
            "complete": "visualization_planner",
            "repair": "repair",
            "refine": "refine",
            "execution": "execution",
            "replan": "runtime_replan",
            "budget_stop": "budget_stop",
        }
    )

    graph.add_conditional_edges(
        "runtime_replan",
        route_after_runtime_replan,
        {
            "router": "router",
            "approval_gate": "approval_gate",
            "complete": "visualization_planner",
            "budget_stop": "budget_stop",
        }
    )

    graph.add_conditional_edges(
        "advance_task",
        route_after_advance,
        {
            "router": "router",
            "complete": "visualization_planner",
        }
    )

    graph.add_conditional_edges(
        "visualization_planner",
        route_after_visualization_plan,
        {
            "plot_generation": "plot_generation",
            "report": "report",
            "budget_stop": "budget_stop",
        }
    )

    graph.add_edge(
        "plot_generation",
        "plot_execution"
    )


    graph.add_conditional_edges(
    "plot_execution",
    route_after_plot_execution,
    {
        "plot_repair": "plot_repair",
        "advance_chart": "advance_chart",
        "budget_stop": "budget_stop",
    }
    )

    graph.add_edge(
    "plot_repair",
    "plot_execution"
    )

    graph.add_conditional_edges(
    "advance_chart",
    route_after_chart_advance,
    {
        "plot_generation": "plot_generation",
        "report": "report",
    }
    )


    graph.add_conditional_edges(
        "report",
        route_to_report_approval,
        {
            "report_publish_gate": "report_publish_gate",
            "artifact": "artifact",
            "budget_stop": "budget_stop",
        }
    )

    # 报告发布关口：报告内容已生成但文件尚未写出，中断可安全重放。
    # 拒绝时不落盘、不给产物，与计划被拒的行为一致。
    graph.add_conditional_edges(
        "report_publish_gate",
        route_after_report_approval,
        {
            "artifact": "artifact",
            "complete": END,
        }
    )

    graph.add_edge(
        "artifact",
        END
    )

    # checkpointer 为 None 时退化为无持久化执行（单次脚本调用）；
    # 传入 SqliteSaver 等实现后，同一 thread_id 可跨进程恢复。
    return graph.compile(checkpointer=checkpointer)
