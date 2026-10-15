# 定义 LangGraph 工作流节点，负责读取和更新共享 State。

import contextvars
import functools
import os
import time
from typing import Any, Callable

from langgraph.types import interrupt

from state import InsightFlowState
from ingestion.resources import (
    build_snapshots,
    snapshots_to_state,
    verify_recorded_sources,
)
from context.context_builder import build_all_contexts
from agents.planner import create_analysis_plan
from agents.router import route_analysis_task
from agents.executor import generate_execution_code
from tools.tool_registry import available_tools, invoke_tool
from agents.repair import repair_execution_code
from agents.critic import review_execution_result
from agents.refine import refine_execution_code
from agents.reporter import generate_analysis_report
from utils.artifacts import save_markdown_report,save_html_report,save_pdf_report
from agents.visualizer import create_visualization_plan
from agents.plotter import generate_plot_code
from agents.plot_repair import repair_plot_code
from pathlib import Path
from core.inference import model_for, resolve_snapshots, resolve_tables
from core.runtime import current_runtime
from utils.idempotency import (
    artifact_operation,
    chart_operation,
    completed_operations,
    plan_approval_id,
    record,
    replay_guard,
    report_approval_id,
    attempt_subject,
    run_agent_call,
)
from utils.result_artifacts import (
    ResultArtifactError,
    build_preview,
    load_result,
    store_result,
    unwrap_records,
)


def _crash_point(name: str) -> None:
    """故障注入点，仅供测试使用。

    通过环境变量 INSIGHTFLOW_CRASH_AT 指定：在文件已经写入、但本节点的
    state 增量尚未提交（Checkpoint 未落盘）的瞬间强杀进程。
    用于验证"文件系统与 SQLite 不属于同一个事务"时会发生什么。
    """
    if os.environ.get("INSIGHTFLOW_CRASH_AT") == name:
        import os as _os

        _os._exit(17)


def _agent_call(
    agent: str,
    subject: str,
    invoke: Callable[[], Any],
    *,
    attempt: int = 1,
) -> tuple[Any, dict, dict]:
    """执行一次 Agent 调用，并取回该次调用的幂等登记与额外 state 增量。

    `subject` 必须已经用 `attempt_subject()` 带上逻辑尝试序号（会重复调用的
    节点），`attempt` 用于账本上的人读标注。两者分开的原因见
    `utils/idempotency.run_agent_call`。

    节点在返回 state 增量时必须一并带上 `completed_operations` 与额外增量
    （例如清空 `pending_llm_calls`），否则这些记录不会进入 Checkpoint：
    "已完成"没落盘会导致恢复时重复执行，
    "未确认清单"没清空会导致每次恢复都重新询问同一件事。
    """
    return run_agent_call(
        _CURRENT_STATE.get() or {},
        node=agent,
        subject=subject,
        call=invoke,
        attempt=attempt,
    )


# 当前节点正在处理的 state。节点函数签名不能带额外参数（LangGraph 会把
# 参数名当成状态通道），因此这里用 ContextVar 传递，避免改动 19 个节点签名。
_CURRENT_STATE: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "insightflow_current_state", default=None
)


def _with_state(func: Callable) -> Callable:
    """把 state 放进 ContextVar，供 `_agent_call` 读取。

    只包装实际调用 Agent 的节点；其余节点不受影响。
    """

    @functools.wraps(func)
    def wrapper(state: InsightFlowState) -> dict:
        token = _CURRENT_STATE.set(state)
        try:
            return func(state)
        finally:
            _CURRENT_STATE.reset(token)

    return wrapper
from utils.run_logger import log_event
from time import perf_counter
from rag.knowledge_service import build_and_retrieve_knowledge

def loader_node(
    state: InsightFlowState
) -> dict:

    # 只把表的身份写进 state（可序列化），真实 DataFrame 由 resolver 按需读回。
    #
    # 先校验已有快照：恢复或重放时若来源已被修改/删除，必须在这里失败。
    # 不能只靠下面的 build_snapshots —— 它会重新计算指纹，
    # 于是"数据变了"会被静默当成一份新快照，历史 Run 就用了新数据。
    recorded = state.get("table_snapshot") or {}
    if recorded:
        verify_recorded_sources(recorded)

    snapshots = build_snapshots(state["data_sources"])
    resolve_snapshots(snapshots_to_state(snapshots))

    return {
        "table_snapshot": snapshots_to_state(snapshots)
    }


def context_node(
    state: InsightFlowState
) -> dict:

    contexts = build_all_contexts(
        resolve_tables(state)
    )

    return {
        "data_context": contexts
    }


def knowledge_retrieval_node(
    state: InsightFlowState
) -> dict:

    knowledge_sources = state.get(
        "knowledge_sources",
        []
    )

    if not knowledge_sources:
        return {
            "retrieved_knowledge": [],
            "retrieved_knowledge_details": []
        }

    analysis_focus = state.get(
        "analysis_focus"
    )

    query = (
        analysis_focus
        or "与当前数据分析相关的业务规则、指标定义和术语"
    )

    retrieved_knowledge_details = (
        build_and_retrieve_knowledge(
            knowledge_sources=knowledge_sources,
            query=query
        )
    )

    retrieved_knowledge = [
        item["content"]
        for item in retrieved_knowledge_details
        if item.get("content")
    ]

    return {
        "retrieved_knowledge": retrieved_knowledge,
        "retrieved_knowledge_details": (retrieved_knowledge_details)
    }


@_with_state
def planner_node(
    state: InsightFlowState
) -> dict:

    retrieved_knowledge = state.get(
        "retrieved_knowledge",
        []
    )

    plan, _ops, _extra = _agent_call(
        "planner",
        "plan",
        lambda: create_analysis_plan(
            contexts=state["data_context"],
            analysis_focus=state.get(
                "analysis_focus"
            ),
            retrieved_knowledge=retrieved_knowledge,
            llm=model_for("planner"),
        ),
    )

    run_id = state.get("run_id")

    if run_id is not None:
        log_event(
            run_id=run_id,
            node="planner",
            event="plan_created",
            status="success",
            details={
                "task_count": len(plan),
                "task_ids": [
                    task["task_id"]
                    for task in plan
                ],
                "retrieved_knowledge_count": len(
                    retrieved_knowledge
                )
            }
        )

    if not plan:
        # 空计划意味着"没有可分析的产出"，不是成功。这里显式标注原因，
        # 否则下游只能看到 failed_tasks == [] 而误判为分析完成。
        return {
            "analysis_plan": [],
            "current_task_index": 0,
            "analysis_status": "no_tasks",
            "status_reason": "Planner 未产出任何分析任务",
            "completed_operations": _ops,
            **_extra,
        }

    return {
        "analysis_plan": plan,
        "current_task_index": 0,
        "completed_operations": _ops,
        **_extra,
    }



def router_node(
    state: InsightFlowState
) -> dict:

    index = state["current_task_index"]

    task = state["analysis_plan"][index]

    tool = route_analysis_task(
        task,
        resolve_tables(state)
    )
    run_id = state.get("run_id")

    if run_id is not None:
        log_event(
            run_id=run_id,
            node="router",
            event="route_selected",
            status="success",
            details={
                "task_id": task["task_id"],
                "route": tool,
                "target_tables": task[
                    "target_tables"
                ]
            }
        )

    return {
        "current_tool": tool
    }


@_with_state
def execution_node(
    state: InsightFlowState
) -> dict:
    started_at = perf_counter()
    index = state["current_task_index"]

    task = state["analysis_plan"][index]

    data_context = {
        table_name: state["data_context"][table_name]
        for table_name in task["target_tables"]
    }

    tool = state["current_tool"]

    if tool is None:
        raise ValueError(
            "Current tool is missing."
        )

    retrieved_knowledge = state.get(
    "retrieved_knowledge",
    []
    )

    # execution_node 每个任务只进入一次：之后的代码由 repair / refine 产出
    # （图上是 repair→tool_execution、refine→tool_execution，不再回到这里）。
    # 因此这里保持 task_id 作为身份就足够，不需要尝试序号。
    code, _ops, _extra = _agent_call(
        "executor",
        task["task_id"],
        lambda: generate_execution_code(
            task=task,
            data_context=data_context,
            route=tool,
            retrieved_knowledge=retrieved_knowledge,
            llm=model_for("executor"),
        ),
    )

    duration_ms = (
    perf_counter() - started_at
    ) * 1000

    run_id = state.get("run_id")

    if run_id is not None:
        log_event(
            run_id=run_id,
            node="execution_generation",
            event="analysis_code_generated",
            status="success",
            details={
                "task_id": task["task_id"],
                "tool": tool,
                "target_tables": task[
                    "target_tables"
                ]
            },
            duration_ms=duration_ms
        )

    return {
        "current_code": code,
        "completed_operations": _ops,
        **_extra,
    }



def tool_execution_node(
    state: InsightFlowState
) -> dict:

    index = state["current_task_index"]
    task = state["analysis_plan"][index]

    tool = state["current_tool"]
    code = state["current_code"]

    if tool is None:
        raise ValueError("Current tool is missing.")

    if code is None:
        raise ValueError("Current code is missing.")

    if tool not in available_tools():
        raise ValueError(
            f"Unsupported tool: {tool}"
        )

    # 参数校验失败与执行失败都从 invoke_tool 以错误字典返回，
    # 交给 route_after_execution 走修复循环，而不是在这里抛异常终止工作流。
    arguments = (
        {"table_names": task["target_tables"], "code": code}
        if tool == "python"
        else {"table_names": task["target_tables"], "query": code}
    )

    execution = invoke_tool(tool, resolve_tables(state), arguments)

    run_id = state.get("run_id")

    # 结果可能含 DataFrame / ndarray，直接放进 state 会让 Checkpoint 序列化失败。
    # 小结果内联，大结果落成带指纹的产物，state 只保留摘要与引用。
    inline_result, result_artifact = (None, None)
    if execution["success"]:
        inline_result, result_artifact = store_result(
            execution["result"],
            run_id=run_id or "no-run",
            task_id=task["task_id"],
            output_root=current_runtime().output_root,
        )

    execution_result = {
        "task_id": task["task_id"],
        "tool_used": tool,
        "code": code,
        "success": execution["success"],
        "result": inline_result,
        "result_artifact": result_artifact,
        "error": execution["error"],
        # 记录产生这条结果的尝试序号，供下面的去重与审计使用
        "retry_count": state.get("retry_count", 0),
        "refine_count": state.get("refine_count", 0),
    }

    if run_id is not None:
        log_event(
            run_id=run_id,
            node="tool_execution",
            event="execution_completed",
            status=(
                "success"
                if execution["success"]
                else "failed"
            ),
            details={
                "task_id": task["task_id"],
                "tool": tool,
                "target_tables": task[
                    "target_tables"
                ],
                "retry_count": state.get(
                    "retry_count",
                    0
                ),
                "refine_count": state.get(
                    "refine_count",
                    0
                ),
                "error": execution["error"]
            }
        )

    previous_results = state.get(
        "execution_results",
        []
    )

    # 与绘图结果同样的道理：execution_results 是追加列表，重放该节点会重复追加。
    # 按 (task_id, retry_count, refine_count) 去重——同一次尝试的重放应替换，
    # 而修复循环里的新尝试（计数变化）会作为新记录保留。
    attempt_key = (
        task["task_id"],
        state.get("retry_count", 0),
        state.get("refine_count", 0),
    )

    deduped = [
        existing
        for existing in previous_results
        if (
            existing.get("task_id"),
            existing.get("retry_count", 0),
            existing.get("refine_count", 0),
        )
        != attempt_key
    ]

    return {
        "execution_results": deduped + [
            execution_result
        ],
        "execution_error": execution["error"],
        # 分类随错误一起进 state：路由据此判断是否值得进入修复循环，
        # 而不是去解析错误文案。
        "execution_error_kind": execution.get("error_kind"),
    }


@_with_state
def repair_node(
    state: InsightFlowState
) -> dict:

    retrieved_knowledge = state.get(
    "retrieved_knowledge",
    []
)

    index = state["current_task_index"]
    task = state["analysis_plan"][index]

    data_context = {
        table_name: state["data_context"][table_name]
        for table_name in task["target_tables"]
    }

    tool = state["current_tool"]
    code = state["current_code"]
    error = state["execution_error"]

    if tool is None:
        raise ValueError("Current tool is missing.")

    if code is None:
        raise ValueError("Current code is missing.")

    if error is None:
        raise ValueError("Execution error is missing.")

    # 本次修复的身份由**入站** retry_count 决定（首轮修复为 0）。
    # 下面的 retry_count+1 是已修复几次的记账，不能拿来做操作 ID：
    # 同一轮修复被 Checkpoint 重放时，入站值相同才算同一次逻辑调用。
    repair_attempt = state.get("retry_count", 0)

    repaired_code, _ops, _extra = _agent_call(
        "repair",
        attempt_subject(task["task_id"], repair_attempt),
        lambda: repair_execution_code(
            task=task,
            data_context=data_context,
            route=tool,
            code=code,
            error=error,
            retrieved_knowledge=retrieved_knowledge,
            llm=model_for("repair"),
        ),
        attempt=repair_attempt + 1,
    )

    retry_count = state.get(
        "retry_count",
        0
    ) + 1

    run_id = state.get("run_id")

    if run_id is not None:
        log_event(
            run_id=run_id,
            node="repair",
            event="repair_generated",
            status="success",
            details={
                "task_id": task["task_id"],
                "tool": tool,
                "retry_count": retry_count,
                "previous_error": error
            }
        )

    return {
        "current_code": repaired_code,
        "retry_count": retry_count,
        "completed_operations": _ops,
        **_extra,
    }




def get_final_execution_results(
    execution_results: list[dict]
) -> list[dict]:

    final_results = {}

    for item in execution_results:
        final_results[item["task_id"]] = item

    return list(final_results.values())

    

def get_final_visualization_results(
    visualization_results: list[dict]
) -> list[dict]:

    final_results = {}

    for item in visualization_results:
        final_results[item["chart_id"]] = item

    return list(final_results.values())


def build_report_execution_previews(
    execution_results: list[dict],
    *,
    run_id: str | None,
    output_root: Path,
) -> list[dict]:
    """把执行结果转成报告 Agent 看得懂的**有界预览**。

    报告只由 LLM 撰写，它需要的是"列名、行数、几条样本"，
    而不是两万行记录。大结果在 state 里本来就只有摘要，这里把摘要与
    （有界的）样本一起给它，并明确标注这是预览——
    既避免把整份数据塞进 Prompt，也不让它误以为看到了全部数据。
    """
    previews: list[dict] = []

    for item in execution_results:
        preview = {
            "task_id": item.get("task_id"),
            "tool_used": item.get("tool_used"),
            "success": item.get("success"),
            "error": item.get("error"),
            "attempt": {
                "retry_count": item.get("retry_count"),
                "refine_count": item.get("refine_count"),
            },
        }

        if item.get("success"):
            preview["result"] = build_preview(
                item.get("result"),
                item.get("result_artifact"),
                run_id=run_id,
                output_root=output_root,
            )
        else:
            preview["result"] = None

        previews.append(preview)

    return previews

    

def advance_task_node(
    state: InsightFlowState
) -> dict:

    current_index = state[
        "current_task_index"
    ]

    current_task = state[
        "analysis_plan"
    ][current_index]

    if state.get(
        "validation_passed",
        False
    ):
        run_id = state.get("run_id")

        if run_id is not None:
            log_event(
                run_id=run_id,
                node="task",
                event="task_completed",
                status="success",
                details={
                    "task_id": current_task[
                        "task_id"
                    ],
                    "retry_count": state.get(
                        "retry_count",
                        0
                    ),
                    "refine_count": state.get(
                        "refine_count",
                        0
                    )
                }
            )

    next_index = current_index + 1

    return {
        "current_task_index": next_index,
        "current_tool": None,
        "current_code": None,
        "execution_error": None,
        "execution_error_kind": None,
        "retry_count": 0,
        "refine_count": 0,
        "critic_feedback": None,
        "validation_passed": False
    }



@_with_state
def critic_node(
    state: InsightFlowState
) -> dict:
    retrieved_knowledge = state.get(
    "retrieved_knowledge",
    []
)
    index = state["current_task_index"]
    task = state["analysis_plan"][index]

    data_context = {
        table_name: state["data_context"][table_name]
        for table_name in task["target_tables"]
    }

    execution_result = state["execution_results"][-1]

    # 每轮精修后都会重新评审同一个任务，因此评审身份必须带 refine_count：
    # 否则第二轮评审会复用第一轮的结论，精修循环等于没有校验。
    review_round = state.get("refine_count", 0)

    review, _ops, _extra = _agent_call(
        "critic",
        attempt_subject(task["task_id"], review_round),
        lambda: review_execution_result(
            task=task,
            data_context=data_context,
            execution_result=execution_result,
            retrieved_knowledge=retrieved_knowledge,
            llm=model_for("critic"),
        ),
        attempt=review_round + 1,
    )

    run_id = state.get("run_id")

    if run_id is not None:
        log_event(
            run_id=run_id,
            node="critic",
            event="validation_completed",
            status=(
                "passed"
                if review["validation_passed"]
                else "failed"
            ),
            details={
                "task_id": task["task_id"],
                "validation_passed": review[
                    "validation_passed"
                ],
                "refine_count": state.get(
                    "refine_count",
                    0
                ),
                "feedback": review[
                    "feedback"
                ]
            }
        )

    return {
        "validation_passed": review["validation_passed"],
        "critic_feedback": review["feedback"],
        "completed_operations": _ops,
        **_extra,
    }



@_with_state
def refine_node(
    state: InsightFlowState
) -> dict:
    
    retrieved_knowledge = state.get(
    "retrieved_knowledge",
    []
)
    index = state["current_task_index"]

    task = state["analysis_plan"][index]

    data_context = {
        table_name: state["data_context"][table_name]
        for table_name in task["target_tables"]
    }

    tool = state["current_tool"]

    code = state["current_code"]

    execution_result = state[
        "execution_results"
    ][-1]

    critic_feedback = state[
        "critic_feedback"
    ]

    if tool is None:
        raise ValueError(
            "Current tool is missing."
        )

    if code is None:
        raise ValueError(
            "Current code is missing."
        )

    if critic_feedback is None:
        raise ValueError(
            "Critic feedback is missing."
        )

    refine_attempt = state.get("refine_count", 0)

    refined_code, _ops, _extra = _agent_call(
        "refine",
        attempt_subject(task["task_id"], refine_attempt),
        lambda: refine_execution_code(
            task=task,
            data_context=data_context,
            route=tool,
            code=code,
            execution_result=execution_result,
            critic_feedback=critic_feedback,
            retrieved_knowledge=retrieved_knowledge,
            llm=model_for("refine"),
        ),
        attempt=refine_attempt + 1,
    )

    refine_count = state.get(
        "refine_count",
        0
    ) + 1

    run_id = state.get("run_id")

    if run_id is not None:
        log_event(
            run_id=run_id,
            node="refine",
            event="refinement_generated",
            status="success",
            details={
                "task_id": task["task_id"],
                "tool": tool,
                "refine_count": refine_count,
                "critic_feedback": critic_feedback
            }
        )

    return {
        "current_code": refined_code,
        "refine_count": refine_count,
        "completed_operations": _ops,
        **_extra,
    }



def failed_task_node(
    state: InsightFlowState
) -> dict:

    index = state["current_task_index"]
    task = state["analysis_plan"][index]

    execution_error = state.get("execution_error")
    critic_feedback = state.get("critic_feedback")

    if execution_error is not None:
        stage = "execution"
        reason = execution_error

    elif critic_feedback is not None:
        stage = "validation"
        reason = critic_feedback

    else:
        raise ValueError(
            "Task failure reason is missing."
        )

    failure = {
        "task_id": task["task_id"],
        "stage": stage,
        "reason": reason,
        # 保留分类，便于事后区分"代码错了"与"沙箱没起来"
        "error_kind": state.get("execution_error_kind") if stage == "execution" else None,
        "code": state.get("current_code"),
        "retry_count": state.get(
            "retry_count",
            0
        ),
        "refine_count": state.get(
            "refine_count",
            0
        )
    }

    run_id = state.get("run_id")

    if run_id is not None:
        log_event(
            run_id=run_id,
            node="task",
            event="task_failed",
            status="failed",
            details={
                "task_id": task["task_id"],
                "stage": stage,
                "reason": reason,
                "error_kind": failure["error_kind"],
                "retry_count": state.get(
                    "retry_count",
                    0
                ),
                "refine_count": state.get(
                    "refine_count",
                    0
                )
            }
        )


    previous_failures = state.get(
        "failed_tasks",
        []
    )

    return {
        "failed_tasks": previous_failures + [
            failure
        ]
    }

@_with_state
def report_node(
    state: InsightFlowState
) -> dict:

    final_execution_results = (
        get_final_execution_results(
            state.get(
                "execution_results",
                []
            )
        )
    )

    final_visualization_results = (
        get_final_visualization_results(
            state.get(
                "visualization_results",
                []
            )
        )
    )

    visualization_plan_by_id = {
        item["chart_id"]: item
        for item in state.get(
            "visualization_plan",
            []
        )
    }

    report_visualizations = []

    for item in final_visualization_results:

        result_item = dict(item)

        visualization_task = (
            visualization_plan_by_id.get(
                item["chart_id"]
            )
        )

        if visualization_task is not None:
            result_item["description"] = (
                visualization_task["description"]
            )

            result_item["chart_type"] = (
                visualization_task["chart_type"]
            )

            result_item["target_tables"] = (
                visualization_task["target_tables"]
            )

            result_item["expected_output"] = (
                visualization_task["expected_output"]
            )

        if result_item.get("path"):
            result_item["markdown_path"] = Path(
                result_item["path"]
            ).name

        report_visualizations.append(
            result_item
        )

    report, _ops, _extra = _agent_call(
        "reporter",
        "report",
        lambda: generate_analysis_report(
            analysis_plan=state["analysis_plan"],
            # 传给报告 Agent 的是**有界预览**而不是完整结果：
            # 完整结果可能有两万行，塞进 Prompt 既超预算也误导模型。
            execution_results=build_report_execution_previews(
                final_execution_results,
                run_id=state.get("run_id"),
                output_root=current_runtime().output_root,
            ),
            failed_tasks=state.get(
                "failed_tasks",
                []
            ),
            visualization_plan=state.get(
                "visualization_plan",
                []
            ),
            visualization_results=report_visualizations,
            llm=model_for("reporter"),
        ),
    )

    run_id = state.get("run_id")

    if run_id is not None:
        log_event(
            run_id=run_id,
            node="report",
            event="report_generated",
            status="success",
            details={
                "analysis_task_count": len(
                    state["analysis_plan"]
                ),
                "execution_result_count": len(
                    final_execution_results
                ),
                "failed_task_count": len(
                    state.get(
                        "failed_tasks",
                        []
                    )
                ),
                "visualization_count": len(
                    report_visualizations
                )
            }
        )

    return {
        "report_content": report,
        "completed_operations": _ops,
        **_extra,
    }


def artifact_node(
    state: InsightFlowState
) -> dict:

    report_content = state[
        "report_content"
    ]

    if report_content is None:
        raise ValueError(
            "Report content is missing."
        )

    run_id = state.get(
        "run_id"
    )

    if run_id is None:
        raise ValueError(
            "Run ID is missing."
        )

    # 写文件是副作用：恢复时若该节点被重放，直接复用已登记的产物，
    # 不重新生成 Markdown/HTML/PDF（PDF 需要启动浏览器，代价明显）。
    op_id = artifact_operation(run_id)
    done = replay_guard(state, op_id, what="报告产物生成")

    if done is not None and done.detail:
        return {
            "report_path": done.detail.get("markdown_path"),
            "analysis_status": done.detail.get("analysis_status"),
            "status_reason": done.detail.get("status_reason"),
            "run_status": done.detail.get("run_status"),
            "artifacts": done.detail.get("artifacts", state.get("artifacts", [])),
            "completed_operations": completed_operations(state),
        }

    output_dir = (
        current_runtime().output_root
        / run_id
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    markdown_path = save_markdown_report(
        report_content,
        output_path=str(
            output_dir
            / "analysis_report.md"
        )
    )

    html_path = save_html_report(
        report_content,
        output_path=str(
            output_dir
            / "analysis_report.html"
        )
    )

    pdf_path = save_pdf_report(
        html_path,
        output_path=str(
            output_dir
            / "analysis_report.pdf"
        )
    )

    # 文件已全部落盘，但"已完成"登记还没提交到 Checkpoint。
    # 这一刻崩溃即可验证文件与 Checkpoint 的一致性边界。
    _crash_point("artifact_files_written")

    log_event(
        run_id=run_id,
        node="artifact",
        event="report_artifacts_saved",
        status="success",
        details={
            "markdown_path": markdown_path,
            "html_path": html_path,
            "pdf_path": pdf_path
        }
    )

    previous_artifacts = state.get(
        "artifacts",
        []
    )

    # 收尾时确定结果语义。空计划不算成功，因此 planner 标注的 no_tasks 需要保留；
    # 有计划但有任务失败则记为 partial。
    plan = state.get("analysis_plan", [])

    if not plan:
        analysis_status = "no_tasks"
        status_reason = state.get("status_reason") or "Planner 未产出任何分析任务"
    elif state.get("failed_tasks"):
        failed_count = len(state["failed_tasks"])
        analysis_status = "partial"
        status_reason = f"{failed_count} 个分析任务最终失败"
    else:
        analysis_status = "completed"
        status_reason = None

    log_event(
        run_id=run_id,
        node="artifact",
        event="analysis_finished",
        status=analysis_status,
        details={
            "tasks_planned": len(plan),
            "tasks_failed": len(state.get("failed_tasks", [])),
            "reason": status_reason
        }
    )

    artifacts = (
        previous_artifacts
        + [
            markdown_path,
            html_path,
            pdf_path
        ]
    )

    # 生命周期与产物登记一起写入同一个 state 增量，保证"完成"与"已登记"一致
    run_status = "completed" if analysis_status in ("completed", "partial") else "failed"

    return {
        "report_path": markdown_path,
        "analysis_status": analysis_status,
        "status_reason": status_reason,
        "run_status": run_status,
        "artifacts": artifacts,
        "completed_operations": record(
            state,
            op_id,
            detail={
                "markdown_path": markdown_path,
                "html_path": html_path,
                "pdf_path": pdf_path,
                "analysis_status": analysis_status,
                "status_reason": status_reason,
                "run_status": run_status,
                "artifacts": artifacts,
            },
        ),
    }


def approval_gate_node(
    state: InsightFlowState
) -> dict:
    """执行前的人工审批关口。

    放在 planner 之后、router 之前：此时还没有任何副作用（未执行工具、
    未写产物），因此中断是安全可重放的。interrupt() 之前不做任何不可重放的
    动作，恢复时这个节点会从头执行，只是这次能拿到审批结果。

    非只读 SQL 不在这里放行：只读约束由 utils/sql_guard.py 独立强制，
    审批只回答"要不要做这项分析"。
    """
    run_id = state.get("run_id") or "unknown"

    plan = state.get("analysis_plan", [])
    sql_task_ids = [
        task["task_id"]
        for task in plan
        if task.get("target_tables") and _task_is_sql(state, task)
    ]

    approval_id = plan_approval_id(run_id)

    # 已审批过（例如恢复后重放本节点）时不再中断
    existing = _find_approval(state, approval_id)
    if existing is not None:
        return {
            "plan_approval": existing,
            "run_status": "running",
        }

    request = {
        "approval_id": approval_id,
        "run_id": run_id,
        "node": "approval_gate",
        "kind": "plan_review",
        "summary": {
            "task_count": len(plan),
            "tasks": [
                {
                    "task_id": task["task_id"],
                    "description": task["description"],
                    "target_tables": task["target_tables"],
                }
                for task in plan
            ],
            # 明确告知：含 SQL 的任务同样只读，审批不会提升权限
            "contains_sql_tasks": bool(sql_task_ids),
            "sql_read_only_enforced": True,
        },
        "created_at": time.time(),
    }

    log_event(
        run_id=run_id,
        node="approval_gate",
        event="approval_requested",
        status="waiting_approval",
        details={"approval_id": approval_id, "kind": "plan_review"},
    )

    # 真正的中断：resume 时 interrupt() 返回 Command(resume=...) 携带的值
    decision = interrupt(request)

    return _apply_decision(state, request, decision)


def _task_is_sql(
    state: InsightFlowState, task: dict
) -> bool:
    """判断任务是否走 SQL。

    只看快照里的资源类型，不读真实数据：审批摘要不需要 DataFrame，
    而按来源类型判定与确定性路由的结论一致（DataFrame→python，SQLite→sql）。
    """
    snapshots = state.get("table_snapshot") or {}

    source_types = [
        snapshots.get(name, {}).get("source_type")
        for name in task.get("target_tables", [])
    ]

    return bool(source_types) and all(
        source_type == "sqlite" for source_type in source_types
    )


def _find_approval(
    state: InsightFlowState, approval_id: str
) -> dict | None:
    for item in state.get("approvals") or []:
        if item.get("approval_id") == approval_id:
            return item
    return None


def _apply_decision(
    state: InsightFlowState, request: dict, decision: Any
) -> dict:
    """把审批结果归档，并给出后续路由依据。

    重复提交同一个审批 ID 时直接复用已归档的决定，不再触发第二次中断，
    避免同一次操作被批准两次。
    """
    payload = decision if isinstance(decision, dict) else {}
    verdict = str(payload.get("decision", "")).lower()

    if verdict not in ("approve", "reject"):
        raise ValueError(
            f"审批决定必须是 approve 或 reject，收到：{payload.get('decision')!r}"
        )

    record_entry = {
        "approval_id": request["approval_id"],
        "run_id": request["run_id"],
        "node": request["node"],
        "kind": request["kind"],
        "decision": verdict,
        "comment": payload.get("comment"),
        "decided_at": time.time(),
    }

    log_event(
        run_id=request["run_id"],
        node="approval_gate",
        event="approval_decided",
        status=verdict,
        details=record_entry,
    )

    approvals = list(state.get("approvals") or [])
    approvals.append(record_entry)

    return {
        "plan_approval": record_entry,
        "approvals": approvals,
        "pending_approval": None,
        "run_status": "completed" if verdict == "approve" else "failed",
        "analysis_status": None if verdict == "approve" else "failed",
        "status_reason": None if verdict == "approve" else "分析计划未获批准",
    }


def report_publish_gate_node(
    state: InsightFlowState
) -> dict:
    """报告发布前的人工确认。

    放在 artifact 之前：报告文件尚未写出，中断恢复不会产生半成品文件。
    """
    run_id = state.get("run_id") or "unknown"
    approval_id = report_approval_id(run_id)

    existing = _find_approval(state, approval_id)
    if existing is not None:
        return {"report_approval": existing, "run_status": "running"}

    request = {
        "approval_id": approval_id,
        "run_id": run_id,
        "node": "report_publish_gate",
        "kind": "report_publish",
        "summary": {
            "report_chars": len(state.get("report_content") or ""),
            "analysis_status": state.get("analysis_status"),
            "failed_tasks": len(state.get("failed_tasks") or []),
        },
        "created_at": time.time(),
    }

    log_event(
        run_id=run_id,
        node="report_publish_gate",
        event="approval_requested",
        status="waiting_approval",
        details={"approval_id": approval_id, "kind": "report_publish"},
    )

    decision = interrupt(request)
    return _apply_decision(state, request, decision)


@_with_state
def visualization_planner_node(
    state: InsightFlowState
) -> dict:

    final_execution_results = (
        get_final_execution_results(
            state.get(
                "execution_results",
                []
            )
        )
    )

    visualization_plan, _ops, _extra = _agent_call(
        "visualizer",
        "plan",
        lambda: create_visualization_plan(
            analysis_plan=state["analysis_plan"],
            execution_results=final_execution_results,
            failed_tasks=state.get(
                "failed_tasks",
                []
            ),
            llm=model_for("visualizer"),
        ),
    )

    run_id = state.get("run_id")

    if run_id is not None:
        log_event(
            run_id=run_id,
            node="visualization_planner",
            event="visualization_plan_created",
            status="success",
            details={
                "chart_count": len(
                    visualization_plan
                ),
                "charts": [
                    {
                        "chart_id": item["chart_id"],
                        "source_task_id": item[
                            "source_task_id"
                        ],
                        "chart_type": item[
                            "chart_type"
                        ]
                    }
                    for item in visualization_plan
                ]
            }
        )

    return {
        "visualization_plan": visualization_plan,
        "current_chart_index": 0,
        "completed_operations": _ops,
        **_extra,
    }


@_with_state
def plot_generation_node(
    state: InsightFlowState
) -> dict:

    index = state["current_chart_index"]

    visualization_task = state[
        "visualization_plan"
    ][index]

    target_tables = visualization_task[
        "target_tables"
    ]

    data_context = {
        table_name: state[
            "data_context"
        ][table_name]
        for table_name in target_tables
    }

    source_task_id = visualization_task[
        "source_task_id"
    ]

    chart_id = visualization_task[
        "chart_id"
    ]

    source_execution_result = next(
        (
            result
            for result in reversed(
                state["execution_results"]
            )
            if (
                result["task_id"]
                == source_task_id
                and result["success"]
            )
        ),
        None
    )

    if source_execution_result is None:
        raise ValueError(
            f"No successful execution result found "
            f"for source task: {source_task_id}"
        )

    generation, _ops, _extra = _agent_call(
        "plotter",
        attempt_subject(chart_id, index),
        lambda: generate_plot_code(
            visualization_task=visualization_task,
            data_context=data_context,
            execution_result=source_execution_result,
            llm=model_for("plotter"),
        ),
    )

    code = generation["code"]

    run_id = state.get("run_id")

    if run_id is not None:
        log_event(
            run_id=run_id,
            node="plot_generation",
            event="plot_code_generated",
            status="success",
            details={
                "chart_id": visualization_task[
                    "chart_id"
                ],
                "source_task_id": source_task_id,
                "chart_type": visualization_task[
                    "chart_type"
                ],
                "target_tables": target_tables,
                "generation_attempts": generation[
                    "attempts"
                ],
                "generation_retry_errors": generation[
                    "retry_errors"
                ]
            }
        )

    return {
        "current_plot_code": code,
        "plot_error": None,
        "completed_operations": _ops,
        **_extra,
    }





def plot_execution_node(
    state: InsightFlowState
) -> dict:

    index = state["current_chart_index"]

    visualization_task = state[
        "visualization_plan"
    ][index]

    code = state["current_plot_code"]

    if code is None:
        raise ValueError(
            "Current plot code is missing."
        )

    target_tables = visualization_task[
        "target_tables"
    ]

    chart_id = visualization_task[
        "chart_id"
    ]

    source_task_id = visualization_task[
        "source_task_id"
    ]

    source_execution_result = next(
        (
            result
            for result in reversed(
                state["execution_results"]
            )
            if (
                result["task_id"]
                == source_task_id
                and result["success"]
            )
        ),
        None
    )

    if source_execution_result is None:
        raise ValueError(
            f"No successful execution result found "
            f"for source task: {source_task_id}"
        )

    run_id = state.get("run_id")

    if run_id is None:
        raise ValueError(
            "Run ID is missing."
        )

    # 绘图有副作用（写 PNG），而 visualization_results 是按 chart_id 追加的列表。
    # 重放该节点会重复追加，因此这里做两件事：
    #   1) 用 chart 级别的幂等键跳过已完成的执行；
    #   2) 写入时按 chart_id 去重，使重放得到与首次一致的列表。
    # 修复循环里的多次尝试保留最后一条（last-wins），不产生重复的图表记录。
    run_id = state.get("run_id") or "unknown"
    op_id = chart_operation(run_id, chart_id)
    done = replay_guard(state, op_id, what=f"图表 {chart_id} 的生成")

    if done is not None and done.detail:
        return {
            "visualization_results": done.detail.get(
                "visualization_results", state.get("visualization_results", [])
            ),
            "plot_error": done.detail.get("plot_error"),
            "completed_operations": completed_operations(state),
        }

    # 绘图走与 Python 分析同一个执行后端：一次性 Docker 容器。
    # 产物路径由工具内部按运行期 output_root 决定，不再在这里拼路径。
    # 绘图需要**真实数据**，而大结果在 state 里只留了摘要。
    # 这里按产物引用把完整结果读回来，并校验归属、路径与 sha256：
    # 直接把 `{"__kind__": "externalized"}` 摘要送进容器会画出一张空图，
    # 而工作流不会报错——这正是 P2-B3 要修掉的静默错误。
    # 读回的是 JSON 记录，仍不把完整内容写回 state。
    try:
        analysis_result = load_result(
            source_execution_result.get("result"),
            source_execution_result.get("result_artifact"),
            run_id=state.get("run_id"),
            output_root=current_runtime().output_root,
        )
    except ResultArtifactError as error:
        return {
            "plot_error": f"绘图所需的分析结果不可用：{error}",
            "visualization_results": state.get("visualization_results", []),
            "completed_operations": completed_operations(state),
        }

    # 绘图的 result_df 需要记录序列，而不是带 __kind__ 的信封。
    execution = invoke_tool("plot", resolve_tables(state), {
        "table_names": target_tables,
        "chart_id": chart_id,
        "code": code,
        "analysis_result": unwrap_records(analysis_result),
    })

    visualization_result = {
        "chart_id": chart_id,
        "code": code,
        "success": execution["success"],
        "path": execution["result"],
        "error": execution["error"]
    }

    if run_id is not None:
        log_event(
            run_id=run_id,
            node="plot_execution",
            event="plot_execution_completed",
            status=(
                "success"
                if execution["success"]
                else "failed"
            ),
            details={
                "chart_id": chart_id,
                "source_task_id": source_task_id,
                "chart_type": visualization_task[
                    "chart_type"
                ],
                "plot_retry_count": state.get(
                    "plot_retry_count",
                    0
                ),
                "path": execution["result"],
                "error": execution["error"]
            }
        )

    previous_results = [
        result
        for result in state.get("visualization_results", [])
        if result.get("chart_id") != chart_id
    ]

    updates = {
        "visualization_results": (
            previous_results
            + [visualization_result]
        ),
        "plot_error": execution["error"]
    }

    if execution["success"]:

        previous_artifacts = state.get(
            "artifacts",
            []
        )

        updates["artifacts"] = (
            previous_artifacts
            + [execution["result"]]
        )

    # PNG 已写入，但登记尚未提交到 Checkpoint
    _crash_point("plot_file_written")

    # 与副作用同一次写入登记，保证"已生成"与"已登记"不会分叉
    updates["completed_operations"] = record(
        state,
        op_id,
        detail={
            "visualization_results": updates["visualization_results"],
            "plot_error": execution["error"],
        },
    )

    return updates


def advance_chart_node(
    state: InsightFlowState
) -> dict:

    current_index = state[
        "current_chart_index"
    ]

    visualization_task = state[
        "visualization_plan"
    ][current_index]

    chart_id = visualization_task[
        "chart_id"
    ]

    latest_result = next(
        (
            result
            for result in reversed(
                state.get(
                    "visualization_results",
                    []
                )
            )
            if result["chart_id"] == chart_id
        ),
        None
    )

    run_id = state.get("run_id")

    if (
        run_id is not None
        and latest_result is not None
    ):
        if latest_result["success"]:
            log_event(
                run_id=run_id,
                node="chart",
                event="chart_completed",
                status="success",
                details={
                    "chart_id": chart_id,
                    "source_task_id": (
                        visualization_task[
                            "source_task_id"
                        ]
                    ),
                    "plot_retry_count": state.get(
                        "plot_retry_count",
                        0
                    ),
                    "path": latest_result[
                        "path"
                    ]
                }
            )

        else:
            log_event(
                run_id=run_id,
                node="chart",
                event="chart_failed",
                status="failed",
                details={
                    "chart_id": chart_id,
                    "source_task_id": (
                        visualization_task[
                            "source_task_id"
                        ]
                    ),
                    "plot_retry_count": state.get(
                        "plot_retry_count",
                        0
                    ),
                    "error": latest_result[
                        "error"
                    ]
                }
            )

    next_index = current_index + 1

    return {
        "current_chart_index": next_index,
        "current_plot_code": None,
        "plot_error": None,
        "plot_retry_count": 0
    }


@_with_state
def plot_repair_node(
    state: InsightFlowState
) -> dict:

    index = state["current_chart_index"]

    visualization_task = state[
        "visualization_plan"
    ][index]

    target_tables = visualization_task[
        "target_tables"
    ]

    data_context = {
        table_name: state[
            "data_context"
        ][table_name]
        for table_name in target_tables
    }

    source_task_id = visualization_task[
        "source_task_id"
    ]

    source_execution_result = next(
        (
            result
            for result in reversed(
                state["execution_results"]
            )
            if (
                result["task_id"]
                == source_task_id
                and result["success"]
            )
        ),
        None
    )

    if source_execution_result is None:
        raise ValueError(
            f"No successful execution result found "
            f"for source task: {source_task_id}"
        )

    code = state["current_plot_code"]

    error = state["plot_error"]

    if code is None:
        raise ValueError(
            "Current plot code is missing."
        )

    if error is None:
        raise ValueError(
            "Plot error is missing."
        )

    chart_id = visualization_task[
        "chart_id"
    ]

    # 与 repair 同理：本次绘图修复的身份由入站 plot_retry_count 决定。
    plot_repair_attempt = state.get("plot_retry_count", 0)

    repaired_code, _ops, _extra = _agent_call(
        "plot_repair",
        attempt_subject(chart_id, plot_repair_attempt),
        lambda: repair_plot_code(
            visualization_task=visualization_task,
            data_context=data_context,
            execution_result=source_execution_result,
            code=code,
            error=error,
            llm=model_for("plot_repair"),
        ),
        attempt=plot_repair_attempt + 1,
    )

    plot_retry_count = state.get(
        "plot_retry_count",
        0
    ) + 1

    run_id = state.get("run_id")

    if run_id is not None:
        log_event(
            run_id=run_id,
            node="plot_repair",
            event="plot_repair_generated",
            status="success",
            details={
                "chart_id": visualization_task[
                    "chart_id"
                ],
                "source_task_id": source_task_id,
                "plot_retry_count": plot_retry_count,
                "previous_error": error
            }
        )

    return {
        "current_plot_code": repaired_code,
        "plot_retry_count": plot_retry_count,
        "completed_operations": _ops,
        **_extra,
    }
