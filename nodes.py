# 定义 LangGraph 工作流节点，负责读取和更新共享 State。

import functools
import os
import time
from typing import Any, Callable

# `BudgetExceeded` 必须在模块级导入：`_with_state` 包装器在**导入期**就把它
# 写进 except 子句（它在第 208 行定义，早于文件下半部分那组 utils 导入），
# 因此不能依赖后面才出现的导入。
from utils.agent_loop import AgentBudget, BudgetExceeded, usage_of

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
from tools.tool_registry import available_tools
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


def _with_state(func: Callable) -> Callable:
    """把 state 绑进 Harness 的运行上下文，并统一收敛预算耗尽。

    只包装实际调用 Agent 的节点；其余节点不受影响。

    `BudgetExceeded` 在这里转成"环境类失败"（归类为 `unavailable`，不进入
    Repair）：预算判定接在真实请求之前，因此它可能从任意一个 Agent 调用点
    抛出，逐处 catch 既啰嗦又容易漏，漏掉就意味着预算耗尽变成未分类异常、
    整个 Run 崩掉而不是安全终止。
    """

    @functools.wraps(func)
    def wrapper(state: InsightFlowState) -> dict:
        with bind_state(state):
            try:
                return func(state)
            except BudgetExceeded as exceeded:
                return _budget_failure(state, func.__name__, exceeded)

    return wrapper


def _budget_failure(
    state: InsightFlowState, node: str, exceeded: BudgetExceeded
) -> dict:
    """把一个 `BudgetExceeded` 转成"能安全收尾"的 state 增量。

    为什么不能让它继续往上抛：预算判定接在真实请求之前，因此它可能从任意
    一个 Agent 调用点抛出。未分类异常会让整个 Run 崩掉，而不是按既有语义
    收敛成失败——那既丢失了"为什么停"的解释，也让 RunStore 停在 running。

    返回的形状按节点语义区分，而不是一套通用字段：

    - `planner`：返回**空计划**。图在计划为空时会走
      `route_to_approval → complete`，从而正常收尾，不会带着不存在的
      计划去执行、也不会因为 `analysis_plan[0]` 越界而崩溃。
    - 其余节点：返回执行/绘图层面的失败记录，由既有的路由判断"不进入修复"。
    """
    log_event(
        run_id=state.get("run_id") or "unknown",
        node=node,
        event="budget_exceeded",
        status="failed",
        details={
            "scope": exceeded.scope,
            "used": exceeded.used,
            "limit": exceeded.limit,
        },
    )

    reason = str(exceeded)
    plan = state.get("analysis_plan") or []
    index = state.get("current_task_index") or 0
    task_id = plan[index]["task_id"] if 0 <= index < len(plan) else "budget"

    existing = list(state.get("failed_tasks") or [])
    # 同一任务已经因预算记过失败就不再追加，避免"失败任务数"被算多
    already = any(
        item.get("task_id") == task_id
        and item.get("error_kind") == BUDGET_ERROR_KIND
        for item in existing
    )
    if not already:
        existing.append(
            {
                "task_id": task_id,
                "reason": reason,
                "error": reason,
                "error_kind": BUDGET_ERROR_KIND,
                "retry_count": 0,
            }
        )

    common = {
        "analysis_status": "failed",
        "status_reason": reason,
        "failed_tasks": existing,
        # **必须把预算增量一起返回**。
        #
        # 真实请求的计数是就地写进当前节点那份 state 的（图会为每个节点复制
        # state），如果这里不把它带出去，LangGraph 就只合并我们返回的这几个键，
        # 累计用量会**回退**到该节点开始时的值：实测出现 total 从 2 掉回 1，
        # 于是已经耗尽的预算又被放行，上限 3 最终发出了 4 次请求。
        #
        # 返回**完整的** `execution_budget`（`limits` + `usage`），不是只返回
        # `usage`：后者会让 `limits` 丢失，判定随即回落到默认上限。
        BUDGET_FIELD: {
            "limits": (
                (state.get(BUDGET_FIELD) or {}).get("limits")
                if isinstance(state.get(BUDGET_FIELD), dict)
                else None
            )
            or AgentBudget().to_dict(),
            "usage": usage_of(state),
        },
    }

    if node == "planner_node":
        # 空计划 → route_to_approval 走 complete，工作流安全收尾。
        # 这也是既有"计划为空"的语义，只是原因说明不同。
        return {
            **common,
            "analysis_plan": [],
            "current_task_index": 0,
            "analysis_status": "no_tasks",
            "status_reason": f"执行预算已用尽，未能产出分析计划：{reason}",
        }

    if node == "execution_node":
        # 图上 `execution → tool_execution` 是一条**无条件**边，因此代码没能
        # 生成时不能什么都不返回：`tool_execution_node` 会去读 `current_code`
        # 并抛 KeyError，把"预算耗尽"变成一次未分类崩溃。
        #
        # 这里只补 `current_code`，**不动** `current_tool`：tool 是 router 已经
        # 定好的，下面 tool_execution 会走它既有的"代码为空"失败路径，
        # 从而按既有语义收敛成任务失败，而不是崩在图里。
        return {**common, "current_code": None}

    return common


from utils.run_logger import log_event
from time import perf_counter
from rag.knowledge_service import build_and_retrieve_knowledge

# 运行治理通过 Harness 进入：模型调用、工具派发与预算都在那一层，
# 节点只解释结果。`BudgetExceeded` 仍要在模块级可用——`_with_state`
# 包装器在导入期就把它写进 except 子句。
from harness.runtime import bind_state, get_harness
from planning import (
    current_task,
    plan_fingerprint,
    validate_plan,
    verify_approved_plan,
)
from utils.agent_loop import BUDGET_ERROR_KIND, BUDGET_FIELD, BudgetExceeded

HARNESS = get_harness()

# 绘图重试上限。与 `conditions.route_after_plot_execution` 共用同一个常量：
# 预算耗尽时把计数直接置满，路由就会跳过 plot_repair——重画同一张图
# 在预算已尽的情况下不会好转，只会再尝试启动一个容器。
# 从 conditions 导入而不在这里另写一个字面量，避免两处上限将来漂移。
from conditions import MAX_PLOT_RETRIES

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
        "table_snapshot": snapshots_to_state(snapshots),
        # 预算在这里落地：只补缺失字段，已有计数原样保留，
        # 因此恢复时不会把已消耗的预算清零。
        **HARNESS.initialize_budget(state),
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


#: 首次规划 + 重新规划的总尝试次数。次数上限写进 state 的 `plan_attempts`，
#: 恢复后从已用次数继续，因此崩溃不会换来新的重试额度。
MAX_PLAN_ATTEMPTS = 3


def _log_plan(
    run_id: str | None,
    event: str,
    status: str,
    validation: Any,
    attempts: int,
) -> None:
    if run_id is None:
        return
    log_event(
        run_id=run_id,
        node="planner",
        event=event,
        status=status,
        details={
            "attempts": attempts,
            "task_count": len(validation.tasks),
            "categories": validation.categories,
            "errors": [error.message for error in validation.errors][:5],
        },
    )


@_with_state
def planner_node(
    state: InsightFlowState
) -> dict:

    retrieved_knowledge = state.get(
        "retrieved_knowledge",
        []
    )

    run_id = state.get("run_id")

    # 计划必须落在本次 Run 的真实资源上：授权集合用快照的键，
    # 来源类型决定现有路由能否用同一个工具执行。
    authorized = set(state.get("data_context") or {})
    source_types = {
        name: snapshot.get("source_type")
        for name, snapshot in (state.get("table_snapshot") or {}).items()
        if isinstance(snapshot, dict)
    }

    # 已经用满次数（例如上一次执行以"计划未通过校验"收尾）就不再尝试：
    # 恢复不能换来新的重试额度。
    if int(state.get("plan_attempts") or 0) >= MAX_PLAN_ATTEMPTS:
        return {
            "analysis_plan": [],
            "current_task_index": 0,
            "analysis_status": "failed",
            "status_reason": (
                f"分析计划在 {MAX_PLAN_ATTEMPTS} 次尝试后仍未通过校验"
            ),
            "plan_attempts": MAX_PLAN_ATTEMPTS,
            "plan_validation_errors": list(state.get("plan_validation_errors") or []),
            "completed_operations": dict(state.get("completed_operations") or {}),
        }

    attempt_index = 0
    feedback = list(state.get("plan_validation_errors") or [])
    operations = dict(state.get("completed_operations") or {})
    budget_extra: dict = {}

    while attempt_index < MAX_PLAN_ATTEMPTS:
        # 每次重新规划是独立的逻辑调用：身份里带尝试序号，幂等层不会把
        # 上一次被拒的计划当成这一次的结果复用。
        #
        # 序号从 0 开始、不接着 state 里的次数往下排：节点被重放时（例如
        # 进程崩溃后从入口重新走到这里）必须用同一个身份命中幂等复用，
        # 否则一次"重放"会变成一次新的模型调用。
        plan, _ops, _extra = HARNESS.model_call(
            "planner",
            attempt_subject("plan", attempt_index),
            lambda: create_analysis_plan(
                contexts=state["data_context"],
                analysis_focus=state.get(
                    "analysis_focus"
                ),
                retrieved_knowledge=retrieved_knowledge,
                validation_feedback=feedback,
                llm=model_for("planner"),
            ),
            attempt=attempt_index + 1,
        )
        attempt_index += 1
        operations.update(_ops or {})
        budget_extra = _extra or {}

        validation = validate_plan(
            plan,
            authorized_tables=authorized,
            source_types=source_types,
        )

        if validation.ok:
            _log_plan(run_id, "plan_validated", "success", validation, attempt_index)

            if validation.is_empty:
                # 空计划意味着"没有可分析的任务"，不是失败。
                return {
                    "analysis_plan": [],
                    "current_task_index": 0,
                    "analysis_status": "no_tasks",
                    "status_reason": "Planner 未生成任何分析任务",
                    "plan_attempts": attempt_index,
                    "plan_validation_errors": [],
                    "completed_operations": operations,
                    **_extra,
                }

            if run_id is not None:
                log_event(
                    run_id=run_id,
                    node="planner",
                    event="plan_created",
                    status="success",
                    details={
                        "task_count": len(validation.tasks),
                        "task_ids": [
                            task["task_id"]
                            for task in validation.tasks
                        ],
                        "retrieved_knowledge_count": len(
                            retrieved_knowledge
                        )
                    }
                )

            return {
                "analysis_plan": validation.tasks,
                "current_task_index": 0,
                "plan_attempts": attempt_index,
                "plan_validation_errors": [],
                "completed_operations": operations,
                **_extra,
            }

        feedback = validation.feedback()
        _log_plan(run_id, "plan_rejected", "failed", validation, attempt_index)

        if attempt_index < MAX_PLAN_ATTEMPTS and run_id is not None:
            log_event(
                run_id=run_id,
                node="planner",
                event="replan_requested",
                status="retrying",
                details={
                    "attempts": attempt_index,
                    "categories": validation.categories,
                },
            )

    # 次数用尽：不再生成计划，也不把无效计划交给 Router/Executor。
    if run_id is not None:
        log_event(
            run_id=run_id,
            node="planner",
            event="replan_exhausted",
            status="failed",
            details={
                "attempts": attempt_index,
                "categories": sorted({item.split("：")[0] for item in feedback})[:5],
            },
        )

    return {
        "analysis_plan": [],
        "current_task_index": 0,
        "analysis_status": "failed",
        "status_reason": (
            f"分析计划在 {attempt_index} 次尝试后仍未通过校验："
            + "；".join(feedback[:3])
        ),
        "plan_attempts": attempt_index,
        "plan_validation_errors": feedback,
        "completed_operations": operations,
        **budget_extra,
    }


def router_node(
    state: InsightFlowState
) -> dict:

    # 执行前确认：现在这份计划就是人工批准的那一份
    verify_approved_plan(state)

    task = current_task(
        state.get("analysis_plan"),
        state.get("current_task_index"),
    )

    if task is None:
        # 计划为空或索引越界都走这里：明确失败，而不是 IndexError
        raise ValueError("路由阶段没有可执行任务：计划为空或索引越界")

    # 路由要按表的来源类型决定工具，因此资源名必须先过校验：
    # 不存在的表在这里被明确拒绝，而不是变成下游的 KeyError。
    HARNESS.ensure_resources(state, task["target_tables"])

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

    data_context = HARNESS.task_context(state, task["target_tables"]).tables

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
    code, _ops, _extra = HARNESS.model_call(
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

    # 缺少工具或代码是**任务级失败**，不是工作流崩溃。
    #
    # 图里 `execution → tool_execution` 是一条无条件边，因此当上游因为预算
    # 耗尽没能产出代码时，这个节点一定会被进入。此时抛 ValueError 会把
    # "预算耗尽、安全终止"变成一次未分类异常（Run 停在 running、原因丢失）。
    # 下面把它转成与"工具返回失败"完全相同的形状，交给既有的
    # `route_after_execution` 走 failed 分支收敛。
    if tool is None or code is None or tool not in available_tools():
        missing = (
            "Current tool is missing."
            if tool is None
            else (
                "Current code is missing."
                if code is None
                else f"Unsupported tool: {tool}"
            )
        )
        log_event(
            run_id=state.get("run_id") or "unknown",
            node="tool_execution",
            event="execution_skipped",
            status="failed",
            details={
                "task_id": task["task_id"],
                "tool": tool,
                "reason": missing,
                "retry_count": state.get("retry_count", 0),
            },
        )
        previous = state.get("execution_results", [])
        attempt = (
            task["task_id"],
            state.get("retry_count", 0),
            state.get("refine_count", 0),
        )
        deduped = [
            existing
            for existing in previous
            if (
                existing.get("task_id"),
                existing.get("retry_count", 0),
                existing.get("refine_count", 0),
            )
            != attempt
        ]
        return {
            "execution_results": deduped
            + [
                {
                    "task_id": task["task_id"],
                    "tool_used": tool,
                    "code": code,
                    "success": False,
                    "result": None,
                    "result_artifact": None,
                    "error": missing,
                    "retry_count": state.get("retry_count", 0),
                    "refine_count": state.get("refine_count", 0),
                }
            ],
            "execution_error": missing,
            # 归类为环境类：重发一次同样的"空代码"不会好转，
            # 因此不进入修复循环，直接收敛成失败。
            "execution_error_kind": BUDGET_ERROR_KIND,
        }

    # 参数校验失败与执行失败都以错误字典返回，交给 route_after_execution
    # 走修复循环，而不是在这里抛异常终止工作流。
    arguments = (
        {"table_names": task["target_tables"], "code": code}
        if tool == "python"
        else {"table_names": task["target_tables"], "query": code}
    )

    # 派发交给 Harness：预算判定、授权资源解析、错误分类与执行时间结算都在
    # 那一层完成。预算耗尽时它记一条 budget_exceeded 并抛出，这里只决定
    # 这次失败长成什么样——环境类失败，路由据此不进入修复（重试不会好转）。
    budget_updates: dict = {}
    try:
        outcome = HARNESS.run_tool(
            state,
            node="tool_execution",
            tool=tool,
            arguments=arguments,
            context={"task_id": task["task_id"]},
        )
    except BudgetExceeded as exceeded:
        execution = {
            "success": False,
            "result": None,
            "error": str(exceeded),
            "error_kind": BUDGET_ERROR_KIND,
        }
    else:
        execution = outcome.execution
        budget_updates = outcome.budget_updates

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
                # 分类随状态一起记录：只看错误文案无法区分"环境类失败"
                # 与"代码写错了"，而路由正是按 error_kind 决定的。
                "error_kind": execution.get("error_kind"),
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
        # P3-A/B：把这次工具调用的预算计数落盘。不写回的话
        # "已消耗"不会进 Checkpoint，恢复后预算就归零了。
        **budget_updates,
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

    data_context = HARNESS.task_context(state, task["target_tables"]).tables

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

    repaired_code, _ops, _extra = HARNESS.model_call(
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
        # P3-A：进入修复就是一次新的 Reason→Act→Observe→Decide 轮次。
        **HARNESS.note_round(state),
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

    data_context = HARNESS.task_context(state, task["target_tables"]).tables

    execution_result = state["execution_results"][-1]

    # 每轮精修后都会重新评审同一个任务，因此评审身份必须带 refine_count：
    # 否则第二轮评审会复用第一轮的结论，精修循环等于没有校验。
    review_round = state.get("refine_count", 0)

    review, _ops, _extra = HARNESS.model_call(
        "critic",
        attempt_subject(task["task_id"], review_round),
        lambda: review_execution_result(
            task=task,
            data_context=data_context,
            execution_result=execution_result,
            retrieved_knowledge=retrieved_knowledge,
            # 当前任务之前的尝试（有界摘要），用来说明这次结果为什么长这样
            attempt_history=HARNESS.history_for_task(state),
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

    data_context = HARNESS.task_context(state, task["target_tables"]).tables

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

    refined_code, _ops, _extra = HARNESS.model_call(
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
        # P3-A：进入精修同样是一次新的循环轮次。
        **HARNESS.note_round(state),
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

    # 没有任何执行结果时不必调用模型。
    #
    # 这条路径由"预算耗尽"暴露出来：执行阶段被预算拦下后，图仍会走到这里
    # （`failed_task → advance → complete → visualization_planner → report`）。
    # 此前它会正常调用 Report Agent，而预算已尽 → 再抛 `BudgetExceeded`；
    # 即便拿到了内容，`artifact_node` 也会因为缺少产物而崩。
    # 因此这里显式收敛：没有可汇报的结果时直接给出结论，不产出产物文件。
    #
    # **不要覆盖已经确定的结论**：计划为空时上游已经把 analysis_status
    # 置为 `no_tasks`（一个合法终态），预算耗尽时置为 `failed`。
    # 这里若无条件写 `failed`，会把"没有任务可分析"错报成"运行失败"——
    # 那正是 `test_planner_rejecting_is_not_reported_as_success` 守住的性质。
    if not get_final_execution_results(state.get("execution_results", [])):
        existing_status = state.get("analysis_status")
        reason = (
            state.get("status_reason")
            or "没有任何成功的分析结果可供生成报告"
        )
        return {
            "report_content": None,
            "analysis_status": existing_status or "failed",
            "status_reason": reason,
        }

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

    report, _ops, _extra = HARNESS.model_call(
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
                ),
                # 这次报告上下文的预算与真实用量记进结构化日志，
                # 使"上下文用了多少"成为可观察事实而不是猜测。
                # estimated_tokens 是估算值（无匹配 tokenizer），字段名里已标注。
                "context": HARNESS.context_report("reporter"),
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

    # 报告内容可能是有意不产出（例如预算耗尽、没有任何执行结果）。
    # 这种情况不该崩在工作流里：直接跳过产物生成，让 Run 以 failed 收尾。
    report_content = state.get("report_content")

    if report_content is None:
        return {
            "artifacts": list(state.get("artifacts") or []),
            "report_path": state.get("report_path"),
            "analysis_status": state.get("analysis_status") or "failed",
            "status_reason": state.get("status_reason")
            or "没有报告内容，未生成产物",
        }

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
            # 人工批准的是这份计划的具体内容，不是一个模糊的"继续"
            "plan_fingerprint": plan_fingerprint(plan),
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

    return _apply_decision(state, request, decision, field="plan_approval")


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
    state: InsightFlowState, request: dict, decision: Any, *, field: str
) -> dict:
    """把审批结果归档，并给出后续路由依据。

    重复提交同一个审批 ID 时直接复用已归档的决定，不再触发第二次中断，
    避免同一次操作被批准两次。

    `field` 指明这次决定归属哪个审批（`plan_approval` / `report_approval`）：
    两个关口的决定必须分开存放，否则报告审批会覆盖计划审批的路由依据。
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
        node=request["node"],
        event="approval_decided",
        status=verdict,
        details=record_entry,
    )

    approvals = list(state.get("approvals") or [])
    approvals.append(record_entry)

    approved = verdict == "approve"

    # 拒绝时的措辞按关口区分：计划被拒与报告被拒是两件事，
    # 用同一句"分析计划未获批准"来描述报告被拒会误导排查。
    if approved:
        reason = None
    elif field == "report_approval":
        reason = "分析报告未获批准发布"
    else:
        reason = "分析计划未获批准"

    updates = {
        field: record_entry,
        "approvals": approvals,
        "pending_approval": None,
        "run_status": "completed" if approved else "failed",
        "analysis_status": None if approved else "failed",
        "status_reason": reason,
    }

    if field == "plan_approval" and approved:
        # 记住"批准的是这一份"，执行前比对；计划被换掉就不执行。
        updates["approved_plan_fingerprint"] = plan_fingerprint(
            state.get("analysis_plan") or []
        )

    return updates


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
    return _apply_decision(
        state, request, decision, field="report_approval"
    )


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

    visualization_plan, _ops, _extra = HARNESS.model_call(
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

    data_context = HARNESS.task_context(state, target_tables).tables

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
        # 没有可用的源结果时绘图无法进行（典型原因：分析阶段被预算拦下，
        # 或上游任务失败）。这里收敛成"这张图失败"，而不是抛未分类异常——
        # 抛异常会让 Run 停在 running，并把"预算耗尽"这个已知原因
        # 淹没在一个通用崩溃里。
        reason = (
            f"绘图所需的分析结果不可用：源任务 {source_task_id} 没有成功的执行结果"
        )
        # run_id 在这个节点里稍后才赋值，这里单独取一次（日志需要它）
        log_run_id = state.get("run_id")
        if log_run_id is not None:
            log_event(
                run_id=log_run_id,
                node="plot_generation",
                event="plot_skipped",
                status="failed",
                details={
                    "chart_id": chart_id,
                    "source_task_id": source_task_id,
                    "reason": reason,
                },
            )
        return {
            "current_plot_code": None,
            "visualization_results": [
                result
                for result in state.get("visualization_results", [])
                if result.get("chart_id") != chart_id
            ]
            + [
                {
                    "chart_id": chart_id,
                    "code": None,
                    "success": False,
                    "path": None,
                    "error": reason,
                }
            ],
            "plot_error": reason,
            # 置满重试计数：重画不会好转，路由直接 advance_chart
            "plot_retry_count": MAX_PLOT_RETRIES,
        }

    generation, _ops, _extra = HARNESS.model_call(
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
        # 绘图计划产出为空（例如预算拦下了代码生成）。
        # 与 `tool_execution_node` 同样的道理：图上 `plot_generation →
        # plot_execution` 是无条件边，这里抛异常会把"预算耗尽、安全终止"
        # 变成未分类崩溃。返回失败结果让路由按既有语义走 advance_chart。
        chart_id = visualization_task["chart_id"]
        reason = "绘图代码为空，未执行绘图"
        previous = [
            result
            for result in state.get("visualization_results", [])
            if result.get("chart_id") != chart_id
        ]
        return {
            "visualization_results": previous
            + [
                {
                    "chart_id": chart_id,
                    "code": None,
                    "success": False,
                    "path": None,
                    "error": reason,
                }
            ],
            "plot_error": reason,
            "plot_retry_count": MAX_PLOT_RETRIES,
        }

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

    # 绘图与 Python/SQL 走同一个派发入口。绘图是开销最大、有副作用的工具，
    # 因此预算与授权不能在这里被绕开。
    budget_updates: dict = {}
    try:
        outcome = HARNESS.run_tool(
            state,
            node="plot_execution",
            tool="plot",
            arguments={
                "table_names": target_tables,
                "chart_id": chart_id,
                "code": code,
                "analysis_result": unwrap_records(analysis_result),
            },
            context={"chart_id": chart_id},
        )
    except BudgetExceeded as exceeded:
        # 预算耗尽属于环境类失败：重画同一张图不会好转，因此**不允许**
        # 进入 plot_repair 循环（那会为了重试而再次尝试启动容器）。
        #
        # 用既有的 plot_retry_count 表达"不再重试"，而不是新增 state 字段：
        # 未在 InsightFlowState 里声明的键会被 LangGraph 静默丢掉，
        # 那样路由读不到标志，还是会把流程送进 plot_repair。
        previous_results = [
            result
            for result in state.get("visualization_results", [])
            if result.get("chart_id") != chart_id
        ]
        return {
            "visualization_results": previous_results
            + [
                {
                    "chart_id": chart_id,
                    "code": code,
                    "success": False,
                    "path": None,
                    "error": str(exceeded),
                }
            ],
            "plot_error": str(exceeded),
            # 置满重试计数 → route_after_plot_execution 直接走 advance_chart
            "plot_retry_count": MAX_PLOT_RETRIES,
        }
    else:
        execution = outcome.execution
        budget_updates = outcome.budget_updates

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
                "error_kind": execution.get("error_kind"),
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

    # 把治理层产生的预算增量（工具调用计数 + 执行时钟结算）并入返回值，
    # 否则绘图消耗不会落盘，恢复后预算就归零了。
    updates.update(budget_updates)

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

    data_context = HARNESS.task_context(state, target_tables).tables

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

    repaired_code, _ops, _extra = HARNESS.model_call(
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
        # P3-A：绘图修复是一个独立的有界循环，也计入总轮次。
        **HARNESS.note_round(state),
        **_extra,
    }
