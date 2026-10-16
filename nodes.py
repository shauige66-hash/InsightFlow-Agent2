# 定义 LangGraph 工作流节点，负责读取和更新共享 State。

import contextvars
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


def _governed_agent_call(
    agent: str,
    subject: str,
    invoke: Callable[[], Any],
    *,
    attempt: int = 1,
) -> tuple[Any, dict, dict]:
    """`run_agent_call` 之上再套一层预算治理。

    分工必须说清楚，否则会出现两套互相打架的计数：

    - `utils.idempotency.run_agent_call` 负责**幂等**：
      同一逻辑调用已完成就复用结果，以及"请求发出但未确认"的人工拦截；
    - 这里负责**总量**：把**真正发生过上游请求**的调用计入
      `execution_budget.llm_calls`。

    两者都不替代对方：幂等管"要不要重发"，预算管"还能不能发"。

    命中幂等复用时不计数——那次没有发生模型调用，计进去会让预算虚高。
    判据来自 `run_agent_call` 的内部标记，标记本身必须**从 state 增量里剔除**，
    否则会写进 Checkpoint（未声明的键会被 LangGraph 丢掉，但留着会误导阅读）。
    """
    payload, operations, extra = run_agent_call(
        _CURRENT_STATE.get() or {},
        node=agent,
        subject=subject,
        call=invoke,
        attempt=attempt,
        # 预算判定放在**幂等复用之后、账本落 started 之前**：
        # 复用不该消耗额度，而超预算时绝不能往账本写一条 started——
        # 那会留下一个永远等不到 completed 的记录，恢复时被判成
        # "未确认的模型调用"并要求人工裁决，于是一个纯粹的预算问题
        # 变成了人工工单。
        before_request=_budget_guard,
    )

    extra = dict(extra or {})
    invoked = bool(extra.pop("__upstream_invoked__", False))
    # 操作 ID 与账本、幂等登记用的是同一个值，日志带上它才能把
    # "这次模型调用"与账本里的 started/completed 对上。它不进 state。
    operation_id = extra.pop("__operation_id__", None)

    if invoked:
        # 计数**不在**这里做：真实请求数由 `utils/agent_loop.after_model_request`
        # 在每次请求前逐次累加（含同一次逻辑调用内部的重试）。
        # 这里曾经额外 `consume_llm_call` 一次，于是同一次请求被算了两次——
        # 实测声明上限 3 时实际只发出 2 次请求就判定耗尽。
        #
        # 这里只做时钟结算：模型调用已结束，若接着去等人审批，时间不再累积。
        settled = settle_active_time(_CURRENT_STATE.get() or {})
        extra[BUDGET_FIELD] = settled[BUDGET_FIELD]

        # P4：把这个 Agent 的上下文度量写进结构化日志。
        # 各 Agent 在裁剪时把结果放进自己的上下文变量，这里按 agent 名取回。
        report = _context_report_of(agent)
        if report:
            log_event(
                run_id=(_CURRENT_STATE.get() or {}).get("run_id") or "unknown",
                node=agent,
                event="llm_context",
                status="ok",
                details={
                    "subject": subject,
                    "operation_id": operation_id,
                    "context": report,
                },
            )

    return payload, operations, extra


def _context_report_of(agent: str) -> dict:
    """取回某个 Agent 最近一次上下文裁剪的度量。

    P4 的度量通过 `contextvars.ContextVar` 传递，因此**按执行上下文隔离**：
    并发执行的两个 Run 各自看到自己的那份，不会互相串写。
    早期实现把度量放在模块级字典里，两个 Run 并发调用同一个 Agent 时
    可能把 A 的度量写进 B 的日志。

    取不到就返回空字典（不因为日志缺字段而影响执行）。
    """
    from utils.context_budget import context_report_of

    return context_report_of(agent)


def _budget_guard() -> None:
    """进入一次**真实模型请求**前判预算；超了抛 `BudgetExceeded`。

    这是模型预算真正生效的地方。只判不记：计由请求计数钩子完成——命中幂等
    复用的调用不该被算成新请求。

    只看 `llm_calls`：这一次请求额外消耗的是一个模型调用额度，不该因为工具
    调用或时间预算刚好用尽就被拒绝；那些分项在各自的入口
    （`governed_tool_call`、`_round_update`）判定。

    `run_agent_call` 在落账本之前调它一次，`utils.llm_retry` 在每次重发之前
    再调一次。两次都只读。
    """
    from utils.agent_loop import check_llm_budget

    # `llm_calls` 在每次请求**发出前**就地累加，因此同一逻辑调用内部的重试
    # 读到的已经是"已发出的请求数"，这里只比较、不再额外加一。
    check_llm_budget(_CURRENT_STATE.get() or {})


def _round_update(*, node: str) -> dict:
    """记一次 Reason→Act→Observe→Decide 轮次。"""
    from utils.agent_loop import consume_round

    return consume_round(_CURRENT_STATE.get() or {})


# 当前节点正在处理的 state。节点函数签名不能带额外参数（LangGraph 会把
# 参数名当成状态通道），因此这里用 ContextVar 传递，避免改动 19 个节点签名。
_CURRENT_STATE: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "insightflow_current_state", default=None
)


def _with_state(func: Callable) -> Callable:
    """把 state 放进 ContextVar，供预算闸门与请求计数钩子读取。

    只包装实际调用 Agent 的节点；其余节点不受影响。

    P3-A：同时在这里确保 state 里有一份预算快照。放在这一层是因为
    所有会消耗预算的节点都被它包着，而 `initialize_budget` 是**幂等**的
    （已有计数不重置），所以恢复后重复进入不会把预算清零。

    P3 定点修复：`BudgetExceeded` 也在这里统一收敛。预算判定现在接在
    **真实请求之前**，因此它会从任意一个 Agent 调用点抛出来（10 处）。
    逐处 catch 既啰嗦又容易漏掉一处，漏掉就意味着预算耗尽变成未分类异常、
    整个 Run 崩掉而不是安全终止。这一层是所有消耗预算的节点的共同入口，
    在这里转成"环境类失败"最稳妥：归类为 `unavailable`，不进入 Repair。
    """

    @functools.wraps(func)
    def wrapper(state: InsightFlowState) -> dict:
        # 预算是"当前 state 已有则保留、没有才建"的语义。
        # 这里只在缺字段时补一个 ContextVar 可见的副本，不写回 state——
        # 写回必须由节点自己的返回值完成，否则会被 LangGraph 当成未声明增量丢弃。
        token = _CURRENT_STATE.set(state)
        try:
            return func(state)
        except BudgetExceeded as exceeded:
            return _budget_failure(state, func.__name__, exceeded)
        finally:
            _CURRENT_STATE.reset(token)

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


def _register_request_hooks() -> None:
    """把预算钩子注册进 `utils/agent_loop`。

    为什么需要这一层：Agent 在**每次真实模型请求**前后都要判预算与计数
    （内部重试的每一次请求也算），但 Agent 不能 import 节点模块——那会形成
    `agents → nodes → agents` 的循环依赖。因此节点层把自己的取值方式注册过去。

    `on_request` 直接**就地更新** state 里的用量。这样做是必要的：
    一次逻辑调用内部可能发出多次请求，而 state 的正式增量要等节点返回才合并；
    不就地累加的话，第二次请求判定时读到的仍是旧计数，上限 1 也拦不住第 2 次。
    就地更新改变的是 LangGraph 传入的那份 state 对象，节点返回的增量是它的
    超集，因此不会互相覆盖。
    """
    from utils.agent_loop import (
        BUDGET_FIELD,
        consume_llm_call,
        set_request_hooks,
        set_state_provider,
    )

    def _state() -> dict | None:
        return _CURRENT_STATE.get()

    def _before(attempt: int) -> None:
        _budget_guard()

    def _count(attempt: int) -> None:
        state = _CURRENT_STATE.get()
        if state is None:
            return
        # 就地累加：同一次逻辑调用内部的后续重试要靠它看到"已经发过几次"，
        # 正式落盘仍由节点返回的增量完成。
        updates = consume_llm_call(state)
        state[BUDGET_FIELD] = updates[BUDGET_FIELD]

    set_state_provider(_state)
    set_request_hooks(before=_before, after=_count)


_register_request_hooks()
from utils.run_logger import log_event
from time import perf_counter
from rag.knowledge_service import build_and_retrieve_knowledge

# P3：有界执行与统一治理。放在这里而不是散进各节点，
# 是为了让"预算判定"与"结构化日志"只有一个出口。
from utils.agent_loop import (
    BUDGET_ERROR_KIND,
    BUDGET_FIELD,
    BudgetExceeded,
    initialize as initialize_budget,
    settle_active_time,
)
from utils.middleware import governed_tool_call

# `BudgetExceeded` 与预算字段要在模块顶部可用：`_with_state` 包装器在导入期
# 就把它写进 except 子句，晚于它的局部导入（原先在函数体里）帮不上忙。
from utils.agent_loop import BudgetExceeded

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
        # P3-A：预算在这里落地。`initialize_budget` 只补缺失字段，
        # 已有计数原样保留，因此恢复时不会把已消耗的预算清零。
        **initialize_budget(state),
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

    plan, _ops, _extra = _governed_agent_call(
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
    code, _ops, _extra = _governed_agent_call(
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

    # 参数校验失败与执行失败都从 invoke_tool 以错误字典返回，
    # 交给 route_after_execution 走修复循环，而不是在这里抛异常终止工作流。
    arguments = (
        {"table_names": task["target_tables"], "code": code}
        if tool == "python"
        else {"table_names": task["target_tables"], "query": code}
    )

    # 权威数据快照一并传入：MCP 后端据此确定授权集合（哪些 dataset_id 可见），
    # direct 后端忽略它。不传的话 MCP 只能按本次点名的表临时授权，
    # 与 P1 的"数据源变化必须失败"语义脱节。
    #
    # P3-B：经统一治理层调用。它在真正执行前判一次预算、执行后记一次
    # 结构化日志与耗时；预算耗尽时抛 BudgetExceeded，这里转成环境类失败，
    # 由既有的 route_after_execution 走"失败"而不是"修复"（重试不会好转）。
    budget_updates: dict = {}
    try:
        with governed_tool_call(state, node="tool_execution", tool=tool) as gate:
            execution = invoke_tool(
                tool,
                resolve_tables(state),
                arguments,
                snapshots=state.get("table_snapshot") or {},
            )
            gate["error_kind"] = execution.get("error_kind")
        # finally 里才结算执行时钟，必须在 with 退出后重读，否则拿到的是
        # 结算前的 clock_open=True
        budget_updates = gate.get("budget_updates") or {}
    except BudgetExceeded as exceeded:
        execution = {
            "success": False,
            "result": None,
            "error": str(exceeded),
            "error_kind": BUDGET_ERROR_KIND,
        }
        log_event(
            run_id=state.get("run_id") or "unknown",
            node="tool_execution",
            event="budget_exceeded",
            status="failed",
            details={"scope": exceeded.scope, "used": exceeded.used, "limit": exceeded.limit},
        )

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

    repaired_code, _ops, _extra = _governed_agent_call(
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
        **_round_update(node="repair"),
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

    review, _ops, _extra = _governed_agent_call(
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

    refined_code, _ops, _extra = _governed_agent_call(
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
        **_round_update(node="refine"),
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

    report, _ops, _extra = _governed_agent_call(
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
                # P4：把这次报告上下文的预算与真实用量记进结构化日志，
                # 使"上下文用了多少"成为可观察事实而不是猜测。
                # estimated_tokens 是估算值（无匹配 tokenizer），字段名里已标注。
                # 度量按执行上下文隔离：并发 Run 之间不会互相串写。
                "context": _context_report_of("reporter"),
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

    return {
        field: record_entry,
        "approvals": approvals,
        "pending_approval": None,
        "run_status": "completed" if approved else "failed",
        "analysis_status": None if approved else "failed",
        "status_reason": reason,
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

    visualization_plan, _ops, _extra = _governed_agent_call(
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

    generation, _ops, _extra = _governed_agent_call(
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

    # 绘图与 Python/SQL 走**同一个**治理入口。此前这里直接调用 invoke_tool，
    # 于是绘图既不计入 tool_calls，也不受预算约束——预算设为 0 时仍会启动
    # 一个 Docker 容器。绘图是有副作用且开销最大的工具，不能是治理的缺口。
    budget_updates: dict = {}
    try:
        with governed_tool_call(state, node="plot_execution", tool="plot") as gate:
            execution = invoke_tool("plot", resolve_tables(state), {
                "table_names": target_tables,
                "chart_id": chart_id,
                "code": code,
                "analysis_result": unwrap_records(analysis_result),
            })
            gate["error_kind"] = execution.get("error_kind")
        # 同 tool_execution：时钟在 finally 里结算，退出 with 之后再取
        budget_updates = gate.get("budget_updates") or {}
    except BudgetExceeded as exceeded:
        # 预算耗尽属于环境类失败：重画同一张图不会好转，因此**不允许**
        # 进入 plot_repair 循环（那会为了重试而再次尝试启动容器）。
        #
        # 用既有的 plot_retry_count 表达"不再重试"，而不是新增 state 字段：
        # 未在 InsightFlowState 里声明的键会被 LangGraph 静默丢掉，
        # 那样路由读不到标志，还是会把流程送进 plot_repair。
        log_event(
            run_id=run_id or "unknown",
            node="plot_execution",
            event="budget_exceeded",
            status="failed",
            details={
                "scope": exceeded.scope,
                "used": exceeded.used,
                "limit": exceeded.limit,
                "chart_id": chart_id,
            },
        )
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

    repaired_code, _ops, _extra = _governed_agent_call(
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
        **_round_update(node="plot_repair"),
        **_extra,
    }
