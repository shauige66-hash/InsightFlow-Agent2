"""统一封装 InsightFlow 工作流运行逻辑，供命令行与 API 入口复用。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Command

from core.llm import ModelFactory
from core.runtime import (
    McpSettings,
    RunRuntime,
    build_thread_config,
    use_runtime,
)
from ingestion.resources import ResourceResolver
from utils.agent_loop import AgentBudget, new_usage


def _resolve_budget(budget: AgentBudget | dict[str, Any] | None) -> AgentBudget:
    """把调用方给的预算收敛成 `AgentBudget`。

    接受 dict 是为了让 CLI/脚本能直接写配置；未知字段被忽略、
    负数被拒绝（与 `AgentBudget.from_dict` 同一套规则，见那里的说明：
    `0` 是合法上限，表示"一次都不许"）。
    """
    if budget is None:
        return AgentBudget()
    if isinstance(budget, AgentBudget):
        return budget
    if isinstance(budget, dict):
        return AgentBudget.from_dict(budget)
    raise TypeError(f"不支持的预算类型：{type(budget).__name__}")

from utils.idempotency import LlmCallAbandoned
from agent_loop.policy import limits_from
from utils.run_logger import create_run_id, log_event
from workflow import create_workflow

DEFAULT_RECURSION_LIMIT = 100


def _budget_snapshot(result: dict[str, Any]) -> dict[str, Any]:
    """从结果里取预算用量快照，供 `run_completed` 记录。

    只看图状态里的 `execution_budget`；没有就返回空字典，不因为缺字段影响收尾。
    """
    from utils.agent_loop import snapshot

    return snapshot(result) if isinstance(result.get("execution_budget"), dict) else {}


def _resume_inputs(
    run_id: str, runtime: RunRuntime, *, grant_retry: bool = False
) -> dict[str, Any]:
    """从 Checkpoint 续跑时提供的增量输入。

    只有 `pending_llm_calls` 需要在这里给出：它是**本次运行**才发现的证据
    （来自账本），不能依赖上一次进程把它写进 Checkpoint。
    其余字段由 LangGraph 从 Checkpoint 恢复。

    `grant_retry` 表示调用方已经决定重发未确认的调用。它必须作为 state 的一部分
    传入：恢复是从上一张 Checkpoint 继续，若只靠 `Command` 传参，
    中断恢复后这个决定就丢了，会被反复询问。
    """
    return {
        "pending_llm_calls": list(runtime.unconfirmed_llm_calls),
        "llm_retry_granted": "retry" if grant_retry else None,
        "require_plan_approval": None,
    }


def _unconfirmed_llm_calls(run_id: str, output_root: Path) -> list[str]:
    """从账本找出"已发出但没有完成记录"的模型调用。

    Checkpoint 只在节点结束时提交，因此"请求已发出、节点未返回"的崩溃在
    state 里没有任何痕迹；只有账本能提供这个证据。这里把它读出来交给工作流，
    由 guard 决定是否暂停等人工确认。
    """
    from utils import llm_ledger

    summary = llm_ledger.summarize(run_id, root=output_root)
    return list(summary["unconfirmed_operations"])


def run_analysis(
    data_sources: list[str],
    analysis_focus: str | None = None,
    knowledge_sources: list[str] | None = None,
    run_id: str | None = None,
    model_factory: ModelFactory | None = None,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    resolver: ResourceResolver | None = None,
    output_root: Path | str = "outputs",
    resume: Any | None = None,
    require_plan_approval: bool = False,
    require_report_approval: bool = False,
    continue_from_checkpoint: bool = False,
    resume_value: Any | None = None,
    mcp: McpSettings | None = None,
    budget: AgentBudget | dict[str, Any] | None = None,
    conversation_context: dict[str, Any] | None = None,
    memories: list[dict[str, Any]] | None = None,
    enable_dynamic_loop: bool = False,
    dynamic_limits: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """执行（或恢复）一次完整分析。

    运行期依赖通过 ContextVar 注入而不是放进 state —— state 会写进 Checkpoint，
    模型客户端与 DataFrame 都无法跨进程恢复。

    `resume` 用于把人工审批决定交回被中断的节点（LangGraph 的 Command）。
    `continue_from_checkpoint` 用于进程异常终止后的续跑：传 None 作为输入，
    让 LangGraph 从最后提交的 Checkpoint 继续，而不是从头重跑。
    `checkpointer` 为 None 时退化为无持久化的一次性执行，便于单次脚本调用。

    `budget` 覆盖默认执行预算（轮次/工具调用/模型请求/墙上时钟）。
    与 `mcp` 一样属于**可信运行期配置**：它决定"这次允许做多少工作"，
    因此只由宿主调用方提供，不从请求体读取。传 None 用默认值。
    注意它只影响**新建**的预算快照；恢复已有 Run 时以 state 里已存的为准。

    注意：从 Checkpoint 续跑意味着最后提交点之后的工作会被重放。
    有副作用的步骤依靠幂等登记跳过，模型调用无法保证恰好一次（见
    utils/idempotency.py 的说明）。
    """
    if run_id is None:
        run_id = create_run_id()

    runtime = RunRuntime(
        run_id=run_id,
        model_factory=_require_factory(model_factory, run_id),
        resolver=resolver or ResourceResolver(),
        output_root=Path(output_root),
        unconfirmed_llm_calls=_unconfirmed_llm_calls(run_id, Path(output_root)),
        # MCP 后端属于可信运行期配置：默认 direct，只有调用方显式指定才走 MCP。
        # 它不进 state，因此恢复时会重新构造，不会把上次的进程状态带过来。
        mcp=mcp or McpSettings(),
    )

    workflow = create_workflow(checkpointer=checkpointer)
    config = build_thread_config(
        run_id, recursion_limit=DEFAULT_RECURSION_LIMIT
    )

    inputs: dict[str, Any] | None = None
    if resume is None and not continue_from_checkpoint:
        inputs = {
            "run_id": run_id,
            "data_sources": data_sources,
            "analysis_focus": analysis_focus,
            "knowledge_sources": knowledge_sources or [],
            "retrieved_knowledge": [],
            "retrieved_knowledge_details": [],
            "execution_results": [],
            "failed_tasks": [],
            "visualization_results": [],
            "artifacts": [],
            "approvals": [],
            "completed_operations": {},
            # P3-A：预算从第一轮就写入 state，之后按增量累加。
            # 恢复时不重建它（见 utils/agent_loop.initialize）。
            "execution_budget": {
                "limits": _resolve_budget(budget).to_dict(),
                "usage": {
                    **new_usage(),
                    "started_at": time.time(),
                    "last_active_at": time.time(),
                },
            },
            "run_status": "running",
            "require_plan_approval": require_plan_approval,
            "require_report_approval": require_report_approval,
            "pending_llm_calls": list(runtime.unconfirmed_llm_calls),
            # 计划校验与重新规划的次数随 Checkpoint 走，恢复后不会重置
            "plan_attempts": 0,
            "plan_validation_errors": [],
            "approved_plan_fingerprint": None,
            # H4：任务级证据按 task_id 累积；多轮上下文与长期记忆在首轮写入，
            # 恢复时从 Checkpoint 读取，不重新检索也不需要重新注入。
            "task_knowledge": {},
            "knowledge_version": None,
            "conversation_context": conversation_context,
            "memories": list(memories or []),
            # H5：开关与限制在创建 Run 时写进 state。恢复沿用同一份，
            # 不会因为重新配置把正在进行的循环放宽。
            "enable_dynamic_loop": bool(enable_dynamic_loop),
            "dynamic_limits": limits_from(dynamic_limits),
            "loop_step": 0,
            "decision_history": [],
            "current_observation": None,
            "current_action": None,
            "action_reason": None,
            "last_observation_fingerprint": None,
            "action_counts": {},
            "plan_revision": 0,
            "runtime_replans": 0,
            "extra_retrievals": 0,
            "retrieval_focus": None,
            "skipped_tasks": [],
            "dynamic_finished": False,
            "termination_reason": None,
        }
    elif continue_from_checkpoint:
        # 续跑时也必须把账本里的未确认调用注入：Checkpoint 不会记得上次进程
        # 在"请求已发出、节点未提交"之间的状态，只有账本有这条证据。
        inputs = _resume_inputs(run_id, runtime)
    else:
        inputs = _resume_inputs(run_id, runtime)

    # 恢复时可以把人工决定交回被中断的节点（Command(resume=...)）
    # 恢复时可以把人工决定交回被中断的节点（Command(resume=...)）。
    # 若该决定是针对未确认模型调用的重发许可，还要写进 state：
    # 仅靠 Command 传参的话，这次 resume 内部再次中断后就丢失了。
    grant_retry = (
        isinstance(resume_value, dict)
        and str(resume_value.get("llm_decision", "")).lower() == "retry"
    )

    if continue_from_checkpoint or resume is not None:
        inputs = _resume_inputs(run_id, runtime, grant_retry=grant_retry)

    if resume_value is not None:
        resume = Command(resume=resume_value)

    # 人工对"未确认的模型调用"选择了不重发。这是受控终止：必须落成终态，
    # 否则图仍停在待执行节点上，Run 会一直显示 running 并反复询问同一个决定。
    # 在 invoke 之前处理，是因为节点内抛出的异常会被 LangGraph 记为"节点失败"，
    # 表现为又一次 interrupt，无法从结果里区分。
    if _declined_llm_replay(run_id, checkpointer, resume_value):
        return _mark_abandoned(run_id, config, checkpointer, runtime, run_id)

    try:
        # 运行期依赖用 ContextVar 绑定：它不参与 Checkpoint，也不跨 Run 串值。
        # 所有日志都在这个上下文内写：日志目录跟随 output_root，
        # 否则工作流级别的日志会落到进程当前目录，绕过隔离目录。
        run_started_at = time.time()
        with use_runtime(runtime):
            # 已完成/已失败的 Run 不再重跑：否则每次 resume 都会重新执行一遍节点，
            # 让追加型字段（如 execution_results）持续增长。终态是幂等的。
            if resume is not None or continue_from_checkpoint:
                persisted = (
                    read_run_state(run_id, checkpointer=checkpointer)
                    if checkpointer
                    else None
                )
                if persisted and persisted.get("run_status") in (
                    "completed",
                    "failed",
                ):
                    log_event(
                        run_id=run_id,
                        node="workflow",
                        event="resume_skipped_terminal",
                        status=persisted.get("run_status"),
                        details={"reason": "Run 已处于终态，跳过重复执行"},
                    )
                    return _read_full_state(run_id, checkpointer)

            log_event(
                run_id=run_id,
                node="workflow",
                event="run_started",
                status="running",
            )

            result = workflow.invoke(
                resume if resume is not None else inputs, config
            )

            log_event(
                run_id=run_id,
                node="workflow",
                event="run_completed",
                status=result.get("run_status", "unknown"),
                details={
                    "run_status": result.get("run_status"),
                    "analysis_status": result.get("analysis_status"),
                    "status_reason": result.get("status_reason"),
                    "analysis_tasks": len(result.get("analysis_plan", [])),
                    "failed_tasks": len(result.get("failed_tasks", [])),
                    "visualizations": len(result.get("visualization_plan", [])),
                    "artifacts": result.get("artifacts", []),
                    # 预算用量与整轮耗时：没有它们就无法回答
                    # "这次跑完花了多少额度、多久"，只能逐节点翻日志。
                    "budget": _budget_snapshot(result),
                },
                duration_ms=round((time.time() - run_started_at) * 1000, 2),
            )
    except LlmCallAbandoned as abandoned:
        # 人工选择不重发未确认的调用：这是受控终止，必须落成终态，
        # 否则 Run 会一直停留在 running。
        #
        # 图此时还停在待执行的节点上（`next` 非空），单靠"不再 invoke"无法
        # 表达终止；因此显式把终态写进 Checkpoint。这是一次纯状态写入，
        # 不推进任何节点，也就不会产生副作用。
        with use_runtime(runtime):
            log_event(
                run_id=run_id,
                node="workflow",
                event="llm_replay_declined",
                status="failed",
                details={"reason": str(abandoned)},
            )

            if checkpointer is not None:
                workflow = create_workflow(checkpointer=checkpointer)
                workflow.update_state(
                    config,
                    {
                        "run_status": "failed",
                        "analysis_status": "failed",
                        "status_reason": str(abandoned),
                    },
                )
                return _read_full_state(run_id, checkpointer)

        raise
    except Exception as error:
        with use_runtime(runtime):
            log_event(
                run_id=run_id,
                node="workflow",
                event="run_failed",
                status="failed",
                details={
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
        raise

    return result


def _declined_llm_replay(
    run_id: str,
    checkpointer: BaseCheckpointSaver | None,
    resume_value: Any,
) -> bool:
    """判断调用方是否对未确认的模型调用选择了"不重发"。

    只有两种情况算：
    - 显式传入非 retry 的决定（例如 `{"llm_decision": "skip"}`）；
    - 持久化状态里已经有 pending 的 llm_replay 决策且本次没有带 retry 许可
      （节点重放时若拿不到 retry 就会放弃，与其等它放弃，不如在这里明确终止）。
    """
    if checkpointer is None:
        return False

    decided_skip = (
        isinstance(resume_value, dict)
        and "llm_decision" in resume_value
        and str(resume_value.get("llm_decision", "")).lower() != "retry"
    )
    if decided_skip:
        return True

    state = read_run_state(run_id, checkpointer=checkpointer)
    if not state:
        return False

    pending = state.get("pending_approval") or {}
    if pending.get("kind") != "llm_replay":
        return False

    # 没有 retry 许可就视为放弃
    if isinstance(resume_value, dict):
        return str(resume_value.get("llm_decision", "")).lower() != "retry"

    return True


def _mark_abandoned(
    run_id: str,
    config: dict[str, Any],
    checkpointer: BaseCheckpointSaver,
    runtime: RunRuntime,
    label: str,
) -> dict[str, Any]:
    """把"未获准重发"写成终态。

    纯状态写入，不推进任何节点，因此不会产生副作用。
    """
    reason = "存在未确认的模型调用，且未获准重发"

    with use_runtime(runtime):
        log_event(
            run_id=run_id,
            node="workflow",
            event="llm_replay_declined",
            status="failed",
            details={"reason": reason},
        )

        workflow = create_workflow(checkpointer=checkpointer)
        workflow.update_state(
            config,
            {
                "run_status": "failed",
                "analysis_status": "failed",
                "status_reason": reason,
            },
        )

    return _read_full_state(run_id, checkpointer)


def _read_full_state(
    run_id: str, checkpointer: BaseCheckpointSaver | None
) -> dict[str, Any]:
    """直接返回 Checkpoint 中的完整状态，不推进执行。"""
    if checkpointer is None:
        return {}

    workflow = create_workflow(checkpointer=checkpointer)
    snapshot = workflow.get_state({"configurable": {"thread_id": run_id}})
    return dict(snapshot.values) if snapshot is not None else {}


def _require_factory(
    factory: ModelFactory | None, run_id: str
) -> ModelFactory:
    """缺少注入时明确失败。

    这里不回退到"自己构造真实客户端"：测试或恢复路径一旦忘记注入，
    就会意外打到真实上游。需要真实客户端时由调用方显式构造。
    """
    if factory is None:
        raise ValueError(
            "run_analysis 需要显式的 model_factory；"
            "请用 build_model_factory(run_id) 构造，测试中注入假模型"
        )
    return factory


# 对外公开的轻量状态字段。真实 DataFrame 不在其中——它不在 state 里，
# 而且调用方需要的是"这次运行到哪一步"，不是完整数据。
PUBLIC_STATE_FIELDS = (
    "run_id",
    "run_status",
    "analysis_status",
    "status_reason",
    "analysis_focus",
    "data_sources",
    "current_task_index",
    "current_chart_index",
    "failed_tasks",
    "artifacts",
    "report_path",
)


def read_run_state(
    run_id: str,
    *,
    checkpointer: BaseCheckpointSaver,
) -> dict[str, Any] | None:
    """读取持久化状态、下一个待执行节点与待审批请求。

    这是"重启后查询运行到哪了"的接口：它只依赖 Checkpoint，
    因此在新进程里同样可用。
    """
    workflow = create_workflow(checkpointer=checkpointer)
    snapshot = workflow.get_state({"configurable": {"thread_id": run_id}})

    if snapshot is None or not snapshot.values:
        return None

    values = snapshot.values
    state = {key: values.get(key) for key in PUBLIC_STATE_FIELDS}

    pending_approval: dict[str, Any] | None = None
    for task in snapshot.tasks or ():
        for item in getattr(task, "interrupts", ()) or ():
            value = getattr(item, "value", None)
            pending_approval = value if isinstance(value, dict) else {"value": value}
            break
        if pending_approval is not None:
            break

    next_nodes = list(snapshot.next or ())

    if pending_approval is not None:
        state["run_status"] = "waiting_approval"
    elif not next_nodes and state.get("run_status") not in ("completed", "failed"):
        state["run_status"] = "interrupted"

    state["pending_approval"] = pending_approval
    state["next_nodes"] = next_nodes
    state["checkpoint_id"] = (
        snapshot.config.get("configurable", {}).get("checkpoint_id")
        if snapshot.config
        else None
    )

    return state
