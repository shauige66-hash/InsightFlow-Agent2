"""把校验过的动作翻译成 state 增量与图上的下一个节点。

控制器不判断"该做什么"（那是策略与模型的职责），只负责记账：
步数、历史、动作计数、终止原因、跳过的任务。所有字段都可 JSON 序列化，
因为整份 state 会进 Checkpoint。
"""

from __future__ import annotations

from typing import Any

from agent_loop.policy import (
    CONTINUE,
    FINISH,
    REFINE,
    REPAIR,
    REPLAN_REMAINING,
    RETRIEVE_MORE,
    SKIP_TASK,
    ActionDecision,
)

ROUTE_BY_ACTION: dict[str, str] = {
    CONTINUE: "advance",
    FINISH: "complete",
    REPAIR: "repair",
    REFINE: "refine",
    RETRIEVE_MORE: "execution",
    REPLAN_REMAINING: "replan",
    SKIP_TASK: "advance",
}

MAX_HISTORY = 20


def route_for(action: str | None) -> str:
    return ROUTE_BY_ACTION.get(str(action or ""), "complete")


def _record_action(state: dict[str, Any], task_id: str, action: str) -> dict[str, Any]:
    counts = {
        key: dict(value or {})
        for key, value in (state.get("action_counts") or {}).items()
    }
    per_task = counts.setdefault(str(task_id), {})
    per_task[action] = int(per_task.get(action, 0)) + 1
    return counts


def _skip_entry(observation: Any, decision: ActionDecision) -> dict[str, Any]:
    stage = "validation" if observation.validation_passed is False else "execution"
    return {
        "task_id": observation.task_id,
        "stage": "skipped",
        "reason": decision.reason or "任务无法安全完成",
        "error_kind": observation.error_kind,
        "code": None,
        "retry_count": observation.retry_count,
        "refine_count": observation.refine_count,
        "inner_stage": stage,
    }


def decision_updates(
    state: dict[str, Any],
    observation: Any,
    decision: ActionDecision,
) -> dict[str, Any]:
    """决策节点要写回 state 的内容。"""
    step = int(observation.loop_step) + 1
    history = list(state.get("decision_history") or [])
    history.append(
        {
            "step": step,
            "task_id": observation.task_id,
            "action": decision.action,
            "reason": decision.reason,
            "proposed": decision.proposed,
            "forced": decision.forced,
            "observation": decision.observation_fingerprint,
            "plan_revision": observation.plan_revision,
        }
    )

    updates: dict[str, Any] = {
        "loop_step": step,
        "decision_history": history[-MAX_HISTORY:],
        "current_action": decision.action,
        "action_reason": decision.reason,
        "current_observation": observation.to_dict(),
        "last_observation_fingerprint": decision.observation_fingerprint,
        "action_counts": _record_action(state, observation.task_id, decision.action),
    }

    if decision.action == RETRIEVE_MORE:
        updates["extra_retrievals"] = int(state.get("extra_retrievals") or 0) + 1
        # 用校验意见或执行错误作为补充检索的依据；两者都没有时退回任务目标
        updates["retrieval_focus"] = (
            observation.critic_feedback
            or observation.error_summary
            or observation.task_goal
        )
    elif decision.action == REPAIR:
        # 修复次数由 repair 节点负责累加（它知道这次修复是否真的发生了）
        updates["retrieval_focus"] = None
    elif decision.action == REFINE:
        updates["retrieval_focus"] = None

    if decision.action == SKIP_TASK:
        skipped = list(state.get("skipped_tasks") or [])
        skipped.append(_skip_entry(observation, decision))
        updates["skipped_tasks"] = skipped
        # 计入失败任务，让最终状态是 partial 而不是 completed
        failed = list(state.get("failed_tasks") or [])
        failed.append(_skip_entry(observation, decision))
        updates["failed_tasks"] = failed
        updates["termination_reason"] = (
            f"任务 {observation.task_id} 被跳过：{decision.reason}"
        )
    elif decision.action == FINISH:
        # 提前结束时如果当前任务并没有通过，它就是一个未完成的任务：
        # 只有把它记进 failed_tasks，收尾状态才会是 partial。否则"模型决定
        # 不做了"会被下游当成"分析成功完成"。
        if (
            observation.validation_passed is not True
            or observation.execution_success is False
        ):
            failed = list(state.get("failed_tasks") or [])
            if not any(item.get("task_id") == observation.task_id for item in failed):
                failed.append(
                    {
                        "task_id": observation.task_id,
                        "stage": (
                            "validation"
                            if observation.validation_passed is False
                            else "execution"
                        ),
                        "reason": decision.reason or "分析在任务通过前结束",
                        "error_kind": observation.error_kind,
                        "code": None,
                        "retry_count": observation.retry_count,
                        "refine_count": observation.refine_count,
                    }
                )
                updates["failed_tasks"] = failed
        updates["analysis_status"] = _finish_status(state, observation)
        updates["termination_reason"] = decision.reason or "分析结束"
        updates["dynamic_finished"] = True

        # 提前结束意味着后面还有任务没执行。只把状态写成 partial 不够：
        # 收尾节点会按 failed_tasks 重新计算状态，未执行的任务必须留痕，
        # 否则"少做了几项"仍会被报成 completed。
        plan = state.get("analysis_plan") or []
        index = int(state.get("current_task_index") or 0)
        pending = plan[index + 1 :]
        if pending:
            failed = list(updates.get("failed_tasks") or state.get("failed_tasks") or [])
            known = {item.get("task_id") for item in failed}
            for item in pending:
                task_id = item.get("task_id")
                if task_id in known:
                    continue
                failed.append(
                    {
                        "task_id": task_id,
                        "stage": "skipped",
                        "reason": decision.reason or "分析提前结束，任务未执行",
                        "error_kind": None,
                        "code": None,
                        "retry_count": 0,
                        "refine_count": 0,
                    }
                )
            updates["failed_tasks"] = failed

    return updates


def _finish_status(state: dict[str, Any], observation: Any) -> str:
    """结束时的分析状态：当前任务没通过、或还有任务没执行，都是 partial。"""
    plan = state.get("analysis_plan") or []
    index = int(state.get("current_task_index") or 0)

    current_ok = (
        observation.validation_passed is True
        and observation.execution_success is not False
    )
    # 后面还有任务没跑：即使当前任务通过，也不能算"分析完成"
    remaining = index + 1 < len(plan)

    if not current_ok or remaining or (state.get("failed_tasks") or []):
        return "partial"
    return "completed"


__all__ = ["MAX_HISTORY", "ROUTE_BY_ACTION", "decision_updates", "route_for"]
