"""动作策略：把模型的动作建议收敛成"当前条件下确实可以执行的动作"。

模型可以建议，但不能决定。这里不调用任何模型，只按观察、计数和预算判断：
这个动作允许吗？不允许时退到哪个安全动作？

为什么必须有这一层：一个"只要失败就 repair"的模型在沙箱不可用、SQL 权限被拒
或凭据错误时会要求无限重试。这些失败重试不会好转，只会烧掉预算并掩盖原因。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from sandbox.protocol import ENVIRONMENT_ERROR_KINDS

CONTINUE = "continue"
FINISH = "finish"
REPAIR = "repair"
REFINE = "refine"
RETRIEVE_MORE = "retrieve_more"
REPLAN_REMAINING = "replan_remaining"
SKIP_TASK = "skip_task"

ACTIONS = (
    CONTINUE,
    FINISH,
    REPAIR,
    REFINE,
    RETRIEVE_MORE,
    REPLAN_REMAINING,
    SKIP_TASK,
)

#: 需要"失败或不合格"才能成立的动作
_REMEDIAL = (REPAIR, REFINE, RETRIEVE_MORE, REPLAN_REMAINING, SKIP_TASK)

#: 重试不会好转的错误：权限、授权、环境、预算
_TERMINAL_KINDS = frozenset(
    {
        *ENVIRONMENT_ERROR_KINDS,
        "sql_permission",
        "sql_authorization",
        "authorization_denied",
        "forbidden",
        "permission_denied",
        "budget_exceeded",
    }
)

DEFAULT_LIMITS: dict[str, int] = {
    # 一次运行最多做几次动态决策（含首次）
    "max_decision_steps": 8,
    # 运行期重新规划计划（不含  的初始计划生成）
    "max_runtime_replans": 2,
    # 每个任务最多补充几次检索
    "max_extra_retrievals": 2,
    # 同一任务、同一动作最多做几次（防止观察不变、动作重复）
    "max_repeated_action_per_task": 2,
}


@dataclass(frozen=True)
class ActionDecision:
    action: str
    reason: str
    proposed: str | None = None
    forced: bool = False
    observation_fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def limits_from(value: dict[str, Any] | None) -> dict[str, int]:
    """合并默认限制与调用方给的值，只接受已知键和正整数。"""
    limits = dict(DEFAULT_LIMITS)
    for key, raw in (value or {}).items():
        if key not in limits:
            continue
        try:
            parsed = int(raw)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            limits[key] = parsed
    return limits


def _budget_exhausted(observation: Any) -> bool:
    remaining = observation.remaining_budget or {}
    llm = remaining.get("llm_calls")
    return isinstance(llm, int) and llm <= 0


def _task_settled(observation: Any) -> bool:
    return observation.validation_passed is True and observation.execution_success is True


def _has_problem(observation: Any) -> bool:
    return (
        observation.execution_success is False
        or observation.validation_passed is False
        or (observation.execution_success is True and observation.validation_passed is None)
    )


def _evidence_needed(observation: Any, state: dict[str, Any]) -> bool:
    if not (state.get("knowledge_sources") or []):
        return False
    if observation.evidence_sufficient:
        return False
    return observation.extra_retrievals < int(
        limits_from(state.get("dynamic_limits"))["max_extra_retrievals"]
    )


def repeated_too_often(observation: Any, action: str, limits: dict[str, int]) -> bool:
    return observation.repeated_actions.get(action, 0) >= int(
        limits["max_repeated_action_per_task"]
    )


def _deny(action: str, reason: str) -> tuple[str | None, str]:
    return action, reason


def validate_action(
    observation: Any,
    proposed: Any,
    *,
    state: dict[str, Any],
    limits: dict[str, int],
) -> ActionDecision:
    """校验模型建议的动作，必要时降级为安全动作。"""
    limits = limits or DEFAULT_LIMITS
    raw = proposed if isinstance(proposed, dict) else {}
    suggested = str(raw.get("action") or "").strip().lower()
    model_reason = str(raw.get("reason") or "").strip()[:300]
    fingerprint = observation.fingerprint()

    def decide(action: str, reason: str, *, forced: bool = False) -> ActionDecision:
        return ActionDecision(
            action=action,
            reason=reason,
            proposed=suggested or None,
            forced=forced,
            observation_fingerprint=fingerprint,
        )

    # 1. 步数上限：不再做决策，直接收尾
    if observation.loop_step >= int(limits["max_decision_steps"]):
        return decide(
            FINISH,
            f"动态决策达到上限 {limits['max_decision_steps']} 步",
            forced=True,
        )

    # 2. 预算耗尽：继续决策只会再发一次必然被拒的请求
    if _budget_exhausted(observation):
        return decide(FINISH, "模型调用预算已耗尽", forced=True)

    # 3. 模型给的不是已知动作
    if suggested not in ACTIONS:
        return decide(
            FINISH,
            f"未知动作 {suggested!r}，安全收尾",
            forced=True,
        )

    settled = _task_settled(observation)
    terminal = observation.error_kind in _TERMINAL_KINDS

    # 4. 不可重试的错误：只允许跳过或收尾
    if terminal and suggested in (REPAIR, REFINE, RETRIEVE_MORE, CONTINUE):
        return decide(
            SKIP_TASK,
            f"{observation.error_kind} 不可通过重试解决，跳过当前任务",
            forced=True,
        )

    # 5. 逐个动作的成立条件
    if suggested == CONTINUE:
        if settled:
            return decide(CONTINUE, model_reason or "任务已通过校验")
        return decide(
            SKIP_TASK,
            "任务未通过校验，不能直接进入下一任务",
            forced=True,
        )

    if suggested == REPAIR:
        if observation.execution_success is not False:
            return decide(
                CONTINUE if settled else SKIP_TASK,
                "没有可修复的执行错误",
                forced=True,
            )
        if repeated_too_often(observation, REPAIR, limits):
            return decide(
                SKIP_TASK,
                f"同一任务已修复 {limits['max_repeated_action_per_task']} 次仍失败",
                forced=True,
            )
        return decide(REPAIR, model_reason or "执行错误可修复")

    if suggested == REFINE:
        if observation.validation_passed is not False:
            return decide(
                CONTINUE if settled else SKIP_TASK,
                "没有质量反馈可改进",
                forced=True,
            )
        if repeated_too_often(observation, REFINE, limits):
            return decide(
                SKIP_TASK,
                f"同一任务已精修 {limits['max_repeated_action_per_task']} 次仍不通过",
                forced=True,
            )
        return decide(REFINE, model_reason or "质量反馈需要改进")

    if suggested == RETRIEVE_MORE:
        if not (state.get("knowledge_sources") or []):
            return decide(
                REFINE if observation.validation_passed is False else SKIP_TASK,
                "本次运行没有知识源，无法补充检索",
                forced=True,
            )
        if observation.extra_retrievals >= int(limits["max_extra_retrievals"]):
            return decide(
                REFINE if observation.validation_passed is False else SKIP_TASK,
                f"补充检索达到上限 {limits['max_extra_retrievals']} 次",
                forced=True,
            )
        if repeated_too_often(observation, RETRIEVE_MORE, limits):
            return decide(
                SKIP_TASK,
                "同样的观察下已补充检索多次，不再重复",
                forced=True,
            )
        if settled:
            return decide(CONTINUE, "任务已通过，无需补充检索", forced=True)
        return decide(RETRIEVE_MORE, model_reason or "证据不足，补充检索")

    if suggested == REPLAN_REMAINING:
        if observation.runtime_replans >= int(limits["max_runtime_replans"]):
            return decide(
                SKIP_TASK if _has_problem(observation) else FINISH,
                f"运行期重新规划达到上限 {limits['max_runtime_replans']} 次",
                forced=True,
            )
        if settled:
            return decide(CONTINUE, "任务已通过，不需要重新规划", forced=True)
        return decide(REPLAN_REMAINING, model_reason or "剩余计划无法完成")

    if suggested == SKIP_TASK:
        if settled:
            return decide(CONTINUE, "任务已完成，不需要跳过", forced=True)
        if observation.error_summary is None and observation.validation_passed is not False:
            return decide(
                FINISH,
                "没有可跳过的问题，安全收尾",
                forced=True,
            )
        return decide(SKIP_TASK, model_reason or "任务无法安全完成")

    # FINISH：任何时候都允许，但要检查是否还有完成不了的工作
    return decide(FINISH, model_reason or "结束分析")


__all__ = [
    "ACTIONS",
    "CONTINUE",
    "DEFAULT_LIMITS",
    "FINISH",
    "REFINE",
    "REPAIR",
    "REPLAN_REMAINING",
    "RETRIEVE_MORE",
    "SKIP_TASK",
    "ActionDecision",
    "limits_from",
    "repeated_too_often",
    "validate_action",
]
