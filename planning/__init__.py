"""计划合法性与任务推进。

`validator` 判断一份 LLM 生成的计划能否执行，`orchestrator` 约束顺序执行时
的状态。两者都是确定性逻辑，不调用模型。
"""

from planning.orchestrator import (
    PlanChangedAfterApproval,
    completed_task_ids,
    current_task,
    dependencies_satisfied,
    verify_approved_plan,
)
from planning.validator import (
    MAX_PLAN_TASKS,
    PlanError,
    PlanValidation,
    plan_fingerprint,
    validate_plan,
)

__all__ = [
    "MAX_PLAN_TASKS",
    "PlanChangedAfterApproval",
    "PlanError",
    "PlanValidation",
    "completed_task_ids",
    "current_task",
    "dependencies_satisfied",
    "plan_fingerprint",
    "validate_plan",
    "verify_approved_plan",
]
