"""顺序任务推进的状态约束。

执行仍按 `analysis_plan` 的顺序逐个进行，这里只提供三件事：安全地取当前任务、
校验依赖是否满足、以及确认"现在执行的计划就是被审批的那一份"。

不实现通用 DAG 调度：依赖只允许指向更早的任务，顺序执行天然满足。
"""

from __future__ import annotations

from typing import Any

from planning.validator import plan_fingerprint
from state import AnalysisTask


class PlanChangedAfterApproval(RuntimeError):
    """审批通过后计划被换成了另一份。"""


def current_task(
    plan: list[AnalysisTask] | None, index: int | None
) -> AnalysisTask | None:
    """当前任务；索引越界时返回 None，由调用方决定如何收敛。

    越界只可能来自非法计划或状态损坏，不该变成 IndexError 让整次运行崩掉。
    """
    if not plan or index is None:
        return None
    if index < 0 or index >= len(plan):
        return None
    return plan[index]


def dependencies_satisfied(
    task: AnalysisTask,
    *,
    succeeded: set[str],
) -> bool:
    """任务声明的依赖是否都成功完成。

    没有 `depends_on` 的旧计划直接返回 True，保持向后兼容。
    """
    dependency_ids = task.get("depends_on") or []
    return all(str(item) in succeeded for item in dependency_ids)


def completed_task_ids(execution_results: list[dict[str, Any]] | None) -> set[str]:
    """成功完成的任务 ID 集合。"""
    return {
        str(item.get("task_id"))
        for item in (execution_results or [])
        if item.get("success")
    }


def verify_approved_plan(state: dict[str, Any]) -> None:
    """执行前确认计划与获批的那一份一致。

    审批之后如果有人替换了 `analysis_plan`，这里必须失败：人工批准的是具体
    内容，不是"随便一份计划"。
    """
    approved = state.get("approved_plan_fingerprint")
    if not approved:
        return
    current = plan_fingerprint(state.get("analysis_plan") or [])
    if current != approved:
        raise PlanChangedAfterApproval(
            "审批通过的计划与当前计划不一致，拒绝执行"
        )


__all__ = [
    "PlanChangedAfterApproval",
    "completed_task_ids",
    "current_task",
    "dependencies_satisfied",
    "verify_approved_plan",
]
