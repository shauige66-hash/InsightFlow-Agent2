"""计划校验：LLM 给出的分析计划是否可执行，由程序判断。

这里不调用模型，也不猜自然语言。校验只看结构化字段：任务形状、任务 ID、
目标表是否是本次 Run 的资源、这些表的来源类型能否被现有路由支持。
"这次分析是否有业务意义"不在校验范围内——那需要业务规则，不是这里能证明的。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from state import AnalysisTask

#: 单个 Run 的任务数上限。计划再长也不会一次跑完，纯粹的无界展开没有价值。
MAX_PLAN_TASKS = 20

REQUIRED_FIELDS = ("task_id", "description", "target_tables", "expected_output")

# 校验错误分类。名字会进日志与重新规划的反馈，保持稳定、可枚举。
NOT_A_LIST = "not_a_list"
TOO_MANY_TASKS = "too_many_tasks"
INVALID_TASK = "invalid_task"
MISSING_FIELD = "missing_field"
EMPTY_FIELD = "empty_field"
DUPLICATE_TASK_ID = "duplicate_task_id"
EMPTY_TARGET_TABLES = "empty_target_tables"
DUPLICATE_TABLE = "duplicate_table"
UNKNOWN_TABLE = "unknown_table"
UNSUPPORTED_SOURCE_MIX = "unsupported_source_mix"
UNKNOWN_DEPENDENCY = "unknown_dependency"
DEPENDENCY_ORDER = "dependency_order"
CYCLIC_DEPENDENCY = "cyclic_dependency"


@dataclass(frozen=True)
class PlanError:
    """一条校验错误。

    `task_id` / `index` 用于定位；反馈给 Planner 时只用 `message`。
    """

    category: str
    message: str
    task_id: str | None = None
    index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "message": self.message,
            "task_id": self.task_id,
            "index": self.index,
        }


@dataclass(frozen=True)
class PlanValidation:
    """一次计划校验的结果。"""

    tasks: list[AnalysisTask] = field(default_factory=list)
    errors: list[PlanError] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def is_empty(self) -> bool:
        """空计划是合法结果：数据不足以支撑任何分析任务。"""
        return self.ok and not self.tasks

    @property
    def categories(self) -> list[str]:
        return sorted({error.category for error in self.errors})

    def feedback(self) -> list[str]:
        """给 Planner 的重新规划依据：具体的、可修正的问题。"""
        return [error.message for error in self.errors]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "task_count": len(self.tasks),
            "categories": self.categories,
            "errors": [error.to_dict() for error in self.errors],
        }


def validate_plan(
    plan: Any,
    *,
    authorized_tables: set[str] | frozenset[str] | None = None,
    source_types: dict[str, str] | None = None,
) -> PlanValidation:
    """校验一份分析计划。

    `authorized_tables` 是本次 Run 的真实资源名集合（快照的键）；
    `source_types` 是资源名到来源类型的映射，用来判断路由是否能支持这个组合。
    两者都不给时只做结构校验。
    """
    if not isinstance(plan, list):
        return PlanValidation(
            errors=[PlanError(NOT_A_LIST, "计划必须是任务列表")]
        )

    if len(plan) > MAX_PLAN_TASKS:
        return PlanValidation(
            errors=[
                PlanError(
                    TOO_MANY_TASKS,
                    f"任务数 {len(plan)} 超过上限 {MAX_PLAN_TASKS}",
                )
            ]
        )

    errors: list[PlanError] = []
    tasks: list[AnalysisTask] = []
    seen_ids: dict[str, int] = {}

    for index, raw in enumerate(plan):
        task, task_errors = _validate_task(
            raw,
            index=index,
            authorized_tables=authorized_tables,
            source_types=source_types,
            seen_ids=seen_ids,
        )
        errors.extend(task_errors)
        if task is not None:
            tasks.append(task)

    if errors:
        return PlanValidation(errors=errors)

    errors.extend(_validate_dependencies(tasks))
    if errors:
        return PlanValidation(errors=errors)

    return PlanValidation(tasks=tasks)


def _validate_task(
    raw: Any,
    *,
    index: int,
    authorized_tables: set[str] | frozenset[str] | None,
    source_types: dict[str, str] | None,
    seen_ids: dict[str, int],
) -> tuple[AnalysisTask | None, list[PlanError]]:
    if not isinstance(raw, dict):
        return None, [PlanError(INVALID_TASK, f"第 {index} 项不是任务对象", index=index)]

    missing = [name for name in REQUIRED_FIELDS if name not in raw]
    if missing:
        return None, [
            PlanError(
                MISSING_FIELD,
                f"第 {index} 项缺少字段：{', '.join(missing)}",
                index=index,
            )
        ]

    task_id = raw["task_id"]
    if not isinstance(task_id, str) or not task_id.strip():
        return None, [
            PlanError(EMPTY_FIELD, f"第 {index} 项 task_id 为空", index=index)
        ]
    task_id = task_id.strip()

    if task_id in seen_ids:
        return None, [
            PlanError(
                DUPLICATE_TASK_ID,
                f"task_id 重复：{task_id}",
                task_id=task_id,
                index=index,
            )
        ]
    seen_ids[task_id] = index

    errors: list[PlanError] = []

    for name in ("description", "expected_output"):
        value = raw[name]
        if not isinstance(value, str) or not value.strip():
            errors.append(
                PlanError(
                    EMPTY_FIELD, f"{task_id} 的 {name} 为空", task_id=task_id, index=index
                )
            )

    target_tables = raw["target_tables"]
    if not isinstance(target_tables, list) or not target_tables:
        errors.append(
            PlanError(
                EMPTY_TARGET_TABLES,
                f"{task_id} 未指定目标表",
                task_id=task_id,
                index=index,
            )
        )
        return None, errors

    names = [str(name).strip() for name in target_tables]
    if any(not name for name in names):
        errors.append(
            PlanError(
                EMPTY_FIELD, f"{task_id} 的目标表有空名字", task_id=task_id, index=index
            )
        )
        return None, errors

    duplicated = sorted({name for name in names if names.count(name) > 1})
    if duplicated:
        errors.append(
            PlanError(
                DUPLICATE_TABLE,
                f"{task_id} 的目标表重复：{', '.join(duplicated)}",
                task_id=task_id,
                index=index,
            )
        )

    if authorized_tables is not None:
        unknown = sorted(set(names) - set(authorized_tables))
        if unknown:
            errors.append(
                PlanError(
                    UNKNOWN_TABLE,
                    f"{task_id} 引用了不存在的资源：{', '.join(unknown)}；"
                    "target_tables 必须用 data_context 的真实资源名",
                    task_id=task_id,
                    index=index,
                )
            )
        else:
            mix = _unsupported_mix(names, source_types)
            if mix is not None:
                errors.append(
                    PlanError(
                        UNSUPPORTED_SOURCE_MIX,
                        f"{task_id} 混合了无法用同一个工具执行的来源：{mix}",
                        task_id=task_id,
                        index=index,
                    )
                )

    if errors:
        return None, errors

    normalized: dict[str, Any] = dict(raw)
    normalized["task_id"] = task_id
    normalized["description"] = str(raw["description"]).strip()
    normalized["expected_output"] = str(raw["expected_output"]).strip()
    normalized["target_tables"] = names
    return normalized, errors  # type: ignore[return-value]


def _unsupported_mix(
    names: list[str], source_types: dict[str, str] | None
) -> str | None:
    """判断这个任务能否被现有路由落到单一工具。

    路由的规则是"全是 DataFrame 走 python、全是 sqlite 走 sql"。
    csv 与 excel 都会被读成 DataFrame，因此只有"sqlite 混入其它来源"才不可执行。
    """
    if not source_types:
        return None

    types = {source_types[name] for name in names if name in source_types}
    if "sqlite" in types and len(types) > 1:
        return ", ".join(
            f"{name}({source_types.get(name, '?')})" for name in names
        )
    return None


def _validate_dependencies(tasks: list[AnalysisTask]) -> list[PlanError]:
    """校验可选的 `depends_on`。

    当前执行是顺序的，因此依赖只能是"指向更早的任务"。这比通用 DAG 弱，
    但它可以只用现有顺序执行满足，不需要额外的调度器。
    """
    position = {task["task_id"]: index for index, task in enumerate(tasks)}
    errors: list[PlanError] = []

    for index, task in enumerate(tasks):
        depends_on = task.get("depends_on")
        if depends_on is None:
            continue
        if not isinstance(depends_on, list):
            errors.append(
                PlanError(
                    INVALID_TASK,
                    f"{task['task_id']} 的 depends_on 必须是列表",
                    task_id=task["task_id"],
                    index=index,
                )
            )
            continue

        for dependency in depends_on:
            dependency = str(dependency)
            if dependency == task["task_id"]:
                errors.append(
                    PlanError(
                        CYCLIC_DEPENDENCY,
                        f"{task['task_id']} 依赖自己",
                        task_id=task["task_id"],
                        index=index,
                    )
                )
                continue
            if dependency not in position:
                errors.append(
                    PlanError(
                        UNKNOWN_DEPENDENCY,
                        f"{task['task_id']} 依赖不存在的任务：{dependency}",
                        task_id=task["task_id"],
                        index=index,
                    )
                )
                continue
            if position[dependency] > index:
                errors.append(
                    PlanError(
                        DEPENDENCY_ORDER,
                        f"{task['task_id']} 依赖了排在其后的任务 {dependency}；"
                        "顺序执行只支持依赖更早的任务",
                        task_id=task["task_id"],
                        index=index,
                    )
                )

    return errors


def plan_fingerprint(tasks: list[AnalysisTask]) -> str:
    """计划的内容指纹，用于确认"审批的是这份计划"。

    只取影响执行的字段，排序固定，因此同一份计划在恢复后得到同一个值。
    """
    canonical = [
        {
            "task_id": task.get("task_id"),
            "description": task.get("description"),
            "target_tables": list(task.get("target_tables") or []),
            "expected_output": task.get("expected_output"),
            "depends_on": list(task.get("depends_on") or []),
        }
        for task in tasks
    ]
    payload = json.dumps(canonical, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "MAX_PLAN_TASKS",
    "PlanError",
    "PlanValidation",
    "plan_fingerprint",
    "validate_plan",
]
