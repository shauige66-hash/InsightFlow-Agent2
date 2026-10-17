"""任务级上下文选择与构造。

上下文有两层控制，职责不同：

- **选什么**：按当前 Agent 与当前任务挑出相关表、相关历史。这一层在这里。
- **压多小**：字符预算与保留字段在 `utils/context_budget`。这里只调用它，
  不另建一套预算。

资源名就是快照的键：CSV 是文件主名（`sales`），Excel 与 SQLite 是
`文件主名.表名`（`sales.orders`）。`database_table` 是 SQLite 内部的物理表名，
**不是**资源名，两者不能混用。

样本行可以缩减，schema、行数与列数不能：模型写查询要靠列名与类型，压掉它们
只会换来一个"查不到的列"的错误。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from utils.context_budget import ContextReport, bound_fields, summarise_history

#: 单张表最多带多少行样本。整表数据本来就不进 Prompt（大结果走产物引用），
#: 这里只兜住"表很多、每张都带样本"的膨胀。
MAX_SAMPLE_ROWS = 5

#: 历史摘要最多保留多少条。与 `summarise_history` 的默认值一致。
MAX_HISTORY_ENTRIES = 8


class UnknownTableError(LookupError):
    """任务引用了当前 Run 不存在的资源。

    这种计划不该进入执行：让它落到工具里只会得到"表不存在"的运行时错误，
    或者更糟——被静默跳过，模型据此编出不存在的列。
    """

    def __init__(self, missing: list[str], available: list[str]) -> None:
        self.missing = tuple(missing)
        self.available = tuple(available)
        super().__init__(
            "任务引用了不存在的资源：" + ", ".join(missing)
            + "；当前 Run 可用：" + (", ".join(available) or "（无）")
        )


@dataclass(frozen=True)
class TableSelection:
    """一次任务级表选择的结果。"""

    tables: dict[str, Any]
    kept: tuple[str, ...] = ()
    excluded: tuple[str, ...] = ()
    samples_trimmed: int = 0


def ensure_known_resources(
    state: dict[str, Any],
    target_tables: list[str] | tuple[str, ...],
) -> tuple[str, ...]:
    """校验这些名字都是本次 Run 的资源；返回规范化后的名字。

    只校验不选择，供"还没到构造上下文、但已经要按表做决定"的地方使用
    （例如路由）。不在这里判断字段级可行性。
    """
    if not target_tables:
        raise ValueError("任务未指定目标表")

    names = [str(name) for name in target_tables]
    duplicated = sorted({name for name in names if names.count(name) > 1})
    if duplicated:
        raise ValueError("任务的目标表重复：" + ", ".join(duplicated))

    available = state.get("data_context") or {}
    missing = [name for name in names if name not in available]
    if missing:
        raise UnknownTableError(missing, sorted(available))

    snapshot = state.get("table_snapshot") or {}
    if snapshot:
        unauthorized = [name for name in names if name not in snapshot]
        if unauthorized:
            raise UnknownTableError(unauthorized, sorted(snapshot))

    return tuple(names)


def select_tables(
    state: dict[str, Any],
    target_tables: list[str] | tuple[str, ...],
    *,
    max_sample_rows: int = MAX_SAMPLE_ROWS,
) -> TableSelection:
    """按任务挑出相关表的上下文。

    资源名先过 `ensure_known_resources`：不存在的表直接拒绝，而不是交给下游
    的 `KeyError` 或者被静默跳过。
    """
    names = ensure_known_resources(state, target_tables)
    available = state.get("data_context") or {}

    tables: dict[str, Any] = {}
    trimmed = 0
    for name in names:
        tables[name], cut = _bound_samples(available[name], max_sample_rows)
        trimmed += cut

    return TableSelection(
        tables=tables,
        kept=tuple(names),
        excluded=tuple(sorted(set(available) - set(names))),
        samples_trimmed=trimmed,
    )


def _bound_samples(context: Any, limit: int) -> tuple[Any, int]:
    """把一张表的样本行压到 `limit` 行以内，其余结构原样保留。

    返回 `(上下文, 被裁掉的行数)`。上下文可能是 dict（schema + data），
    也可能是别的形状——不认识就原样返回，不猜。
    """
    if not isinstance(context, dict):
        return context, 0

    bounded = dict(context)
    data = bounded.get("data")
    cut = 0

    if isinstance(data, list):
        cut = max(0, len(data) - limit)
        if cut:
            bounded["data"] = data[:limit]
    elif isinstance(data, dict):
        for key in ("head", "tail"):
            rows = data.get(key)
            if isinstance(rows, list) and len(rows) > limit:
                cut += len(rows) - limit
                bounded["data"] = {**data, key: rows[:limit]}
                data = bounded["data"]

    return bounded, cut


def select_history(
    state: dict[str, Any],
    *,
    limit: int = MAX_HISTORY_ENTRIES,
) -> list[dict[str, Any]]:
    """按任务挑出必要的历史执行摘要。

    当前任务自己的尝试排在最后，这样"保留最近 N 条"截断时先丢的是别的任务，
    而不是正在修的那个任务的历史。输出仍然只含 `summarise_history` 的固定
    字段，不含结果数据本身。
    """
    results = list(state.get("execution_results") or [])
    if not results:
        return []

    plan = state.get("analysis_plan") or []
    index = state.get("current_task_index") or 0
    current_task_id = (
        plan[index].get("task_id") if 0 <= index < len(plan) else None
    )

    if current_task_id is None:
        ordered = results
    else:
        # 稳定排序：其它任务的记录保持原顺序在前，当前任务的记录在后
        ordered = [item for item in results if item.get("task_id") != current_task_id]
        ordered += [item for item in results if item.get("task_id") == current_task_id]

    return summarise_history(ordered, max_entries=limit)


def build_context(
    agent: str,
    fields: dict[str, Any],
    *,
    selection: TableSelection | None = None,
    history: list[dict[str, Any]] | None = None,
    task_id: str | None = None,
    budget: int | None = None,
) -> tuple[dict[str, Any], ContextReport]:
    """把一次 Agent 的输入压进预算，并产出可观测的度量。

    这里不重新拼 Prompt 字段：字段名与结构由各 Agent 决定，压多少由
    `utils.context_budget` 决定。把两者拆开，是为了让"选中了什么"和
    "压了多少"各自都能单独测试。
    """
    bounded, report = bound_fields(
        agent,
        fields,
        budget=budget,
        task_id=task_id,
        kept_tables=list(selection.kept) if selection else None,
        excluded_tables=list(selection.excluded) if selection else None,
        history_entries=len(history) if history is not None else None,
    )
    return bounded, report


__all__ = [
    "MAX_HISTORY_ENTRIES",
    "MAX_SAMPLE_ROWS",
    "TableSelection",
    "UnknownTableError",
    "build_context",
    "ensure_known_resources",
    "select_history",
    "select_tables",
]
