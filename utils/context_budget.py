"""上下文预算与选择。

## 解决的三个实际问题

1. **没有预算，也没有度量。** 各节点的 Prompt 由 `json.dumps(..., indent=2)`
   直接拼出来，大小只受上游数据规模影响。P2-B3 给了报告 Agent **有界预览**
   （`build_preview`：20 条 / 8 KiB），但那是"报告那一条路径"的约束，
   不是按节点的预算，也没有记录"这次用了多少"。

2. **不是所有字段都该一视同仁。** 执行结果里既有几十行的样本数据，
   也有决定下游判断的错误信息与审批结论。简单按总长度截断会先丢掉
   错误信息——那恰恰是路由与人工复核最需要的东西。

3. **没有统一的观察点。** 想知道"这次运行的 Prompt 有多大"只能逐节点加打印。

## 设计取舍

- **预算是字符数，不是 Token 数。** 本仓库没有与所用模型匹配的 tokenizer，
  凭空算 Token 只会给出看似精确的错数。因此对外只承诺**字符预算**，
  另外给出一个明确标注为估算的 Token 参考值。
- **关键字段永不截断。** `error` / `error_kind` / `status_reason` /
  `approvals` / `pending_approval` / `task_id` 等属于"影响正确性与恢复语义"
  的字段，无论预算多紧都原样保留。
- **确定性。** 同样的输入必然产生同样的输出：不依赖 dict 迭代顺序之外的
  随机性，也不调用模型去"压缩"（那会引入新的付费调用与不确定性）。
"""

from __future__ import annotations

import contextvars
import json
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# 上下文度量的传递：按执行上下文隔离
# ---------------------------------------------------------------------------

#: 最近一次上下文构造的度量，**按执行上下文隔离**。
#:
#: 早期实现把它放在每个 Agent 模块的模块级字典里。`RunCoordinator` 用
#: `ThreadPoolExecutor` 并发执行多个 Run，于是两个 Run 同时调用同一个 Agent 时，
#: 后写的那个会覆盖先写的——结果是 **A Run 的上下文度量被记到 B Run 的日志里**。
#: 模块级可变状态在并发下没有归属，换成 `ContextVar` 就没有这个问题：
#: 每个执行上下文看到自己 set 的那份。注意线程池**不会**把提交线程的上下文
#: 复制进工作线程（工作线程看到的是 `default`），隔离性正是由此而来。
#:
#: 结构是 `{agent 名: 度量}`，因为同一次执行里可能调用多个 Agent。
_context_reports: contextvars.ContextVar[dict[str, dict[str, Any]]] = (
    contextvars.ContextVar("insightflow_context_reports", default={})
)


def publish_context_report(node: str, report: dict[str, Any]) -> None:
    """记录某个 Agent 本次的上下文度量。

    整体替换而不是 `clear()` + `update()`：后者是两步操作，中间可能被
    另一个线程观察到"空度量"。
    """
    current = dict(_context_reports.get())
    current[node] = dict(report)
    _context_reports.set(current)


def context_report_of(node: str) -> dict[str, Any]:
    """取回某个 Agent **本次执行**的上下文度量；没有则返回空字典。"""
    return dict(_context_reports.get().get(node) or {})


def clear_context_reports() -> None:
    """清空本执行上下文的度量（测试用）。"""
    _context_reports.set({})

# 估算比例：英文字符约 4 个/token，中文约 1 个/token。
# 这里取 3.5 作为两种文本混合时的粗略折算，**明确标注为估算**。
CHARS_PER_TOKEN_ESTIMATE = 3.5

#: 各节点的上下文预算（字符）。
#:
#: 取值依据：既有 Prompt 模板本身约 1~3 KiB；正常执行的样本与历史结果
#: 在几 KiB 量级。预算留出宽裕余量，只在异常膨胀（例如上千条历史结果、
#: 超长错误堆栈）时才生效，因此正常路径完全不受影响。
DEFAULT_CONTEXT_BUDGETS: dict[str, int] = {
    "planner": 24_000,
    "executor": 32_000,
    "repair": 32_000,
    "refine": 32_000,
    "critic": 24_000,
    "visualizer": 20_000,
    "plotter": 20_000,
    "plot_repair": 20_000,
    "reporter": 48_000,
}

#: 兜底预算：未列出的节点用这个值。
FALLBACK_CONTEXT_BUDGET = 32_000

#: 这些键承载**正确性与恢复语义**，任何情况下都不截断。
PRESERVE_KEYS: frozenset[str] = frozenset(
    {
        "task_id",
        "chart_id",
        "run_id",
        "error",
        "error_kind",
        "status_reason",
        "reason",
        "success",
        "analysis_status",
        "run_status",
        "approvals",
        "pending_approval",
        "approval_id",
        "decision",
        "kind",
        "node",
        "retry_count",
        "refine_count",
        "plot_retry_count",
        "truncated",
        "row_count",
        "columns",
        "preview",
        "__kind__",
    }
)

#: 截断标记。放在被截断的值里，让模型知道"这不是全部"。
TRUNCATION_MARKER = "...[已按上下文预算截断]"


def estimate_tokens(text: str) -> int:
    """**估算** Token 数。

    本仓库没有与所用模型匹配的 tokenizer，因此这里只能按字符比例折算。
    返回值是估算值，**不得**当作计费 Token 使用，也不得用于对外声称
    "节省了多少 Token"。
    """
    if not text:
        return 0
    return max(1, int(len(text) / CHARS_PER_TOKEN_ESTIMATE))


def budget_for(node: str) -> int:
    return DEFAULT_CONTEXT_BUDGETS.get(node, FALLBACK_CONTEXT_BUDGET)


@dataclass
class ContextReport:
    """一次上下文构造的度量结果。

    字符规模之外还记录"选了哪些表、排除了哪些表、带了几条历史"，
    否则无法回答"这次上下文为什么这么大"。

    `kept_tables` / `excluded_tables` 由调用方按任务级选择填入；
    直接调用 `bound_fields` 的路径留空，不影响既有用法。
    """

    node: str
    budget_chars: int
    original_chars: int = 0
    final_chars: int = 0
    truncated_fields: list[str] = field(default_factory=list)
    task_id: str | None = None
    kept_tables: list[str] = field(default_factory=list)
    excluded_tables: list[str] = field(default_factory=list)
    history_entries: int | None = None
    # 知识证据、长期记忆与缓存命中的可观测字段。它们只描述“注入了多少”，
    # 不携带内容本身，因此可以安全进日志。
    knowledge_chunks: int = 0
    memory_entries: int = 0
    knowledge_cache_hit: bool = False

    @property
    def truncated(self) -> bool:
        return bool(self.truncated_fields)

    @property
    def over_budget(self) -> bool:
        """保留字段本身就超预算。

        这类超预算不是 bug：`NODE_PRESERVE_FIELDS` 里的字段（数据 schema、
        任务约束、错误反馈）宁可超也不压掉。记录下来，避免把它当成
        "预算没生效"。
        """
        return self.final_chars > self.budget_chars

    @property
    def estimated_tokens(self) -> int:
        return estimate_tokens("x" * self.final_chars)

    @property
    def original_estimated_tokens(self) -> int:
        return estimate_tokens("x" * self.original_chars)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node": self.node,
            "task_id": self.task_id,
            "budget_chars": self.budget_chars,
            "original_chars": self.original_chars,
            "final_chars": self.final_chars,
            # 明确标注：这是估算，不是真实计费 Token
            "estimated_tokens": self.estimated_tokens,
            "token_estimate_method": f"chars/{CHARS_PER_TOKEN_ESTIMATE}",
            "truncated": self.truncated,
            "truncated_fields": sorted(self.truncated_fields),
            "over_budget": self.over_budget,
            "kept_tables": list(self.kept_tables),
            "excluded_tables": list(self.excluded_tables),
            "history_entries": self.history_entries,
            "knowledge_chunks": self.knowledge_chunks,
            "memory_entries": self.memory_entries,
            "knowledge_cache_hit": self.knowledge_cache_hit,
        }


def _json_length(value: Any) -> int:
    """一个值序列化后的字符数。无法序列化时退回 `str()`。"""
    try:
        return len(json.dumps(value, ensure_ascii=False, default=str))
    except (TypeError, ValueError):
        return len(str(value))


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    if limit <= len(TRUNCATION_MARKER):
        return TRUNCATION_MARKER[:limit]
    return text[: limit - len(TRUNCATION_MARKER)] + TRUNCATION_MARKER


def bound_value(
    value: Any,
    *,
    limit: int,
    path: str = "",
    report: ContextReport | None = None,
) -> Any:
    """把一个值压到 `limit` 字符以内。

    规则：

    - 保留键（见 `PRESERVE_KEYS`）原样返回，不计入截断；
    - 字符串超长直接截断并加标记；
    - 列表按**均分后的每项额度**递归压缩，因此少量超长项不会被整条丢弃，
      也不会因为"第一项就超预算"而丢掉全部；
    - 字典按剩余额度均分给各可选键，缩完再实测一次，仍超就继续收紧。
    """
    if isinstance(value, str):
        if len(value) <= limit:
            return value
        if report is not None:
            report.truncated_fields.append(path or "value")
        return _truncate_text(value, limit)

    if isinstance(value, (int, float, bool)) or value is None:
        return value

    if isinstance(value, list):
        measured = _json_length(value)
        if measured <= limit:
            return value

        if not value:
            return value

        # 每条记录的 JSON 骨架（键名、引号、逗号、括号）本身就要占字符。
        # 不预留的话，条目一多，"每项额度"会被骨架吃光，
        # 于是整体仍然远超预算——这是实测踩到的问题。
        skeleton = _json_length([{}] * len(value)) - 2  # 减去外层 "[]"
        note_space = min(160, max(0, limit // 10))
        available = max(1, limit - note_space - skeleton)

        items = list(value)

        # 条目过多时先限量：留着 200 条各自空壳，不如给少量条目真实内容。
        max_items = max(1, available // 200)
        if len(items) > max_items:
            dropped = len(items) - max_items
            items = items[:max_items]
        else:
            dropped = 0

        # **必须在这里登记截断**。列表压缩不一定经过字符串分支
        # （条目变少、或子字段在各自额度内变动都可能），
        # 漏登记会让 `ContextReport.truncated` 报 False，
        # 也就是"明明丢了东西却报告说没丢"——那正是这个字段存在的意义。
        if report is not None:
            report.truncated_fields.append(path or "list")

        per_item = max(1, available // len(items))

        kept: list[Any] = []
        carried = 0
        for item in items:
            item_limit = per_item + carried
            bounded_item = bound_value(
                item, limit=item_limit, path=path, report=report
            )
            kept.append(bounded_item)
            # 用不完的额度让给后面的项（前面的项小则后面的项更宽松）
            carried = max(0, item_limit - _json_length(bounded_item))

        if dropped:
            kept.append(
                {
                    "context_note": (
                        f"另有 {dropped} 条同结构记录因上下文预算未展开"
                    ),
                    "dropped": dropped,
                }
            )
        else:
            kept.append(
                {
                    "context_note": (
                        f"共 {len(value)} 条同结构记录，已按每条约 {per_item} 字符"
                        "压缩展示"
                    ),
                }
            )
        return kept

    if isinstance(value, dict):
        measured = _json_length(value)
        if measured <= limit:
            return value

        preserved = {
            key: item for key, item in value.items() if key in PRESERVE_KEYS
        }
        optional = {
            key: item for key, item in value.items() if key not in PRESERVE_KEYS
        }

        result: dict[str, Any] = dict(preserved)
        if not optional:
            # 全是保留键：不能再缩，原样返回（保留键优先于预算）
            return result

        # 剩余额度按可选键**按体积比例**分配：大的键拿更多绝对额度，
        # 但每个键都按比例缩小，不会出现"大的被砍光、小的原样留着"。
        optional_budget = max(1, limit - _json_length(preserved))

        for _attempt in range(3):
            sizes = {key: max(1, _json_length(item)) for key, item in optional.items()}
            total = sum(sizes.values())
            allocated: dict[str, Any] = {}
            for key, item in optional.items():
                share = max(1, int(optional_budget * sizes[key] / total))
                # 压缩本身也算截断：即使子字段没有触发字符串截断，
                # "按额度缩小"同样意味着这份上下文不再完整。
                if report is not None:
                    report.truncated_fields.append(
                        f"{path}.{key}" if path else key
                    )
                allocated[key] = bound_value(
                    item,
                    limit=share,
                    path=f"{path}.{key}" if path else key,
                    report=report,
                )

            result = {**preserved, **allocated}
            if _json_length(result) <= limit:
                return result

            # 仍超：收紧可选额度再来一次
            optional_budget = max(1, int(optional_budget * 0.7))

        return result

    # 其它类型（DataFrame 之类）本来就不该进 Prompt；序列化后按字符串处理
    return bound_value(str(value), limit=limit, path=path, report=report)


def select_context(
    node: str,
    payload: dict[str, Any],
    *,
    budget: int | None = None,
) -> tuple[dict[str, Any], ContextReport]:
    """按节点预算裁剪一次 Prompt 的输入。

    返回 `(裁剪后的输入, 度量结果)`。度量结果用于结构化日志，
    使"这次用了多少上下文"成为可观察事实，而不是靠猜。
    """
    effective = budget if budget is not None else budget_for(node)
    report = ContextReport(
        node=node,
        budget_chars=effective,
        original_chars=_json_length(payload),
    )

    bounded = bound_value(payload, limit=effective, path="", report=report)

    # 裁剪后仍可能略超（保留键本身就很长）：那也不动保留键，
    # 因为丢掉错误信息比超一点预算更糟。
    report.final_chars = _json_length(bounded)

    return bounded, report


def summarise_history(
    execution_results: list[dict],
    *,
    max_entries: int = 8,
) -> list[dict]:
    """把多任务历史压成**确定性**摘要。

    只保留判定与展示真正需要的字段，不保留整份结果：
    结果是"可确定性重建"的（同样的输入必然得到同样的摘要），
    因此恢复后重新构造不会产生漂移。
    """
    summary: list[dict] = []
    for item in execution_results[-max_entries:]:
        summary.append(
            {
                "task_id": item.get("task_id"),
                "success": item.get("success"),
                "error": item.get("error"),
                "error_kind": item.get("error_kind"),
                "attempt": {
                    "retry_count": item.get("retry_count"),
                    "refine_count": item.get("refine_count"),
                },
            }
        )
    return summary


# ---------------------------------------------------------------------------
# 字段级预算：给各 Agent 的 `build_*_context` 共用
# ---------------------------------------------------------------------------

#: 每个 LLM 节点里**不可压缩**的字段名。
#:
#: 键必须与各 Agent `inputs` 字典里的真实字段名一致——写错名字不会报错，
#: 只会让"该保留的字段"被当作可压缩字段处理，那是这条防线最危险的失效方式。
#: 有一条测试逐一核对这里与各 Agent 实际 inputs 键的对应关系。
#:
#: 选取原则：
#:
#: - **用户问题与业务知识**（`analysis_focus` / `retrieved_knowledge`）——决定"要做什么"；
#: - **失败反馈**（`error` / `execution_result` / `failed_tasks`）——决定"下一步修什么"；
#: - **出错代码**（`code`）——不看代码无法修；
#: - **数据 schema**（`data_context`）——决定"能查什么"。
#:
#: 代价是：这些字段本身超预算时最终仍会超（`data_context` 尤其明显，
#: 但它承载表结构与样本，压掉会让模型写出查不到的列）。
#: 这是刻意取舍——超预算比让模型在缺信息的情况下决策要好。
NODE_PRESERVE_FIELDS: dict[str, frozenset[str]] = {
    # planner：用户问题 + 业务知识 + 数据 schema（可行性依据）
    "planner": frozenset(
        {"analysis_focus", "retrieved_knowledge", "data_context"}
    ),
    # executor：任务定义 + 知识 + 数据 schema。
    #
    # 注意这里**没有** `error` / `execution_result`：executor 的 inputs 里
    # 根本没有这两个键。上一次失败的信息由上游写进**任务描述**再传进来，
    # 因此它随 `task` 一起被保留。写一个不存在的字段名不会报错，
    # 只会让真正该保留的字段被当成可压缩字段——有测试逐一对齐这层关系。
    "executor": frozenset(
        {"task", "retrieved_knowledge", "route", "data_context"}
    ),
    # repair：错误反馈与出错代码不可压缩，否则无法定位问题
    "repair": frozenset({"task", "retrieved_knowledge", "route", "code", "error"}),
    # refine：出错代码 + critic 反馈 + 上一次执行结果（里面含 error）
    "refine": frozenset(
        {
            "task",
            "retrieved_knowledge",
            "route",
            "code",
            "execution_result",
            "critic_feedback",
        }
    ),
    # critic：任务期望与执行结果（含 error/error_kind）是判定依据
    "critic": frozenset({"task", "retrieved_knowledge", "execution_result"}),
    # visualizer：计划与失败任务决定要不要出图
    "visualizer": frozenset({"analysis_plan", "failed_tasks"}),
    # plotter：图表定义与样本 schema 决定画什么
    "plotter": frozenset({"visualization_task", "data_context"}),
    # plot_repair：出错信息与出错代码
    "plot_repair": frozenset({"visualization_task", "code", "error"}),
    # reporter：计划、失败任务与可视化结果是报告骨架
    "reporter": frozenset(
        {"analysis_plan", "failed_tasks", "visualization_results"}
    ),
}


def bound_fields(
    node: str,
    fields: dict[str, Any],
    *,
    budget: int | None = None,
    task_id: str | None = None,
    kept_tables: list[str] | None = None,
    excluded_tables: list[str] | None = None,
    history_entries: int | None = None,
    knowledge_chunks: int = 0,
    memory_entries: int = 0,
    knowledge_cache_hit: bool = False,
) -> tuple[dict[str, Any], ContextReport]:
    """按节点预算压缩一组**命名字段**。

    与 `select_context` 的区别：输入是原始对象（可能是 list / dict / str），
    不是已经序列化好的 Prompt 文本。用于各 Agent 在拼 Prompt **之前**
    控制上下文规模。

    压缩规则：

    1. `NODE_PRESERVE_FIELDS[node]` 里的字段**原样保留**，不参与压缩；
    2. 其余字段按当前体积**比例**分配剩余额度，逐个递归压缩；
    3. 若保留字段本身就超预算，最终结果仍会超——这是刻意的取舍：
       丢掉错误反馈比超预算更糟。

    其余关键字参数只描述"这次选了哪些表、带了几条历史"，写进报告供观测，
    不影响压缩结果。
    """
    effective = budget if budget is not None else budget_for(node)
    report = ContextReport(
        node=node,
        budget_chars=effective,
        original_chars=_json_length(fields),
        task_id=task_id,
        kept_tables=list(kept_tables or []),
        excluded_tables=list(excluded_tables or []),
        history_entries=history_entries,
        knowledge_chunks=knowledge_chunks,
        memory_entries=memory_entries,
        knowledge_cache_hit=knowledge_cache_hit,
    )

    preserved_names = NODE_PRESERVE_FIELDS.get(node, frozenset())

    preserved = {name: value for name, value in fields.items() if name in preserved_names}
    optional = {name: value for name, value in fields.items() if name not in preserved_names}

    if not optional or _json_length(fields) <= effective:
        report.final_chars = _json_length(fields)
        return dict(fields), report

    remaining = max(1, effective - _json_length(preserved))
    sizes = {name: max(1, _json_length(value)) for name, value in optional.items()}
    total = sum(sizes.values())

    bounded_optional: dict[str, Any] = {}
    for name, value in optional.items():
        share = max(1, int(remaining * sizes[name] / total))
        bounded_optional[name] = bound_value(
            value, limit=share, path=name, report=report
        )

    result = {**bounded_optional, **preserved}
    report.final_chars = _json_length(result)

    return result, report
