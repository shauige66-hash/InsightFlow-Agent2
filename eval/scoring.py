"""P5-B：判分与指标口径。

## 口径必须分清

任务**完成率**（completion）与**正确率**（correctness）是两件事：

- 完成：工作流跑到了终态并产出了可判分的结果；
- 正确：产出的数值与**独立算出的标准答案**一致。

一个 Run 可以"完成但算错"，也可以"没完成但也没算错"。
把两者合成一个数字会掩盖最需要看的失败模式，因此这里分开统计。

部分完成（`analysis_status == "partial"`，有计划但有任务失败）
单列一档：它既不是完全成功，也不是彻底失败。

## 数值比较

浮点比较用**容差**而不是相等：不同实现（pandas / SQLite / Python）
在最后一位上可能有差异。容差取 1e-6 相对量级，比"金额保留两位"的
业务精度更严，因此不会放过真实的算错。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: 浮点比较容差。业务金额保留两位，这个容差远小于一分钱。
FLOAT_TOLERANCE = 1e-6

#: 任务终态判定
STATUS_CORRECT = "correct"
STATUS_INCORRECT = "incorrect"
STATUS_INCOMPLETE = "incomplete"
STATUS_PARTIAL = "partial"
STATUS_ERROR = "error"

#: 计入"完成"的状态
COMPLETED_STATUSES = {STATUS_CORRECT, STATUS_INCORRECT, STATUS_PARTIAL}


def values_match(actual: Any, expected: Any) -> bool:
    """数值/结构的宽松但严格的比较。

    适用于**复合单行**答案（例如"营收最高的订单"返回 `{order_id, revenue}`）：
    实际结果多带一列不影响正确性，因此忽略多余的键。

    分组答案请用 `grouped_match`——那里多余的键意味着"多出一组"，
    是实打实的错误。两种语义分开，避免一个宽松规则掩盖另一种错误。
    """
    if isinstance(expected, bool) or isinstance(actual, bool):
        return actual is expected

    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(float(actual) - float(expected)) <= FLOAT_TOLERANCE * max(
            1.0, abs(float(expected))
        )

    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        # 键也按数值语义归一：同一分组键在不同路径下可能是 1062 或 '1062'，
        # 那是类型差异而不是答案差异。缺键必须判错，不能因为归一而放过。
        normalized_actual = {_coerce_key(key): value for key, value in actual.items()}
        for key, value in expected.items():
            normalized_key = _coerce_key(key)
            if normalized_key not in normalized_actual:
                return False
            if not values_match(normalized_actual[normalized_key], value):
                return False
        return True

    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return False
        return all(values_match(a, e) for a, e in zip(actual, expected))

    return actual == expected


def grouped_match(actual: Any, expected: dict) -> bool:
    """分组答案的**严格**比较：键集合必须完全一致。

    与 `values_match` 的差别只有一条：多余的键判错。

    为什么必须严格：分组结果少一组或多一组都是错误答案，
    而"忽略多余键"的宽松规则会把它判成正确——实测中
    `{**expected, "NoSuchRegion": 1.0}` 会被误判为通过。
    """
    if not isinstance(actual, dict):
        return False

    normalized_actual = {_coerce_key(key): value for key, value in actual.items()}
    normalized_expected = {_coerce_key(key): value for key, value in expected.items()}

    if set(normalized_actual) != set(normalized_expected):
        return False

    return all(
        values_match(normalized_actual[key], value)
        for key, value in normalized_expected.items()
    )


def _coerce_number(value: Any) -> Any:
    """把 numpy / Decimal 之类收敛成 Python 原生数，便于比较与序列化。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    try:
        import decimal

        if isinstance(value, decimal.Decimal):
            return float(value)
    except ImportError:  # pragma: no cover
        pass
    # numpy 标量：有 item() 且不是容器
    item = getattr(value, "item", None)
    if callable(item) and getattr(value, "shape", None) == ():
        try:
            return item()
        except (TypeError, ValueError):
            return value
    return value


def _coerce_key(value: Any) -> Any:
    """分组键归一。

    同一个分组键在不同路径下可能是 `1062`（int）或 `'1062'`（str）：
    SQLite 返回 int，而 pandas 的 `groupby` 在某些构造下给出 numpy 整数、
    序列化成 JSON 后又变成字符串。**这是类型差异，不是答案差异**，
    因此按数值语义归一后再比较。

    不能反过来把期望值都变成字符串：那会让 `'01'` 与 `1` 相等，
    也会掩盖真实的类型错误。
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else value

    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        pass
    try:
        number = float(text)
    except ValueError:
        return str(value)
    return int(number) if number.is_integer() else number


def extract_answer(task: Any, records: list[dict]) -> Any:
    """从执行结果里取出与标准答案可比对的形状。

    三种形状，各自的语义必须说清楚，否则很容易把"复合结果"
    误当成"分组结果"而判错（这是实测踩到的）：

    - `scalar`：单个数值，取 `value_column`（缺列时退回唯一列）；
    - `row`：**一行复合结果**（例如"营收最高的订单 + 它的营收"），
      整行按 `{列: 值}` 返回；
    - `rows_grouped`：**按分组键展开的多行**，返回 `{键: 指标}`。
    """
    if not records:
        return None

    shape = getattr(task, "shape", "scalar")

    if shape == "scalar":
        row = records[0]
        column = getattr(task, "value_column", None)
        if column and column in row:
            return _coerce_number(row[column])
        # 单列结果直接取值
        if len(row) == 1:
            return _coerce_number(next(iter(row.values())))
        # 多列且未指定指标列：按整行返回，交给 values_match 按键比对
        return {str(key): _coerce_number(value) for key, value in row.items()}

    if shape == "row":
        if len(records) != 1:
            return None
        return {str(key): _coerce_number(value) for key, value in records[0].items()}

    if shape == "rows_grouped":
        key_column = getattr(task, "key_column", None)
        value_column = getattr(task, "value_column", None)
        if not key_column or not value_column:
            return None
        result: dict[Any, Any] = {}
        for row in records:
            if key_column not in row or value_column not in row:
                return None
            result[_coerce_key(row[key_column])] = _coerce_number(row[value_column])
        return result

    return None


@dataclass
class TaskOutcome:
    """单个任务的评测结果。"""

    task_id: str
    family: str
    status: str
    expected: Any = None
    actual: Any = None
    reason: str | None = None
    duration_seconds: float = 0.0
    llm_calls: int = 0
    tool_calls: int = 0
    analysis_status: str | None = None
    mcp_seconds: float = 0.0

    @property
    def completed(self) -> bool:
        return self.status in COMPLETED_STATUSES

    @property
    def correct(self) -> bool:
        return self.status == STATUS_CORRECT

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "family": self.family,
            "status": self.status,
            "completed": self.completed,
            "correct": self.correct,
            "expected": self.expected,
            "actual": self.actual,
            "reason": self.reason,
            "analysis_status": self.analysis_status,
            "duration_seconds": round(self.duration_seconds, 3),
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "mcp_seconds": round(self.mcp_seconds, 3),
        }


@dataclass
class EvalSummary:
    """一次评测的汇总指标。

    完成率与正确率**分开**给出；`correct_rate` 的分母是**全部任务**，
    因此"没跑完"也会拉低正确率——这正是想暴露的信息。
    """

    total: int = 0
    outcomes: list[TaskOutcome] = field(default_factory=list)

    def add(self, outcome: TaskOutcome) -> None:
        self.outcomes.append(outcome)
        self.total += 1

    def _count(self, predicate) -> int:
        return sum(1 for item in self.outcomes if predicate(item))

    @property
    def completed(self) -> int:
        return self._count(lambda item: item.completed)

    @property
    def correct(self) -> int:
        return self._count(lambda item: item.correct)

    @property
    def incorrect(self) -> int:
        return self._count(lambda item: item.status == STATUS_INCORRECT)

    @property
    def partial(self) -> int:
        return self._count(lambda item: item.status == STATUS_PARTIAL)

    @property
    def incomplete(self) -> int:
        return self._count(lambda item: item.status == STATUS_INCOMPLETE)

    @property
    def errored(self) -> int:
        return self._count(lambda item: item.status == STATUS_ERROR)

    @property
    def completion_rate(self) -> float:
        return self.completed / self.total if self.total else 0.0

    @property
    def correctness_rate(self) -> float:
        """关键数值正确率：分母是全部任务，不只是完成的任务。"""
        return self.correct / self.total if self.total else 0.0

    @property
    def conditional_accuracy(self) -> float:
        """在**已完成**的任务里，算对的比例。

        与 `correctness_rate` 一起看才有意义：
        前者说明"跑起来的时候算得对不对"，后者说明端到端可靠性。
        """
        return self.correct / self.completed if self.completed else 0.0

    @property
    def total_seconds(self) -> float:
        return sum(item.duration_seconds for item in self.outcomes)

    @property
    def total_llm_calls(self) -> int:
        return sum(item.llm_calls for item in self.outcomes)

    @property
    def total_tool_calls(self) -> int:
        return sum(item.tool_calls for item in self.outcomes)

    @property
    def total_mcp_seconds(self) -> float:
        return sum(item.mcp_seconds for item in self.outcomes)

    def by_family(self) -> dict[str, dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}
        for item in self.outcomes:
            bucket = grouped.setdefault(
                item.family, {"total": 0, "completed": 0, "correct": 0}
            )
            bucket["total"] += 1
            bucket["completed"] += int(item.completed)
            bucket["correct"] += int(item.correct)
        for bucket in grouped.values():
            bucket["completion_rate"] = round(
                bucket["completed"] / bucket["total"], 4
            )
            bucket["correctness_rate"] = round(bucket["correct"] / bucket["total"], 4)
        return grouped

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "completed": self.completed,
            "correct": self.correct,
            "incorrect": self.incorrect,
            "partial": self.partial,
            "incomplete": self.incomplete,
            "errored": self.errored,
            "completion_rate": round(self.completion_rate, 4),
            "correctness_rate": round(self.correctness_rate, 4),
            "conditional_accuracy": round(self.conditional_accuracy, 4),
            "total_seconds": round(self.total_seconds, 3),
            "total_llm_calls": self.total_llm_calls,
            "total_tool_calls": self.total_tool_calls,
            "total_mcp_seconds": round(self.total_mcp_seconds, 3),
            "by_family": self.by_family(),
            "outcomes": [item.to_dict() for item in self.outcomes],
        }
