"""P3-A：有界 Agent 循环的控制层。

## 这个模块解决什么

现有 19 节点图已经是**有界**的：`MAX_RETRIES` / `MAX_REFINES` /
`MAX_PLOT_RETRIES` 各自限制本地循环，`recursion_limit` 兜住全局。那些上限
是**每类操作的局部计数**，缺的是统一的、跨操作的总量治理：

- 一个 Run 里"工具被调用多少次"没有上限，只受各类 retry 上限间接约束；
- 没有整体时间预算，图可以在很多次合法重试里长时间占用资源；
- 局部计数分散在 `retry_count` / `refine_count` / `plot_retry_count` 里，
  没有单一的"这个 Run 还能不能继续"的判断点。

因此这里**不重写图**，只在既有节点之上补一层计数器与判定：

    Reason（模型给出下一步）
      → Act（执行工具）
      → Observe（工具结果与分类）
      → Decide（在预算内继续 / 修复 / 收敛失败）

## 预算为什么放在 state 里

预算必须**跨进程恢复后不重置**。放在进程内存里的话，崩溃重启就归零，
"有界"就名不副实。因此计数进 `InsightFlowState`，随 Checkpoint 一起持久化。

时间预算是个例外：它是**墙上时钟**，不能持久化"已用秒数"（那会在
暂停等待人工审批时把审批时间也算进去）。因此记的是**首次开始的时刻**，
恢复后按它重算剩余时间；没有开始时刻就现场补一个。

## 与既有机制的关系

- 不新建幂等或重试系统：逻辑尝试 ID 仍来自 `utils/idempotency`；
- 不取代 `MAX_RETRIES` 等局部上限：它们是更细的闸门，两者同时生效；
- 失败是**收敛**而不是崩溃：超预算时抛出 `BudgetExceeded`，
  节点捕获后转成正常的失败记录，让工作流按既有路径收尾。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

# state 里保存预算快照的字段名
BUDGET_FIELD = "execution_budget"

# 预算收敛失败使用的错误分类。归到已有的 `unavailable` 语义下：
# 预算耗尽不是"代码写错了"，重试同样的代码不会好转，
# 因此不该进入 Repair——这与 P2-B 对环境类错误的处理一致。
BUDGET_ERROR_KIND = "unavailable"


class BudgetExceeded(RuntimeError):
    """某一项执行预算已经用尽。

    带 `scope` 便于上层区分是哪一类预算（rounds / tool_calls / llm_calls / seconds）。
    """

    def __init__(self, scope: str, used: float, limit: float, message: str | None = None):
        self.scope = scope
        self.used = used
        self.limit = limit
        super().__init__(
            message
            or f"执行预算已用尽（{scope}）：已用 {used} / 上限 {limit}"
        )


@dataclass(frozen=True)
class AgentBudget:
    """一个 Run 的有界执行预算。

    所有上限都是**总量**，不是每轮的量。默认值取得足够宽，使既有工作流
    在正常路径下完全不受影响；它的作用是兜住"合法但无休止"的重试组合。
    """

    max_rounds: int = 24
    """Reason→Act→Observe→Decide 的总轮次上限。

    默认 24：既有的最坏合法路径是 1 次执行 + 3 次 repair + 3 次 refine，
    再加上图自身的节点步进，远低于这个数；而一旦出现互相触发的组合爆炸，
    这里会先于 recursion_limit 收敛住。
    """

    max_tool_calls: int = 40
    """工具调用总次数上限（含 python / plot / sql）。"""

    max_llm_calls: int = 60
    """模型调用总次数上限。

    与 `utils/llm_ledger` 的分工：账本记录"每次调用是否确认完成"，
    这里只负责"总量是否超了"。两者都不替代对方。
    """

    max_seconds: float = 1800.0
    """墙上时钟预算（秒）。默认 30 分钟。"""

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_rounds": self.max_rounds,
            "max_tool_calls": self.max_tool_calls,
            "max_llm_calls": self.max_llm_calls,
            "max_seconds": self.max_seconds,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "AgentBudget":
        """从 state 里恢复预算上限。

        只认已知字段，缺失或类型不对时回落到默认值：预算配置是宿主决定的，
        不该因为 state 里少一个键就让整次运行失败。

        **0 是合法上限**（"一次都不许"），必须与"字段缺失"区分开。
        早期实现用 `value <= 0` 过滤，于是 `max_tool_calls=0` 被当成缺失、
        回落到默认的 40——一个声明"禁止工具调用"的 Run 反而拿到了 40 次额度。
        这类错误很隐蔽：限制看起来生效了，实际被悄悄放宽。
        """
        if not payload:
            return cls()

        known: dict[str, Any] = {}
        for field_name in ("max_rounds", "max_tool_calls", "max_llm_calls", "max_seconds"):
            value = payload.get(field_name)
            # bool 是 int 的子类，必须显式排除，否则 True 会变成上限 1
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if value < 0:
                continue
            known[field_name] = (
                int(value) if field_name != "max_seconds" else float(value)
            )
        return cls(**known)


def new_usage() -> dict[str, Any]:
    """一份全新的用量记录。"""
    return {
        "rounds": 0,
        "tool_calls": 0,
        "llm_calls": 0,
        # 首次开始时刻。墙上时钟不能持久化"已用秒数"：
        # 等待人工审批的时间不属于执行消耗，恢复时按开始时刻重算。
        "started_at": None,
    }


def usage_of(state: dict[str, Any]) -> dict[str, Any]:
    """读出 state 里的用量；没有就返回一份新的（不写回）。"""
    payload = state.get(BUDGET_FIELD)
    if not isinstance(payload, dict):
        return new_usage()
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return new_usage()
    merged = new_usage()
    merged.update(
        {key: value for key, value in usage.items() if key in merged}
    )
    return merged


def budget_of(state: dict[str, Any]) -> AgentBudget:
    payload = state.get(BUDGET_FIELD)
    if isinstance(payload, dict):
        return AgentBudget.from_dict(payload.get("limits"))
    return AgentBudget()


def initialize(state: dict[str, Any]) -> dict[str, Any]:
    """确保 state 里有一份预算快照，并在首次调用时打上开始时刻。

    返回要写回 state 的增量。已经存在时不重置任何计数——这正是
    "恢复后不重置预算"的实现点。
    """
    payload = state.get(BUDGET_FIELD)

    if not isinstance(payload, dict):
        return {
            BUDGET_FIELD: {
                "limits": AgentBudget().to_dict(),
                "usage": {**new_usage(), "started_at": time.time()},
            }
        }

    usage = usage_of(state)
    if usage.get("started_at") is None:
        # 老 Checkpoint（P3 之前）没有这个字段，现场补一个
        return {
            BUDGET_FIELD: {
                "limits": AgentBudget.from_dict(payload.get("limits")).to_dict(),
                "usage": {**usage, "started_at": time.time()},
            }
        }

    return {}


def elapsed_seconds(state: dict[str, Any], *, now: float | None = None) -> float:
    usage = usage_of(state)
    started = usage.get("started_at")
    if not isinstance(started, (int, float)):
        return 0.0
    return max(0.0, (now if now is not None else time.time()) - float(started))


def check(state: dict[str, Any], *, now: float | None = None) -> None:
    """预算是否还能继续。超了就抛 `BudgetExceeded`。

    只读，不改 state：判定与计数分开，调用方先判定再决定要不要计一次。
    """
    budget = budget_of(state)
    usage = usage_of(state)

    for scope, used, limit in (
        ("rounds", usage["rounds"], budget.max_rounds),
        ("tool_calls", usage["tool_calls"], budget.max_tool_calls),
        ("llm_calls", usage["llm_calls"], budget.max_llm_calls),
    ):
        if used >= limit:
            raise BudgetExceeded(scope, used, limit)

    spent = elapsed_seconds(state, now=now)
    if spent >= budget.max_seconds:
        raise BudgetExceeded("seconds", round(spent, 1), budget.max_seconds)

def _bump(state: dict[str, Any], key: str) -> dict[str, Any]:
    payload = state.get(BUDGET_FIELD)
    if not isinstance(payload, dict):
        payload = {
            "limits": AgentBudget().to_dict(),
            "usage": {**new_usage(), "started_at": time.time()},
        }

    usage = usage_of(state)
    usage[key] = int(usage.get(key, 0)) + 1

    return {
        BUDGET_FIELD: {
            "limits": AgentBudget.from_dict(payload.get("limits")).to_dict(),
            "usage": usage,
        }
    }


def consume_round(state: dict[str, Any]) -> dict[str, Any]:
    """记一次 Reason→Act→Observe→Decide 轮次。"""
    return _bump(state, "rounds")


def consume_tool_call(state: dict[str, Any]) -> dict[str, Any]:
    return _bump(state, "tool_calls")


def consume_llm_call(state: dict[str, Any]) -> dict[str, Any]:
    return _bump(state, "llm_calls")


def snapshot(state: dict[str, Any], *, now: float | None = None) -> dict[str, Any]:
    """当前预算与用量，用于日志与可观测性。"""
    budget = budget_of(state)
    usage = usage_of(state)
    return {
        "limits": budget.to_dict(),
        "usage": {**usage, "seconds": round(elapsed_seconds(state, now=now), 3)},
        "remaining": {
            "rounds": max(0, budget.max_rounds - usage["rounds"]),
            "tool_calls": max(0, budget.max_tool_calls - usage["tool_calls"]),
            "llm_calls": max(0, budget.max_llm_calls - usage["llm_calls"]),
            "seconds": max(
                0.0, round(budget.max_seconds - elapsed_seconds(state, now=now), 3)
            ),
        },
    }
