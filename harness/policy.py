"""执行策略：预算判定、消耗与结算的统一入口。

预算状态留在 `utils/agent_loop` 与图状态里。它必须随 Checkpoint 持久化，
因此不能有第二份计数器；这里只编码"哪种操作看哪些额度、什么时候计一次"。

判定与计数分开是有原因的。判定发生在真实请求发出之前，超预算时不能往账本
写 `started`；计数发生在请求发出之前一点点，这样即使请求失败也占用了额度。
两者读同一份用量，所以同一个请求只会被算一次——这一点由
`tests/test_llm_request_budget_discriminating.py` 固定。
"""

from __future__ import annotations

from typing import Any

from utils import agent_loop

#: 预算快照在 state 里的键。这里再导出一次，避免上层到处 import 两个模块。
BUDGET_FIELD = agent_loop.BUDGET_FIELD


def initialize(state: dict[str, Any]) -> dict[str, Any]:
    """补齐预算快照并打上开始时刻。

    已有计数不重置，这正是"恢复后预算不归零"的实现点；因此重复调用是安全的。
    """
    return agent_loop.initialize(state)


def ensure_model_request_allowed(state: dict[str, Any]) -> None:
    """一次真实模型请求发出前，只看模型请求额度。

    不检查工具调用与时间额度：这一请求额外消耗的是一个模型请求额度，
    其它分项在各自的入口判定。混在一起会让"工具额度刚好用尽"误杀一次
    本来合法的模型调用。
    """
    agent_loop.check_llm_budget(state)


def ensure_tool_allowed(state: dict[str, Any]) -> None:
    """工具调用前检查全部额度。

    工具是开销最大、且有副作用的操作，因此轮次、工具次数、模型请求与执行
    时间任何一项用尽都应当拒绝。`max_rounds` 目前也是在这里生效。
    """
    agent_loop.check(state)


def count_model_request(state: dict[str, Any]) -> dict[str, Any]:
    """记一次真实模型请求，并就地写回 state。

    就地写回是必需的：一次逻辑调用内部可能因为结构化输出解析失败而重发，
    后续重试必须看到"已经发过几次"，而 state 的正式增量要等节点返回才合并。
    就地更新改变的是 LangGraph 传给该节点的 state 对象，节点返回的增量是它的
    超集，因此不会互相覆盖。
    """
    updates = agent_loop.consume_llm_call(state)
    state[BUDGET_FIELD] = updates[BUDGET_FIELD]
    return updates


def count_tool_call(state: dict[str, Any]) -> dict[str, Any]:
    """记一次工具调用。"""
    return agent_loop.consume_tool_call(state)


def count_round(state: dict[str, Any]) -> dict[str, Any]:
    """记一次 Reason→Act→Observe→Decide 轮次。"""
    return agent_loop.consume_round(state)


def settle(state: dict[str, Any]) -> dict[str, Any]:
    """结算执行时钟：把"距上次操作"的那段算清并让时钟停住。

    模型调用或工具调用结束后调用。之后若进入人工审批等待，`elapsed_seconds`
    不再增长——等待不该消耗执行预算。
    """
    return agent_loop.settle_active_time(state)


def usage(state: dict[str, Any]) -> dict[str, Any]:
    return agent_loop.usage_of(state)


def budget(state: dict[str, Any]) -> agent_loop.AgentBudget:
    return agent_loop.budget_of(state)


def remaining(state: dict[str, Any]) -> dict[str, Any]:
    """额度余量，供日志与收尾判断使用。"""
    return agent_loop.snapshot(state)["remaining"]


__all__ = [
    "BUDGET_FIELD",
    "budget",
    "count_model_request",
    "count_round",
    "count_tool_call",
    "ensure_model_request_allowed",
    "ensure_tool_allowed",
    "initialize",
    "remaining",
    "settle",
    "usage",
]
