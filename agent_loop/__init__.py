"""运行期动态决策闭环：观察 → 决策 → 校验 → 执行 → 再观察。

与 H3 的区别：H3 只管"初始计划是否可执行"，在执行前就结束了；这里管
"执行中观察到的结果是否说明该换一条路"，输入是真实的任务结果。

模块分工：

- `observation`：把 state 压成可序列化的观察，确定性、有界；
- `decision`：模型在固定动作集合里选一个（结构化协议）；
- `policy`：确定性校验，模型不能直接决定路由；
- `controller`：把动作翻译成 state 增量与下一个节点。
"""

from agent_loop.controller import ROUTE_BY_ACTION, decision_updates, route_for
from agent_loop.observation import Observation, build_observation
from agent_loop.policy import (
    ACTIONS,
    CONTINUE,
    DEFAULT_LIMITS,
    FINISH,
    REFINE,
    REPAIR,
    REPLAN_REMAINING,
    RETRIEVE_MORE,
    SKIP_TASK,
    ActionDecision,
    limits_from,
    validate_action,
)

__all__ = [
    "ACTIONS",
    "CONTINUE",
    "DEFAULT_LIMITS",
    "FINISH",
    "Observation",
    "REFINE",
    "REPAIR",
    "REPLAN_REMAINING",
    "RETRIEVE_MORE",
    "ROUTE_BY_ACTION",
    "SKIP_TASK",
    "ActionDecision",
    "build_observation",
    "decision_updates",
    "limits_from",
    "route_for",
    "validate_action",
]
