"""动态决策协议：给定观察，让模型在固定动作集合里选一个并说明理由。

协议是结构化的：模型只能返回 `{"action": ..., "reason": ...}`，动作名必须是
枚举之一，最终由 `agent_loop.policy` 校验。自由文本永远不会直接驱动路由。

为什么保持单一决策 Agent：决策需要"看懂结果与目标是否一致"的语义判断，
用一个专门的提示比在每个节点里塞分支更清楚；它不新增分析能力，也不碰工具。
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

from agent_loop.policy import (
    ACTIONS,
    CONTINUE,
    FINISH,
    REFINE,
    REPAIR,
    REPLAN_REMAINING,
    RETRIEVE_MORE,
    SKIP_TASK,
)
from utils.agent_loop import after_model_request, before_model_request
from utils.llm_retry import invoke_json_with_retry

NODE_NAME = "decider"

DECIDER_PROMPT = """
你是一个数据分析运行期的决策 Agent。

下面是一次分析任务当前的**实际观察结果**（JSON）。它来自真实的执行结果、
校验反馈与检索证据，不是计划，也不是推测：

{observation}

本轮的可用动作与成立条件：

- continue：当前任务已经通过校验，进入下一个任务；
- finish：分析可以结束（目标已达成，或继续下去没有意义）；
- repair：执行代码有错误，且错误属于可以通过改代码解决的类型；
- refine：结果质量不达标，critic 给出了可执行的改进意见；
- retrieve_more：业务证据不足（检索到的片段与任务目标不匹配或为空）；
- replan_remaining：当前计划中尚未执行的任务已无法完成，需要调整剩余任务；
- skip_task：当前任务在现有数据与权限下无法安全完成，记录原因后继续后续任务。

判断时必须依据观察里的字段，而不是猜测：

- `error_kind` 是权限、授权、环境或预算类错误时，不要建议 repair / refine：
  重试不会让权限变成允许；
- `evidence_refs` 为空且任务依赖业务规则时，才建议 retrieve_more；
- `validation_passed` 为 true 且 `execution_success` 为 true 时，不要建议修复；
- 剩余预算 `remaining_budget` 很小时，优先收尾而不是发起更多尝试。

已经做过的动作与次数（避免重复）：

{history}

当前限制：

{limits}

只返回合法 JSON，不要输出其他文字。

返回格式：

{{
    "action": "{actions}",
    "reason": "一句话说明依据"
}}
"""

_VALIDATOR = None


def _validate(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("决策结果必须是 JSON 对象")
    action = payload.get("action")
    if not isinstance(action, str) or not action.strip():
        raise ValueError(f"决策结果缺少 action：{action!r}")
    reason = payload.get("reason")
    # 不在协议里直接拒绝未知动作：那会让一次无效输出变成图上的异常。
    # 未知动作交给 `agent_loop.policy` 收敛为安全收尾，并留下"模型想做什么"。
    return {
        "action": action.strip().lower(),
        "reason": str(reason or "").strip()[:300],
    }


def format_history(history: list[dict[str, Any]]) -> str:
    if not history:
        return "无"
    lines = [
        f"- 第 {item.get('step')} 步 {item.get('action')}：{item.get('reason') or ''}"
        for item in history[-5:]
    ]
    return "\n".join(lines)


def format_limits(limits: dict[str, int]) -> str:
    return json.dumps(limits, ensure_ascii=False, sort_keys=True)


def build_inputs(
    observation: Any,
    *,
    history: list[dict[str, Any]],
    limits: dict[str, int],
) -> dict[str, Any]:
    return {
        "observation": json.dumps(
            observation.to_dict(), ensure_ascii=False, indent=2, default=str
        ),
        "history": format_history(history),
        "limits": format_limits(limits),
        "actions": " | ".join(
            [
                CONTINUE,
                FINISH,
                REPAIR,
                REFINE,
                RETRIEVE_MORE,
                REPLAN_REMAINING,
                SKIP_TASK,
            ]
        ),
    }


def decide_action(
    observation: Any,
    *,
    history: list[dict[str, Any]],
    limits: dict[str, int],
    llm: Any,
) -> dict[str, Any]:
    """让模型在动作集合里选一个。返回值仍要经过策略校验。"""
    prompt = PromptTemplate.from_template(DECIDER_PROMPT)
    chain = prompt | llm | JsonOutputParser()
    inputs = build_inputs(observation, history=history, limits=limits)

    return invoke_json_with_retry(
        chain,
        inputs,
        validate=_validate,
        before_request=before_model_request,
        on_request=after_model_request,
    )


__all__ = ["DECIDER_PROMPT", "build_inputs", "decide_action", "format_history"]
