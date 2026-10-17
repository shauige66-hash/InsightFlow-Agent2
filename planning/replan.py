"""运行期计划修订：只重写还没执行完的部分。

与 H3 的初始规划不是一回事：

- H3 在**执行前**决定"这份计划能不能跑"，并允许有限次重写整份计划；
- 这里在**执行中**发现后续任务已无法完成（数据里没有需要的指标、
  证据不支持原假设），只调整当前位置及其后的任务。

已完成的索引不会被覆盖：调用方只提交"尚未执行"的切片，节点负责把它拼回
原计划的前缀之后。任务 ID 也不能与已完成的任务重复，否则报告与幂等登记会混淆。
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

from utils.agent_loop import after_model_request, before_model_request
from utils.llm_retry import invoke_json_with_retry

NODE_NAME = "planner"

REPLAN_PROMPT = """
你是一个数据分析计划的修订 Agent。

分析已经开始执行，但现在发现当前的计划无法合理完成。请只重新设计
**从当前任务开始、尚未执行的任务**。

原始分析重点：

{analysis_focus}

已经完成的计划（不能修改，也不要在输出里重复）：

{completed_tasks}

当前任务（执行结果不满足要求）：

{current_task}

当前任务的观察结果：

{observation}

原本还未执行的任务（供参考，可以保留、修改或删除）：

{remaining_tasks}

本次可用的数据资源（只能使用这些表的真实资源名）：

{data_context}

要求：

1. 只输出尚未执行的部分，不要包含已完成的任务；
2. task_id 不得与已完成任务重复，建议从 {next_task_number} 开始编号；
3. target_tables 必须是 data_context 中真实存在的资源名；
4. 每个任务必须包含 task_id、description、target_tables、expected_output；
5. 如果现有数据无法支撑原来的分析目标，就设计能达成的替代任务，
   而不是保留一个注定失败的任务；
6. 只返回合法 JSON。

返回格式：

{{
    "tasks": [
        {{
            "task_id": "task_3",
            "description": "分析内容",
            "target_tables": ["table_name"],
            "expected_output": "预期结果"
        }}
    ]
}}
"""


def _validate(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("修订结果必须是 JSON 对象")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("修订结果缺少 tasks 列表")
    return tasks


def replan_remaining_tasks(
    *,
    analysis_focus: str | None,
    completed_tasks: list[dict[str, Any]],
    current_task: dict[str, Any],
    observation: dict[str, Any],
    remaining_tasks: list[dict[str, Any]],
    data_context: dict[str, Any],
    next_task_number: int,
    llm: Any,
) -> list[dict[str, Any]]:
    """让模型只给出"尚未执行"的新任务列表。"""
    prompt = PromptTemplate.from_template(REPLAN_PROMPT)
    chain = prompt | llm | JsonOutputParser()

    inputs = {
        "analysis_focus": analysis_focus or "未指定",
        "completed_tasks": json.dumps(
            completed_tasks, ensure_ascii=False, indent=2, default=str
        ),
        "current_task": json.dumps(
            current_task, ensure_ascii=False, indent=2, default=str
        ),
        "observation": json.dumps(
            observation, ensure_ascii=False, indent=2, default=str
        ),
        "remaining_tasks": json.dumps(
            remaining_tasks, ensure_ascii=False, indent=2, default=str
        ),
        "data_context": json.dumps(
            data_context, ensure_ascii=False, indent=2, default=str
        ),
        "next_task_number": next_task_number,
    }

    return invoke_json_with_retry(
        chain,
        inputs,
        validate=_validate,
        before_request=before_model_request,
        on_request=after_model_request,
    )


__all__ = ["REPLAN_PROMPT", "replan_remaining_tasks"]
