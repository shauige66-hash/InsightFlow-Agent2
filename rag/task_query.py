"""从当前任务构造检索问句。

检索问句决定"这次为什么去查知识库"。只用任务描述会让"按地区细分销售额"
查不到指标定义，只用关键词又会丢掉目标表约束，所以拼三样东西：任务描述、
期望产出、分析重点；上一轮对话只带最近一条用户消息，避免把整段历史塞进问句。
"""

from __future__ import annotations

from typing import Any, Iterable


def build_task_query(
    task: dict[str, Any],
    *,
    analysis_focus: str | None = None,
    previous_context: Iterable[str] | None = None,
) -> str:
    parts: list[str] = []

    description = str(task.get("description") or "").strip()
    if description:
        parts.append(description)

    expected = str(task.get("expected_output") or "").strip()
    if expected:
        parts.append(f"期望产出：{expected}")

    tables = [str(name) for name in (task.get("target_tables") or []) if name]
    if tables:
        parts.append(f"涉及数据：{', '.join(sorted(tables))}")

    focus = (analysis_focus or "").strip()
    if focus:
        parts.append(f"分析重点：{focus}")

    for item in list(previous_context or [])[-1:]:
        text = str(item or "").strip()
        if text:
            parts.append(f"上一轮问题：{text}")

    return "；".join(parts)


__all__ = ["build_task_query"]
