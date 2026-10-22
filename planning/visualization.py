"""Deterministic validation of chart structure and source-result lineage."""
from __future__ import annotations

from typing import Any
import re

MAX_CHARTS = 3


def validate_visualization_plan(plan: Any, *, analysis_plan: list[dict], execution_results: list[dict], failed_tasks: list[dict]) -> list[dict]:
    if not isinstance(plan, list) or len(plan) > MAX_CHARTS:
        raise ValueError(f"图表计划必须是最多 {MAX_CHARTS} 项的列表")
    validated, rejected = partition_visualization_plan(plan, analysis_plan=analysis_plan,
        execution_results=execution_results, failed_tasks=failed_tasks)
    if rejected:
        raise ValueError(rejected[0]["error"])
    return validated


def partition_visualization_plan(plan: Any, *, analysis_plan: list[dict], execution_results: list[dict], failed_tasks: list[dict]) -> tuple[list[dict], list[dict]]:
    """Apply the same strict source rules per chart; invalid siblings cannot veto valid ones."""
    if not isinstance(plan, list):
        return [], [{"chart_id": "visualization_plan", "source_task_id": None,
                     "error": "图表计划必须是列表", "stage": "planning"}]
    tasks = {task["task_id"]: task for task in analysis_plan}
    final = {item["task_id"]: item for item in execution_results}
    failed = {item.get("task_id") for item in failed_tasks}
    seen = set()
    validated = []
    rejected = []
    for index, item in enumerate(plan):
        try:
            if not isinstance(item, dict):
                raise ValueError("每个图表必须是对象")
            for field in ("chart_id", "description", "source_task_id", "chart_type", "expected_output"):
                if not isinstance(item.get(field), str) or not item[field].strip():
                    raise ValueError(f"图表缺少非空字符串字段 {field}")
            chart_id = item["chart_id"]
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", chart_id) or chart_id in seen:
                raise ValueError("chart_id 必须唯一且仅包含字母、数字、下划线、连字符")
            source = item["source_task_id"]
            if source in failed:
                raise ValueError(f"图表 {chart_id} 的来源任务 {source} 最终失败，不可用于绘图")
            if source not in tasks or not final.get(source, {}).get("success"):
                raise ValueError(f"图表 {chart_id} 没有最终成功的来源任务 {source}")
            if item["chart_type"] not in ("line", "bar", "scatter"):
                raise ValueError("chart_type 只能是 line、bar、scatter")
            tables = item.get("target_tables")
            if not isinstance(tables, list) or not tables or any(not isinstance(name, str) for name in tables) or len(set(tables)) != len(tables):
                raise ValueError("图表目标表必须是非空且不重复的字符串列表")
            if not set(tables).issubset(tasks[source]["target_tables"]):
                raise ValueError("图表目标表必须属于来源分析任务")
            if len(validated) == MAX_CHARTS:
                raise ValueError(f"最多保留 {MAX_CHARTS} 个合格图表")
        except ValueError as error:
            candidate = item if isinstance(item, dict) else {}
            rejected.append({"chart_id": str(candidate.get("chart_id") or "visualization_plan"),
                "source_task_id": candidate.get("source_task_id") if isinstance(candidate.get("source_task_id"), str) else None,
                "candidate_index": index + 1, "stage": "planning", "error": str(error)})
            continue
        seen.add(chart_id)
        validated.append(dict(item))
    return validated, rejected
