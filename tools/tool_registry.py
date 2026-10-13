"""工具的统一调用入口。

各工具自身在参数非法时会 `raise`，而执行失败时会返回错误字典。
调用方如果两种都要处理，很容易漏掉前者，让一次本可修复的调用
直接终止整条工作流。这里把两类结果统一成同一种形状：

    {"success": bool, "result": Any, "error": str | None}

另外按「工具名 + 数据表指纹」缓存工具实例。原先每次调用都重建工具闭包，
在修复循环里会反复构造。
"""

from __future__ import annotations

from typing import Any, Callable

from ingestion.models import TableResource
from tools.python_tool import create_python_tool
from tools.sql_tool import create_sql_tool

Result = dict[str, Any]

# 工具实例缓存。键包含表资源指纹，因此换数据集不会拿到过期的工具。
_TOOL_CACHE: dict[tuple[str, tuple[str, ...]], Any] = {}
_CACHE_LIMIT = 32


def _tables_fingerprint(tables: dict[str, TableResource]) -> tuple[str, ...]:
    """表的身份由资源名与来源路径共同决定。"""
    return tuple(
        sorted(f"{name}@{resource.source_path}" for name, resource in tables.items())
    )


TOOL_BUILDERS: dict[str, Callable[[dict[str, TableResource]], Any]] = {
    "python": create_python_tool,
    "sql": create_sql_tool,
}


def available_tools() -> tuple[str, ...]:
    return tuple(TOOL_BUILDERS)


def _get_tool(tool_name: str, tables: dict[str, TableResource]) -> Any:
    builder = TOOL_BUILDERS[tool_name]
    key = (tool_name, _tables_fingerprint(tables))

    tool = _TOOL_CACHE.get(key)
    if tool is None:
        tool = builder(tables)
        if len(_TOOL_CACHE) >= _CACHE_LIMIT:
            _TOOL_CACHE.pop(next(iter(_TOOL_CACHE)))
        _TOOL_CACHE[key] = tool

    return tool


def _failure(error: Exception) -> Result:
    return {"success": False, "result": None, "error": str(error)}


def _normalize(outcome: Any) -> Result:
    if not isinstance(outcome, dict):
        return {"success": False, "result": None, "error": "工具返回了非字典结果"}

    return {
        "success": bool(outcome.get("success", False)),
        "result": outcome.get("result"),
        "error": outcome.get("error"),
    }


def invoke_tool(
    tool_name: str,
    tables: dict[str, TableResource],
    arguments: dict[str, Any],
) -> Result:
    """调用工具并保证不抛异常。

    参数校验失败与执行失败在调用方看来是同一种可重试的失败，
    因此统一返回错误字典。
    """
    if tool_name not in TOOL_BUILDERS:
        return _failure(ValueError(f"Unsupported tool: {tool_name}"))

    try:
        return _normalize(_get_tool(tool_name, tables).invoke(arguments))
    except Exception as error:  # noqa: BLE001 - 工具失败统一转为结果
        return _failure(error)


def clear_tool_cache() -> None:
    """供测试隔离使用。"""
    _TOOL_CACHE.clear()
