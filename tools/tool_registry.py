"""工具的统一调用入口。

各工具自身在参数非法时会 `raise`，而执行失败时会返回错误字典。
调用方如果两种都要处理，很容易漏掉前者，让一次本可修复的调用
直接终止整条工作流。这里把两类结果统一成同一种形状：

    {"success": bool, "result": Any, "error": str | None}

另外按「工具名 + 数据表指纹」缓存工具实例。原先每次调用都重建工具闭包，
在修复循环里会反复构造。

P2-A：`python` 与 `plot` 的执行都发生在一次性 Docker 容器中，本模块不含
任何同进程执行模型代码的路径。
"""

from __future__ import annotations

from typing import Any, Callable

from ingestion.models import TableResource
from tools.plot_tool import create_plot_tool
from tools.python_tool import create_python_tool
from tools.sql_tool import create_sql_tool

Result = dict[str, Any]

# 工具实例缓存。键包含表资源指纹（名 + 路径 + 内容哈希），
# 因此换数据集或同一路径换内容都不会拿到过期的工具。
_TOOL_CACHE: dict[tuple[str, tuple[str, ...]], Any] = {}
_CACHE_LIMIT = 32


def _tables_fingerprint(tables: dict[str, TableResource]) -> tuple[str, ...]:
    """表的身份由资源名、来源路径与内容哈希共同决定。

    只带路径是不够的：同一路径下文件内容变化时（测试里很常见），
    缓存键必须跟着变，否则会复用基于旧 DataFrame 构造的工具实例。
    内容哈希来自 P1 的 TableSnapshot，与该层的指纹语义保持一致。
    """
    return tuple(
        sorted(
            f"{name}@{resource.source_path}@{resource.content_hash or '-'}"
            for name, resource in tables.items()
        )
    )


TOOL_BUILDERS: dict[str, Callable[[dict[str, TableResource]], Any]] = {
    "python": create_python_tool,
    "sql": create_sql_tool,
    "plot": create_plot_tool,
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


def _failure(error: Exception, error_kind: str | None = None) -> Result:
    return {
        "success": False,
        "result": None,
        "error": str(error),
        "error_kind": error_kind,
    }


def _normalize(outcome: Any) -> Result:
    if not isinstance(outcome, dict):
        return {
            "success": False,
            "result": None,
            "error": "工具返回了非字典结果",
            "error_kind": None,
        }

    return {
        "success": bool(outcome.get("success", False)),
        "result": outcome.get("result"),
        "error": outcome.get("error"),
        # 错误分类必须原样穿过这一层：下游路由靠它决定是否进入修复循环
        "error_kind": outcome.get("error_kind"),
    }


def invoke_tool(
    tool_name: str,
    tables: dict[str, TableResource],
    arguments: dict[str, Any],
) -> Result:
    """调用工具并保证不抛异常。

    参数校验失败与执行失败在调用方看来都是 `success=False`，
    但 `error_kind` 保留了"是否值得重试"的信息。
    """
    if tool_name not in TOOL_BUILDERS:
        return _failure(ValueError(f"Unsupported tool: {tool_name}"), "unsupported_tool")

    try:
        return _normalize(_get_tool(tool_name, tables).invoke(arguments))
    except Exception as error:  # noqa: BLE001 - 工具失败统一转为结果
        return _failure(error)


def clear_tool_cache() -> None:
    """供测试隔离使用。"""
    _TOOL_CACHE.clear()
