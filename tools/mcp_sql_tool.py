# 经 MCP 调用 SQL 的工具。
#
# 与 `tools/sql_tool.py` 的关系：那个是 direct 后端，直接在宿主进程执行；
# 这个是 mcp 后端，把请求交给独立 Server 子进程，由 Server 复用**同一份**
# SQL 策略（`utils/sql_policy`）。两者返回的 `result` 结构完全一致，
# 因此上层节点不需要知道本次走的是哪条路径。
#
# 关键约束：MCP 通道不可用、启动失败或超时，就是 unavailable/timeout，
# **绝不**退回到 direct 执行。回退等于在用户没有选择的情况下换了执行路径。

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from core.runtime import RuntimeContextMissing, current_runtime
from ingestion.models import TableResource
from mcp_bridge.client import (
    McpStdioClient,
    McpTimeout,
    McpUnavailable,
    reap_process,
)
from mcp_bridge.contract import (
    ERROR_KIND_MCP_TIMEOUT,
    ERROR_KIND_MCP_UNAVAILABLE,
    TOOL_RUN_READONLY_SQL,
)
from utils.mcp_runtime import dataset_ids_for_resources, ensure_mcp_config


def _mcp_crash_point(name: str) -> None:
    """故障注入点，仅供测试使用。

    与 `nodes._crash_point` 同样的约定：环境变量命中即强杀进程。
    放在这里是因为"外部 MCP 调用已完成、节点尚未提交"这个瞬间只在本模块内
    可见——它正是验证跨进程恢复语义的关键位置。
    """
    import os

    if os.environ.get("INSIGHTFLOW_CRASH_AT") == name:
        os._exit(17)


def call_mcp_sql(
    *,
    run_id: str,
    snapshots: dict[str, Any],
    dataset_ids: list[str],
    query: str,
    tables: dict[str, TableResource],
) -> dict[str, Any]:
    """经 MCP 执行一次只读 SQL，返回工具契约形状的结果。

    抽成独立函数，便于在没有 LangChain 包装的情况下直接测试，
    也避免把"会话生命周期"藏在工具闭包里看不见。

    会话按调用创建、按调用销毁：MCP 会话不能进 Checkpoint，
    长驻一个共享会话会让恢复语义变得依赖进程状态。
    """
    runtime = current_runtime()

    try:
        config_path = ensure_mcp_config(
            run_id=run_id,
            snapshots=snapshots,
            workspace_root=runtime.workspace_root(),
            output_root=runtime.output_root,
            # 私有根目录在 run 目录之外：run 目录会被只读挂载进沙箱作为 /history，
            # 放在那里模型代码能直接读到宿主绝对路径（实测可复现）
            private_root=runtime.mcp_private_root(),
        )
    except Exception as error:  # noqa: BLE001
        # 写不出私有配置同样是"通道不可用"，必须归类。
        # 若让它裸抛，LangChain 会把它变成一条无分类的错误，
        # 下游路由就会误以为这是可以靠改代码修复的问题。
        return _failure(f"MCP 配置不可用：{error}", ERROR_KIND_MCP_UNAVAILABLE)

    client = McpStdioClient(
        config_path=config_path,
        server_module=runtime.mcp.server_module,
        project_root=Path(__file__).resolve().parent.parent,
        startup_timeout=runtime.mcp.startup_timeout,
        call_timeout=runtime.mcp.call_timeout,
    )

    try:
        client.start()
    except McpTimeout as error:
        client.close()
        return _failure(str(error), ERROR_KIND_MCP_TIMEOUT)
    except McpUnavailable as error:
        client.close()
        return _failure(str(error), ERROR_KIND_MCP_UNAVAILABLE)
    except Exception as error:  # noqa: BLE001 - 一律归类为通道不可用
        client.close()
        return _failure(f"MCP 启动失败：{error}", ERROR_KIND_MCP_UNAVAILABLE)

    try:
        try:
            # 动态发现不等于自动授权：只调用允许清单内的工具
            if TOOL_RUN_READONLY_SQL not in client.allowed_tools():
                return _failure(
                    f"MCP Server 未提供允许的工具：{TOOL_RUN_READONLY_SQL}",
                    ERROR_KIND_MCP_UNAVAILABLE,
                )

            outcome = client.call_tool(
                TOOL_RUN_READONLY_SQL,
                {"query": query, "dataset_ids": dataset_ids},
            )
        except McpTimeout as error:
            # 超时会话已被标记失效，这里必须显式归类成 timeout，
            # 否则会被当成 SQL 写错而去触发无意义的修复轮次。
            return _failure(str(error), ERROR_KIND_MCP_TIMEOUT)
        except McpUnavailable as error:
            return _failure(str(error), ERROR_KIND_MCP_UNAVAILABLE)
        except Exception as error:  # noqa: BLE001
            return _failure(f"MCP 调用失败：{error}", ERROR_KIND_MCP_UNAVAILABLE)
    finally:
        pid = client._child_pid
        client.close()
        # 兜底：正常路径已被 SDK 的 Job Object 回收；万一关闭阶段出错，
        # 这里确保不留下孤儿进程。
        if pid:
            reap_process(pid)

    # 故障注入点：MCP 调用**已经完成**、但本节点的 state 增量还没提交。
    # 用来验证"外部工具副作用与 Checkpoint 不在同一个事务里"时的恢复行为。
    _mcp_crash_point("after_mcp_call")

    if not outcome["success"]:
        return outcome

    payload = outcome["result"]
    if not isinstance(payload, dict):
        return _failure("MCP 返回了非结构化结果", "invalid_result")

    return {
        "success": True,
        "result": {
            "columns": payload.get("columns") or [],
            "rows": payload.get("rows") or [],
            "row_count": payload.get("row_count"),
            "bytes": payload.get("bytes"),
            "truncated": payload.get("truncated"),
        },
        "error": None,
        "error_kind": None,
    }


def _failure(error: str, error_kind: str) -> dict[str, Any]:
    return {"success": False, "result": None, "error": error, "error_kind": error_kind}


def create_mcp_sql_tool(
    tables: dict[str, TableResource],
    *,
    snapshots: dict[str, Any] | None = None,
):
    """构造经 MCP 执行 SQL 的工具。"""

    @tool
    def run_sql_query_via_mcp(
        table_names: list[str],
        query: str
    ) -> dict[str, Any]:
        """经 MCP Server 对已授权的 SQLite 数据表执行只读查询。"""

        if not table_names:
            raise ValueError("At least one table is required.")

        try:
            runtime = current_runtime()
        except RuntimeContextMissing:
            return _failure(
                "MCP SQL 调用需要运行期上下文", ERROR_KIND_MCP_UNAVAILABLE
            )

        resources: list[TableResource] = []
        for name in table_names:
            resource = tables.get(name)
            if resource is None:
                raise ValueError(f"Table not found: {name}")
            if resource.source_type != "sqlite":
                raise ValueError(f"Table is not a SQLite resource: {name}")
            resources.append(resource)

        # 授权沿用 P1 的权威快照；没有快照时按本次点名的资源构造，
        # 这样工具在单元测试里也能独立使用。
        authoritative = snapshots if snapshots is not None else _snapshots_of(resources)

        return call_mcp_sql(
            run_id=runtime.run_id,
            snapshots=authoritative,
            dataset_ids=dataset_ids_for_resources(runtime.run_id, resources),
            query=query.strip(),
            tables=tables,
        )

    return run_sql_query_via_mcp


def _snapshots_of(resources: list[TableResource]) -> dict[str, Any]:
    """从资源推导最小快照集合（仅用于缺少 state 快照的场景）。"""
    snapshots: dict[str, Any] = {}
    for resource in resources:
        snapshots[str(resource.name)] = {
            "name": str(resource.name),
            "source_type": str(resource.source_type),
            "source_path": str(resource.source_path),
            "content_hash": resource.content_hash or "",
            "file_size": 0,
            "locator": resource.database_table,
            "database_table": resource.database_table,
        }
    return snapshots
