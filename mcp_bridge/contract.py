"""MCP 宿主与 Server 之间共享的契约。

这个模块被**两侧同时导入**，因此工具名、参数名、结果形状只有一份定义：

- 宿主侧：`mcp_bridge/client.py` 用它拼参数、校验 Schema；
- Server 侧：`mcp_bridge/server.py` 用它注册工具。

放在独立模块而不是任一侧，是为了避免"客户端按 A 拼参数、服务端按 B 解析"
这种只有集成时才暴露的错位。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

# 允许通过 MCP 暴露的工具。实际可调用工具必须在这个集合内：
# 动态发现（list_tools）只告诉我们"服务端声明了什么"，
# 不等于"我们就允许调用它"——授权来自这里的显式清单。
TOOL_DISCOVER_DATASETS = "discover_datasets"
TOOL_GET_SCHEMA = "get_schema"
TOOL_RUN_READONLY_SQL = "run_readonly_sql"

ALLOWED_MCP_TOOLS: frozenset[str] = frozenset(
    {
        TOOL_DISCOVER_DATASETS,
        TOOL_GET_SCHEMA,
        TOOL_RUN_READONLY_SQL,
    }
)

# MCP 后端在工具契约里的 error_kind 取值。
#
# 环境/协议类失败映射到 unavailable 与 timeout，让工作流短路到失败，
# 而不是让模型去"修"一个根本没通起来的 MCP 通道。
ERROR_KIND_MCP_UNAVAILABLE = "unavailable"
ERROR_KIND_MCP_TIMEOUT = "timeout"


@dataclass(frozen=True)
class DatasetRef:
    """Server 侧可见的一次 Run 授权资源。

    `dataset_id` 是服务端分配的稳定标识，也是模型唯一能引用的东西：
    模型拿不到宿主绝对路径，只能拿 id 换数据。
    """

    dataset_id: str
    name: str
    source_type: str
    source_path: str
    locator: str | None = None
    content_hash: str | None = None
    database_table: str | None = None
    columns: list[str] = field(default_factory=list)
    row_count: int | None = None


def dataset_id_for(run_id: str, name: str) -> str:
    """由 Run 与资源名派生 dataset_id。

    只由稳定输入构成，因此同一个 Run 里同一个资源每次得到同一个 id，
    跨进程恢复后仍然一致；不同 Run 之间不会撞车。
    """
    return f"{run_id}:{name}"


def public_dataset(dataset: DatasetRef) -> dict[str, Any]:
    """转成可以返回给模型的形状。

    **刻意不含 `source_path`**：模型不应看到宿主绝对路径，否则越权读取
    就只剩一道"我们记得别打开它"的软约束。需要数据时按 dataset_id 请求。
    """
    payload: dict[str, Any] = {
        "dataset_id": dataset.dataset_id,
        "name": dataset.name,
        "source_type": dataset.source_type,
        "row_count": dataset.row_count,
        "columns": list(dataset.columns),
        "content_hash": dataset.content_hash,
    }

    if dataset.database_table:
        payload["database_table"] = dataset.database_table
        payload["able_to"] = ["schema", "readonly_sql"]
    elif dataset.source_type in ("csv", "excel"):
        payload["locator"] = dataset.locator
        payload["able_to"] = ["schema"]
    else:
        payload["able_to"] = ["schema"]

    return payload


def workspace_path(workspace_root: Path, candidate: str | Path) -> Path:
    """把路径限制在 workspace 之内，越界即拒绝。

    Server 是独立进程，不能假设调用方传进来的路径可信；
    这一步是 Server 侧防任意宿主路径读取的硬边界。
    """
    root = Path(workspace_root).resolve()
    resolved = Path(candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise PermissionError(
            f"路径越界：{candidate} 不在 workspace 之内"
        ) from error
    return resolved


def registry_payload(
    *,
    run_id: str,
    workspace_root: str,
    datasets: list[DatasetRef],
) -> dict[str, Any]:
    """生成交给 Server 的私有配置内容。

    这个文件由宿主写出、只读传给 Server；模型既看不到它的内容，
    也无法通过工具参数改写它——工具参数里根本没有这个路径。
    """
    return {
        "run_id": run_id,
        "workspace_root": workspace_root,
        "datasets": [
            {
                "dataset_id": dataset.dataset_id,
                "name": dataset.name,
                "source_type": dataset.source_type,
                "source_path": dataset.source_path,
                "locator": dataset.locator,
                "content_hash": dataset.content_hash,
                "database_table": dataset.database_table,
                "columns": list(dataset.columns),
                "row_count": dataset.row_count,
            }
            for dataset in datasets
        ],
    }


def dataset_from_payload(payload: dict[str, Any]) -> DatasetRef:
    return DatasetRef(
        dataset_id=str(payload["dataset_id"]),
        name=str(payload["name"]),
        source_type=str(payload["source_type"]),
        source_path=str(payload["source_path"]),
        locator=payload.get("locator"),
        content_hash=payload.get("content_hash"),
        database_table=payload.get("database_table"),
        columns=list(payload.get("columns") or []),
        row_count=payload.get("row_count"),
    )


# 各工具的参数名。客户端与 Server 共用，避免拼写漂移。
PARAM_DATASET_ID = "dataset_id"
PARAM_DATASET_IDS = "dataset_ids"
PARAM_QUERY = "query"

ToolKind = Literal["discover", "schema", "sql"]
