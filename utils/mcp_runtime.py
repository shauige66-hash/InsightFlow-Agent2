"""MCP 的运行期接线。

MCP 的 Client、Session、事件循环线程与子进程句柄都**不能进 Checkpoint**：
它们绑在当前进程上，恢复后在新进程里毫无意义，序列化还会直接失败。
因此它们和模型客户端一样放在 `RunRuntime` 里，由 `use_runtime` 按上下文注入。

数据集映射（资源名 -> dataset_id）也在这里按需构建：
它可以从 state 的快照推导出来，因此不需要额外持久化，
跨进程恢复后重新构建即可。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from mcp_bridge.contract import DatasetRef, dataset_id_for

if TYPE_CHECKING:  # pragma: no cover - 仅用于类型标注
    from ingestion.models import TableResource


def dataset_ids_for_resources(
    run_id: str,
    resources: list["TableResource"],
) -> list[str]:
    """把若干资源映射成它们在本 Run 中的 dataset_id。

    映射只由 (run_id, 资源名) 决定，与 P1 的资源名一致，
    因此同一个 Run 里重复计算结果相同，跨进程恢复也一致。
    """
    return [dataset_id_for(run_id, str(resource.name)) for resource in resources]


def dataset_refs_for_resources(
    run_id: str,
    resources: list["TableResource"],
) -> list[DatasetRef]:
    """资源 -> Server 侧授权描述。

    含宿主路径，只用于写私有配置，不返回给模型。
    """
    refs: list[DatasetRef] = []
    for resource in resources:
        refs.append(
            DatasetRef(
                dataset_id=dataset_id_for(run_id, str(resource.name)),
                name=str(resource.name),
                source_type=str(resource.source_type),
                source_path=str(Path(resource.source_path)),
                locator=resource.database_table or None,
                content_hash=resource.content_hash,
                database_table=resource.database_table or None,
                columns=sorted(resource.dataframe.columns)
                if resource.dataframe is not None
                else [],
                row_count=int(len(resource.dataframe))
                if resource.dataframe is not None
                else None,
            )
        )
    return refs


def ensure_mcp_config(
    *,
    run_id: str,
    snapshots: dict[str, Any],
    workspace_root: Path,
    output_root: Path,
    limits: dict[str, Any] | None = None,
    private_root: Path | None = None,
) -> Path:
    """按本次 Run 的权威快照写出（或刷新）MCP 授权配置。

    配置写到私有根目录（默认 `<output_root>/mcp_private/<run_id>/`），
    **不在** run 目录内——run 目录会被只读挂载进沙箱，放在那里等于把
    Server 配置交给模型代码。

    每次调用都重写：恢复后的新进程里内容指纹可能已变化，
    沿用旧配置等于绕过 P1 的"数据源变化必须失败"。
    """
    from mcp_bridge.registry import write_authorizations

    path, _refs = write_authorizations(
        run_id=run_id,
        snapshots=snapshots,
        workspace_root=workspace_root,
        output_root=output_root,
        limits=limits,
        private_root=private_root,
    )
    return path
