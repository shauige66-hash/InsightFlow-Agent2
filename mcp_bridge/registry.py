"""Run 级数据集授权。

MCP Server 能访问什么，完全由这里生成的私有配置决定：

- dataset_id 由服务端按 `run_id + 资源名` 派生，稳定且跨 Run 不撞车；
- 配置写到 `<output_root>/<run_id>/mcp/` 这个**受控私有位置**——
  既不在 Docker 沙箱的可写目录（staging）里，也不是模型工具参数能触及的路径；
- 配置里含宿主路径（Server 需要它才能打开文件），因此这个文件本身不是给模型看的；
  工具返回给模型的数据集描述里**不含路径**。

内容指纹沿用 P1 的语义：`TableSnapshot.content_hash` 决定资源身份，
同一路径内容变化时必须被视为另一份数据，而不是同一份数据的更新。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ingestion.resources import TableSnapshot
from mcp_bridge.contract import (
    DatasetRef,
    dataset_id_for,
    registry_payload,
)

# 配置文件名固定：宿主写出、Server 读取，双方都不接受调用方覆盖
CONFIG_FILENAME = "authorizations.json"

# 私有配置根目录相对 output_root 的后缀。
#
# 用**后缀**而不是固定子目录名（早期是 `output_root/mcp_private`）：
# `run_id` 由调用方指定，而 `api_runs.RUN_ID_PATTERN` 合法接受 `mcp_private`。
# 一旦某个 Run 的 id 就叫这个名字，它的 run 目录就等于私有根目录，
# 而被只读挂载进容器的 `/history` 正是 run 目录——于是**所有 Run 的私有配置**
# 都会暴露给该 Run 的沙箱（实测可复现）。平级目录从结构上消除了这种重叠。
PRIVATE_ROOT_SUFFIX = ".mcp_private"


def mcp_private_root(output_root: Path) -> Path:
    """私有配置根目录（与 output_root 平级）。"""
    return Path(str(output_root) + PRIVATE_ROOT_SUFFIX)


def _normalized(path: Path) -> str:
    """规范化路径用于归属比较：展开链接与 `..`，并统一大小写。

    Windows 上大小写不敏感而 `is_relative_to` 按字符串比较，
    不做 `normcase` 会用 `OUTPUTS` 这类别名绕过检查。
    """
    return os.path.normcase(str(Path(path).resolve()))


def _is_inside(candidate: Path, parent: Path) -> bool:
    normalized_candidate = _normalized(candidate)
    normalized_parent = _normalized(parent)
    if normalized_candidate == normalized_parent:
        return True
    return normalized_candidate.startswith(
        normalized_parent.rstrip("\\/") + os.sep
    )


def ensure_private_root_outside(
    private_root: Path, output_root: Path
) -> Path:
    """确认私有根目录不落在 output_root 之内。

    output_root 的整棵目录树都会被按 Run 只读挂载进沙箱，
    因此私有配置放在里面就等于交给模型代码。
    """
    if _is_inside(private_root, output_root):
        raise PermissionError(
            f"私有配置根目录不能位于 output_root 之内："
            f"{private_root} 落在 {output_root} 里"
        )
    return Path(private_root)


def mcp_config_dir(
    output_root: Path, run_id: str, *, private_root: Path | None = None
) -> Path:
    """本次 Run 的 MCP 私有配置目录。"""
    root = (
        Path(private_root)
        if private_root is not None
        else mcp_private_root(output_root)
    )
    ensure_private_root_outside(root, Path(output_root))
    return root / run_id


def config_path(
    output_root: Path, run_id: str, *, private_root: Path | None = None
) -> Path:
    return mcp_config_dir(output_root, run_id, private_root=private_root) / CONFIG_FILENAME


def dataset_refs_from_snapshots(
    run_id: str,
    snapshots: dict[str, Any],
) -> list[DatasetRef]:
    """把本次 Run 的权威快照转换成 Server 侧的授权资源。

    `snapshots` 是 state 里的 `table_snapshot`（`name -> 快照字典`），
    由 P1 的 Loader 写入，带内容指纹。
    """
    refs: list[DatasetRef] = []

    for name, raw in sorted(snapshots.items()):
        snapshot = (
            raw if isinstance(raw, TableSnapshot) else TableSnapshot.from_dict(raw)
        )
        refs.append(
            DatasetRef(
                dataset_id=dataset_id_for(run_id, snapshot.name),
                name=snapshot.name,
                source_type=snapshot.source_type,
                source_path=str(snapshot.source_path),
                locator=snapshot.locator,
                content_hash=snapshot.content_hash,
                database_table=snapshot.database_table,
                columns=[],
                row_count=None,
            )
        )

    return refs


def write_authorizations(
    *,
    run_id: str,
    snapshots: dict[str, Any],
    workspace_root: Path,
    output_root: Path,
    limits: dict[str, Any] | None = None,
    private_root: Path | None = None,
) -> tuple[Path, list[DatasetRef]]:
    """写出 Server 的授权配置，返回 (配置路径, 授权资源列表)。

    配置落在**私有根目录**下（默认 `<output_root>/mcp_private/<run_id>/`），
    不放在 run 目录里：run 目录会被只读挂载进沙箱作为 /history，
    放在那里模型代码能直接读到宿主绝对路径（实测可复现）。

    每次启动 MCP 都重写：恢复后的新进程里内容指纹可能已经变化，
    沿用旧配置等于绕过 P1 的"数据源变化必须失败"语义。
    """
    refs = dataset_refs_from_snapshots(run_id, snapshots)

    payload = registry_payload(
        run_id=run_id,
        # workspace 是 Server 允许触达的边界，越界路径一律拒绝
        workspace_root=str(Path(workspace_root).resolve()),
        datasets=refs,
    )
    if limits:
        payload["limits"] = limits

    target = config_path(output_root, run_id, private_root=private_root)
    _prepare_private_dir(target)
    _write_config(target, payload)

    return target, refs


def _prepare_private_dir(target: Path) -> None:
    """建立私有目录，并拒绝被符号链接顶替。

    私有配置是"服务端信任的输入"：如果 mcp_private 或其中某一级被替换成
    指向别处的符号链接，写出去的配置就可能落到我们以为之外的位置。
    因此写之前逐级确认不是链接。
    """
    target.parent.mkdir(parents=True, exist_ok=True)

    for candidate in (target.parent, target):
        if candidate.is_symlink():
            raise PermissionError(f"私有配置路径是符号链接，拒绝写入：{candidate}")


def _write_config(target: Path, payload: dict[str, Any]) -> None:
    """原子写出配置。

    先写临时文件再 `os.replace`：避免读到写了一半的半截 JSON。
    临时文件放在同目录下，保证 replace 是同一文件系统内的重命名。
    """
    import os

    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, target)
