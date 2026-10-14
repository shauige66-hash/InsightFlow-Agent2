"""表资源的可序列化快照与解析器。

P1 需要把执行状态写进 Checkpoint，但 `TableResource` 里含 pandas DataFrame，
既无法 JSON 编码，也不该跨进程恢复。因此把"表在哪里、是什么"与"表的实际内容"
分开：

- `TableSnapshot`：只含可序列化元数据（来源路径、内容指纹、sheet / 表名）。
  它是 state 中唯一与表相关的字段。
- `ResourceResolver`：按快照把真实 DataFrame 读回来，并校验来源是否变化。

校验用内容指纹而不是 mtime：mtime 在检出、复制、容器挂载后都可能变，
用它会误判为"数据已变化"；内容哈希只在内容真的不同时才失败。

数据源在快照之后被改动时，恢复会明确失败而不是静默使用新数据——
静默换数据会让历史 Run 的结论无法复现。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ingestion.models import TableResource

# 计算内容指纹时的分块大小，避免一次把大文件读进内存
_HASH_CHUNK = 1024 * 1024


class ResourceUnavailableError(RuntimeError):
    """表资源无法按快照恢复。"""

    def __init__(self, resource_name: str, reason: str) -> None:
        super().__init__(f"无法恢复表资源 {resource_name!r}：{reason}")
        self.resource_name = resource_name
        self.reason = reason


@dataclass(frozen=True)
class TableSnapshot:
    """一张表的可序列化身份信息。"""

    name: str
    source_type: str
    source_path: str
    content_hash: str
    file_size: int
    # Excel 的 sheet 名或 SQLite 的表名；CSV 为 None
    locator: str | None = None
    database_table: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TableSnapshot":
        return cls(
            name=payload["name"],
            source_type=payload["source_type"],
            source_path=payload["source_path"],
            content_hash=payload["content_hash"],
            file_size=int(payload.get("file_size", 0)),
            locator=payload.get("locator"),
            database_table=payload.get("database_table"),
        )


def file_content_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _detect_source_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".xlsx":
        return "excel"
    if suffix in {".db", ".sqlite", ".sqlite3"}:
        return "sqlite"
    raise ValueError(f"Unsupported data source: {path.suffix}")


def _sqlite_table_names(path: Path) -> list[str]:
    with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True) as connection:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
    return [row[0] for row in rows]


def verify_recorded_sources(
    recorded: dict[str, dict[str, Any]],
) -> None:
    """校验已有快照的来源是否仍然可用。

    为什么必须显式校验：`build_snapshots` 会**重新计算**指纹，因此
    "重新加载"本身发现不了数据变化——它只会得到一份新的快照，
    让历史 Run 在不知不觉中用了新数据。恢复路径必须在重建快照之前
    拿记录中的指纹与当前文件比对。

    任何一条不一致就抛错，避免部分表用新数据、部分用旧数据。
    """
    snapshots = snapshots_from_state(recorded)

    for snapshot in snapshots.values():
        _verify_source(snapshot)


def build_snapshots(data_sources: list[str]) -> dict[str, TableSnapshot]:
    """为一批数据源建立快照，键与 `load_data_sources` 的资源名保持一致。"""
    snapshots: dict[str, TableSnapshot] = {}

    for source in data_sources:
        path = Path(source)

        if not path.exists():
            raise FileNotFoundError(f"Data source not found: {path}")
        if not path.is_file():
            raise IsADirectoryError(f"Data source is not a file: {path}")

        source_type = _detect_source_type(path)
        digest = file_content_hash(path)
        size = path.stat().st_size

        if source_type == "csv":
            table_name = path.stem
            snapshots[table_name] = TableSnapshot(
                name=table_name,
                source_type="csv",
                source_path=str(path),
                content_hash=digest,
                file_size=size,
            )

        elif source_type == "excel":
            sheets = pd.read_excel(path, sheet_name=None)
            for sheet_name in sheets:
                table_name = f"{path.stem}.{sheet_name}"
                snapshots[table_name] = TableSnapshot(
                    name=table_name,
                    source_type="excel",
                    source_path=str(path),
                    content_hash=digest,
                    file_size=size,
                    locator=str(sheet_name),
                )

        else:
            for table_name in _sqlite_table_names(path):
                key = f"{path.stem}.{table_name}"
                snapshots[key] = TableSnapshot(
                    name=key,
                    source_type="sqlite",
                    source_path=str(path),
                    content_hash=digest,
                    file_size=size,
                    locator=table_name,
                    database_table=table_name,
                )

    return snapshots


def _verify_source(snapshot: TableSnapshot) -> Path:
    """确认来源文件仍然存在且内容未变。"""
    path = Path(snapshot.source_path)

    if not path.exists():
        raise ResourceUnavailableError(
            snapshot.name, f"来源文件已不存在：{path}"
        )
    if not path.is_file():
        raise ResourceUnavailableError(
            snapshot.name, f"来源路径不是文件：{path}"
        )

    try:
        current = file_content_hash(path)
    except PermissionError as error:
        raise ResourceUnavailableError(
            snapshot.name, f"无权限读取来源文件：{path}"
        ) from error

    if current != snapshot.content_hash:
        raise ResourceUnavailableError(
            snapshot.name,
            "来源内容已变化，拒绝用新数据恢复历史 Run"
            f"（快照 {snapshot.content_hash[:12]}… 当前 {current[:12]}…）",
        )

    return path


def _build_from_snapshot(snapshot: TableSnapshot) -> TableResource:
    path = _verify_source(snapshot)

    if snapshot.source_type == "csv":
        frame = pd.read_csv(path)
        return TableResource(
            name=snapshot.name,
            source_type="csv",
            source_path=str(path),
            dataframe=frame,
        )

    if snapshot.source_type == "excel":
        sheets = pd.read_excel(path, sheet_name=None)
        if snapshot.locator not in sheets:
            raise ResourceUnavailableError(
                snapshot.name, f"Excel 中已找不到 sheet：{snapshot.locator}"
            )
        return TableResource(
            name=snapshot.name,
            source_type="excel",
            source_path=str(path),
            dataframe=sheets[snapshot.locator],
        )

    if snapshot.source_type == "sqlite":
        # 校验表仍存在，避免执行阶段才以 SQL 错误的形式暴露
        if snapshot.database_table not in _sqlite_table_names(path):
            raise ResourceUnavailableError(
                snapshot.name,
                f"SQLite 中已找不到表：{snapshot.database_table}",
            )
        return TableResource(
            name=snapshot.name,
            source_type="sqlite",
            source_path=str(path),
            database_table=snapshot.database_table,
        )

    raise ResourceUnavailableError(
        snapshot.name, f"未知的资源类型：{snapshot.source_type}"
    )


def _cache_key(snapshot: TableSnapshot) -> str:
    """缓存键包含内容指纹，来源一变就不会命中旧资源。"""
    return f"{snapshot.source_path}|{snapshot.content_hash}|{snapshot.name}"


class ResourceResolver:
    """按快照解析出真实表资源，并缓存已加载的对象。

    缓存以「来源路径 + 内容指纹 + 资源名」为键：内容变了自然失效，
    需要强制丢弃（例如测试之间）时调用 `invalidate_all`。
    """

    def __init__(self) -> None:
        self._cache: dict[str, TableResource] = {}

    def resolve(self, snapshot: TableSnapshot) -> TableResource:
        key = _cache_key(snapshot)

        cached = self._cache.get(key)
        if cached is not None:
            return cached

        resource = _build_from_snapshot(snapshot)
        self._cache[key] = resource
        return resource

    def resolve_all(
        self, snapshots: dict[str, TableSnapshot]
    ) -> dict[str, TableResource]:
        return {name: self.resolve(snapshot) for name, snapshot in snapshots.items()}

    def invalidate_all(self) -> None:
        self._cache.clear()

    @property
    def cached_count(self) -> int:
        return len(self._cache)


# 进程内共享一个解析器：同一个 Run 的多次节点调用复用同一份 DataFrame，
# 避免每个节点都重新读盘。资源本身按内容指纹隔离，因此跨 Run 也安全。
_default_resolver = ResourceResolver()


def get_resolver() -> ResourceResolver:
    return _default_resolver


def snapshots_to_state(
    snapshots: dict[str, TableSnapshot],
) -> dict[str, dict[str, Any]]:
    """转成可直接放进 state 的普通 dict（Checkpoint 只接受可序列化值）。"""
    return {name: snapshot.to_dict() for name, snapshot in snapshots.items()}


def snapshots_from_state(
    payload: dict[str, dict[str, Any]] | None,
) -> dict[str, TableSnapshot]:
    if not payload:
        return {}
    return {name: TableSnapshot.from_dict(item) for name, item in payload.items()}


def snapshots_to_json(snapshots: dict[str, TableSnapshot]) -> str:
    return json.dumps(snapshots_to_state(snapshots), ensure_ascii=False, sort_keys=True)
