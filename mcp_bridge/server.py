"""MCP Server：把一次 Run 的授权资源以三个只读工具暴露出去。

进程模型：由宿主以独立子进程启动，经 stdio 与 MCP Client 通信。
**它不是沙箱**——这里是宿主上的普通 Python 进程，权限与启动它的用户相同。
它的安全边界来自两件事，而不是来自进程隔离：

1. 授权只有一份：可访问哪些资源由宿主写出的私有配置决定，
   工具参数里没有路径，模型只能按 dataset_id 引用；
2. 执行复用 `utils/sql_policy`：Authorizer、只读连接、查询预算与结果上限
   与 direct 工具是同一份实现，不存在"MCP 绕过加固"的可能。

stdout 纪律：MCP 1.x 没有把 fd1 从协议通道分离出来，stdout 上的任何普通文本
都会被当成协议消息。因此本进程所有日志一律写 stderr，工具函数不得打印。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from mcp.server.fastmcp import FastMCP

from mcp_bridge.contract import (
    TOOL_DISCOVER_DATASETS,
    TOOL_GET_SCHEMA,
    TOOL_RUN_READONLY_SQL,
    DatasetRef,
    dataset_from_payload,
    public_dataset,
    workspace_path,
)
from utils.sql_policy import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_ROWS,
    DEFAULT_TIMEOUT_SECONDS,
    SqlPolicyError,
    SqlTimeoutError,
    execute_readonly_sql,
)


class ResourceNotAuthorized(PermissionError):
    """请求的 dataset_id 不在本次 Run 的授权集合内。"""


class ResourceUnavailable(RuntimeError):
    """授权资源存在，但文件已丢失或内容已变化。"""


class RunAuthorizations:
    """一次 Run 的授权资源集合。

    每个 Server 实例只加载一份，且只加载属于它这次 Run 的配置。
    刻意不做任何"按名字去别处找找看"的兜底：找不到就是拒绝。
    """

    def __init__(
        self,
        *,
        run_id: str,
        workspace_root: Path,
        datasets: dict[str, DatasetRef],
        max_rows: int = DEFAULT_MAX_ROWS,
        max_bytes: int = DEFAULT_MAX_BYTES,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.run_id = run_id
        self.workspace_root = Path(workspace_root).resolve()
        self.datasets = datasets
        self.max_rows = max_rows
        self.max_bytes = max_bytes
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> "RunAuthorizations":
        workspace_root = Path(payload["workspace_root"]).resolve()
        datasets = {
            item["dataset_id"]: dataset_from_payload(item)
            for item in payload.get("datasets", [])
        }
        limits = payload.get("limits") or {}
        return cls(
            run_id=str(payload["run_id"]),
            workspace_root=workspace_root,
            datasets=datasets,
            max_rows=int(limits.get("max_rows", DEFAULT_MAX_ROWS)),
            max_bytes=int(limits.get("max_bytes", DEFAULT_MAX_BYTES)),
            timeout_seconds=float(
                limits.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
            ),
        )

    def resolve(self, dataset_id: str) -> DatasetRef:
        """按 id 取授权资源，并校验路径与内容都仍然合法。

        未知 id 与越权 id 走同一条拒绝路径，并且**不透露**授权集合里有哪些 id
        （错误信息不回显请求值之外的任何东西）。
        """
        dataset = self.datasets.get(dataset_id)
        if dataset is None:
            raise ResourceNotAuthorized(f"未授权的 dataset_id：{dataset_id}")

        # 路径必须在 workspace 之内：配置由宿主写出，但仍然按不可信输入处理，
        # 避免配置被改动后变成任意宿主路径读取。
        path = workspace_path(self.workspace_root, dataset.source_path)

        if not path.exists():
            raise ResourceUnavailable(f"数据源已丢失：{dataset_id}")

        return dataset

    def resolve_many(self, dataset_ids: list[str]) -> list[DatasetRef]:
        if not dataset_ids:
            raise ResourceNotAuthorized("必须指定 dataset_ids")
        return [self.resolve(dataset_id) for dataset_id in dataset_ids]

    def public_listing(self) -> list[dict[str, Any]]:
        return [public_dataset(dataset) for dataset in self.datasets.values()]


def _describe_columns(rows: list[Any], columns: list[str]) -> list[dict[str, Any]]:
    """对少量样本推断列类型。

    只做"看起来像什么"的粗推断，不声称是严格类型系统：
    目的是让模型知道该怎么写查询，而不是充当 schema 校验器。
    """

    def infer(values: list[Any]) -> str:
        seen = {type(value).__name__ for value in values if value is not None}
        if not seen:
            return "null"
        if seen <= {"int"}:
            return "integer"
        if seen <= {"int", "float"}:
            return "number"
        if seen <= {"bool"}:
            return "boolean"
        return "text"

    described: list[dict[str, Any]] = []
    for index, column in enumerate(columns):
        values = [
            row.get(column) if isinstance(row, dict) else None for row in rows
        ]
        described.append(
            {
                "name": column,
                "inferred_type": infer(values),
                "sample_values": [value for value in values[:3]],
            }
        )
    return described


def _read_tabular(dataset: DatasetRef) -> list[dict[str, Any]]:
    """读取 CSV / Excel 的少量样本用于 schema 推断。

    按 dataset 自身声明的类型与 locator 读，不接受调用方给的任何路径或 sheet 名。

    pandas 在模块顶层导入、不在这里导入：FastMCP 把同步工具函数放到工作线程里
    执行，如果这时才首次导入 pandas，实测会在 CSV 路径上**永久卡住**
    （SQLite 路径不走 pandas，所以看起来"只有 CSV 有问题"）。
    导入放在模块加载期，就发生在 stdio 循环启动之前。
    """
    path = Path(dataset.source_path)

    if dataset.source_type == "csv":
        frame = pd.read_csv(path, nrows=50)
    elif dataset.source_type == "excel":
        sheets = pd.read_excel(path, sheet_name=None, nrows=50)
        locator = dataset.locator
        if locator and locator in sheets:
            frame = sheets[locator]
        elif sheets:
            frame = next(iter(sheets.values()))
        else:
            frame = pd.DataFrame()
    else:
        raise ResourceUnavailable(f"不支持的资源类型：{dataset.source_type}")

    return frame.to_dict(orient="records")


def build_server(auth: RunAuthorizations) -> FastMCP:
    """构造绑定到一份授权集合的 MCP Server。

    每个 Run 一个实例：工具闭包捕获 `auth`，因此不存在"换个参数访问别的 Run"
    的入口。
    """
    server = FastMCP("insightflow-mcp")

    @server.tool(name=TOOL_DISCOVER_DATASETS)
    def discover_datasets() -> dict[str, Any]:
        """列出本次 Run 已授权的数据资源。"""
        return {
            "run_id": auth.run_id,
            "datasets": auth.public_listing(),
            "count": len(auth.datasets),
        }

    @server.tool(name=TOOL_GET_SCHEMA)
    def get_schema(dataset_id: str) -> dict[str, Any]:
        """按 dataset_id 查询数据结构。"""
        dataset = auth.resolve(dataset_id)

        if dataset.source_type == "sqlite":
            return _sqlite_schema(auth, dataset)

        rows = _read_tabular(dataset)
        columns = list(dataset.columns) or (
            [str(key) for key in rows[0]] if rows else []
        )
        return {
            "dataset_id": dataset.dataset_id,
            "name": dataset.name,
            "source_type": dataset.source_type,
            "columns": _describe_columns(rows, columns),
            "sampled_rows": len(rows),
            "row_count": dataset.row_count,
        }

    @server.tool(name=TOOL_RUN_READONLY_SQL)
    def run_readonly_sql(
        query: str, dataset_ids: list[str] | None = None
    ) -> dict[str, Any]:
        """对已授权的 SQLite 数据表执行只读查询。"""
        selected = auth.resolve_many(list(dataset_ids or []))

        sqlite_datasets = [
            dataset for dataset in selected if dataset.source_type == "sqlite"
        ]
        if not sqlite_datasets:
            raise ResourceUnavailable("指定的资源不是 SQLite 数据源")

        database_paths = {
            str(Path(dataset.source_path).resolve()) for dataset in sqlite_datasets
        }
        if len(database_paths) != 1:
            raise ResourceUnavailable(
                "跨库查询不允许：所有数据表必须来自同一个 SQLite 文件"
            )

        allowed_tables = frozenset(
            dataset.database_table or dataset.name for dataset in sqlite_datasets
        )

        try:
            outcome = execute_readonly_sql(
                next(iter(database_paths)),
                str(query),
                allowed_tables=allowed_tables,
                max_rows=auth.max_rows,
                max_bytes=auth.max_bytes,
                timeout_seconds=auth.timeout_seconds,
            )
        except SqlTimeoutError as error:
            # 以工具错误返回（isError=True），由客户端映射成 timeout 分类
            raise RuntimeError(str(error)) from error
        except SqlPolicyError as error:
            raise RuntimeError(str(error)) from error

        return {
            "columns": outcome["columns"],
            "rows": outcome["rows"],
            "row_count": outcome["row_count"],
            "bytes": outcome["bytes"],
            "truncated": outcome["truncated"],
        }

    return server


def _sqlite_schema(auth: RunAuthorizations, dataset: DatasetRef) -> dict[str, Any]:
    """读取授权表的结构。

    不执行调用方提供的 SQL，也不读 `sqlite_master` 里未授权的表：
    只用 PRAGMA table_info 查询**本数据集自己的**授权表名。
    """
    import sqlite3

    table = dataset.database_table or dataset.name
    connection = sqlite3.connect(
        f"file:{Path(dataset.source_path).as_posix()}?mode=ro", uri=True
    )
    try:
        info = connection.execute(
            f'PRAGMA table_info("{table}")'
        ).fetchall()
    finally:
        connection.close()

    if not info:
        # 表不存在时不编造结构
        return {
            "dataset_id": dataset.dataset_id,
            "name": dataset.name,
            "source_type": dataset.source_type,
            "database_table": table,
            "columns": [],
            "note": "授权表不存在或没有字段信息",
        }

    return {
        "dataset_id": dataset.dataset_id,
        "name": dataset.name,
        "source_type": dataset.source_type,
        "database_table": table,
        "columns": [
            {
                "name": row[1],
                "declared_type": row[2] or "unspecified",
                "nullable": not bool(row[3]),
                "primary_key": bool(row[5]),
            }
            for row in info
        ],
    }


def load_authorizations(config_path: Path) -> RunAuthorizations:
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    return RunAuthorizations.from_config(payload)


def main(argv: list[str] | None = None) -> int:
    """入口：`python -m mcp_bridge.server --config <path>`。

    配置路径只能由启动方给，工具参数里没有它，模型无法改写授权。
    """
    import argparse

    parser = argparse.ArgumentParser(description="InsightFlow MCP Server")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    auth = load_authorizations(Path(args.config))
    build_server(auth).run("stdio")
    return 0


if __name__ == "__main__":
    # 兜底：任何未捕获异常也走 stderr，避免污染 stdout 的协议通道
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as error:  # noqa: BLE001
        print(f"mcp server fatal: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
