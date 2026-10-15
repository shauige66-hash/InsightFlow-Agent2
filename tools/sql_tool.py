# 把 SQLite 查询能力包装成可供 AI 调用的 SQL Tool。
#
# 本工具在**宿主进程内**执行 SQL，隔离强度低于 Python/Plot 的 Docker 沙箱。
# 因此采用多层纵深，且不宣称具备容器级隔离：
#
#   1. 词法只读校验（utils/sql_guard）—— 拦掉明显非只读与多语句，fail-closed；
#   2. SQLite Authorizer        —— 按操作码拒绝写/DDL/ATTACH 与越权表；
#   3. mode=ro 连接             —— 即使前两层被绕过，文件本身也写不进去；
#   4. 进度回调超时             —— 用单调时钟的查询预算中断长查询；
#   5. 有界 fetchmany           —— 不把整个结果集读进内存再截断。
#
# 第 2～5 层的实现在 `utils/sql_policy.py`：MCP Server 必须受同一套约束，
# 因此策略只有一份，这里只负责"授权从哪来"和"结果怎么回给调用方"。

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from ingestion.models import TableResource
from mcp_bridge.contract import DatasetRef
from utils.sql_guard import SqlNotReadOnly
from utils.sql_policy import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_ROWS,
    DEFAULT_TIMEOUT_SECONDS,
    SqlPolicyError,
    SqlTimeoutError,
    execute_readonly_sql,
)

# 重新导出，保持既有调用方与测试的导入路径可用
__all__ = [
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_ROWS",
    "DEFAULT_TIMEOUT_SECONDS",
    "SqlPolicyError",
    "SqlTimeoutError",
    "create_sql_tool",
    "execute_readonly_sql",
    "run_query",
]

from utils.sql_policy import run_query  # noqa: E402  (再导出，保持旧导入路径)


def _dataset_of(resource: TableResource) -> DatasetRef:
    """把单个资源看成一份授权数据集。

    授权表名取**该资源自己的物理表名**，而不是"资源名与本次表名按位置对应"。
    后者是错的：`table_names` 的顺序由模型决定，一旦它与 `tables` 的迭代顺序
    不一致，就会把 A 表的授权安到 B 表上。
    """
    return DatasetRef(
        dataset_id=str(resource.name),
        name=str(resource.name),
        source_type=str(resource.source_type),
        source_path=str(resource.source_path),
        locator=resource.database_table or None,
        content_hash=resource.content_hash,
        database_table=resource.database_table or None,
    )


def _authorized_targets(
    sqlite_resources: list[TableResource],
) -> tuple[str, frozenset[str]]:
    """从本次点名的资源推出 (库文件路径, 授权物理表集合)。"""
    source_paths = {
        str(Path(resource.source_path).resolve()) for resource in sqlite_resources
    }

    # 只允许访问当前 Run 登记的资源所属的那个库文件。
    # 路径来自 resource.source_path，由 Loader 在本次 Run 解析并校验过，
    # 模型无法指定任意宿主路径。
    if len(source_paths) != 1:
        raise ValueError(
            "SQL multi-table queries require all target "
            "tables to come from the same SQLite database."
        )

    allowed = frozenset(
        dataset.database_table or dataset.name
        for dataset in (_dataset_of(resource) for resource in sqlite_resources)
    )
    return next(iter(source_paths)), allowed


def create_sql_tool(
    tables: dict[str, TableResource],
    *,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_bytes: int = DEFAULT_MAX_BYTES,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
):

    @tool
    def run_sql_query(
        table_names: list[str],
        query: str
    ) -> dict[str, Any]:
        """对同一个 SQLite 数据库中的一个或多个数据表执行只读 SQL 查询。"""

        if not table_names:
            raise ValueError(
                "At least one table is required."
            )

        sqlite_resources = []

        for table_name in table_names:

            if table_name not in tables:
                raise ValueError(
                    f"Table not found: {table_name}"
                )

            resource = tables[table_name]

            if resource.source_type != "sqlite":
                raise ValueError(
                    f"Table is not a SQLite resource: {table_name}"
                )

            sqlite_resources.append(resource)

        try:
            database_path, allowed_tables = _authorized_targets(sqlite_resources)
        except ValueError as error:
            return {
                "success": False,
                "result": None,
                "error": str(error),
                "error_kind": "execution_error",
            }

        cleaned_query = query.strip()

        try:
            outcome = execute_readonly_sql(
                database_path,
                cleaned_query,
                allowed_tables=allowed_tables,
                max_rows=max_rows,
                max_bytes=max_bytes,
                timeout_seconds=timeout_seconds,
            )
        except SqlNotReadOnly as error:
            # 非只读 SQL 属于调用方写错了查询：可修复，不是环境问题
            return {
                "success": False,
                "result": None,
                "error": str(error),
                "error_kind": "execution_error",
            }
        except SqlTimeoutError as error:
            return {
                "success": False,
                "result": None,
                "error": str(error),
                # 超时属于资源类失败：重发同样的查询不会好转
                "error_kind": "timeout",
            }
        except SqlPolicyError as error:
            return {
                "success": False,
                "result": None,
                "error": str(error),
                "error_kind": "execution_error",
            }
        except Exception as error:  # noqa: BLE001 - 统一转成结果字典
            return {
                "success": False,
                "result": None,
                "error": str(error),
                "error_kind": "execution_error",
            }

        return {
            "success": True,
            "result": {
                "columns": outcome["columns"],
                "rows": outcome["rows"],
                "row_count": outcome["row_count"],
                "bytes": outcome["bytes"],
                # 截断必须显式标记：否则部分结果会被当成完整结果使用
                "truncated": outcome["truncated"],
            },
            "error": None,
            "error_kind": None,
        }

    return run_sql_query
