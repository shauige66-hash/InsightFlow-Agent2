# 把 SQLite 查询能力包装成可供 AI 调用的 SQL Tool。

from pathlib import Path
from typing import Any
import sqlite3

import pandas as pd
from langchain_core.tools import tool

from ingestion.models import TableResource
from utils.sql_guard import SqlNotReadOnly, ensure_read_only


def _read_only_uri(database_path: str) -> str:
    """构造 SQLite 只读 URI。

    用 URI 形式而不是直接传路径：`sqlite3.connect(path)` 以读写模式打开，
    这是"只读"承诺与其实现不一致的常见来源。
    """
    path = Path(database_path).resolve()
    # as_posix 保证 Windows 反斜杠不会破坏 URI
    return f"file:{path.as_posix()}?mode=ro"


def create_sql_tool(
    tables: dict[str, TableResource]
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

        source_paths = {
            str(
                Path(resource.source_path).resolve()
            )
            for resource in sqlite_resources
        }

        if len(source_paths) != 1:
            raise ValueError(
                "SQL multi-table queries require all target "
                "tables to come from the same SQLite database."
            )

        database_path = next(
            iter(source_paths)
        )

        cleaned_query = query.strip()

        # 词法只读校验：拒绝非 SELECT/WITH、多语句、注释绕过与写关键字。
        # 这是第一道闸门，但词法判定本身不是权限隔离——真正的边界是下面
        # 以 mode=ro 打开连接（第二道闸门）。两者必须同时生效。
        try:
            ensure_read_only(cleaned_query)
        except SqlNotReadOnly as error:
            return {
                "success": False,
                "result": None,
                "error": str(error)
            }

        try:
            # 只读 URI 打开：即使词法校验被绕过，SQLite 也会拒绝任何写入。
            # 注意两点：不能只用 `sqlite3.connect(path)`（读写模式），
            # 且 URI 形式必须显式传 uri=True，否则字符串会被当成文件名。
            with sqlite3.connect(
                _read_only_uri(database_path), uri=True
            ) as connection:

                result_df = pd.read_sql_query(
                    cleaned_query,
                    connection
                )

            return {
                "success": True,
                "result": result_df.to_dict(
                    orient="records"
                ),
                "error": None
            }

        except Exception as error:
            return {
                "success": False,
                "result": None,
                "error": str(error)
            }

    return run_sql_query