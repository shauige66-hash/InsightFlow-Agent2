# 把 SQLite 查询能力包装成可供 AI 调用的 SQL Tool。
#
# 本工具在**宿主进程内**执行 SQL，隔离强度低于 Python/Plot 的 Docker 沙箱。
# 因此这里采用多层纵深，且不宣称具备容器级隔离：
#
#   1. 词法只读校验（utils/sql_guard）—— 拦掉明显非只读与多语句，fail-closed；
#   2. SQLite Authorizer        —— 在 SQLite 内部按操作码拒绝写/DDL/ATTACH 与越权表；
#   3. mode=ro 连接             —— 即使前两层被绕过，文件本身也写不进去；
#   4. 进度回调超时             —— 用单调时钟的查询预算中断长查询；
#   5. 有界 fetchmany           —— 不把整个结果集读进内存再截断。
#
# 前两层是"能否执行"，后两层是"执行多久、返回多少"。

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from ingestion.models import TableResource
from utils.sql_guard import SqlNotReadOnly, ensure_read_only

# 默认预算。行数与字节数的关系：行数控制"多少条"，字节数控制"多大"，
# 两者都要，因为一行也可能非常大（例如拼接后的长字符串）。
DEFAULT_MAX_ROWS = 10_000
DEFAULT_MAX_BYTES = 4 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 10.0

# 进度回调每多少个 SQLite 虚拟机指令被调用一次。
# 太小会让查询本身变慢，太大则中断不及时。
_PROGRESS_INTERVAL = 2_000

# 允许的 Authorizer 操作码。
#
# 刻意采用"显式白名单"：只有真正为完成一次只读 SELECT 所必需的操作码才放行，
# 其余（含所有写、DDL、ATTACH/DETACH、PRAGMA、事务控制）一律拒绝。
_ALLOWED_ACTIONS = frozenset(
    {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        sqlite3.SQLITE_RECURSIVE,
    }
)

# 操作码 → 人读名字。仅用于把拒绝原因写清楚，不做权限判断。
_ACTION_NAMES = {
    value: name
    for name, value in vars(sqlite3).items()
    if name.startswith("SQLITE_") and isinstance(value, int)
}

# SQLite 的元数据表。允许读取会让模型看到库里还有哪些表/视图，
# 也可能间接暴露建表语句，因此显式拒绝，只允许访问本次授权的表。
_METADATA_TABLES = frozenset({"sqlite_master", "sqlite_schema", "sqlite_temp_master"})


class SqlPolicyError(ValueError):
    """查询违反本工具的访问策略（越权表、非授权操作）。"""


class SqlTimeoutError(RuntimeError):
    """查询超出时间预算，已被中断。"""


def _read_only_uri(database_path: str) -> str:
    """构造 SQLite 只读 URI。

    用 URI 形式而不是直接传路径：`sqlite3.connect(path)` 以读写模式打开，
    这是"只读"承诺与其实现不一致的常见来源。
    """
    path = Path(database_path).resolve()
    # as_posix 保证 Windows 反斜杠不会破坏 URI
    return f"file:{path.as_posix()}?mode=ro"


def _action_name(action: int) -> str:
    return _ACTION_NAMES.get(action, f"action_{action}")


def _make_authorizer(allowed_tables: frozenset[str]):
    """构造回调式 Authorizer。

    返回 `SQLITE_OK` 放行、`SQLITE_DENY` 拒绝。拒绝时把原因写进闭包里的
    列表：Authorizer 自身无法抛异常（回调里的异常会被 SQLite 吞成通用错误），
    因此用这个列表把具体原因带回上层。
    """

    denials: list[str] = []

    def authorizer(
        action: int,
        arg1: str | None,
        arg2: str | None,
        database_name: str | None,
        trigger_name: str | None,
    ) -> int:
        if action not in _ALLOWED_ACTIONS:
            denials.append(
                f"不允许的操作 {_action_name(action)}"
                + (f"（对象 {arg1}）" if arg1 else "")
            )
            return sqlite3.SQLITE_DENY

        # 只读访问的表必须在本次请求的授权集合内
        if action in (sqlite3.SQLITE_READ, sqlite3.SQLITE_SELECT) and arg1:
            table = str(arg1)
            if table in _METADATA_TABLES:
                denials.append(f"不允许访问元数据表 {table}")
                return sqlite3.SQLITE_DENY
            if table not in allowed_tables:
                denials.append(f"未授权的数据表 {table}")
                return sqlite3.SQLITE_DENY

        return sqlite3.SQLITE_OK

    return authorizer, denials


def _make_reader(
    connection: sqlite3.Connection,
    *,
    query: str,
    max_rows: int,
    max_bytes: int,
    timeout_seconds: float,
):
    """有界读取：逐批取行，达到行数或字节上限即停。

    为什么不能先 `read_sql_query` 再截断：那等于把完整结果（可能几十万行）
    先读进内存，内存峰值由数据决定，"限制"形同虚设。这里用游标分批取，
    内存上限由 batch 大小决定。
    """
    cursor = connection.cursor()
    deadline = time.monotonic() + timeout_seconds

    timed_out = {"value": False}

    def progress_handler() -> int:
        if time.monotonic() >= deadline:
            timed_out["value"] = True
            return 1  # 非 0 → SQLite 中断当前语句
        return 0

    connection.set_progress_handler(progress_handler, _PROGRESS_INTERVAL)

    def interrupted() -> SqlTimeoutError:
        return SqlTimeoutError(
            f"查询超过 {timeout_seconds:.1f} 秒的时间预算，已被中断。"
        )

    # execute() 本身就可能跑很久（大连接/排序都发生在这一步），
    # 因此超时处理必须同时覆盖 execute 与取行两个阶段。
    try:
        cursor.execute(query)
    except sqlite3.OperationalError as error:
        if timed_out["value"]:
            raise interrupted() from error
        raise

    columns = (
        [description[0] for description in cursor.description]
        if cursor.description
        else []
    )

    rows: list[dict[str, Any]] = []
    truncated = False
    total_bytes = 0

    try:
        while len(rows) < max_rows:
            batch = cursor.fetchmany(min(500, max_rows - len(rows)))
            if not batch:
                break

            for record in batch:
                item = {
                    column: _jsonable(value)
                    for column, value in zip(columns, record)
                }
                size = len(json.dumps(item, ensure_ascii=False, default=str))
                if total_bytes + size > max_bytes:
                    truncated = True
                    break
                rows.append(item)
                total_bytes += size

            if truncated:
                break
        else:
            # 行数正好用满：再探一行判断是否真的还有更多
            if len(rows) >= max_rows and cursor.fetchmany(1):
                truncated = True
    except sqlite3.OperationalError as error:
        if timed_out["value"]:
            raise interrupted() from error
        raise
    finally:
        connection.set_progress_handler(None, 0)
        cursor.close()

    return {
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "bytes": total_bytes,
        "truncated": truncated,
    }


def _jsonable(value: Any) -> Any:
    """把 SQLite 返回的标量转成可 JSON 表达的形式。

    BLOB 不能直接 JSON 化，转成 base64 会掩盖它是二进制这一事实，
    这里改成显式标记，避免调用方误当成文本。
    """
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"__kind__": "blob", "size": len(bytes(value))}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def run_query(
    database_path: str,
    query: str,
    *,
    allowed_tables: frozenset[str],
    max_rows: int = DEFAULT_MAX_ROWS,
    max_bytes: int = DEFAULT_MAX_BYTES,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """在只读、授权受限、有界时间与体积的前提下执行一次查询。

    抽成独立函数是为了让策略层可以在不经过 LangChain 工具包装的情况下被测试，
    也便于把"策略"与"工具装配"分开。
    """
    ensure_read_only(query)

    connection = sqlite3.connect(_read_only_uri(database_path), uri=True)
    try:
        authorizer, denials = _make_authorizer(allowed_tables)
        connection.set_authorizer(authorizer)

        try:
            result = _make_reader(
                connection,
                query=query,
                max_rows=max_rows,
                max_bytes=max_bytes,
                timeout_seconds=timeout_seconds,
            )
        except sqlite3.DatabaseError as error:
            # Authorizer 拒绝会以 DatabaseError("not authorized") 形式抛出；
            # 具体原因在闭包列表里，用它给出可读的错误。
            if denials:
                raise SqlPolicyError(denials[0]) from error
            raise
    finally:
        # 无论成功、超时还是策略拒绝，都关闭连接释放文件句柄与锁
        connection.close()

    return result


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

        source_paths = {
            str(
                Path(resource.source_path).resolve()
            )
            for resource in sqlite_resources
        }

        # 只允许访问当前 Run 登记的资源所属的那个库文件。
        # 模型不能指定任意宿主路径：路径来自 resource.source_path，
        # 由 Loader 在本次 Run 解析并校验过。
        if len(source_paths) != 1:
            raise ValueError(
                "SQL multi-table queries require all target "
                "tables to come from the same SQLite database."
            )

        database_path = next(
            iter(source_paths)
        )

        # 授权的表 = 本次请求点名的表对应的**物理表名**。
        # 用 database_table 而不是资源别名：Authorizer 看到的是真实表名。
        allowed_tables = frozenset(
            resource.database_table or resource.name
            for resource in sqlite_resources
        )

        cleaned_query = query.strip()

        # 第一层：词法只读校验。拒绝非 SELECT/WITH、多语句与注释绕过。
        try:
            ensure_read_only(cleaned_query)
        except SqlNotReadOnly as error:
            # 非只读 SQL 属于调用方写错了查询：可修复，不是环境问题
            return {
                "success": False,
                "result": None,
                "error": str(error),
                "error_kind": "execution_error"
            }

        try:
            outcome = run_query(
                database_path,
                cleaned_query,
                allowed_tables=allowed_tables,
                max_rows=max_rows,
                max_bytes=max_bytes,
                timeout_seconds=timeout_seconds,
            )
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
        except Exception as error:
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
