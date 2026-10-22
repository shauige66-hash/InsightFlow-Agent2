"""SQLite 只读执行策略。

P2-B 把这套约束实现在 `tools/sql_tool.py` 里。P2-C 引入 MCP Server 后，
Server 也必须受同一套约束——因此这里把**策略本体**抽出来，由 direct 工具与
MCP Server 共同调用。

抽取的动机不是"分层好看"，而是：如果 Server 侧另写一份 SQL 执行，
两边的 authorizer、超时与结果上限就会各自漂移，
"引入 MCP 不得放宽任何既有安全条件"这条要求会变成靠自觉维持。

本模块只负责"给定库文件与授权表集合，安全地执行一条只读查询"。
词法校验与"哪些表被授权"由调用方决定，因为 direct 与 MCP 的授权来源不同。
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from utils.sql_guard import ensure_read_only

DEFAULT_MAX_ROWS = 10_000
DEFAULT_MAX_BYTES = 4 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 10.0

# 进度回调每多少个 SQLite 虚拟机指令被调用一次。
# 太小会让查询本身变慢，太大则中断不及时。
_PROGRESS_INTERVAL = 2_000

# 允许的 Authorizer 操作码。
#
# 显式白名单：只有完成一次只读 SELECT 所必需的操作码才放行，
# 其余（含所有写、DDL、ATTACH/DETACH、PRAGMA、事务控制）一律拒绝。
_ALLOWED_ACTIONS = frozenset(
    {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        sqlite3.SQLITE_RECURSIVE,
    }
)

_ACTION_NAMES = {
    value: name
    for name, value in vars(sqlite3).items()
    if name.startswith("SQLITE_") and isinstance(value, int)
}

# SQLite 元数据表：允许读取会让模型看到库里还有哪些表与建表语句，
# 因此显式拒绝，只允许访问本次授权的表。
_METADATA_TABLES = frozenset({"sqlite_master", "sqlite_schema", "sqlite_temp_master"})


class SqlPolicyError(ValueError):
    """查询违反执行策略（越权表、非授权操作）。"""


class SqlTimeoutError(RuntimeError):
    """查询超出时间预算，已被中断。"""


def read_only_uri(database_path: str) -> str:
    """构造 SQLite 只读 URI。

    用 URI 形式而不是直接传路径：`sqlite3.connect(path)` 以读写模式打开，
    这是"只读"承诺与其实现不一致的常见来源。`uri=True` 是必须的，
    否则整串会被当成文件名。
    """
    path = Path(database_path).resolve()
    # as_posix 保证 Windows 反斜杠不会破坏 URI
    return f"file:{path.as_posix()}?mode=ro"


def _action_name(action: int) -> str:
    return _ACTION_NAMES.get(action, f"action_{action}")


def make_authorizer(allowed_tables: frozenset[str]):
    """构造回调式 Authorizer。

    返回 `(authorizer, denials)`。拒绝原因写进闭包里的列表：
    Authorizer 自身无法抛异常（回调里的异常会被 SQLite 吞成通用错误），
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


def _jsonable(value: Any) -> Any:
    """把 SQLite 标量转成可 JSON 表达的形式。

    BLOB 不直接 JSON 化，也不悄悄转成文本（那会让调用方误当成字符串），
    改成显式标记。
    """
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"__kind__": "blob", "size": len(bytes(value))}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _make_reader(
    connection: sqlite3.Connection,
    *,
    query: str,
    max_rows: int,
    max_bytes: int,
    timeout_seconds: float,
):
    """有界读取：逐批取行，达到行数或字节上限即停。

    不能先 `read_sql_query` 再截断——那等于把完整结果读进内存，
    内存峰值由数据决定，"限制"形同虚设。这里用游标分批取，
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
            # 行数正好用满：再探一行，只有确实还有更多才标记截断
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

    不做词法校验（`ensure_read_only`）——那是调用方的第一道闸门，
    这里负责的是它之后的 Authorizer、超时与结果上限。
    """
    connection = sqlite3.connect(read_only_uri(database_path), uri=True)
    try:
        # Bound individual values before SQLite allocates/materializes them.
        # Row-count and JSON limits below only apply after allocation.
        connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, max(64 * 1024, max_bytes))
        connection.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 256 * 1024)
        connection.setlimit(sqlite3.SQLITE_LIMIT_COLUMN, 512)
        connection.setlimit(sqlite3.SQLITE_LIMIT_EXPR_DEPTH, 100)
        authorizer, denials = make_authorizer(allowed_tables)
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
            # Authorizer 拒绝会以 DatabaseError("not authorized") 抛出；
            # 具体原因在闭包列表里，用它给出可读的错误。
            if denials:
                raise SqlPolicyError(denials[0]) from error
            raise
    finally:
        # 无论成功、超时还是策略拒绝，都关闭连接释放文件句柄与锁
        connection.close()

    return result


def execute_readonly_sql(
    database_path: str,
    query: str,
    *,
    allowed_tables: frozenset[str],
    max_rows: int = DEFAULT_MAX_ROWS,
    max_bytes: int = DEFAULT_MAX_BYTES,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """完整的一次只读查询：词法校验 → 策略执行。

    direct 工具与 MCP Server 都走这一条路径，因此两边的安全条件不可能不一致。
    """
    ensure_read_only(query)
    return run_query(
        database_path,
        query,
        allowed_tables=allowed_tables,
        max_rows=max_rows,
        max_bytes=max_bytes,
        timeout_seconds=timeout_seconds,
    )
