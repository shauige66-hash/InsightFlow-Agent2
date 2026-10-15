"""P2-B2：SQLite 查询的纵深防护。

本工具在**宿主进程内**执行 SQL，隔离强度低于 Python/Plot 的 Docker 沙箱，
因此这里逐层验证，并且不把任何一层说成容器级隔离：

    1. 词法只读校验        —— utils/sql_guard
    2. SQLite Authorizer   —— 按操作码拒绝写/DDL/ATTACH/PRAGMA 与越权表
    3. mode=ro 连接        —— 文件本身写不进去
    4. 进度回调超时        —— 单调时钟的查询预算
    5. 有界 fetchmany      —— 不把完整结果读进内存再截断

所有用例只用本文件自建的临时数据库，**不触碰任何业务数据库**。
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path

import pytest

from ingestion.models import TableResource
from tools.sql_tool import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_ROWS,
    SqlTimeoutError,
    run_query,
)
from tools.tool_registry import invoke_tool

ORDERS_ROWS = [
    (1, 101, 10.0),
    (2, 101, 20.0),
    (3, 102, 5.0),
    (4, 102, 7.5),
    (5, 103, 9.0),
]

CUSTOMERS_ROWS = [
    (101, "Alice", "East"),
    (102, "Bob", "West"),
    (103, "Carol", "East"),
]

PRODUCTS_ROWS = [
    (1, "widget", 2.5),
    (2, "gadget", 9.0),
]

SECRETS_ROWS = [
    (1, "top-secret-value"),
]


def build_database(path: Path, *, extra_rows: int = 0) -> None:
    """建一个测试库：orders / customers / products / secrets。

    products 与 secrets **不在**测试授权的表集合里，用来验证越权读取被拒绝。
    """
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER, amount REAL)"
        )
        connection.executemany("INSERT INTO orders VALUES (?, ?, ?)", ORDERS_ROWS)

        connection.execute(
            "CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT, region TEXT)"
        )
        connection.executemany("INSERT INTO customers VALUES (?, ?, ?)", CUSTOMERS_ROWS)

        connection.execute(
            "CREATE TABLE products (product_id INTEGER PRIMARY KEY, name TEXT, price REAL)"
        )
        connection.executemany("INSERT INTO products VALUES (?, ?, ?)", PRODUCTS_ROWS)

        connection.execute(
            "CREATE TABLE secrets (secret_id INTEGER PRIMARY KEY, value TEXT)"
        )
        connection.executemany("INSERT INTO secrets VALUES (?, ?)", SECRETS_ROWS)

        if extra_rows:
            connection.execute("CREATE TABLE big (n INTEGER)")
            connection.executemany(
                "INSERT INTO big VALUES (?)", [(index,) for index in range(extra_rows)]
            )

        connection.commit()
    finally:
        connection.close()


def database_fingerprint(path: Path) -> str:
    """库内容与结构的指纹：表清单 + 各表行数与内容哈希。"""
    connection = sqlite3.connect(path)
    try:
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        ]
        parts = []
        for table in tables:
            rows = connection.execute(f"SELECT * FROM {table}").fetchall()
            digest = hashlib.sha256(repr(rows).encode("utf-8")).hexdigest()
            parts.append(f"{table}:{len(rows)}:{digest}")
            schema = connection.execute(
                "SELECT sql FROM sqlite_master WHERE name = ?", (table,)
            ).fetchone()
            parts.append(f"schema:{schema[0] if schema else ''}")
        return "|".join(parts)
    finally:
        connection.close()


@pytest.fixture
def database(tmp_path: Path) -> Path:
    path = tmp_path / "test_sales.sqlite"
    build_database(path, extra_rows=400)
    return path


@pytest.fixture
def tables(database: Path) -> dict[str, TableResource]:
    """授权 orders 与 customers；products 与 secrets 存在但未授权。"""
    return {
        "orders": TableResource(
            name="orders",
            source_type="sqlite",
            source_path=str(database),
            database_table="orders",
        ),
        "customers": TableResource(
            name="customers",
            source_type="sqlite",
            source_path=str(database),
            database_table="customers",
        ),
    }


# run_query 的直接调用用它模拟"本次请求授权的表"。
# products / secrets / sqlite_master 刻意不在其中，用于验证越权读取被拒绝。
ALLOWED = frozenset({"orders", "customers", "big", "blobs"})


def query(database: Path, sql: str, **kwargs):
    return run_query(str(database), sql, allowed_tables=ALLOWED, **kwargs)


def run_sql(tables: dict[str, TableResource], sql: str, table_names=None):
    names = table_names or list(tables)
    return invoke_tool("sql", tables, {"table_names": names, "query": sql})


class TestReadOnlyQueriesStillWork:
    """加固不能把正常查询挡掉。"""

    @pytest.mark.parametrize(
        "sql, expected_rows",
        [
            ("SELECT * FROM orders", 5),
            ("select * from orders", 5),  # 小写
            ("SELECT customer_id, SUM(amount) AS total FROM orders GROUP BY customer_id", 3),
            (
                "WITH big AS (SELECT * FROM orders WHERE amount > 8) "
                "SELECT COUNT(*) AS n FROM big",
                1,
            ),
            (
                "SELECT c.name, SUM(o.amount) AS total FROM orders o "
                "JOIN customers c ON c.customer_id = o.customer_id GROUP BY c.name",
                3,
            ),
            ("SELECT region, COUNT(*) AS n FROM customers GROUP BY region", 2),
            # 注释与字符串字面量里的关键字不应被误判
            ("SELECT * FROM orders -- 这里的 delete 只是注释", 5),
            ("SELECT 'delete' AS label FROM orders LIMIT 1", 1),
            ("/* drop table orders */ SELECT COUNT(*) AS n FROM orders", 1),
        ],
    )
    def test_query_is_allowed_and_returns_rows(
        self, database: Path, sql: str, expected_rows: int
    ) -> None:
        outcome = query(database, sql)

        assert outcome["row_count"] == expected_rows, outcome
        assert outcome["truncated"] is False
        assert outcome["columns"]

    def test_multi_table_join_across_authorized_tables(self, database: Path) -> None:
        outcome = query(
            database,
            "SELECT o.order_id, c.name FROM orders o "
            "JOIN customers c ON c.customer_id = o.customer_id ORDER BY o.order_id",
        )

        assert outcome["row_count"] == 5
        assert outcome["rows"][0]["name"] == "Alice"


class TestAuthorizerBlocksDangerousOperations:
    """Authorizer 层：即使词法校验被绕过也要拒绝。

    注意分层事实：DELETE / PRAGMA / ATTACH 这些语句**在词法层就被拒了**
    （见 TestLexicalLayerAlsoRejects），所以经 run_query 走不到 Authorizer。
    要单独验证第二层，必须绕过第一层——下面用可写连接直接施加 Authorizer，
    证明它独立于词法校验生效。这正是"纵深"的含义：
    任何一层单独失效都不应导致写入或越权访问发生。
    """

    def test_authorizer_denies_write_on_a_writable_connection(
        self, tmp_path: Path
    ) -> None:
        """在**可写**连接上，Authorizer 也必须挡住 DELETE。

        连接刻意不用 mode=ro：这样"写入被拒绝"只可能来自 Authorizer，
        而不是只读文件系统兜底。
        """
        from utils.sql_policy import make_authorizer

        path = tmp_path / "writable.sqlite"
        build_database(path)
        before = database_fingerprint(path)

        connection = sqlite3.connect(path)  # 读写模式
        authorizer, denials = make_authorizer(ALLOWED)
        connection.set_authorizer(authorizer)
        try:
            with pytest.raises(sqlite3.DatabaseError):
                connection.execute("DELETE FROM orders")
            assert denials, "Authorizer 应记录拒绝原因"
        finally:
            connection.close()

        assert database_fingerprint(path) == before, "Authorizer 未挡住写入"

    @pytest.mark.parametrize(
        "action",
        [
            "SQLITE_INSERT",
            "SQLITE_UPDATE",
            "SQLITE_DELETE",
            "SQLITE_CREATE_TABLE",
            "SQLITE_DROP_TABLE",
            "SQLITE_ALTER_TABLE",
            "SQLITE_ATTACH",
            "SQLITE_DETACH",
            "SQLITE_PRAGMA",
            "SQLITE_TRANSACTION",
            "SQLITE_CREATE_INDEX",
            "SQLITE_DROP_INDEX",
            "SQLITE_CREATE_VIEW",
            "SQLITE_DROP_VIEW",
            "SQLITE_CREATE_TRIGGER",
            "SQLITE_DROP_TRIGGER",
            "SQLITE_ANALYZE",
            "SQLITE_REINDEX",
        ],
    )
    def test_authorizer_denies_every_write_related_action(self, action: str) -> None:
        """逐个操作码验证：写、DDL、ATTACH/DETACH、PRAGMA、事务一律 DENY。"""
        from utils.sql_policy import make_authorizer

        authorizer, denials = make_authorizer(ALLOWED)
        code = getattr(sqlite3, action)

        result = authorizer(code, "orders", None, "main", None)

        assert result == sqlite3.SQLITE_DENY, f"{action} 未被拒绝"
        assert denials, f"{action} 未记录原因"

    @pytest.mark.parametrize(
        "action",
        ["SQLITE_SELECT", "SQLITE_READ", "SQLITE_FUNCTION", "SQLITE_RECURSIVE"],
    )
    def test_authorizer_allows_the_read_actions(self, action: str) -> None:
        """只读查询必需的操作码必须放行，否则正常查询会被挡掉。"""
        from utils.sql_policy import make_authorizer

        authorizer, _denials = make_authorizer(ALLOWED)

        assert authorizer(getattr(sqlite3, action), "orders", None, "main", None) == (
            sqlite3.SQLITE_OK
        )

    def test_authorizer_denies_unauthorized_table_read(self) -> None:
        from utils.sql_policy import make_authorizer

        authorizer, denials = make_authorizer(ALLOWED)

        result = authorizer(sqlite3.SQLITE_READ, "secrets", "value", "main", None)

        assert result == sqlite3.SQLITE_DENY
        assert any("secrets" in reason for reason in denials)

    def test_authorizer_denies_metadata_table_read(self) -> None:
        from utils.sql_policy import make_authorizer

        authorizer, denials = make_authorizer(ALLOWED)

        result = authorizer(sqlite3.SQLITE_READ, "sqlite_master", "name", "main", None)

        assert result == sqlite3.SQLITE_DENY
        assert any("元数据" in reason for reason in denials)

    def test_authorizer_allows_authorized_table_read(self) -> None:
        from utils.sql_policy import make_authorizer

        authorizer, denials = make_authorizer(ALLOWED)

        result = authorizer(sqlite3.SQLITE_READ, "orders", "amount", "main", None)

        assert result == sqlite3.SQLITE_OK
        assert denials == []


class TestLexicalLayerAlsoRejects:
    """经真实工具入口时，危险语句必须在第一层就被拒。"""

    @pytest.mark.parametrize(
        "sql",
        [
            "DELETE FROM orders",
            "INSERT INTO orders VALUES (99, 101, 1.0)",
            "UPDATE orders SET amount = 0",
            "DROP TABLE orders",
            "CREATE TABLE evil (x INTEGER)",
            "ALTER TABLE orders ADD COLUMN evil TEXT",
            "TRUNCATE TABLE orders",
            "REPLACE INTO orders VALUES (1, 101, 0.0)",
            "VACUUM",
            "REINDEX",
            "PRAGMA table_info(orders)",
            "PRAGMA writable_schema = 1",
            "ATTACH DATABASE 'other.db' AS other",
            "DETACH DATABASE main",
            "SELECT 1; DROP TABLE orders",
        ],
    )
    def test_statement_is_refused_by_the_tool(self, tables, sql: str) -> None:
        outcome = run_sql(tables, sql)

        assert outcome["success"] is False, f"未被拒绝：{sql}"
        assert outcome["error"]
        assert outcome["error_kind"] == "execution_error"


class TestUnauthorizedTableAccess:
    """只能访问本次请求授权的表。"""

    def test_unlisted_table_is_refused(self, tables) -> None:
        outcome = run_sql(tables, "SELECT * FROM products")

        assert outcome["success"] is False
        assert "products" in outcome["error"]

    def test_secrets_table_is_refused(self, tables) -> None:
        outcome = run_sql(tables, "SELECT * FROM secrets")

        assert outcome["success"] is False
        assert "secrets" in outcome["error"]

    def test_metadata_table_is_refused(self, tables) -> None:
        outcome = run_sql(tables, "SELECT name FROM sqlite_master")

        assert outcome["success"] is False
        assert "sqlite_master" in outcome["error"] or "元数据" in outcome["error"]

    def test_authorized_table_can_be_queried(self, tables) -> None:
        outcome = run_sql(tables, "SELECT * FROM orders WHERE order_id = 1")

        assert outcome["success"] is True
        assert outcome["result"]["row_count"] == 1


class TestQueryTimeout:
    def test_slow_query_is_interrupted(self, database: Path) -> None:
        """真正耗时的查询必须被中断，而不是只检查回调存在。"""
        # 400 行的笛卡尔自连接 = 6.4e7 行中间结果，远超 0.5 秒预算
        started = time.monotonic()
        with pytest.raises(SqlTimeoutError):
            query(
                database,
                "SELECT COUNT(*) FROM big a, big b, big c",
                timeout_seconds=0.5,
            )
        elapsed = time.monotonic() - started

        # 中断确实发生，且不会拖到查询自然结束
        assert elapsed < 30, f"超时耗时过长: {elapsed:.2f}s"

    def test_timeout_surfaces_as_timeout_error_kind(self, tables) -> None:
        """经工具入口时，超时必须分类为 timeout，而不是普通执行错误。"""
        from tools.sql_tool import create_sql_tool

        # big 也必须出现在 table_names 里：授权集合由本次请求点名的表推导，
        # 没点名就会被 Authorizer 当成越权访问（那是另一条测试的事）。
        scoped = dict(tables)
        scoped["big"] = TableResource(
            name="big",
            source_type="sqlite",
            source_path=tables["orders"].source_path,
            database_table="big",
        )

        # 预算设为 0：进度回调第一次触发就判定超时，断言不依赖机器速度。
        # invoke_tool 走的是默认预算，这里直接构造工具以显式给定预算。
        tool = create_sql_tool(scoped, timeout_seconds=0.0)

        outcome = tool.invoke(
            {
                "table_names": ["orders", "customers", "big"],
                "query": "SELECT COUNT(*) FROM big a, big b, big c",
            }
        )

        assert outcome["success"] is False
        assert outcome["error_kind"] == "timeout"
        assert "时间预算" in outcome["error"]

    def test_fast_query_is_not_interrupted(self, database: Path) -> None:
        outcome = query(database, "SELECT COUNT(*) AS n FROM orders", timeout_seconds=5.0)

        assert outcome["row_count"] == 1
        assert outcome["rows"][0]["n"] == 5

    def test_connection_is_released_after_timeout(self, database: Path) -> None:
        """超时后连接必须被关闭，否则会残留文件句柄与锁。

        验证方式：超时之后立刻以独占方式打开同一个库做一次写事务。
        若连接还持有锁，这一步会失败。
        """
        with pytest.raises(SqlTimeoutError):
            query(
                database,
                "SELECT COUNT(*) FROM big a, big b, big c",
                timeout_seconds=0.3,
            )

        connection = sqlite3.connect(database, timeout=2.0)
        try:
            connection.execute("CREATE TABLE probe_after_timeout (x INTEGER)")
            connection.rollback()
        finally:
            connection.close()


class TestBoundedResults:
    def test_row_count_limit_truncates_and_flags(self, database: Path) -> None:
        outcome = query(database, "SELECT * FROM big", max_rows=10)

        assert outcome["row_count"] == 10
        assert outcome["truncated"] is True, "截断必须显式标记"
        assert outcome["columns"] == ["n"]

    def test_exact_row_count_is_not_flagged_as_truncated(self, database: Path) -> None:
        """结果正好等于上限时，只有确实还有更多行才标记截断。"""
        outcome = query(database, "SELECT * FROM big LIMIT 10", max_rows=10)

        assert outcome["row_count"] == 10
        assert outcome["truncated"] is False

    def test_large_single_row_is_bounded_by_bytes(self, database: Path) -> None:
        """单行超大时由字节上限兜住，不能只看行数。"""
        outcome = query(
            database,
            "SELECT group_concat(n, '') AS blob FROM big",
            max_rows=100,
            max_bytes=1024,
        )

        assert outcome["bytes"] <= 1024
        assert outcome["truncated"] is True
        # 超大单行被拒绝后应返回 0 行，而不是塞进一行超限数据
        assert outcome["row_count"] == 0

    def test_default_limits_are_sane(self) -> None:
        assert DEFAULT_MAX_ROWS > 0
        assert DEFAULT_MAX_BYTES > 0

    def test_result_is_json_serializable(self, database: Path) -> None:
        import json

        outcome = query(database, "SELECT * FROM orders")

        encoded = json.dumps(outcome, ensure_ascii=False)
        assert json.loads(encoded)["row_count"] == 5

    def test_blob_is_marked_not_decoded(self, database: Path) -> None:
        """BLOB 不能悄悄转成文本：显式标记，避免被误当成字符串使用。"""
        connection = sqlite3.connect(database)
        try:
            connection.execute("CREATE TABLE blobs (payload BLOB)")
            connection.execute("INSERT INTO blobs VALUES (?)", (b"\x00\x01\x02",))
            connection.commit()
        finally:
            connection.close()

        outcome = run_query(
            str(database),
            "SELECT payload FROM blobs",
            allowed_tables=frozenset({"blobs"}),
        )

        assert outcome["rows"][0]["payload"] == {"__kind__": "blob", "size": 3}


class TestDatabaseIsUnchanged:
    """查询失败或被拒绝后，库的内容与结构必须完全不变。"""

    def test_database_fingerprint_is_stable_across_all_rejections(
        self, database: Path, tables
    ) -> None:
        before = database_fingerprint(database)

        attacks = [
            "DELETE FROM orders",
            "UPDATE orders SET amount = 0",
            "INSERT INTO orders VALUES (99, 101, 1.0)",
            "DROP TABLE orders",
            "CREATE TABLE evil (x INTEGER)",
            "ALTER TABLE orders ADD COLUMN evil TEXT",
            "ATTACH DATABASE 'evil.db' AS evil",
            "PRAGMA writable_schema = 1",
            "VACUUM",
            "SELECT * FROM secrets",
            "SELECT * FROM products",
            "SELECT * FROM sqlite_master",
            "SELECT * FROM no_such_table",
            "SELECT bogus_column FROM orders",
        ]

        for sql in attacks:
            run_sql(tables, sql)

        assert database_fingerprint(database) == before, "库内容或结构被改动"

    def test_no_side_files_are_left_behind(self, database: Path, tables) -> None:
        """只读查询不应产生 -wal/-journal 之类的残留。"""
        before = sorted(path.name for path in database.parent.iterdir())

        run_sql(tables, "SELECT * FROM orders")
        run_sql(tables, "DELETE FROM orders")

        after = sorted(path.name for path in database.parent.iterdir())
        assert after == before, f"产生了额外文件: {set(after) - set(before)}"

    def test_failed_query_does_not_leak_a_transaction(self, database: Path) -> None:
        """构造一个执行中失败的查询，随后库仍可正常写。"""
        run_sql(
            {
                "orders": TableResource(
                    name="orders",
                    source_type="sqlite",
                    source_path=str(database),
                    database_table="orders",
                )
            },
            "SELECT * FROM orders WHERE no_such_column = 1",
        )

        connection = sqlite3.connect(database, timeout=2.0)
        try:
            connection.execute("INSERT INTO customers VALUES (999, 'Probe', 'North')")
            connection.rollback()
        finally:
            connection.close()


class TestResourceAuthorization:
    """数据源路径来自本次 Run 登记的资源，不能由模型指定。"""

    def test_non_sqlite_table_is_refused(self, tmp_path: Path) -> None:
        import pandas as pd

        tables = {
            "csv_table": TableResource(
                name="csv_table",
                source_type="csv",
                source_path=str(tmp_path / "x.csv"),
                dataframe=pd.DataFrame({"a": [1]}),
            )
        }

        outcome = run_sql(tables, "SELECT * FROM csv_table")

        assert outcome["success"] is False
        assert "SQLite" in outcome["error"]

    def test_unknown_table_name_is_refused(self, tables) -> None:
        outcome = run_sql(tables, "SELECT 1", table_names=["orders", "ghost"])

        assert outcome["success"] is False
        assert "ghost" in outcome["error"]

    def test_tables_from_different_databases_are_refused(self, tmp_path: Path) -> None:
        first = tmp_path / "a.sqlite"
        second = tmp_path / "b.sqlite"
        build_database(first)
        build_database(second)

        tables = {
            "orders": TableResource(
                name="orders", source_type="sqlite", source_path=str(first),
                database_table="orders",
            ),
            "customers": TableResource(
                name="customers", source_type="sqlite", source_path=str(second),
                database_table="customers",
            ),
        }

        outcome = run_sql(tables, "SELECT * FROM orders")

        assert outcome["success"] is False
        assert "same SQLite database" in outcome["error"]

    def test_multi_table_query_within_one_database_works(self, tables) -> None:
        outcome = run_sql(
            tables,
            "SELECT COUNT(*) AS n FROM orders o JOIN customers c "
            "ON c.customer_id = o.customer_id",
        )

        assert outcome["success"] is True
        assert outcome["result"]["rows"][0]["n"] == 5


class TestToolContract:
    def test_result_shape_is_stable(self, tables) -> None:
        success = run_sql(tables, "SELECT * FROM orders")
        failure = run_sql(tables, "DELETE FROM orders")

        for outcome in (success, failure):
            assert set(outcome) == {"success", "result", "error", "error_kind"}

    def test_success_payload_carries_truncation_metadata(self, tables) -> None:
        outcome = run_sql(tables, "SELECT * FROM orders")

        assert set(outcome["result"]) == {
            "columns",
            "rows",
            "row_count",
            "bytes",
            "truncated",
        }

    def test_empty_result_set_keeps_columns(self, tables) -> None:
        outcome = run_sql(tables, "SELECT * FROM orders WHERE amount > 1000")

        assert outcome["success"] is True
        assert outcome["result"]["row_count"] == 0
        assert outcome["result"]["columns"] == ["order_id", "customer_id", "amount"]
