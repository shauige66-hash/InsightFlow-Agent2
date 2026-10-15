"""P1 补验收：SQL 只读是否真正接入工具链。

从**真实调用路径**验证 `tool_registry → sql_tool → sqlite3`，
而不是只测 `utils/sql_guard` 这个独立函数。

两层防护都要生效，缺一不可：
1. 词法校验 `ensure_read_only` —— 拒绝非只读语句、多语句、注释绕过；
2. 连接以 `mode=ro` 打开 —— 即使词法层被绕过，SQLite 也会拒绝写入。

用专用测试数据库，不触碰任何业务数据。测试结束校验数据库内容未变。
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from ingestion.models import TableResource
from tools.tool_registry import clear_tool_cache, invoke_tool


@pytest.fixture
def sqlite_db(tmp_path: Path) -> Path:
    """专用测试数据库：orders / customers 两张表。"""
    path = tmp_path / "test_sales.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER, amount REAL)"
        )
        connection.execute(
            "INSERT INTO orders VALUES (1, 101, 10.0), (2, 102, 20.0), (3, 101, 30.0)"
        )
        connection.execute(
            "CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT)"
        )
        connection.execute("INSERT INTO customers VALUES (101, 'Alice'), (102, 'Bob')")
    return path


@pytest.fixture
def tables(sqlite_db: Path) -> dict[str, TableResource]:
    return {
        "test_sales.orders": TableResource(
            name="test_sales.orders",
            source_type="sqlite",
            source_path=str(sqlite_db),
            database_table="orders",
        ),
        "test_sales.customers": TableResource(
            name="test_sales.customers",
            source_type="sqlite",
            source_path=str(sqlite_db),
            database_table="customers",
        ),
    }


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_tool_cache()
    yield
    clear_tool_cache()


def database_fingerprint(path: Path) -> str:
    """数据库整体内容的指纹（含所有表的数据）。"""
    digest = hashlib.sha256()
    with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True) as connection:
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        ]
        for table in tables:
            digest.update(table.encode())
            for row in connection.execute(f"SELECT * FROM {table} ORDER BY rowid"):
                digest.update(repr(row).encode())
    return digest.hexdigest()


def run_sql(tables: dict, query: str):
    """经真实工具注册表执行，与工作流内的调用路径一致。"""
    return invoke_tool("sql", tables, {"table_names": list(tables), "query": query})


class TestReadOnlyStatementsWork:
    def test_select_returns_rows(self, tables) -> None:
        outcome = run_sql(tables, "SELECT * FROM orders ORDER BY order_id")

        assert outcome["success"] is True
        assert outcome["result"]["row_count"] == 3

    def test_aggregate_query_works(self, tables) -> None:
        outcome = run_sql(
            tables, "SELECT customer_id, SUM(amount) AS total FROM orders GROUP BY customer_id"
        )

        assert outcome["success"] is True
        assert outcome["result"]["row_count"] == 2

    def test_cte_select_works(self, tables) -> None:
        outcome = run_sql(
            tables,
            "WITH big AS (SELECT * FROM orders WHERE amount > 15) SELECT COUNT(*) AS n FROM big",
        )

        assert outcome["success"] is True
        assert outcome["result"]["rows"][0]["n"] == 2

    def test_join_across_tables_works(self, tables) -> None:
        outcome = run_sql(
            tables,
            "SELECT c.name, SUM(o.amount) AS total FROM orders o "
            "JOIN customers c ON c.customer_id = o.customer_id GROUP BY c.name",
        )

        assert outcome["success"] is True
        assert outcome["result"]["row_count"] == 2


class TestWriteStatementsAreRefused:
    """经真实工具入口拒绝写操作。"""

    @pytest.mark.parametrize(
        "query",
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
            "ATTACH DATABASE 'x.db' AS other",
            "DETACH DATABASE main",
        ],
    )
    def test_statement_is_rejected(self, tables, query: str) -> None:
        outcome = run_sql(tables, query)

        assert outcome["success"] is False, f"未被拒绝：{query}"
        assert outcome["error"], "拒绝时必须给出原因"

    @pytest.mark.parametrize(
        "query",
        [
            # 注释前缀绕过：startswith("with") 会放行，实际是 DELETE
            "/* x */ WITH t AS (SELECT 1) DELETE FROM orders",
            "WITH t AS (SELECT 1) DELETE FROM orders",
            # 多语句
            "SELECT 1; DROP TABLE orders",
            "SELECT * FROM orders; DELETE FROM orders",
            # 空白与换行绕过
            "\n\n  DELETE FROM orders",
            "-- comment\nDELETE FROM orders",
            # 大小写
            "delete from orders",
            "DeLeTe FrOm orders",
        ],
    )
    def test_bypass_attempt_is_rejected(self, tables, query: str) -> None:
        outcome = run_sql(tables, query)

        assert outcome["success"] is False, f"绕过成功：{query!r}"


class TestDatabaseIsUnchanged:
    def test_content_hash_is_stable_across_dangerous_attempts(
        self, tables, sqlite_db: Path
    ) -> None:
        """执行一批危险语句后，数据库内容必须完全不变。"""
        before = database_fingerprint(sqlite_db)

        dangerous = [
            "DELETE FROM orders",
            "INSERT INTO orders VALUES (99, 101, 1.0)",
            "UPDATE orders SET amount = 0",
            "DROP TABLE orders",
            "/* x */ WITH t AS (SELECT 1) DELETE FROM orders",
            "SELECT 1; DROP TABLE orders",
            "PRAGMA journal_mode = WAL",
            "ATTACH DATABASE 'evil.db' AS evil",
        ]
        for query in dangerous:
            outcome = run_sql(tables, query)
            assert outcome["success"] is False, f"危险语句被执行：{query}"

        after = database_fingerprint(sqlite_db)

        assert before == after, "数据库内容发生了变化"

    def test_row_counts_unchanged(self, tables, sqlite_db: Path) -> None:
        run_sql(tables, "DELETE FROM orders")
        run_sql(tables, "DROP TABLE orders")

        with sqlite3.connect(sqlite_db) as connection:
            orders = connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
            customers = connection.execute(
                "SELECT COUNT(*) FROM customers"
            ).fetchone()[0]

        assert orders == 3
        assert customers == 2

    def test_evil_table_was_not_created(self, tables, sqlite_db: Path) -> None:
        run_sql(tables, "CREATE TABLE evil (x INTEGER)")

        with sqlite3.connect(sqlite_db) as connection:
            names = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }

        assert "evil" not in names


class TestConnectionIsActuallyReadOnly:
    """第二层防护：即使绕过词法校验，连接本身也必须拒绝写入。"""

    def test_read_only_uri_rejects_writes(self, sqlite_db: Path) -> None:
        from utils.sql_policy import read_only_uri

        uri = read_only_uri(str(sqlite_db))
        assert "mode=ro" in uri

        with sqlite3.connect(uri, uri=True) as connection:
            with pytest.raises(sqlite3.OperationalError, match="readonly"):
                connection.execute("DELETE FROM orders")

    def test_read_write_connection_would_allow_the_bypass(
        self, sqlite_db: Path
    ) -> None:
        """反证：以读写模式打开时，同样的语句会被执行。

        这说明仅靠词法校验不够，`mode=ro` 是必要的第二层。
        """
        with sqlite3.connect(sqlite_db) as connection:
            connection.execute("CREATE TABLE proof (x INTEGER)")

        with sqlite3.connect(sqlite_db) as connection:
            names = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        assert "proof" in names

        # 收尾：清理反证用的表
        with sqlite3.connect(sqlite_db) as connection:
            connection.execute("DROP TABLE proof")


class TestGuardIsWiredIntoTheChain:
    def test_sql_tool_module_uses_the_guard(self) -> None:
        """确认只读校验确实接在执行链路上，而不是只有独立测试通过。

        P2-C 把策略本体抽到 `utils/sql_policy.py`（MCP Server 必须复用同一份），
        因此这里分两处核对：工具侧必须调用共享策略入口，
        策略侧必须真的装上 Authorizer、超时与有界读取。
        """
        from pathlib import Path as _Path

        root = _Path(__file__).resolve().parent.parent
        tool_source = (root / "tools" / "sql_tool.py").read_text(encoding="utf-8")
        policy_source = (root / "utils" / "sql_policy.py").read_text(encoding="utf-8")

        # 工具侧：走共享策略，且不自行实现只读判定
        assert "execute_readonly_sql" in tool_source
        assert "SqlNotReadOnly" in tool_source

        # 策略侧：三层硬防护 + 有界读取必须真的实现
        assert "ensure_read_only" in policy_source
        assert "mode=ro" in policy_source
        assert "set_authorizer" in policy_source
        assert "set_progress_handler" in policy_source
        assert "fetchmany" in policy_source

        # 旧的弱校验（仅前缀判断）不应再作为唯一防线
        assert 'startswith(\n            ("select", "with")' not in tool_source
        assert 'startswith(\n            ("select", "with")' not in policy_source

    def test_registry_reports_rejection_not_exception(self, tables) -> None:
        """参数与安全类错误都应以结果字典返回，便于修复循环处理。"""
        outcome = run_sql(tables, "DELETE FROM orders")

        # sql_tool 现在与其他工具一致，显式给出 error_kind
        assert set(outcome) == {"success", "result", "error", "error_kind"}
        assert outcome["success"] is False
