"""工具层行为。

重点验证两件事：
1. Python / SQL 工具对"参数非法"与"执行失败"给出统一的结果形状，
   调用方不需要分别处理异常与错误字典。
2. 安全约束真的生效：SQL 只接受只读语句，Python 工具的产物约定不会被静默跳过。

这里断言的是当前实现的实际边界，不宣称它们构成安全沙箱——
真正的资源隔离属于 P2，见 docs 的已知限制。
"""

from __future__ import annotations

import pandas as pd
import pytest

from ingestion.models import TableResource
from tools.python_tool import create_python_tool
from tools.sql_tool import create_sql_tool
from tools.tool_registry import available_tools, clear_tool_cache, invoke_tool

from tests.conftest import SALES_SQLITE


@pytest.fixture(autouse=True)
def _fresh_tool_cache():
    clear_tool_cache()
    yield
    clear_tool_cache()


@pytest.fixture
def frame_tables() -> dict[str, TableResource]:
    return {
        "customers": TableResource(
            name="customers",
            source_type="csv",
            source_path="data/customers.csv",
            dataframe=pd.DataFrame(
                {
                    "customer_id": ["C001", "C002", "C003"],
                    "region": ["East", "West", "East"],
                }
            ),
        )
    }


@pytest.fixture
def sqlite_tables() -> dict[str, TableResource]:
    return {
        "sales.orders": TableResource(
            name="sales.orders",
            source_type="sqlite",
            source_path=str(SALES_SQLITE),
            database_table="orders",
        )
    }


class TestPythonTool:
    def test_successful_analysis_returns_result(self, frame_tables) -> None:
        tool = create_python_tool(frame_tables)

        outcome = tool.invoke(
            {
                "table_names": ["customers"],
                "code": "result = len(df)",
            }
        )

        assert outcome["success"] is True
        assert outcome["result"] == 3
        assert outcome["error"] is None

    def test_missing_result_variable_is_reported(self, frame_tables) -> None:
        """生成代码必须把结论写入 result；缺失时按执行失败返回而不是静默通过。"""
        tool = create_python_tool(frame_tables)

        outcome = tool.invoke(
            {
                "table_names": ["customers"],
                "code": "total = len(df)",
            }
        )

        assert outcome["success"] is False
        assert "result" in outcome["error"]

    def test_runtime_error_is_captured(self, frame_tables) -> None:
        tool = create_python_tool(frame_tables)

        outcome = tool.invoke(
            {
                "table_names": ["customers"],
                "code": "result = df['missing_column'].sum()",
            }
        )

        assert outcome["success"] is False
        assert outcome["result"] is None
        assert outcome["error"]

    def test_unknown_table_raises_from_tool_but_registry_catches(
        self, frame_tables
    ) -> None:
        """工具自身对参数错误抛异常；注册表把它转成统一结果。"""
        with pytest.raises(ValueError, match="Table not found"):
            create_python_tool(frame_tables).invoke(
                {"table_names": ["nope"], "code": "result = 1"}
            )

        outcome = invoke_tool(
            "python", frame_tables, {"table_names": ["nope"], "code": "result = 1"}
        )
        assert outcome["success"] is False
        assert "Table not found" in outcome["error"]

    def test_multiple_tables_exposed_via_dfs(self, frame_tables) -> None:
        tables = dict(frame_tables)
        tables["orders"] = TableResource(
            name="orders",
            source_type="csv",
            source_path="data/orders.csv",
            dataframe=pd.DataFrame({"customer_id": ["C001"], "amount": [120]}),
        )

        outcome = create_python_tool(tables).invoke(
            {
                "table_names": ["customers", "orders"],
                "code": "result = sorted(dfs.keys())",
            }
        )

        assert outcome["success"] is True
        assert outcome["result"] == ["customers", "orders"]

    def test_original_dataframe_is_not_mutated(self, frame_tables) -> None:
        """工具内部对 DataFrame 做副本，生成代码不应污染原始数据。"""
        original_columns = list(frame_tables["customers"].dataframe.columns)

        create_python_tool(frame_tables).invoke(
            {
                "table_names": ["customers"],
                "code": "df['new_col'] = 1\nresult = len(df.columns)",
            }
        )

        assert list(frame_tables["customers"].dataframe.columns) == original_columns


class TestSqlTool:
    def test_select_returns_rows(self, sqlite_tables) -> None:
        outcome = create_sql_tool(sqlite_tables).invoke(
            {"table_names": ["sales.orders"], "query": "SELECT * FROM orders"}
        )

        assert outcome["success"] is True
        assert len(outcome["result"]) == 3

    def test_with_clause_is_allowed(self, sqlite_tables) -> None:
        outcome = create_sql_tool(sqlite_tables).invoke(
            {
                "table_names": ["sales.orders"],
                "query": (
                    "WITH totals AS (SELECT SUM(amount) AS s FROM orders) "
                    "SELECT * FROM totals"
                ),
            }
        )

        assert outcome["success"] is True

    @pytest.mark.parametrize(
        "query",
        [
            "DROP TABLE orders",
            "DELETE FROM orders",
            "UPDATE orders SET amount = 0",
            "INSERT INTO orders VALUES (9, 9, 9)",
            "CREATE TABLE x (a INT)",
        ],
    )
    def test_write_statements_are_rejected(self, sqlite_tables, query: str) -> None:
        outcome = create_sql_tool(sqlite_tables).invoke(
            {"table_names": ["sales.orders"], "query": query}
        )

        assert outcome["success"] is False
        assert "Only SELECT or WITH" in outcome["error"]

    def test_leading_whitespace_is_tolerated(self, sqlite_tables) -> None:
        outcome = create_sql_tool(sqlite_tables).invoke(
            {"table_names": ["sales.orders"], "query": "   \n  SELECT 1"}
        )

        assert outcome["success"] is True

    def test_sql_error_is_captured(self, sqlite_tables) -> None:
        outcome = create_sql_tool(sqlite_tables).invoke(
            {"table_names": ["sales.orders"], "query": "SELECT * FROM missing"}
        )

        assert outcome["success"] is False
        assert outcome["error"]

    def test_cross_database_query_is_rejected(self, sqlite_tables, frame_tables) -> None:
        """同一查询只能针对一个数据库文件。"""
        mixed = {**sqlite_tables, **frame_tables}

        outcome = invoke_tool(
            "sql",
            mixed,
            {"table_names": ["sales.orders", "customers"], "query": "SELECT 1"},
        )

        assert outcome["success"] is False


class TestToolRegistry:
    def test_available_tools_are_python_and_sql(self) -> None:
        assert set(available_tools()) == {"python", "sql"}

    def test_unknown_tool_returns_error_not_exception(self, frame_tables) -> None:
        outcome = invoke_tool(
            "plot", frame_tables, {"table_names": ["customers"], "code": "pass"}
        )

        assert outcome["success"] is False
        assert "Unsupported tool" in outcome["error"]

    def test_result_shape_is_always_the_same(self, frame_tables) -> None:
        success = invoke_tool(
            "python", frame_tables, {"table_names": ["customers"], "code": "result = 1"}
        )
        failure = invoke_tool(
            "python", frame_tables, {"table_names": ["customers"], "code": "raise ValueError('x')"}
        )

        for outcome in (success, failure):
            assert set(outcome) == {"success", "result", "error"}

    def test_tool_instance_is_reused_for_same_tables(self, frame_tables) -> None:
        arguments = {"table_names": ["customers"], "code": "result = 1"}

        first = invoke_tool("python", frame_tables, arguments)
        second = invoke_tool("python", frame_tables, arguments)

        assert first == second

    def test_clear_cache_does_not_break_invocation(self, frame_tables) -> None:
        arguments = {"table_names": ["customers"], "code": "result = 1"}
        invoke_tool("python", frame_tables, arguments)

        clear_tool_cache()

        outcome = invoke_tool("python", frame_tables, arguments)
        assert outcome["success"] is True
