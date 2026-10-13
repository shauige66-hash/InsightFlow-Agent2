"""工具路由。

`route_analysis_task` 是纯函数，只看资源元数据，不调用 LLM —— 这是
"不让模型决定执行方式"这一设计的具体落点，因此边界行为要钉死。
"""

from __future__ import annotations

import pandas as pd
import pytest

from agents.router import route_analysis_task
from ingestion.models import TableResource


def dataframe_resource(name: str = "customers") -> TableResource:
    return TableResource(
        name=name,
        source_type="csv",
        source_path=f"data/{name}.csv",
        dataframe=pd.DataFrame({"amount": [1, 2]}),
    )


def sqlite_resource(name: str = "sales.orders") -> TableResource:
    return TableResource(
        name=name,
        source_type="sqlite",
        source_path="data/sales.sqlite",
        database_table=name.split(".")[-1],
    )


def task(target_tables: list[str], task_id: str = "task_1") -> dict:
    return {
        "task_id": task_id,
        "description": "d",
        "target_tables": target_tables,
        "expected_output": "e",
    }


class TestRouting:
    def test_dataframe_tables_route_to_python(self) -> None:
        tables = {"customers": dataframe_resource()}

        assert route_analysis_task(task(["customers"]), tables) == "python"

    def test_sqlite_tables_route_to_sql(self) -> None:
        tables = {"sales.orders": sqlite_resource("sales.orders")}

        assert route_analysis_task(task(["sales.orders"]), tables) == "sql"

    def test_multiple_dataframe_tables_route_to_python(self) -> None:
        tables = {
            "customers": dataframe_resource("customers"),
            "orders": dataframe_resource("orders"),
        }

        assert route_analysis_task(task(["customers", "orders"]), tables) == "python"

    def test_multiple_sqlite_tables_route_to_sql(self) -> None:
        tables = {
            "sales.orders": sqlite_resource("sales.orders"),
            "sales.customers": sqlite_resource("sales.customers"),
        }

        assert (
            route_analysis_task(task(["sales.orders", "sales.customers"]), tables)
            == "sql"
        )

    def test_excel_resource_with_dataframe_routes_to_python(self) -> None:
        """Excel 资源带 DataFrame，因此与 CSV 同样走 Python。"""
        excel = TableResource(
            name="company.sheet1",
            source_type="excel",
            source_path="data/company.xlsx",
            dataframe=pd.DataFrame({"a": [1]}),
        )

        route = route_analysis_task(
            task(["company.sheet1"]), {"company.sheet1": excel}
        )
        assert route == "python"


class TestRoutingErrors:
    def test_mixed_sources_raise(self) -> None:
        """同一任务混用 DataFrame 与 sqlite 无法用单一工具完成，必须显式失败。"""
        tables = {
            "customers": dataframe_resource("customers"),
            "sales.orders": sqlite_resource("sales.orders"),
        }

        with pytest.raises(ValueError, match="Cannot route mixed data sources"):
            route_analysis_task(task(["customers", "sales.orders"]), tables)

    def test_missing_table_raises(self) -> None:
        """未知表名必须报错，不能静默路由到某个工具。

        当前实现用未加守卫的下标访问，抛出的是 KeyError。这里锁定"必须抛异常"
        这一行为契约；异常类型不统一的问题记录在已知限制中。
        """
        with pytest.raises((KeyError, ValueError)):
            route_analysis_task(task(["unknown_table"]), {})

    def test_empty_target_tables_routes_to_python(self) -> None:
        """target_tables 为空时 all() 为真，落到 python 分支。
        这是既有行为，计划非空由 Planner 侧的计划校验负责。"""
        assert route_analysis_task(task([]), {}) == "python"
