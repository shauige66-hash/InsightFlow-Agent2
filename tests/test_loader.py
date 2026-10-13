"""数据加载与 TableResource 契约。

加载器决定后续路由（有 DataFrame 走 Python，sqlite 走 SQL），
因此它对每种数据源产出的 TableResource 形状必须稳定。
"""

from __future__ import annotations

import pytest

from ingestion.loader import load_data_sources
from tests.conftest import CUSTOMERS_CSV, FINANCIALS_CSV, ORDERS_CSV, SALES_SQLITE


class TestCsvLoading:
    def test_csv_becomes_dataframe_resource(self) -> None:
        tables = load_data_sources([str(CUSTOMERS_CSV)])

        assert set(tables) == {"customers"}
        resource = tables["customers"]
        assert resource.source_type == "csv"
        assert resource.dataframe is not None
        assert list(resource.dataframe.columns) == [
            "customer_id",
            "customer_name",
            "region",
        ]
        assert len(resource.dataframe) == 5
        assert resource.database_table is None

    def test_resource_name_is_file_stem(self) -> None:
        tables = load_data_sources([str(ORDERS_CSV)])
        assert "orders" in tables

    def test_multiple_csv_sources_are_merged(self) -> None:
        tables = load_data_sources([str(CUSTOMERS_CSV), str(FINANCIALS_CSV)])

        assert set(tables) == {"customers", "financials"}
        assert all(r.dataframe is not None for r in tables.values())


class TestSqliteLoading:
    def test_each_table_becomes_its_own_resource(self) -> None:
        """SQLite 的每张表独立成一个资源，键名带库名前缀避免同名文件冲突。"""
        tables = load_data_sources([str(SALES_SQLITE)])

        assert set(tables) == {"sales.customers", "sales.orders"}
        for resource in tables.values():
            assert resource.source_type == "sqlite"
            assert resource.database_table is not None
            # sqlite 资源不预载 DataFrame，数据在执行阶段由 SQL 取
            assert resource.dataframe is None

    def test_database_table_is_bare_name(self) -> None:
        tables = load_data_sources([str(SALES_SQLITE)])

        assert tables["sales.orders"].database_table == "orders"
        assert tables["sales.customers"].database_table == "customers"


class TestExcelLoading:
    def test_each_sheet_becomes_its_own_resource(self, project_root) -> None:
        workbook = project_root / "data" / "company.xlsx"
        if not workbook.exists():
            pytest.skip("company.xlsx 不存在，跳过 Excel 用例")

        tables = load_data_sources([str(workbook)])

        assert tables, "Excel 至少应产出一张表"
        for key, resource in tables.items():
            assert key.startswith("company.")
            assert resource.source_type == "excel"
            assert resource.dataframe is not None


class TestLoaderErrors:
    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="Data source not found"):
            load_data_sources(["data/does_not_exist.csv"])

    def test_unsupported_suffix_raises(self, tmp_path) -> None:
        unsupported = tmp_path / "notes.json"
        unsupported.write_text("{}", encoding="utf-8")

        with pytest.raises(ValueError, match="Unsupported data source"):
            load_data_sources([str(unsupported)])

    def test_empty_source_list_returns_empty_mapping(self) -> None:
        assert load_data_sources([]) == {}
