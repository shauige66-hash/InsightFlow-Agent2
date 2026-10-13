"""Adaptive Context 的分层采样策略。

上下文是最容易悄悄膨胀的部分：schema 永远完整内联，行数据按规模分档。
这里把三档边界钉住，因为分档判据 `rows * cols` 一旦算错，
大表会被整表塞进 prompt。
"""

from __future__ import annotations

import pandas as pd
import pytest

from context.context_builder import (
    build_all_contexts,
    build_context,
    build_dataframe_context,
    build_sqlite_context,
)
from ingestion.models import TableResource

from tests.conftest import CUSTOMERS_CSV, SALES_SQLITE


def frame_resource(frame: pd.DataFrame, name: str = "t") -> TableResource:
    return TableResource(
        name=name,
        source_type="csv",
        source_path=f"data/{name}.csv",
        dataframe=frame,
    )


class TestContextLevels:
    def test_small_table_returns_all_rows(self) -> None:
        """rows * cols <= 500 时整表内联。"""
        frame = pd.DataFrame({"a": range(50), "b": range(50)})  # 100 单元格

        context = build_dataframe_context(frame_resource(frame))

        assert context["context_level"] == "full"
        assert isinstance(context["data"], list)
        assert len(context["data"]) == 50

    def test_boundary_at_500_cells_is_full(self) -> None:
        frame = pd.DataFrame({f"c{i}": [1, 2] for i in range(250)})  # 500 单元格

        context = build_dataframe_context(frame_resource(frame))

        assert context["context_level"] == "full"

    def test_medium_table_returns_head_and_tail(self) -> None:
        """500 < cells <= 5000 时只给头尾各 10 行。"""
        frame = pd.DataFrame(
            {f"c{index}": range(200) for index in range(10)}  # 200 * 10 = 2000 单元格
        )

        context = build_dataframe_context(frame_resource(frame))

        assert context["context_level"] == "sampled"
        assert set(context["data"]) == {"head", "tail"}
        assert len(context["data"]["head"]) == 10
        assert len(context["data"]["tail"]) == 10

    def test_large_table_returns_five_rows_each_side(self) -> None:
        frame = pd.DataFrame({"a": range(3000), "b": range(3000)})  # 6000 单元格

        context = build_dataframe_context(frame_resource(frame))

        assert context["context_level"] == "minimal"
        assert len(context["data"]["head"]) == 5
        assert len(context["data"]["tail"]) == 5

    def test_row_and_column_counts_are_reported(self) -> None:
        frame = pd.DataFrame({"a": range(7), "b": range(7)})

        context = build_dataframe_context(frame_resource(frame))

        assert context["row_count"] == 7
        assert context["column_count"] == 2


class TestSchemaInlining:
    def test_schema_lists_every_column(self) -> None:
        """schema 不截断，模型必须能看到全部列名与类型。"""
        frame = pd.DataFrame({f"col_{i}": [1] for i in range(30)})

        context = build_dataframe_context(frame_resource(frame))

        assert len(context["schema"]["columns"]) == 30
        assert "col_29" in context["schema"]["columns"]

    def test_schema_survives_minimal_level(self) -> None:
        """即使行数据被压缩到 5 行，schema 仍然完整——这是有意的取舍。"""
        frame = pd.DataFrame(
            {f"col_{i}": range(1000) for i in range(10)}  # 10000 单元格
        )

        context = build_dataframe_context(frame_resource(frame))

        assert context["context_level"] == "minimal"
        assert len(context["schema"]["columns"]) == 10

    def test_dtype_is_stringified(self) -> None:
        frame = pd.DataFrame({"n": [1, 2], "s": ["a", "b"]})

        columns = build_dataframe_context(frame_resource(frame))["schema"]["columns"]

        assert all(isinstance(value, str) for value in columns.values())


class TestSqliteContext:
    def test_sqlite_context_includes_database_table(self) -> None:
        resource = TableResource(
            name="sales.orders",
            source_type="sqlite",
            source_path=str(SALES_SQLITE),
            database_table="orders",
        )

        context = build_sqlite_context(resource)

        assert context["database_table"] == "orders"
        assert context["context_level"] == "minimal"
        assert context["row_count"] == 3
        assert len(context["data"]) == 3

    def test_sqlite_context_requires_database_table(self) -> None:
        resource = TableResource(
            name="sales.orders",
            source_type="sqlite",
            source_path=str(SALES_SQLITE),
        )

        with pytest.raises(ValueError, match="SQLite table name missing"):
            build_sqlite_context(resource)


class TestContextErrors:
    def test_dataframe_context_rejects_sqlite_resource(self) -> None:
        resource = TableResource(
            name="sales.orders",
            source_type="sqlite",
            source_path=str(SALES_SQLITE),
            database_table="orders",
        )

        with pytest.raises(ValueError, match="not a DataFrame resource"):
            build_dataframe_context(resource)

    def test_unsupported_resource_type_raises(self) -> None:
        resource = TableResource(
            name="weird",
            source_type="parquet",
            source_path="data/weird.parquet",
        )

        with pytest.raises(ValueError, match="Unsupported resource type"):
            build_context(resource)


class TestBuildAllContexts:
    def test_covers_every_table(self) -> None:
        tables = {
            "customers": frame_resource(
                pd.read_csv(CUSTOMERS_CSV), "customers"
            ),
            "sales.orders": TableResource(
                name="sales.orders",
                source_type="sqlite",
                source_path=str(SALES_SQLITE),
                database_table="orders",
            ),
        }

        contexts = build_all_contexts(tables)

        assert set(contexts) == {"customers", "sales.orders"}
        assert contexts["customers"]["context_level"] == "full"
        assert contexts["sales.orders"]["context_level"] == "minimal"

    def test_empty_tables_returns_empty_mapping(self) -> None:
        assert build_all_contexts({}) == {}


def test_context_is_json_serializable() -> None:
    """data_context 会被 json.dumps 进 prompt，必须可序列化。"""
    import json

    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})

    context = build_dataframe_context(frame_resource(frame))

    json.dumps(context)
