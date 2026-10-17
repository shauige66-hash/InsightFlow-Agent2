"""表资源快照与解析。

覆盖 P1 要求的恢复能力：CSV、Excel 多 Sheet、SQLite，以及
"来源变化必须拒绝恢复"的数据版本校验。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from ingestion.resources import (
    ResourceResolver,
    ResourceUnavailableError,
    TableSnapshot,
    build_snapshots,
    snapshots_from_state,
    snapshots_to_state,
)


@pytest.fixture
def csv_source(tmp_path: Path) -> Path:
    path = tmp_path / "customers.csv"
    path.write_text(
        "customer_id,customer_name,region\nC001,Alice,East\nC002,Bob,West\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def excel_source(tmp_path: Path) -> Path:
    """两个 Sheet 的工作簿。"""
    path = tmp_path / "company.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"id": [1, 2], "name": ["a", "b"]}).to_excel(
            writer, sheet_name="employees", index=False
        )
        pd.DataFrame({"month": ["Jan"], "revenue": [100]}).to_excel(
            writer, sheet_name="financials", index=False
        )
    return path


@pytest.fixture
def sqlite_source(tmp_path: Path) -> Path:
    path = tmp_path / "sales.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE orders (id INTEGER, amount REAL)")
        connection.execute("INSERT INTO orders VALUES (1, 10.0), (2, 20.0)")
        connection.execute("CREATE TABLE regions (id INTEGER, name TEXT)")
    return path


class TestSnapshotConstruction:
    def test_csv_snapshot_is_json_serializable(self, csv_source: Path) -> None:
        snapshots = build_snapshots([str(csv_source)])

        assert "customers" in snapshots
        payload = snapshots_to_state(snapshots)

        # 必须能过 Checkpoint 的序列化
        assert json.loads(json.dumps(payload)) == payload

    def test_snapshot_carries_identity_and_fingerprint(self, csv_source: Path) -> None:
        snapshot = build_snapshots([str(csv_source)])["customers"]

        assert snapshot.source_type == "csv"
        assert snapshot.source_path == str(csv_source)
        assert snapshot.content_hash
        assert snapshot.file_size > 0

    def test_excel_produces_one_snapshot_per_sheet(self, excel_source: Path) -> None:
        snapshots = build_snapshots([str(excel_source)])

        assert set(snapshots) == {"company.employees", "company.financials"}
        assert snapshots["company.employees"].locator == "employees"
        assert snapshots["company.employees"].source_type == "excel"

    def test_sqlite_produces_one_snapshot_per_table(self, sqlite_source: Path) -> None:
        snapshots = build_snapshots([str(sqlite_source)])

        assert set(snapshots) == {"sales.orders", "sales.regions"}
        assert snapshots["sales.orders"].database_table == "orders"
        assert snapshots["sales.orders"].source_type == "sqlite"

    def test_missing_source_is_reported_at_snapshot_time(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            build_snapshots([str(tmp_path / "nope.csv")])

    def test_unsupported_extension_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.txt"
        path.write_text("hello", encoding="utf-8")

        with pytest.raises(ValueError, match="Unsupported"):
            build_snapshots([str(path)])


class TestResolverRestoresResources:
    def test_csv_restores_dataframe(self, csv_source: Path) -> None:
        resolver = ResourceResolver()
        snapshots = build_snapshots([str(csv_source)])

        tables = resolver.resolve_all(snapshots)

        assert set(tables) == {"customers"}
        assert len(tables["customers"].dataframe) == 2
        assert list(tables["customers"].dataframe.columns)[:2] == [
            "customer_id",
            "customer_name",
        ]

    def test_excel_restores_the_right_sheet(self, excel_source: Path) -> None:
        resolver = ResourceResolver()
        snapshots = build_snapshots([str(excel_source)])

        tables = resolver.resolve_all(snapshots)

        assert set(tables["company.employees"].dataframe.columns) == {"id", "name"}
        assert set(tables["company.financials"].dataframe.columns) == {
            "month",
            "revenue",
        }

    def test_sqlite_resource_has_no_dataframe(self, sqlite_source: Path) -> None:
        """SQLite 表由 SQL 工具按需查询，不整表载入。"""
        resolver = ResourceResolver()
        snapshots = build_snapshots([str(sqlite_source)])

        tables = resolver.resolve_all(snapshots)

        assert tables["sales.orders"].dataframe is None
        assert tables["sales.orders"].database_table == "orders"

    def test_repeated_resolution_reuses_cache(self, csv_source: Path) -> None:
        resolver = ResourceResolver()
        snapshots = build_snapshots([str(csv_source)])

        first = resolver.resolve(snapshots["customers"])
        second = resolver.resolve(snapshots["customers"])

        assert first is second, "同一份来源被重复读取"
        assert resolver.cached_count == 1


class TestVersionValidation:
    def test_changed_content_is_refused(self, csv_source: Path) -> None:
        resolver = ResourceResolver()
        snapshots = build_snapshots([str(csv_source)])

        csv_source.write_text("customer_id\nZZZ\n", encoding="utf-8")

        with pytest.raises(ResourceUnavailableError, match="内容已变化"):
            resolver.resolve(snapshots["customers"])

    def test_deleted_source_is_refused(self, csv_source: Path) -> None:
        resolver = ResourceResolver()
        snapshots = build_snapshots([str(csv_source)])

        csv_source.unlink()

        with pytest.raises(ResourceUnavailableError, match="不存在"):
            resolver.resolve(snapshots["customers"])

    def test_removed_excel_sheet_is_refused(self, excel_source: Path) -> None:
        snapshots = build_snapshots([str(excel_source)])
        target = snapshots["company.employees"]

        # 重写工作簿并删掉其中一个 sheet（内容指纹也随之变化，
        # 但这里通过伪造指纹来单独验证 sheet 检查）
        with pd.ExcelWriter(excel_source) as writer:
            pd.DataFrame({"id": [1]}).to_excel(writer, sheet_name="other", index=False)

        from ingestion.resources import file_content_hash

        forged = TableSnapshot(
            name=target.name,
            source_type=target.source_type,
            source_path=target.source_path,
            content_hash=file_content_hash(excel_source),
            file_size=target.file_size,
            locator="employees",
        )

        with pytest.raises(ResourceUnavailableError, match="sheet"):
            ResourceResolver().resolve(forged)

    def test_removed_sqlite_table_is_refused(self, sqlite_source: Path) -> None:
        snapshots = build_snapshots([str(sqlite_source)])
        target = snapshots["sales.orders"]

        with sqlite3.connect(sqlite_source) as connection:
            connection.execute("DROP TABLE orders")

        from ingestion.resources import file_content_hash

        forged = TableSnapshot(
            name=target.name,
            source_type=target.source_type,
            source_path=target.source_path,
            content_hash=file_content_hash(sqlite_source),
            file_size=target.file_size,
            locator="orders",
            database_table="orders",
        )

        with pytest.raises(ResourceUnavailableError, match="表"):
            ResourceResolver().resolve(forged)

    def test_cache_does_not_serve_stale_data_for_same_path(
        self, csv_source: Path
    ) -> None:
        """同一路径、内容变化后必须重新读取，不能命中旧缓存。"""
        resolver = ResourceResolver()
        first_snapshot = build_snapshots([str(csv_source)])["customers"]
        resolver.resolve(first_snapshot)

        csv_source.write_text(
            "customer_id,customer_name,region\nN001,New,North\n", encoding="utf-8"
        )
        second_snapshot = build_snapshots([str(csv_source)])["customers"]

        # 指纹不同 => 缓存键不同 => 不会复用旧 DataFrame
        resource = resolver.resolve(second_snapshot)

        assert len(resource.dataframe) == 1
        assert resource.dataframe.iloc[0]["customer_id"] == "N001"


class TestSnapshotRoundTrip:
    def test_state_round_trip_preserves_everything(self, excel_source: Path) -> None:
        snapshots = build_snapshots([str(excel_source)])

        payload = snapshots_to_state(snapshots)
        restored = snapshots_from_state(json.loads(json.dumps(payload)))

        assert restored == snapshots

    def test_empty_state_yields_no_snapshots(self) -> None:
        assert snapshots_from_state(None) == {}
        assert snapshots_from_state({}) == {}
