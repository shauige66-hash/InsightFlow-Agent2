"""大结果外置后的恢复与消费。

背景缺陷：`load_result` 已定义但工作流里没有消费路径。结果超过 32 KiB 被外置后，
state 里只剩 `{"__kind__": "externalized", ...}` 摘要，而绘图节点把它当数据用——
图会画错（或画空），工作流却不报错。报告节点则把整份数据 JSON 塞进 Prompt。

本文件覆盖两条消费路径的区别：

    需要真实数据（绘图）  → load_result，校验归属/路径/哈希
    只需要概要（报告 ）   → build_preview，有界样本 + 明确的"这是预览"
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from utils.result_artifacts import (
    INLINE_LIMIT_BYTES,
    PREVIEW_MAX_RECORDS,
    ResultArtifactError,
    build_preview,
    from_jsonable,
    load_result,
    store_result,
    to_jsonable,
    unwrap_records,
)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "outputs"


RUN_ID = "p2b3-run"
TASK_ID = "task_1"


def store(value, root: Path, run_id: str = RUN_ID, task_id: str = TASK_ID):
    return store_result(value, run_id=run_id, task_id=task_id, output_root=root)


def big_frame(rows: int = 20_000) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": [f"C{index:06d}" for index in range(rows)],
            "amount": [index * 1.5 for index in range(rows)],
            "region": ["East" if index % 2 else "West" for index in range(rows)],
        }
    )


class TestSmallResultsStayInline:
    def test_small_result_is_inlined(self, root: Path) -> None:
        inline, artifact = store({"total": 42}, root)

        assert artifact is None
        assert inline == {"total": 42}

    def test_small_dataframe_is_inlined_with_kind_marker(self, root: Path) -> None:
        frame = pd.DataFrame({"a": [1, 2, 3]})
        inline, artifact = store(frame, root)

        assert artifact is None
        assert inline["__kind__"] == "dataframe"
        assert inline["columns"] == ["a"]

    def test_inlined_result_round_trips(self, root: Path) -> None:
        frame = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        inline, artifact = store(frame, root)

        restored = load_result(inline, artifact, run_id=RUN_ID, output_root=root)

        assert isinstance(restored, pd.DataFrame)
        pd.testing.assert_frame_equal(restored, frame)


class TestLargeResultsAreExternalized:
    def test_result_over_limit_is_externalized(self, root: Path) -> None:
        frame = big_frame()
        inline, artifact = store(frame, root)

        assert artifact is not None
        assert inline["__kind__"] == "externalized"
        assert inline["row_count"] == 20_000
        assert inline["size"] > INLINE_LIMIT_BYTES
        # 摘要是明确的：带着列名，但不含任何一行数据
        assert inline["columns"] == ["customer_id", "amount", "region"]
        assert "records" not in inline

    def test_artifact_file_exists_and_hash_matches(self, root: Path) -> None:
        _, artifact = store(big_frame(), root)

        path = Path(artifact["path"])
        assert path.exists()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]

    def test_state_remains_checkpoint_serializable(self, root: Path) -> None:
        """外置后 state 里只留摘要，必须能被 JSON 序列化。"""
        inline, artifact = store(big_frame(), root)

        encoded = json.dumps(
            {"result": inline, "result_artifact": artifact}, ensure_ascii=False
        )
        restored = json.loads(encoded)

        assert restored["result"]["__kind__"] == "externalized"

    def test_20000_rows_are_fully_recoverable(self, root: Path) -> None:
        frame = big_frame()
        inline, artifact = store(frame, root)

        restored = load_result(inline, artifact, run_id=RUN_ID, output_root=root)

        assert isinstance(restored, pd.DataFrame)
        assert len(restored) == 20_000
        assert list(restored.columns) == list(frame.columns)
        # 抽查首尾，确认不是只还原了一部分
        assert restored.iloc[0]["customer_id"] == "C000000"
        assert restored.iloc[-1]["customer_id"] == "C019999"
        assert restored.iloc[-1]["amount"] == pytest.approx(19_999 * 1.5)

    def test_recovery_works_in_a_fresh_process_view(self, root: Path) -> None:
        """模拟"重启后重新加载"：只用 state 里的摘要与引用即可还原。

        用足够大的数据集以确保走外置路径（小结果内联，不需要重新加载）。
        """
        frame = big_frame()
        inline, artifact = store(frame, root)
        assert artifact is not None

        # 只保留可持久化的两个字段，丢掉原始对象
        persisted_result = json.loads(json.dumps(inline))
        persisted_artifact = json.loads(json.dumps(artifact))
        del frame

        restored = load_result(
            persisted_result, persisted_artifact, run_id=RUN_ID, output_root=root
        )

        assert len(restored) == 20_000


class TestIntegrityFailuresAreExplicit:
    def test_missing_file_fails_loudly(self, root: Path) -> None:
        _, artifact = store(big_frame(), root)
        Path(artifact["path"]).unlink()

        with pytest.raises(ResultArtifactError) as captured:
            load_result(None, artifact, run_id=RUN_ID, output_root=root)

        assert "丢失" in str(captured.value)

    def test_hash_mismatch_fails_loudly(self, root: Path) -> None:
        _, artifact = store(big_frame(), root)
        path = Path(artifact["path"])
        path.write_text("[{\"tampered\": true}]", encoding="utf-8")

        with pytest.raises(ResultArtifactError) as captured:
            load_result(None, artifact, run_id=RUN_ID, output_root=root)

        assert "已变化" in str(captured.value)

    def test_cross_run_reference_is_refused(self, root: Path) -> None:
        """另一个 Run 的产物不能被本次 Run 引用。

        用足够大的数据集，确保确实产生了产物文件；行数太少会走内联路径，
        那条路径不涉及引用，也就测不到归属校验。
        """
        _, other_artifact = store(big_frame(), root, run_id="other-run")
        assert other_artifact is not None

        with pytest.raises(ResultArtifactError) as captured:
            load_result(
                None, other_artifact, run_id=RUN_ID, output_root=root
            )

        assert "不属于本次 Run" in str(captured.value)

    def test_path_outside_output_root_is_refused(self, tmp_path: Path) -> None:
        root = tmp_path / "outputs"
        outside = tmp_path / "outside.json"
        outside.write_text("[{\"secret\": 1}]", encoding="utf-8")

        artifact = {
            "path": str(outside),
            "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
            "size": outside.stat().st_size,
            "row_count": 1,
        }

        with pytest.raises(ResultArtifactError) as captured:
            load_result(None, artifact, run_id=RUN_ID, output_root=root)

        assert "越界" in str(captured.value)

    def test_directory_instead_of_file_is_refused(self, root: Path) -> None:
        directory = root / RUN_ID / "results"
        directory.mkdir(parents=True, exist_ok=True)

        artifact = {
            "path": str(directory),
            "sha256": "0" * 64,
            "size": 0,
            "row_count": None,
        }

        with pytest.raises(ResultArtifactError):
            load_result(None, artifact, run_id=RUN_ID, output_root=root)


class TestPreviewIsBoundedAndLabelled:
    def test_preview_of_large_result_is_truncated(self, root: Path) -> None:
        inline, artifact = store(big_frame(), root)

        preview = build_preview(inline, artifact, run_id=RUN_ID, output_root=root)

        assert preview["kind"] == "externalized"
        assert preview["truncated"] is True
        assert preview["total_row_count"] == 20_000
        assert preview["preview_rows"] <= PREVIEW_MAX_RECORDS
        assert len(preview["sample"]) <= PREVIEW_MAX_RECORDS
        # 明确告知这是预览
        assert "预览" in preview["note"]

    def test_preview_does_not_embed_all_rows(self, root: Path) -> None:
        """把预览塞进 Prompt 时不能带全量数据。"""
        inline, artifact = store(big_frame(), root)
        preview = build_preview(inline, artifact, run_id=RUN_ID, output_root=root)

        encoded = json.dumps(preview, ensure_ascii=False)

        assert len(encoded) < 16 * 1024, f"预览过大: {len(encoded)} 字节"

    def test_preview_reports_columns_and_row_count(self, root: Path) -> None:
        inline, artifact = store(big_frame(), root)
        preview = build_preview(inline, artifact, run_id=RUN_ID, output_root=root)

        assert preview["columns"] == ["customer_id", "amount", "region"]
        assert preview["row_count"] == 20_000

    def test_preview_of_inline_result_is_marked_inline(self, root: Path) -> None:
        inline, artifact = store([{"a": 1}, {"a": 2}], root)

        preview = build_preview(inline, artifact, run_id=RUN_ID, output_root=root)

        assert preview["kind"] == "inline"
        assert preview["total_row_count"] == 2
        assert preview["truncated"] is False

    def test_preview_of_scalar_result(self, root: Path) -> None:
        inline, artifact = store(42, root)

        preview = build_preview(inline, artifact, run_id=RUN_ID, output_root=root)

        assert preview["kind"] == "scalar"
        assert preview["value"] == 42

    def test_preview_degrades_when_artifact_is_missing(self, root: Path) -> None:
        """预览可降级：读不到产物时说明原因，而不是让报告整体失败。"""
        inline, artifact = store(big_frame(), root)
        Path(artifact["path"]).unlink()

        preview = build_preview(inline, artifact, run_id=RUN_ID, output_root=root)

        assert preview["available"] is False
        assert preview["truncated"] is True
        assert "不可用" in preview["note"]
        # 仍保留形状信息，便于报告说明"结果存在但无法读取"
        assert preview["total_row_count"] == 20_000

    def test_preview_distinguishes_from_real_data(self, root: Path) -> None:
        """预览与真实数据必须能区分：预览里没有完整记录。"""
        inline, artifact = store(big_frame(), root)

        real = load_result(inline, artifact, run_id=RUN_ID, output_root=root)
        preview = build_preview(inline, artifact, run_id=RUN_ID, output_root=root)

        assert len(real) == 20_000
        assert len(preview["sample"]) < len(real)
        assert preview["truncated"] is True


class TestTypeRoundTrips:
    @pytest.mark.parametrize(
        "value",
        [
            42,
            3.5,
            "text",
            True,
            None,
            [1, 2, 3],
            {"a": 1, "b": [2, 3]},
            {"nested": {"deep": {"value": 1}}},
        ],
    )
    def test_plain_values_round_trip(self, value) -> None:
        assert from_jsonable(to_jsonable(value)) == value

    def test_dataframe_round_trip(self) -> None:
        frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})

        restored = from_jsonable(to_jsonable(frame))

        assert isinstance(restored, pd.DataFrame)
        assert list(restored.columns) == ["a", "b"]
        assert restored["a"].tolist() == [1, 2]

    def test_empty_dataframe_keeps_columns(self) -> None:
        frame = pd.DataFrame(columns=["a", "b"])

        restored = from_jsonable(to_jsonable(frame))

        assert isinstance(restored, pd.DataFrame)
        assert list(restored.columns) == ["a", "b"]
        assert len(restored) == 0

    def test_series_round_trip(self) -> None:
        series = pd.Series([1, 2, 3], name="counts")

        restored = from_jsonable(to_jsonable(series))

        assert isinstance(restored, pd.Series)
        assert restored.tolist() == [1, 2, 3]

    def test_ndarray_round_trip(self) -> None:
        array = np.array([[1, 2], [3, 4]])

        restored = from_jsonable(to_jsonable(array))

        assert isinstance(restored, np.ndarray)
        assert restored.tolist() == [[1, 2], [3, 4]]

    def test_numpy_scalars_become_python_scalars(self) -> None:
        assert isinstance(to_jsonable(np.int64(5)), int)
        assert isinstance(to_jsonable(np.float64(1.5)), float)
        assert isinstance(to_jsonable(np.bool_(True)), bool)

    def test_timestamp_round_trip(self) -> None:
        stamp = pd.Timestamp("2024-01-02T03:04:05")

        encoded = to_jsonable(stamp)

        assert encoded == "2024-01-02T03:04:05"

    def test_externalized_marker_is_not_mistaken_for_data(self) -> None:
        """未知标记不能被"还原"成数据——那会把摘要当内容用。"""
        summary = {"__kind__": "externalized", "row_count": 100}

        assert from_jsonable(summary) == summary


class TestUnwrapForPlotting:
    """绘图的 result_df 需要记录序列，不是带标记的信封。"""

    def test_dataframe_envelope_unwraps_to_records(self) -> None:
        records = unwrap_records(
            {"__kind__": "dataframe", "columns": ["a"], "records": [{"a": 1}, {"a": 2}]}
        )

        assert records == [{"a": 1}, {"a": 2}]

    def test_plain_records_pass_through(self) -> None:
        records = [{"a": 1}]

        assert unwrap_records(records) == records

    def test_scalar_passes_through(self) -> None:
        assert unwrap_records(42) == 42

    def test_plot_tool_builds_a_dataframe_from_records(self) -> None:
        """端到端衔接：拆封后的记录必须能被绘图侧的推导函数识别。"""
        from sandbox.bootstrap import build_result_dataframe

        records = unwrap_records(
            {
                "__kind__": "dataframe",
                "columns": ["region", "amount"],
                "records": [{"region": "East", "amount": 3}, {"region": "West", "amount": 5}],
            }
        )

        frame = build_result_dataframe(records)

        assert frame is not None
        assert len(frame) == 2
        assert list(frame.columns) == ["region", "amount"]
