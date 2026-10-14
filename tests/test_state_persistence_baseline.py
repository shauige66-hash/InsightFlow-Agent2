"""P1-A 前置失败测试：P0 的 state 无法被 Checkpointer 持久化。

这些用例在改造前应当失败，用来证明问题真实存在而不是假设。
改造完成后它们会转为通过，并成为回归保护。

失败原因分三类：
1. `tables` 里是 TableResource（含 pandas DataFrame），JSON 无法编码；
2. `execution_results[].result` 可能是 DataFrame / ndarray 等任意对象；
3. `model_factory` 是运行期对象（含 resolver 闭包），跨进程恢复没有意义。
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from ingestion.models import TableResource
from state import InsightFlowState

from tests.conftest import CUSTOMERS_CSV, SALES_SQLITE

# 真实运行快照中的字段：P1 必须让这些值都能被 JSON 编码
SERIALIZABLE_KEYS = (
    "run_id",
    "data_sources",
    "knowledge_sources",
    "analysis_plan",
    "current_task_index",
    "execution_results",
    "failed_tasks",
    "artifacts",
    "visualization_plan",
    "analysis_status",
)

# 运行期对象或不可序列化值：这些键不允许进入 checkpoint
RUNTIME_ONLY_KEYS = ("tables", "model_factory")


def _resource_frame() -> TableResource:
    return TableResource(
        name="customers",
        source_type="csv",
        source_path=str(CUSTOMERS_CSV),
        dataframe=pd.read_csv(CUSTOMERS_CSV),
    )


class TestCurrentStateCannotBePersisted:
    def test_dataframe_resource_is_not_json_serializable(self) -> None:
        resource = _resource_frame()

        with pytest.raises(TypeError):
            json.dumps({"tables": {"customers": resource}})

    def test_dataframe_in_execution_result_is_not_json_serializable(self) -> None:
        payload = {
            "execution_results": [
                {
                    "task_id": "task_1",
                    "tool_used": "python",
                    "code": "result = df",
                    "success": True,
                    "result": pd.DataFrame({"amount": [1, 2]}),
                    "error": None,
                }
            ]
        }

        with pytest.raises(TypeError):
            json.dumps(payload)

    def test_numpy_scalar_in_execution_result_is_not_json_serializable(self) -> None:
        import numpy as np

        payload = {"execution_results": [{"result": np.int64(5)}]}

        with pytest.raises(TypeError):
            json.dumps(payload)

    def test_runtime_objects_are_no_longer_in_state(self) -> None:
        """改造后 state 不再持有 DataFrame 与模型客户端。

        这两个键一旦留在 state 里，Checkpoint 就会在序列化时失败，
        或者在恢复后持有一个跨进程无意义的对象。
        """
        for key in RUNTIME_ONLY_KEYS:
            assert key not in InsightFlowState.__annotations__, (
                f"state 仍声明了运行期键 {key}"
            )

    def test_state_declares_serializable_resource_identity(self) -> None:
        """表以可序列化快照的形式存在于 state 中。"""
        assert "table_snapshot" in InsightFlowState.__annotations__

    def test_state_annotations_are_json_encodable(self) -> None:
        """真实运行快照中的字段都必须能 JSON 编码。"""
        sample = {
            "run_id": "run_x",
            "data_sources": ["data/customers.csv"],
            "knowledge_sources": [],
            "table_snapshot": {
                "customers": {
                    "name": "customers",
                    "source_type": "csv",
                    "source_path": "data/customers.csv",
                    "content_hash": "abc",
                    "file_size": 114,
                    "locator": None,
                    "database_table": None,
                }
            },
            "analysis_plan": [],
            "current_task_index": 0,
            "execution_results": [
                {
                    "task_id": "task_1",
                    "tool_used": "python",
                    "code": "result = 1",
                    "success": True,
                    "result": {"__kind__": "externalized", "size": 100},
                    "result_artifact": {
                        "path": "outputs/run_x/results/task_1.json",
                        "sha256": "deadbeef",
                        "size": 100,
                        "row_count": 3,
                    },
                    "error": None,
                }
            ],
            "failed_tasks": [],
            "artifacts": ["outputs/run_x/analysis_report.md"],
            "visualization_plan": [],
            "analysis_status": "completed",
            "run_status": "completed",
            "approvals": [
                {"approval_id": "ap-1", "decision": "approve", "node": "plan_review"}
            ],
            "pending_approval": None,
            "completed_operations": {
                "artifact:run_x": {"status": "completed", "path": "x.md"}
            },
        }

        for key in SERIALIZABLE_KEYS:
            assert key in sample, f"样本缺少字段 {key}"

        encoded = json.dumps(sample, ensure_ascii=False)
        assert json.loads(encoded) == sample

    def test_sqlite_resource_is_serializable_without_dataframe(self) -> None:
        """sqlite 资源本身只含元数据，说明问题集中在 DataFrame 上。"""
        resource = TableResource(
            name="sales.orders",
            source_type="sqlite",
            source_path=str(SALES_SQLITE),
            database_table="orders",
        )

        snapshot = {
            "name": resource.name,
            "source_type": resource.source_type,
            "source_path": resource.source_path,
            "database_table": resource.database_table,
        }

        assert json.loads(json.dumps(snapshot)) == snapshot
