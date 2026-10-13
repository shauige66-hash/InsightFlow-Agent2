"""State 结构契约。

`InsightFlowState` 是节点之间唯一的通信协议，字段语义一旦漂移，
条件路由就会读到错误的键。这里同时为 P1 的 Checkpoint 改造留一份
"哪些字段不能直接持久化"的可执行清单。
"""

from __future__ import annotations

import importlib
import inspect
import json

import pandas as pd
import pytest

from ingestion.models import TableResource
from state import (
    AnalysisTask,
    ExecutionResult,
    InsightFlowState,
    TaskFailure,
    VisualizationResult,
    VisualizationTask,
)

# 条件路由与节点实际读取的键。缺失这些键会导致 KeyError 或错误分支。
ROUTING_KEYS = (
    "current_task_index",
    "analysis_plan",
    "current_chart_index",
    "visualization_plan",
    "execution_error",
    "retry_count",
    "refine_count",
    "validation_passed",
    "plot_error",
    "plot_retry_count",
)

# 已知不能直接 JSON 序列化的字段。P1 引入 Checkpointer 时必须处理。
NON_SERIALIZABLE_KEYS = ("tables",)


def test_state_declares_routing_keys() -> None:
    for key in ROUTING_KEYS:
        assert key in InsightFlowState.__annotations__, f"state 缺少路由键 {key}"


def test_state_is_total_false() -> None:
    """total=False 允许节点只返回增量更新，这是 LangGraph 的合并前提。"""
    assert InsightFlowState.__total__ is False


def test_task_payloads_are_plain_json() -> None:
    """四个任务/结果结构必须能被 json.dumps 直接编码。"""
    task: AnalysisTask = {
        "task_id": "task_1",
        "description": "d",
        "target_tables": ["customers"],
        "expected_output": "e",
    }
    visualization: VisualizationTask = {
        "chart_id": "chart_1",
        "description": "d",
        "source_task_id": "task_1",
        "target_tables": ["customers"],
        "chart_type": "bar",
        "expected_output": "e",
    }
    execution: ExecutionResult = {
        "task_id": "task_1",
        "tool_used": "python",
        "code": "result = 1",
        "success": True,
        "result": [{"value": 1}],
        "error": None,
    }
    failure: TaskFailure = {
        "task_id": "task_1",
        "stage": "execution",
        "reason": "boom",
        "code": None,
        "retry_count": 1,
        "refine_count": 0,
    }
    visual_result: VisualizationResult = {
        "chart_id": "chart_1",
        "code": "pass",
        "success": True,
        "path": "outputs/x/chart_1.png",
        "error": None,
    }

    for payload in (task, visualization, execution, failure, visual_result):
        json.dumps(payload)


def test_execution_result_allows_non_serializable_result() -> None:
    """`ExecutionResult.result` 承载任意对象（可能是 DataFrame），
    因此按该字段整体序列化 state 会在 P1 失败。"""
    frame = pd.DataFrame({"amount": [1, 2]})
    payload: ExecutionResult = {
        "task_id": "task_1",
        "tool_used": "python",
        "code": "result = df",
        "success": True,
        "result": frame,
        "error": None,
    }

    with pytest.raises(TypeError):
        json.dumps(payload)


def test_tables_key_is_not_serializable() -> None:
    """`tables` 存的是带 DataFrame 的 TableResource，不能进 Checkpoint。"""
    resource = TableResource(
        name="customers",
        source_type="csv",
        source_path="data/customers.csv",
        dataframe=pd.DataFrame({"customer_id": ["C001"]}),
    )

    with pytest.raises(TypeError):
        json.dumps({"tables": {"customers": resource}})


def test_table_resource_metadata_is_serializable() -> None:
    """去掉 dataframe 后，恢复所需的最小信息是可序列化的 —— 这是 P1 的改造方向。"""
    resource = TableResource(
        name="sales.orders",
        source_type="sqlite",
        source_path="data/sales.sqlite",
        database_table="orders",
    )

    snapshot = {
        "name": resource.name,
        "source_type": resource.source_type,
        "source_path": resource.source_path,
        "database_table": resource.database_table,
    }

    assert json.loads(json.dumps(snapshot)) == snapshot


@pytest.mark.parametrize(
    "module_name",
    ["nodes", "workflow", "conditions", "state"],
)
def test_core_modules_import_without_credentials(module_name: str) -> None:
    """核心模块导入阶段不得要求任何密钥，否则测试无法在无凭证环境运行。"""
    module = importlib.import_module(module_name)
    assert module is not None


def test_nodes_do_not_read_init_tracking_module_scope() -> None:
    """节点函数只应通过 state 取数据，不应依赖模块级可变追踪变量。"""
    import nodes

    source = inspect.getsource(nodes)
    for forbidden in ("_current_run_id", "_current_agent", "global "):
        assert forbidden not in source, f"nodes.py 出现全局可变状态：{forbidden}"
