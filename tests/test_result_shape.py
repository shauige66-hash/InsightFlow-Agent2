"""结果结构描述：让绘图模型不用猜键名。

真实模型在 `analysis_result` 上猜了一个不存在的键（`hv_sales_contribution`），
KeyError 之后修复循环也无从下手——因为它看到的只有"键不存在"。
"""

from __future__ import annotations

from agents.result_shape import describe_result_shape


def test_a_dict_result_lists_its_actual_keys() -> None:
    # 传入的是 ExecutionResult 的形状：结构描述会先解包 result 字段
    shape = describe_result_shape(
        {"task_id": "task_1", "success": True, "result": {"ratio": 0.61, "regions": ["East"]}}
    )

    assert "只能使用下面这些顶层键" in shape
    assert "`ratio`" in shape
    assert "`regions`" in shape
    assert "KeyError" in shape


def test_a_list_result_describes_its_elements() -> None:
    shape = describe_result_shape({"result": [{"region": "East", "ratio": 0.61}]})

    assert "列表，长度 1" in shape
    assert "region" in shape
    assert "不要当成字典按键取" in shape


def test_a_scalar_result_says_not_to_index_it() -> None:
    shape = describe_result_shape({"result": 42})

    assert "标量" in shape
    assert "不要按键或下标访问" in shape


def test_a_none_result_points_at_the_dataframes() -> None:
    shape = describe_result_shape({"result": None})

    assert "result_df" in shape


def test_an_externalised_result_is_not_described_as_a_dict() -> None:
    """大结果在 state 里只留摘要；把它当字典描述会误导模型。"""
    shape = describe_result_shape({"result": {"__kind__": "externalized", "sha256": "x"}})

    assert "已外置" in shape
    assert "dfs" in shape
    assert "sha256" not in shape
