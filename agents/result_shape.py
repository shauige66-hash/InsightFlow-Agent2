"""把结果的结构描述出来，给绘图/修复提示词用。

动机来自真实模型：模型拿不到结构时会猜键名（`hv_sales_contribution` 这种），
猜错就抛 KeyError，而修复循环只看得到同一个 KeyError，改三次也改不对。
把"实际存在哪些键"写进提示词，是成本最低的修法。
"""

from __future__ import annotations

from typing import Any

MAX_KEYS = 25
MAX_SAMPLE = 3


def _type_name(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "str"
    if isinstance(value, dict):
        return f"dict({len(value)} keys)"
    if isinstance(value, (list, tuple)):
        return f"list({len(value)})"
    return type(value).__name__


def describe_result_shape(execution_result: Any) -> str:
    """返回一段给模型看的结构说明；无法判断时返回空串。"""
    value = execution_result
    if isinstance(execution_result, dict) and "result" in execution_result:
        value = execution_result.get("result")

    if value is None:
        return "analysis_result 为 None：请改用 result_df 或 dfs 中的数据。"

    if isinstance(value, dict) and "__kind__" in value:
        # 大结果在 state 里只留摘要，绘图时由宿主读回完整内容。
        return "分析结果已外置为产物：绘图请使用 result_df 或 dfs，不要按字典键访问。"

    if isinstance(value, dict):
        lines = [
            "analysis_result 是一个字典，**只能使用下面这些顶层键**：",
            "",
        ]
        for index, (key, item) in enumerate(value.items()):
            if index >= MAX_KEYS:
                lines.append(f"- …（共 {len(value)} 个键，其余省略）")
                break
            sample = ""
            if isinstance(item, (list, tuple)) and item:
                sample = f"，首元素是 {_type_name(item[0])}"
            elif isinstance(item, dict) and item:
                sample = f"，键为 {list(item.keys())[:MAX_SAMPLE]}"
            lines.append(f"- `{key}`：{_type_name(item)}{sample}")
        lines.append("")
        lines.append("取不存在的键会抛 KeyError，无法修复；不确定就用 `.get()` 或先判断。")
        return "\n".join(lines)

    if isinstance(value, (list, tuple)):
        head = value[0] if value else None
        lines = [f"analysis_result 是一个列表，长度 {len(value)}，元素类型 {_type_name(head)}。"]
        if isinstance(head, dict) and head:
            lines.append(f"每个元素的键：{list(head.keys())[:MAX_KEYS]}")
        lines.append("请用下标或遍历访问，不要当成字典按键取。")
        return "\n".join(lines)

    return f"analysis_result 是一个标量（{_type_name(value)}）：直接使用，不要按键或下标访问。"
