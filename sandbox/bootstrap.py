"""沙箱执行作用域的构造。

这个模块被**宿主与容器共同导入**，因此同一份逻辑不会出现两个实现：

- 宿主侧（`tools/plot_tool.py`）用它构造 `result_df`，行为与 P1 一致；
- 容器侧（`sandbox/run.py`）用它重建模型代码依赖的全部符号。

它只依赖 pandas，不导入 matplotlib——`plt` 由调用方注入，这样本模块
在宿主上有 matplotlib 与否都能工作。
"""

from __future__ import annotations

from typing import Any

import pandas as pd

# 可用的内置函数白名单。
#
# 刻意不包含 `__import__`、`open`、`eval`、`exec`、`compile`：
# 模型代码不能自行导入模块或直接读写文件，只能通过注入的 `pd` / `plt`
# 与数据集交互。容器本身另有只读根、无网络与资源限制，这里是纵深的一层，
# 不是唯一防线。
#
# 包含异常类：生成代码里 `try/except` 是常见写法（例如"缺列就换个算法"），
# 拿掉异常类会让这类正常代码直接 NameError。异常类不提供任何访问能力，
# 放行它们不会放宽上面的边界。
SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "hasattr": hasattr,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    # 异常类型
    "Exception": Exception,
    "BaseException": BaseException,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "AttributeError": AttributeError,
    "ImportError": ImportError,
    "NameError": NameError,
    "ZeroDivisionError": ZeroDivisionError,
    "ArithmeticError": ArithmeticError,
    "RuntimeError": RuntimeError,
    "StopIteration": StopIteration,
    "NotImplementedError": NotImplementedError,
    "OverflowError": OverflowError,
    "MemoryError": MemoryError,
}


def build_result_dataframe(analysis_result: Any) -> pd.DataFrame | None:
    """从上一步分析结果推导绘图用的 DataFrame。

    保持与 P1 完全一致的判定顺序：DataFrame → list[dict] → 单 DataFrame 值的
    dict → 全标量 dict。返回 None 表示无法推导，此时的语义是"没有可用的
    result_df"，而不是错误。
    """
    if isinstance(analysis_result, pd.DataFrame):
        return analysis_result.copy()

    if isinstance(analysis_result, list):
        if all(isinstance(item, dict) for item in analysis_result):
            return pd.DataFrame(analysis_result)

    if isinstance(analysis_result, dict):

        dataframe_candidates = []

        for value in analysis_result.values():

            if isinstance(value, pd.DataFrame):
                dataframe_candidates.append(value.copy())

            elif isinstance(value, list) and all(
                isinstance(item, dict) for item in value
            ):
                dataframe_candidates.append(pd.DataFrame(value))

        if len(dataframe_candidates) == 1:
            return dataframe_candidates[0]

        if all(
            not isinstance(value, (dict, list, tuple, set, pd.DataFrame))
            for value in analysis_result.values()
        ):
            return pd.DataFrame([analysis_result])

    return None


def build_python_scope(dfs: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """构造 Python 分析的作用域：`dfs`、`pd`，单表时额外 `df`。

    `df` 与 `dfs` 是同一份对象（P1 的 python_tool 就是 `dfs[name]`，不是副本），
    保持这一点可以让 `df is dfs[name]` 这类写法不产生行为差异。
    """
    scope: dict[str, Any] = {
        "dfs": dfs,
        "pd": pd,
        "__builtins__": SAFE_BUILTINS,
    }

    if len(dfs) == 1:
        only_name = next(iter(dfs))
        scope["df"] = dfs[only_name]

    return scope


def build_plot_scope(
    dfs: dict[str, pd.DataFrame],
    *,
    plt: Any,
    analysis_result: Any,
    output_path: str,
) -> dict[str, Any]:
    """构造绘图作用域：`dfs`、`plt`、`analysis_result`、`result_df`、`output_path`。

    绘图代码需要 `output_path` 才能保存图片；该路径由宿主决定并映射到容器内
    的固定位置，模型无法指定。
    """
    scope: dict[str, Any] = {
        "dfs": dfs,
        "plt": plt,
        "analysis_result": analysis_result,
        "result_df": build_result_dataframe(analysis_result),
        "output_path": output_path,
        "__builtins__": SAFE_BUILTINS,
    }

    if len(dfs) == 1:
        only_name = next(iter(dfs))
        scope["df"] = dfs[only_name]

    return scope
