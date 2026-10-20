"""沙箱执行作用域的构造。

这个模块被**宿主与容器共同导入**，因此同一份逻辑不会出现两个实现：

- 宿主侧（`tools/plot_tool.py`）用它构造 `result_df`，行为与 P1 一致；
- 容器侧（`sandbox/run.py`）用它重建模型代码依赖的全部符号。

它只依赖 pandas，不导入 matplotlib——`plt` 由调用方注入，这样本模块
在宿主上有 matplotlib 与否都能工作。
"""

from __future__ import annotations

import builtins
import sys
from typing import Any

import pandas as pd
import numpy as np
import matplotlib.ticker as mtick

# 允许导入的模块。
#
# 一开始的做法是"禁止一切 import"，把 pd / plt 注入命名空间就够了。真实的模型
# 不这么写：算时间要 `import time`、算周期要 `datetime`、画图例要
# `matplotlib.patches`，于是每换一个任务就冒出一个新的 ImportError，修复循环也
# 改不动（它只知道"不允许 import"）。
#
# 改成白名单：只放行纯计算、无文件与网络能力的模块。容器本身仍有只读根、无网络、
# 非 root 与资源上限，这里是纵深的一层，不是唯一防线。
ALLOWED_IMPORTS = frozenset(
    {
        # 纯计算与文本处理
        "math", "statistics", "decimal", "fractions", "random",
        "re", "json", "string", "textwrap", "unicodedata",
        "datetime", "calendar", "time",
        "itertools", "functools", "collections", "operator", "copy", "enum",
        # 数据处理与绘图
        "numpy", "pandas", "matplotlib",
        "matplotlib.pyplot", "matplotlib.ticker", "matplotlib.patches",
        "matplotlib.colors", "matplotlib.dates", "matplotlib.lines",
        "matplotlib.transforms", "matplotlib.figure", "matplotlib.axes",
    }
)

_ALLOWED_ROOTS = frozenset(name.split(".")[0] for name in ALLOWED_IMPORTS)

_REAL_IMPORT = builtins.__import__


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: ANN001
    """只放行白名单模块，其余给出可执行的错误信息。"""
    if level:
        raise ImportError("沙箱不支持相对导入，请使用绝对模块名。")

    allowed = name in ALLOWED_IMPORTS or name.split(".")[0] in _ALLOWED_ROOTS
    if allowed:
        return _REAL_IMPORT(name, globals, locals, fromlist, level)

    raise ImportError(
        f"沙箱不允许 import（{name}）。可用的模块：math/statistics/datetime/time/json/re/"
        "itertools/collections 等标准库，以及 numpy/pandas/matplotlib 及其子模块；"
        "数据已通过 pd/dfs 提供，绘图用 plt。"
    )


def _print_to_stderr(*args, **kwargs):  # noqa: ANN002, ANN003
    """`print` 写 stderr，不碰 stdout。

    stdout 是宿主解析结果用的协议流（最后一行是 JSON）。模型用 print 调试很常见，
    直接不给它会得到 NameError；给一个重定向到 stderr 的版本，既保住协议，
    也让调试输出能被 executor 捕获下来。
    """
    kwargs.pop("file", None)
    print(*args, file=sys.stderr, **kwargs)

# 包含异常类：生成代码里 `try/except` 是常见写法（例如"缺列就换个算法"），
# 拿掉异常类会让这类正常代码直接 NameError。异常类不提供任何访问能力，
# 放行它们不会放宽上面的边界。
SAFE_BUILTINS: dict[str, Any] = {
    # 白名单式导入：放行纯计算与绘图模块，其余拒绝并说明可用的替代
    "__import__": _guarded_import,
    # print 重定向到 stderr：stdout 是协议流，但调试输出不该让代码直接失败
    "print": _print_to_stderr,
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
    # 类型判断与迭代：绘图/分析代码里极其常见，且不提供任何访问能力。
    # 真实模型第一次跑就在这里挂了（NameError: isinstance），mock 从没触发过。
    "isinstance": isinstance,
    "issubclass": issubclass,
    "type": type,
    "getattr": getattr,
    "iter": iter,
    "next": next,
    "reversed": reversed,
    "slice": slice,
    "frozenset": frozenset,
    "callable": callable,
    "hash": hash,
    "repr": repr,
    "format": format,
    "divmod": divmod,
    "pow": pow,
    "ord": ord,
    "chr": chr,
    "map": map,
    "filter": filter,
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
        "pd": pd,
        "np": np,
        "mtick": mtick,
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
