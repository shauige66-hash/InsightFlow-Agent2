"""沙箱与生成代码之间的"隐含契约"。

真实模型踩到的坑：写 `import pandas as pd`、用 `isinstance`、算时间要 `import time`、
画图例要 `matplotlib.patches`。mock 模型从不写这些，所以只有真实调用才暴露出来。
"""

from __future__ import annotations

import pytest

from sandbox.bootstrap import (
    ALLOWED_IMPORTS,
    SAFE_BUILTINS,
    build_plot_scope,
    build_python_scope,
)


class TestImportAllowlist:
    @pytest.mark.parametrize(
        "name",
        ["time", "datetime", "math", "statistics", "json", "re", "pandas", "numpy"],
    )
    def test_pure_computation_modules_are_importable(self, name: str) -> None:
        """真实模型确实需要这些：算时间、算统计量、序列化中间结果。"""
        assert SAFE_BUILTINS["__import__"](name) is not None
        assert name in ALLOWED_IMPORTS

    @pytest.mark.parametrize(
        "name", ["matplotlib.pyplot", "matplotlib.ticker", "matplotlib.patches"]
    )
    def test_matplotlib_submodules_are_importable(self, name: str) -> None:
        assert SAFE_BUILTINS["__import__"](name) is not None

    @pytest.mark.parametrize("name", ["os", "subprocess", "socket", "pathlib", "shutil", "pickle"])
    def test_unsafe_modules_are_refused_with_guidance(self, name: str) -> None:
        with pytest.raises(ImportError) as error:
            SAFE_BUILTINS["__import__"](name)

        message = str(error.value)
        assert "不允许 import" in message
        assert "可用的模块" in message

    def test_relative_imports_are_refused(self) -> None:
        with pytest.raises(ImportError):
            SAFE_BUILTINS["__import__"]("something", level=1)


class TestSandboxBuiltins:
    @pytest.mark.parametrize(
        "name", ["isinstance", "type", "getattr", "iter", "next", "reversed", "callable", "format"]
    )
    def test_common_safe_builtins_are_available(self, name: str) -> None:
        assert name in SAFE_BUILTINS

    @pytest.mark.parametrize("name", ["open", "eval", "exec", "compile"])
    def test_dangerous_builtins_are_absent(self, name: str) -> None:
        assert name not in SAFE_BUILTINS

    def test_setattr_is_not_exposed(self) -> None:
        """setattr 能改注入对象的状态，不在白名单里。"""
        assert "setattr" not in SAFE_BUILTINS
        assert "delattr" not in SAFE_BUILTINS


class TestScopes:
    def test_the_plot_scope_provides_pandas_numpy_and_ticker(self) -> None:
        scope = build_plot_scope({}, plt=object(), analysis_result={}, output_path="x.png")

        assert "pd" in scope
        assert "np" in scope
        assert "mtick" in scope
        assert scope["output_path"] == "x.png"

    def test_the_python_scope_provides_pandas(self) -> None:
        scope = build_python_scope({})

        assert "pd" in scope
        assert "dfs" in scope


class TestPrint:
    """真实模型的调试习惯：写 print。mock 从不写。"""

    def test_print_writes_to_stderr(self, capsys) -> None:
        SAFE_BUILTINS["print"]("调试信息")

        captured = capsys.readouterr()
        assert "调试信息" in captured.err
        assert captured.out == ""  # stdout 必须留给协议流

    def test_print_ignores_an_explicit_file(self, capsys) -> None:
        """模型可能写 print(..., file=sys.stderr)；不能因此报错。"""
        SAFE_BUILTINS["print"]("again", file=None)

        assert "again" in capsys.readouterr().err
