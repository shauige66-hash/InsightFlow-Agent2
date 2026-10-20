"""沙箱与绘图代码之间的"隐含契约"。

真实模型第一次跑绘图时踩到的两个坑：写 `import pandas as pd`、用 `isinstance`。
mock 模型从不写这些，所以只有真实调用才暴露出来。
"""

from __future__ import annotations

import pytest

from sandbox.bootstrap import SAFE_BUILTINS, build_plot_scope, build_python_scope
from tools.plot_tool import strip_provided_imports


class TestImportStripping:
    def test_canonical_imports_are_removed(self) -> None:
        code = (
            "import pandas as pd\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "fig = plt.figure()\n"
        )

        stripped = strip_provided_imports(code)

        assert "import" not in stripped
        assert "fig = plt.figure()" in stripped

    def test_unrelated_imports_are_left_alone(self) -> None:
        """只剥"已注入模块"的等价写法；真正的动态导入仍然会被沙箱拒绝。"""
        code = "import json\nfrom matplotlib.ticker import FuncFormatter\nx = 1\n"

        stripped = strip_provided_imports(code)

        assert "import json" in stripped
        assert "from matplotlib.ticker import FuncFormatter" in stripped

    def test_indented_imports_are_removed_too(self) -> None:
        code = "def draw():\n    import matplotlib.pyplot as plt\n    return plt.figure()\n"

        stripped = strip_provided_imports(code)

        assert "import matplotlib" not in stripped
        assert "return plt.figure()" in stripped


class TestSandboxBuiltins:
    @pytest.mark.parametrize(
        "name", ["isinstance", "type", "getattr", "iter", "next", "reversed", "callable", "format"]
    )
    def test_common_safe_builtins_are_available(self, name: str) -> None:
        assert name in SAFE_BUILTINS

    @pytest.mark.parametrize("name", ["open", "eval", "exec", "compile"])
    def test_dangerous_builtins_are_absent(self, name: str) -> None:
        assert name not in SAFE_BUILTINS

    def test_import_fails_with_an_actionable_message(self) -> None:
        blocked = SAFE_BUILTINS["__import__"]

        with pytest.raises(ImportError) as error:
            blocked("matplotlib")

        message = str(error.value)
        assert "不允许 import" in message
        assert "plt" in message  # 告诉模型该用什么

    def test_setattr_is_not_exposed(self) -> None:
        """setattr 能改注入对象的状态，不在白名单里。"""
        assert "setattr" not in SAFE_BUILTINS
        assert "delattr" not in SAFE_BUILTINS


class TestScopes:
    def test_the_plot_scope_provides_pandas_and_numpy(self) -> None:
        scope = build_plot_scope({}, plt=object(), analysis_result={}, output_path="x.png")

        assert "pd" in scope
        assert "np" in scope
        assert scope["output_path"] == "x.png"

    def test_the_python_scope_provides_pandas(self) -> None:
        scope = build_python_scope({})

        assert "pd" in scope
        assert "dfs" in scope
