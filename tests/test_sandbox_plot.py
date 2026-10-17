"""绘图在沙箱中的真实行为。

重点是"真的画出来了"和"中文没有变成方框"，而不是断言参数串或文件存在性。
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from sandbox.executor import DockerSandboxExecutor
from sandbox.protocol import SandboxJob, SandboxLimits

pytestmark = pytest.mark.sandbox

# 含标题、横纵轴与图例的中文绘图代码；覆盖三种常见的文字位置。
CJK_PLOT_CODE = (
    "fig = plt.figure(figsize=(6, 4))\n"
    "plt.bar(['华东', '华北', '华南'], [12, 7, 9], label='销售额')\n"
    "plt.title('各地区销售额对比')\n"
    "plt.xlabel('地区')\n"
    "plt.ylabel('销售额（万元）')\n"
    "plt.legend()\n"
    "plt.tight_layout()\n"
    "plt.savefig(output_path, bbox_inches='tight')\n"
    "plt.close()\n"
    "result = output_path\n"
)


@pytest.fixture(scope="module", autouse=True)
def require_docker() -> None:
    executor = DockerSandboxExecutor()
    if not executor.available():
        pytest.skip("Docker daemon 不可用")
    if not executor.image_present():
        pytest.skip("沙箱镜像未构建：先运行 sandbox/build_image.py")


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "plot-run"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def plot_tables():
    import pandas as pd

    from ingestion.models import TableResource

    return {
        "sales": TableResource(
            name="sales",
            source_type="csv",
            source_path="/nonexistent/sales.csv",
            dataframe=pd.DataFrame({"region": ["华东", "华北"], "amount": [12, 7]}),
        )
    }


def run_plot(code: str, *, run_dir: Path, filename: str = "chart_1.png"):
    executor = DockerSandboxExecutor()
    job = SandboxJob(
        mode="plot",
        table_names=["sales"],
        code=code,
        analysis_result=[{"region": "华东", "amount": 12}, {"region": "华北", "amount": 7}],
        output_filename=filename,
        limits=SandboxLimits(timeout_seconds=120),
        op_label="plot-test",
    )
    return executor.run(
        job, run_id="plot-isolation", run_dir=run_dir, tables=plot_tables()
    )


def png_dimensions(path: Path) -> tuple[int, int]:
    """从 PNG 头读取宽高，确认产物是真实图片而不是空文件。"""
    header = path.read_bytes()[:24]
    assert header[:8] == b"\x89PNG\r\n\x1a\n", "不是合法 PNG"
    width, height = struct.unpack(">II", header[16:24])
    return width, height


class TestRealPngGeneration:
    def test_png_is_created_and_valid(self, run_dir: Path) -> None:
        outcome = run_plot(CJK_PLOT_CODE, run_dir=run_dir)

        assert outcome.success is True, outcome.error

        produced = run_dir / "chart_1.png"
        assert produced.exists(), "容器未把 PNG 写回宿主输出目录"

        width, height = png_dimensions(produced)
        assert width > 100 and height > 100, f"图片尺寸异常: {width}x{height}"
        # 真实绘图不会只有几百字节
        assert produced.stat().st_size > 5000, (
            f"PNG 过小，疑似空白: {produced.stat().st_size} bytes"
        )

    def test_cjk_text_renders_without_missing_glyphs(self, run_dir: Path) -> None:
        """中文标题/坐标轴/图例必须真正渲染，而不是方框。

        matplotlib 在缺字形时会发 "Glyph ... missing from font" 警告，
        容器把计数带出来作为判据——只看文件存在无法区分"画了方框"。
        """
        outcome = run_plot(CJK_PLOT_CODE, run_dir=run_dir, filename="cjk.png")

        assert outcome.success is True, outcome.error
        assert outcome.diagnostics.get("missing_glyph_count") == 0, (
            f"存在缺字形，中文渲染为方框: {outcome.diagnostics}"
        )

    def test_output_is_inside_the_run_directory(self, run_dir: Path) -> None:
        """产物必须落在本次 Run 的目录，而不是进程当前目录。"""
        outcome = run_plot(CJK_PLOT_CODE, run_dir=run_dir, filename="scoped.png")

        assert outcome.success is True, outcome.error
        assert (run_dir / "scoped.png").exists()

    def test_plot_code_without_savefig_is_rejected(self, run_dir: Path) -> None:
        """没有产出图片时必须失败，而不是返回一个不存在的路径。"""
        outcome = run_plot("result = output_path", run_dir=run_dir, filename="missing.png")

        assert outcome.success is False
        assert "expected image file" in outcome.error

    def test_plot_result_df_is_available(self, run_dir: Path) -> None:
        """result_df 由 analysis_result 推导，保持 P1 的语义。"""
        code = (
            "rows = 0 if result_df is None else len(result_df)\n"
            "fig = plt.figure()\n"
            "plt.plot([1, 2, 3], [1, 4, 9])\n"
            "plt.title('数值')\n"
            "plt.savefig(output_path)\n"
            "plt.close()\n"
            "result = rows\n"
        )

        outcome = run_plot(code, run_dir=run_dir, filename="result_df.png")

        assert outcome.success is True, outcome.error
        # 绘图模式下 result 约定为图片路径（与  一致），
        # 模型写进 result 的值经 diagnostics 带出
        assert outcome.result.endswith("result_df.png")
        assert outcome.diagnostics.get("model_result") == 2

    def test_invalid_chart_filename_is_rejected(self, run_dir: Path) -> None:
        """输出文件名由宿主决定，不能包含路径分隔符。"""
        outcome = run_plot(
            CJK_PLOT_CODE, run_dir=run_dir, filename="../escape.png"
        )

        assert outcome.success is False
        assert "filename" in outcome.error.lower()

    def test_two_runs_do_not_share_plot_output(self, tmp_path: Path) -> None:
        """两个 Run 的绘图产物互相隔离。"""
        dir_a = tmp_path / "run-a"
        dir_b = tmp_path / "run-b"
        dir_a.mkdir()
        dir_b.mkdir()

        first = run_plot(CJK_PLOT_CODE, run_dir=dir_a, filename="shared.png")
        second = run_plot(CJK_PLOT_CODE, run_dir=dir_b, filename="shared.png")

        assert first.success and second.success

        a_png = dir_a / "shared.png"
        b_png = dir_b / "shared.png"

        assert a_png.exists() and b_png.exists()
        # 两个 Run 各自持有自己的文件，不是同一路径
        assert a_png.resolve() != b_png.resolve()
