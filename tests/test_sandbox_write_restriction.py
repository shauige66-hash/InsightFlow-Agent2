"""P2-A 补验收：容器写入权限边界。

背景（实测复现过的真实漏洞）：整份 Run 目录原先以可写模式挂载为 `/out`，
于是模型代码可以这样篡改同一 Run 的历史产物：

    df.to_csv('/out/analysis_report.md')   # 覆盖已有报告
    df.to_csv('/out/evil_new.csv')         # 凭空新建文件

作用域里没有 `open` / `__import__` **不能**阻止这件事——pandas 与 matplotlib
自己的方法就会写文件。因此边界必须落在**挂载权限**上：容器只写 staging，
宿主校验后再发布到稳定的 Run 路径。

这些用例用无害临时文件与无意义内容，不触碰任何真实敏感数据。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sandbox.executor import DockerSandboxExecutor
from sandbox.protocol import SandboxJob, SandboxLimits

pytestmark = pytest.mark.sandbox

HISTORICAL_REPORT = "HISTORICAL REPORT CONTENT\n"


@pytest.fixture(scope="module", autouse=True)
def require_docker() -> None:
    executor = DockerSandboxExecutor()
    if not executor.available():
        pytest.skip("Docker daemon 不可用")
    if not executor.image_present():
        pytest.skip("沙箱镜像未构建：先运行 sandbox/build_image.py")


@pytest.fixture
def guarded_run(tmp_path: Path):
    """构造一个已存在历史产物的 Run 目录。"""
    run_dir = tmp_path / "guarded-run"
    run_dir.mkdir(parents=True, exist_ok=True)

    report = run_dir / "analysis_report.md"
    report.write_text(HISTORICAL_REPORT, encoding="utf-8")

    chart = run_dir / "chart_0.png"
    chart.write_bytes(b"\x89PNG\r\n\x1a\nHISTORICAL-CHART-BYTES")

    return run_dir, report, chart


def tables():
    import pandas as pd

    from ingestion.models import TableResource

    return {
        "t": TableResource(
            name="t",
            source_type="csv",
            source_path="/nonexistent/t.csv",
            dataframe=pd.DataFrame({"a": [1, 2, 3]}),
        )
    }


def run_code(code: str, *, run_dir: Path, filename: str | None = None, mode: str = "python"):
    executor = DockerSandboxExecutor()
    job = SandboxJob(
        mode=mode,
        table_names=["t"],
        code=code,
        output_filename=filename,
        limits=SandboxLimits(timeout_seconds=60),
        op_label="write-guard",
    )
    return executor.run(job, run_id="guarded", run_dir=run_dir, tables=tables())


class TestHistoricalArtifactsAreProtected:
    """容器不能修改同一 Run 已发布的产物。"""

    def test_cannot_overwrite_existing_report(self, guarded_run) -> None:
        run_dir, report, _ = guarded_run

        run_code("df.to_csv('/out/analysis_report.md', index=False)\nresult = 'attempted'",
                 run_dir=run_dir)

        # 历史报告必须逐字节不变
        assert report.read_text(encoding="utf-8") == HISTORICAL_REPORT

    def test_cannot_overwrite_existing_chart(self, guarded_run) -> None:
        run_dir, _, chart = guarded_run
        original = chart.read_bytes()

        run_code(
            "df.to_csv('/out/chart_0.png', index=False)\nresult = 'attempted'",
            run_dir=run_dir,
        )

        assert chart.read_bytes() == original

    def test_cannot_create_arbitrary_new_file(self, guarded_run) -> None:
        run_dir, _, _ = guarded_run

        run_code("df.to_csv('/out/evil_new.csv', index=False)\nresult = 'attempted'",
                 run_dir=run_dir)

        assert not (run_dir / "evil_new.csv").exists()

    def test_failed_plot_does_not_publish_a_chart(self, guarded_run) -> None:
        """绘图失败时不得把任何东西发布到 Run 路径。"""
        run_dir, _, _ = guarded_run

        outcome = run_code(
            "df.to_csv('/out/chart_9.png', index=False)\nresult = 'x'",
            run_dir=run_dir,
            filename="chart_9.png",
            mode="plot",
        )

        assert outcome.success is False
        assert not (run_dir / "chart_9.png").exists()


class TestOnlyStagingIsWritable:
    def test_writes_land_in_staging_not_in_run_dir(self, guarded_run) -> None:
        """容器的写入确实发生了，但落在 staging（宿主会丢弃）。"""
        run_dir, _, _ = guarded_run

        run_code("df.to_csv('/out/probe_artifact.csv', index=False)\nresult = 'ok'",
                 run_dir=run_dir)

        staged = list((run_dir / "sandbox").rglob("staging/probe_artifact.csv"))
        assert staged, "预期写入落在 staging 目录"
        assert not (run_dir / "probe_artifact.csv").exists()

    def test_history_mount_is_read_only(self, guarded_run) -> None:
        """容器可以读 /history，但改不了。"""
        run_dir, _, _ = guarded_run

        outcome = run_code(
            "df.to_csv('/history/tamper.csv', index=False)\nresult = 'attempted'",
            run_dir=run_dir,
        )

        assert outcome.success is False
        assert not (run_dir / "tamper.csv").exists()

    def test_traversal_out_of_out_is_blocked(self, guarded_run) -> None:
        run_dir, report, _ = guarded_run

        outcome = run_code(
            "df.to_csv('/out/../escaped.csv', index=False)\nresult = 'attempted'",
            run_dir=run_dir,
        )

        assert outcome.success is False
        assert not (run_dir.parent / "escaped.csv").exists()
        assert report.read_text(encoding="utf-8") == HISTORICAL_REPORT


class TestSymlinkHandling:
    """符号链接的处理。

    **本机的能力限制（如实记录）**：
      - 宿主：Windows 未开启开发者模式，`os.symlink` 报 WinError 1314（缺特权）；
      - 容器内：模型代码作用域不含 `os`，无法自行创建链接。
    因此这里**无法**构造真实符号链接来做端到端验证。改用两条：
      1. 直接验证发布校验对"看起来像链接"的路径会拒绝（打桩 `is_symlink`）；
      2. 用可创建的真实**硬链接**验证发布逻辑不会被链接形式绕过。
    真实符号链接的验证留待具备相应权限的环境执行。
    """

    def test_publish_rejects_symlinked_artifact(self, tmp_path: Path, monkeypatch) -> None:
        """发布校验必须先判链接，再读内容。"""
        import struct

        from sandbox import executor as executor_module

        staging = tmp_path / "staging"
        staging.mkdir()
        (staging / "chart_1.png").write_bytes(
            b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + struct.pack(">II", 10, 10)
        )

        # 模拟"该路径是符号链接"：真实链接依赖本机特权，无法在此构造
        real_is_symlink = Path.is_symlink

        def fake_is_symlink(self: Path) -> bool:
            if self.name == "chart_1.png":
                return True
            return real_is_symlink(self)

        monkeypatch.setattr(Path, "is_symlink", fake_is_symlink)

        executor = executor_module.DockerSandboxExecutor()
        with pytest.raises(executor_module.SandboxOutputRejected) as captured:
            executor._publish_output(
                staging_dir=staging,
                run_dir=tmp_path,
                filename="chart_1.png",
                limits=SandboxLimits(),
            )

        assert "symlink" in str(captured.value).lower()
        assert not (tmp_path / "chart_1.png").exists()

    def test_hardlinked_artifact_is_still_validated(self, tmp_path: Path) -> None:
        """硬链接不是符号链接：仍按普通文件走完整校验，不会绕过类型检查。"""
        import os

        staging = tmp_path / "staging"
        staging.mkdir()
        target = staging / "backing.bin"
        target.write_bytes(b"not a png")

        link = staging / "chart_1.png"
        try:
            os.link(target, link)
        except OSError:
            pytest.skip("本机不允许创建硬链接")

        executor = DockerSandboxExecutor()
        with pytest.raises(Exception) as captured:
            executor._publish_output(
                staging_dir=staging,
                run_dir=tmp_path,
                filename="chart_1.png",
                limits=SandboxLimits(),
            )

        # 内容是 "not a png"，应在类型校验被拒，而不是被当成链接放行或拒绝
        assert "PNG" in str(captured.value)

    def test_container_cannot_reach_outside_paths_via_history_mount(
        self, guarded_run
    ) -> None:
        """即使试图借 /history 的相对路径往外走，也读不到挂载之外。"""
        run_dir, _, _ = guarded_run
        outside = run_dir.parent / "outside-secret.txt"
        outside.write_text("OUTSIDE SECRET\n", encoding="utf-8")

        outcome = run_code(
            "df.to_csv('/history/../outside-secret.txt', index=False)\nresult = 'attempted'",
            run_dir=run_dir,
        )

        assert outcome.success is False
        assert outside.read_text(encoding="utf-8") == "OUTSIDE SECRET\n"


class TestHostPublishValidation:
    """宿主发布前的校验：路径、大小、文件类型、合法性。"""

    def test_non_png_content_is_rejected(self, tmp_path: Path) -> None:
        staging = tmp_path / "staging"
        staging.mkdir()
        (staging / "fake.png").write_bytes(b"this is not a png at all")

        executor = DockerSandboxExecutor()
        with pytest.raises(Exception) as captured:
            executor._publish_output(
                staging_dir=staging,
                run_dir=tmp_path,
                filename="fake.png",
                limits=SandboxLimits(),
            )

        assert "PNG" in str(captured.value)
        assert not (tmp_path / "fake.png").exists()

    def test_empty_file_is_rejected(self, tmp_path: Path) -> None:
        staging = tmp_path / "staging"
        staging.mkdir()
        (staging / "empty.png").write_bytes(b"")

        executor = DockerSandboxExecutor()
        with pytest.raises(Exception) as captured:
            executor._publish_output(
                staging_dir=staging,
                run_dir=tmp_path,
                filename="empty.png",
                limits=SandboxLimits(),
            )

        assert "empty" in str(captured.value).lower()

    def test_oversized_file_is_rejected(self, tmp_path: Path) -> None:
        import struct

        staging = tmp_path / "staging"
        staging.mkdir()
        payload = (
            b"\x89PNG\r\n\x1a\n"
            + b"\x00\x00\x00\rIHDR"
            + struct.pack(">II", 10, 10)
            + b"\x00" * 8192
        )
        (staging / "big.png").write_bytes(payload)

        executor = DockerSandboxExecutor()
        with pytest.raises(Exception) as captured:
            executor._publish_output(
                staging_dir=staging,
                run_dir=tmp_path,
                filename="big.png",
                limits=SandboxLimits(max_output_file_bytes=1024),
            )

        assert "too large" in str(captured.value).lower()

    def test_missing_staged_file_is_rejected(self, tmp_path: Path) -> None:
        staging = tmp_path / "staging"
        staging.mkdir()

        executor = DockerSandboxExecutor()
        with pytest.raises(Exception) as captured:
            executor._publish_output(
                staging_dir=staging,
                run_dir=tmp_path,
                filename="absent.png",
                limits=SandboxLimits(),
            )

        assert "missing" in str(captured.value).lower()

    def test_valid_png_is_published_and_removed_from_staging(self, tmp_path: Path) -> None:
        import struct

        staging = tmp_path / "staging"
        staging.mkdir()
        (staging / "good.png").write_bytes(
            b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + struct.pack(">II", 20, 30)
        )

        executor = DockerSandboxExecutor()
        published = executor._publish_output(
            staging_dir=staging,
            run_dir=tmp_path,
            filename="good.png",
            limits=SandboxLimits(),
        )

        assert published == tmp_path / "good.png"
        assert published.exists()
        assert not (staging / "good.png").exists(), "发布后应从 staging 移走"

    def test_publish_replaces_own_previous_artifact(self, tmp_path: Path) -> None:
        """同一 Run 重放时覆盖自己的产物（幂等要求）。"""
        import struct

        staging = tmp_path / "staging"
        staging.mkdir()
        png = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + struct.pack(">II", 5, 5)

        (tmp_path / "chart_1.png").write_bytes(b"OLD")
        (staging / "chart_1.png").write_bytes(png)

        executor = DockerSandboxExecutor()
        published = executor._publish_output(
            staging_dir=staging,
            run_dir=tmp_path,
            filename="chart_1.png",
            limits=SandboxLimits(),
        )

        assert published.read_bytes() == png
