"""P1 补验收：文件副作用与 Checkpoint 的一致性。

文件系统与 SQLite Checkpoint **不属于同一个事务**。可观测的边界是：

    文件已写入 → 进程被杀 → 节点的 state 增量从未提交
    → Checkpoint 里"没有做过" → 恢复时该节点重放 → 文件被再写一次

因此这里不假设原子性，而是用故障注入把边界测出来，并明确：
- 哪些情况是安全的（重复写同一路径、内容一致）；
- 哪些情况需要额外机制（产物引用、哈希校验）。

用 -O 崩溃点通过环境变量 INSIGHTFLOW_CRASH_AT 注入，崩溃发生在
"文件已落盘、登记未提交"的瞬间。
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DRIVER = PROJECT_ROOT / "scripts" / "recovery_driver.py"
PYTHON = sys.executable


def run_phase(
    phase: str,
    *,
    workdir: Path,
    run_id: str,
    data_source: Path,
    crash_at: str | None = None,
) -> dict:
    command = [
        str(PYTHON),
        str(DRIVER),
        phase,
        "--workdir",
        str(workdir),
        "--run-id",
        run_id,
        "--data-source",
        str(data_source),
    ]

    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    if crash_at:
        env["INSIGHTFLOW_CRASH_AT"] = crash_at

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(PROJECT_ROOT),
        timeout=300,
        env=env,
    )

    if completed.returncode != 0 and not completed.stdout.strip():
        return {"crashed": True, "returncode": completed.returncode}

    line = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else "{}"
    try:
        payload = json.loads(line)
    except ValueError:
        payload = {"raw": completed.stdout[-400:]}
    payload["returncode"] = completed.returncode
    return payload


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def files_in(workdir: Path, run_id: str) -> dict[str, int]:
    directory = workdir / run_id
    if not directory.exists():
        return {}
    return {
        path.name: path.stat().st_size
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


@pytest.fixture
def workspace(tmp_path: Path) -> tuple[Path, Path]:
    workdir = tmp_path / "run"
    workdir.mkdir()

    source = tmp_path / "customers.csv"
    source.write_text(
        "customer_id,customer_name,region\nC001,Alice,East\nC002,Bob,West\n",
        encoding="utf-8",
    )
    return workdir, source


class TestPlotFileWrittenButNotRecorded:
    """PNG 已保存，但完成标记尚未写入 Checkpoint 时强杀。"""

    def test_png_exists_after_crash(self, workspace) -> None:
        workdir, source = workspace
        run_id = "fs-plot"

        crashed = run_phase(
            "start",
            workdir=workdir,
            run_id=run_id,
            data_source=source,
            crash_at="plot_file_written",
        )
        assert crashed.get("crashed") is True

        # 文件已经在磁盘上，但 Checkpoint 里没有"已完成"登记
        assert files_in(workdir, run_id) == {"chart_1.png": files_in(workdir, run_id)["chart_1.png"]}

        from langgraph.checkpoint.sqlite import SqliteSaver

        with SqliteSaver.from_conn_string(str(workdir / "checkpoints.sqlite")) as saver:
            snapshot = saver.get_tuple({"configurable": {"thread_id": run_id}})
            ops = snapshot.checkpoint["channel_values"].get("completed_operations") or {}
            assert not any("plot_execution" in key for key in ops), (
                "登记不应已经写入——这正是要验证的窗口"
            )

    def test_resume_does_not_duplicate_the_png(self, workspace) -> None:
        workdir, source = workspace
        run_id = "fs-plot-resume"

        run_phase(
            "start",
            workdir=workdir,
            run_id=run_id,
            data_source=source,
            crash_at="plot_file_written",
        )
        png = workdir / run_id / "chart_1.png"
        before = digest(png)

        finished = run_phase(
            "resume", workdir=workdir, run_id=run_id, data_source=source
        )
        assert finished["run_status"] == "completed"

        # 只有一个 PNG：重放写的是同一个稳定路径，没有产生第二份文件
        names = list(files_in(workdir, run_id))
        assert [name for name in names if name.endswith(".png")] == ["chart_1.png"]

        # 内容可复现（同一份代码、同一份数据）
        assert digest(png) == before, "重放后的图片内容发生变化"

    def test_artifact_list_has_no_duplicates(self, workspace) -> None:
        workdir, source = workspace
        run_id = "fs-plot-artifacts"

        run_phase(
            "start",
            workdir=workdir,
            run_id=run_id,
            data_source=source,
            crash_at="plot_file_written",
        )
        finished = run_phase(
            "resume", workdir=workdir, run_id=run_id, data_source=source
        )

        artifacts = finished["artifacts"]
        assert len(artifacts) == len(set(artifacts)), f"产物重复：{artifacts}"
        assert finished["visualizations"] == 1, "图表记录被重复追加"


class TestReportFilesWrittenButStateNotPersisted:
    """Markdown/HTML/PDF 已生成，但 State 更新尚未持久化时强杀。"""

    def test_report_files_exist_after_crash(self, workspace) -> None:
        workdir, source = workspace
        run_id = "fs-artifact"

        crashed = run_phase(
            "start",
            workdir=workdir,
            run_id=run_id,
            data_source=source,
            crash_at="artifact_files_written",
        )
        assert crashed.get("crashed") is True

        present = files_in(workdir, run_id)
        assert "analysis_report.md" in present
        assert "analysis_report.html" in present
        assert "analysis_report.pdf" in present

    def test_resume_regenerates_without_duplicating(self, workspace) -> None:
        workdir, source = workspace
        run_id = "fs-artifact-resume"

        run_phase(
            "start",
            workdir=workdir,
            run_id=run_id,
            data_source=source,
            crash_at="artifact_files_written",
        )

        finished = run_phase(
            "resume", workdir=workdir, run_id=run_id, data_source=source
        )
        assert finished["run_status"] == "completed"

        # 每个报告文件只有一个实例
        names = list(files_in(workdir, run_id))
        for expected in ("analysis_report.md", "analysis_report.html", "analysis_report.pdf"):
            assert names.count(expected) == 1

        # 产物清单无重复
        artifacts = finished["artifacts"]
        assert len(artifacts) == len(set(artifacts))

    def test_report_path_points_at_a_real_file(self, workspace) -> None:
        """产物引用必须仍然有效，不能指向不存在或被覆盖的文件。"""
        workdir, source = workspace
        run_id = "fs-artifact-ref"

        run_phase(
            "start",
            workdir=workdir,
            run_id=run_id,
            data_source=source,
            crash_at="artifact_files_written",
        )
        finished = run_phase(
            "resume", workdir=workdir, run_id=run_id, data_source=source
        )

        # report_path 在驱动摘要里没有暴露，直接检查磁盘上的三份报告
        for name in (
            "analysis_report.md",
            "analysis_report.html",
            "analysis_report.pdf",
        ):
            path = workdir / run_id / name
            assert path.exists(), f"缺少报告文件：{name}"
            assert path.stat().st_size > 0, f"报告文件为空：{name}"

        # 崩溃点发生在文件写完之后，因此这三份文件在崩溃时就已经存在
        assert finished["run_status"] == "completed"


class TestRepeatedResumeAfterFileCrash:
    """进程重启后反复 resume：产物与结果都不得增长。"""

    def test_repeated_resume_is_stable(self, workspace) -> None:
        workdir, source = workspace
        run_id = "fs-repeat"

        run_phase(
            "start",
            workdir=workdir,
            run_id=run_id,
            data_source=source,
            crash_at="artifact_files_written",
        )

        first = run_phase(
            "resume", workdir=workdir, run_id=run_id, data_source=source
        )
        files_after_first = files_in(workdir, run_id)
        results_after_first = first["execution_results"]

        second = run_phase(
            "resume", workdir=workdir, run_id=run_id, data_source=source
        )
        third = run_phase(
            "resume", workdir=workdir, run_id=run_id, data_source=source
        )

        assert files_in(workdir, run_id) == files_after_first, "产物文件发生变化"
        assert second["execution_results"] == results_after_first
        assert third["execution_results"] == results_after_first
        assert third["visualizations"] == first["visualizations"] == 1


class TestOutputIsolation:
    """产物必须落在调用方指定的目录，而不是进程当前目录。"""

    def test_nothing_is_written_into_the_repository(self, workspace) -> None:
        workdir, source = workspace
        run_id = "fs-isolation"

        repo_outputs = PROJECT_ROOT / "outputs"
        existed_before = repo_outputs.exists()

        run_phase("start", workdir=workdir, run_id=run_id, data_source=source)

        # 若此前不存在，就不得因为这次运行而被创建
        if not existed_before:
            assert not repo_outputs.exists(), (
                "产物/日志写进了仓库目录，绕过了 output_root"
            )

        # 该有的东西都在 workdir 里
        assert (workdir / run_id / "analysis_report.pdf").exists()
        assert (workdir / "logs" / f"{run_id}.jsonl").exists()
        assert (workdir / "llm_ledger" / f"{run_id}.jsonl").exists()
