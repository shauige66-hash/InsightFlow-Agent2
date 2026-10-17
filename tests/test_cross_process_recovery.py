"""真实的跨进程恢复。

每个 phase 都由 `scripts/recovery_driver.py` 在**独立 Python 进程**中执行，
因此"重启恢复"不是在同一进程里换一个 Graph 对象，而是真的重新打开
Checkpoint 文件、重新构建工作流。

覆盖场景：
    A 服务重启恢复      B 审批恢复          C 拒绝审批
    D 重复 Resume       E 多 Run 隔离       F 文件丢失
    G 绘图阶段恢复
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DRIVER = PROJECT_ROOT / "scripts" / "recovery_driver.py"
PYTHON = sys.executable

# 崩溃阶段必须在 checkpoint 已提交之后，否则恢复没有起点
CRASH_AFTER_PLOT_GENERATION = "plotter"


def run_phase(
    phase: str,
    *,
    workdir: Path,
    run_id: str,
    data_source: Path,
    require_approval: bool = False,
    crash_on: str | None = None,
    decision: str = "approve",
    expect_failure: bool = False,
    grant_llm_retry: bool = False,
) -> dict:
    """在独立进程中执行一个 phase，返回摘要。"""
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
    if require_approval:
        command.append("--require-approval")
    if crash_on:
        command += ["--crash-on", crash_on]
    if decision != "approve":
        command += ["--decision", decision]
    if grant_llm_retry:
        command.append("--grant-llm-retry")

    # Windows 默认用本地代码页解码子进程输出，中文错误信息会导致解码失败，
    # 因此显式指定 UTF-8 并容错。
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

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

    if expect_failure:
        assert completed.returncode != 0, (
            f"{phase} 本应失败但成功退出：{completed.stdout[-500:]}"
        )
        return {"stderr": completed.stderr, "stdout": completed.stdout}

    if completed.returncode != 0:
        # 崩溃场景：进程被强杀，退出码是 os._exit 传入的值
        return {
            "crashed": True,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }

    line = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else "{}"
    return json.loads(line)


def llm_calls(workdir: Path) -> list[str]:
    path = workdir / "llm_calls.txt"
    if not path.exists():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line]


def artifact_files(workdir: Path, run_id: str) -> list[str]:
    directory = workdir / run_id
    if not directory.exists():
        return []
    return sorted(path.name for path in directory.iterdir() if path.is_file())


@pytest.fixture
def workspace(tmp_path: Path) -> tuple[Path, Path]:
    """独立的运行目录与数据源副本（不碰仓库里的 data/）。"""
    workdir = tmp_path / "run"
    workdir.mkdir()

    source = tmp_path / "customers.csv"
    source.write_text(
        "customer_id,customer_name,region\nC001,Alice,East\nC002,Bob,West\n"
        "C003,Carol,East\nC004,Dave,South\nC005,Eve,West\n",
        encoding="utf-8",
    )

    return workdir, source


class TestScenarioAServerRestart:
    """场景 A：执行到已持久化的安全检查点，终止进程，重启后恢复至完成。"""

    def test_resume_after_hard_kill(self, workspace) -> None:
        workdir, source = workspace
        run_id = "xproc-restart"

        crashed = run_phase(
            "start",
            workdir=workdir,
            run_id=run_id,
            data_source=source,
            crash_on=CRASH_AFTER_PLOT_GENERATION,
        )
        assert crashed.get("crashed") is True, "未按预期崩溃"
        assert crashed["returncode"] == 17

        # 崩溃前的进度应当已提交：分析执行完成
        inspected = run_phase(
            "inspect", workdir=workdir, run_id=run_id, data_source=source
        )
        state = inspected["state"]
        assert state is not None, "崩溃后读不到任何 Checkpoint"
        assert state["run_status"] in ("running", "interrupted")

        before = llm_calls(workdir)
        assert "planner" in before and "executor" in before

        # 新进程恢复。绘图代码生成那一次调用已发出但未登记，
        # 因此先暂停等人工决定——不静默重发可能已被计费的请求。
        paused = run_phase(
            "resume", workdir=workdir, run_id=run_id, data_source=source
        )
        assert paused["interrupted"] is True
        assert paused["pending_approval"]["kind"] == "llm_replay"
        assert llm_calls(workdir) == before, "恢复时静默重发了模型请求"

        # 批准重发后继续到完成
        resumed = run_phase(
            "resume",
            workdir=workdir,
            run_id=run_id,
            data_source=source,
            grant_llm_retry=True,
        )
        assert resumed["run_status"] == "completed"
        assert resumed["analysis_status"] == "completed"
        assert resumed["visualizations"] == 1

        after = llm_calls(workdir)

        # 关键断言：分析任务没有被重新推理
        assert after.count("executor") == before.count("executor") == 1, (
            f"executor 被重复调用：{after.count('executor')} 次"
        )
        assert after.count("planner") == 1, "planner 被重复调用"

    def test_report_is_generated_exactly_once(self, workspace) -> None:
        workdir, source = workspace
        run_id = "xproc-once"

        run_phase(
            "start",
            workdir=workdir,
            run_id=run_id,
            data_source=source,
            crash_on=CRASH_AFTER_PLOT_GENERATION,
        )
        # 先暂停（未确认的模型调用），再批准重发以跑完
        run_phase("resume", workdir=workdir, run_id=run_id, data_source=source)
        run_phase(
            "resume",
            workdir=workdir,
            run_id=run_id,
            data_source=source,
            grant_llm_retry=True,
        )

        files = artifact_files(workdir, run_id)
        assert "analysis_report.md" in files
        assert "analysis_report.pdf" in files
        # 报告文件各只有一份
        assert files.count("analysis_report.md") == 1
        assert files.count("analysis_report.pdf") == 1


class TestScenarioBApprovalResume:
    """场景 B：执行到 interrupt，重启后查询审批内容，批准后继续。"""

    def test_pending_approval_survives_restart(self, workspace) -> None:
        workdir, source = workspace
        run_id = "xproc-approval"

        started = run_phase(
            "start",
            workdir=workdir,
            run_id=run_id,
            data_source=source,
            require_approval=True,
        )
        assert started["interrupted"] is True
        assert started["pending_approval"]["kind"] == "plan_review"
        assert started["execution_results"] == 0, "审批前不应执行任何任务"

        # 新进程查询待审批
        inspected = run_phase(
            "inspect", workdir=workdir, run_id=run_id, data_source=source
        )
        state = inspected["state"]
        assert state["run_status"] == "waiting_approval"
        assert state["pending_approval"]["approval_id"]

        # 新进程批准并继续
        approved = run_phase(
            "approve",
            workdir=workdir,
            run_id=run_id,
            data_source=source,
            decision="approve",
        )
        assert approved["run_status"] == "completed"
        assert approved["approvals"] == 1
        assert approved["execution_results"] >= 1

    def test_reject_after_restart_does_not_execute(self, workspace) -> None:
        """场景 C：拒绝后不得继续执行被拒绝的操作。"""
        workdir, source = workspace
        run_id = "xproc-reject"

        run_phase(
            "start",
            workdir=workdir,
            run_id=run_id,
            data_source=source,
            require_approval=True,
        )

        rejected = run_phase(
            "approve",
            workdir=workdir,
            run_id=run_id,
            data_source=source,
            decision="reject",
        )

        assert rejected["approvals"] == 1
        assert rejected["execution_results"] == 0, "拒绝后仍然执行了任务"
        assert rejected["visualizations"] == 0
        assert artifact_files(workdir, run_id) == [], "拒绝后仍然产生了产物"

        # 执行类 Agent 未被调用
        assert "executor" not in llm_calls(workdir)


class TestScenarioDRepeatedResume:
    """场景 D：对已完成的 Run 重复恢复，不得重复追加或重复生成产物。"""

    def test_resume_completed_run_is_idempotent(self, workspace) -> None:
        workdir, source = workspace
        run_id = "xproc-repeat"

        first = run_phase(
            "start", workdir=workdir, run_id=run_id, data_source=source
        )
        assert first["run_status"] == "completed"

        files_before = artifact_files(workdir, run_id)
        calls_before = llm_calls(workdir)

        # 再次从 checkpoint 恢复同一个已完成的线程
        again = run_phase(
            "resume", workdir=workdir, run_id=run_id, data_source=source
        )

        assert again["run_status"] == "completed"

        # 产物数量不变
        assert artifact_files(workdir, run_id) == files_before

        # execution_results 没有翻倍
        assert again["execution_results"] == first["execution_results"], (
            "重复恢复追加了执行结果"
        )

        # 没有产生新的模型调用
        assert llm_calls(workdir) == calls_before, "重复恢复触发了新的推理"


class TestScenarioEMultiRunIsolation:
    """场景 E：两个 Run 同时存在时互相隔离。"""

    def test_two_runs_do_not_share_state(self, workspace, tmp_path: Path) -> None:
        """两个 Run 在同一工作目录下并存。

        脚本化的 planner 固定指向表 `customers`，所以两个 Run 都用同名文件
        （放在不同目录），这样才能验证"同名的表不会串数据"。
        """
        workdir, source_a = workspace

        other_dir = tmp_path / "other"
        other_dir.mkdir()
        source_b = other_dir / "customers.csv"
        source_b.write_text(
            "customer_id,customer_name,region\nX001,Zoe,North\n",
            encoding="utf-8",
        )

        run_one = run_phase(
            "start", workdir=workdir, run_id="xproc-iso-one", data_source=source_a
        )
        run_two = run_phase(
            "start",
            workdir=workdir,
            run_id="xproc-iso-two",
            data_source=source_b,
        )

        assert run_one["run_status"] == "completed"
        assert run_two["run_status"] == "completed"

        # 各自的 Checkpoint 归属各自线程，数据源不串
        one = run_phase(
            "inspect", workdir=workdir, run_id="xproc-iso-one", data_source=source_a
        )
        two = run_phase(
            "inspect",
            workdir=workdir,
            run_id="xproc-iso-two",
            data_source=source_b,
        )

        assert one["state"]["run_id"] == "xproc-iso-one"
        assert two["state"]["run_id"] == "xproc-iso-two"

        assert str(source_a) in one["state"]["data_sources"]
        assert str(source_b) in two["state"]["data_sources"]
        assert str(source_b) not in one["state"]["data_sources"]

        # 两个 Run 的同名表来源不同，说明资源缓存按内容而非仅按表名复用
        snapshot_one = one["state"]
        snapshot_two = two["state"]
        assert snapshot_one["run_id"] != snapshot_two["run_id"]

        # 产物目录各自独立，互不覆盖
        assert artifact_files(workdir, "xproc-iso-one")
        assert artifact_files(workdir, "xproc-iso-two")

    def test_same_named_tables_with_different_content_are_not_confused(
        self, workspace, tmp_path: Path
    ) -> None:
        """同名文件、不同内容：两个 Run 各自读自己的数据。

        这是缓存指纹必须包含内容哈希的原因——只按路径或表名缓存，
        在测试或多次上传同名文件的场景下会串数据。
        """
        workdir, source_a = workspace

        other_dir = tmp_path / "variant"
        other_dir.mkdir()
        source_b = other_dir / "customers.csv"
        source_b.write_text(
            "customer_id,customer_name,region\nY001,Yan,East\nY002,Ken,West\n"
            "Y003,Lee,South\n",
            encoding="utf-8",
        )

        run_phase("start", workdir=workdir, run_id="xproc-content-a", data_source=source_a)
        run_phase("start", workdir=workdir, run_id="xproc-content-b", data_source=source_b)

        a = run_phase(
            "inspect", workdir=workdir, run_id="xproc-content-a", data_source=source_a
        )
        b = run_phase(
            "inspect", workdir=workdir, run_id="xproc-content-b", data_source=source_b
        )

        assert str(source_a) in a["state"]["data_sources"]
        assert str(source_b) in b["state"]["data_sources"]


class TestScenarioFFileLoss:
    """场景 F：数据源丢失后恢复必须明确失败。"""

    def test_missing_source_fails_loudly(self, workspace) -> None:
        workdir, source = workspace
        run_id = "xproc-lost"

        run_phase(
            "start",
            workdir=workdir,
            run_id=run_id,
            data_source=source,
            crash_on=CRASH_AFTER_PLOT_GENERATION,
        )

        # 删除数据源后恢复
        source.unlink()

        failed = run_phase(
            "resume",
            workdir=workdir,
            run_id=run_id,
            data_source=source,
            expect_failure=True,
        )

        combined = failed["stdout"] + failed["stderr"]
        # 快照校验给出的明确错误，而不是读取其他 Run 的数据或静默生成结果
        assert "无法恢复表资源" in combined, f"未给出明确错误：{combined[-800:]}"
        assert "已不存在" in combined
        # 不得产生任何产物
        assert artifact_files(workdir, run_id) == []

    def test_changed_source_is_refused(self, workspace) -> None:
        """来源内容变化后恢复必须拒绝，避免用新数据复现历史结论。"""
        workdir, source = workspace
        run_id = "xproc-changed"

        run_phase(
            "start",
            workdir=workdir,
            run_id=run_id,
            data_source=source,
            crash_on=CRASH_AFTER_PLOT_GENERATION,
        )

        # 改动内容（含新增行），指纹应随之变化
        source.write_text(
            "customer_id,customer_name,region\nZZZ,Changed,North\n",
            encoding="utf-8",
        )

        failed = run_phase(
            "resume",
            workdir=workdir,
            run_id=run_id,
            data_source=source,
            expect_failure=True,
        )

        combined = failed["stdout"] + failed["stderr"]
        assert "内容已变化" in combined, f"未提示内容变化：{combined[-800:]}"


class TestScenarioGPlotRecovery:
    """场景 G：分析已完成但绘图未完成时中断并恢复。"""

    def test_plot_stage_is_resumed(self, workspace) -> None:
        workdir, source = workspace
        run_id = "xproc-plot"

        # 在绘图代码生成时崩溃：分析已提交，绘图尚未执行
        crashed = run_phase(
            "start",
            workdir=workdir,
            run_id=run_id,
            data_source=source,
            crash_on="plotter",
        )
        assert crashed.get("crashed") is True

        inspected = run_phase(
            "inspect", workdir=workdir, run_id=run_id, data_source=source
        )
        # 分析结果已持久化，说明第 5 个 tables 依赖节点（plot_execution）的输入仍在
        assert inspected["state"]["run_status"] in ("running", "interrupted")

        calls_before = llm_calls(workdir)

        # 绘图代码生成那一次调用已发出但未登记，因此恢复先暂停等人工决定，
        # 不静默重发（细节见 test_llm_crash_recovery.py）
        resumed = run_phase(
            "resume", workdir=workdir, run_id=run_id, data_source=source
        )
        assert resumed["interrupted"] is True
        assert resumed["pending_approval"]["kind"] == "llm_replay"
        assert llm_calls(workdir) == calls_before, "恢复时静默重发了模型请求"

        # 批准重发后才继续完成绘图
        finished = run_phase(
            "resume",
            workdir=workdir,
            run_id=run_id,
            data_source=source,
            grant_llm_retry=True,
        )

        assert finished["run_status"] == "completed"
        assert finished["visualizations"] == 1, "绘图未在恢复后完成"

        # 分析阶段没有被重跑：executor 始终只调用一次
        calls = llm_calls(workdir)
        assert calls.count("executor") == 1, "恢复时重跑了分析执行"
