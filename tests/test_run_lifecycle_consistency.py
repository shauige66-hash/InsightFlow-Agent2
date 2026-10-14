"""P1 补验收：RunStore 与 Checkpointer 的状态一致性。

两者分别持久化生命周期与图状态，因此需要验证几个容易出问题的点：

- 进程被杀后，登记为 running 的 Run 能否被识别为"需要恢复"；
- 它会不永远停留在 running；
- 并发 resume 不会启动两个执行线程；
- 拒绝审批 / 失败 / 完成三种终态与 Checkpoint 一致。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from services.run_store import STALE_HEARTBEAT_SECONDS, RunStore

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DRIVER = PROJECT_ROOT / "scripts" / "recovery_driver.py"
PYTHON = sys.executable


def run_driver(*args: str, env_extra: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", **(env_extra or {})}
    return subprocess.run(
        [str(PYTHON), str(DRIVER), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(PROJECT_ROOT),
        timeout=300,
        env=env,
    )


@pytest.fixture
def store(tmp_path: Path) -> RunStore:
    return RunStore(tmp_path / "runs.sqlite")


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


class TestStaleRunningDetection:
    """running 状态在进程消失后必须能被识别，而不是永久 running。"""

    def test_running_run_owned_by_a_dead_process_is_stale(self, store: RunStore) -> None:
        store.create("run_dead", data_sources=["a.csv"])
        store.update_status("run_dead", "running")

        # 模拟持有者进程已消失：直接写入一个不存在的 pid
        with store._connect() as connection:
            connection.execute(
                "UPDATE runs SET owner_pid = ?, heartbeat_at = ? WHERE run_id = ?",
                (999_999_999, time.time(), "run_dead"),
            )

        assert store.is_stale_running("run_dead") is True

    def test_expired_heartbeat_is_stale(self, store: RunStore) -> None:
        store.create("run_quiet", data_sources=["a.csv"])
        store.update_status("run_quiet", "running")

        with store._connect() as connection:
            connection.execute(
                "UPDATE runs SET heartbeat_at = ? WHERE run_id = ?",
                (time.time() - STALE_HEARTBEAT_SECONDS - 60, "run_quiet"),
            )

        assert store.is_stale_running("run_quiet") is True

    def test_live_run_is_not_stale(self, store: RunStore) -> None:
        store.create("run_live", data_sources=["a.csv"])
        store.update_status("run_live", "running")
        store.touch("run_live")

        assert store.is_stale_running("run_live") is False

    def test_terminal_states_are_never_stale(self, store: RunStore) -> None:
        for status in ("completed", "failed", "waiting_approval"):
            run_id = f"run_{status}"
            store.create(run_id, data_sources=["a.csv"])
            store.update_status(run_id, status)
            assert store.is_stale_running(run_id) is False

    def test_touch_updates_heartbeat(self, store: RunStore) -> None:
        store.create("run_touch", data_sources=["a.csv"])
        store.update_status("run_touch", "running")

        before = store.get("run_touch").heartbeat_at
        time.sleep(0.05)
        store.touch("run_touch")

        assert store.get("run_touch").heartbeat_at > before


class TestStaleAfterRealProcessKill:
    """真实进程被强杀后，登记状态能反映"需要恢复"。"""

    def test_killed_run_is_running_and_stale_after_owner_exits(
        self, workspace, tmp_path: Path
    ) -> None:
        workdir, source = workspace
        run_id = "lifecycle-killed"

        # 进程在绘图时被强杀，Registry 里留下 running（由 driver 之外的地方登记）
        completed = run_driver(
            "start",
            "--workdir",
            str(workdir),
            "--run-id",
            run_id,
            "--data-source",
            str(source),
            "--crash-on",
            "plotter",
            "--crash-at",
            "after_request",
        )
        assert completed.returncode == 17

        # 驱动脚本本身不写 RunStore，这里模拟 API 侧登记后进程消失的情形
        store = RunStore(tmp_path / "runs.sqlite")
        store.create(run_id, data_sources=[str(source)])
        store.update_status(run_id, "running")

        with store._connect() as connection:
            connection.execute(
                "UPDATE runs SET owner_pid = ?, heartbeat_at = ? WHERE run_id = ?",
                (999_999_999, time.time(), run_id),
            )

        record = store.get(run_id)
        assert record.status == "running"
        assert store.is_stale_running(run_id) is True

        # 恢复后应进入终态，不再停留在 running
        run_driver(
            "resume",
            "--workdir",
            str(workdir),
            "--run-id",
            run_id,
            "--data-source",
            str(source),
            "--grant-llm-retry",
        )

        with SqliteSaver.from_conn_string(str(workdir / "checkpoints.sqlite")) as saver:
            snapshot = saver.get_tuple({"configurable": {"thread_id": run_id}})
            assert snapshot.checkpoint["channel_values"]["run_status"] == "completed"


class TestConcurrentResume:
    """同一 Run 的并发 resume 不得启动两个执行线程。"""

    def test_second_resume_is_rejected_while_first_is_active(
        self, workspace, tmp_path: Path
    ) -> None:
        """执行器认为 Run 正在运行时，resume 必须拒绝。"""
        from services.run_coordinator import RunCoordinator

        workdir, source = workspace
        store = RunStore(tmp_path / "runs.sqlite")
        coordinator = RunCoordinator(
            store,
            output_root=workdir,
            checkpointer_path=workdir / "checkpoints.sqlite",
            model_factory_builder=lambda run_id: None,
            max_workers=1,
        )

        run_id = "lifecycle-concurrent"
        store.create(run_id, data_sources=[str(source)])
        store.update_status(run_id, "waiting_approval", pending_approval={"a": 1})

        # 占住执行器，模拟"上一次 resume 仍在跑"
        release = threading.Event()
        coordinator.executor.submit(run_id, lambda: release.wait(timeout=30))

        try:
            with pytest.raises(ValueError, match="正在执行"):
                coordinator.resume(run_id, resume_value={"decision": "approve"})
        finally:
            release.set()
            coordinator.shutdown()

    def test_resume_rejects_completed_run(self, store: RunStore) -> None:
        from services.run_coordinator import RunCoordinator

        coordinator = RunCoordinator(store, model_factory_builder=lambda run_id: None)
        store.create("run_done", data_sources=["a.csv"])
        store.update_status("run_done", "completed")

        try:
            with pytest.raises(ValueError, match="不可恢复"):
                coordinator.resume("run_done")
        finally:
            coordinator.shutdown()

    def test_executor_reports_active_state(self, store: RunStore) -> None:
        from services.run_coordinator import RunCoordinator

        coordinator = RunCoordinator(store, model_factory_builder=lambda run_id: None)
        release = threading.Event()

        try:
            coordinator.executor.submit("run_active", lambda: release.wait(timeout=30))
            assert coordinator.executor.is_active("run_active") is True

            release.set()
            coordinator.executor.wait("run_active", timeout=10)
            assert coordinator.executor.is_active("run_active") is False
        finally:
            release.set()
            coordinator.shutdown()


class TestTerminalStatesMatchCheckpoint:
    """终态在 RunStore 与 Checkpoint 之间必须一致。"""

    def test_completed_run_agrees(self, workspace, tmp_path: Path) -> None:
        workdir, source = workspace
        run_id = "lifecycle-completed"

        completed = run_driver(
            "start",
            "--workdir",
            str(workdir),
            "--run-id",
            run_id,
            "--data-source",
            str(source),
        )
        assert completed.returncode == 0
        summary = json.loads(completed.stdout.strip().splitlines()[-1])
        assert summary["run_status"] == "completed"

        with SqliteSaver.from_conn_string(str(workdir / "checkpoints.sqlite")) as saver:
            snapshot = saver.get_tuple({"configurable": {"thread_id": run_id}})
            values = snapshot.checkpoint["channel_values"]
            assert values["run_status"] == "completed"
            assert values["analysis_status"] == "completed"

    def test_rejected_run_reports_failed_and_does_no_work(
        self, workspace, tmp_path: Path
    ) -> None:
        workdir, source = workspace
        run_id = "lifecycle-rejected"

        started = run_driver(
            "start",
            "--workdir",
            str(workdir),
            "--run-id",
            run_id,
            "--data-source",
            str(source),
            "--require-approval",
        )
        summary = json.loads(started.stdout.strip().splitlines()[-1])
        assert summary["interrupted"] is True

        rejected = run_driver(
            "approve",
            "--workdir",
            str(workdir),
            "--run-id",
            run_id,
            "--data-source",
            str(source),
            "--decision",
            "reject",
        )
        summary = json.loads(rejected.stdout.strip().splitlines()[-1])

        assert summary["run_status"] == "failed"
        assert summary["status_reason"] == "分析计划未获批准"
        assert summary["execution_results"] == 0

        with SqliteSaver.from_conn_string(str(workdir / "checkpoints.sqlite")) as saver:
            snapshot = saver.get_tuple({"configurable": {"thread_id": run_id}})
            values = snapshot.checkpoint["channel_values"]
            assert values["run_status"] == "failed"
            assert values["analysis_status"] == "failed"

    def test_waiting_approval_is_reported(self, workspace) -> None:
        workdir, source = workspace
        run_id = "lifecycle-waiting"

        started = run_driver(
            "start",
            "--workdir",
            str(workdir),
            "--run-id",
            run_id,
            "--data-source",
            str(source),
            "--require-approval",
        )
        summary = json.loads(started.stdout.strip().splitlines()[-1])

        assert summary["interrupted"] is True
        assert summary["pending_approval"]["kind"] == "plan_review"
        # 审批期间不得执行任何任务
        assert summary["execution_results"] == 0
