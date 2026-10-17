"""RunStore 的非破坏性迁移。

P2-C 之前的 `runs` 表没有 `sql_backend` 列。升级后必须：

- 自动补列，不需要人工迁移；
- 既有行的数据**一字不改**；
- 旧 Run 落在 `direct` 上——那正是它们当初实际使用的后端，
  历史语义不被改写。

反向验证也重要：不能因为加了默认值就把旧 Run 说成 mcp。
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

import pytest

from services.run_store import RunStore

# 基线（c1a1802）的建表语句，不含 sql_backend
LEGACY_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    status        TEXT NOT NULL,
    data_sources  TEXT NOT NULL,
    analysis_focus TEXT,
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL,
    error         TEXT,
    pending_approval TEXT,
    approvals     TEXT NOT NULL DEFAULT '[]',
    owner_pid     INTEGER,
    heartbeat_at  REAL
);
"""


def build_legacy_database(path: Path) -> None:
    """造一个 P2-C 结构的库，塞进几行历史数据。"""
    connection = sqlite3.connect(str(path))
    try:
        connection.executescript(LEGACY_SCHEMA)
        now = time.time()
        connection.execute(
            "INSERT INTO runs (run_id, status, data_sources, analysis_focus, "
            "created_at, updated_at, error, pending_approval, approvals, "
            "owner_pid, heartbeat_at) "
            "VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, '[]', NULL, NULL)",
            (
                "run_legacy_done",
                "completed",
                json.dumps(["customers.csv"]),
                "历史分析",
                now - 1000,
                now - 900,
            ),
        )
        connection.execute(
            "INSERT INTO runs (run_id, status, data_sources, analysis_focus, "
            "created_at, updated_at, error, pending_approval, approvals, "
            "owner_pid, heartbeat_at) "
            "VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, NULL, NULL)",
            (
                "run_legacy_waiting",
                "waiting_approval",
                json.dumps(["sales.sqlite"]),
                "等待审批",
                now - 500,
                now - 400,
                json.dumps({"approval_id": "ap-1", "node": "approval_gate"}),
                json.dumps([{"approval_id": "ap-0", "decision": "reject"}]),
            ),
        )
        connection.commit()
    finally:
        connection.close()


class TestMigration:
    def test_legacy_database_gains_the_column(self, tmp_path: Path) -> None:
        db = tmp_path / "runs.sqlite"
        build_legacy_database(db)

        # 迁移前：确实没有这一列
        connection = sqlite3.connect(str(db))
        try:
            before = {
                row[1] for row in connection.execute("PRAGMA table_info(runs)")
            }
        finally:
            connection.close()
        assert "sql_backend" not in before

        RunStore(db)

        connection = sqlite3.connect(str(db))
        try:
            after = {row[1] for row in connection.execute("PRAGMA table_info(runs)")}
        finally:
            connection.close()
        assert "sql_backend" in after

    def test_legacy_rows_default_to_direct(self, tmp_path: Path) -> None:
        db = tmp_path / "runs.sqlite"
        build_legacy_database(db)

        store = RunStore(db)

        assert store.get("run_legacy_done").sql_backend == "direct"
        assert store.get("run_legacy_waiting").sql_backend == "direct"

    def test_migration_preserves_existing_data(self, tmp_path: Path) -> None:
        """迁移只加列，不改写既有行。"""
        db = tmp_path / "runs.sqlite"
        build_legacy_database(db)

        store = RunStore(db)

        done = store.get("run_legacy_done")
        assert done.status == "completed"
        assert done.data_sources == ["customers.csv"]
        assert done.analysis_focus == "历史分析"

        waiting = store.get("run_legacy_waiting")
        assert waiting.status == "waiting_approval"
        assert waiting.pending_approval == {
            "approval_id": "ap-1",
            "node": "approval_gate",
        }
        assert waiting.approvals == [{"approval_id": "ap-0", "decision": "reject"}]

    def test_migration_is_idempotent(self, tmp_path: Path) -> None:
        db = tmp_path / "runs.sqlite"
        build_legacy_database(db)

        for _ in range(3):
            RunStore(db)

        connection = sqlite3.connect(str(db))
        try:
            columns = [row[1] for row in connection.execute("PRAGMA table_info(runs)")]
        finally:
            connection.close()

        assert columns.count("sql_backend") == 1

    def test_new_runs_after_migration_can_select_mcp(self, tmp_path: Path) -> None:
        """迁移后的库里新建 Run 仍能写 mcp。"""
        db = tmp_path / "runs.sqlite"
        build_legacy_database(db)
        store = RunStore(db)

        store.create("run_new_mcp", data_sources=["a.csv"], sql_backend="mcp")

        assert store.get("run_new_mcp").sql_backend == "mcp"
        assert store.get("run_legacy_done").sql_backend == "direct"

    def test_legacy_row_read_without_column_in_select(self, tmp_path: Path) -> None:
        """即使某条查询没取到新列，记录对象也要给出 direct 而不是崩。"""
        from services.run_store import _row_to_record

        db = tmp_path / "runs.sqlite"
        build_legacy_database(db)

        connection = sqlite3.connect(str(db))
        connection.row_factory = sqlite3.Row
        try:
            # 显式只选旧列，模拟按旧列清单读取
            row = connection.execute(
                "SELECT run_id, status, data_sources, analysis_focus, created_at, "
                "updated_at, error, pending_approval, approvals, owner_pid, "
                "heartbeat_at FROM runs WHERE run_id = ?",
                ("run_legacy_done",),
            ).fetchone()
        finally:
            connection.close()

        record = _row_to_record(row)
        assert record.sql_backend == "direct"
        assert record.mcp_settings.mode == "direct"

    def test_invalid_backend_is_refused_on_write(self, tmp_path: Path) -> None:
        store = RunStore(tmp_path / "runs.sqlite")

        with pytest.raises(ValueError):
            store.create("run_bad", data_sources=["a.csv"], sql_backend="shell")

        with pytest.raises(ValueError):
            store.set_sql_backend("run_bad", "MCP")

    def test_set_sql_backend_is_explicit_and_audited(self, tmp_path: Path) -> None:
        """改写后端是显式操作，且只影响目标 Run。"""
        store = RunStore(tmp_path / "runs.sqlite")
        store.create("run_a", data_sources=["a.csv"], sql_backend="direct")
        store.create("run_b", data_sources=["b.csv"], sql_backend="direct")

        store.set_sql_backend("run_a", "mcp")

        assert store.get_sql_backend("run_a") == "mcp"
        assert store.get_sql_backend("run_b") == "direct"

    def test_record_rebuilds_mcp_settings_from_stored_mode(
        self, tmp_path: Path
    ) -> None:
        store = RunStore(tmp_path / "runs.sqlite")
        store.create("run_mcp", data_sources=["a.csv"], sql_backend="mcp")
        store.create("run_direct", data_sources=["b.csv"], sql_backend="direct")

        assert store.get("run_mcp").mcp_settings.is_mcp is True
        assert store.get("run_direct").mcp_settings.is_mcp is False


class TestStaleRunningDetection:
    """`is_stale_running` 必须在持有者**已经死掉**时立刻为真。

    回归背景：Windows 上 `OpenProcess` 对一个已被杀掉但尚未回收的进程
    仍会返回有效句柄，只判断句柄会把**僵尸进程当成存活**。
    于是"执行者已崩溃"的 Run 只能靠 120 秒心跳超时才被判为过期，
    恢复接口在那之前一直返回 409——跨进程恢复用例因此偶发失败。
    """

    def test_terminated_process_is_not_alive(self) -> None:
        """核心回归：被杀掉且未回收的子进程必须判定为已死。"""
        import ctypes
        import subprocess
        import sys

        from services.run_store import _process_alive

        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"]
        )
        try:
            time.sleep(1.0)
            assert _process_alive(process.pid) is True

            if os.name == "nt":
                handle = ctypes.windll.kernel32.OpenProcess(
                    0x0001, False, process.pid
                )
                assert handle, "无法打开子进程句柄"
                try:
                    ctypes.windll.kernel32.TerminateProcess(handle, 7)
                    time.sleep(0.5)
                    # 此时进程已退出但尚未 wait() 回收
                    assert _process_alive(process.pid) is False, (
                        "僵尸进程被误判为存活"
                    )
                finally:
                    ctypes.windll.kernel32.CloseHandle(handle)
            else:
                process.kill()
                time.sleep(0.5)
                process.wait()
                assert _process_alive(process.pid) is False
        finally:
            process.kill()
            process.wait()

    def test_missing_pid_is_not_alive(self) -> None:
        from services.run_store import _process_alive

        assert _process_alive(None) is False
        assert _process_alive(0) is False
        # 一个几乎不可能存在的 pid
        assert _process_alive(999_999) is False

    def test_own_pid_is_alive(self) -> None:
        from services.run_store import _process_alive

        assert _process_alive(os.getpid()) is True

    def test_dead_owner_makes_a_running_run_stale_without_waiting(
        self, tmp_path: Path
    ) -> None:
        """持有者已死时，无需等心跳过期就应判定为 stale。"""
        db = tmp_path / "runs.sqlite"
        store = RunStore(db)
        store.create("run_crashed", data_sources=["a.csv"])
        store.update_status("run_crashed", "running")
        # touch 会记下**当前**进程作为持有者，心跳也是刚刚
        store.touch("run_crashed")

        # 伪造一个已不存在的持有者，但心跳保持"刚刚更新"
        with sqlite3.connect(db) as connection:
            connection.execute(
                "UPDATE runs SET owner_pid = ?, heartbeat_at = ? WHERE run_id = ?",
                (999_999, time.time(), "run_crashed"),
            )
            connection.commit()

        assert store.get("run_crashed").status == "running"
        assert store.is_stale_running("run_crashed") is True, (
            "持有者已消失但心跳很新，仍应判定为 stale"
        )

    def test_live_owner_with_fresh_heartbeat_is_not_stale(
        self, tmp_path: Path
    ) -> None:
        store = RunStore(tmp_path / "runs.sqlite")
        store.create("run_live", data_sources=["a.csv"])
        store.update_status("run_live", "running")
        # touch 记录的是当前进程 pid，因此持有者是"活着"的
        store.touch("run_live")

        assert store.is_stale_running("run_live") is False

    def test_non_running_records_are_never_stale(self, tmp_path: Path) -> None:
        store = RunStore(tmp_path / "runs.sqlite")
        store.create("run_done", data_sources=["a.csv"])
        store.update_status("run_done", "completed")

        assert store.is_stale_running("run_done") is False
        assert store.is_stale_running("no_such_run") is False
