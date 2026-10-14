"""Run 生命周期登记。

Checkpoint 保存的是图状态，但不适合用来回答"现在有哪些 Run、各自处于什么阶段"
这类运维问题——遍历所有 thread 代价高。这里用一个独立的轻量登记表：

- 与 Checkpoint 同一个 SQLite 文件，避免多一个需要运维的存储；
- 只存索引与生命周期，权威状态仍在 Checkpoint 里，两者不一致时以 Checkpoint 为准；
- 进程重启后可直接查询，因此"服务重启后仍能看到待审批"是查表而不是查内存。

生命周期取值：
    pending          已登记，尚未开始
    running          正在执行
    waiting_approval 已中断，等待人工审批
    completed        正常结束（含 partial：有计划但有任务失败）
    failed           执行失败（异常终止或分析未产出）
    interrupted      被主动中断，可恢复
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

RUN_STATUSES = (
    "pending",
    "running",
    "waiting_approval",
    "completed",
    "failed",
    "interrupted",
)

DEFAULT_RUN_DB = Path("outputs") / "runs" / "runs.sqlite"

_SCHEMA = """
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

# 心跳超过这个时长就认为持有者已消失。取一个明显大于单步耗时的值：
# 太长会把真正卡住的 Run 一直显示为 running，太短会误判正在执行的 Run。
STALE_HEARTBEAT_SECONDS = 120.0


def _process_alive(pid: int | None) -> bool:
    """检查 pid 是否仍然存在。

    只用于判断"上次执行者是否还在"，不作为安全边界。
    """
    if not pid:
        return False

    if pid == os.getpid():
        return True

    try:
        if os.name == "nt":
            import ctypes

            handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
            if not handle:
                return False
            ctypes.windll.kernel32.CloseHandle(handle)
            return True

        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


@dataclass
class RunRecord:
    """一条 Run 登记。"""

    run_id: str
    status: str
    data_sources: list[str] = field(default_factory=list)
    analysis_focus: str | None = None
    created_at: float = 0.0
    updated_at: float = 0.0
    error: str | None = None
    pending_approval: dict[str, Any] | None = None
    approvals: list[dict[str, Any]] = field(default_factory=list)
    owner_pid: int | None = None
    heartbeat_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "data_sources": self.data_sources,
            "analysis_focus": self.analysis_focus,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
            "pending_approval": self.pending_approval,
            "approvals": self.approvals,
            "owner_pid": self.owner_pid,
            "heartbeat_at": self.heartbeat_at,
        }


def run_database_path(root: Path | str = "outputs") -> Path:
    return Path(root) / "runs" / "runs.sqlite"


class RunStore:
    """Run 登记的读写。"""

    def __init__(self, database: Path | str | None = None) -> None:
        self._path = Path(database) if database is not None else DEFAULT_RUN_DB
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(_SCHEMA)
            self._migrate(connection)

    @staticmethod
    def _migrate(connection: sqlite3.Connection) -> None:
        """为已存在的库补上后加的列。

        `CREATE TABLE IF NOT EXISTS` 不会改动已有表，因此新增列需要显式补齐，
        否则升级后读取会直接失败。
        """
        existing = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(runs)").fetchall()
        }

        for column, ddl in (
            ("owner_pid", "INTEGER"),
            ("heartbeat_at", "REAL"),
        ):
            if column not in existing:
                connection.execute(f"ALTER TABLE runs ADD COLUMN {column} {ddl}")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(str(self._path))
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def create(
        self,
        run_id: str,
        *,
        data_sources: list[str],
        analysis_focus: str | None = None,
        status: str = "pending",
    ) -> RunRecord:
        now = time.time()
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO runs "
                "(run_id, status, data_sources, analysis_focus, created_at, "
                " updated_at, error, pending_approval, approvals) "
                "VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, '[]')",
                (
                    run_id,
                    status,
                    json.dumps(data_sources, ensure_ascii=False),
                    analysis_focus,
                    now,
                    now,
                ),
            )
        return self.get(run_id)  # type: ignore[return-value]

    def get(self, run_id: str) -> RunRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return _row_to_record(row) if row is not None else None

    def list(self, *, limit: int = 50) -> list[RunRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM runs ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_row_to_record(row) for row in rows]

    def update_status(
        self,
        run_id: str,
        status: str,
        *,
        error: str | None = None,
        pending_approval: dict[str, Any] | None = None,
    ) -> None:
        if status not in RUN_STATUSES:
            raise ValueError(f"未知的运行状态：{status}")

        # 进入 running 时记下持有者，便于事后判断"这个 running 是不是已经死了"
        owner = os.getpid() if status == "running" else None
        heartbeat = time.time() if status == "running" else None

        with self._connect() as connection:
            connection.execute(
                "UPDATE runs SET status = ?, updated_at = ?, error = ?, "
                "pending_approval = ?, owner_pid = COALESCE(?, owner_pid), "
                "heartbeat_at = COALESCE(?, heartbeat_at) WHERE run_id = ?",
                (
                    status,
                    time.time(),
                    error,
                    json.dumps(pending_approval, ensure_ascii=False)
                    if pending_approval
                    else None,
                    owner,
                    heartbeat,
                    run_id,
                ),
            )

    def touch(self, run_id: str) -> None:
        """刷新心跳，表示执行者仍然活着。"""
        with self._connect() as connection:
            connection.execute(
                "UPDATE runs SET heartbeat_at = ?, updated_at = ?, "
                "owner_pid = ? WHERE run_id = ?",
                (time.time(), time.time(), os.getpid(), run_id),
            )

    def is_stale_running(self, run_id: str) -> bool:
        """判断一个 running 状态的 Run 是否已经失去执行者。

        为什么需要它：`running` 无法区分"正在执行"与"执行者已崩溃"。
        仅凭状态会让被杀死的 Run 永远显示为 running。这里用
        "持有者进程是否还在" 与 "心跳是否过期" 两个信号来判断，
        它是启发式的**可观测性**手段，不是自动重启机制。
        """
        record = self.get(run_id)
        if record is None or record.status != "running":
            return False

        if not _process_alive(record.owner_pid):
            return True

        if record.heartbeat_at is None:
            return True

        return (time.time() - record.heartbeat_at) > STALE_HEARTBEAT_SECONDS

    def record_approval(self, run_id: str, approval: dict[str, Any]) -> None:
        record = self.get(run_id)
        if record is None:
            raise KeyError(f"未登记的 run：{run_id}")

        approvals = list(record.approvals)
        approvals.append(approval)

        with self._connect() as connection:
            connection.execute(
                "UPDATE runs SET approvals = ?, updated_at = ?, "
                "pending_approval = NULL WHERE run_id = ?",
                (json.dumps(approvals, ensure_ascii=False), time.time(), run_id),
            )


def _row_to_record(row: sqlite3.Row) -> RunRecord:
    return RunRecord(
        run_id=row["run_id"],
        status=row["status"],
        data_sources=json.loads(row["data_sources"] or "[]"),
        analysis_focus=row["analysis_focus"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        error=row["error"],
        pending_approval=json.loads(row["pending_approval"])
        if row["pending_approval"]
        else None,
        approvals=json.loads(row["approvals"] or "[]"),
        owner_pid=row["owner_pid"],
        heartbeat_at=row["heartbeat_at"],
    )
