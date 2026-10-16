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
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:  # pragma: no cover - 仅用于类型标注
    from core.runtime import McpSettings

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
    heartbeat_at  REAL,
    sql_backend   TEXT NOT NULL DEFAULT 'direct',
    require_plan_approval INTEGER NOT NULL DEFAULT 0,
    require_report_approval INTEGER NOT NULL DEFAULT 0
);
"""

# 心跳超过这个时长就认为持有者已消失。取一个明显大于单步耗时的值：
# 太长会把真正卡住的 Run 一直显示为 running，太短会误判正在执行的 Run。
#
# 注意这只是**兜底**：正常路径靠 `_process_alive` 判断持有者是否还在，
# 心跳超时用于"进程还活着但已经卡死"的情况。
STALE_HEARTBEAT_SECONDS = 120.0

#: Windows `GetExitCodeProcess` 在进程仍在运行时的返回值。
_STILL_ACTIVE = 259


def _process_alive(pid: int | None) -> bool:
    """检查 pid 是否仍然是**活着的**进程。

    只用于判断"上次执行者是否还在"，不作为安全边界。

    Windows 上必须查退出码，不能只看 `OpenProcess` 是否成功：
    一个已经被杀掉但尚未被回收的进程仍然是一个内核对象，
    `OpenProcess` 会返回有效句柄（哪怕只请求
    `PROCESS_QUERY_LIMITED_INFORMATION`）。只判断句柄会把
    **僵尸进程当成存活**，于是"执行者已崩溃"的 Run 只能靠
    120 秒心跳超时才被判为过期——恢复接口在那之前一直返回 409。
    这正是跨进程恢复用例偶发失败的原因。

    因此这里用 `GetExitCodeProcess`：退出码仍是 `STILL_ACTIVE`
    才算活着，否则说明它已经退出（僵尸或已回收）。
    """
    if not pid:
        return False

    if pid == os.getpid():
        return True

    try:
        if os.name == "nt":
            import ctypes

            kernel32 = ctypes.windll.kernel32
            # PROCESS_QUERY_LIMITED_INFORMATION：只需查询权限，
            # 不要请求 PROCESS_QUERY_INFORMATION，它会被权限策略拒绝
            handle = kernel32.OpenProcess(0x1000, False, pid)
            if not handle:
                # 打不开通常意味着进程已不存在；也可能是权限不足，
                # 此时保守地当作"不存在"，因为心跳超时仍会兜底
                return False
            try:
                exit_code = ctypes.c_ulong()
                ok = kernel32.GetExitCodeProcess(
                    handle, ctypes.byref(exit_code)
                )
                if not ok:
                    return False
                return exit_code.value == _STILL_ACTIVE
            finally:
                kernel32.CloseHandle(handle)

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
    # SQL 后端模式：创建时确定，之后不再改变。
    # 恢复必须依据它，否则一个 mcp 模式的 Run 可能被恢复成 direct。
    sql_backend: str = "direct"
    # 是否要求计划审批：同样在创建时确定，恢复时必须沿用。
    # 硬写成 True 会让"崩溃点晚于审批门"的 Run 恢复时又被拦一次。
    require_plan_approval: bool = False
    # 是否要求报告发布前确认：同上。
    require_report_approval: bool = False

    @property
    def mcp_settings(self) -> "McpSettings":
        """按登记的后端模式重建 MCP 配置。

        只重建**模式**；Server 模块、超时与私有目录仍由当前进程的可信配置
        决定，不从数据库读取——那些值不该成为可持久化的攻击面。
        """
        from core.runtime import McpSettings

        return McpSettings(mode=self.sql_backend)

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
            # 只暴露模式名。MCP 私有配置路径、Server 启动命令与环境变量
            # 一律不出现在这里——它们是宿主内部实现，不是运行信息。
            "sql_backend": self.sql_backend,
            "require_plan_approval": self.require_plan_approval,
            "require_report_approval": self.require_report_approval,
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

        非破坏性：只 `ADD COLUMN`，不重建表、不改动既有行。
        `sql_backend` 带 `DEFAULT 'direct'`，因此**旧 Run 自动落在 direct 上**
        ——那正是它们当初实际使用的后端，历史语义不会被改写。
        刻意不设 `NOT NULL`：SQLite 的 `ADD COLUMN` 对既有行只做默认值填充，
        这里保持与建表语句一致的宽松性，避免在旧库上因约束失败而启动不了。
        """
        existing = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(runs)").fetchall()
        }

        for column, ddl in (
            ("owner_pid", "INTEGER"),
            ("heartbeat_at", "REAL"),
            ("sql_backend", "TEXT NOT NULL DEFAULT 'direct'"),
            # 审批要求也是"创建时确定、恢复时必须沿用"的设置。
            # 旧 Run 默认 0（不要求审批）——与它们当初的行为一致。
            ("require_plan_approval", "INTEGER NOT NULL DEFAULT 0"),
            ("require_report_approval", "INTEGER NOT NULL DEFAULT 0"),
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

    @staticmethod
    def _normalize_backend(sql_backend: str) -> str:
        """收敛并校验后端取值。

        用 `SqlBackend` 而不是各写一份字面量：写入库里的值必须与运行期
        接受的值来自同一个定义，否则将来加后端时会出现"API 能收、库里不认"。
        """
        from core.runtime import SqlBackend

        try:
            return SqlBackend(sql_backend).value
        except ValueError as error:
            raise ValueError(f"不支持的 SQL 后端模式：{sql_backend}") from error

    def create(
        self,
        run_id: str,
        *,
        data_sources: list[str],
        analysis_focus: str | None = None,
        status: str = "pending",
        sql_backend: str = "direct",
        require_plan_approval: bool = False,
        require_report_approval: bool = False,
    ) -> RunRecord:
        now = time.time()
        backend = self._normalize_backend(sql_backend)

        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO runs "
                "(run_id, status, data_sources, analysis_focus, created_at, "
                " updated_at, error, pending_approval, approvals, sql_backend, "
                " require_plan_approval, require_report_approval) "
                "VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, '[]', ?, ?, ?)",
                (
                    run_id,
                    status,
                    json.dumps(data_sources, ensure_ascii=False),
                    analysis_focus,
                    now,
                    now,
                    backend,
                    1 if require_plan_approval else 0,
                    1 if require_report_approval else 0,
                ),
            )
        return self.get(run_id)  # type: ignore[return-value]

    def get_sql_backend(self, run_id: str) -> str:
        """读取登记的后端模式；未登记的 Run 视为 direct。"""
        record = self.get(run_id)
        return record.sql_backend if record is not None else "direct"

    def set_sql_backend(self, run_id: str, sql_backend: str) -> None:
        """改写已登记 Run 的后端模式。

        正常流程不需要它（模式在 create 时一次写定）。保留是为了让运维脚本
        能在明确知情的前提下切换某个历史 Run 的后端。
        """
        backend = self._normalize_backend(sql_backend)

        with self._connect() as connection:
            connection.execute(
                "UPDATE runs SET sql_backend = ?, updated_at = ? WHERE run_id = ?",
                (backend, time.time(), run_id),
            )

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
    keys = row.keys()
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
        # 未迁移过的行没有这列；按 direct 处理，与列默认值一致
        sql_backend=(row["sql_backend"] or "direct")
        if "sql_backend" in keys
        else "direct",
        require_plan_approval=bool(row["require_plan_approval"])
        if "require_plan_approval" in keys
        else False,
        require_report_approval=bool(row["require_report_approval"])
        if "require_report_approval" in keys
        else False,
    )
