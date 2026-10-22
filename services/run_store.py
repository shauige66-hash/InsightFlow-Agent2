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
from services.run_identity import RunConflict, LeaseLost, current_owner, fingerprint, lease_token_for

if TYPE_CHECKING:  # pragma: no cover - 仅用于类型标注
    from core.runtime import McpSettings

RUN_STATUSES = (
    # queued：已登记、等待 worker 抢占（只有队列模式会用到）
    "queued",
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
    require_report_approval INTEGER NOT NULL DEFAULT 0,
    owner TEXT NOT NULL DEFAULT 'local-demo',
    request_hash TEXT,
    request_payload TEXT NOT NULL DEFAULT '{}',
    lease_token TEXT,
    lease_expires_at REAL
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
                # Access denied is not proof of death. Let the lease expire.
                return kernel32.GetLastError() == 5
            try:
                exit_code = ctypes.c_ulong()
                ok = kernel32.GetExitCodeProcess(
                    handle, ctypes.byref(exit_code)
                )
                if not ok:
                    return True
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

    owner: str = "local-demo"
    request_hash: str | None = None
    request_payload: dict[str, Any] = field(default_factory=dict)
    lease_token: str | None = None
    lease_expires_at: float | None = None

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
            "owner": self.owner,
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
    """Owner-scoped registration with insert-only identity and atomic execution leases."""
    def __init__(self, database: Path | str | None = None) -> None:
        self._path = Path(database) if database is not None else DEFAULT_RUN_DB
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(_SCHEMA)
            self._migrate(connection)

    @staticmethod
    def _migrate(connection):
        existing = {row["name"] for row in connection.execute("PRAGMA table_info(runs)")}
        columns = [
            ("owner_pid", "INTEGER"), ("heartbeat_at", "REAL"),
            ("sql_backend", "TEXT NOT NULL DEFAULT 'direct'"),
            ("require_plan_approval", "INTEGER NOT NULL DEFAULT 0"),
            ("require_report_approval", "INTEGER NOT NULL DEFAULT 0"),
            ("owner", "TEXT NOT NULL DEFAULT 'local-demo'"),
            ("request_hash", "TEXT"), ("request_payload", "TEXT NOT NULL DEFAULT '{}'"),
            ("lease_token", "TEXT"), ("lease_expires_at", "REAL"),
        ]
        for name, ddl in columns:
            if name not in existing:
                connection.execute(f"ALTER TABLE runs ADD COLUMN {name} {ddl}")

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(str(self._path), timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _normalize_backend(value):
        from core.runtime import SqlBackend
        try:
            return SqlBackend(value).value
        except ValueError as error:
            raise ValueError(f"不支持的 SQL 后端模式：{value}") from error

    def create(self, run_id, *, data_sources, analysis_focus=None, status="pending",
               sql_backend="direct", require_plan_approval=False, require_report_approval=False,
               owner=None, request_payload=None):
        owner = current_owner(owner)
        payload = request_payload if request_payload is not None else {
            "data_sources": data_sources, "analysis_focus": analysis_focus,
            "sql_backend": self._normalize_backend(sql_backend),
            "require_plan_approval": bool(require_plan_approval),
            "require_report_approval": bool(require_report_approval),
        }
        digest = fingerprint(payload)
        now = time.time()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO runs (run_id,status,data_sources,analysis_focus,created_at,updated_at,"
                "approvals,sql_backend,require_plan_approval,require_report_approval,owner,request_hash,request_payload) "
                "VALUES (?,?,?,?,?,?,'[]',?,?,?,?,?,?) ON CONFLICT(run_id) DO NOTHING",
                (run_id,status,json.dumps(data_sources),analysis_focus,now,now,
                 self._normalize_backend(sql_backend),int(require_plan_approval),int(require_report_approval),
                 owner,digest,json.dumps(payload,ensure_ascii=False)),
            )
            row = connection.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row["owner"] != owner or row["request_hash"] != digest:
                raise RunConflict("run_id 已存在且不属于同一请求")
        return _row_to_record(row)

    def get(self, run_id, *, owner=None):
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM runs WHERE run_id=? AND owner=?", (run_id,current_owner(owner))).fetchone()
        return _row_to_record(row) if row else None

    def list(self, *, limit=50, owner=None):
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM runs WHERE owner=? ORDER BY updated_at DESC LIMIT ?", (current_owner(owner),limit)).fetchall()
        return [_row_to_record(row) for row in rows]

    def get_sql_backend(self, run_id):
        record = self.get(run_id)
        return record.sql_backend if record else "direct"

    def set_sql_backend(self, run_id, sql_backend):
        with self._connect() as connection:
            connection.execute("UPDATE runs SET sql_backend=?,updated_at=? WHERE run_id=? AND owner=?", (self._normalize_backend(sql_backend),time.time(),run_id,current_owner()))

    def claim(self, run_id, token, *, owner=None, resume=False, ttl=STALE_HEARTBEAT_SECONDS):
        owner = current_owner(owner)
        # Local SQLite runs share a host: proven process death permits immediate
        # recovery. PostgreSQL workers only use expiry, never another host's PID.
        previous = self.get(run_id, owner=owner) if resume else None
        if previous and previous.lease_token and not _process_alive(previous.owner_pid):
            with self._connect() as connection:
                connection.execute("UPDATE runs SET lease_expires_at=0 WHERE run_id=? AND owner=? AND lease_token=? AND owner_pid IS ?",
                    (run_id,owner,previous.lease_token,previous.owner_pid))
        allowed = ("waiting_approval","interrupted","failed","running","queued") if resume else ("pending","queued")
        marks = ",".join("?" for _ in allowed)
        now = time.time()
        with self._connect() as connection:
            cursor = connection.execute(
                f"UPDATE runs SET status='running',lease_token=?,lease_expires_at=?,heartbeat_at=?,owner_pid=?,updated_at=? "
                f"WHERE run_id=? AND owner=? AND status IN ({marks}) AND (lease_token IS NULL OR lease_expires_at<=?)",
                (token,now+ttl,now,os.getpid(),now,run_id,owner,*allowed,now),
            )
            return cursor.rowcount == 1

    def assert_lease(self, run_id, token, *, owner=None):
        record = self.get(run_id, owner=owner)
        if record is None or record.lease_token != token or not record.lease_expires_at or record.lease_expires_at <= time.time():
            raise LeaseLost("运行执行权已失效，拒绝继续提交")

    def touch(self, run_id, *, token=None, owner=None):
        token = token or lease_token_for(run_id)
        now = time.time()
        with self._connect() as connection:
            if token:
                cursor = connection.execute("UPDATE runs SET heartbeat_at=?,updated_at=?,lease_expires_at=? WHERE run_id=? AND owner=? AND lease_token=? AND lease_expires_at>?", (now,now,now+STALE_HEARTBEAT_SECONDS,run_id,current_owner(owner),token,now))
                if cursor.rowcount != 1:
                    raise LeaseLost("运行租约续期失败")
            else:
                connection.execute("UPDATE runs SET heartbeat_at=?,updated_at=?,owner_pid=? WHERE run_id=? AND owner=? AND lease_token IS NULL", (now,now,os.getpid(),run_id,current_owner(owner)))

    def update_status(self, run_id, status, *, error=None, pending_approval=None, owner=None, token=None):
        if status not in RUN_STATUSES:
            raise ValueError(f"未知的运行状态：{status}")
        token = token or lease_token_for(run_id)
        now = time.time()
        where = "run_id=? AND owner=?"
        args = [status,now,error,json.dumps(pending_approval,ensure_ascii=False) if pending_approval else None]
        terminal = status != "running"
        lease_set = ",lease_token=NULL,lease_expires_at=NULL" if terminal else ""
        if status=="running" and not token:
            lease_set += ",owner_pid=?,heartbeat_at=?"
            args += [os.getpid(),now]
        args += [run_id,current_owner(owner)]
        if token:
            where += " AND lease_token=? AND lease_expires_at>?"
            args += [token,now]
        else:
            where += " AND (lease_token IS NULL OR lease_expires_at<=?)"
            args += [now]
        with self._connect() as connection:
            cursor = connection.execute(f"UPDATE runs SET status=?,updated_at=?,error=?,pending_approval=?{lease_set} WHERE {where}", args)
            if token and cursor.rowcount != 1:
                raise LeaseLost("旧执行者不得修改运行状态")

    def is_stale_running(self, run_id):
        record = self.get(run_id)
        if record is None or record.status != "running":
            return False
        if record.lease_token:
            return not _process_alive(record.owner_pid) or not record.lease_expires_at or record.lease_expires_at <= time.time()
        return not _process_alive(record.owner_pid) or record.heartbeat_at is None or time.time()-record.heartbeat_at > STALE_HEARTBEAT_SECONDS

    def record_approval(self, run_id, approval):
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM runs WHERE run_id=? AND owner=?", (run_id,current_owner())).fetchone()
            if row is None:
                raise KeyError("未找到运行")
            approvals = json.loads(row["approvals"] or "[]")
            existing=next((item for item in approvals if item.get("approval_id")==approval.get("approval_id")),None)
            if existing and existing.get("decision") != approval.get("decision"):
                raise RunConflict("同一审批已经提交不同决定")
            if existing is None:
                approvals.append(approval)
            connection.execute("UPDATE runs SET approvals=?,updated_at=? WHERE run_id=? AND owner=?", (json.dumps(approvals,ensure_ascii=False),time.time(),run_id,current_owner()))
            return existing or approval


def _row_to_record(row):
    values=dict(row)
    def get(key,default=None):
        return values.get(key,default)
    return RunRecord(
        run_id=get("run_id"),status=get("status"),data_sources=json.loads(get("data_sources") or "[]"),
        analysis_focus=get("analysis_focus"),created_at=get("created_at",0),updated_at=get("updated_at",0),
        error=get("error"),pending_approval=json.loads(get("pending_approval")) if get("pending_approval") else None,
        approvals=json.loads(get("approvals") or "[]"),owner_pid=get("owner_pid"),heartbeat_at=get("heartbeat_at"),
        sql_backend=get("sql_backend") or "direct",require_plan_approval=bool(get("require_plan_approval",False)),
        require_report_approval=bool(get("require_report_approval",False)),owner=get("owner","local-demo"),request_hash=get("request_hash"),
        request_payload=json.loads(get("request_payload") or "{}"),lease_token=get("lease_token"),lease_expires_at=get("lease_expires_at"),
    )
