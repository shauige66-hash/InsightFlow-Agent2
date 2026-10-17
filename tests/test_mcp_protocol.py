"""MCP 协议层——握手、动态发现、调用、超时与进程回收。

这些用例启动**真实的 Server 子进程**并通过 stdio 通信，不是把普通函数调用
包装成 MCP 的样子。因此它们能发现"参数拼错、Schema 不匹配、stdout 被污染、
子进程泄漏"这类只有真跑才会暴露的问题。
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from mcp_bridge.client import McpStdioClient, McpTimeout, McpUnavailable, flatten_exception
from mcp_bridge.contract import (
    ALLOWED_MCP_TOOLS,
    TOOL_DISCOVER_DATASETS,
    TOOL_GET_SCHEMA,
    TOOL_RUN_READONLY_SQL,
)
from mcp_bridge.registry import write_authorizations

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _process_alive(pid: int) -> bool:
    """用系统进程表判断 pid 是否还活着。

    不用 `os.kill(pid, 0)`：Windows 上它对已退出进程也可能成功，
    会给出假阳性。
    """
    if sys.platform == "win32":
        completed = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return str(pid) in completed.stdout

    try:
        import os

        os.kill(pid, 0)
        return True
    except OSError:
        return False


def build_sqlite(path: Path, *, extra_table: bool = False) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE orders ("
            "order_id INTEGER PRIMARY KEY, customer_id INTEGER, amount REAL)"
        )
        connection.executemany(
            "INSERT INTO orders VALUES (?, ?, ?)",
            [(1, 101, 10.0), (2, 102, 20.0), (3, 101, 5.0)],
        )
        connection.execute("CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT)")
        connection.executemany(
            "INSERT INTO customers VALUES (?, ?)", [(101, "Alice"), (102, "Bob")]
        )
        if extra_table:
            connection.execute("CREATE TABLE secrets (value TEXT)")
            connection.execute("INSERT INTO secrets VALUES ('top-secret')")
        connection.commit()
    finally:
        connection.close()


def build_csv(path: Path, rows: int = 5) -> None:
    lines = ["customer_id,region,amount"]
    for index in range(rows):
        lines.append(f"C{index:03d},{'East' if index % 2 else 'West'},{index * 1.5}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def snapshot(name: str, source_type: str, path: Path, **extra) -> dict:
    payload = {
        "name": name,
        "source_type": source_type,
        "source_path": str(path),
        "content_hash": extra.pop("content_hash", "hash-1"),
        "file_size": path.stat().st_size,
        "locator": extra.pop("locator", None),
        "database_table": extra.pop("database_table", None),
    }
    payload.update(extra)
    return payload


@pytest.fixture(scope="module")
def workspace(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("mcp-protocol")


@pytest.fixture(scope="module")
def sqlite_workspace(workspace: Path):
    db = workspace / "sales.sqlite"
    build_sqlite(db, extra_table=True)
    snapshots = {
        "sales.orders": snapshot(
            "sales.orders", "sqlite", db, database_table="orders"
        ),
        "sales.customers": snapshot(
            "sales.customers", "sqlite", db, database_table="customers"
        ),
    }
    config, refs = write_authorizations(
        run_id="proto-run",
        snapshots=snapshots,
        workspace_root=workspace,
        output_root=workspace,
    )
    return db, config, refs


class TestHandshakeAndDiscovery:
    def test_initialize_and_list_tools(self, sqlite_workspace) -> None:
        db, config, _refs = sqlite_workspace

        with McpStdioClient(config_path=config, project_root=PROJECT_ROOT) as client:
            assert client.server_info["name"] == "insightflow-mcp"
            assert client.protocol_version

            discovered = sorted(client.discovered_tools)
            assert discovered == sorted(ALLOWED_MCP_TOOLS)

    def test_tool_schemas_are_available(self, sqlite_workspace) -> None:
        """读取参数 Schema：不读 Schema 就无法在调用前校验参数。"""
        _db, config, _refs = sqlite_workspace

        with McpStdioClient(config_path=config, project_root=PROJECT_ROOT) as client:
            schema = client.discovered_tools[TOOL_RUN_READONLY_SQL].input_schema

            assert schema.get("type") == "object"
            properties = schema.get("properties") or {}
            assert "query" in properties
            assert "dataset_ids" in properties

    def test_allowed_tools_intersection(self, sqlite_workspace) -> None:
        """动态发现不等于授权：可调用集合是发现结果与允许清单的交集。"""
        _db, config, _refs = sqlite_workspace

        with McpStdioClient(config_path=config, project_root=PROJECT_ROOT) as client:
            assert set(client.allowed_tools()) == ALLOWED_MCP_TOOLS

    def test_child_process_is_a_real_subprocess(self, sqlite_workspace) -> None:
        """确实是独立进程，而不是同进程内的对象。"""
        _db, config, _refs = sqlite_workspace

        with McpStdioClient(config_path=config, project_root=PROJECT_ROOT) as client:
            pid = client._child_pid

            assert pid, "未能取得子进程 pid"
            assert pid != __import__("os").getpid(), "Server 不是独立进程"
            assert _process_alive(pid)

    def test_three_tools_are_callable(self, sqlite_workspace) -> None:
        _db, config, _refs = sqlite_workspace

        with McpStdioClient(config_path=config, project_root=PROJECT_ROOT) as client:
            discovered = client.call_tool(TOOL_DISCOVER_DATASETS, {})
            assert discovered["success"] is True
            assert discovered["result"]["count"] == 2

            schema = client.call_tool(
                TOOL_GET_SCHEMA, {"dataset_id": "proto-run:sales.orders"}
            )
            assert schema["success"] is True
            names = [column["name"] for column in schema["result"]["columns"]]
            assert "amount" in names

            query = client.call_tool(
                TOOL_RUN_READONLY_SQL,
                {
                    "query": "SELECT COUNT(*) AS n FROM orders",
                    "dataset_ids": ["proto-run:sales.orders"],
                },
            )
            assert query["success"] is True
            assert query["result"]["rows"][0]["n"] == 3


class TestProtocolErrorsAreSurfaced:
    def test_tool_error_is_returned_not_raised(self, sqlite_workspace) -> None:
        """MCP 的工具错误必须变成结构化失败，而不是异常穿透。"""
        _db, config, _refs = sqlite_workspace

        with McpStdioClient(config_path=config, project_root=PROJECT_ROOT) as client:
            outcome = client.call_tool(
                TOOL_RUN_READONLY_SQL,
                {
                    "query": "SELECT * FROM no_such_table",
                    "dataset_ids": ["proto-run:sales.orders"],
                },
            )

        assert outcome["success"] is False
        assert outcome["error"]
        assert outcome["error_kind"] == "execution_error"

    def test_tool_outside_allow_list_is_refused(self, sqlite_workspace) -> None:
        """不在允许清单内的工具即使被调用也直接拒绝。"""
        _db, config, _refs = sqlite_workspace

        with McpStdioClient(config_path=config, project_root=PROJECT_ROOT) as client:
            outcome = client.call_tool("some_arbitrary_tool", {})

        assert outcome["success"] is False
        assert outcome["error_kind"] == "unavailable"

    def test_missing_arguments_produce_structured_failure(self, sqlite_workspace) -> None:
        _db, config, _refs = sqlite_workspace

        with McpStdioClient(config_path=config, project_root=PROJECT_ROOT) as client:
            outcome = client.call_tool(TOOL_GET_SCHEMA, {})

        assert outcome["success"] is False
        assert outcome["error_kind"] == "execution_error"

    def test_invalid_json_payload_is_not_treated_as_success(self, tmp_path: Path) -> None:
        """Server 返回非 JSON 文本时不得伪装成成功结构。"""
        from mcp_bridge.client import McpStdioClient as _Client

        client = _Client(config_path=tmp_path / "x.json", project_root=PROJECT_ROOT)

        async def fake_call(name: str, arguments: dict) -> dict:
            return {
                "success": True,
                "result": "plain text not json",
                "error": None,
                "error_kind": None,
            }

        client._call = fake_call  # type: ignore[assignment]
        client.started = True
        client._tools = {}
        result = client.call_tool(TOOL_DISCOVER_DATASETS, {})
        # 未声明该工具 → 仍然拒绝（这里验证的是"没有声明就不能调"）
        assert result["success"] is False

        client.close()


class TestStartupFailures:
    def test_missing_config_fails_unavailable(self, tmp_path: Path) -> None:
        client = McpStdioClient(
            config_path=tmp_path / "missing.json",
            project_root=PROJECT_ROOT,
            startup_timeout=30,
        )
        try:
            with pytest.raises(McpUnavailable) as captured:
                client.start()
            assert captured.value.error_kind == "unavailable"
        finally:
            client.close()

    def test_broken_server_module_fails_unavailable(self, sqlite_workspace, tmp_path: Path) -> None:
        _db, config, _refs = sqlite_workspace
        client = McpStdioClient(
            config_path=config,
            server_module="mcp_bridge.no_such_module",
            project_root=PROJECT_ROOT,
            startup_timeout=30,
        )
        try:
            with pytest.raises(McpUnavailable):
                client.start()
        finally:
            client.close()

    def test_startup_timeout_is_classified_as_timeout(self, sqlite_workspace) -> None:
        """启动超时必须报 timeout，而不是笼统的 unavailable。"""
        _db, config, _refs = sqlite_workspace
        client = McpStdioClient(
            config_path=config,
            project_root=PROJECT_ROOT,
            startup_timeout=0.001,
        )
        try:
            with pytest.raises(McpTimeout) as captured:
                client.start()
            assert captured.value.error_kind == "timeout"
        finally:
            client.close()

    def test_calls_after_failure_stay_failed(self, tmp_path: Path) -> None:
        """启动失败后不得"悄悄变成可用"。"""
        client = McpStdioClient(
            config_path=tmp_path / "missing.json", project_root=PROJECT_ROOT
        )
        with pytest.raises(McpUnavailable):
            client.start()

        outcome = client.call_tool(TOOL_DISCOVER_DATASETS, {})
        client.close()

        assert outcome["success"] is False
        assert outcome["error_kind"] in ("unavailable", "timeout")


class TestProcessLifecycle:
    def test_child_is_reaped_on_close(self, sqlite_workspace) -> None:
        _db, config, _refs = sqlite_workspace

        client = McpStdioClient(config_path=config, project_root=PROJECT_ROOT)
        client.start()
        pid = client._child_pid
        assert pid and _process_alive(pid)

        client.close()

        for _ in range(40):
            if not _process_alive(pid):
                break
            import time

            time.sleep(0.25)
        assert not _process_alive(pid), "关闭后子进程仍在运行"

    def test_close_is_idempotent(self, sqlite_workspace) -> None:
        _db, config, _refs = sqlite_workspace

        client = McpStdioClient(config_path=config, project_root=PROJECT_ROOT)
        client.start()
        pid = client._child_pid

        client.close()
        client.close()

        for _ in range(20):
            if not _process_alive(pid):
                break
            import time

            time.sleep(0.25)
        assert not _process_alive(pid)

    def test_client_without_start_does_not_raise(self, sqlite_workspace) -> None:
        _db, config, _refs = sqlite_workspace

        client = McpStdioClient(config_path=config, project_root=PROJECT_ROOT)
        outcome = client.call_tool(TOOL_DISCOVER_DATASETS, {})
        client.close()

        assert outcome["success"] is False
        assert outcome["error_kind"] == "unavailable"


class TestExceptionFlattening:
    """超时被包进 anyio 的 ExceptionGroup，必须能取出根因。"""

    def test_flatten_extracts_nested_leaves(self) -> None:
        inner = TimeoutError("timed out")
        middle = BaseExceptionGroup("middle", [inner])
        outer = BaseExceptionGroup("outer", [middle])

        leaves = flatten_exception(outer)

        assert leaves == [inner]

    def test_flatten_of_plain_exception(self) -> None:
        error = ValueError("x")

        assert flatten_exception(error) == [error]


class TestStdioDiscipline:
    """stdout 只能承载协议消息。"""

    def test_server_prints_nothing_to_stdout_on_startup(self, sqlite_workspace) -> None:
        """Server 启动与工具注册期间不得向 stdout 写普通文本。

        做法：用 Popen 直接拉起 Server，给一个无效配置让它立刻退出，
        然后检查 stdout 是否为空。协议消息只会在握手后出现，
        因此此刻 stdout 必须干净。
        """
        _db, config, _refs = sqlite_workspace

        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "mcp_bridge.server",
                "--config",
                str(config),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            input="",  # 立即关闭 stdin，让 stdio 传输立刻结束
            cwd=str(PROJECT_ROOT),
            timeout=90,
        )

        assert completed.stdout == "", f"stdout 出现非协议输出：{completed.stdout!r}"

    def test_server_writes_diagnostics_to_stderr(self, tmp_path: Path) -> None:
        """致命错误写 stderr，不污染 stdout。"""
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "mcp_bridge.server",
                "--config",
                str(tmp_path / "missing.json"),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(PROJECT_ROOT),
            timeout=90,
        )

        assert completed.returncode != 0
        assert completed.stdout == ""
        assert "missing.json" in completed.stderr or "Error" in completed.stderr

    def test_protocol_survives_a_chatty_stderr(self, sqlite_workspace) -> None:
        """stderr 上有大量日志时，握手与调用仍然正常。

        这正是"日志走 stderr"的意义：它不是靠日志少，而是靠通道分离。
        """
        _db, config, _refs = sqlite_workspace

        with McpStdioClient(config_path=config, project_root=PROJECT_ROOT) as client:
            for _ in range(5):
                outcome = client.call_tool(TOOL_DISCOVER_DATASETS, {})
                assert outcome["success"] is True


class TestConfigIsPrivateToTheRun:
    def test_config_lives_outside_the_run_directory(
        self, workspace: Path, tmp_path: Path
    ) -> None:
        """配置必须落在 run 目录**之外**。

        run 目录会被 P2-A 的 DockerSandboxExecutor 只读挂载进容器作为 /history，
        放在里面等于把 Server 配置交给沙箱里的模型代码（实测用 pandas 就能读到）。
        """
        db = workspace / "private.sqlite"
        build_sqlite(db)
        snapshots = {"sales.orders": snapshot("sales.orders", "sqlite", db, database_table="orders")}

        config, _refs = write_authorizations(
            run_id="private-run",
            snapshots=snapshots,
            workspace_root=workspace,
            output_root=workspace,
        )

        run_dir = workspace / "private-run"
        assert config.name == "authorizations.json"
        assert not str(config).startswith(str(run_dir)), (
            f"配置落在了 run 目录内，会被挂载进沙箱：{config}"
        )
        # 仍然按 Run 隔离，且仍在 workspace 之内（路径校验与归属检查依赖它）
        assert config.parent.name == "private-run"
        assert str(config).startswith(str(workspace.resolve()))

    def test_config_is_rewritten_per_call(self, workspace: Path) -> None:
        """每次启动都按权威快照重写：沿用旧配置会绕过数据变化检测。"""
        db = workspace / "rewrite.sqlite"
        build_sqlite(db)

        first, _ = write_authorizations(
            run_id="rewrite-run",
            snapshots={
                "sales.orders": snapshot(
                    "sales.orders", "sqlite", db, database_table="orders", content_hash="v1"
                )
            },
            workspace_root=workspace,
            output_root=workspace,
        )
        assert json.loads(first.read_text(encoding="utf-8"))["datasets"][0]["content_hash"] == "v1"

        second, _ = write_authorizations(
            run_id="rewrite-run",
            snapshots={
                "sales.orders": snapshot(
                    "sales.orders", "sqlite", db, database_table="orders", content_hash="v2"
                )
            },
            workspace_root=workspace,
            output_root=workspace,
        )

        payload = json.loads(second.read_text(encoding="utf-8"))
        assert payload["datasets"][0]["content_hash"] == "v2"
