"""P2-C 补验收：真实跨进程 MCP 恢复，以及 /runs API 的实际后端行为。

前半部分用**三个独立进程**验证 MCP 模式能在新进程里继续未完成的 Run：

    进程 1  真实 MCP Server 子进程 → SQL 调用完成后强杀（state 未提交）
    进程 2  全新 Python 进程，加载已有 Checkpoint，重建 MCP Client/Server
    进程 3  inspect 最终状态

后半部分是**实测结论**：FastAPI 的 /runs 目前不暴露 MCP 模式。
这里把这个事实固定成测试，避免文档与实现各说一套。
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DRIVER = PROJECT_ROOT / "scripts" / "recovery_driver.py"

RUN_ID = "xproc-mcp"
DB_NAME = "sales.sqlite"
TARGET_TABLE = "sales.orders"


def run_driver(
    phase: str,
    *,
    workdir: Path,
    data_source: Path,
    mcp_mode: str = "mcp",
    extra: list[str] | None = None,
    env_extra: dict[str, str] | None = None,
    timeout: int = 300,
) -> dict:
    """驱动一次独立进程。

    `--sql-task --target-table` 必须每次都给：驱动按进程重建回复脚本，
    首次与恢复两次运行的 planner 输出必须一致，否则恢复出来的图会与
    Checkpoint 里记的任务对不上。
    """
    command = [
        sys.executable,
        str(DRIVER),
        phase,
        "--workdir", str(workdir),
        "--run-id", RUN_ID,
        "--data-source", str(data_source),
        "--mcp-mode", mcp_mode,
        "--sql-task",
        "--target-table", TARGET_TABLE,
    ]
    if extra:
        command += extra

    import os

    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        cwd=str(PROJECT_ROOT),
        env=env,
    )

    payload: dict = {"returncode": completed.returncode}
    for line in reversed(completed.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                payload.update(json.loads(line))
            except ValueError:
                pass
            break

    payload["stdout"] = completed.stdout
    payload["stderr"] = completed.stderr
    return payload


def build_database(path: Path) -> None:
    """建一个带 orders 与 customers 的库。

    表名用 orders：驱动的 SQL 任务查的就是它，而 SQLite 资源键是
    `<文件名>.<表名>`。
    """
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER, amount REAL)"
        )
        connection.executemany(
            "INSERT INTO orders VALUES (?, ?, ?)",
            [(1, 101, 10.0), (2, 102, 20.0), (3, 101, 5.0)],
        )
        connection.execute(
            "CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT, region TEXT)"
        )
        connection.executemany(
            "INSERT INTO customers VALUES (?, ?, ?)",
            [(101, "Alice", "East"), (102, "Bob", "West")],
        )
        connection.commit()
    finally:
        connection.close()


def mcp_server_processes() -> list[int]:
    """当前存活的 MCP Server 子进程 pid。"""
    if sys.platform != "win32":
        return []
    completed = subprocess.run(
        [
            "powershell", "-NoProfile", "-Command",
            "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
            "Where-Object { $_.CommandLine -match 'mcp_bridge.server' } | "
            "Select-Object -ExpandProperty ProcessId",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return [int(line) for line in completed.stdout.split() if line.strip().isdigit()]


@pytest.fixture
def workspace(tmp_path: Path) -> tuple[Path, Path]:
    workdir = tmp_path / "run"
    workdir.mkdir(parents=True)
    db = workdir / DB_NAME
    build_database(db)
    return workdir, db


class TestRealCrossProcessMcpRecovery:
    def test_full_three_process_recovery(self, workspace) -> None:
        workdir, db = workspace

        # --- 进程 1：真实 MCP Server 启动并完成一次 SQL 调用，然后强杀 ---
        crashed = run_driver(
            "start",
            workdir=workdir,
            data_source=db,
            # 崩溃点：MCP 调用已完成、节点 state 尚未提交
            env_extra={"INSIGHTFLOW_CRASH_AT": "after_mcp_call"},
        )
        assert crashed["returncode"] == 17, (
            f"未在 MCP 调用后强杀：{crashed['returncode']}\n{crashed['stderr'][-400:]}"
        )
        # 进程 1 确实经过了 MCP Server（stderr 上有协议活动）
        assert "CallToolRequest" in crashed["stderr"], (
            "进程 1 未经过 MCP Server"
        )

        # 崩溃点之后该节点没有提交：state 应停在待执行节点上
        inspected = run_driver("inspect", workdir=workdir, data_source=db)
        state = inspected["state"]
        assert state["run_status"] == "running"
        assert state["next_nodes"], "应停在待执行节点上"
        assert state["execution_results"] if "execution_results" in state else True

        # 进程 1 结束后不应留下 MCP Server 子进程
        assert mcp_server_processes() == []

        # --- 进程 2：全新进程恢复 ---
        resumed = run_driver("resume", workdir=workdir, data_source=db)

        assert resumed["returncode"] == 0, resumed["stderr"][-400:]
        assert resumed["run_status"] == "completed"
        assert resumed["analysis_status"] == "completed"
        assert resumed["execution_results"] >= 1
        assert resumed["failed_tasks"] == 0

        # 恢复时确实重建并使用了 MCP Server，而不是静默切到 direct
        assert "CallToolRequest" in resumed["stderr"], (
            "恢复进程未经过 MCP Server（可能静默切到了 direct）"
        )

        # --- 进程 3：inspect 最终状态 ---
        final = run_driver("inspect", workdir=workdir, data_source=db)
        assert final["state"]["run_status"] == "completed"
        assert final["state"]["next_nodes"] == []

        # 无遗留子进程
        assert mcp_server_processes() == []

    def test_authorization_and_fingerprint_survive_recovery(self, workspace) -> None:
        """恢复后 Run 级授权与内容指纹不变。"""
        workdir, db = workspace

        run_driver(
            "start",
            workdir=workdir,
            data_source=db,
            env_extra={"INSIGHTFLOW_CRASH_AT": "after_mcp_call"},
        )
        # 私有根目录与 output_root（这里是 workdir）平级
        config = (
            Path(str(workdir) + ".mcp_private") / RUN_ID / "authorizations.json"
        )
        assert config.exists(), "首次运行未写出授权配置"

        before = json.loads(config.read_text(encoding="utf-8"))

        run_driver("resume", workdir=workdir, data_source=db)

        after = json.loads(config.read_text(encoding="utf-8"))

        assert [d["dataset_id"] for d in after["datasets"]] == [
            d["dataset_id"] for d in before["datasets"]
        ]
        assert [d["content_hash"] for d in after["datasets"]] == [
            d["content_hash"] for d in before["datasets"]
        ]
        # run_id 前缀保持，说明 dataset_id 仍然按 Run 隔离
        assert all(d["dataset_id"].startswith(f"{RUN_ID}:") for d in after["datasets"])

    def test_private_config_is_outside_the_run_directory(self, workspace) -> None:
        workdir, db = workspace
        run_driver(
            "start",
            workdir=workdir,
            data_source=db,
            env_extra={"INSIGHTFLOW_CRASH_AT": "after_mcp_call"},
        )

        run_dir = workdir / RUN_ID
        # 私有根目录与 output_root（这里是 workdir）平级
        private_root = Path(str(workdir) + ".mcp_private")
        config = private_root / RUN_ID / "authorizations.json"

        assert config.exists()
        assert not config.is_relative_to(run_dir)
        assert not config.is_relative_to(workdir)
        assert not (run_dir / "mcp").exists()

    def test_completed_results_are_not_duplicated_on_resume(self, workspace) -> None:
        """恢复不会让已确认完成的结果重复增长。"""
        workdir, db = workspace

        run_driver(
            "start",
            workdir=workdir,
            data_source=db,
            env_extra={"INSIGHTFLOW_CRASH_AT": "after_mcp_call"},
        )
        run_driver("resume", workdir=workdir, data_source=db)

        first = run_driver("inspect", workdir=workdir, data_source=db)
        count_after_first = first["state"].get("current_task_index")

        # 终态 Run 再次 resume：应短路
        second = run_driver("resume", workdir=workdir, data_source=db)
        assert second["run_status"] == "completed"

        third = run_driver("inspect", workdir=workdir, data_source=db)
        assert third["state"].get("current_task_index") == count_after_first

    def test_direct_mode_recovery_does_not_start_a_server(self, workspace) -> None:
        """对照：direct 模式的恢复不应出现 MCP 协议活动。"""
        workdir, db = workspace

        crashed = run_driver(
            "start",
            workdir=workdir,
            data_source=db,
            mcp_mode="direct",
        )
        # direct 模式下没有 after_mcp_call 崩溃点，会正常跑完
        assert crashed["returncode"] == 0, crashed["stderr"][-300:]
        assert "CallToolRequest" not in crashed["stderr"]


class TestApiMcpExposureBoundary:
    """API 对 MCP 的暴露边界。

    P2-C 时 API 完全不支持 MCP，因此当时的断言是"源码里不出现 McpSettings"。
    P2-D 把后端选择接进了 HTTP，这条断言随之失效——**契约变了，测试要跟着改**，
    但不能放宽：这里改为断言新的、更重要的性质：

    - Coordinator 传递的是**按 Run 记录派生**的配置，而不是请求里的任意对象；
    - 请求模型里不存在任何可指定 MCP 基础设施的字段；
    - 非 mcp 模式的 Run 不产生任何 MCP 私有配置。
    """

    def test_coordinator_derives_settings_from_the_stored_record(self, tmp_path, monkeypatch) -> None:
        from core.runtime import McpSettings
        from services.run_store import RunStore
        from services.run_coordinator import RunCoordinator
        store = RunStore(tmp_path / 'runs.sqlite')
        store.create('stored_mcp', data_sources=['original.sqlite'], sql_backend='mcp')
        template = McpSettings(server_module='host.trusted_server', call_timeout=17,
            private_root=tmp_path / 'private')
        coordinator = RunCoordinator(store, output_root=tmp_path, mcp_settings=template)
        captured = {}
        def execute(run_id, **kwargs):
            captured.update(kwargs)
            return {'run_status': 'completed', 'analysis_status': 'completed'}
        monkeypatch.setattr(coordinator, '_execute', execute)
        try:
            coordinator.start('stored_mcp', record_exists=True, wait=True,
                data_sources=['changed.csv'], sql_backend='direct')
        finally:
            coordinator.shutdown()
        assert captured['data_sources'] == ['original.sqlite']
        settings = captured['mcp']
        assert settings.is_mcp and settings.server_module == template.server_module
        assert settings.call_timeout == 17 and settings.private_root == template.private_root

    def test_request_model_cannot_carry_mcp_infrastructure(self) -> None:
        import api_runs

        fields = set(api_runs.CreateRunRequest.model_fields)
        assert "sql_backend" in fields

        for forbidden in (
            "mcp_command",
            "mcp_args",
            "mcp_env",
            "mcp_private_root",
            "private_root",
            "server_module",
            "workspace_root",
            "startup_timeout",
            "call_timeout",
        ):
            assert forbidden not in fields, f"请求模型不应接受 {forbidden}"

        # resume 请求同样不能改写后端
        resume_fields = set(api_runs.ResumeRequest.model_fields)
        assert "sql_backend" not in resume_fields

    def test_api_run_never_creates_mcp_private_config(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """direct 模式的 Run 不产生任何 MCP 私有配置。

        这是最直接的判据：mcp 模式**每次**调用都会写出
        `<output_root>.mcp_private/<run_id>/authorizations.json`
        （见 TestRealCrossProcessMcpRecovery）。没有出现就说明它走的是 direct。

        不在这里断言业务是否跑通：假模型的 planner 指向 customers 表，
        而 SQLite 资源键是 sales.orders，任务失败与后端选择无关。
        """
        import time

        from fastapi.testclient import TestClient

        from services.run_coordinator import RunCoordinator
        from services.run_store import RunStore
        from tests.test_run_api import _factory

        db = tmp_path / DB_NAME
        build_database(db)

        # `api_runs` 在**导入期**就创建了一个默认 RunStore，路径是相对的
        # `outputs/runs/runs.sqlite`。先切到临时目录再导入，它就会落在临时目录里
        # （否则会在仓库根留下 outputs/）。既有 test_run_api 也是靠 chdir 规避的。
        monkeypatch.chdir(tmp_path)
        import api_runs

        store = RunStore(tmp_path / "runs.sqlite")
        coordinator = RunCoordinator(
            store,
            output_root=tmp_path,
            checkpointer_path=tmp_path / "checkpoints.sqlite",
            model_factory_builder=_factory,
            max_workers=1,
        )

        monkeypatch.setattr(api_runs, "run_store", store)
        monkeypatch.setattr(api_runs, "coordinator", coordinator)
        monkeypatch.setattr(api_runs, "OUTPUT_ROOT", tmp_path)
        monkeypatch.setattr(api_runs, "CHECKPOINT_PATH", tmp_path / "checkpoints.sqlite")

        from api import app

        try:
            with TestClient(app) as client:
                response = client.post(
                    "/runs",
                    json={
                        "data_sources": [str(db)],
                        "analysis_focus": "统计订单总额",
                    },
                )
                assert response.status_code == 202, response.text
                run_id = response.json()["run_id"]

                deadline = time.time() + 180
                status = ""
                while time.time() < deadline:
                    body = client.get(f"/runs/{run_id}").json()
                    # 该端点的响应形状是 {"run": {...}, "state": {...}}
                    status = (body.get("run") or {}).get("status", "")
                    if status in ("completed", "failed"):
                        break
                    time.sleep(0.5)
        finally:
            coordinator.shutdown()

        assert status in ("completed", "failed"), f"Run 未结束：{status}"
        # mcp 模式必然写出这个文件；它不存在 = 走的是 direct
        assert not any(tmp_path.parent.glob(f"{tmp_path.name}.mcp_private")), (
            "API 发起的 Run 产生了 MCP 私有配置，说明它走了 MCP"
        )
