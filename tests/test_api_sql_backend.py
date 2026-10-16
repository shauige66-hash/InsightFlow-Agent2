"""P2-D1：SQL 后端模式经 HTTP API 的传递、持久化与恢复。

覆盖：
- 请求模型接受 direct / mcp，缺省为 direct（兼容既有客户端）。
- 非法模式被拒（400），且请求无法夹带 Server 启动命令、私有目录、环境变量。
- 后端模式写入 RunStore（不是只在内存里），并有非破坏性迁移。
- 恢复时读取 record 里最初保存的模式，而不是恢复进程的默认值。
- direct 不启动 MCP Server；mcp 真的启动并出现 list_tools / call_tool。
- 查询 Run 信息返回 sql_backend，但不泄漏私有配置路径。
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import tools.mcp_sql_tool as mcp_sql_tool
from core.config import Settings
from core.llm import CallContext, ModelFactory
from services.run_coordinator import RunCoordinator
from services.run_store import RunStore

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# SQLite 资源的键是 `<文件名>.<表名>`，因此这里必须点名 sales.orders
SQL_TARGET = "sales.orders"
SQL_CODE = "SELECT COUNT(*) AS n FROM orders"

SQL_REPLIES: dict[str, Any] = {
    "planner": {
        "tasks": [
            {
                "task_id": "task_1",
                "description": "统计订单总数",
                "target_tables": [SQL_TARGET],
                "expected_output": "订单总数",
            }
        ]
    },
    "executor": {"code": SQL_CODE},
    "repair": {"code": SQL_CODE},
    "refine": {"code": SQL_CODE},
    "critic": {"validation_passed": True, "feedback": "ok"},
    "visualizer": {"charts": []},
    "reporter": "# Analysis Report\n\n## Overview\n订单总数 3。\n",
    "visualizer_fallback": {"charts": []},
}


class _SqlScripted(BaseChatModel):
    agent_name: str = ""

    @property
    def _llm_type(self) -> str:
        return f"api-sql-{self.agent_name}"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        reply = SQL_REPLIES.get(self.agent_name, {"tasks": []})
        content = (
            reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)
        )
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content))]
        )


def sql_factory(run_id: str) -> ModelFactory:
    """让任务真正路由到 SQL 工具的假模型工厂。"""

    def resolver(agent_name: str) -> _SqlScripted:
        return _SqlScripted(agent_name=agent_name)

    return ModelFactory(
        settings=Settings(
            api_key="sk-fake",
            base_url="http://127.0.0.1:8000/v1",
            model="deepseek-chat",
        ),
        context=CallContext(run_id=run_id, client_app="insightflow-agent2"),
        resolver=resolver,
    )



def build_sqlite(path: Path) -> None:
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


@pytest.fixture
def api(tmp_path: Path, monkeypatch):
    """真实 FastAPI 路由 + 独立临时运行数据库。"""
    monkeypatch.chdir(tmp_path)
    import api_runs as module

    store = RunStore(tmp_path / "runs.sqlite")
    coordinator = RunCoordinator(
        store,
        output_root=tmp_path,
        checkpointer_path=tmp_path / "checkpoints.sqlite",
        model_factory_builder=sql_factory,
        max_workers=1,
    )

    monkeypatch.setattr(module, "run_store", store)
    monkeypatch.setattr(module, "coordinator", coordinator)
    monkeypatch.setattr(module, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(module, "CHECKPOINT_PATH", tmp_path / "checkpoints.sqlite")

    from api import app

    with TestClient(app) as client:
        yield client, store, tmp_path

    coordinator.shutdown()


def wait_for_status(
    client: TestClient, run_id: str, expected: set[str], timeout: float = 180.0
) -> dict:
    deadline = time.time() + timeout
    last = {}
    while time.time() < deadline:
        body = client.get(f"/runs/{run_id}").json()
        last = body.get("run") or {}
        if last.get("status") in expected:
            return last
        time.sleep(0.4)
    raise AssertionError(f"Run {run_id} 未在 {timeout}s 内进入 {expected}：{last}")


class TestRequestModelAcceptsBackend:
    def test_direct_mode_run(self, api, tmp_path: Path) -> None:
        client, store, root = api
        db = tmp_path / "sales.sqlite"
        build_sqlite(db)

        response = client.post(
            "/runs",
            json={
                "data_sources": [str(db)],
                "analysis_focus": "订单总额",
                "sql_backend": "direct",
                "require_plan_approval": True,
            },
        )

        assert response.status_code == 202, response.text
        body = response.json()
        assert body["sql_backend"] == "direct"
        assert store.get(body["run_id"]).sql_backend == "direct"

    def test_mcp_mode_run(self, api, tmp_path: Path) -> None:
        client, store, root = api
        db = tmp_path / "sales.sqlite"
        build_sqlite(db)

        response = client.post(
            "/runs",
            json={
                "data_sources": [str(db)],
                "sql_backend": "mcp",
                "require_plan_approval": True,
            },
        )

        assert response.status_code == 202, response.text
        body = response.json()
        assert body["sql_backend"] == "mcp"
        assert store.get(body["run_id"]).sql_backend == "mcp"

    def test_unspecified_defaults_to_direct(self, api, tmp_path: Path) -> None:
        """未指定时默认 direct：既有客户端行为不变。"""
        client, store, root = api
        db = tmp_path / "sales.sqlite"
        build_sqlite(db)

        response = client.post(
            "/runs",
            json={"data_sources": [str(db)], "require_plan_approval": True},
        )

        assert response.status_code == 202, response.text
        assert response.json()["sql_backend"] == "direct"
        assert store.get(response.json()["run_id"]).sql_backend == "direct"

    @pytest.mark.parametrize(
        "value",
        ["shell", "", "MCP", "Direct", "direct;rm", "mcp --evil", "../../etc"],
    )
    def test_invalid_backend_is_rejected(self, api, tmp_path: Path, value: str) -> None:
        client, store, root = api
        db = tmp_path / "sales.sqlite"
        build_sqlite(db)

        response = client.post(
            "/runs",
            json={"data_sources": [str(db)], "sql_backend": value},
        )

        assert response.status_code == 400, response.text
        assert "sql_backend" in response.json()["detail"]
        # 未登记任何 Run
        assert store.list() == []

    @pytest.mark.parametrize(
        "value",
        [
            {"command": "evil", "args": ["-c", "steal"]},
            ["mcp", "--config", "C:/secret"],
            123,
        ],
    )
    def test_backend_field_cannot_carry_launch_config(self, api, tmp_path: Path, value) -> None:
        """请求无法借模式字段指定 Server 启动命令、私有目录或宿主路径。"""
        client, store, root = api
        db = tmp_path / "sales.sqlite"
        build_sqlite(db)

        response = client.post(
            "/runs",
            json={"data_sources": [str(db)], "sql_backend": value},
        )

        # 类型不符 → 422；枚举不符 → 400。两者都不接受非字符串模式名。
        assert response.status_code in (400, 422), response.text
        assert store.list() == []

    def test_request_model_has_no_mcp_infrastructure_fields(self) -> None:
        """请求模型里不存在可指定 MCP 基础设施的字段。"""
        from api_runs import CreateRunRequest

        fields = set(CreateRunRequest.model_fields)
        assert "sql_backend" in fields
        for forbidden in (
            "mcp_command",
            "mcp_args",
            "mcp_env",
            "mcp_private_root",
            "private_root",
            "server_module",
            "workspace_root",
            "call_timeout",
            "startup_timeout",
        ):
            assert forbidden not in fields, f"请求模型不应接受 {forbidden}"


class TestBackendPersistence:
    def test_backend_is_in_the_database_not_only_memory(
        self, api, tmp_path: Path
    ) -> None:
        """换一个 RunStore 实例（模拟进程重启）仍能读到模式。"""
        client, store, root = api
        db = tmp_path / "sales.sqlite"
        build_sqlite(db)

        run_id = client.post(
            "/runs",
            json={
                "data_sources": [str(db)],
                "sql_backend": "mcp",
                "require_plan_approval": True,
            },
        ).json()["run_id"]

        # 直接查数据库文件，不经过 API
        connection = sqlite3.connect(str(store._path))
        try:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT sql_backend FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        finally:
            connection.close()

        assert row["sql_backend"] == "mcp"

        # 新实例（新进程会这样做）读出来仍是 mcp
        reopened = RunStore(store._path)
        assert reopened.get(run_id).sql_backend == "mcp"

    def test_run_info_exposes_backend(self, api, tmp_path: Path) -> None:
        client, store, root = api
        db = tmp_path / "sales.sqlite"
        build_sqlite(db)

        run_id = client.post(
            "/runs",
            json={
                "data_sources": [str(db)],
                "sql_backend": "mcp",
                "require_plan_approval": True,
            },
        ).json()["run_id"]

        body = client.get(f"/runs/{run_id}").json()
        assert body["run"]["sql_backend"] == "mcp"
        listed = client.get("/runs").json()["runs"]
        assert listed[0]["sql_backend"] == "mcp"

    def test_run_info_does_not_leak_mcp_private_paths(self, api, tmp_path: Path) -> None:
        """运行信息里不得出现私有配置路径或 Server 内部实现。"""
        client, store, root = api
        db = tmp_path / "sales.sqlite"
        build_sqlite(db)

        run_id = client.post(
            "/runs",
            json={
                "data_sources": [str(db)],
                "sql_backend": "mcp",
                "require_plan_approval": True,
            },
        ).json()["run_id"]

        encoded = json.dumps(client.get(f"/runs/{run_id}").json(), ensure_ascii=False)

        assert "mcp_private" not in encoded
        assert "authorizations" not in encoded
        assert "server_module" not in encoded
        assert "mcp_bridge" not in encoded


class TestResumeUsesStoredBackend:
    def test_resume_keeps_mcp_even_if_caller_sends_direct(
        self, api, tmp_path: Path, monkeypatch
    ) -> None:
        """恢复时后端取自记录，请求体无法把它改回 direct。"""
        client, store, root = api
        db = tmp_path / "sales.sqlite"
        build_sqlite(db)

        run_id = client.post(
            "/runs",
            json={
                "data_sources": [str(db)],
                "analysis_focus": "订单总额",
                "sql_backend": "mcp",
                "require_plan_approval": True,
            },
        ).json()["run_id"]

        wait_for_status(client, run_id, {"waiting_approval"})

        # 恢复时试图指定 direct —— 请求模型里根本没有这个字段
        response = client.post(
            f"/runs/{run_id}/resume",
            json={"resume_value": {"decision": "approve"}, "sql_backend": "direct"},
        )
        assert response.status_code == 202, response.text
        # 响应回显的是记录里的 mcp，说明额外字段被忽略、后端没有被改
        assert response.json()["sql_backend"] == "mcp"
        assert store.get(run_id).sql_backend == "mcp"


class TestRealMcpProcessUsage:
    """不只看响应 JSON：验证真实是否启动了 MCP Server。"""

    def test_direct_mode_never_starts_mcp(self, api, tmp_path: Path, monkeypatch) -> None:
        client, store, root = api
        db = tmp_path / "sales.sqlite"
        build_sqlite(db)

        stats = {"starts": 0, "calls": 0}
        original = mcp_sql_tool.McpStdioClient

        class Counting(original):  # type: ignore[misc, valid-type]
            def start(self):
                stats["starts"] += 1
                return super().start()

            def call_tool(self, name, arguments=None, **kwargs):
                stats["calls"] += 1
                return super().call_tool(name, arguments, **kwargs)

        monkeypatch.setattr(mcp_sql_tool, "McpStdioClient", Counting)

        run_id = client.post(
            "/runs",
            json={
                "data_sources": [str(db)],
                "analysis_focus": "订单总额",
                "sql_backend": "direct",
            },
        ).json()["run_id"]

        wait_for_status(client, run_id, {"completed", "failed"})

        assert stats["starts"] == 0, "direct 模式不应启动 MCP Server"
        assert not any(root.parent.glob(f"{root.name}.mcp_private"))

    def test_mcp_mode_really_starts_and_calls_the_server(
        self, api, tmp_path: Path, monkeypatch
    ) -> None:
        client, store, root = api
        db = tmp_path / "sales.sqlite"
        build_sqlite(db)

        stats = {"starts": 0, "calls": 0, "tools": [], "stderr": ""}
        original = mcp_sql_tool.McpStdioClient

        class Counting(original):  # type: ignore[misc, valid-type]
            def start(self):
                stats["starts"] += 1
                result = super().start()
                stats["tools"] = sorted(self._tools)
                return result

            def call_tool(self, name, arguments=None, **kwargs):
                stats["calls"] += 1
                stats["tools"].append(name)
                return super().call_tool(name, arguments, **kwargs)

        monkeypatch.setattr(mcp_sql_tool, "McpStdioClient", Counting)

        run_id = client.post(
            "/runs",
            json={
                "data_sources": [str(db)],
                "analysis_focus": "订单总额",
                "sql_backend": "mcp",
            },
        ).json()["run_id"]

        wait_for_status(client, run_id, {"completed", "failed"}, timeout=240)

        assert stats["starts"] >= 1, "mcp 模式必须真实启动 MCP Server"
        assert stats["calls"] >= 1, "mcp 模式必须真实发生 call_tool"
        assert "run_readonly_sql" in stats["tools"]

        # 私有配置写在 output_root 之外
        private_config = root.parent / f"{root.name}.mcp_private" / run_id / "authorizations.json"
        assert private_config.exists()
        payload = json.loads(private_config.read_text(encoding="utf-8"))
        assert all(
            item["dataset_id"].startswith(f"{run_id}:") for item in payload["datasets"]
        )
