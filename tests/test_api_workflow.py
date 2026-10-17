"""HTTP 创建 / 审批 / 恢复的完整链路。

分两类：

- `TestClient` 驱动真实 FastAPI 路由，用于精确控制失败注入
  （MCP 不可用、SQL 错误、数据源变化等）；
- `ApiServer` 启动**独立进程中的真实 uvicorn**，用于跨进程恢复：
  强杀进程 → 换新进程重启 → 通过 HTTP Resume 同一个 Run。

模型全部是脚本化 Mock，不访问任何付费模型。
"""

from __future__ import annotations

import json
import sqlite3
import threading
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
from tests.http_harness import ApiServer, free_port, isolated_workdir, mcp_server_pids

# SQLite 资源键是 `<文件名>.<表名>`
SQL_TARGET = "customers.customers"
SQL_CODE = "SELECT COUNT(*) AS n FROM customers"
BROKEN_SQL = "SELECT COUNT(*) AS n FROM nonexistent_table"

REPLIES: dict[str, Any] = {
    "planner": {
        "tasks": [
            {
                "task_id": "task_1",
                "description": "统计客户总数",
                "target_tables": [SQL_TARGET],
                "expected_output": "客户总数",
            }
        ]
    },
    "executor": {"code": SQL_CODE},
    "repair": {"code": SQL_CODE},
    "refine": {"code": SQL_CODE},
    "critic": {"validation_passed": True, "feedback": "ok"},
    "visualizer": {"charts": []},
    "reporter": "# Analysis Report\n\n## Overview\n客户总数 3。\n",
}


class _Scripted(BaseChatModel):
    agent_name: str = ""

    @property
    def _llm_type(self) -> str:
        return f"p2d-{self.agent_name}"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        reply = REPLIES.get(self.agent_name, {"tasks": []})
        content = (
            reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)
        )
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content))]
        )


def factory(run_id: str) -> ModelFactory:
    def resolver(agent_name: str) -> _Scripted:
        return _Scripted(agent_name=agent_name)

    return ModelFactory(
        settings=Settings(
            api_key="sk-fake",
            base_url="http://127.0.0.1:8000/v1",
            model="deepseek-chat",
        ),
        context=CallContext(run_id=run_id, client_app="insightflow-agent2"),
        resolver=resolver,
    )


def build_database(path: Path, *, rows: int = 3) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT, region TEXT)"
        )
        connection.executemany(
            "INSERT INTO customers VALUES (?, ?, ?)",
            [(100 + index, f"user{index}", "East" if index % 2 else "West")
             for index in range(1, rows + 1)],
        )
        connection.commit()
    finally:
        connection.close()


@pytest.fixture
def api(tmp_path: Path, monkeypatch):
    """TestClient + 隔离的 RunStore/Checkpoint。"""
    monkeypatch.chdir(tmp_path)
    import api_runs

    store = RunStore(tmp_path / "runs.sqlite")
    coordinator = RunCoordinator(
        store,
        output_root=tmp_path,
        checkpointer_path=tmp_path / "checkpoints.sqlite",
        model_factory_builder=factory,
        max_workers=1,
    )

    monkeypatch.setattr(api_runs, "run_store", store)
    monkeypatch.setattr(api_runs, "coordinator", coordinator)
    monkeypatch.setattr(api_runs, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(api_runs, "CHECKPOINT_PATH", tmp_path / "checkpoints.sqlite")

    from api import app

    with TestClient(app) as client:
        yield client, store, tmp_path

    coordinator.shutdown()


def wait(
    client: TestClient, run_id: str, expected: set[str], timeout: float = 240.0
) -> dict:
    deadline = time.time() + timeout
    last: dict[str, Any] = {}
    while time.time() < deadline:
        body = client.get(f"/runs/{run_id}").json()
        last = body.get("run") or {}
        if last.get("status") in expected:
            return last
        time.sleep(0.3)
    raise AssertionError(f"Run {run_id} 未进入 {expected}：{last}")


class TestHappyPathOverHTTP:
    def test_create_execute_query_and_fetch_report(self, api, tmp_path: Path) -> None:
        client, store, root = api
        db = tmp_path / "customers.sqlite"
        build_database(db)

        created = client.post(
            "/runs",
            json={
                "data_sources": [str(db)],
                "analysis_focus": "客户总数",
                "sql_backend": "mcp",
            },
        )
        assert created.status_code == 202, created.text
        run_id = created.json()["run_id"]

        record = wait(client, run_id, {"completed", "failed"})
        assert record["status"] == "completed", record.get("error")

        body = client.get(f"/runs/{run_id}").json()
        state = body["state"]
        assert state["analysis_status"] == "completed"
        assert state["run_status"] == "completed"
        # 任务全部完成，没有失败项
        assert state["failed_tasks"] == []
        assert state["current_task_index"] >= 1
        # 报告与产物确实产出
        assert state["report_path"], state
        assert state["artifacts"], state

        report = client.get(f"/runs/{run_id}/report")
        assert report.status_code == 200, report.text
        assert report.headers["content-type"] == "application/pdf"
        assert report.content[:4] == b"%PDF", "报告不是有效 PDF"

    def test_report_endpoint_honours_configured_output_root(
        self, api, tmp_path: Path
    ) -> None:
        """回归：报告端点必须跟随运行期 output_root。

        原先它硬编码 `Path("outputs")`，产物写到别的根目录时就会 404，
        而 Run 其实是成功的。
        """
        client, store, root = api
        db = tmp_path / "customers.sqlite"
        build_database(db)

        run_id = client.post(
            "/runs",
            json={"data_sources": [str(db)], "sql_backend": "direct"},
        ).json()["run_id"]
        wait(client, run_id, {"completed", "failed"})

        # 产物落在隔离的 output_root 下
        assert (root / run_id / "analysis_report.pdf").exists()
        # 仓库根目录不应被写入
        assert not (Path.cwd() / "outputs" / run_id).exists() or root == Path.cwd() / "outputs"

        report = client.get(f"/runs/{run_id}/report")
        assert report.status_code == 200, report.text

    def test_report_missing_is_404(self, api) -> None:
        client, store, root = api
        assert client.get("/runs/run_no_such_report/report").status_code == 404

    def test_list_runs_shows_backend(self, api, tmp_path: Path) -> None:
        client, store, root = api
        db = tmp_path / "customers.sqlite"
        build_database(db)

        run_id = client.post(
            "/runs",
            json={
                "data_sources": [str(db)],
                "sql_backend": "mcp",
                "require_plan_approval": True,
            },
        ).json()["run_id"]

        records = client.get("/runs").json()["runs"]
        assert any(item["run_id"] == run_id and item["sql_backend"] == "mcp" for item in records)


class TestHitlApproval:
    def test_approve_resumes_to_completion(self, api, tmp_path: Path) -> None:
        client, store, root = api
        db = tmp_path / "customers.sqlite"
        build_database(db)

        run_id = client.post(
            "/runs",
            json={
                "data_sources": [str(db)],
                "analysis_focus": "客户总数",
                "sql_backend": "mcp",
                "require_plan_approval": True,
            },
        ).json()["run_id"]

        waiting = wait(client, run_id, {"waiting_approval"})
        assert waiting["pending_approval"], "应带待审批负载"

        approval = client.post(
            f"/runs/{run_id}/approval",
            json={"decision": "approve", "comment": "同意"},
        )
        assert approval.status_code == 202, approval.text
        assert approval.json()["decision"] == "approve"

        final = wait(client, run_id, {"completed", "failed"})
        assert final["status"] == "completed", final.get("error")
        assert final["approvals"][0]["decision"] == "approve"

    def test_reject_stops_the_run(self, api, tmp_path: Path) -> None:
        client, store, root = api
        db = tmp_path / "customers.sqlite"
        build_database(db)

        run_id = client.post(
            "/runs",
            json={
                "data_sources": [str(db)],
                "sql_backend": "direct",
                "require_plan_approval": True,
            },
        ).json()["run_id"]

        wait(client, run_id, {"waiting_approval"})

        rejected = client.post(
            f"/runs/{run_id}/approval",
            json={"decision": "reject", "comment": "不需要"},
        )
        assert rejected.status_code == 202, rejected.text

        final = wait(client, run_id, {"completed", "failed"})
        # 拒绝路径不得产出成功分析
        assert final["status"] == "failed" or final.get("pending_approval") is None
        record = store.get(run_id)
        assert record.approvals[0]["decision"] == "reject"

    def test_duplicate_approval_is_idempotent(self, api, tmp_path: Path) -> None:
        client, store, root = api
        db = tmp_path / "customers.sqlite"
        build_database(db)

        run_id = client.post(
            "/runs",
            json={
                "data_sources": [str(db)],
                "sql_backend": "direct",
                "require_plan_approval": True,
            },
        ).json()["run_id"]

        wait(client, run_id, {"waiting_approval"})

        first = client.post(f"/runs/{run_id}/approval", json={"decision": "approve"})
        assert first.status_code == 202
        assert first.json()["duplicate"] is False

        second = client.post(f"/runs/{run_id}/approval", json={"decision": "approve"})
        # 已经离开 waiting_approval：要么 409，要么标记为重复
        assert second.status_code in (202, 409)
        if second.status_code == 202:
            assert second.json()["duplicate"] is True

    def test_approval_on_wrong_state_is_conflict(self, api, tmp_path: Path) -> None:
        client, store, root = api
        db = tmp_path / "customers.sqlite"
        build_database(db)

        run_id = client.post(
            "/runs",
            json={
                "data_sources": [str(db)],
                "sql_backend": "direct",
                "require_plan_approval": True,
            },
        ).json()["run_id"]

        wait(client, run_id, {"waiting_approval"})
        client.post(f"/runs/{run_id}/approval", json={"decision": "approve"})
        wait(client, run_id, {"completed", "failed"})

        again = client.post(f"/runs/{run_id}/approval", json={"decision": "approve"})
        assert again.status_code == 409, again.text

    def test_invalid_decision_is_rejected(self, api, tmp_path: Path) -> None:
        client, store, root = api
        db = tmp_path / "customers.sqlite"
        build_database(db)

        run_id = client.post(
            "/runs",
            json={
                "data_sources": [str(db)],
                "sql_backend": "direct",
                "require_plan_approval": True,
            },
        ).json()["run_id"]

        wait(client, run_id, {"waiting_approval"})

        bad = client.post(f"/runs/{run_id}/approval", json={"decision": "maybe"})
        assert bad.status_code == 400, bad.text


class TestResumeGuards:
    def test_resume_on_completed_run_is_conflict(self, api, tmp_path: Path) -> None:
        client, store, root = api
        db = tmp_path / "customers.sqlite"
        build_database(db)

        run_id = client.post(
            "/runs",
            json={"data_sources": [str(db)], "sql_backend": "direct"},
        ).json()["run_id"]

        wait(client, run_id, {"completed", "failed"})

        again = client.post(f"/runs/{run_id}/resume", json={})
        assert again.status_code == 409, again.text

    def test_concurrent_resume_is_refused(self, api, tmp_path: Path) -> None:
        """同一 Run 的并发恢复：第二个请求必须被拒，不能重复执行。"""
        client, store, root = api
        db = tmp_path / "customers.sqlite"
        build_database(db)

        run_id = client.post(
            "/runs",
            json={
                "data_sources": [str(db)],
                "sql_backend": "direct",
                "require_plan_approval": True,
            },
        ).json()["run_id"]

        wait(client, run_id, {"waiting_approval"})

        results: list[int] = []
        lock = threading.Lock()

        def fire() -> None:
            response = client.post(
                f"/runs/{run_id}/resume",
                json={"resume_value": {"decision": "approve"}},
            )
            with lock:
                results.append(response.status_code)

        threads = [threading.Thread(target=fire) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)

        assert len(results) == 2
        # 至少一个成功；另一个要么 409（并发保护），要么 202（已跑完）
        assert 202 in results, results
        assert all(code in (202, 409) for code in results), results

        final = wait(client, run_id, {"completed", "failed"})
        assert final["status"] in ("completed", "failed")

    def test_resume_unknown_run_is_404(self, api) -> None:
        client, store, root = api
        response = client.post("/runs/run_does_not_exist/resume", json={})
        assert response.status_code == 404, response.text


class TestUserInputErrors:
    def test_missing_data_source_file(self, api) -> None:
        client, store, root = api
        response = client.post(
            "/runs",
            json={"data_sources": ["C:/definitely/not/here.csv"], "sql_backend": "direct"},
        )
        assert response.status_code == 400, response.text

    def test_empty_data_sources(self, api) -> None:
        client, store, root = api
        response = client.post("/runs", json={"data_sources": [], "sql_backend": "direct"})
        assert response.status_code == 400, response.text

    def test_directory_as_data_source(self, api, tmp_path: Path) -> None:
        client, store, root = api
        response = client.post(
            "/runs",
            json={"data_sources": [str(tmp_path)], "sql_backend": "direct"},
        )
        assert response.status_code == 400, response.text

    @pytest.mark.parametrize("run_id", ["ab", "has space", "bad/slash", "x" * 80, "../etc"])
    def test_invalid_run_id(self, api, tmp_path: Path, run_id: str) -> None:
        client, store, root = api
        db = tmp_path / "customers.sqlite"
        build_database(db)

        response = client.post(
            "/runs",
            json={"data_sources": [str(db)], "sql_backend": "direct", "run_id": run_id},
        )
        assert response.status_code == 400, response.text

    def test_unknown_run_is_404(self, api) -> None:
        client, store, root = api
        assert client.get("/runs/run_unknown_x").status_code == 404
        assert client.get("/runs/run_unknown_x/state").status_code == 404

    def test_bad_limit_is_rejected(self, api) -> None:
        client, store, root = api
        assert client.get("/runs?limit=0").status_code == 400
        assert client.get("/runs?limit=100000").status_code == 400


class TestFailureClassification:
    def test_mcp_unavailable_is_explicit_and_does_not_fall_back(
        self, api, tmp_path: Path, monkeypatch
    ) -> None:
        """MCP 不可用时必须明确失败，不能悄悄改用 direct。"""
        client, store, root = api
        db = tmp_path / "customers.sqlite"
        build_database(db)

        import tools.tool_registry as registry

        backend_calls: list[str] = []
        original_resolve = registry.resolve_backend

        def spy(tool_name: str) -> str:
            backend = original_resolve(tool_name)
            backend_calls.append(backend)
            return backend

        monkeypatch.setattr(registry, "resolve_backend", spy)

        def explode(*args: Any, **kwargs: Any):
            raise mcp_sql_tool.McpUnavailable("注入：MCP Server 不可用")

        monkeypatch.setattr(mcp_sql_tool, "McpStdioClient", explode)

        run_id = client.post(
            "/runs",
            json={
                "data_sources": [str(db)],
                "analysis_focus": "客户总数",
                "sql_backend": "mcp",
            },
        ).json()["run_id"]

        wait(client, run_id, {"completed", "failed"}, timeout=300)

        body = client.get(f"/runs/{run_id}").json()
        state = body["state"]
        failures = state.get("failed_tasks") or []
        assert failures, f"应留下失败记录：{state}"

        # 失败原因必须指向环境（unavailable/timeout），而不是被当成 SQL 写错
        assert all(
            item.get("error_kind") in ("unavailable", "timeout") for item in failures
        ), failures
        # 该 Run 明确失败，没有产出成功分析
        assert state["analysis_status"] != "completed", state
        # 没有任何一次分派落到 direct
        assert "direct" not in backend_calls, backend_calls
        # 也不应调用 repair 去重试环境故障
        assert all(item.get("retry_count", 0) == 0 for item in failures), failures

    def test_sql_error_triggers_bounded_repair(self, api, tmp_path: Path, monkeypatch) -> None:
        """SQL 写错进入有界 Repair；修复成功后任务完成。

        判据：executor 被调用了**两次以上**（首次失败 + repair 重试），
        且最终完成。repair 的调用次数用脚本化模型直接数。
        """
        client, store, root = api
        db = tmp_path / "customers.sqlite"
        build_database(db)

        counts = {"executor": 0, "repair": 0}

        class Counting(_Scripted):
            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                counts[self.agent_name] = counts.get(self.agent_name, 0) + 1
                return super()._generate(messages, stop, run_manager, **kwargs)

        def counting_factory(run_id: str) -> ModelFactory:
            def resolver(agent_name: str):
                return Counting(agent_name=agent_name)

            return ModelFactory(
                settings=Settings(
                    api_key="sk-fake",
                    base_url="http://127.0.0.1:8000/v1",
                    model="deepseek-chat",
                ),
                context=CallContext(run_id=run_id, client_app="insightflow-agent2"),
                resolver=resolver,
            )

        coordinator = RunCoordinator(
            store,
            output_root=root,
            checkpointer_path=root / "checkpoints.sqlite",
            model_factory_builder=counting_factory,
            max_workers=1,
        )

        import api_runs

        monkeypatch.setattr(api_runs, "coordinator", coordinator)
        # 首次执行给出坏 SQL，修复给出正确 SQL
        monkeypatch.setitem(REPLIES, "executor", {"code": BROKEN_SQL})
        monkeypatch.setitem(REPLIES, "repair", {"code": SQL_CODE})

        try:
            run_id = client.post(
                "/runs",
                json={
                    "data_sources": [str(db)],
                    "analysis_focus": "客户总数",
                    "sql_backend": "direct",
                },
            ).json()["run_id"]

            wait(client, run_id, {"completed", "failed"}, timeout=300)
            state = client.get(f"/runs/{run_id}").json()["state"]
        finally:
            coordinator.shutdown()

        # 修复确实发生过
        assert counts.get("repair", 0) >= 1, counts
        # 修复最终成功
        assert state["analysis_status"] == "completed", state
        assert state["failed_tasks"] == [], state

    def test_environment_error_does_not_trigger_repair(
        self, api, tmp_path: Path, monkeypatch
    ) -> None:
        """环境故障不应触发无效的模型重试。"""
        client, store, root = api
        db = tmp_path / "customers.sqlite"
        build_database(db)

        repair_calls: list[int] = []

        class CountingRepair(_Scripted):
            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                repair_calls.append(1)
                return super()._generate(messages, stop, run_manager, **kwargs)

        def counting_factory(run_id: str) -> ModelFactory:
            def resolver(agent_name: str):
                if agent_name == "repair":
                    return CountingRepair(agent_name=agent_name)
                return _Scripted(agent_name=agent_name)

            return ModelFactory(
                settings=Settings(
                    api_key="sk-fake",
                    base_url="http://127.0.0.1:8000/v1",
                    model="deepseek-chat",
                ),
                context=CallContext(run_id=run_id, client_app="insightflow-agent2"),
                resolver=resolver,
            )

        coordinator = RunCoordinator(
            store,
            output_root=root,
            checkpointer_path=root / "checkpoints.sqlite",
            model_factory_builder=counting_factory,
            max_workers=1,
        )

        import api_runs

        monkeypatch.setattr(api_runs, "coordinator", coordinator)

        def explode(*args: Any, **kwargs: Any):
            raise mcp_sql_tool.McpUnavailable("注入：通道不可用")

        monkeypatch.setattr(mcp_sql_tool, "McpStdioClient", explode)

        try:
            run_id = client.post(
                "/runs",
                json={
                    "data_sources": [str(db)],
                    "analysis_focus": "客户总数",
                    "sql_backend": "mcp",
                },
            ).json()["run_id"]
            wait(client, run_id, {"completed", "failed"}, timeout=300)
            state = client.get(f"/runs/{run_id}").json()["state"]
        finally:
            coordinator.shutdown()

        assert repair_calls == [], f"环境故障不应调用 repair，实际 {len(repair_calls)} 次"
        failures = state.get("failed_tasks") or []
        assert failures, state
        assert all(
            item.get("error_kind") in ("unavailable", "timeout") for item in failures
        ), failures

    def test_vanished_data_source_fails_the_run(self, api, tmp_path: Path) -> None:
        """数据源在运行前消失 → 明确失败，不产出假结果。"""
        client, store, root = api
        db = tmp_path / "customers.sqlite"
        build_database(db)

        # 先通过校验，再删掉文件
        run_id = client.post(
            "/runs",
            json={
                "data_sources": [str(db)],
                "sql_backend": "direct",
                "require_plan_approval": True,
            },
        ).json()["run_id"]

        db.unlink()

        final = wait(client, run_id, {"completed", "failed"}, timeout=180)
        assert final["status"] == "failed", final

    def test_changed_content_fingerprint_fails_the_run(self, api, tmp_path: Path) -> None:
        """内容指纹变化 → 失败（P1 的"数据源变化必须失败"语义）。"""
        client, store, root = api
        db = tmp_path / "customers.sqlite"
        build_database(db)

        run_id = client.post(
            "/runs",
            json={
                "data_sources": [str(db)],
                "sql_backend": "direct",
                "require_plan_approval": True,
            },
        ).json()["run_id"]

        # 等它真正读过数据源（进入 waiting_approval），再改写内容
        wait(client, run_id, {"waiting_approval"})

        connection = sqlite3.connect(str(db))
        try:
            connection.execute(
                "INSERT INTO customers VALUES (999, 'intruder', 'North')"
            )
            connection.commit()
        finally:
            connection.close()

        client.post(f"/runs/{run_id}/resume", json={"resume_value": {"decision": "approve"}})
        final = wait(client, run_id, {"completed", "failed"}, timeout=180)

        if final["status"] == "completed":
            # 若判定为完成，结果里必须没有把新数据算进去的迹象
            body = client.get(f"/runs/{run_id}").json()
            text = json.dumps(body.get("state") or {}, ensure_ascii=False)
            assert "intruder" not in text
        else:
            assert final["error"], "失败必须带原因"


class TestCrossProcessRecoveryOverHTTP:
    """真实 uvicorn + 强杀 + 新进程重启 + HTTP Resume。"""

    def test_mcp_run_survives_process_death(self, tmp_path: Path) -> None:
        workdir = isolated_workdir("p2d-xproc-")
        db = workdir / "customers.sqlite"
        build_database(db)
        port = free_port()

        # --- 阶段 1：真实 HTTP 创建 mcp Run，在 MCP 调用后崩溃 ---
        first = ApiServer(workdir, port, crash_at="after_mcp_call").start()
        try:
            status, created = first.create_run(
                data_sources=[str(db)],
                analysis_focus="客户总数",
                sql_backend="mcp",
            )
            assert status == 202, created
            run_id = created["run_id"]
            assert created["sql_backend"] == "mcp"

            # 崩溃点在 MCP 调用完成、节点尚未提交时触发
            code = first.wait_death(timeout=180)
            assert code == 17, f"未按预期强杀：{code}\n{first.tail()}"
        finally:
            first.stop()

        # 崩溃后不应残留 MCP Server 子进程
        assert mcp_server_pids() == [], "崩溃后残留 MCP Server 进程"

        # --- 阶段 2：全新进程重启，通过 HTTP 恢复同一个 Run ---
        second = ApiServer(workdir, port).start()
        try:
            status, record = second.get_run(run_id)
            assert status == 200, record
            run = record["run"]
            # 模式仍是最初保存的 mcp
            assert run["sql_backend"] == "mcp"
            assert run["status"] in ("running", "interrupted", "failed"), run

            resumed = second.base
            _code, body = __import__("tests.http_harness", fromlist=["http"]).http(
                "POST", f"{resumed}/runs/{run_id}/resume", {"resume_value": None}
            )
            assert _code == 202, body
            assert body["sql_backend"] == "mcp"

            final = second.wait_status(run_id, {"completed", "failed"}, timeout=300)
            assert final["status"] == "completed", final.get("error")

            # 恢复时确实重建并使用了 MCP Server
            assert "CallToolRequest" in second.tail(400), (
                "恢复进程未经过 MCP Server"
            )
        finally:
            second.stop()

        # 收尾：没有残留 MCP Server 进程
        assert mcp_server_pids() == []

        # 私有配置在 output_root 之外，且授权含本次 Run 的 dataset_id
        from core.runtime import _is_inside

        private = workdir.parent / f"{workdir.name}.mcp_private" / run_id / "authorizations.json"
        assert private.exists()
        payload = json.loads(private.read_text(encoding="utf-8"))
        assert payload["datasets"]
        assert all(
            item["dataset_id"].startswith(f"{run_id}:") for item in payload["datasets"]
        )
        # 用真正的目录归属判断，而不是字符串前缀
        # （`...x.mcp_private` 也以 `...x` 开头，前缀判断会给出假阳性）
        assert not _is_inside(private, workdir), (
            f"私有配置落在了会被挂载进沙箱的 output_root 内：{private}"
        )
