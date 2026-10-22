"""Run 生命周期 API 与输入校验。

TestClient 直接调用 ASGI 应用，不起真实服务器。
模型用脚本化假模型注入，因此不访问任何模型服务。

只验证接口契约、生命周期流转与安全校验；执行语义由
test_cross_process_recovery.py 以真实进程覆盖。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from core.config import Settings
from core.llm import CallContext, ModelFactory

REPLIES: dict[str, Any] = {
    "planner": {
        "tasks": [
            {
                "task_id": "task_1",
                "description": "统计客户总数",
                "target_tables": ["customers"],
                "expected_output": "客户总数",
            }
        ]
    },
    "executor": {"code": "result = len(df)"},
    "repair": {"code": "result = len(df)"},
    "refine": {"code": "result = len(df)"},
    "critic": {"validation_passed": True, "feedback": "ok"},
    "visualizer": {"charts": []},
    "reporter": "# Analysis Report\n\n## Overview\n共 5 位客户。\n",
}


class _Scripted(BaseChatModel):
    agent_name: str = ""

    @property
    def _llm_type(self) -> str:
        return f"api-{self.agent_name}"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        reply = REPLIES[self.agent_name]
        content = (
            reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)
        )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])


def _factory(run_id: str) -> ModelFactory:
    def resolver(agent_name: str) -> _Scripted:
        return _Scripted(agent_name=agent_name)

    return ModelFactory(
        settings=Settings(
            api_key="sk-fake", base_url="http://127.0.0.1:8000/v1", model="deepseek-chat"
        ),
        context=CallContext(run_id=run_id, client_app="insightflow-agent2"),
        resolver=resolver,
    )


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    """把 API 的存储与产物都指向临时目录，避免污染仓库。"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("INSIGHTFLOW_EXECUTION","inline")
    import tools.python_tool as python_tool
    import nodes
    from sandbox.protocol import SandboxOutcome
    def scripted_python(*, job, tables, **kwargs):
        # This module tests HTTP lifecycle, not Docker execution. Evaluate only
        # the fixed scripted test operation, never arbitrary generated code.
        assert job.mode == "python"
        assert job.code.strip() == "result = len(df)"
        return SandboxOutcome(success=True,result=len(tables[job.table_names[0]].dataframe),error=None)
    def no_pdf_browser(*args, **kwargs):
        raise RuntimeError("PDF browser deliberately disabled in API unit tests")
    monkeypatch.setattr(python_tool,"execute_in_sandbox",scripted_python)
    monkeypatch.setattr(nodes,"save_pdf_report",no_pdf_browser)

    import api_runs
    from services.run_coordinator import RunCoordinator
    from services.run_store import RunStore

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
    # 读取状态必须与执行写入同一个 Checkpoint 文件
    monkeypatch.setattr(api_runs, "CHECKPOINT_PATH", tmp_path / "checkpoints.sqlite")

    from api import app

    with TestClient(app) as test_client:
        yield test_client

    coordinator.shutdown()


@pytest.fixture
def data_source(tmp_path: Path) -> Path:
    path = tmp_path / "customers.csv"
    path.write_text(
        "customer_id,customer_name,region\n"
        "C001,Alice,East\nC002,Bob,West\nC003,Carol,East\n",
        encoding="utf-8",
    )
    return path


def wait_for_status(client: TestClient, run_id: str, expected: set[str], timeout: float = 120.0):
    """轮询直到进入期望状态。"""
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        response = client.get(f"/runs/{run_id}")
        assert response.status_code == 200
        last = response.json()
        if last["run"]["status"] in expected:
            return last
        time.sleep(0.3)
    raise AssertionError(f"等待 {expected} 超时，最后状态：{last}")


class TestCompatibility:
    def test_health_still_works(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_v1_request_structure_is_preserved(self, client: TestClient) -> None:
        """V1 的 /analyze/upload 端点与请求字段必须保留。

        这里只验证接口存在与参数校验，不真正跑分析（那需要模型）。
        """
        schema = client.get("/openapi.json").json()

        assert "/analyze/upload" in schema["paths"]

        # 参数名必须保持不变，否则现有调用方会失效
        operation = schema["paths"]["/analyze/upload"]["post"]
        body = operation["requestBody"]["content"]["multipart/form-data"]["schema"]
        ref = body.get("$ref", "")
        component = schema["components"]["schemas"][ref.split("/")[-1]]
        assert "files" in component["properties"]
        assert "knowledge_files" in component["properties"]
        assert "analysis_focus" in component["properties"]


class TestRunLifecycle:
    def test_create_returns_run_id_immediately(self, client: TestClient, data_source: Path) -> None:
        response = client.post(
            "/runs", json={"data_sources": [str(data_source)]}
        )

        assert response.status_code == 202
        payload = response.json()
        assert payload["run_id"]
        assert payload["status"] in ("pending","running")

    def test_run_reaches_completed(self, client: TestClient, data_source: Path) -> None:
        created = client.post("/runs", json={"data_sources": [str(data_source)]})
        run_id = created.json()["run_id"]

        final = wait_for_status(client, run_id, {"completed", "failed"})
        assert final["run"]["status"] == "completed"
        assert final["run"]["error"] is None

    def test_state_endpoint_returns_lightweight_state(
        self, client: TestClient, data_source: Path
    ) -> None:
        created = client.post("/runs", json={"data_sources": [str(data_source)]})
        run_id = created.json()["run_id"]
        wait_for_status(client, run_id, {"completed", "failed"})

        response = client.get(f"/runs/{run_id}/state")
        assert response.status_code == 200
        state = response.json()

        assert state["run_id"] == run_id
        assert state["run_status"] == "completed"
        assert state["analysis_status"] == "completed"

        # 轻量：不得暴露完整数据上下文或表内容
        assert "data_context" not in state
        assert "table_snapshot" not in state
        assert "tables" not in state

    def test_run_appears_in_listing(self, client: TestClient, data_source: Path) -> None:
        created = client.post("/runs", json={"data_sources": [str(data_source)]})
        run_id = created.json()["run_id"]
        wait_for_status(client, run_id, {"completed", "failed"})

        listing = client.get("/runs").json()["runs"]
        assert any(record["run_id"] == run_id for record in listing)

    def test_unknown_run_returns_404(self, client: TestClient) -> None:
        assert client.get("/runs/nonexistent-run").status_code == 404
        assert client.get("/runs/nonexistent-run/state").status_code == 404


class TestApprovalLifecycle:
    def test_run_waits_for_approval(self, client: TestClient, data_source: Path) -> None:
        created = client.post(
            "/runs",
            json={
                "data_sources": [str(data_source)],
                "require_plan_approval": True,
            },
        )
        run_id = created.json()["run_id"]

        waiting = wait_for_status(client, run_id, {"waiting_approval"})
        assert waiting["run"]["status"] == "waiting_approval"
        assert waiting["run"]["pending_approval"]["kind"] == "plan_review"

        state = client.get(f"/runs/{run_id}/state").json()
        assert state["pending_approval"]["approval_id"]

    def test_approve_completes_the_run(self, client: TestClient, data_source: Path) -> None:
        created = client.post(
            "/runs",
            json={"data_sources": [str(data_source)], "require_plan_approval": True},
        )
        run_id = created.json()["run_id"]
        wait_for_status(client, run_id, {"waiting_approval"})

        response = client.post(
            f"/runs/{run_id}/approval",
            json={"decision": "approve", "comment": "同意"},
        )
        assert response.status_code == 202

        final = wait_for_status(client, run_id, {"completed", "failed"})
        assert final["run"]["status"] == "completed"
        assert final["run"]["approvals"][0]["decision"] == "approve"

    def test_reject_stops_the_run(self, client: TestClient, data_source: Path) -> None:
        created = client.post(
            "/runs",
            json={"data_sources": [str(data_source)], "require_plan_approval": True},
        )
        run_id = created.json()["run_id"]
        wait_for_status(client, run_id, {"waiting_approval"})

        client.post(f"/runs/{run_id}/approval", json={"decision": "reject"})
        final = wait_for_status(client, run_id, {"completed", "failed", "interrupted"})

        state = client.get(f"/runs/{run_id}/state").json()
        assert state.get("analysis_status") in (None, "failed") or final["run"][
            "status"
        ] == "failed"

    def test_duplicate_approval_is_reported_not_reapplied(
        self, client: TestClient, data_source: Path
    ) -> None:
        """重复审批必须被识别，不得触发第二次执行。"""
        created = client.post(
            "/runs",
            json={"data_sources": [str(data_source)], "require_plan_approval": True},
        )
        run_id = created.json()["run_id"]
        wait_for_status(client, run_id, {"waiting_approval"})

        first = client.post(f"/runs/{run_id}/approval", json={"decision": "approve"})
        assert first.json()["duplicate"] is False

        wait_for_status(client, run_id, {"completed", "failed"})

        second = client.post(f"/runs/{run_id}/approval", json={"decision": "approve"})
        # 已完成且审批已归档：要么 409（没有待审批），要么报告为重复
        assert second.status_code in (202, 409)
        if second.status_code == 202:
            assert second.json()["duplicate"] is True

    def test_approval_on_run_without_pending_is_conflict(
        self, client: TestClient, data_source: Path
    ) -> None:
        created = client.post("/runs", json={"data_sources": [str(data_source)]})
        run_id = created.json()["run_id"]
        wait_for_status(client, run_id, {"completed", "failed"})

        response = client.post(f"/runs/{run_id}/approval", json={"decision": "approve"})
        assert response.status_code == 409

    def test_invalid_decision_is_rejected(self, client: TestClient, data_source: Path) -> None:
        created = client.post(
            "/runs",
            json={"data_sources": [str(data_source)], "require_plan_approval": True},
        )
        run_id = created.json()["run_id"]
        wait_for_status(client, run_id, {"waiting_approval"})

        response = client.post(f"/runs/{run_id}/approval", json={"decision": "maybe"})
        assert response.status_code == 400


class TestInputValidation:
    @pytest.mark.parametrize(
        "bad_run_id",
        [
            "../../etc/passwd",
            "..\\..\\windows\\system32",
            "a",
            "has space",
            "semi;colon",
            "x" * 65,
        ],
    )
    def test_path_traversal_in_run_id_is_rejected(
        self, client: TestClient, bad_run_id: str
    ) -> None:
        """run_id 参与文件路径与线程名，必须拒绝穿越与非法字符。"""
        response = client.get(f"/runs/{bad_run_id}")
        assert response.status_code in (400, 404), (
            f"{bad_run_id!r} 未被拒绝：{response.status_code}"
        )

    def test_create_with_traversal_run_id_is_rejected(
        self, client: TestClient, data_source: Path
    ) -> None:
        response = client.post(
            "/runs",
            json={"data_sources": [str(data_source)], "run_id": "../escape"},
        )
        assert response.status_code == 400

    def test_missing_data_source_is_rejected(self, client: TestClient) -> None:
        response = client.post("/runs", json={"data_sources": ["no/such/file.csv"]})
        assert response.status_code == 400
        assert "不存在" in response.json()["detail"]

    def test_directory_as_data_source_is_rejected(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        response = client.post("/runs", json={"data_sources": [str(tmp_path)]})
        assert response.status_code == 400
        assert "不是文件" in response.json()["detail"]

    def test_empty_data_sources_is_rejected(self, client: TestClient) -> None:
        response = client.post("/runs", json={"data_sources": []})
        assert response.status_code == 400

    def test_resume_requires_a_resumable_status(
        self, client: TestClient, data_source: Path
    ) -> None:
        created = client.post("/runs", json={"data_sources": [str(data_source)]})
        run_id = created.json()["run_id"]
        wait_for_status(client, run_id, {"completed", "failed"})

        response = client.post(f"/runs/{run_id}/resume", json={})
        assert response.status_code == 409
