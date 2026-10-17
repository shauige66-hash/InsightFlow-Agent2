"""报告发布前的人工确认。

P1 只实现了**计划审批**（`approval_gate` / `kind="plan_review"`）。
`report_publish_gate_node` 与 `report_approval_id` 当时就写好了，
但**从未注册进图**——是死代码，永远不会被调用。

本文件覆盖把它接通之后的行为，重点是两件事：

1. 报告内容已生成、文件尚未写出时中断，恢复可安全重放；
2. 两个关口（计划 / 报告）的决定互不覆盖。
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

from core.config import Settings
from core.llm import CallContext, ModelFactory
from ingestion.resources import ResourceResolver
from services.analysis_service import run_analysis
from services.run_coordinator import RunCoordinator
from services.run_store import RunStore
from utils.idempotency import plan_approval_id, report_approval_id

TARGET = "customers.customers"
CODE = "SELECT COUNT(*) AS n FROM customers"

REPLIES: dict[str, Any] = {
    "planner": {
        "tasks": [
            {
                "task_id": "task_1",
                "description": "统计客户总数",
                "target_tables": [TARGET],
                "expected_output": "客户总数",
            }
        ]
    },
    "executor": {"code": CODE},
    "repair": {"code": CODE},
    "refine": {"code": CODE},
    "critic": {"validation_passed": True, "feedback": "ok"},
    "visualizer": {"charts": []},
    "reporter": "# Analysis Report\n\n## Overview\n客户总数 3。\n",
}


class _Scripted(BaseChatModel):
    agent_name: str = ""

    @property
    def _llm_type(self) -> str:
        return f"p2d-report-{self.agent_name}"

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


def build_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT, region TEXT)"
        )
        connection.executemany(
            "INSERT INTO customers VALUES (?, ?, ?)",
            [(101, "Alice", "East"), (102, "Bob", "West"), (103, "Cara", "East")],
        )
        connection.commit()
    finally:
        connection.close()


class TestReportApprovalGateIsWired:
    def test_gate_is_registered_in_the_graph(self) -> None:
        """回归：这个节点曾经是死代码，从未注册。"""
        from workflow import create_workflow

        graph = create_workflow().get_graph()
        assert "report_publish_gate" in graph.nodes

    def test_report_approval_id_is_stable(self) -> None:
        """恢复后重新计算必须得到同一个 ID，否则重复审批保护会失效。"""
        first = report_approval_id("run_x")
        assert first == report_approval_id("run_x")
        # 与计划审批的 ID 必须不同，否则两个关口会互相覆盖
        assert first != plan_approval_id("run_x")
        assert "report_publish" in first


class TestReportApprovalInterrupts:
    """中断与恢复必须走真实 Checkpoint。

    `interrupt()` 与 `Command(resume=...)` 都要求 Checkpointer：
    没有它 LangGraph 会直接报 "Cannot use Command(resume=...) without checkpointer"。
    因此这里两段式：第一段跑到中断，第二段用同一个 saver 恢复。
    """

    @staticmethod
    def _saver(tmp_path: Path):
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()

    def test_run_pauses_before_writing_the_report(self, tmp_path: Path) -> None:
        db = tmp_path / "customers.sqlite"
        build_database(db)

        result = run_analysis(
            run_id="p2d-report-pause",
            data_sources=[str(db)],
            model_factory=factory("p2d-report-pause"),
            resolver=ResourceResolver(),
            output_root=tmp_path,
            require_report_approval=True,
            checkpointer=self._saver(tmp_path),
        )

        interrupts = result.get("__interrupt__") or []
        assert interrupts, f"应在报告发布前中断：{result.get('run_status')}"

        payload = getattr(interrupts[0], "value", None)
        assert isinstance(payload, dict), payload
        assert payload["kind"] == "report_publish", payload
        assert payload["node"] == "report_publish_gate", payload

        summary = payload["summary"]
        # 报告内容确实已生成
        assert summary["report_chars"] > 0, summary
        # 中断点在报告生成之后、状态收尾之前，因此这里只断言"没有失败任务"，
        # 不断言 analysis_status（它由后续节点收尾写入）
        assert summary["failed_tasks"] == 0, summary

        # 关键：中断发生在文件写出**之前**
        assert not (tmp_path / "p2d-report-pause" / "analysis_report.md").exists()

    def test_approve_writes_the_report(self, tmp_path: Path) -> None:
        db = tmp_path / "customers.sqlite"
        build_database(db)

        run_id = "p2d-report-approve"
        saver = self._saver(tmp_path)

        run_analysis(
            run_id=run_id,
            data_sources=[str(db)],
            model_factory=factory(run_id),
            resolver=ResourceResolver(),
            output_root=tmp_path,
            require_report_approval=True,
            checkpointer=saver,
        )

        result = run_analysis(
            run_id=run_id,
            data_sources=[str(db)],
            model_factory=factory(run_id),
            resolver=ResourceResolver(),
            output_root=tmp_path,
            require_report_approval=True,
            continue_from_checkpoint=True,
            resume_value={"decision": "approve"},
            checkpointer=saver,
        )

        assert result["analysis_status"] == "completed", result.get("status_reason")
        report_path = Path(result["report_path"])
        assert report_path.exists(), report_path
        assert "Analysis Report" in report_path.read_text(encoding="utf-8")

        kinds = sorted(item["kind"] for item in result["approvals"])
        assert kinds == ["report_publish"], kinds

    def test_reject_does_not_publish(self, tmp_path: Path) -> None:
        db = tmp_path / "customers.sqlite"
        build_database(db)

        run_id = "p2d-report-reject"
        saver = self._saver(tmp_path)

        run_analysis(
            run_id=run_id,
            data_sources=[str(db)],
            model_factory=factory(run_id),
            resolver=ResourceResolver(),
            output_root=tmp_path,
            require_report_approval=True,
            checkpointer=saver,
        )

        result = run_analysis(
            run_id=run_id,
            data_sources=[str(db)],
            model_factory=factory(run_id),
            resolver=ResourceResolver(),
            output_root=tmp_path,
            require_report_approval=True,
            continue_from_checkpoint=True,
            resume_value={"decision": "reject"},
            checkpointer=saver,
        )

        assert result["analysis_status"] == "failed", result
        assert result["run_status"] == "failed", result
        assert result["status_reason"] == "分析报告未获批准发布", result

        # 被拒的报告不落盘、也不给出产物
        assert not (tmp_path / run_id / "analysis_report.md").exists()
        assert not result.get("artifacts")

    def test_default_is_off(self, tmp_path: Path) -> None:
        """未开启时保持既有全自动行为（报告直接落盘）。"""
        db = tmp_path / "customers.sqlite"
        build_database(db)

        result = run_analysis(
            run_id="p2d-report-off",
            data_sources=[str(db)],
            model_factory=factory("p2d-report-off"),
            resolver=ResourceResolver(),
            output_root=tmp_path,
        )

        assert result["analysis_status"] == "completed", result.get("status_reason")
        assert not result.get("__interrupt__")
        assert Path(result["report_path"]).exists()


class TestBothGatesAreIndependent:
    def test_plan_and_report_decisions_do_not_overwrite_each_other(
        self, tmp_path: Path
    ) -> None:
        """两个关口各存各的决定。

        早期 `_apply_decision` 无条件把结果写进 `plan_approval`，
        报告审批会覆盖计划审批的路由依据。
        """
        db = tmp_path / "customers.sqlite"
        build_database(db)

        run_id = "p2d-both-gates"

        # 第一次：停在计划审批
        first = run_analysis(
            run_id=run_id,
            data_sources=[str(db)],
            model_factory=factory(run_id),
            resolver=ResourceResolver(),
            output_root=tmp_path,
            require_plan_approval=True,
            require_report_approval=True,
            checkpointer=None,
        )
        interrupts = first.get("__interrupt__") or []
        assert interrupts
        assert getattr(interrupts[0], "value", {})["kind"] == "plan_review"

        # 带 Checkpoint 跑完整流程，便于观察两个决定
        from langgraph.checkpoint.memory import InMemorySaver

        saver = InMemorySaver()

        paused = run_analysis(
            run_id="p2d-both-gates-cp",
            data_sources=[str(db)],
            model_factory=factory("p2d-both-gates-cp"),
            resolver=ResourceResolver(),
            output_root=tmp_path,
            require_plan_approval=True,
            require_report_approval=True,
            checkpointer=saver,
        )
        blocked = paused.get("__interrupt__") or []
        assert blocked, "应停在计划审批"
        assert getattr(blocked[0], "value", {})["kind"] == "plan_review"

        # 批准计划 → 继续跑到报告审批
        second = run_analysis(
            run_id="p2d-both-gates-cp",
            data_sources=[str(db)],
            model_factory=factory("p2d-both-gates-cp"),
            resolver=ResourceResolver(),
            output_root=tmp_path,
            continue_from_checkpoint=True,
            resume_value={"decision": "approve"},
            checkpointer=saver,
        )
        pending = second.get("__interrupt__") or []
        assert pending, f"批准计划后应停在报告审批：{second.get('run_status')}"
        assert getattr(pending[0], "value", {})["kind"] == "report_publish"

        # 批准报告 → 完成；两个决定都留档
        final = run_analysis(
            run_id="p2d-both-gates-cp",
            data_sources=[str(db)],
            model_factory=factory("p2d-both-gates-cp"),
            resolver=ResourceResolver(),
            output_root=tmp_path,
            continue_from_checkpoint=True,
            resume_value={"decision": "approve"},
            checkpointer=saver,
        )

        kinds = sorted(item["kind"] for item in final["approvals"])
        assert kinds == ["plan_review", "report_publish"], kinds
        assert final["analysis_status"] == "completed", final.get("status_reason")
        assert Path(final["report_path"]).exists()


class TestReportApprovalOverHttp:
    @pytest.fixture
    def api(self, tmp_path: Path, monkeypatch):
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

    def _wait(self, client: TestClient, run_id: str, expected: set[str]) -> dict:
        deadline = time.time() + 180
        last: dict[str, Any] = {}
        while time.time() < deadline:
            last = (client.get(f"/runs/{run_id}").json().get("run") or {})
            if last.get("status") in expected:
                return last
            time.sleep(0.3)
        raise AssertionError(f"未进入 {expected}：{last}")

    def test_http_report_approval_round_trip(self, api, tmp_path: Path) -> None:
        client, store, root = api
        db = tmp_path / "customers.sqlite"
        build_database(db)

        created = client.post(
            "/runs",
            json={
                "data_sources": [str(db)],
                "analysis_focus": "客户总数",
                "sql_backend": "mcp",
                "require_report_approval": True,
            },
        )
        assert created.status_code == 202, created.text
        body = created.json()
        assert body["require_report_approval"] is True
        run_id = body["run_id"]

        waiting = self._wait(client, run_id, {"waiting_approval"})
        assert waiting["pending_approval"]["kind"] == "report_publish", waiting
        # 落库的开关也要在
        assert store.get(run_id).require_report_approval is True

        approved = client.post(
            f"/runs/{run_id}/approval", json={"decision": "approve"}
        )
        assert approved.status_code == 202, approved.text

        final = self._wait(client, run_id, {"completed", "failed"})
        assert final["status"] == "completed", final.get("error")

        report = client.get(f"/runs/{run_id}/report")
        assert report.status_code == 200, report.text
        assert report.content[:4] == b"%PDF"

    def test_report_approval_switch_survives_resume(self, api, tmp_path: Path) -> None:
        """恢复时按记录沿用开关，而不是恢复进程的默认值。"""
        client, store, root = api
        db = tmp_path / "customers.sqlite"
        build_database(db)

        run_id = client.post(
            "/runs",
            json={
                "data_sources": [str(db)],
                "sql_backend": "direct",
                "require_report_approval": True,
            },
        ).json()["run_id"]

        self._wait(client, run_id, {"waiting_approval"})
        assert store.get_sql_backend(run_id) == "direct"
        assert store.get(run_id).require_report_approval is True

        client.post(f"/runs/{run_id}/approval", json={"decision": "approve"})
        final = self._wait(client, run_id, {"completed", "failed"})
        assert final["status"] == "completed", final.get("error")
