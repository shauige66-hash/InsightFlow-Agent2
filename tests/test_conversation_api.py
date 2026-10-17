"""多轮对话 API：会话、轮次、Run、幂等与重启恢复。

用真实的 `RunCoordinator` + 脚本化模型 + 工具替身，接口与执行链路都是真的；
只有模型和容器被替换。
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import api_runs
import storage
from services.run_coordinator import RunCoordinator
from services.run_store import RunStore
from tests.test_e2e_business_scenarios import ScriptedFactory
from tests.test_report_approval import build_database

REPLIES: dict[str, Any] = {
    "planner": {
        "tasks": [
            {
                "task_id": "task_1",
                "description": "统计客户总数",
                "target_tables": ["customers.customers"],
                "expected_output": "客户数",
            }
        ]
    },
    "executor": {"code": "SELECT COUNT(*) AS n FROM customers"},
    "critic": {"validation_passed": True, "feedback": "ok"},
    "visualizer": {"charts": []},
    "reporter": "# 分析结果\n\n共 3 位客户。\n",
}


@pytest.fixture(autouse=True)
def _fake_tool(monkeypatch: pytest.MonkeyPatch):
    from harness import tool_dispatcher

    def fake_invoke(tool_name, tables, arguments, *, snapshots=None):
        return {"success": True, "result": [{"n": 3}], "error": None, "error_kind": None}

    monkeypatch.setattr(tool_dispatcher, "invoke_tool", fake_invoke)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """隔离的本地存储 + 真实 Coordinator。"""
    monkeypatch.setenv("INSIGHTFLOW_STORAGE", "local")
    monkeypatch.setenv("INSIGHTFLOW_LOCAL_DB", str(tmp_path / "conversations.sqlite"))
    monkeypatch.setenv("INSIGHTFLOW_OWNER", "api-test")
    monkeypatch.delenv("INSIGHTFLOW_REDIS_ENABLED", raising=False)

    database = tmp_path / "customers.sqlite"
    build_database(database)

    storage.reset()
    storage.configure()

    monkeypatch.setattr(api_runs, "OUTPUT_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(api_runs, "CHECKPOINT_PATH", tmp_path / "checkpoints" / "cp.sqlite")

    store = RunStore(tmp_path / "runs.sqlite")
    # 脚本可替换：失败用例需要换一份模型输出，而不能继承默认回复
    replies = {"replies": copy.deepcopy(REPLIES)}
    coordinator = RunCoordinator(
        store,
        output_root=tmp_path / "outputs",
        checkpointer_path=tmp_path / "checkpoints" / "cp.sqlite",
        model_factory_builder=lambda run_id: ScriptedFactory(
            run_id, replies["replies"]
        ).build(),
    )
    monkeypatch.setattr(api_runs, "run_store", store)
    monkeypatch.setattr(api_runs, "coordinator", coordinator)

    client = TestClient(api_runs.app if hasattr(api_runs, "app") else _app())
    yield {
        "client": client,
        "tmp": tmp_path,
        "source": database,
        "coordinator": coordinator,
        "store": store,
        "replies": replies,
    }

    coordinator.shutdown()
    storage.reset()


def _app():
    import api

    return api.app


def make_session(env, *, title: str = "销售分析") -> str:
    response = env["client"].post(
        "/sessions",
        json={"title": title, "dataset_refs": [str(env["source"])]},
    )
    assert response.status_code == 201, response.text
    return response.json()["session_id"]


def post_message(env, session_id: str, content: str, **extra) -> dict[str, Any]:
    payload = {"content": content, **extra}
    response = env["client"].post(f"/sessions/{session_id}/messages", json=payload)
    assert response.status_code == 202, response.text
    return response.json()


def wait_turn(env, turn_id: str, *, timeout: float = 120) -> dict[str, Any]:
    run_id = env["client"].get(f"/turns/{turn_id}").json()["run_id"]
    env["coordinator"].executor.wait(run_id, timeout=timeout)
    response = env["client"].get(f"/turns/{turn_id}")
    assert response.status_code == 200, response.text
    return response.json()


class TestSessionLifecycle:
    def test_create_and_read_a_session(self, env) -> None:
        session_id = make_session(env)

        fetched = env["client"].get(f"/sessions/{session_id}")
        assert fetched.status_code == 200
        payload = fetched.json()
        assert payload["session_id"] == session_id
        assert payload["owner"] == "api-test"

    def test_listing_and_deleting(self, env) -> None:
        session_id = make_session(env)

        listed = env["client"].get("/sessions").json()["sessions"]
        assert [item["session_id"] for item in listed] == [session_id]

        assert env["client"].delete(f"/sessions/{session_id}").status_code == 200
        assert env["client"].get(f"/sessions/{session_id}").status_code == 404
        assert env["client"].get("/sessions").json()["sessions"] == []

    def test_a_session_requires_a_dataset(self, env) -> None:
        response = env["client"].post("/sessions", json={"title": "空的"})
        assert response.status_code == 400


class TestMessageFlow:
    def test_a_message_creates_a_turn_and_a_run(self, env) -> None:
        session_id = make_session(env)

        accepted = post_message(env, session_id, "分析一下客户数量")

        assert accepted["run_id"].startswith("run_")
        assert accepted["turn_id"].startswith("turn_")
        assert accepted["status"] == "running"

    def test_the_turn_completes_and_carries_the_result(self, env) -> None:
        session_id = make_session(env)
        accepted = post_message(env, session_id, "分析一下客户数量")

        turn = wait_turn(env, accepted["turn_id"])

        assert turn["status"] == "completed"
        result = env["client"].get(f"/turns/{accepted['turn_id']}/result").json()
        assert result["status"] == "completed"
        assert Path(result["result"]["report_path"]).exists()

    def test_the_assistant_message_is_written_only_from_real_output(self, env) -> None:
        session_id = make_session(env)
        accepted = post_message(env, session_id, "分析一下客户数量")
        wait_turn(env, accepted["turn_id"])

        messages = env["client"].get(f"/sessions/{session_id}/messages").json()["messages"]
        roles = [item["role"] for item in messages]

        assert roles == ["user", "assistant"]
        assert "客户" in messages[1]["content"]

    def test_client_message_id_is_idempotent(self, env) -> None:
        session_id = make_session(env)

        first = post_message(env, session_id, "分析客户数量", client_message_id="c-1")
        wait_turn(env, first["turn_id"])
        second = post_message(env, session_id, "分析客户数量", client_message_id="c-1")

        assert second["deduplicated"] is True
        assert second["turn_id"] == first["turn_id"]
        assert second["run_id"] == first["run_id"]

        messages = env["client"].get(f"/sessions/{session_id}/messages").json()["messages"]
        assert len(messages) == 2

    def test_a_second_message_creates_a_second_turn(self, env) -> None:
        session_id = make_session(env)

        first = post_message(env, session_id, "分析客户数量")
        wait_turn(env, first["turn_id"])
        second = post_message(env, session_id, "把结果按地区细分")
        wait_turn(env, second["turn_id"])

        assert second["run_id"] != first["run_id"]
        turns = env["client"].get(f"/sessions/{session_id}/turns").json()["turns"]
        assert [turn["seq"] for turn in turns] == [1, 2]
        assert all(turn["status"] == "completed" for turn in turns)

    def test_the_second_turn_carries_the_first_turns_result(self, env) -> None:
        from contextlib import contextmanager

        from core.checkpoints import open_checkpointer
        from workflow import create_workflow

        session_id = make_session(env)
        first = post_message(env, session_id, "分析客户数量")
        wait_turn(env, first["turn_id"])
        second = post_message(env, session_id, "把结果按地区细分")
        wait_turn(env, second["turn_id"])

        with open_checkpointer(api_runs.CHECKPOINT_PATH) as saver:
            snapshot = create_workflow(checkpointer=saver).get_state(
                {"configurable": {"thread_id": second["run_id"]}}
            )

        context = snapshot.values["conversation_context"]
        assert context["recent_messages"], context
        assert context["previous_turns"], context
        # 摘要来自上一轮真实结果，不是凭空生成的
        assert "客户" in context["previous_turns"][0]["summary"]


class TestFailureAndApproval:
    def test_a_failed_run_marks_the_turn_failed(self, env, monkeypatch: pytest.MonkeyPatch) -> None:
        env["replies"]["replies"] = {
            **copy.deepcopy(REPLIES),
            "planner": {
                "tasks": [
                    {
                        "task_id": "task_1",
                        "description": "x",
                        "target_tables": ["customers.invented"],
                        "expected_output": "y",
                    }
                ]
            },
        }

        session_id = make_session(env)
        accepted = post_message(env, session_id, "分析不存在的表")
        turn = wait_turn(env, accepted["turn_id"])

        assert turn["status"] == "failed"
        assert turn["result"] == {}
        messages = env["client"].get(f"/sessions/{session_id}/messages").json()["messages"]
        # 失败时不能写一条"已完成"的助手消息
        assert [item["role"] for item in messages] == ["user"]

    def test_new_messages_and_run_resume_are_different_paths(self, env) -> None:
        session_id = make_session(env)
        accepted = post_message(
            env, session_id, "统计客户数量", require_plan_approval=True
        )
        turn = wait_turn(env, accepted["turn_id"])
        assert turn["status"] == "waiting_approval"

        # 恢复同一个 Run：不产生新的轮次
        resumed = env["client"].post(
            f"/runs/{accepted['run_id']}/approval",
            json={"approval_id": "plan_approval_" + accepted["run_id"], "decision": "approve"},
        )
        assert resumed.status_code == 202, resumed.text
        env["coordinator"].executor.wait(accepted["run_id"], timeout=120)
        turns = env["client"].get(f"/sessions/{session_id}/turns").json()["turns"]
        assert len(turns) == 1

        # 新消息：新的 Turn + 新的 Run
        second = post_message(env, session_id, "再分析一次")
        assert second["run_id"] != accepted["run_id"]
        wait_turn(env, second["turn_id"])
        assert len(env["client"].get(f"/sessions/{session_id}/turns").json()["turns"]) == 2


class TestRestartRecovery:
    def test_sessions_and_messages_survive_a_restart(self, env) -> None:
        session_id = make_session(env)
        accepted = post_message(env, session_id, "分析客户数量")
        wait_turn(env, accepted["turn_id"])

        # 模拟进程重启：丢掉连接池与 Engine
        storage.dispose_all()
        storage.configure()

        fetched = env["client"].get(f"/sessions/{session_id}")
        assert fetched.status_code == 200
        messages = env["client"].get(f"/sessions/{session_id}/messages").json()["messages"]
        assert [item["role"] for item in messages] == ["user", "assistant"]

    def test_turn_status_survives_a_restart(self, env) -> None:
        session_id = make_session(env)
        accepted = post_message(env, session_id, "分析客户数量")
        wait_turn(env, accepted["turn_id"])

        storage.dispose_all()
        storage.configure()

        turn = env["client"].get(f"/turns/{accepted['turn_id']}").json()
        assert turn["status"] == "completed"
        assert turn["result"]["report_path"]
