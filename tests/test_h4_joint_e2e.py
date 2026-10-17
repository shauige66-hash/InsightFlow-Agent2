"""H1～H4 联合链路：会话 → 计划 → 任务检索 → 上下文 → 报告 → 记忆。

真实 PostgreSQL + 真实 Redis + 真实 Coordinator；只有模型脚本化、工具替身。
跳过表示本机缺少容器，而不是用 SQLite / Mock Redis 替代后宣称通过。
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
from tests.postgres_support import (  # noqa: F401
    clear_redis,
    postgres,
    redis_available,
    redis_url,
    requires_postgres,
    requires_redis,
)
from tests.test_e2e_business_scenarios import ScriptedFactory
from tests.test_report_approval import build_database

pytestmark = [requires_postgres]

VIP_DOC = (
    "# VIP 客户\n\nVIP 客户指年消费金额超过 10000 元的客户。\n"
    "VIP 客户的售后响应时限为 4 小时。\n"
)

REPLIES: dict[str, Any] = {
    "planner": {
        "tasks": [
            {
                "task_id": "task_1",
                "description": "统计 VIP 客户数量",
                "target_tables": ["customers.customers"],
                "expected_output": "VIP 客户数",
            }
        ]
    },
    "executor": {"code": "SELECT COUNT(*) AS n FROM customers"},
    "critic": {"validation_passed": True, "feedback": "ok"},
    "visualizer": {"charts": []},
    "reporter": "# 分析结果\n\nVIP 客户 2 位。\n",
}


@pytest.fixture(autouse=True)
def _fake_tool(monkeypatch: pytest.MonkeyPatch):
    from harness import tool_dispatcher

    def fake_invoke(tool_name, tables, arguments, *, snapshots=None):
        return {"success": True, "result": [{"n": 2}], "error": None, "error_kind": None}

    monkeypatch.setattr(tool_dispatcher, "invoke_tool", fake_invoke)


@pytest.fixture
def app_env(postgres, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """postgres 模式 + Redis 缓存 + 真实 Coordinator。"""
    monkeypatch.setenv("INSIGHTFLOW_STORAGE", "postgres")
    monkeypatch.setenv("INSIGHTFLOW_DATABASE_URL", storage.database_url())
    monkeypatch.setenv("INSIGHTFLOW_OWNER", "joint-test")
    if redis_available():
        clear_redis()
        monkeypatch.setenv("INSIGHTFLOW_REDIS_ENABLED", "true")
        monkeypatch.setenv("INSIGHTFLOW_REDIS_URL", redis_url())
    else:
        monkeypatch.setenv("INSIGHTFLOW_REDIS_ENABLED", "false")

    storage.reset()
    storage.configure()

    from rag.retrieval import reset_retriever

    reset_retriever()

    database = tmp_path / "customers.sqlite"
    build_database(database)
    knowledge = tmp_path / "vip_rules.md"
    knowledge.write_text(VIP_DOC, encoding="utf-8")

    monkeypatch.setattr(api_runs, "OUTPUT_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(api_runs, "CHECKPOINT_PATH", tmp_path / "checkpoints" / "cp.sqlite")

    store = RunStore(tmp_path / "runs.sqlite")
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

    import api

    yield {
        "client": TestClient(api.app),
        "coordinator": coordinator,
        "store": store,
        "source": database,
        "knowledge": knowledge,
        "replies": replies,
        "tmp": tmp_path,
    }

    coordinator.shutdown()
    storage.reset()
    reset_retriever()


def make_session(env, *, knowledge: bool = True) -> str:
    payload = {"title": "联合测试", "dataset_refs": [str(env["source"])]}
    if knowledge:
        payload["knowledge_refs"] = [str(env["knowledge"])]
    response = env["client"].post("/sessions", json=payload)
    assert response.status_code == 201, response.text
    return response.json()["session_id"]


def post(env, session_id: str, content: str, **extra) -> dict[str, Any]:
    response = env["client"].post(
        f"/sessions/{session_id}/messages", json={"content": content, **extra}
    )
    assert response.status_code == 202, response.text
    return response.json()


def finish(env, turn_id: str) -> dict[str, Any]:
    run_id = env["client"].get(f"/turns/{turn_id}").json()["run_id"]
    env["coordinator"].executor.wait(run_id, timeout=180)
    return env["client"].get(f"/turns/{turn_id}").json()


class TestThreeTurnScenario:
    def test_three_turns_with_memory_and_cache(self, app_env) -> None:
        env = app_env
        session_id = make_session(env)

        first = post(env, session_id, "分析最近三个月的地区销售额")
        turn1 = finish(env, first["turn_id"])
        assert turn1["status"] == "completed"

        # 第二轮引用上一轮结果；同一分析重点，任务级问句因此可命中缓存
        second = post(
            env, session_id, "把刚才的结果按客户类型细分", analysis_focus="VIP 客户分析"
        )
        turn2 = finish(env, second["turn_id"])
        assert turn2["status"] == "completed"
        assert second["run_id"] != first["run_id"]

        third = post(
            env, session_id, "结合 VIP 客户规则再分析一次", analysis_focus="VIP 客户分析"
        )
        turn3 = finish(env, third["turn_id"])
        assert turn3["status"] == "completed"

        turns = env["client"].get(f"/sessions/{session_id}/turns").json()["turns"]
        assert [item["seq"] for item in turns] == [1, 2, 3]

        # 报告里能看到知识来源（第三轮带知识库）
        result = env["client"].get(f"/turns/{third['turn_id']}/result").json()["result"]
        report = Path(result["report_path"]).read_text(encoding="utf-8")
        assert "## 参考来源" in report
        assert "vip_rules.md" in report

        # 相同 RAG 查询命中 Redis（第二轮与第三轮问句相同）
        if redis_available():
            from rag.retrieval import get_retriever

            stats = get_retriever().stats(knowledge_sources=[str(env["knowledge"])])
            assert stats["cache"]["hits"] >= 1, stats

        # 用户确认一条偏好
        created = env["client"].post(
            "/memories",
            json={
                "kind": "report_format",
                "key": "chart_style",
                "value": "今后图表优先使用柱状图。",
                "status": "confirmed",
                "provenance": {"session_id": session_id, "turn_id": third["turn_id"]},
            },
        )
        assert created.status_code == 201, created.text

        # 新会话：能召回偏好，但不会把上一会话的临时数据当成事实
        new_session = make_session(env)
        recalled = env["client"].get(
            "/memories/retrieve", params={"q": "图表用什么", "scope": "global"}
        ).json()["memories"]
        assert [item["key"] for item in recalled] == ["chart_style"]

        new_turn = post(env, new_session, "画一张图表")
        assert finish(env, new_turn["turn_id"])["status"] == "completed"

        from core.checkpoints import open_checkpointer
        from workflow import create_workflow

        with open_checkpointer(api_runs.CHECKPOINT_PATH) as saver:
            snapshot = create_workflow(checkpointer=saver).get_state(
                {"configurable": {"thread_id": new_turn["run_id"]}}
            )
        memories = snapshot.values["memories"]
        assert [item["key"] for item in memories] == ["chart_style"]
        # 新会话没有把上一会话的消息当成本轮历史
        assert not (snapshot.values["conversation_context"] or {}).get(
            "recent_messages"
        )


class TestRestartKeepsEverything:
    def test_messages_memory_and_index_survive(self, app_env) -> None:
        env = app_env
        session_id = make_session(env)
        accepted = post(env, session_id, "统计 VIP 客户数量")
        assert finish(env, accepted["turn_id"])["status"] == "completed"

        memory = env["client"].post(
            "/memories",
            json={
                "kind": "preference",
                "key": "currency",
                "value": "今后销售额统一用人民币元表示。",
                "status": "confirmed",
            },
        ).json()

        # 进程重启：丢弃 Engine / 连接池 / 检索单例
        from rag.retrieval import reset_retriever

        storage.dispose_all()
        storage.configure()
        reset_retriever()

        session = env["client"].get(f"/sessions/{session_id}")
        assert session.status_code == 200
        messages = env["client"].get(f"/sessions/{session_id}/messages").json()["messages"]
        assert [item["role"] for item in messages] == ["user", "assistant"]

        listed = env["client"].get("/memories").json()["memories"]
        assert any(item["key"] == "currency" for item in listed)

        # 知识索引仍在：同样的问句可以直接检索到
        from rag.retrieval import get_retriever

        outcome = get_retriever().retrieve(
            knowledge_sources=[str(env["knowledge"])], query="VIP 客户的判定标准"
        )
        assert [item.source for item in outcome.evidence] == ["vip_rules.md"]


class TestInvalidation:
    def test_knowledge_update_and_memory_delete(self, app_env) -> None:
        env = app_env
        from rag.retrieval import get_retriever

        retriever = get_retriever()
        before = retriever.retrieve(
            knowledge_sources=[str(env["knowledge"])], query="VIP 客户的年消费门槛"
        )
        assert "10000" in before.evidence[0].content

        env["knowledge"].write_text(
            "# VIP 客户\n\nVIP 客户指年消费金额超过 50000 元的客户。\n",
            encoding="utf-8",
        )
        after = retriever.retrieve(
            knowledge_sources=[str(env["knowledge"])], query="VIP 客户的年消费门槛"
        )
        assert after.knowledge_version != before.knowledge_version
        assert "50000" in after.evidence[0].content

        memory = env["client"].post(
            "/memories",
            json={
                "kind": "preference",
                "key": "currency",
                "value": "今后销售额统一用人民币元表示。",
                "status": "confirmed",
            },
        ).json()
        assert env["client"].delete(f"/memories/{memory['memory_id']}").status_code == 200

        recalled = env["client"].get(
            "/memories/retrieve", params={"q": "人民币"}
        ).json()["memories"]
        assert recalled == []
