"""P1-B 前置失败测试：真实 Checkpointer 无法持久化当前工作流的 state。

比 JSON 检查更强：这里用真实的 SqliteSaver 驱动真实工作流，跑一个节点，
让 Checkpointer 自己去序列化 state。改造前它会因为 DataFrame 而失败，
改造后必须通过。

用固定的假模型，不访问任何模型服务。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.sqlite import SqliteSaver

from core.config import Settings
from core.llm import CallContext, ModelFactory
from services.analysis_service import run_analysis
from tools.tool_registry import clear_tool_cache

from tests.conftest import CALL_RECORDS, CUSTOMERS_CSV

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
    "critic": {"validation_passed": True, "feedback": "ok"},
    "visualizer": {"charts": []},
    "reporter": "# Analysis Report\n\n## Overview\n共 5 位客户。\n",
}


class _Scripted(BaseChatModel):
    agent_name: str = ""
    record_key: str = ""

    @property
    def _llm_type(self) -> str:
        return f"p1-{self.agent_name}"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        CALL_RECORDS.setdefault(self.record_key, []).append(list(messages))
        reply = REPLIES[self.agent_name]
        content = (
            reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)
        )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])


def _factory(run_id: str) -> ModelFactory:
    CALL_RECORDS[run_id] = []

    def resolver(agent_name: str) -> _Scripted:
        return _Scripted(agent_name=agent_name, record_key=run_id)

    return ModelFactory(
        settings=Settings(
            api_key="sk-fake", base_url="http://127.0.0.1:8000/v1", model="deepseek-chat"
        ),
        context=CallContext(run_id=run_id, client_app="insightflow-agent2"),
        resolver=resolver,
    )


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    clear_tool_cache()
    yield tmp_path
    CALL_RECORDS.clear()
    clear_tool_cache()


class TestRealCheckpointerAgainstWorkflow:
    def test_sqlite_checkpointer_can_persist_a_run(self, tmp_path: Path) -> None:
        """用一个真实持久化的 Checkpointer 跑完整工作流。

        改造前：Checkpointer 序列化 state 时遇到 pandas DataFrame 会失败。
        改造后：state 只含可序列化字段，checkpoint 应正常写入。
        """
        db_path = tmp_path / "checkpoints.sqlite"

        with SqliteSaver.from_conn_string(str(db_path)) as saver:
            result = run_analysis(
                run_id="p1-checkpoint",
                data_sources=[str(CUSTOMERS_CSV)],
                model_factory=_factory("p1-checkpoint"),
                checkpointer=saver,
            )

            assert result["analysis_status"] == "completed"

            checkpoint = saver.get_tuple({"configurable": {"thread_id": "p1-checkpoint"}})
            assert checkpoint is not None, "未写入任何 checkpoint"
            assert checkpoint.checkpoint["channel_values"].get("run_id") == (
                "p1-checkpoint"
            )

        assert db_path.exists(), "checkpoint 文件未生成"

    def test_checkpoint_contains_no_runtime_objects(self, tmp_path: Path) -> None:
        """checkpoint 内不得出现模型客户端、resolver 或 DataFrame。"""
        db_path = tmp_path / "checkpoints.sqlite"

        with SqliteSaver.from_conn_string(str(db_path)) as saver:
            run_analysis(
                run_id="p1-clean",
                data_sources=[str(CUSTOMERS_CSV)],
                model_factory=_factory("p1-clean"),
                checkpointer=saver,
            )

            checkpoint = saver.get_tuple({"configurable": {"thread_id": "p1-clean"}})
            values = checkpoint.checkpoint["channel_values"]

            for forbidden in ("model_factory", "tables"):
                assert forbidden not in values, f"checkpoint 含运行期对象: {forbidden}"

            for key in ("table_snapshot", "data_sources"):
                assert key in values, f"缺少可恢复所需的字段: {key}"

    def test_checkpoint_survives_new_graph_instance(self, tmp_path: Path) -> None:
        """用新的 Checkpointer 连接同一文件，应能读回之前的 state。"""
        db_path = tmp_path / "checkpoints.sqlite"

        with SqliteSaver.from_conn_string(str(db_path)) as saver:
            run_analysis(
                run_id="p1-reopen",
                data_sources=[str(CUSTOMERS_CSV)],
                model_factory=_factory("p1-reopen"),
                checkpointer=saver,
            )

        # 全新连接：模拟进程重启后的读取
        with SqliteSaver.from_conn_string(str(db_path)) as reopened:
            checkpoint = reopened.get_tuple(
                {"configurable": {"thread_id": "p1-reopen"}}
            )
            assert checkpoint is not None
            values = checkpoint.checkpoint["channel_values"]
            assert values.get("run_id") == "p1-reopen"
            assert values.get("analysis_status") == "completed"

    def test_checkpoint_file_has_no_plaintext_api_key(self, tmp_path: Path) -> None:
        """checkpoint 文件里不得出现明文密钥。"""
        db_path = tmp_path / "checkpoints.sqlite"

        with SqliteSaver.from_conn_string(str(db_path)) as saver:
            run_analysis(
                run_id="p1-nosecret",
                data_sources=[str(CUSTOMERS_CSV)],
                model_factory=_factory("p1-nosecret"),
                checkpointer=saver,
            )

        raw = sqlite3.connect(str(db_path)).execute(
            "SELECT COUNT(*) FROM sqlite_master"
        ).fetchone()
        assert raw is not None

        blob = db_path.read_bytes()
        for forbidden in (b"sk-fake", b"sk-", b"api_key"):
            assert forbidden not in blob, f"checkpoint 含敏感内容: {forbidden!r}"
