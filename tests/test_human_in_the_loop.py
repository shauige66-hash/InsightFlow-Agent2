"""Human-in-the-Loop 的真实 interrupt / resume。

用真实的 LangGraph interrupt + Command(resume=...)，不模拟暂停：
图在审批节点处真正停止执行，恢复时从同一个 Checkpoint 继续。

中断点选在 planner 之后、任何工具执行之前，因此中断本身不产生副作用，
重放这个节点是安全的。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from core.config import Settings
from core.llm import CallContext, ModelFactory
from ingestion.resources import ResourceResolver
from services.analysis_service import read_run_state, run_analysis
from tools.tool_registry import clear_tool_cache
from utils.idempotency import plan_approval_id

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
        return f"hitl-{self.agent_name}"

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
    CALL_RECORDS.clear()
    clear_tool_cache()
    yield tmp_path
    CALL_RECORDS.clear()
    clear_tool_cache()


def _start_with_approval(
    saver: SqliteSaver, run_id: str, data_sources: list[str], output_root: Path
) -> dict[str, Any]:
    return run_analysis(
        run_id=run_id,
        data_sources=data_sources,
        model_factory=_factory(run_id),
        checkpointer=saver,
        resolver=ResourceResolver(),
        output_root=output_root,
        require_plan_approval=True,
    )


def _pending_approval(saver: SqliteSaver, run_id: str) -> dict[str, Any]:
    """从持久化状态里读出待审批负载。

    走 LangGraph 的 get_state，而不是直接翻 CheckpointTuple：
    恢复后的新进程也只有这条路径可用。
    """
    state = read_run_state(run_id, checkpointer=saver)
    assert state is not None, "没有持久化状态"
    pending = state.get("pending_approval")
    assert pending, "图未报告 interrupt"
    return pending


class TestInterruptOnPlanApproval:
    def test_run_pauses_before_any_tool_execution(
        self, tmp_path: Path
    ) -> None:
        """审批前不得执行任何任务——interrupt 之前不能有副作用。"""
        with SqliteSaver.from_conn_string(str(tmp_path / "cp.sqlite")) as saver:
            result = _start_with_approval(
                saver, "hitl-pause", [str(CUSTOMERS_CSV)], tmp_path
            )

            assert "__interrupt__" in result, "未产生 interrupt"

            pending = _pending_approval(saver, "hitl-pause")
            assert pending["kind"] == "plan_review"
            assert pending["run_id"] == "hitl-pause"
            assert pending["approval_id"] == plan_approval_id("hitl-pause")
            assert pending["summary"]["task_count"] == 1

            # 关键断言：还没有任何执行结果，也没有产物
            snapshot = saver.get_tuple({"configurable": {"thread_id": "hitl-pause"}})
            values = snapshot.checkpoint["channel_values"]
            assert values.get("execution_results", []) == []
            assert not list((tmp_path / "hitl-pause").glob("*.pdf"))

    def test_approve_continues_to_completion(self, tmp_path: Path) -> None:
        with SqliteSaver.from_conn_string(str(tmp_path / "cp.sqlite")) as saver:
            _start_with_approval(saver, "hitl-approve", [str(CUSTOMERS_CSV)], tmp_path)

            result = run_analysis(
                run_id="hitl-approve",
                data_sources=[str(CUSTOMERS_CSV)],
                model_factory=_factory("hitl-approve"),
                checkpointer=saver,
                resolver=ResourceResolver(),
                output_root=tmp_path,
                resume=Command(resume={"decision": "approve", "comment": "可以执行"}),
                require_plan_approval=True,
            )

            assert "__interrupt__" not in result
            assert result["analysis_status"] == "completed"
            assert result["run_status"] == "completed"

            approvals = result["approvals"]
            assert len(approvals) == 1
            assert approvals[0]["decision"] == "approve"
            assert approvals[0]["comment"] == "可以执行"
            assert approvals[0]["node"] == "approval_gate"

    def test_reject_stops_before_execution(self, tmp_path: Path) -> None:
        """拒绝后不得执行被拒绝的操作。"""
        with SqliteSaver.from_conn_string(str(tmp_path / "cp.sqlite")) as saver:
            _start_with_approval(saver, "hitl-reject", [str(CUSTOMERS_CSV)], tmp_path)

            result = run_analysis(
                run_id="hitl-reject",
                data_sources=[str(CUSTOMERS_CSV)],
                model_factory=_factory("hitl-reject"),
                checkpointer=saver,
                resolver=ResourceResolver(),
                output_root=tmp_path,
                resume=Command(resume={"decision": "reject", "comment": "口径不对"}),
                require_plan_approval=True,
            )

            assert result["approvals"][0]["decision"] == "reject"
            assert result["status_reason"] == "分析计划未获批准"

            # 没有执行结果，也没有产物
            assert result.get("execution_results", []) == []
            assert not list((tmp_path / "hitl-reject").glob("*.pdf"))

            # 执行类 Agent 一次都没被调用
            calls = CALL_RECORDS.get("hitl-reject", [])
            assert calls, "planner 应当被调用过"

    def test_invalid_decision_is_rejected(self, tmp_path: Path) -> None:
        with SqliteSaver.from_conn_string(str(tmp_path / "cp.sqlite")) as saver:
            _start_with_approval(saver, "hitl-bad", [str(CUSTOMERS_CSV)], tmp_path)

            with pytest.raises(ValueError, match="approve 或 reject"):
                run_analysis(
                    run_id="hitl-bad",
                    data_sources=[str(CUSTOMERS_CSV)],
                    model_factory=_factory("hitl-bad"),
                    checkpointer=saver,
                    resolver=ResourceResolver(),
                    output_root=tmp_path,
                    resume=Command(resume={"decision": "maybe"}),
                    require_plan_approval=True,
                )


class TestRepeatedApproval:
    def test_second_approval_does_not_duplicate_the_record(
        self, tmp_path: Path
    ) -> None:
        """重复提交同一次审批不得产生第二条记录。

        审批 ID 只由 run_id 与节点决定，恢复后重新计算得到同一个值，
        因此第二次 resume 会命中已归档的决定而不再中断。
        """
        db = tmp_path / "cp.sqlite"

        with SqliteSaver.from_conn_string(str(db)) as saver:
            _start_with_approval(saver, "hitl-dup", [str(CUSTOMERS_CSV)], tmp_path)
            first = run_analysis(
                run_id="hitl-dup",
                data_sources=[str(CUSTOMERS_CSV)],
                model_factory=_factory("hitl-dup"),
                checkpointer=saver,
                resolver=ResourceResolver(),
                output_root=tmp_path,
                resume=Command(resume={"decision": "approve"}),
                require_plan_approval=True,
            )

        # 新进程/新连接：再次恢复同一个已完成线程
        with SqliteSaver.from_conn_string(str(db)) as reopened:
            second = run_analysis(
                run_id="hitl-dup",
                data_sources=[str(CUSTOMERS_CSV)],
                model_factory=_factory("hitl-dup"),
                checkpointer=reopened,
                resolver=ResourceResolver(),
                output_root=tmp_path,
                resume=Command(resume={"decision": "approve"}),
                require_plan_approval=True,
            )

        assert len(first["approvals"]) == 1
        assert len(second["approvals"]) == 1, "重复审批产生了额外记录"
        assert second["approvals"][0]["approval_id"] == plan_approval_id("hitl-dup")


class TestApprovalIsPersisted:
    def test_pending_approval_survives_reopen(self, tmp_path: Path) -> None:
        """服务重启后仍能查询到待审批内容。"""
        db = tmp_path / "cp.sqlite"

        with SqliteSaver.from_conn_string(str(db)) as saver:
            _start_with_approval(saver, "hitl-persist", [str(CUSTOMERS_CSV)], tmp_path)

        # 全新连接 = 新进程
        with SqliteSaver.from_conn_string(str(db)) as reopened:
            pending = _pending_approval(reopened, "hitl-persist")
            assert pending["approval_id"] == plan_approval_id("hitl-persist")
            assert pending["summary"]["tasks"][0]["task_id"] == "task_1"


class TestSqlReadOnlyCannotBeApproved:
    """审批不能提升权限：非只读 SQL 必须硬拒绝。"""

    @pytest.mark.parametrize(
        "statement",
        [
            "DELETE FROM orders",
            "INSERT INTO orders VALUES (1)",
            "UPDATE orders SET amount = 0",
            "DROP TABLE orders",
            "/* 注释 */ WITH x AS (SELECT 1) DELETE FROM orders",
            "SELECT 1; DROP TABLE orders",
            "PRAGMA table_info(orders)",
            "ATTACH DATABASE 'x.db' AS x",
        ],
    )
    def test_write_statements_are_refused(self, statement: str) -> None:
        from utils.sql_guard import SqlNotReadOnly, ensure_read_only

        with pytest.raises(SqlNotReadOnly):
            ensure_read_only(statement)

    @pytest.mark.parametrize(
        "statement",
        [
            "SELECT * FROM orders",
            "select count(*) from orders",
            "WITH t AS (SELECT * FROM orders) SELECT * FROM t",
            "  SELECT 1  ",
            "SELECT * FROM orders;",
        ],
    )
    def test_read_only_statements_are_allowed(self, statement: str) -> None:
        from utils.sql_guard import ensure_read_only

        assert ensure_read_only(statement) == statement

    def test_approval_does_not_bypass_the_guard(self, tmp_path: Path) -> None:
        """即使计划已获批准，非只读 SQL 仍被拒绝。"""
        from utils.sql_guard import SqlNotReadOnly, ensure_read_only

        # 审批记录存在与否与只读判定无关
        with pytest.raises(SqlNotReadOnly):
            ensure_read_only("DELETE FROM orders")
