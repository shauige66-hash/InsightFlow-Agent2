"""有界重新规划、任务推进与审批边界。

用 Scripted 模型按调用次数给出不同计划：非法计划触发重新规划，合法计划进入
执行。工具用替身，避免为这些逻辑测试启动容器。
"""

from __future__ import annotations

import copy
import json
import threading
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

import harness
import nodes
from core.runtime import RunRuntime, use_runtime
from planning import plan_fingerprint
from planning.orchestrator import PlanChangedAfterApproval
from services.analysis_service import run_analysis
from tests.test_budget_real_entry_points import CountingFactory
from tests.test_report_approval import REPLIES, build_database

TARGET = "customers.customers"
VALID_PLAN = {
    "tasks": [
        {
            "task_id": "task_1",
            "description": "统计客户总数",
            "target_tables": [TARGET],
            "expected_output": "客户总数",
        }
    ]
}
INVALID_PLAN = {
    "tasks": [
        {
            "task_id": "task_1",
            "description": "统计不存在的表",
            "target_tables": ["customers.invented"],
            "expected_output": "无",
        }
    ]
}


@pytest.fixture(autouse=True)
def _fake_tool(monkeypatch: pytest.MonkeyPatch):
    """工具替身：这里的重点是计划与推进，不是容器。"""
    from harness import tool_dispatcher

    def fake_invoke(tool_name, tables, arguments, *, snapshots=None):
        return {"success": True, "result": [{"n": 3}], "error": None, "error_kind": None}

    monkeypatch.setattr(tool_dispatcher, "invoke_tool", fake_invoke)


@pytest.fixture(autouse=True)
def _restore_replies():
    original = copy.deepcopy(REPLIES)
    yield
    REPLIES.clear()
    REPLIES.update(copy.deepcopy(original))


def chat_result(payload: dict[str, Any]) -> ChatResult:
    return ChatResult(
        generations=[
            ChatGeneration(message=AIMessage(content=json.dumps(payload, ensure_ascii=False)))
        ]
    )


def run_with_plans(
    tmp_path: Path,
    *,
    run_id: str,
    plans: list[dict[str, Any]],
    budget: dict[str, Any] | None = None,
    require_plan_approval: bool = False,
    checkpointer=None,
    resume=None,
):
    """按顺序返回 `plans` 里的计划，超出后重复最后一个。"""
    database = tmp_path / run_id / "customers.sqlite"
    database.parent.mkdir(parents=True, exist_ok=True)
    if not database.exists():
        # 同一个 run_id 在审批前后会调用两次，不能重复建表
        build_database(database)

    counting = CountingFactory(run_id)
    seen = {"planner": 0}

    def responder(agent_name: str, total: int):
        if agent_name != "planner":
            return None
        index = min(seen["planner"], len(plans) - 1)
        seen["planner"] += 1
        return chat_result(plans[index])

    counting.responder = responder

    result = run_analysis(
        run_id=run_id,
        data_sources=[str(database)],
        model_factory=counting.build(),
        output_root=tmp_path,
        budget=budget,
        require_plan_approval=require_plan_approval,
        checkpointer=checkpointer,
        resume=resume,
    )
    return result, counting, seen


def read_events(tmp_path: Path, run_id: str, event: str) -> list[dict]:
    path = tmp_path / "logs" / f"{run_id}.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("event") == event
    ]


class TestValidPlanGoesStraightThrough:
    def test_a_valid_plan_is_not_replanned(self, tmp_path: Path) -> None:
        result, counting, seen = run_with_plans(
            tmp_path, run_id="h3-valid", plans=[VALID_PLAN]
        )

        assert result["analysis_status"] == "completed", result.get("status_reason")
        assert seen["planner"] == 1, "合法计划不该触发重新规划"
        assert read_events(tmp_path, "h3-valid", "plan_validated")
        assert read_events(tmp_path, "h3-valid", "plan_created")
        assert not read_events(tmp_path, "h3-valid", "replan_requested")
        assert result["plan_attempts"] == 1

    def test_the_executed_plan_is_the_validated_one(self, tmp_path: Path) -> None:
        """执行的是 Validator 归一化后的计划，而不是模型原始输出。"""
        result, _counting, _seen = run_with_plans(
            tmp_path, run_id="h3-normalised", plans=[VALID_PLAN]
        )

        assert result["analysis_plan"][0]["target_tables"] == [TARGET]


class TestBoundedReplanning:
    def test_an_invalid_table_triggers_one_replan(self, tmp_path: Path) -> None:
        result, _counting, seen = run_with_plans(
            tmp_path, run_id="h3-replan", plans=[INVALID_PLAN, VALID_PLAN]
        )

        assert result["analysis_status"] == "completed", result.get("status_reason")
        assert seen["planner"] == 2
        assert result["plan_attempts"] == 2

        rejected = read_events(tmp_path, "h3-replan", "plan_rejected")
        assert rejected, "非法计划没有被记录"
        assert "unknown_table" in rejected[0]["details"]["categories"]
        assert read_events(tmp_path, "h3-replan", "replan_requested")
        assert read_events(tmp_path, "h3-replan", "plan_validated")

    def test_consecutive_invalid_plans_stop_at_the_limit(self, tmp_path: Path) -> None:
        result, counting, seen = run_with_plans(
            tmp_path, run_id="h3-exhausted", plans=[INVALID_PLAN]
        )

        assert seen["planner"] == nodes.MAX_PLAN_ATTEMPTS
        assert result["plan_attempts"] == nodes.MAX_PLAN_ATTEMPTS
        assert result["analysis_status"] == "failed"
        assert result["analysis_plan"] == []
        assert "未通过校验" in result["status_reason"]

        assert len(read_events(tmp_path, "h3-exhausted", "plan_rejected")) == nodes.MAX_PLAN_ATTEMPTS
        assert read_events(tmp_path, "h3-exhausted", "replan_exhausted")
        # 无效计划绝不能进入执行
        assert result["execution_results"] == []

    def test_the_budget_stops_replanning(self, tmp_path: Path) -> None:
        """预算耗尽后不再生成计划：连第一次请求都不该发出。"""
        result, counting, seen = run_with_plans(
            tmp_path,
            run_id="h3-budget",
            plans=[INVALID_PLAN],
            budget={"max_llm_calls": 0},
        )

        assert counting.count == 0, counting.requests
        assert seen["planner"] == 0
        assert result["analysis_plan"] == []
        assert result["analysis_status"] in ("no_tasks", "failed")
        assert not read_events(tmp_path, "h3-budget", "replan_requested")

    def test_replanning_only_covers_plan_generation(self, tmp_path: Path) -> None:
        """工具执行失败不走重新规划；那是 Repair 的职责。"""
        result, _counting, seen = run_with_plans(
            tmp_path, run_id="h3-no-replan-on-tool", plans=[VALID_PLAN]
        )

        assert result["analysis_status"] == "completed"
        assert seen["planner"] == 1


class TestCheckpointKeepsTheLimit:
    def test_a_resumed_run_does_not_get_fresh_attempts(self, tmp_path: Path) -> None:
        """`plan_attempts` 已在 state 里用满时，Planner 一次都不调用。"""
        state: dict[str, Any] = {
            "run_id": "h3-resume-limit",
            "data_context": {TARGET: {"schema": {}, "row_count": 1}},
            "table_snapshot": {TARGET: {"name": TARGET, "source_type": "sqlite"}},
            "retrieved_knowledge": [],
            "analysis_focus": None,
            "plan_attempts": nodes.MAX_PLAN_ATTEMPTS,
            "plan_validation_errors": ["上一轮的失败原因"],
        }

        with use_runtime(
            RunRuntime(
                run_id="h3-resume-limit",
                model_factory=None,  # type: ignore[arg-type]
                resolver=None,  # type: ignore[arg-type]
                output_root=tmp_path,
            )
        ):
            updates = nodes.planner_node(state)

        assert updates["analysis_plan"] == []
        assert updates["plan_attempts"] == nodes.MAX_PLAN_ATTEMPTS
        assert updates["analysis_status"] == "failed"


class TestApprovalBoundary:
    def test_invalid_plans_never_reach_approval(self, tmp_path: Path) -> None:
        result, _counting, seen = run_with_plans(
            tmp_path,
            run_id="h3-approval-invalid",
            plans=[INVALID_PLAN],
            require_plan_approval=True,
        )

        # 三次尝试都在审批之前完成，被拒绝的计划没有产生 interrupt
        assert seen["planner"] == nodes.MAX_PLAN_ATTEMPTS
        assert result.get("__interrupt__") is None
        assert result["analysis_status"] == "failed"

    def test_the_approval_records_the_fingerprint_of_the_approved_plan(
        self, tmp_path: Path
    ) -> None:
        run_id = "h3-approval-fingerprint"

        with SqliteSaver.from_conn_string(str(tmp_path / "cp.sqlite")) as saver:
            run_with_plans(
                tmp_path,
                run_id=run_id,
                plans=[VALID_PLAN],
                require_plan_approval=True,
                checkpointer=saver,
            )

            from workflow import create_workflow

            paused = create_workflow(checkpointer=saver).get_state(
                {"configurable": {"thread_id": run_id}}
            )
            plan = paused.values["analysis_plan"]
            expected = plan_fingerprint(plan)

            pending = next(
                item.value
                for task in paused.tasks
                for item in (task.interrupts or ())
            )
            assert pending["summary"]["plan_fingerprint"] == expected

            result = run_with_plans(
                tmp_path,
                run_id=run_id,
                plans=[VALID_PLAN],
                require_plan_approval=True,
                checkpointer=saver,
                resume=Command(resume={"decision": "approve"}),
            )[0]

        assert result["analysis_status"] == "completed"
        assert result["approved_plan_fingerprint"] == expected

    def test_a_swapped_plan_is_refused_before_execution(self) -> None:
        """审批后计划被换掉时必须拒绝执行，而不是照常跑另一份计划。"""
        plan = VALID_PLAN["tasks"]
        state: dict[str, Any] = {
            "run_id": "h3-swapped",
            "analysis_plan": plan,
            "current_task_index": 0,
            "approved_plan_fingerprint": "not-the-approved-one",
        }

        with pytest.raises(PlanChangedAfterApproval):
            nodes.router_node(state)

    def test_matching_fingerprints_are_allowed(self, tmp_path: Path) -> None:
        """与获批计划一致时照常路由（这里只验证校验不误杀）。"""
        from planning.orchestrator import verify_approved_plan

        plan = VALID_PLAN["tasks"]
        state = {
            "analysis_plan": plan,
            "approved_plan_fingerprint": plan_fingerprint(plan),
        }

        verify_approved_plan(state)  # 不抛


class TestRunsAreIsolated:
    def test_attempt_counters_belong_to_their_own_run(self, tmp_path: Path) -> None:
        first, _counting1, seen1 = run_with_plans(
            tmp_path, run_id="h3-iso-a", plans=[INVALID_PLAN, VALID_PLAN]
        )
        second, _counting2, seen2 = run_with_plans(
            tmp_path, run_id="h3-iso-b", plans=[VALID_PLAN]
        )

        assert first["plan_attempts"] == 2
        assert second["plan_attempts"] == 1
        assert seen1["planner"] == 2
        assert seen2["planner"] == 1

    def test_the_limit_counter_is_per_state(self) -> None:
        """两个 state 各自的 plan_attempts 互不影响。"""
        seen: dict[str, int] = {}
        errors: list[BaseException] = []

        def worker(name: str, attempts: int) -> None:
            try:
                with harness.bind_state({"run_id": name, "plan_attempts": attempts}):
                    seen[name] = harness.current_state()["plan_attempts"]
            except BaseException as error:  # noqa: BLE001
                errors.append(error)

        threads = [
            threading.Thread(target=worker, args=("run-a", 2)),
            threading.Thread(target=worker, args=("run-b", 0)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        assert not errors, errors
        assert seen == {"run-a": 2, "run-b": 0}
