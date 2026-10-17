"""H5 运行期重新规划：只改未执行部分，修订必须重新校验与重新审批。

与 H3 的初始规划分开验证：这里关心的是"执行中发现后续任务做不了"之后的
行为，而不是"计划一开始就不合法"。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langgraph.types import Command

import storage
from core.checkpoints import open_checkpointer
from planning import plan_fingerprint
from services.analysis_service import run_analysis
from tests.scripted_queue import ScriptedQueueFactory, decision, task
from tests.test_report_approval import build_database

LIMITS = {"max_runtime_replans": 2, "max_decision_steps": 8}


@pytest.fixture(autouse=True)
def _local_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("INSIGHTFLOW_STORAGE", "local")
    monkeypatch.setenv("INSIGHTFLOW_LOCAL_DB", str(tmp_path / "conversations.sqlite"))
    monkeypatch.setenv("INSIGHTFLOW_OWNER", "h5-replan")
    monkeypatch.delenv("INSIGHTFLOW_REDIS_ENABLED", raising=False)
    storage.reset()
    storage.configure()
    yield
    storage.reset()


@pytest.fixture
def database(tmp_path: Path) -> Path:
    path = tmp_path / "customers.sqlite"
    build_database(path)
    return path


@pytest.fixture(autouse=True)
def _fake_tool(monkeypatch: pytest.MonkeyPatch):
    from harness import tool_dispatcher

    def fake_invoke(tool_name, tables, arguments, *, snapshots=None):
        payload = " ".join(
            str(value) for value in (arguments or {}).values() if isinstance(value, str)
        )
        if "BROKEN" in payload:
            return {
                "success": False,
                "result": None,
                "error": "执行失败：BROKEN",
                "error_kind": "python_error",
            }
        return {"success": True, "result": [{"n": 3}], "error": None, "error_kind": None}

    monkeypatch.setattr(tool_dispatcher, "invoke_tool", fake_invoke)


THREE_TASKS = [
    task("task_1", description="统计客户总数"),
    task("task_3", description="按地区细分"),
    task("task_5", description="按客户类型细分"),
]


def queues(
    *,
    planner: list[Any],
    decider: list[Any],
    executor: list[Any] | None = None,
) -> dict[str, list[Any]]:
    return {
        "planner": planner,
        "executor": executor
        or [{"code": "SELECT COUNT(*) AS n FROM customers"}],
        "repair": [{"code": "SELECT COUNT(*) AS n FROM customers"}],
        "refine": [{"code": "SELECT COUNT(*) AS n FROM customers"}],
        "critic": [{"validation_passed": True, "feedback": "ok"}],
        "visualizer": [{"charts": []}],
        "reporter": ["# 报告\n\n共 3 位客户。\n"],
        "decider": decider,
    }


def run(database: Path, tmp_path: Path, *, run_id: str, queues_, approval: bool = False):
    factory = ScriptedQueueFactory(run_id, queues_)
    result = run_analysis(
        run_id=run_id,
        data_sources=[str(database)],
        model_factory=factory.build(),
        output_root=tmp_path,
        enable_dynamic_loop=True,
        dynamic_limits=LIMITS,
        require_plan_approval=approval,
    )
    return result, factory


class TestRuntimeReplan:
    def test_only_the_remaining_tasks_are_rewritten(self, database, tmp_path) -> None:
        result, factory = run(
            database,
            tmp_path,
            run_id="h5-replan",
            queues_=queues(
                planner=[
                    {"tasks": THREE_TASKS},
                    # 修订：只给尚未完成的部分
                    {"tasks": [task("task_6", description="按地区统计客户数（数据只支持这一层）")]},
                ],
                executor=[
                    {"code": "SELECT COUNT(*) AS n FROM customers"},
                    {"code": "BROKEN"},
                    {"code": "SELECT COUNT(*) AS n FROM customers"},
                ],
                decider=[
                    decision("continue"),          # task_1 通过
                    decision("replan_remaining"),  # task_3 无法完成
                    decision("continue"),          # 修订后的任务通过
                ],
            ),
        )

        assert result["analysis_status"] == "completed"
        assert result["plan_revision"] == 1
        assert result["runtime_replans"] == 1

        plan = result["analysis_plan"]
        assert plan[0]["task_id"] == "task_1", "已完成任务被改写"
        assert [item["task_id"] for item in plan[1:]] == ["task_6"]

        # 已完成任务的执行结果没有被重复写入
        task_1_results = [
            item for item in result["execution_results"] if item["task_id"] == "task_1"
        ]
        assert len(task_1_results) == 1
        # 原来的 task_5 没有被执行
        assert not [
            item for item in result["execution_results"] if item["task_id"] == "task_5"
        ]

    def test_an_invalid_revision_is_rejected(self, database, tmp_path) -> None:
        result, _factory = run(
            database,
            tmp_path,
            run_id="h5-replan-invalid",
            queues_=queues(
                planner=[
                    {"tasks": THREE_TASKS},
                    {
                        "tasks": [
                            task("task_6", tables=["customers.invented"], description="不存在的表")
                        ]
                    },
                ],
                executor=[
                    {"code": "SELECT COUNT(*) AS n FROM customers"},
                    {"code": "BROKEN"},
                ],
                decider=[decision("continue"), decision("replan_remaining")],
            ),
        )

        assert result["runtime_replans"] == 1
        assert result["dynamic_finished"] is True
        assert result["analysis_status"] == "partial"
        assert "未通过校验" in (result["termination_reason"] or "")
        # 非法的新任务没有被执行
        assert not [
            item for item in result["execution_results"] if item["task_id"] == "task_6"
        ]

    def test_a_revision_cannot_reuse_a_completed_task_id(self, database, tmp_path) -> None:
        result, _factory = run(
            database,
            tmp_path,
            run_id="h5-replan-collision",
            queues_=queues(
                planner=[
                    {"tasks": THREE_TASKS},
                    {"tasks": [task("task_1", description="重复使用已完成的任务 ID")]},
                ],
                executor=[
                    {"code": "SELECT COUNT(*) AS n FROM customers"},
                    {"code": "BROKEN"},
                ],
                decider=[decision("continue"), decision("replan_remaining")],
            ),
        )

        assert result["runtime_replans"] == 1
        assert result["dynamic_finished"] is True
        assert "冲突" in (result["termination_reason"] or "")

    def test_runtime_replans_are_capped(self, database, tmp_path) -> None:
        result, factory = run(
            database,
            tmp_path,
            run_id="h5-replan-cap",
            queues_=queues(
                planner=[
                    {"tasks": THREE_TASKS},
                    {"tasks": [task("task_6", description="修订一"), task("task_7", description="修订二")]},
                    {"tasks": [task("task_8", description="再修订")]},
                ],
                executor=[
                    {"code": "SELECT COUNT(*) AS n FROM customers"},
                    {"code": "BROKEN"},
                    {"code": "BROKEN"},
                    {"code": "BROKEN"},
                ],
                decider=[decision("continue")] + [decision("replan_remaining")],
            ),
        )

        assert result["runtime_replans"] <= 2
        actions = [item["action"] for item in result["decision_history"]]
        assert actions.count("replan_remaining") <= 2
        # 达到上限后不再修订，任务被跳过收尾
        assert result["analysis_status"] == "partial"
        assert result["skipped_tasks"], result["decision_history"][-1]


class TestReplanUnderApproval:
    def _scripted(self, run_id: str, planner_queue: list[Any]) -> dict[str, list[Any]]:
        """每个阶段单独一份脚本：恢复时 Planner 只会被"修订"调用一次。"""
        return queues(
            planner=planner_queue,
            executor=[
                {"code": "SELECT COUNT(*) AS n FROM customers"},
                {"code": "BROKEN"},
                {"code": "SELECT COUNT(*) AS n FROM customers"},
            ],
            decider=[
                decision("continue"),
                decision("replan_remaining"),
                decision("continue"),
            ],
        )

    def _advance_to_revision(self, database, tmp_path, run_id: str):
        """先批准初始计划，跑到"运行期修订"的那个中断上。"""
        checkpoint = tmp_path / "cp.sqlite"
        initial = [{"tasks": THREE_TASKS}]
        revision = [{"tasks": [task("task_6", description="修订任务")]}]

        with open_checkpointer(checkpoint) as saver:
            first = run_analysis(
                run_id=run_id,
                data_sources=[str(database)],
                model_factory=ScriptedQueueFactory(
                    run_id, self._scripted(run_id, initial)
                ).build(),
                output_root=tmp_path,
                enable_dynamic_loop=True,
                dynamic_limits=LIMITS,
                require_plan_approval=True,
                checkpointer=saver,
            )
        assert first.get("__interrupt__"), "初始计划没有触发审批"
        assert first["__interrupt__"][0].value["summary"]["plan_revision"] == 0

        with open_checkpointer(checkpoint) as saver:
            second = run_analysis(
                run_id=run_id,
                data_sources=[str(database)],
                model_factory=ScriptedQueueFactory(
                    run_id, self._scripted(run_id, revision)
                ).build(),
                output_root=tmp_path,
                enable_dynamic_loop=True,
                dynamic_limits=LIMITS,
                require_plan_approval=True,
                checkpointer=saver,
                resume=Command(resume={"decision": "approve"}),
            )
        return second, revision

    def test_a_revision_requires_a_new_approval(self, database, tmp_path) -> None:
        run_id = "h5-replan-hitl"
        paused, revision = self._advance_to_revision(database, tmp_path, run_id)

        assert paused.get("__interrupt__"), "运行期修订没有触发审批"
        pending = paused["__interrupt__"][0].value
        assert pending["summary"]["plan_revision"] == 1
        assert pending["approval_id"].endswith(":r1")
        assert "task_6" in json.dumps(pending["summary"], ensure_ascii=False)

        approved_fingerprint = plan_fingerprint(paused["analysis_plan"])

        with open_checkpointer(tmp_path / "cp.sqlite") as saver:
            resumed = run_analysis(
                run_id=run_id,
                data_sources=[str(database)],
                model_factory=ScriptedQueueFactory(
                    run_id, self._scripted(run_id, revision)
                ).build(),
                output_root=tmp_path,
                enable_dynamic_loop=True,
                dynamic_limits=LIMITS,
                require_plan_approval=True,
                checkpointer=saver,
                resume=Command(resume={"decision": "approve"}),
            )

        assert resumed["analysis_status"] == "completed"
        assert resumed["approved_plan_fingerprint"] == approved_fingerprint

    def test_a_rejected_revision_is_not_executed(self, database, tmp_path) -> None:
        run_id = "h5-replan-reject"
        paused, revision = self._advance_to_revision(database, tmp_path, run_id)
        assert paused.get("__interrupt__")

        with open_checkpointer(tmp_path / "cp.sqlite") as saver:
            resumed = run_analysis(
                run_id=run_id,
                data_sources=[str(database)],
                model_factory=ScriptedQueueFactory(
                    run_id, self._scripted(run_id, revision)
                ).build(),
                output_root=tmp_path,
                enable_dynamic_loop=True,
                dynamic_limits=LIMITS,
                require_plan_approval=True,
                checkpointer=saver,
                resume=Command(resume={"decision": "reject"}),
            )

        # 被拒的新任务不执行，计划退回已完成的前缀
        assert not [
            item for item in resumed["execution_results"] if item["task_id"] == "task_6"
        ]
        assert [item["task_id"] for item in resumed["analysis_plan"]] == ["task_1"]
        assert resumed["dynamic_finished"] is True
        assert "未获批准" in (resumed["termination_reason"] or "")

    def test_the_paused_revision_does_not_record_duplicate_approvals(
        self, database, tmp_path
    ) -> None:
        run_id = "h5-replan-idempotent"
        paused, revision = self._advance_to_revision(database, tmp_path, run_id)
        approval_id = paused["__interrupt__"][0].value["approval_id"]

        with open_checkpointer(tmp_path / "cp.sqlite") as saver:
            resumed = run_analysis(
                run_id=run_id,
                data_sources=[str(database)],
                model_factory=ScriptedQueueFactory(
                    run_id, self._scripted(run_id, revision)
                ).build(),
                output_root=tmp_path,
                enable_dynamic_loop=True,
                dynamic_limits=LIMITS,
                require_plan_approval=True,
                checkpointer=saver,
                resume=Command(resume={"decision": "approve"}),
            )

        recorded = [
            item for item in resumed["approvals"] if item["approval_id"] == approval_id
        ]
        assert len(recorded) == 1
