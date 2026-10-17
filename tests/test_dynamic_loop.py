"""H5 动态决策：动作矩阵、循环上限与预算。

脚本化模型按顺序给出观察后的动作选择；工具是替身，但走真实的 LangGraph 节点、
Harness 的模型/工具入口与 Checkpoint。断言看的是**实际走了哪条路径**，
不是"决策函数被调用了"。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import storage
from services.analysis_service import run_analysis
from tests.scripted_queue import ScriptedQueueFactory, decision, task
from tests.test_report_approval import build_database


@pytest.fixture(autouse=True)
def _local_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("INSIGHTFLOW_STORAGE", "local")
    monkeypatch.setenv("INSIGHTFLOW_LOCAL_DB", str(tmp_path / "conversations.sqlite"))
    monkeypatch.setenv("INSIGHTFLOW_OWNER", "h5-test")
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


def install_tool(monkeypatch: pytest.MonkeyPatch, *, fail_markers: dict[str, str] | None = None):
    """工具替身：代码里出现指定标记时返回对应错误。

    这是模拟"执行结果"，不是把决策逻辑写进测试关键词：决策仍然由脚本化模型
    在观察结果之后做出。SQL 与 Python 的参数键不同，因此检查所有字符串值。
    """
    from harness import tool_dispatcher

    markers = fail_markers or {}

    def fake_invoke(tool_name, tables, arguments, *, snapshots=None):
        payload = " ".join(
            str(value) for value in (arguments or {}).values() if isinstance(value, str)
        )
        for marker, kind in markers.items():
            if marker in payload:
                return {
                    "success": False,
                    "result": None,
                    "error": f"执行失败：{marker}",
                    "error_kind": kind,
                }
        rows = [{"n": 3}]
        return {"success": True, "result": rows, "error": None, "error_kind": None}

    monkeypatch.setattr(tool_dispatcher, "invoke_tool", fake_invoke)


def run(
    database: Path,
    tmp_path: Path,
    *,
    run_id: str,
    queues: dict[str, list[Any]],
    enable: bool = True,
    limits: dict[str, int] | None = None,
    budget: dict[str, Any] | None = None,
    knowledge: list[str] | None = None,
):
    factory = ScriptedQueueFactory(run_id, queues)
    result = run_analysis(
        run_id=run_id,
        data_sources=[str(database)],
        knowledge_sources=knowledge or [],
        model_factory=factory.build(),
        output_root=tmp_path,
        enable_dynamic_loop=enable,
        dynamic_limits=limits,
        budget=budget,
    )
    return result, factory


def events_of(tmp_path: Path, run_id: str, event: str) -> list[dict]:
    path = tmp_path / "logs" / f"{run_id}.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("event") == event
    ]


BASE_QUEUES: dict[str, list[Any]] = {
    "executor": [{"code": "SELECT COUNT(*) AS n FROM customers"}],
    # 修复与精修的默认脚本：与执行一致，保证"再试一次"不会因为脚本缺失而崩
    "repair": [{"code": "SELECT COUNT(*) AS n FROM customers"}],
    "refine": [{"code": "SELECT COUNT(*) AS n FROM customers"}],
    "critic": [{"validation_passed": True, "feedback": "ok"}],
    "visualizer": [{"charts": []}],
    "reporter": ["# 报告\n\n共 3 位客户。\n"],
}


def queues_with(**overrides: list[Any]) -> dict[str, list[Any]]:
    merged = {name: list(items) for name, items in BASE_QUEUES.items()}
    merged.update(overrides)
    return merged


class TestNormalPath:
    def test_a_passing_task_continues_then_finishes(self, database, tmp_path, monkeypatch) -> None:
        install_tool(monkeypatch)
        result, factory = run(
            database,
            tmp_path,
            run_id="h5-continue",
            queues=queues_with(
                planner=[{"tasks": [task("task_1"), task("task_2", description="第二个任务")]}],
                decider=[decision("continue"), decision("finish")],
            ),
        )

        assert result["analysis_status"] == "completed"
        history = result["decision_history"]
        assert [item["action"] for item in history] == ["continue", "finish"]
        assert result["loop_step"] == 2
        # 真实模型请求数包含两次决策
        assert factory.calls.count("decider") == 2

    def test_disabled_loop_keeps_the_fixed_topology(self, database, tmp_path, monkeypatch) -> None:
        install_tool(monkeypatch)
        result, factory = run(
            database,
            tmp_path,
            run_id="h5-disabled",
            queues=queues_with(planner=[{"tasks": [task("task_1")]}]),
            enable=False,
        )

        assert result["analysis_status"] == "completed"
        assert result["decision_history"] == []
        assert factory.calls.count("decider") == 0
        assert result["loop_step"] == 0


class TestRepairAndRefine:
    def test_a_repairable_error_goes_through_repair(self, database, tmp_path, monkeypatch) -> None:
        install_tool(monkeypatch, fail_markers={"BROKEN": "python_error"})
        result, factory = run(
            database,
            tmp_path,
            run_id="h5-repair",
            queues=queues_with(
                planner=[{"tasks": [task("task_1")]}],
                executor=[{"code": "BROKEN"}, {"code": "SELECT COUNT(*) AS n FROM customers"}],
                decider=[decision("repair"), decision("continue"), decision("finish")],
            ),
        )

        assert result["analysis_status"] == "completed"
        assert "repair" in factory.calls
        assert [item["action"] for item in result["decision_history"]][:2] == [
            "repair",
            "continue",
        ]

    def test_quality_feedback_goes_through_refine(self, database, tmp_path, monkeypatch) -> None:
        install_tool(monkeypatch)
        result, factory = run(
            database,
            tmp_path,
            run_id="h5-refine",
            queues=queues_with(
                planner=[{"tasks": [task("task_1")]}],
                critic=[
                    {"validation_passed": False, "feedback": "结果缺少区域细分"},
                    {"validation_passed": True, "feedback": "ok"},
                ],
                decider=[decision("refine"), decision("continue"), decision("finish")],
            ),
        )

        assert result["analysis_status"] == "completed"
        assert "refine" in factory.calls
        # 精修之后又重新校验了一次
        assert factory.calls.count("critic") == 2


class TestExtraRetrieval:
    def test_business_gap_triggers_extra_retrieval_with_new_evidence(
        self, database, tmp_path, monkeypatch
    ) -> None:
        install_tool(monkeypatch)
        knowledge = tmp_path / "vip.md"
        knowledge.write_text(
            "# VIP 客户\n\nVIP 客户指年消费金额超过 10000 元的客户。\n", encoding="utf-8"
        )

        result, factory = run(
            database,
            tmp_path,
            run_id="h5-retrieve",
            queues=queues_with(
                planner=[{"tasks": [task("task_1", description="统计 VIP 客户数量")]}],
                critic=[
                    {"validation_passed": False, "feedback": "缺少 VIP 客户的判定口径"},
                    {"validation_passed": True, "feedback": "ok"},
                ],
                decider=[
                    decision("retrieve_more"),
                    decision("continue"),
                    decision("finish"),
                ],
            ),
            knowledge=[str(knowledge)],
        )

        assert result["analysis_status"] == "completed"
        assert result["extra_retrievals"] == 1
        # 第二次执行时，问句里带上了补充检索的依据
        executor_prompts = factory.prompts["executor"]
        assert len(executor_prompts) == 2

        evidence = result["task_knowledge"]["task_1"]
        assert any(item["source"] == "vip.md" for item in evidence)

    def test_extra_retrieval_is_refused_without_a_knowledge_base(
        self, database, tmp_path, monkeypatch
    ) -> None:
        install_tool(monkeypatch)
        result, factory = run(
            database,
            tmp_path,
            run_id="h5-retrieve-none",
            queues=queues_with(
                planner=[{"tasks": [task("task_1")]}],
                critic=[
                    {"validation_passed": False, "feedback": "缺依据"},
                    {"validation_passed": True, "feedback": "ok"},
                ],
                decider=[decision("retrieve_more"), decision("continue"), decision("finish")],
            ),
        )

        assert result["extra_retrievals"] == 0
        forced = [item for item in result["decision_history"] if item["forced"]]
        assert forced and forced[0]["action"] in ("refine", "skip_task")
        assert result["analysis_status"] in ("completed", "partial")


class TestSkipAndTerminalErrors:
    def test_a_permission_error_is_not_repaired(self, database, tmp_path, monkeypatch) -> None:
        install_tool(monkeypatch, fail_markers={"DENIED": "sql_permission"})
        result, factory = run(
            database,
            tmp_path,
            run_id="h5-permission",
            queues=queues_with(
                planner=[{"tasks": [task("task_1")]}],
                executor=[{"code": "DENIED"}],
                # 模型坚持要修，但策略不允许
                decider=[decision("repair"), decision("finish")],
            ),
        )

        history = result["decision_history"]
        assert history[0]["action"] == "skip_task"
        assert history[0]["forced"] is True
        assert "repair" not in factory.calls, "权限错误仍然触发了修复调用"
        assert result["analysis_status"] == "partial"
        assert result["skipped_tasks"][0]["task_id"] == "task_1"

    def test_skipping_continues_with_the_next_task(self, database, tmp_path, monkeypatch) -> None:
        install_tool(monkeypatch, fail_markers={"DENIED": "authorization_denied"})
        result, factory = run(
            database,
            tmp_path,
            run_id="h5-skip",
            queues=queues_with(
                planner=[{"tasks": [task("task_1"), task("task_2", description="第二个任务")]}],
                executor=[
                    {"code": "DENIED"},
                    {"code": "SELECT COUNT(*) AS n FROM customers"},
                ],
                decider=[decision("skip_task"), decision("continue"), decision("finish")],
            ),
        )

        assert result["analysis_status"] == "partial"
        assert result["skipped_tasks"][0]["task_id"] == "task_1"
        assert factory.calls.count("executor") == 2, "跳过后应继续执行下一个任务"


class TestBounds:
    def test_decision_steps_stop_at_the_limit(self, database, tmp_path, monkeypatch) -> None:
        install_tool(monkeypatch)
        result, factory = run(
            database,
            tmp_path,
            run_id="h5-steps",
            queues=queues_with(
                planner=[{"tasks": [task("task_1"), task("task_2", description="第二个任务")]}],
                # 模型一直想继续，但步数上限先到
                decider=[decision("continue")],
            ),
            limits={"max_decision_steps": 1},
        )

        assert result["decision_history"][-1]["action"] == "finish"
        assert result["decision_history"][-1]["forced"] is True
        assert "上限" in (result["termination_reason"] or "")
        # 决策在上限处收尾：只发生了两次决策，不会继续给第三个任务做判断
        assert len(result["decision_history"]) == 2
        assert factory.calls.count("decider") == 2

    def test_repeated_failed_action_falls_back(self, database, tmp_path, monkeypatch) -> None:
        install_tool(monkeypatch, fail_markers={"BROKEN": "python_error"})
        result, factory = run(
            database,
            tmp_path,
            run_id="h5-repeat",
            queues=queues_with(
                planner=[{"tasks": [task("task_1")]}],
                executor=[{"code": "BROKEN"}],
                # 修复也没能改好：同一个动作不能无限重复
                repair=[{"code": "BROKEN"}],
                decider=[decision("repair")],
            ),
        )

        actions = [item["action"] for item in result["decision_history"]]
        assert actions.count("repair") <= 2
        assert actions[-1] in ("skip_task", "finish")
        assert result["analysis_status"] == "partial"

    def test_model_budget_stops_the_loop(self, database, tmp_path, monkeypatch) -> None:
        install_tool(monkeypatch)
        result, factory = run(
            database,
            tmp_path,
            run_id="h5-budget",
            queues=queues_with(
                planner=[{"tasks": [task("task_1")]}],
                decider=[decision("repair")],
            ),
            budget={"max_llm_calls": 3},
        )

        usage = (result["execution_budget"] or {}).get("usage") or {}
        assert usage.get("llm_calls", 0) <= 3
        assert result["analysis_status"] in ("partial", "failed")


class TestDecisionContract:
    def test_an_unknown_action_ends_safely(self, database, tmp_path, monkeypatch) -> None:
        install_tool(monkeypatch)
        result, factory = run(
            database,
            tmp_path,
            run_id="h5-unknown",
            queues=queues_with(
                planner=[{"tasks": [task("task_1")]}],
                decider=[{"action": "delete_everything", "reason": "?"}],
            ),
        )

        assert result["decision_history"][0]["action"] == "finish"
        assert result["decision_history"][0]["forced"] is True

    def test_every_decision_is_a_tracked_model_request(self, database, tmp_path, monkeypatch) -> None:
        install_tool(monkeypatch)
        result, factory = run(
            database,
            tmp_path,
            run_id="h5-accounting",
            queues=queues_with(
                planner=[{"tasks": [task("task_1")]}],
                decider=[decision("continue"), decision("finish")],
            ),
        )

        usage = (result["execution_budget"] or {}).get("usage") or {}
        assert usage["llm_calls"] == len(factory.calls)

        ledger = [
            json.loads(line)
            for line in (tmp_path / "llm_ledger" / "h5-accounting.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        started = [item for item in ledger if item["phase"] == "started"]
        completed = [item for item in ledger if item["phase"] == "completed"]
        assert len(started) == len(completed) == len(factory.calls)
