"""H2 + H3 联合验证：校验通过的计划与任务级上下文必须指向同一批表。

单个阶段的测试各自成立还不够：如果 Validator 用的是 A 集合、Context Manager
选的是 B 集合，两者都"通过"，执行时却会看到不该看的表。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from services.analysis_service import run_analysis
from tests.test_e2e_business_scenarios import ScriptedFactory
from tests.test_report_approval import REPLIES

TABLES = ("orders", "customers", "products", "regions", "returns")


def build_dataset(path: Path) -> Path:
    with sqlite3.connect(path) as connection:
        for index, table in enumerate(TABLES):
            connection.execute(f"CREATE TABLE {table} (id INTEGER, amount REAL)")
            connection.execute(f"INSERT INTO {table} VALUES (?, ?)", (index, index * 2.0))
    return path


def resource(name: str) -> str:
    return f"shop.{name}"


def plan_of(tables: list[str], *, task_id: str = "task_1") -> dict[str, Any]:
    return {
        "tasks": [
            {
                "task_id": task_id,
                "description": f"统计 {tables[0]}",
                "target_tables": tables,
                "expected_output": "行数",
            }
        ]
    }


@pytest.fixture(autouse=True)
def _fake_tool(monkeypatch: pytest.MonkeyPatch):
    from harness import tool_dispatcher

    def fake_invoke(tool_name, tables, arguments, *, snapshots=None):
        if tool_name == "plot":
            return {"success": True, "result": "chart.png", "error": None, "error_kind": None}
        return {"success": True, "result": [{"n": 1}], "error": None, "error_kind": None}

    monkeypatch.setattr(tool_dispatcher, "invoke_tool", fake_invoke)


@pytest.fixture(autouse=True)
def _restore_replies():
    import copy

    original = copy.deepcopy(REPLIES)
    yield
    REPLIES.clear()
    REPLIES.update(copy.deepcopy(original))


def read_events(tmp_path: Path, run_id: str, event: str) -> list[dict]:
    path = tmp_path / "logs" / f"{run_id}.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("event") == event
    ]


class TestValidatedPlanMatchesSelectedContext:
    def test_each_task_sees_exactly_the_tables_the_validator_accepted(
        self, tmp_path: Path
    ) -> None:
        from tests.test_budget_real_entry_points import CountingFactory

        source = build_dataset(tmp_path / "shop.sqlite")
        plans = [
            plan_of([resource("invented")]),  # 第一次被拒，触发重新规划
            {
                "tasks": [
                    {
                        "task_id": "task_1",
                        "description": "订单量",
                        "target_tables": [resource("orders")],
                        "expected_output": "行数",
                    },
                    {
                        "task_id": "task_2",
                        "description": "区域量",
                        "target_tables": [resource("regions")],
                        "expected_output": "行数",
                    },
                ]
            },
        ]
        replies = {
            "executor": {"code": "SELECT COUNT(*) AS n FROM orders"},
            "critic": {"validation_passed": True, "feedback": "ok"},
            "visualizer": {"charts": []},
            "reporter": "# Report\n",
        }
        run_id = "joint-tables"
        counting = CountingFactory(run_id)
        counting.base_replies = replies
        seen = {"planner": 0}

        def responder(agent_name: str, total: int):
            if agent_name != "planner":
                return None
            payload = plans[min(seen["planner"], len(plans) - 1)]
            seen["planner"] += 1
            from langchain_core.messages import AIMessage
            from langchain_core.outputs import ChatGeneration, ChatResult

            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content=json.dumps(payload, ensure_ascii=False)
                        )
                    )
                ]
            )

        counting.responder = responder

        # 其余 Agent 仍用默认脚本回复
        REPLIES.update(replies)

        result = run_analysis(
            run_id=run_id,
            data_sources=[str(source)],
            model_factory=counting.build(),
            output_root=tmp_path,
        )

        assert result["analysis_status"] == "completed", result.get("status_reason")
        assert seen["planner"] == 2, "第一次计划应被拒绝并重新规划"

        events = read_events(tmp_path, run_id, "llm_context")
        kept = {
            event["node"]: event["details"]["context"]["kept_tables"] for event in events
        }

        # Planner 需要全局视图；执行类 Agent 只看到被校验通过的那张表
        assert sorted(kept["planner"]) == sorted(resource(name) for name in TABLES)
        executor_kept = [
            event["details"]["context"]["kept_tables"]
            for event in events
            if event["node"] == "executor"
        ]
        assert sorted(executor_kept) == [[resource("orders")], [resource("regions")]]

        # 校验器接受的计划就是实际执行的计划
        accepted = [task["target_tables"] for task in result["analysis_plan"]]
        assert accepted == [[resource("orders")], [resource("regions")]]


class TestReplanningDoesNotResetBudgetOrLedger:
    def test_budget_and_ledger_survive_a_replan(self, tmp_path: Path) -> None:
        from tests.test_budget_real_entry_points import CountingFactory

        source = build_dataset(tmp_path / "shop.sqlite")
        run_id = "joint-budget"
        plans = [plan_of([resource("invented")]), plan_of([resource("orders")])]
        counting = CountingFactory(run_id)
        seen = {"planner": 0}

        def responder(agent_name: str, total: int):
            if agent_name != "planner":
                return None
            payload = plans[min(seen["planner"], len(plans) - 1)]
            seen["planner"] += 1
            from langchain_core.messages import AIMessage
            from langchain_core.outputs import ChatGeneration, ChatResult

            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content=json.dumps(payload, ensure_ascii=False)
                        )
                    )
                ]
            )

        counting.responder = responder
        REPLIES.update(
            {
                "executor": {"code": "SELECT COUNT(*) AS n FROM orders"},
                "critic": {"validation_passed": True, "feedback": "ok"},
                "visualizer": {"charts": []},
                "reporter": "# Report\n",
            }
        )

        result = run_analysis(
            run_id=run_id,
            data_sources=[str(source)],
            model_factory=counting.build(),
            output_root=tmp_path,
            budget={"max_llm_calls": 20},
        )

        usage = (result.get("execution_budget") or {}).get("usage") or {}
        # 重新规划是额外的真实请求，必须计入：计数口径与  一致
        assert usage["llm_calls"] == counting.count > 0
        assert result["plan_attempts"] == 2

        ledger = [
            json.loads(line)
            for line in (tmp_path / "llm_ledger" / f"{run_id}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        started = [item for item in ledger if item["phase"] == "started"]
        completed = [item for item in ledger if item["phase"] == "completed"]
        # 账本按逻辑调用计；重规划不是"复用"，因此有两条 planner 记录
        planner_ops = {item["operation_id"] for item in started if item["node"] == "planner"}
        assert len(planner_ops) == 2
        assert len(started) == len(completed)


class TestKnowledgeAndReportStillFlow:
    def test_rag_knowledge_survives_validation(self, tmp_path: Path) -> None:
        from tests.conftest import BUSINESS_RULES_MD

        source = build_dataset(tmp_path / "shop.sqlite")
        run_id = "joint-rag"
        replies = {
            "planner": plan_of([resource("orders")]),
            "executor": {"code": "SELECT COUNT(*) AS n FROM orders"},
            "critic": {"validation_passed": True, "feedback": "ok"},
            "visualizer": {"charts": []},
            "reporter": "# Report\n\n共 1 行。\n",
        }

        result = run_analysis(
            run_id=run_id,
            data_sources=[str(source)],
            knowledge_sources=[str(BUSINESS_RULES_MD)],
            model_factory=ScriptedFactory(run_id, replies).build(),
            output_root=tmp_path,
            analysis_focus="高价值客户",
        )

        assert result["analysis_status"] == "completed", result.get("status_reason")
        assert result["retrieved_knowledge"], "业务知识没有被检索到"
        assert result["report_content"].strip(), "报告为空"

    def test_the_chart_path_still_completes(self, tmp_path: Path) -> None:
        source = build_dataset(tmp_path / "shop.sqlite")
        run_id = "joint-plot"
        replies = {
            "planner": plan_of([resource("orders")]),
            "executor": {"code": "SELECT COUNT(*) AS n FROM orders"},
            "critic": {"validation_passed": True, "feedback": "ok"},
            "visualizer": {
                "charts": [
                    {
                        "chart_id": "chart_1",
                        "source_task_id": "task_1",
                        "chart_type": "bar",
                        "title": "按区域",
                        "description": "订单",
                        "target_tables": [resource("orders")],
                        "expected_output": "柱状图",
                    }
                ]
            },
            "plotter": {"code": "print(1)"},
            "plot_repair": {"code": "print(1)"},
            "reporter": "# Report\n\n共 1 行。\n",
        }

        result = run_analysis(
            run_id=run_id,
            data_sources=[str(source)],
            model_factory=ScriptedFactory(run_id, replies).build(),
            output_root=tmp_path,
        )

        assert result["analysis_status"] == "completed", result.get("status_reason")
        assert result["visualization_results"], "图表结果缺失"

        events = read_events(tmp_path, run_id, "llm_context")
        plotter_kept = [
            event["details"]["context"]["kept_tables"]
            for event in events
            if event["node"] == "plotter"
        ]
        assert plotter_kept == [[resource("orders")]]
