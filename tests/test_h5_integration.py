"""H1～H5 联合验证：动态循环在真实链路里的行为与成本。

重点不是"新增了决策节点"，而是同一份任务在开启与关闭动态循环时**走了不同
的路径**，并且开启后的代价有明确上限。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

import storage
from services.analysis_service import run_analysis
from tests.scripted_queue import ScriptedQueueFactory, decision, task
from tests.test_report_approval import build_database

VIP_DOC = (
    "# VIP 客户\n\nVIP 客户指年消费金额超过 10000 元的客户。\n"
    "VIP 客户按客户类型细分后需要单独统计。\n"
)


@pytest.fixture(autouse=True)
def _local_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("INSIGHTFLOW_STORAGE", "local")
    monkeypatch.setenv("INSIGHTFLOW_LOCAL_DB", str(tmp_path / "conversations.sqlite"))
    monkeypatch.setenv("INSIGHTFLOW_OWNER", "h5-integration")
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
        if "BOOM" in payload:
            return {
                "success": False,
                "result": None,
                "error": "沙箱不可用",
                "error_kind": "unavailable",
            }
        if "DENIED" in payload:
            return {
                "success": False,
                "result": None,
                "error": "权限被拒绝",
                "error_kind": "sql_permission",
            }
        return {"success": True, "result": [{"n": 3}], "error": None, "error_kind": None}

    monkeypatch.setattr(tool_dispatcher, "invoke_tool", fake_invoke)


def base_queues(**overrides: list[Any]) -> dict[str, list[Any]]:
    queues: dict[str, list[Any]] = {
        "executor": [{"code": "SELECT COUNT(*) AS n FROM customers"}],
        "repair": [{"code": "SELECT COUNT(*) AS n FROM customers"}],
        "refine": [{"code": "SELECT COUNT(*) AS n FROM customers"}],
        "critic": [{"validation_passed": True, "feedback": "ok"}],
        "visualizer": [{"charts": []}],
        "reporter": ["# 报告\n\n共 3 位客户。\n"],
    }
    queues.update(overrides)
    return queues


def execute(
    database: Path,
    tmp_path: Path,
    *,
    run_id: str,
    queues: dict[str, list[Any]],
    enable: bool,
    knowledge: list[str] | None = None,
):
    factory = ScriptedQueueFactory(run_id, queues)
    started = time.perf_counter()
    result = run_analysis(
        run_id=run_id,
        data_sources=[str(database)],
        knowledge_sources=knowledge or [],
        model_factory=factory.build(),
        output_root=tmp_path,
        enable_dynamic_loop=enable,
        dynamic_limits={"max_decision_steps": 8, "max_runtime_replans": 2},
    )
    return result, factory, round((time.perf_counter() - started) * 1000, 1)


def metrics(result: dict[str, Any], factory, duration_ms: float) -> dict[str, Any]:
    usage = (result.get("execution_budget") or {}).get("usage") or {}
    return {
        "analysis_status": result.get("analysis_status"),
        "decisions": len(result.get("decision_history") or []),
        "decision_calls": factory.calls.count("decider"),
        "llm_calls": usage.get("llm_calls"),
        "tool_calls": usage.get("tool_calls"),
        "runtime_replans": result.get("runtime_replans"),
        "extra_retrievals": result.get("extra_retrievals"),
        "executions": len(result.get("execution_results") or []),
        "termination": result.get("termination_reason"),
        "duration_ms": duration_ms,
    }


class TestScenarioBRemediation:
    """同一个"第一次结果不合格"的任务：固定编排与动态闭环走不同的路。"""

    def test_dynamic_loop_changes_the_path_when_evidence_is_missing(
        self, database, tmp_path
    ) -> None:
        knowledge = tmp_path / "vip.md"
        knowledge.write_text(VIP_DOC, encoding="utf-8")
        common = {
            "planner": [{"tasks": [task("task_1", description="统计 VIP 客户数量")]}],
            "executor": [{"code": "SELECT COUNT(*) AS n FROM customers"}],
            "critic": [
                {"validation_passed": False, "feedback": "缺少 VIP 客户的判定口径"},
                {"validation_passed": True, "feedback": "ok"},
            ],
        }

        # H5 关闭：固定编排走 refine（不会补检索）
        off, off_factory, off_ms = execute(
            database,
            tmp_path,
            run_id="int-off",
            queues={**base_queues(), **common},
            enable=False,
            knowledge=[str(knowledge)],
        )
        # H5 开启：观察到证据不足，先补充检索再重新执行
        on, on_factory, on_ms = execute(
            database,
            tmp_path,
            run_id="int-on",
            queues={
                **base_queues(),
                **common,
                "decider": [
                    decision("retrieve_more"),
                    decision("continue"),
                    decision("finish"),
                ],
            },
            enable=True,
            knowledge=[str(knowledge)],
        )

        off_metrics = metrics(off, off_factory, off_ms)
        on_metrics = metrics(on, on_factory, on_ms)

        # 两条路径确实不同：一个 refine，一个补检索
        assert "refine" in off_factory.calls
        assert "refine" not in on_factory.calls
        assert on["extra_retrievals"] == 1
        assert on_factory.calls.count("executor") == 2, on_factory.calls

        # 动态路径多了一次决策与一次检索，但仍在有界范围内
        assert on_metrics["decision_calls"] == 2
        assert on_metrics["llm_calls"] > off_metrics["llm_calls"]
        assert on_metrics["llm_calls"] - off_metrics["llm_calls"] <= 3
        assert on["analysis_status"] == "completed"
        assert off["analysis_status"] == "completed"


class TestScenarioCPlanChange:
    def test_runtime_replan_completes_where_fixed_flow_reports_partial(
        self, database, tmp_path
    ) -> None:
        tasks = [
            task("task_1", description="统计客户总数"),
            task("task_2", description="按不存在的维度细分"),
        ]
        common = {
            "planner": [
                {"tasks": tasks},
                # 修订只给能做的任务；H5 关闭时不会走到这里
                {"tasks": [task("task_3", description="按地区细分")]},
            ],
            "executor": [
                {"code": "SELECT COUNT(*) AS n FROM customers"},
                # 环境类失败：重试不会好转，固定编排只能记失败
                {"code": "BOOM"},
                {"code": "SELECT COUNT(*) AS n FROM customers"},
            ],
        }

        off, off_factory, off_ms = execute(
            database,
            tmp_path,
            run_id="int-replan-off",
            queues={**base_queues(), **common},
            enable=False,
        )
        on, on_factory, on_ms = execute(
            database,
            tmp_path,
            run_id="int-replan-on",
            queues={
                **base_queues(),
                **common,
                "decider": [
                    decision("continue"),
                    decision("replan_remaining"),
                    decision("continue"),
                ],
            },
            enable=True,
        )

        # 固定编排：任务失败 → partial，不会调整计划
        assert off["analysis_status"] == "partial"
        assert off["plan_revision"] == 0

        # 动态闭环：运行期修订 → 重新校验 → 完成
        assert on["analysis_status"] == "completed"
        assert on["plan_revision"] == 1
        assert on["runtime_replans"] == 1
        assert [item["task_id"] for item in on["analysis_plan"]] == ["task_1", "task_3"]


class TestScenarioFSafety:
    def test_a_permission_error_converges_without_extra_tool_calls(
        self, database, tmp_path
    ) -> None:
        result, factory, _ = execute(
            database,
            tmp_path,
            run_id="int-permission",
            queues=base_queues(
                planner=[{"tasks": [task("task_1")]}],
                executor=[{"code": "DENIED"}],
                decider=[decision("repair"), decision("finish")],
            ),
            enable=True,
        )

        usage = (result.get("execution_budget") or {}).get("usage") or {}
        assert result["analysis_status"] == "partial"
        assert usage.get("tool_calls") == 1, "权限被拒后仍在调用工具"
        assert "repair" not in factory.calls
        assert result["skipped_tasks"][0]["task_id"] == "task_1"

    def test_an_unauthorized_table_never_reaches_execution(self, database, tmp_path) -> None:
        result, factory, _ = execute(
            database,
            tmp_path,
            run_id="int-unauthorized",
            queues=base_queues(
                planner=[{"tasks": [task("task_1", tables=["customers.secret"])]}],
                decider=[decision("continue")],
            ),
            enable=True,
        )

        assert result["execution_results"] == []
        assert result["analysis_status"] == "failed"
        assert "executor" not in factory.calls


class TestScenarioDConversation:
    """多轮 + 长期记忆 + 动态决策：三者协作但互不替代。"""

    def test_a_dynamic_turn_uses_confirmed_memory_and_previous_result(
        self, database, tmp_path
    ) -> None:
        memory = {
            "memory_id": "mem_1",
            "kind": "preference",
            "key": "currency",
            "value": "今后销售额统一用人民币元表示。",
            "scope": "global",
            "status": "confirmed",
        }
        conversation = {
            "recent_messages": [
                {"role": "user", "content": "分析最近三个月的销售额"},
                {"role": "assistant", "content": "共 3 位客户。"},
            ],
            "previous_turns": [
                {"seq": 1, "status": "completed", "summary": "上一轮统计了客户数量"}
            ],
            "dataset_refs": [str(database)],
        }

        factory = ScriptedQueueFactory(
            "int-conversation",
            {
                **base_queues(),
                "planner": [{"tasks": [task("task_1", description="按地区统计销售额")]}],
                "decider": [decision("continue")],
            },
        )
        result = run_analysis(
            run_id="int-conversation",
            data_sources=[str(database)],
            model_factory=factory.build(),
            output_root=tmp_path,
            enable_dynamic_loop=True,
            conversation_context=conversation,
            memories=[memory],
        )

        assert result["analysis_status"] == "completed"
        assert result["current_action"] == "continue"

        planner_prompt = factory.prompts["planner"][0]
        # 已确认的偏好与上一轮真实摘要都进入了规划上下文
        assert "人民币" in planner_prompt
        assert "上一轮统计了客户数量" in planner_prompt
        # 长期记忆不会改变工具权限：仍然是只读 SQL
        assert result["execution_results"][0]["tool_used"] == "sql"


class TestOnOffComparison:
    def test_the_same_clean_task_costs_more_when_the_loop_is_on(
        self, database, tmp_path
    ) -> None:
        """诚实记录：干净的普通任务上，动态循环只是额外开销。"""
        common = {"planner": [{"tasks": [task("task_1")]}]}

        off, off_factory, off_ms = execute(
            database,
            tmp_path,
            run_id="cmp-off",
            queues={**base_queues(), **common},
            enable=False,
        )
        on, on_factory, on_ms = execute(
            database,
            tmp_path,
            run_id="cmp-on",
            queues={**base_queues(), **common, "decider": [decision("continue")]},
            enable=True,
        )

        off_metrics = metrics(off, off_factory, off_ms)
        on_metrics = metrics(on, on_factory, on_ms)

        assert off_metrics["analysis_status"] == on_metrics["analysis_status"] == "completed"
        assert off_metrics["decisions"] == 0
        assert on_metrics["decisions"] == 1
        # 开启后至少多一次模型请求，不会更省
        assert on_metrics["llm_calls"] == off_metrics["llm_calls"] + 1
        assert on_metrics["tool_calls"] == off_metrics["tool_calls"]

        target = os.getenv("INSIGHTFLOW_H5_METRICS_OUT")
        if target:
            path = Path(target)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {"dynamic_loop_off": off_metrics, "dynamic_loop_on": on_metrics},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
