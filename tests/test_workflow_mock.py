"""端到端 Mock 工作流与 V1 基线。

用假模型与白名单工具响应跑完整条 LangGraph 流程，验证：
- 一次运行能走到产出报告；
- 各 Agent 的调用身份被正确传递（ModelGate 追踪的前提）；
- 循环边界真的会终止；
- 基线可度量项（调用次数、工具次数、耗时、产物）被记录下来。

不伪造 token 与成本：假模型不返回 usage，因此这些指标记为不可用而不是 0。
这里不验收Docker代码执行/真实绘图；SQLite只读查询仍用真实工具。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from core.config import Settings
from core.llm import CallContext, ModelFactory
from services.analysis_service import run_analysis
from tools.tool_registry import clear_tool_cache

from tests.conftest import CALL_RECORDS, CUSTOMERS_CSV, ORDERS_CSV, SALES_SQLITE

DEFAULT_REPLIES: dict[str, Any] = {
    "planner": {
        "tasks": [
            {
                "task_id": "task_1",
                "description": "统计客户数量",
                "target_tables": ["customers"],
                "expected_output": "客户总数",
            }
        ]
    },
    "executor": {"code": "result = len(df)"},
    "critic": {"validation_passed": True, "feedback": "结果符合任务要求"},
    "visualizer": {"charts": []},
    "reporter": "# Analysis Report\n\n## Overview\n共 5 位客户。\n",
}


class ScriptedAgentModel(BaseChatModel):
    """按 agent_name 返回预设内容的假模型。"""

    agent_name: str = ""
    record_key: str = ""
    replies: dict[str, Any] = {}

    @property
    def _llm_type(self) -> str:
        return f"scripted-{self.agent_name}"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        CALL_RECORDS.setdefault(self.record_key, []).append(list(messages))

        reply = self.replies[self.agent_name]
        content = (
            reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)
        )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])


def build_factory(
    run_id: str,
    *,
    agent_replies: dict[str, Any] | None = None,
) -> ModelFactory:
    """构造按 Agent 分发的假模型工厂；调用记录写入 CALL_RECORDS[run_id]。"""
    replies = {**DEFAULT_REPLIES, **(agent_replies or {})}
    CALL_RECORDS[run_id] = []

    def resolver(agent_name: str) -> ScriptedAgentModel:
        if agent_name not in replies:
            raise AssertionError(f"未配置 {agent_name} 的假响应")
        return ScriptedAgentModel(
            agent_name=agent_name, record_key=run_id, replies=replies
        )

    return ModelFactory(
        settings=Settings(
            api_key="sk-fake",
            base_url="http://127.0.0.1:8000/v1",
            model="deepseek-chat",
        ),
        context=CallContext(run_id=run_id, client_app="insightflow-agent2"),
        resolver=resolver,
    )


def agent_call_count(run_id: str, agent_name: str) -> int:
    """统计某 Agent 被调用次数。

    LangChain 只在请求时解析 resolver，因此记录键是 (run_id, agent_name) 的形态，
    这里按 agent 前缀过滤。
    """
    total = 0
    for messages in CALL_RECORDS.get(run_id, []):
        agent = getattr(messages[0], "additional_kwargs", {}).get("agent_name")
        if agent is None or agent == agent_name:
            total += 1
    return total


@pytest.fixture(autouse=True)
def _isolate_outputs(tmp_path, monkeypatch):
    """产物与日志写入临时目录，避免污染仓库。"""
    monkeypatch.chdir(tmp_path)
    from harness import tool_dispatcher
    from tests.support.scripted_runtime_tools import invoke_scripted_tool, no_browser_pdf
    import nodes

    real_invoke = tool_dispatcher.invoke_tool
    def known_tool(tool_name, tables, arguments, *, snapshots=None):
        # Workflow/identity assertions do not exercise Python isolation. Only
        # these fixed replies are modeled; no model-provided code is executed.
        if tool_name == "sql":
            return real_invoke(tool_name, tables, arguments, snapshots=snapshots)
        if arguments.get("code", "").strip() == "result = undefined_name":
            return {"success": False, "result": None, "error": "NameError: undefined_name", "error_kind": "execution_error"}
        return invoke_scripted_tool(tool_name, tables, arguments, snapshots=snapshots)
    monkeypatch.setattr(tool_dispatcher, "invoke_tool", known_tool)
    monkeypatch.setattr(nodes, "save_pdf_report", no_browser_pdf)
    clear_tool_cache()
    yield tmp_path
    CALL_RECORDS.clear()
    clear_tool_cache()


class TestEndToEndMockWorkflow:
    def test_single_table_analysis_completes(self) -> None:
        factory = build_factory("run_e2e_single")

        result = run_analysis(
            run_id="run_e2e_single",
            data_sources=[str(CUSTOMERS_CSV)],
            analysis_focus="统计客户数量",
            model_factory=factory,
        )

        assert result["run_id"] == "run_e2e_single"
        assert len(result["analysis_plan"]) == 1
        assert result["failed_tasks"] == []

    def test_execution_produced_a_result(self) -> None:
        factory = build_factory("run_e2e_exec")

        result = run_analysis(
            run_id="run_e2e_exec",
            data_sources=[str(CUSTOMERS_CSV)],
            analysis_focus="统计客户数量",
            model_factory=factory,
        )

        executions = result["execution_results"]
        assert executions, "至少应有一条执行结果"
        assert executions[-1]["success"] is True
        assert executions[-1]["result"] == 5
        assert executions[-1]["tool_used"] == "python"

    def test_each_agent_receives_its_own_model(self) -> None:
        """追踪身份按 Agent 区分——这是 ModelGate 能分 Agent 统计的前提。"""
        factory = build_factory("run_e2e_agents")

        run_analysis(
            run_id="run_e2e_agents",
            data_sources=[str(CUSTOMERS_CSV)],
            analysis_focus="统计客户数量",
            model_factory=factory,
        )

        calls = agent_call_count("run_e2e_agents", "planner")
        assert calls >= 1, "planner 应被调用"

    def test_report_artifact_is_written(self) -> None:
        factory = build_factory("run_e2e_report")

        result = run_analysis(
            run_id="run_e2e_report",
            data_sources=[str(CUSTOMERS_CSV)],
            analysis_focus="统计客户数量",
            model_factory=factory,
        )

        assert result["report_content"].startswith("# Analysis Report")
        report_path = Path(result["report_path"])
        assert report_path.exists()

        run_dir = Path("outputs") / "run_e2e_report"
        assert (run_dir / "analysis_report.md").exists()
        assert (run_dir / "analysis_report.html").exists()

    def test_run_log_is_written(self) -> None:
        factory = build_factory("run_e2e_log")

        run_analysis(
            run_id="run_e2e_log",
            data_sources=[str(CUSTOMERS_CSV)],
            analysis_focus="统计客户数量",
            model_factory=factory,
        )

        log_path = Path("outputs") / "logs" / "run_e2e_log.jsonl"
        assert log_path.exists()

        events = [
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
        ]
        names = {event["event"] for event in events}
        assert "run_started" in names
        assert "run_completed" in names

    def test_two_tables_can_be_analysed_together(self) -> None:
        factory = build_factory("run_e2e_multi")

        result = run_analysis(
            run_id="run_e2e_multi",
            data_sources=[str(CUSTOMERS_CSV), str(ORDERS_CSV)],
            analysis_focus="按地区统计客户与订单",
            model_factory=factory,
        )

        assert result["execution_results"], "应有执行结果"
        assert result["failed_tasks"] == []


class TestBoundedLoops:
    def test_repair_loop_stops_at_limit(self) -> None:
        """执行持续失败时，重试达到上限必须转为失败任务而不是无限循环。"""
        broken = {"code": "result = undefined_name"}
        factory = build_factory(
            "run_retry",
            agent_replies={
                "executor": broken,
                "repair": broken,
                "critic": {"validation_passed": False, "feedback": "不正确"},
                "refine": broken,
            },
        )

        result = run_analysis(
            run_id="run_retry",
            data_sources=[str(CUSTOMERS_CSV)],
            analysis_focus="统计客户数量",
            model_factory=factory,
        )

        assert result["failed_tasks"], "最终应记录失败任务"
        repaired = sum(
            1
            for item in result["execution_results"]
            if item["code"] == "result = undefined_name"
        )
        # MAX_RETRIES=3 -> 首次 + 最多 3 次修复
        assert repaired <= 4

    def test_planner_rejecting_is_not_reported_as_success(self) -> None:
        """空计划不能因为 failed_tasks 为空就被当成成功。

        工作流会正常走完（不崩溃），但必须显式标注"未产出分析任务"，
        否则下游只看到空失败列表就会误判为分析完成。
        """
        factory = build_factory(
            "run_empty_plan", agent_replies={"planner": {"tasks": []}}
        )

        result = run_analysis(
            run_id="run_empty_plan",
            data_sources=[str(CUSTOMERS_CSV)],
            model_factory=factory,
        )

        assert result["analysis_plan"] == []
        assert result["failed_tasks"] == []
        assert result["analysis_status"] == "no_tasks"
        assert result["status_reason"], "必须给出未完成原因"

    def test_successful_run_is_marked_completed(self) -> None:
        factory = build_factory("run_completed_status")

        result = run_analysis(
            run_id="run_completed_status",
            data_sources=[str(CUSTOMERS_CSV)],
            analysis_focus="统计客户数量",
            model_factory=factory,
        )

        assert result["analysis_status"] == "completed"
        assert result["status_reason"] is None

    def test_partial_failure_is_marked_partial(self) -> None:
        """有计划但任务最终失败时记为 partial，并给出失败数量。"""
        broken = {"code": "result = undefined_name"}
        factory = build_factory(
            "run_partial",
            agent_replies={
                "executor": broken,
                "repair": broken,
                "critic": {"validation_passed": False, "feedback": "不正确"},
                "refine": broken,
            },
        )

        result = run_analysis(
            run_id="run_partial",
            data_sources=[str(CUSTOMERS_CSV)],
            analysis_focus="统计客户数量",
            model_factory=factory,
        )

        assert result["failed_tasks"], "应有失败任务"
        assert result["analysis_status"] == "partial"
        assert "失败" in result["status_reason"]

    def test_status_is_recorded_in_run_log(self) -> None:
        """结果语义要进 JSONL，便于事后审计而不是只看返回值。"""
        factory = build_factory("run_log_status")

        run_analysis(
            run_id="run_log_status",
            data_sources=[str(CUSTOMERS_CSV)],
            analysis_focus="统计客户数量",
            model_factory=factory,
        )

        log_path = Path("outputs") / "logs" / "run_log_status.jsonl"
        events = [
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
        ]
        finished = [e for e in events if e["event"] == "analysis_finished"]
        assert finished, "应有 analysis_finished 事件"
        assert finished[0]["status"] == "completed"


class TestBaselineMetrics:
    """V1 基线：记录可稳定测量的量，作为后续对比的基准。"""

    def test_baseline_is_measurable(self, tmp_path) -> None:
        factory = build_factory("run_baseline")

        started = time.perf_counter()
        result = run_analysis(
            run_id="run_baseline",
            data_sources=[str(CUSTOMERS_CSV), str(ORDERS_CSV)],
            analysis_focus="按地区统计客户与订单",
            model_factory=factory,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000

        executions = result["execution_results"]
        baseline = {
            "run_id": result["run_id"],
            "llm_calls_total": len(CALL_RECORDS.get("run_baseline", [])),
            "tool_calls": len(executions),
            "tools_used": sorted({item["tool_used"] for item in executions}),
            "tasks_planned": len(result["analysis_plan"]),
            "tasks_failed": len(result["failed_tasks"]),
            "artifacts": sorted(Path(a).name for a in result["artifacts"]),
            "elapsed_ms": round(elapsed_ms, 1),
            # 假模型不返回 usage，这些指标不可用而不是 0
            "prompt_tokens": None,
            "completion_tokens": None,
            "estimated_cost_usd": None,
        }

        (tmp_path / "baseline.json").write_text(
            json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # 基线必须可复现：同样的 Mock 配置得到同样的调用结构
        assert baseline["llm_calls_total"] >= 4
        assert baseline["tool_calls"] >= 1
        assert baseline["tasks_failed"] == 0
        assert baseline["prompt_tokens"] is None
        assert baseline["estimated_cost_usd"] is None
        assert "analysis_report.md" in baseline["artifacts"]

    def test_sqlite_source_is_routed(self) -> None:
        """sqlite 资源不预载 DataFrame，路由应给出 sql 或退回 python 而不崩溃。"""
        factory = build_factory(
            "run_sqlite",
            agent_replies={
                "planner": {
                    "tasks": [
                        {
                            "task_id": "task_1",
                            "description": "统计订单数",
                            "target_tables": ["sales.orders"],
                            "expected_output": "订单总数",
                        }
                    ]
                },
                "executor": {"code": "SELECT COUNT(*) AS n FROM orders"},
            },
        )

        result = run_analysis(
            run_id="run_sqlite",
            data_sources=[str(SALES_SQLITE)],
            analysis_focus="统计订单数",
            model_factory=factory,
        )

        assert result["execution_results"], "应有执行结果"
        assert result["execution_results"][-1]["tool_used"] == "sql"
