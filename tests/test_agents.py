"""Agent 层：模型注入、输出契约与重试。

这些用例取代原先 tests/ 下的打印脚本。它们不访问网络——`network_guard`
会在 socket 层拦住真实上游，因此"忘了注入假模型"会直接失败而不是产生付费调用。
"""

from __future__ import annotations

import pytest
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agents.critic import review_execution_result
from agents.executor import generate_execution_code
from agents.planner import create_analysis_plan
from agents.repair import repair_execution_code
from agents.visualizer import create_visualization_plan
from utils.llm_retry import StructuredOutputError, invoke_json_with_retry

TASK = {
    "task_id": "task_1",
    "description": "按地区统计高价值客户",
    "target_tables": ["customers"],
    "expected_output": "每个地区的高价值客户数",
}
CONTEXT = {
    "customers": {
        "schema": {"table_name": "customers", "columns": {"region": "object"}},
        "row_count": 5,
        "context_level": "full",
        "data": [{"region": "East"}],
    }
}


class TestPlanner:
    def test_returns_task_list(self, scripted_model) -> None:
        model, _ = scripted_model(
            [
                {
                    "tasks": [
                        {
                            "task_id": "task_1",
                            "description": "d",
                            "target_tables": ["customers"],
                            "expected_output": "e",
                        }
                    ]
                }
            ]
        )

        plan = create_analysis_plan(contexts=CONTEXT, llm=model)

        assert isinstance(plan, list)
        assert plan[0]["task_id"] == "task_1"

    def test_prompt_carries_business_knowledge(self, scripted_model) -> None:
        """业务规则必须进入 prompt，否则接地约束无从生效。"""
        model, calls = scripted_model([{"tasks": []}])

        create_analysis_plan(
            contexts=CONTEXT,
            retrieved_knowledge=["High-value customer = total > $400"],
            llm=model,
        )

        prompt = str(calls[0])
        assert "High-value customer = total > $400" in prompt

    def test_missing_tasks_field_is_retried(self, scripted_model) -> None:
        """JSON 合法但缺 tasks 时必须重试，而不是抛 KeyError 终止工作流。"""
        model, calls = scripted_model(
            [
                {"unexpected": "shape"},
                {"tasks": []},
            ]
        )

        plan = create_analysis_plan(contexts=CONTEXT, llm=model)

        assert plan == []
        assert len(calls) == 2

    def test_retry_prompt_includes_previous_failure(self, scripted_model) -> None:
        """第二次尝试必须带上上一次的失败原因，否则 temperature=0 下会重复同样的错误。"""
        model, calls = scripted_model([{"bad": 1}, {"tasks": []}])

        create_analysis_plan(contexts=CONTEXT, llm=model)

        assert "tasks" in str(calls[1])
        assert len(str(calls[1])) > len(str(calls[0]))

    def test_non_json_output_exhausts_attempts(self, scripted_model) -> None:
        """模型持续返回非 JSON 时，重试次数用尽后给出明确失败。"""
        model, calls = scripted_model([""], simulate_parse_failure=True)

        with pytest.raises(StructuredOutputError):
            create_analysis_plan(contexts=CONTEXT, llm=model)

        assert len(calls) == 3


class TestExecutor:
    def test_returns_generated_code(self, scripted_model) -> None:
        model, _ = scripted_model([{"code": "result = len(df)"}])

        code = generate_execution_code(
            task=TASK, data_context=CONTEXT, route="python", llm=model
        )

        assert code == "result = len(df)"

    def test_missing_code_field_is_retried(self, scripted_model) -> None:
        model, calls = scripted_model([{"other": 1}, {"code": "result = 1"}])

        code = generate_execution_code(
            task=TASK, data_context=CONTEXT, route="python", llm=model
        )

        assert code == "result = 1"
        assert len(calls) == 2

    @pytest.mark.parametrize("route", ["python", "sql"])
    def test_route_is_forwarded_to_prompt(self, scripted_model, route: str) -> None:
        model, calls = scripted_model([{"code": "SELECT 1"}])

        generate_execution_code(
            task=TASK, data_context=CONTEXT, route=route, llm=model
        )

        assert route in str(calls[0])


class TestRepair:
    def test_returns_repaired_code(self, scripted_model) -> None:
        model, calls = scripted_model([{"code": "result = 2"}])

        code = repair_execution_code(
            task=TASK,
            data_context=CONTEXT,
            route="python",
            code="result = df['missing']",
            error="KeyError: missing",
            llm=model,
        )

        assert code == "result = 2"
        assert "KeyError: missing" in str(calls[0])

    def test_missing_code_field_is_retried(self, scripted_model) -> None:
        model, calls = scripted_model([{}, {"code": "result = 1"}])

        code = repair_execution_code(
            task=TASK,
            data_context=CONTEXT,
            route="python",
            code="x",
            error="boom",
            llm=model,
        )

        assert code == "result = 1"
        assert len(calls) == 2


class TestCritic:
    def test_returns_validation_verdict(self, scripted_model) -> None:
        model, _ = scripted_model(
            [{"validation_passed": True, "feedback": "符合业务规则"}]
        )

        review = review_execution_result(
            task=TASK,
            data_context=CONTEXT,
            execution_result={
                "task_id": "task_1",
                "tool_used": "python",
                "code": "result = 1",
                "success": True,
                "result": 1,
                "error": None,
            },
            llm=model,
        )

        assert review["validation_passed"] is True
        assert review["feedback"] == "符合业务规则"

    def test_missing_feedback_field_is_retried(self, scripted_model) -> None:
        model, calls = scripted_model(
            [
                {"validation_passed": True},
                {"validation_passed": False, "feedback": "未使用规定阈值"},
            ]
        )

        review = review_execution_result(
            task=TASK,
            data_context=CONTEXT,
            execution_result={
                "task_id": "task_1",
                "tool_used": "python",
                "code": "c",
                "success": True,
                "result": 1,
                "error": None,
            },
            llm=model,
        )

        assert review["validation_passed"] is False
        assert len(calls) == 2


class TestVisualizer:
    def test_returns_chart_plan(self, scripted_model) -> None:
        model, _ = scripted_model(
            [
                {
                    "charts": [
                        {
                            "chart_id": "chart_1",
                            "description": "d",
                            "source_task_id": "task_1",
                            "target_tables": ["customers"],
                            "chart_type": "bar",
                            "expected_output": "e",
                        }
                    ]
                }
            ]
        )

        charts = create_visualization_plan(
            analysis_plan=[TASK], execution_results=[], failed_tasks=[], llm=model
        )

        assert charts[0]["chart_id"] == "chart_1"

    def test_missing_charts_field_is_retried(self, scripted_model) -> None:
        model, calls = scripted_model([{"charts": None}, {"charts": []}])

        charts = create_visualization_plan(
            analysis_plan=[TASK], execution_results=[], failed_tasks=[], llm=model
        )

        assert charts == []
        assert len(calls) == 2


class TestRetryWrapper:
    def test_validate_exception_triggers_retry(self, scripted_model) -> None:
        model, calls = scripted_model([{"a": 1}, {"a": 2}])
        chain = ChatPromptTemplate.from_template("{x}") | model | JsonOutputParser()

        def validate(payload: dict) -> int:
            if payload["a"] != 2:
                raise ValueError("a must be 2")
            return payload["a"]

        result = invoke_json_with_retry(chain, {"x": "y"}, validate=validate)

        assert result == 2
        assert len(calls) == 2

    def test_raises_after_exhausting_attempts(self, scripted_model) -> None:
        model, calls = scripted_model([{"a": 1}])
        chain = ChatPromptTemplate.from_template("{x}") | model | JsonOutputParser()

        def validate(payload: dict) -> int:
            raise ValueError("always invalid")

        with pytest.raises(StructuredOutputError):
            invoke_json_with_retry(chain, {"x": "y"}, validate=validate)

        assert len(calls) == 3

    def test_non_dict_result_is_rejected(self, scripted_model) -> None:
        """模型返回 JSON 数组等非对象结构时同样按失败处理。"""
        model, calls = scripted_model(["[1, 2, 3]"])
        chain = ChatPromptTemplate.from_template("{x}") | model | JsonOutputParser()

        with pytest.raises(StructuredOutputError):
            invoke_json_with_retry(chain, {"x": "y"})

        assert len(calls) == 3
