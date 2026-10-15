"""P2-A 补验收：错误分类是否贯穿工作流并影响重试次数。

链路：
    sandbox/executor.py → tools/*.py → tools/tool_registry.py
    → nodes.py → conditions.route_after_execution

P1 的 `route_after_execution` 只看 `execution_error` 是否存在，因此
"沙箱没起来"也会去让模型修代码——修不好，还浪费模型调用。
本文件用 Mock LLM 的调用计数验证修复后的行为。

判据不用错误字符串：类型从沙箱一路传到路由，断言的是**结构化字段**。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from core.llm import build_model_factory
from core.runtime import RunRuntime, use_runtime
from core.config import Settings
from core.llm import CallContext, ModelFactory
from ingestion.models import TableResource
from ingestion.resources import ResourceResolver
from services.analysis_service import run_analysis
from sandbox.executor import DockerSandboxExecutor

from tests.conftest import CALL_RECORDS

pytestmark = pytest.mark.sandbox

PLAN = {
    "tasks": [
        {
            "task_id": "task_1",
            "description": "统计客户数量",
            "target_tables": ["customers"],
            "expected_output": "客户数",
        }
    ]
}

REPLIES = {
    "planner": PLAN,
    "executor": {"code": "result = len(df)"},
    "repair": {"code": "result = len(df)"},
    "refine": {"code": "result = len(df)"},
    "critic": {"validation_passed": True, "feedback": "ok"},
    "visualizer": {"charts": []},
    "reporter": "# Analysis Report\n\n## Overview\n共 3 位客户。\n",
}


@pytest.fixture(scope="module", autouse=True)
def require_docker() -> None:
    executor = DockerSandboxExecutor()
    if not executor.available():
        pytest.skip("Docker daemon 不可用")
    if not executor.image_present():
        pytest.skip("沙箱镜像未构建：先运行 sandbox/build_image.py")


def scripted_factory(run_id: str, reply_overrides: dict | None = None) -> ModelFactory:
    """脚本化假模型：记录每个 Agent 的调用次数。"""
    import json as _json

    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessage, BaseMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    replies = {**REPLIES, **(reply_overrides or {})}
    CALL_RECORDS[run_id] = []

    class _Scripted(BaseChatModel):
        agent_name: str = ""
        record_key: str = ""

        @property
        def _llm_type(self) -> str:
            return f"errtest-{self.agent_name}"

        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs,
        ) -> ChatResult:
            CALL_RECORDS[self.record_key].append(self.agent_name)
            reply = replies[self.agent_name]
            content = (
                reply
                if isinstance(reply, str)
                else _json.dumps(reply, ensure_ascii=False)
            )
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content=content))]
            )

    def resolver(agent_name: str) -> _Scripted:
        return _Scripted(agent_name=agent_name, record_key=run_id)

    return ModelFactory(
        settings=Settings(
            api_key="sk-fake", base_url="http://127.0.0.1:8000/v1", model="deepseek-chat"
        ),
        context=CallContext(run_id=run_id, client_app="insightflow-agent2"),
        resolver=resolver,
    )


def agent_calls(run_id: str) -> list[str]:
    return list(CALL_RECORDS.get(run_id, []))


def table(tmp_path: Path) -> Path:
    path = tmp_path / "customers.csv"
    path.write_text(
        "customer_id,region\nC001,East\nC002,West\nC003,East\n", encoding="utf-8"
    )
    return path


def run_workflow(run_id: str, source: Path, tmp_path: Path, **kwargs):
    return run_analysis(
        run_id=run_id,
        data_sources=[str(source)],
        model_factory=scripted_factory(run_id, kwargs.pop("reply_overrides", None)),
        resolver=ResourceResolver(),
        output_root=tmp_path,
        **kwargs,
    )


class TestEnvironmentErrorsDoNotTriggerRepair:
    """环境/资源类失败不应进入 LLM 修复循环。"""

    def test_sandbox_unavailable_skips_repair(self, tmp_path: Path, monkeypatch) -> None:
        """Docker 不可用：修代码没意义，repair 调用次数应为 0。"""
        import tools.python_tool as python_tool

        broken = DockerSandboxExecutor(docker="not-a-real-docker-binary")
        monkeypatch.setattr(python_tool, "get_executor", lambda **_: broken)

        source = table(tmp_path)
        result = run_workflow("err-unavailable", source, tmp_path)

        calls = agent_calls("err-unavailable")
        assert calls.count("repair") == 0, f"环境错误不该触发修复: {calls}"
        assert result["failed_tasks"], "应记录为失败任务"
        assert result["failed_tasks"][0]["error_kind"] == "unavailable"
        # 有失败任务时工作流把整体判为 partial（而不是 completed），
        # 这是既有语义：result_node 依 failed_tasks 与成功任务数共同决定。
        assert result["analysis_status"] == "partial"

    def test_image_missing_skips_repair(self, tmp_path: Path, monkeypatch) -> None:
        """镜像不存在：同样属于环境问题。"""
        import tools.python_tool as python_tool

        missing = DockerSandboxExecutor(image="insightflow-sandbox:no-such-tag")
        monkeypatch.setattr(python_tool, "get_executor", lambda **_: missing)

        source = table(tmp_path)
        result = run_workflow("err-noimage", source, tmp_path)

        calls = agent_calls("err-noimage")
        assert calls.count("repair") == 0, f"镜像缺失不该触发修复: {calls}"
        assert result["failed_tasks"][0]["error_kind"] == "unavailable"

    def test_timeout_skips_repair(self, tmp_path: Path, monkeypatch) -> None:
        """执行超时：重发同样的代码只会再超时一次。"""
        from sandbox import executor as executor_module

        original = executor_module.SandboxLimits

        def tiny_timeout(**kwargs):
            kwargs.setdefault("timeout_seconds", 5)
            return original(**kwargs)

        monkeypatch.setattr(executor_module, "SandboxLimits", tiny_timeout)
        import tools.python_tool as python_tool

        monkeypatch.setattr(python_tool, "SandboxLimits", tiny_timeout, raising=False)

        source = table(tmp_path)
        result = run_workflow(
            "err-timeout",
            source,
            tmp_path,
            reply_overrides={"executor": {"code": "while True:\n    pass"}},
        )

        calls = agent_calls("err-timeout")
        assert calls.count("repair") == 0, f"超时不该触发修复: {calls}"
        assert result["failed_tasks"][0]["error_kind"] == "timeout"


class TestExecutionErrorsStillRepair:
    """模型代码本身的错误仍应进入有界修复循环。"""

    def test_code_error_triggers_bounded_repair(self, tmp_path: Path) -> None:
        """执行异常：进入有界修复循环，每轮都真正请求模型，且被 MAX_RETRIES 限定。

        **P2-B1 更新说明**：本用例原先断言 `repair` 的模型调用次数为 1。
        那锁定的是当时的缺陷——`llm_operation` 的 subject 不含逻辑尝试序号，
        第 2、3 次修复被 `already_done` 误判为同一次操作而直接复用第 1 次的代码。
        P2-B1 修复后每轮修复都是独立的逻辑尝试，因此调用次数应为 MAX_RETRIES。
        详细的逐轮身份断言见 tests/test_llm_attempt_identity.py。
        """
        from conditions import MAX_RETRIES

        source = table(tmp_path)
        result = run_workflow(
            "err-code",
            source,
            tmp_path,
            reply_overrides={
                "executor": {"code": "raise ValueError('always fails')"},
                "repair": {"code": "raise ValueError('still fails')"},
            },
        )

        calls = agent_calls("err-code")
        attempts = [item["retry_count"] for item in result["execution_results"]]

        # 代码错误必须触发修复（与"环境错误零修复"形成对照），且每轮各问一次模型
        assert calls.count("repair") == MAX_RETRIES, f"修复 Agent 调用次数: {calls}"
        # 修复循环跑到上限：retry_count 依次为 0,1,2,3
        assert attempts == [0, 1, 2, 3], f"尝试序列: {attempts}"
        assert max(attempts) == MAX_RETRIES
        assert result["failed_tasks"]
        assert result["failed_tasks"][0]["error_kind"] == "execution_error"

    def test_error_kind_reaches_state_and_log(self, tmp_path: Path) -> None:
        """分类要能落到 state 与 JSONL 日志，便于事后归因。"""
        source = table(tmp_path)
        run_workflow(
            "err-logged",
            source,
            tmp_path,
            reply_overrides={
                "executor": {"code": "raise ValueError('boom')"},
                "repair": {"code": "raise ValueError('boom')"},
            },
        )

        log_path = tmp_path / "logs" / "err-logged.jsonl"
        events = [
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        failed = [e for e in events if e["event"] == "task_failed"]
        assert failed, "应有 task_failed 事件"
        assert failed[0]["details"].get("error_kind") == "execution_error"


class TestErrorKindContractIsStructured:
    """分类必须是结构化字段，而不是靠解析错误文案。"""

    def test_tool_result_carries_error_kind(self, tmp_path: Path) -> None:
        from tools.tool_registry import invoke_tool

        tables = {
            "customers": TableResource(
                name="customers",
                source_type="csv",
                source_path=str(tmp_path / "c.csv"),
                dataframe=pd.DataFrame({"a": [1, 2, 3]}),
            )
        }

        runtime = RunRuntime(
            run_id="contract",
            model_factory=build_model_factory("contract"),
            resolver=ResourceResolver(),
            output_root=tmp_path,
        )

        with use_runtime(runtime):
            ok = invoke_tool("python", tables, {"table_names": ["customers"], "code": "result = 1"})
            bad = invoke_tool(
                "python", tables, {"table_names": ["customers"], "code": "raise ValueError('x')"}
            )

        assert set(ok) == {"success", "result", "error", "error_kind"}
        assert ok["success"] is True and ok["error_kind"] is None
        assert bad["success"] is False and bad["error_kind"] == "execution_error"

    def test_unknown_tool_is_classified(self, tmp_path: Path) -> None:
        from tools.tool_registry import invoke_tool

        outcome = invoke_tool("ruby", {}, {})

        assert outcome["success"] is False
        assert outcome["error_kind"] == "unsupported_tool"

    def test_plot_path_also_carries_error_kind(self, tmp_path: Path, monkeypatch) -> None:
        """绘图路径同样带分类。"""
        import tools.plot_tool as plot_tool

        broken = DockerSandboxExecutor(docker="not-a-real-docker-binary")
        monkeypatch.setattr(plot_tool, "get_executor", lambda **_: broken, raising=False)
        import tools.python_tool as python_tool

        monkeypatch.setattr(python_tool, "get_executor", lambda **_: broken)

        tables = {
            "customers": TableResource(
                name="customers",
                source_type="csv",
                source_path=str(tmp_path / "c.csv"),
                dataframe=pd.DataFrame({"a": [1, 2, 3]}),
            )
        }

        runtime = RunRuntime(
            run_id="plot-contract",
            model_factory=build_model_factory("plot-contract"),
            resolver=ResourceResolver(),
            output_root=tmp_path,
        )

        with use_runtime(runtime):
            from tools.tool_registry import invoke_tool

            outcome = invoke_tool(
                "plot",
                tables,
                {
                    "table_names": ["customers"],
                    "chart_id": "chart_1",
                    "code": "result = output_path",
                    "analysis_result": [{"a": 1}],
                },
            )

        assert outcome["success"] is False
        assert outcome["error_kind"] == "unavailable"
