"""LLM 工厂与调用追踪。

要验证的核心性质：追踪头真的会出现在 HTTP 请求上（ModelGate 才能看见），
且并发运行之间不会串值——这是"不用全局可变追踪变量"的实际检验。
"""

from __future__ import annotations

import pytest

from core.config import Settings
from core.llm import CallContext, ModelFactory, build_model_factory, create_llm

FAKE_SETTINGS = Settings(
    api_key="sk-fake-modelgate-client-key",
    base_url="http://127.0.0.1:8000/v1",
    model="deepseek-chat",
    client_app="insightflow-agent2",
    timeout_seconds=30.0,
)


class TestCreateLlm:
    def test_tracing_headers_are_set(self) -> None:
        context = CallContext(run_id="run_test_1", client_app="insightflow-agent2")

        llm = create_llm(FAKE_SETTINGS, context, "planner")

        headers = llm.default_headers
        assert headers["X-Run-Id"] == "run_test_1"
        assert headers["X-Agent-Name"] == "planner"
        assert headers["X-Client-App"] == "insightflow-agent2"

    def test_model_and_base_url_come_from_settings(self) -> None:
        context = CallContext(run_id="r", client_app="app")

        llm = create_llm(FAKE_SETTINGS, context, "executor")

        assert llm.model_name == "deepseek-chat"
        assert "127.0.0.1:8000" in str(llm.openai_api_base)

    def test_client_side_retry_is_disabled(self) -> None:
        """客户端重试必须关闭：否则一次逻辑调用会被放大成多次真实请求，
        与 ModelGate 侧统计和业务层 JSON 修复循环叠加后口径失真。"""
        context = CallContext(run_id="r", client_app="app")

        llm = create_llm(FAKE_SETTINGS, context, "critic")

        assert llm.max_retries == 0

    def test_missing_api_key_fails_fast(self) -> None:
        settings = Settings(api_key="", base_url="http://x/v1", model="m")
        context = CallContext(run_id="r", client_app="app")

        with pytest.raises(ValueError, match="DEEPSEEK_API_KEY is missing"):
            create_llm(settings, context, "planner")

    def test_missing_model_fails_fast(self) -> None:
        settings = Settings(api_key="sk-x", base_url="http://x/v1", model="")
        context = CallContext(run_id="r", client_app="app")

        with pytest.raises(ValueError, match="DEEPSEEK_MODEL is missing"):
            create_llm(settings, context, "planner")


class TestModelFactory:
    def test_returns_same_instance_per_agent(self) -> None:
        """同一 Agent 复用同一个客户端，避免每次调用都新建连接池。"""
        factory = ModelFactory(
            settings=FAKE_SETTINGS,
            context=CallContext(run_id="r", client_app="app"),
        )

        assert factory.for_agent("planner") is factory.for_agent("planner")

    def test_different_agents_get_different_instances(self) -> None:
        factory = ModelFactory(
            settings=FAKE_SETTINGS,
            context=CallContext(run_id="r", client_app="app"),
        )

        assert factory.for_agent("planner") is not factory.for_agent("executor")

    def test_resolver_bypasses_real_client_construction(self) -> None:
        """注入点必须真的绕过 ChatOpenAI，否则测试仍需要凭证。"""
        built: list[str] = []

        def resolver(agent_name: str) -> str:
            built.append(agent_name)
            return f"fake-{agent_name}"

        factory = ModelFactory(
            settings=Settings(api_key="", base_url="", model=""),
            context=CallContext(run_id="r", client_app="app"),
            resolver=resolver,
        )

        assert factory.for_agent("planner") == "fake-planner"
        assert built == ["planner"]

    def test_resolver_result_is_cached(self) -> None:
        calls: list[str] = []

        def resolver(agent_name: str) -> str:
            calls.append(agent_name)
            return "fake"

        factory = ModelFactory(
            settings=FAKE_SETTINGS,
            context=CallContext(run_id="r", client_app="app"),
            resolver=resolver,
        )

        factory.for_agent("critic")
        factory.for_agent("critic")

        assert calls == ["critic"]


class TestRunIsolation:
    def test_two_runs_do_not_share_tracing(self) -> None:
        """两个 run 的工厂互不影响——追踪信息不是模块级全局变量。"""
        first = ModelFactory(
            settings=FAKE_SETTINGS,
            context=CallContext(run_id="run_a", client_app="app"),
        )
        second = ModelFactory(
            settings=FAKE_SETTINGS,
            context=CallContext(run_id="run_b", client_app="app"),
        )

        llm_a = first.for_agent("planner")
        llm_b = second.for_agent("planner")

        assert llm_a.default_headers["X-Run-Id"] == "run_a"
        assert llm_b.default_headers["X-Run-Id"] == "run_b"
        assert llm_a is not llm_b


def test_build_model_factory_uses_settings_client_app() -> None:
    factory = build_model_factory("run_x", FAKE_SETTINGS)

    assert factory.context.run_id == "run_x"
    assert factory.context.client_app == "insightflow-agent2"
