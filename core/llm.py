"""LLM 工厂与调用追踪。

一次工作流内的所有 Agent 共享同一份追踪上下文（run_id + client_app），
并各自持有一个带自己 agent_name 的客户端实例。客户端按 (run_id, agent_name)
缓存，因此同一个 Agent 在一次运行里复用同一个 httpx 连接池，
而并发运行之间不会互相串值——缓存键包含 run_id，追踪信息不是全局变量。

追踪头直接写在 client 的 default_headers 上：LangChain 的 `metadata`
只进入本地回调，不会成为 HTTP 头，因此无法被 ModelGate 看见。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_openai import ChatOpenAI

from core import config as config_module
from core.config import Settings, get_settings


@dataclass(frozen=True)
class CallContext:
    """一次运行的追踪上下文。不可变，便于安全地传给各 Agent。"""

    run_id: str
    client_app: str

    @classmethod
    def from_settings(cls, run_id: str, settings: Settings) -> "CallContext":
        return cls(run_id=run_id, client_app=settings.client_app)


@dataclass
class ModelFactory:
    """按 Agent 身份提供 LLM 实例。

    `resolver` 是测试注入点：传入一个返回假模型的函数即可完全不构造真实客户端。
    """

    settings: Settings
    context: CallContext
    resolver: object | None = None
    _cache: dict[str, object] = field(default_factory=dict)

    def for_agent(self, agent_name: str):
        if agent_name in self._cache:
            return self._cache[agent_name]

        model = self._resolve(agent_name)
        self._cache[agent_name] = model
        return model

    def _resolve(self, agent_name: str):
        if self.resolver is not None:
            return self.resolver(agent_name)  # type: ignore[operator]

        return create_llm(self.settings, self.context, agent_name)


def create_llm(
    settings: Settings,
    context: CallContext,
    agent_name: str,
) -> ChatOpenAI:
    """构造指向 ModelGate 的客户端，并带上该 Agent 的追踪头。"""
    config_module.validate(settings)

    return ChatOpenAI(
        model=settings.model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        temperature=0,
        timeout=settings.timeout_seconds,
        # 重试交给 ModelGate 与业务层的 JSON 修复循环，客户端不再叠加一层，
        # 否则一次逻辑调用会被放大成多次真实请求，统计口径会失真。
        max_retries=0,
        default_headers={
            "X-Client-App": context.client_app,
            "X-Agent-Name": agent_name,
            "X-Run-Id": context.run_id,
        },
    )


def build_model_factory(run_id: str, settings: Settings | None = None) -> ModelFactory:
    settings = settings or get_settings()
    return ModelFactory(
        settings=settings,
        context=CallContext.from_settings(run_id, settings),
    )


def create_default_llm() -> ChatOpenAI:
    """Agent 未收到注入客户端时的回退构造。

    不带追踪头，因此只应出现在单次脚本调用等没有 run 上下文的入口；
    正常的工作流路径一律通过 ModelFactory 注入，以保留 run_id 与 agent_name。
    """
    settings = get_settings()
    config_module.validate(settings)

    return ChatOpenAI(
        model=settings.model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        temperature=0,
        timeout=settings.timeout_seconds,
        max_retries=0,
    )
