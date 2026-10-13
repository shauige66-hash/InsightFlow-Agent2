"""统一的 Agent 模型注入点。

各 Agent 需要"本 Agent 的 LLM 实例"，而实例由调用方按 run 构造并注入。
节点通过这里取，而不是自己 `create_llm()`，这样：

- 单元测试可以注入假模型，不构造任何真实客户端；
- 一次运行内同一 Agent 复用同一实例，携带正确的 run_id 与 agent_name。

`model_factory` 存在 state 里是为了避免全局可变追踪变量。它是运行期资源句柄、
不可序列化，因此 P1 引入 Checkpoint 时必须移出 state；届时改为由
langgraph 的 config/configurable 传入，语义不变。
"""

from __future__ import annotations

from typing import Any

from core.llm import ModelFactory, build_model_factory
from state import InsightFlowState


def model_for(state: InsightFlowState, agent_name: str) -> Any:
    """取该 Agent 本次运行要用的模型实例。"""
    factory = state.get("model_factory")

    if factory is None:
        # 未注入时（例如直接调用单个节点函数）退回无追踪的默认构造。
        factory = build_model_factory(run_id=str(state.get("run_id") or "no-run"))

    return factory.for_agent(agent_name)


def factory_from(model_factory: ModelFactory | None, run_id: str) -> ModelFactory:
    return model_factory if model_factory is not None else build_model_factory(run_id)
