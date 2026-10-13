"""共享测试夹具。

设计要点：
- 单元测试默认不触碰网络。`network_guard` 在 socket 层拦截真实上游域名，
  把"忘了替换 LLM"变成明确失败，而不是一次静默的付费调用。
- `SocketModel` 是脚本化的假 LLM：按顺序吐出预设内容，因此可以精确驱动
  "解析失败 → 重试 → 成功"这类路径，也能断言实际发生了多少次模型调用。
- 所有测试数据来自仓库内固定的 `data/` 与 `knowledge/`，不下载、不生成随机数据。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from tests.network_guard import NetworkAccessBlocked, install_network_guard

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"

# 仓库内固定的小规模数据集，避免测试依赖外部下载。
CUSTOMERS_CSV = DATA_DIR / "customers.csv"
ORDERS_CSV = DATA_DIR / "orders.csv"
FINANCIALS_CSV = DATA_DIR / "financials.csv"
SALES_SQLITE = DATA_DIR / "sales.sqlite"
BUSINESS_RULES_MD = KNOWLEDGE_DIR / "business_rules.md"


@pytest.fixture(scope="session", autouse=True)
def block_real_network() -> None:
    install_network_guard()


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def data_dir() -> Path:
    return DATA_DIR


class SocketModel(BaseChatModel):
    """按脚本返回预设内容的假 LLM。

    responses 中的每一项按调用顺序出队；用尽后重复最后一项（便于测循环上限）。

    调用记录放在模块级注册表里，而不是实例字段：LangChain 在组合链时会复制
    Runnable，实例上的 list 会连同旧值一起被复制，测试将看不到新记录。
    用只读的实例 id 作键即可绕开这一点。
    """

    responses: list[str] = []
    recorder_key: str = ""
    simulate_parse_failure: bool = False

    @property
    def _llm_type(self) -> str:
        return "socket-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        recorded = CALL_RECORDS.setdefault(self.recorder_key, [])
        recorded.append(list(messages))

        if not self.responses:
            raise AssertionError("SocketModel 没有可用响应，测试配置不完整")

        index = min(len(recorded) - 1, len(self.responses) - 1)
        content = self.responses[index]

        if self.simulate_parse_failure:
            # 非 JSON 文本，触发 JsonOutputParser 抛错，用于验证重试路径
            content = "抱歉，我无法按要求输出。"

        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content))]
        )


# 假模型的调用记录。键为 SocketModel.recorder_key，值为每次收到的 messages。
CALL_RECORDS: dict[str, list[list[BaseMessage]]] = {}


def json_response(payload: dict[str, Any]) -> str:
    """把 dict 序列化成模型会返回的 JSON 文本。"""
    return json.dumps(payload, ensure_ascii=False)


@pytest.fixture
def scripted_model():
    """构造脚本化假 LLM 的工厂。

    返回 (model, calls)；calls 是该模型实例的调用记录列表，随调用实时增长，
    因此可以在断言里直接读长度与内容。
    """

    def _build(
        responses: list[str] | list[dict[str, Any]],
        *,
        simulate_parse_failure: bool = False,
    ) -> tuple[SocketModel, list[list[BaseMessage]]]:
        normalized = [
            json_response(item) if isinstance(item, dict) else item
            for item in responses
        ]
        key = uuid4().hex
        calls: list[list[BaseMessage]] = []
        CALL_RECORDS[key] = calls

        model = SocketModel(
            responses=normalized,
            recorder_key=key,
            simulate_parse_failure=simulate_parse_failure,
        )
        return model, calls

    yield _build

    CALL_RECORDS.clear()


@pytest.fixture
def plan_payload():
    """一份符合 Planner 输出契约的分析计划。"""

    def _build(task_count: int = 2, table: str = "customers") -> dict[str, Any]:
        return {
            "tasks": [
                {
                    "task_id": f"task_{index}",
                    "description": f"分析任务 {index}",
                    "target_tables": [table],
                    "expected_output": "聚合指标",
                }
                for index in range(1, task_count + 1)
            ]
        }

    return _build


__all__ = [
    "BUSINESS_RULES_MD",
    "CUSTOMERS_CSV",
    "DATA_DIR",
    "FINANCIALS_CSV",
    "KNOWLEDGE_DIR",
    "NetworkAccessBlocked",
    "ORDERS_CSV",
    "PROJECT_ROOT",
    "SALES_SQLITE",
    "SocketModel",
    "json_response",
]
