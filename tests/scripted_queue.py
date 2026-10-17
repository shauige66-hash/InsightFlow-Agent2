"""按队列返回脚本回复的模型工厂。

`ScriptedFactory` 每个 Agent 只有一份固定回复，够用于单向流程；运行期动态
决策需要"第一次说 A、第二次说 B"，因此这里按 Agent 维护一个队列，队列取尽
后重复最后一个值（避免脚本写漏时直接崩掉，且行为可预测）。
"""

from __future__ import annotations

import json
import threading
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from core.config import Settings
from core.llm import CallContext, ModelFactory


class ScriptedQueueFactory:
    def __init__(
        self, run_id: str, queues: dict[str, list[Any]]
    ) -> None:
        self.run_id = run_id
        self.queues = {name: list(items) for name, items in queues.items()}
        self.calls: list[str] = []
        self.prompts: dict[str, list[str]] = {}
        self._lock = threading.Lock()

    def next_reply(self, agent_name: str) -> Any:
        queue = self.queues.get(agent_name)
        if not queue:
            return {"tasks": []}
        with self._lock:
            if len(queue) > 1:
                return queue.pop(0)
            return queue[0]

    def build(self) -> ModelFactory:
        owner = self

        class _Queued(BaseChatModel):
            agent_name: str = ""

            @property
            def _llm_type(self) -> str:
                return f"h5-{self.agent_name}"

            def _generate(
                self,
                messages: list[BaseMessage],
                stop: list[str] | None = None,
                run_manager: CallbackManagerForLLMRun | None = None,
                **kwargs: Any,
            ) -> ChatResult:
                owner.calls.append(self.agent_name)
                owner.prompts.setdefault(self.agent_name, []).append(
                    "\n".join(str(message.content) for message in messages)
                )
                reply = owner.next_reply(self.agent_name)
                content = (
                    reply
                    if isinstance(reply, str)
                    else json.dumps(reply, ensure_ascii=False)
                )
                return ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content=content))]
                )

        def resolver(agent_name: str) -> _Queued:
            return _Queued(agent_name=agent_name)

        return ModelFactory(
            settings=Settings(
                api_key="sk-fake",
                base_url="http://127.0.0.1:8000/v1",
                model="deepseek-chat",
            ),
            context=CallContext(run_id=self.run_id, client_app="insightflow-agent2"),
            resolver=resolver,
        )


def task(
    task_id: str,
    *,
    description: str = "统计客户数量",
    tables: list[str] | None = None,
    expected: str = "客户数",
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "description": description,
        "target_tables": tables or ["customers.customers"],
        "expected_output": expected,
    }


def decision(action: str, reason: str = "") -> dict[str, Any]:
    return {"action": action, "reason": reason or action}


__all__ = ["ScriptedQueueFactory", "decision", "task"]
