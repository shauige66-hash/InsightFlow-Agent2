"""离线演示：一次 Agent 调用怎么经过 Harness 的预算与追踪控制。

不访问网络，也不构造真实模型客户端：调用链里用一段固定返回的脚本来代替
LLM，因此输出完全确定。可以看到三件事：

1. 一次逻辑调用内部的解析重试，每一次真实请求都占用模型请求额度；
2. 额度用尽后，下一次调用在**发出请求之前**被拒绝；
3. 每次成功调用在 LLM Ledger 里留下成对的 `started` / `completed`，
   并带上与 usage 对得上的 `operation_id`。

运行：

    python -m scripts.manual.harness_demo
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from core.runtime import RunRuntime, use_runtime
from harness import bind_state, get_harness
from utils import llm_ledger
from utils.agent_loop import (
    BUDGET_FIELD,
    BudgetExceeded,
    after_model_request,
    before_model_request,
    usage_of,
)

RUN_ID = "harness-demo"
DEFAULT_TRACE_DIR = Path("outputs") / "harness_demo"


class _ScriptedChain:
    """替身 LLM：前 `bad` 次返回无法解析的结构，之后返回合法结构。"""

    def __init__(self, bad: int) -> None:
        self.bad = bad
        self.requests: list[int] = []

    def invoke(self, _inputs: Any) -> Any:
        self.requests.append(len(self.requests) + 1)
        if len(self.requests) <= self.bad:
            return []
        return {"ok": True}


def _validate(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("ok") is not True:
        raise ValueError("missing ok")
    return payload


def run_demo(
    *,
    max_llm_calls: int = 2,
    trace_dir: Path | str = DEFAULT_TRACE_DIR,
    clean: bool = True,
) -> dict[str, Any]:
    """跑一次最小的受治理 Agent 调用序列，返回可断言的观测结果。"""
    trace_root = Path(trace_dir)
    if clean and trace_root.exists():
        shutil.rmtree(trace_root)

    runtime = RunRuntime(
        run_id=RUN_ID,
        # 演示直接用脚本文本替代模型，因此不经过 ModelFactory。
        model_factory=None,  # type: ignore[arg-type]
        resolver=None,  # type: ignore[arg-type]
        output_root=trace_root,
    )
    harness = get_harness()

    state: dict[str, Any] = {
        "run_id": RUN_ID,
        "completed_operations": {},
        "approvals": [],
        "pending_llm_calls": [],
        "llm_retry_granted": None,
    }
    state.update(harness.initialize_budget(state))
    state[BUDGET_FIELD]["limits"]["max_llm_calls"] = max_llm_calls

    from utils.llm_retry import invoke_json_with_retry

    events: list[dict[str, Any]] = []

    with use_runtime(runtime), bind_state(state):
        chain = _ScriptedChain(bad=1)
        payload, _operations, extra = harness.model_call(
            "executor",
            "demo#0",
            lambda: invoke_json_with_retry(
                chain,
                {},
                validate=_validate,
                max_attempts=3,
                # 与九个 Agent 用的是同一对钩子：Harness 在安装时把它们
                # 指向预算判定与请求计数。
                before_request=before_model_request,
                on_request=after_model_request,
            ),
        )
        # 节点会把返回的增量并入自己的 state 更新，演示同样要并，
        # 否则时钟结算结果只停留在返回值里。
        state.update(extra)
        events.append(
            {
                "step": "retrying-call",
                "requests": len(chain.requests),
                "llm_calls_used": usage_of(state)["llm_calls"],
                "result": payload,
            }
        )

        try:
            harness.model_call("executor", "demo#1", lambda: {"ok": True})
        except BudgetExceeded as exceeded:
            events.append(
                {
                    "step": "refused",
                    "scope": exceeded.scope,
                    "used": exceeded.used,
                    "limit": exceeded.limit,
                    # 被拒绝的调用不应留下账本记录
                    "llm_calls_used": usage_of(state)["llm_calls"],
                }
            )
        else:
            raise AssertionError("额度用尽后仍然发出了请求")

    ledger_entries = llm_ledger.read_entries(RUN_ID, root=trace_root)
    result = {
        "events": events,
        "usage": usage_of(state),
        "remaining": harness.policy.remaining(state),
        "ledger": {
            "started": sum(1 for item in ledger_entries if item["phase"] == "started"),
            "completed": sum(
                1 for item in ledger_entries if item["phase"] == "completed"
            ),
            "operation_ids": sorted({item["operation_id"] for item in ledger_entries}),
        },
        "trace_dir": str(trace_root),
        "ledger_path": str(llm_ledger.ledger_path(RUN_ID, root=trace_root)),
    }
    return result


def main() -> None:
    result = run_demo()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
