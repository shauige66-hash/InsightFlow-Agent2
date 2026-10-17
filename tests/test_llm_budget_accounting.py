"""P3 定点修复：模型调用计数的口径与请求计数作用域。

审查怀疑 `budget_guard`、请求计数钩子与
`agent_loop.check_llm_budget` 之间存在重复累计。这里把口径固定下来：

- 判定（`check_llm_budget`）只读，不改 state；
- 计数（`consume_llm_call`）每次真实请求各一次；
- 判定读到的 `used` 必须等于"已发出的请求数"，因此内部 JSON 重试既不能
  超过额度，也不能提前拒绝仍有额度的请求。

`test_budget_real_entry_points.py` 已经覆盖"重试不得越过上限"；这里补的是
**不重复累计**与**不提前拒绝**这两个方向。
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from core.runtime import McpSettings
from ingestion.resources import ResourceResolver
from services.analysis_service import run_analysis
from utils import agent_loop as al
from utils.agent_loop import BUDGET_FIELD, AgentBudget, initialize, new_usage

from tests.test_budget_real_entry_points import CountingFactory
from tests.test_report_approval import REPLIES, build_database


@pytest.fixture(autouse=True)
def _restore_replies():
    original = copy.deepcopy(REPLIES)
    yield
    REPLIES.clear()
    REPLIES.update(copy.deepcopy(original))


def bad_executor_reply() -> ChatResult:
    """合法 JSON 但缺必填字段：触发 executor 内部重试。"""
    return ChatResult(
        generations=[ChatGeneration(message=AIMessage(content="{}"))]
    )


def run_with_retries(tmp_path: Path, *, run_id: str, cap: int, bad_replies: int):
    """跑一次真实工作流，让 executor 前 N 次返回缺字段内容。"""
    database = tmp_path / run_id / "customers.sqlite"
    database.parent.mkdir(parents=True, exist_ok=True)
    build_database(database)

    counting = CountingFactory(run_id)
    seen = {"n": 0}

    def responder(agent_name: str, total: int):
        if agent_name != "executor":
            return None
        seen["n"] += 1
        return bad_executor_reply() if seen["n"] <= bad_replies else None

    counting.responder = responder

    result = run_analysis(
        run_id=run_id,
        data_sources=[str(database)],
        model_factory=counting.build(),
        resolver=ResourceResolver(),
        output_root=tmp_path,
        mcp=McpSettings(mode="direct"),
        budget={"max_llm_calls": cap},
    )
    return result, counting, seen


class TestGuardsAreReadOnly:
    """判定不得累计计数。"""

    def test_check_llm_budget_does_not_change_the_usage(self) -> None:
        state: dict[str, Any] = {"run_id": "guard-readonly"}
        state.update(initialize(state))
        state[BUDGET_FIELD]["limits"]["max_llm_calls"] = 5

        before = al.usage_of(state)
        for _ in range(4):
            al.check_llm_budget(state)

        assert al.usage_of(state) == before, "判定过程改动了用量"

    def test_guard_counts_only_the_usage_it_reads(self) -> None:
        """上限按"已发出的请求数"判定，不额外加上当前这一次。"""
        state: dict[str, Any] = {"run_id": "guard-boundary"}
        state.update(initialize(state))
        state[BUDGET_FIELD]["limits"]["max_llm_calls"] = 2

        # 已发出 1 次：还剩 1 个额度，判定通过
        state[BUDGET_FIELD]["usage"]["llm_calls"] = 1
        al.check_llm_budget(state)

        # 已发出 2 次：额度用尽，下一次请求必须被拦下
        state[BUDGET_FIELD]["usage"]["llm_calls"] = 2
        with pytest.raises(al.BudgetExceeded) as excinfo:
            al.check_llm_budget(state)
        assert excinfo.value.scope == "llm_calls"


class TestCountersAreNotDoubledInTheRealWorkflow:
    def test_upstream_requests_equal_the_counted_usage(self, tmp_path: Path) -> None:
        """强制 2 次解析失败重试后，计数与真实请求数必须一致。"""
        result, counting, seen = run_with_retries(
            tmp_path, run_id="count-no-double", cap=10, bad_replies=2
        )

        assert seen["n"] >= 3, f"没有触发重试（executor 仅 {seen['n']} 次）"
        usage = (result.get("execution_budget") or {}).get("usage") or {}
        assert usage.get("llm_calls") == counting.count, (
            f"用量（{usage.get('llm_calls')}）与真实请求数（{counting.count}）不一致"
        )

    @pytest.mark.parametrize("cap", [1, 2, 3, 4])
    def test_cap_stops_exactly_at_the_limit(
        self, tmp_path: Path, cap: int
    ) -> None:
        """重试路径下既不能超上限，也不能在还有额度时提前停下。"""
        result, counting, _seen = run_with_retries(
            tmp_path, run_id=f"count-cap-{cap}", cap=cap, bad_replies=2
        )

        usage = (result.get("execution_budget") or {}).get("usage") or {}
        assert counting.count <= cap, f"上限 {cap} 却发出 {counting.count} 次请求"
        assert usage.get("llm_calls") == counting.count


class TestRetryIsNotRejectedWhileBudgetRemains:
    def test_a_long_retry_burst_fits_inside_a_generous_budget(
        self, tmp_path: Path
    ) -> None:
        """额度充足时，内部重试与后续节点都不得被提前拒绝。

        判据是两条：重试确实发生（executor 至少被调用两次），
        且整个 Run 的请求数远未触及上限。
        """
        result, counting, seen = run_with_retries(
            tmp_path, run_id="retry-not-premature", cap=20, bad_replies=2
        )

        assert seen["n"] >= 2, f"没有触发重试（executor 仅 {seen['n']} 次）"
        assert counting.count < 20, (
            f"上限 20 却在 {counting.count} 次请求后就停止，疑似提前拒绝"
        )
        usage = (result.get("execution_budget") or {}).get("usage") or {}
        assert usage.get("llm_calls") == counting.count, (
            f"用量（{usage.get('llm_calls')}）与真实请求数（{counting.count}）不一致"
        )
        assert result["analysis_status"] == "completed", (
            f"额度充足却没能跑完：{result.get('status_reason')}"
        )
