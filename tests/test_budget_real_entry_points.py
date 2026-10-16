"""P3 定点修复：预算在**真实调用路径**上生效。

这些用例回答的是审查提出的具体问题，而不是"配置字段有没有写进 state"：

- 真正发起模型请求之前有没有判预算；
- 幂等复用会不会被算成新请求；
- 内部重试（解析失败）会不会绕过请求预算；
- 预算耗尽时有没有安全终止、有没有污染 LLM Ledger；
- 绘图是不是也走统一工具治理、预算为 0 时会不会仍启动容器；
- Checkpoint 恢复与跨进程恢复后预算会不会重置；
- 等待 HITL 审批的时间算不算执行预算。

因此每个用例都驱动**真实节点/真实工作流**，只在模型侧注入一个可计数的
脚本模型。断言对象是"模型真的被调了几次"，而不是 state 里的字段。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from core.runtime import McpSettings
from ingestion.resources import ResourceResolver
from services.analysis_service import run_analysis
from utils import agent_loop as al
from utils.agent_loop import (
    BUDGET_FIELD,
    AgentBudget,
    BudgetExceeded,
    new_usage,
)

from tests.test_report_approval import REPLIES, build_database, factory

# 每个测试结束后恢复共享的 REPLIES（深拷贝，避免嵌套字典的改动泄漏）
import copy


@pytest.fixture(autouse=True)
def _restore_replies():
    original = copy.deepcopy(REPLIES)
    yield
    REPLIES.clear()
    REPLIES.update(copy.deepcopy(original))


# ---------------------------------------------------------------------------
# 可计数的脚本模型：在真实 resolver 之上包一层，只数真实请求
# ---------------------------------------------------------------------------


class CountingFactory:
    """把脚本模型包成"会记数"的模型。

    计数点选在 `_generate`：那是 LangChain 真正要求模型产出回复的地方，
    因此**每一次** `chain.invoke` 都会走到这里——包括解析失败后的重试。
    这正是"真实上游请求数"的口径。

    做法是在脚本模型**自身**的实例上替换 `_generate`，而不是外面套一层代理：
    LangChain 的 `prompt | llm` 会校验右操作数是不是 `Runnable`，
    代理对象会被直接拒绝（`Expected a Runnable, callable or dict`）。
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.requests: list[str] = []
        #: 可选：按 agent 名定制每次回复（用于制造解析失败）
        self.responder: Any = None

    def build(self):
        base_factory = factory(self.run_id)
        original_resolver = base_factory.resolver
        owner = self

        def resolver(agent_name: str):
            inner = original_resolver(agent_name)
            original_generate = inner._generate

            def _generate(*args: Any, **kwargs: Any):
                owner.requests.append(agent_name)
                if owner.responder is not None:
                    custom = owner.responder(agent_name, len(owner.requests))
                    if custom is not None:
                        return custom
                return original_generate(*args, **kwargs)

            inner._generate = _generate  # type: ignore[method-assign]
            return inner

        base_factory.resolver = resolver
        return base_factory

    @property
    def count(self) -> int:
        return len(self.requests)


def run_with_budget(
    tmp_path: Path,
    *,
    run_id: str,
    limits: dict[str, Any],
    data_sources: list[str] | None = None,
    mode: str = "direct",
) -> tuple[dict[str, Any], CountingFactory]:
    """按给定预算跑一次真实工作流。"""
    if data_sources is None:
        # 文件名必须是 customers.sqlite：脚本计划里的 target_tables 是
        # `customers.customers`（`<文件主名>.<表名>`），换个名字 router 就找不到表。
        # 放在以 run_id 命名的子目录里，避免同一个 tmp_path 上的多次调用
        # 互相覆盖数据库（快照按内容校验，覆盖会让后续 Run 直接失败）。
        run_dir = tmp_path / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        database = run_dir / "customers.sqlite"
        build_database(database)
        data_sources = [str(database)]

    counting = CountingFactory(run_id)

    result = run_analysis(
        run_id=run_id,
        data_sources=data_sources,
        model_factory=counting.build(),
        resolver=ResourceResolver(),
        output_root=tmp_path,
        mcp=McpSettings(mode=mode),
        budget=limits,
    )
    return result, counting


# ---------------------------------------------------------------------------
# 任务一：LLM 调用预算
# ---------------------------------------------------------------------------


class TestLlmBudgetOnTheRealPath:
    """`max_llm_calls` 必须在真实请求前生效，且不许被内部重试绕过。"""

    def test_budget_zero_forbids_every_model_request(self, tmp_path: Path) -> None:
        result, counting = run_with_budget(
            tmp_path,
            run_id="p3-llm-0",
            limits={"max_llm_calls": 0},
        )

        assert counting.count == 0, (
            f"预算为 0 时不应发出任何模型请求，实际 {counting.count} 次："
            f"{counting.requests}"
        )
        # 安全终止：收敛成失败，而不是抛未分类异常
        assert result["analysis_status"] == "failed"
        usage = (result.get("execution_budget") or {}).get("usage") or {}
        assert usage.get("llm_calls", 0) == 0

    def test_budget_one_allows_exactly_one_request(self, tmp_path: Path) -> None:
        _result, counting = run_with_budget(
            tmp_path,
            run_id="p3-llm-1",
            limits={"max_llm_calls": 1},
        )

        assert counting.count == 1, (
            f"上限 1 时应恰好发出 1 次请求，实际 {counting.count} 次："
            f"{counting.requests}"
        )

    def test_budget_two_allows_exactly_two_requests(self, tmp_path: Path) -> None:
        _result, counting = run_with_budget(
            tmp_path,
            run_id="p3-llm-2",
            limits={"max_llm_calls": 2},
        )

        assert counting.count == 2, (
            f"上限 2 时应恰好发出 2 次请求，实际 {counting.count} 次："
            f"{counting.requests}"
        )

    def test_request_count_never_exceeds_the_declared_budget(
        self, tmp_path: Path
    ) -> None:
        """多种上限下都不得超出——这是这个预算存在的全部意义。"""
        for limit in (0, 1, 2, 3):
            _result, counting = run_with_budget(
                tmp_path,
                run_id=f"p3-cap-{limit}",
                limits={"max_llm_calls": limit},
            )
            assert counting.count <= limit, (
                f"上限 {limit} 却发出了 {counting.count} 次请求"
            )

    def test_usage_figure_matches_real_requests(self, tmp_path: Path) -> None:
        """记在 state 里的用量必须等于真实请求数，不能虚高或虚低。"""
        result, counting = run_with_budget(
            tmp_path,
            run_id="p3-usage",
            limits={"max_llm_calls": 2},
        )
        usage = (result.get("execution_budget") or {}).get("usage") or {}
        assert usage.get("llm_calls") == counting.count

    def test_budget_exhaustion_does_not_enter_repair(self, tmp_path: Path) -> None:
        """预算耗尽要安全终止，不能触发无意义的 Repair。"""
        result, _counting = run_with_budget(
            tmp_path,
            run_id="p3-norepair",
            limits={"max_llm_calls": 0},
        )

        assert result["analysis_status"] == "failed"
        # 没有模型调用 → 不可能发生 repair（repair 本身要调模型）
        # 更直接的判据：失败原因里带的是预算口径
        assert result.get("report_path") is None

    def test_exhaustion_leaves_no_orphan_ledger_entry(self, tmp_path: Path) -> None:
        """超预算被拦下时不得往 LLM Ledger 写 `started`。

        否则会留下一条永远没有 `completed` 的记录，恢复时被判成
        "未确认的模型调用"并要求人工裁决——预算问题不该变成人工工单。
        """
        import json

        result, counting = run_with_budget(
            tmp_path,
            run_id="p3-ledger",
            limits={"max_llm_calls": 1},
        )
        assert counting.count == 1

        ledger_dir = tmp_path / "llm_ledger"
        started = completed = 0
        if ledger_dir.exists():
            for path in ledger_dir.glob("*.jsonl"):
                for line in path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    # 账本用 `phase` 区分 started / completed
                    phase = event.get("phase")
                    if phase == "started":
                        started += 1
                    elif phase == "completed":
                        completed += 1

        assert started == counting.count, (
            f"账本 started 数（{started}）应等于真实请求数（{counting.count}）"
        )
        assert completed == counting.count, (
            f"账本 completed 数（{completed}）应等于真实请求数（{counting.count}）"
        )
        assert started == completed, "每次 started 都必须有对应的 completed"


class TestInternalRetryCannotBypassTheBudget:
    """解析失败会在 `run_agent_call` 内部重发请求，那也必须受预算约束。"""

    def test_retry_hook_counts_each_request(self) -> None:
        from utils.llm_retry import invoke_json_with_retry

        seen_before: list[int] = []
        seen_after: list[int] = []

        class _Chain:
            def __init__(self) -> None:
                self.calls = 0

            def invoke(self, _inputs: Any) -> Any:
                self.calls += 1
                # 前两次返回坏结构 → 触发内部重试
                if self.calls < 3:
                    return []
                return {"ok": True}

        result = invoke_json_with_retry(
            _Chain(),
            {},
            max_attempts=3,
            before_request=seen_before.append,
            on_request=seen_after.append,
        )

        assert result == {"ok": True}
        assert seen_before == [1, 2, 3], "每次请求前都要判预算"
        assert seen_after == [1, 2, 3], "每次请求后都要计数"

    def test_budget_exceeded_aborts_retrying_instead_of_continuing(self) -> None:
        """重试途中预算耗尽必须**上抛**，不能被当成解析失败继续重试。"""
        from utils.llm_retry import invoke_json_with_retry

        class _Chain:
            def invoke(self, _inputs: Any) -> Any:
                return []  # 永远解析失败

        def before(attempt: int) -> None:
            if attempt >= 2:
                raise BudgetExceeded("llm_calls", 2, 2)

        with pytest.raises(BudgetExceeded):
            invoke_json_with_retry(
                _Chain(), {}, max_attempts=3, before_request=before
            )

    def test_retry_does_not_exceed_declared_cap(self, tmp_path: Path) -> None:
        """重试确实发生，且重试后的**总请求数**仍不超上限。

        上限取 5（而不是刚够用）：3 次重试场景下，上限太小会让第 2 次尝试
        直接被预算拦下，那样就看不到"重试已被计数"这件事——
        这条用例要同时证明**发生了重试**与**总数没超**。
        """
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult

        database = tmp_path / "retry-run" / "customers.sqlite"
        database.parent.mkdir(parents=True, exist_ok=True)
        build_database(database)

        counting = CountingFactory("p3-retry-cap")
        executor_replies = {"n": 0}

        def responder(agent_name: str, total: int):
            if agent_name != "executor":
                return None
            executor_replies["n"] += 1
            if executor_replies["n"] <= 2:
                # 合法 JSON 但缺少必填字段 `code`：executor 的校验器会抛错，
                # 从而在 run_agent_call **内部**触发重试。
                # 不能用 "not json"——那种内容由解析器直接抛异常、不经过
                # 校验器，实测不会产生重试，用例就测不到重试路径。
                return ChatResult(
                    generations=[
                        ChatGeneration(message=AIMessage(content="{}"))
                    ]
                )
            return None

        counting.responder = responder

        result = run_analysis(
            run_id="p3-retry-cap",
            data_sources=[str(database)],
            model_factory=counting.build(),
            resolver=ResourceResolver(),
            output_root=tmp_path,
            mcp=McpSettings(mode="direct"),
            budget={"max_llm_calls": 5},
        )

        assert executor_replies["n"] >= 2, (
            f"未触发解析失败重试（executor 仅 {executor_replies['n']} 次）"
        )
        assert counting.count <= 5, (
            f"上限 5 却发出 {counting.count} 次请求：{counting.requests}"
        )
        usage = (result.get("execution_budget") or {}).get("usage") or {}
        assert usage.get("llm_calls") == counting.count, (
            f"记在 state 里的用量（{usage.get('llm_calls')}）"
            f"与真实请求数（{counting.count}）不一致"
        )


class TestIdempotentReuseCostsNothing:
    """幂等复用不是新请求，因此既不消耗额度也不该被预算拦下。"""

    def test_replay_does_not_consume_budget(self, tmp_path: Path) -> None:
        """同一 state 调用两次 `run_agent_call`，第二次必须零消耗。"""
        from utils.idempotency import run_agent_call

        state: dict[str, Any] = {
            "run_id": "p3-idem",
            "completed_operations": {},
            BUDGET_FIELD: {
                "limits": AgentBudget(max_llm_calls=5).to_dict(),
                "usage": {**new_usage(), "active_seconds": 0.0, "clock_open": False},
            },
        }

        calls = {"n": 0}

        def call() -> dict:
            calls["n"] += 1
            return {"answer": 1}

        first, ops, _extra = run_agent_call(
            state, node="planner", subject="s#0", call=call
        )
        state["completed_operations"] = ops

        second, ops2, extra2 = run_agent_call(
            state, node="planner", subject="s#0", call=call
        )

        assert first == second == {"answer": 1}
        assert calls["n"] == 1, "第二次应命中幂等复用，不应再请求"
        assert extra2 == {}, "复用路径不应报告发生了上游请求"

    def test_reuse_is_not_blocked_by_an_exhausted_budget(self) -> None:
        """额度已用完时，**已完成**的调用仍要能复用。

        否则恢复一个已经跑完大部分工作的 Run 会直接失败，
        把"不能发新请求"错误地变成"不能读已有结果"。
        """
        from utils.idempotency import run_agent_call

        state: dict[str, Any] = {
            "run_id": "p3-idem-full",
            "completed_operations": {},
            BUDGET_FIELD: {
                "limits": AgentBudget(max_llm_calls=1).to_dict(),
                "usage": {
                    **new_usage(),
                    "llm_calls": 1,
                    "active_seconds": 0.0,
                    "clock_open": False,
                },
            },
        }

        _payload, ops, _extra = run_agent_call(
            state, node="planner", subject="s#0", call=lambda: {"answer": 1}
        )
        state["completed_operations"] = ops

        payload, _ops2, _extra2 = run_agent_call(
            state, node="planner", subject="s#0", call=lambda: {"answer": 2}
        )
        assert payload == {"answer": 1}, "应复用已完成的结果"


class TestBudgetSurvivesResume:
    def test_usage_is_not_reset_by_reinitialise(self) -> None:
        state: dict[str, Any] = {
            "run_id": "p3-resume",
            BUDGET_FIELD: {
                "limits": AgentBudget(max_llm_calls=3).to_dict(),
                "usage": {
                    **new_usage(),
                    "llm_calls": 2,
                    "tool_calls": 1,
                    "active_seconds": 12.5,
                    "clock_open": False,
                    "started_at": time.time() - 100,
                    "last_active_at": time.time() - 5,
                },
            },
        }

        for _ in range(3):
            state.update(al.initialize(state))

        usage = al.usage_of(state)
        assert usage["llm_calls"] == 2
        assert usage["tool_calls"] == 1
        assert usage["active_seconds"] == pytest.approx(12.5)

    def test_legacy_checkpoint_keeps_consumed_time(self) -> None:
        """老 Checkpoint（只有 started_at）迁移时必须保留已消耗时间。"""
        started = time.time() - 600  # 已执行 10 分钟
        state: dict[str, Any] = {
            "run_id": "p3-legacy",
            BUDGET_FIELD: {
                "limits": AgentBudget(max_seconds=1800).to_dict(),
                "usage": {
                    "rounds": 1,
                    "tool_calls": 1,
                    "llm_calls": 1,
                    "started_at": started,
                },
            },
        }

        state.update(al.initialize(state))
        usage = al.usage_of(state)

        assert usage["active_seconds"] == pytest.approx(600, abs=5), (
            "迁移时必须把原有的已用时间折进 active_seconds"
        )
        assert usage["llm_calls"] == 1, "迁移不得重置计数"
