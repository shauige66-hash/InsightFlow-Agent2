"""H1：老 Checkpoint 的时间预算迁移语义。

P3 早期的预算快照只有 `started_at`，没有 `active_seconds`，当时的"已消耗
时间"按墙钟算，因此把人工审批的等待也算进了执行预算。老格式没有记录执行与
等待的分界，无法精确重建；这里的规则是保守且可解释的：

- 能证明当时正在等待审批（`pending_approval.created_at`）时只折算等待之前
  的那一段，等待不计入；
- 否则按墙钟折算但以 `max_seconds` 封顶，避免"等了两小时的老 Run 一恢复
  就立刻失败"。

无论走哪条分支，已消耗的时间都不退回，计数也不重置。
"""

from __future__ import annotations

import pytest

from utils import agent_loop as al
from utils.agent_loop import (
    BUDGET_FIELD,
    AgentBudget,
    BudgetExceeded,
    check,
    elapsed_seconds,
    initialize,
    usage_of,
)

START = 1_000_000.0


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> dict[str, float]:
    """只替换 `utils.agent_loop` 读到的时钟。"""
    state = {"now": START}
    monkeypatch.setattr(al.time, "time", lambda: state["now"])
    return state


def legacy_state(
    *,
    started_at: float,
    max_seconds: float = 1800.0,
    pending_approval: dict | None = None,
    llm_calls: int = 1,
) -> dict:
    """老格式快照：有 started_at，没有 active_seconds。"""
    state: dict = {
        "run_id": "legacy",
        BUDGET_FIELD: {
            "limits": AgentBudget(max_seconds=max_seconds).to_dict(),
            "usage": {
                "rounds": 1,
                "tool_calls": 1,
                "llm_calls": llm_calls,
                "started_at": started_at,
            },
        },
    }
    if pending_approval is not None:
        state["pending_approval"] = pending_approval
    return state


class TestWaitingIsNotChargedToExecution:
    def test_only_work_before_the_wait_is_carried(self, clock: dict) -> None:
        """执行 60 秒后开始等待，等待 2.5 小时后恢复：只折算 60 秒。"""
        wait_started = START + 60
        state = legacy_state(
            started_at=START,
            pending_approval={"kind": "plan_review", "created_at": wait_started},
        )
        # 等待 2.5 小时
        clock["now"] = wait_started + 2.5 * 3600

        state.update(initialize(state))
        usage = usage_of(state)

        assert usage["active_seconds"] == pytest.approx(60.0), (
            f"审批等待被算进了执行预算：{usage['active_seconds']}"
        )
        assert usage["llm_calls"] == 1, "迁移不得重置计数"

        # 恢复后仍有余量，不应立即失败
        check(state)

    def test_a_long_wait_does_not_exhaust_the_budget(self, clock: dict) -> None:
        state = legacy_state(
            started_at=START,
            max_seconds=1800.0,
            pending_approval={"kind": "report_publish", "created_at": START + 30},
        )
        clock["now"] = START + 10 * 3600
        state.update(initialize(state))

        assert elapsed_seconds(state) == pytest.approx(30.0)
        check(state)  # 不抛异常


class TestUnknownSplitFallsBackConservatively:
    def test_wall_clock_is_capped_at_the_limit(self, clock: dict) -> None:
        """无法判断等待时按墙钟折算，但不超过 max_seconds。"""
        state = legacy_state(started_at=START, max_seconds=600.0)
        clock["now"] = START + 5 * 3600  # 远超预算
        state.update(initialize(state))

        usage = usage_of(state)
        assert usage["active_seconds"] == pytest.approx(600.0), (
            "未封顶：老 Run 会因为一次长等待把额度一次性烧完"
        )
        # 封顶后不再有余量，但已消耗的时间没有被退回
        with pytest.raises(BudgetExceeded):
            check(state)

    def test_short_wall_clock_is_carried_as_is(self, clock: dict) -> None:
        state = legacy_state(started_at=START, max_seconds=1800.0)
        clock["now"] = START + 120
        state.update(initialize(state))

        assert usage_of(state)["active_seconds"] == pytest.approx(120.0)

    def test_a_decided_approval_is_not_reinterpreted(self, clock: dict) -> None:
        """审批已归档（只有 decided_at）时不得凭它反推等待区间。"""
        state = legacy_state(started_at=START, max_seconds=1800.0)
        state["approvals"] = [
            {"approval_id": "a1", "decision": "approve", "decided_at": START + 60}
        ]
        clock["now"] = START + 300
        state.update(initialize(state))

        assert usage_of(state)["active_seconds"] == pytest.approx(300.0), (
            "已归档的审批不应被当成等待区间"
        )


class TestNewFormatIsUntouched:
    def test_states_with_active_seconds_are_not_migrated(self, clock: dict) -> None:
        state: dict = {
            "run_id": "new",
            BUDGET_FIELD: {
                "limits": AgentBudget(max_seconds=1800.0).to_dict(),
                "usage": {
                    "rounds": 1,
                    "tool_calls": 1,
                    "llm_calls": 2,
                    "active_seconds": 12.5,
                    "clock_open": False,
                    "started_at": START - 10_000,
                    "last_active_at": START - 5,
                },
            },
        }
        clock["now"] = START + 5 * 3600

        state.update(initialize(state))
        usage = usage_of(state)

        assert usage["active_seconds"] == pytest.approx(12.5), (
            "新格式的准确执行时间被改写了"
        )
        assert usage["clock_open"] is False
        assert usage["llm_calls"] == 2

    def test_a_fresh_state_carries_nothing(self) -> None:
        state: dict = {"run_id": "fresh"}
        state.update(initialize(state))
        assert usage_of(state)["active_seconds"] == 0.0
        assert usage_of(state)["clock_open"] is False
