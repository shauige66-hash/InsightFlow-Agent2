"""老 Checkpoint 的时间预算迁移语义。

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

from pathlib import Path

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


class TestRealInterruptResume:
    """真实 interrupt / resume 下迁移规则是否可达。

    上面的用例直接构造带 `pending_approval` 的 state，验证的是规则本身。
    这里跑真实 LangGraph，回答两个问题：审批负载到底落在哪里，
    以及老格式的预算在恢复时会不会被补记等待时间。
    """

    def test_the_approval_payload_lives_in_the_task_not_in_state(
        self, tmp_path: Path
    ) -> None:
        from langgraph.checkpoint.sqlite import SqliteSaver

        from ingestion.resources import ResourceResolver
        from services.analysis_service import read_run_state, run_analysis
        from tests.conftest import CUSTOMERS_CSV
        from tests.test_human_in_the_loop import _factory

        run_id = "migration-interrupt"

        with SqliteSaver.from_conn_string(str(tmp_path / "cp.sqlite")) as saver:
            run_analysis(
                run_id=run_id,
                data_sources=[str(CUSTOMERS_CSV)],
                model_factory=_factory(run_id),
                checkpointer=saver,
                resolver=ResourceResolver(),
                output_root=tmp_path,
                require_plan_approval=True,
            )

            config = {"configurable": {"thread_id": run_id}}
            values = saver.get_tuple(config).checkpoint["channel_values"]
            assert "pending_approval" not in values, (
                "state 里出现了审批负载；迁移的等待分支变得可达，需要重新评估该分支"
            )

            pending = read_run_state(run_id, checkpointer=saver).get("pending_approval") or {}
            assert pending.get("kind") == "plan_review"
            # 证据存在，只是在 checkpoint 的 task 里，不在 state 里
            assert isinstance(pending.get("created_at"), (int, float))

    def test_a_legacy_budget_is_not_charged_the_wait_on_resume(
        self, tmp_path: Path
    ) -> None:
        """老格式预算从审批中断处恢复：等待不计入，整段墙钟也不会被补记。

        loader 已经提交过，恢复时不重放，因此 `initialize` 不会执行；那条
        "折算墙钟并封顶"的分支在这里不生效。结果是不会因为等待而失败，
        也不会把等待算进执行时间。
        """
        import time

        from langgraph.checkpoint.sqlite import SqliteSaver
        from langgraph.types import Command

        from ingestion.resources import ResourceResolver
        from services.analysis_service import run_analysis
        from tests.conftest import CUSTOMERS_CSV
        from tests.test_human_in_the_loop import _factory
        from workflow import create_workflow

        run_id = "migration-legacy-resume"
        started_at = time.time() - 2.5 * 3600

        with SqliteSaver.from_conn_string(str(tmp_path / "cp.sqlite")) as saver:
            run_analysis(
                run_id=run_id,
                data_sources=[str(CUSTOMERS_CSV)],
                model_factory=_factory(run_id),
                checkpointer=saver,
                resolver=ResourceResolver(),
                output_root=tmp_path,
                require_plan_approval=True,
            )

            config = {"configurable": {"thread_id": run_id}}
            create_workflow(checkpointer=saver).update_state(
                config,
                {
                    BUDGET_FIELD: {
                        "limits": AgentBudget(max_seconds=1800.0).to_dict(),
                        "usage": {
                            "rounds": 1,
                            "tool_calls": 0,
                            "llm_calls": 1,
                            "started_at": started_at,
                        },
                    }
                },
            )
            persisted = saver.get_tuple(config).checkpoint["channel_values"]
            assert "active_seconds" not in persisted[BUDGET_FIELD]["usage"], (
                "没有写成老格式，这个用例就不成立"
            )

            result = run_analysis(
                run_id=run_id,
                data_sources=[str(CUSTOMERS_CSV)],
                model_factory=_factory(run_id),
                checkpointer=saver,
                resolver=ResourceResolver(),
                output_root=tmp_path,
                resume=Command(resume={"decision": "approve"}),
                require_plan_approval=True,
            )

        assert result["analysis_status"] == "completed", result.get("status_reason")
        usage = usage_of(result)
        # 计数不被重置
        assert usage["llm_calls"] >= 1
        # 2.5 小时的等待没有被算进执行时间，也没有被折算成 1800 秒的封顶值
        assert elapsed_seconds(result) < 120, (
            f"等待被计入了执行时间：{elapsed_seconds(result)}"
        )
