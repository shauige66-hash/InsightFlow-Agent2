"""P3 定点修复：HITL 时间预算语义，以及 P4 上下文度量的并发隔离。

## 任务三：等待人工审批不计入执行预算

`AgentBudget` 的文档说等待人工审批不应计入执行时间，但早期实现是
`elapsed_seconds = now - started_at`——**等待时间照样被算进去**，
文档与行为不符。

这里的用例用**受控时钟**（monkeypatch `time.time`）模拟超过 30 分钟的
审批暂停，不通过真实 sleep 等待。

## 任务四：并发 Run 之间的上下文度量串扰

9 个 Agent 早期把上下文度量放在模块级字典里，而 `RunCoordinator` 用
`ThreadPoolExecutor` 并发执行多个 Run。这里构造真实的并发交错，
证明会串写，并证明改成 `ContextVar` 后不再串写。
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from utils import agent_loop as al
from utils.agent_loop import (
    AgentBudget,
    BUDGET_FIELD,
    BudgetExceeded,
    elapsed_seconds,
    new_usage,
)
from utils.context_budget import (
    clear_context_reports,
    context_report_of,
    publish_context_report,
)

CLOCK_START = 1_000_000.0


def state_at(
    *,
    now: float,
    max_seconds: float = 1800.0,
    active_seconds: float = 0.0,
    clock_open: bool = False,
) -> dict[str, Any]:
    """一份带受控时钟起点的预算 state。"""
    return {
        "run_id": "hitl",
        BUDGET_FIELD: {
            "limits": AgentBudget(max_seconds=max_seconds).to_dict(),
            "usage": {
                **new_usage(),
                "started_at": now,
                "last_active_at": now,
                "active_seconds": active_seconds,
                "clock_open": clock_open,
            },
        },
    }


class TestWaitingForApprovalIsNotExecutionTime:
    def test_a_thirty_minute_pause_does_not_consume_the_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """模拟 30 分钟审批暂停：执行时间不增长，预算不被判定耗尽。

        时间推进的正确方式是"操作开始 → 操作结束"：只调用 `settle` 而不先
        `capture`，时钟一直是停的，那段时间本来就算作空闲。因此这里按真实
        序列写：capture 开时钟，5 秒后 settle 关时钟。
        """
        clock = {"now": CLOCK_START}
        monkeypatch.setattr(al.time, "time", lambda: clock["now"])

        state = state_at(now=CLOCK_START)
        state.update(al.capture_active_time(state))  # 操作开始
        clock["now"] = CLOCK_START + 5
        state.update(al.settle_active_time(state))  # 5 秒后结束

        executed = elapsed_seconds(state)
        assert executed == pytest.approx(5.0, abs=0.01)

        # 审批暂停 31 分钟——超过 max_seconds(1800) 的 30 分钟预算
        clock["now"] = CLOCK_START + 5 + 31 * 60

        after_pause = elapsed_seconds(state)
        assert after_pause == pytest.approx(executed, abs=0.01), (
            f"等待审批被计入了执行时间：{executed} → {after_pause}"
        )

        # 所以此时预算**没有**耗尽
        al.check(state)  # 不抛异常

    def test_work_after_a_long_pause_is_still_counted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """等待不计入，但等待之后做的真实工作必须计入。"""
        clock = {"now": CLOCK_START}
        monkeypatch.setattr(al.time, "time", lambda: clock["now"])

        state = state_at(now=CLOCK_START)

        # 第一段：5 秒
        state.update(al.capture_active_time(state))
        clock["now"] = CLOCK_START + 5
        state.update(al.settle_active_time(state))

        # 暂停 31 分钟
        clock["now"] = CLOCK_START + 5 + 31 * 60
        assert elapsed_seconds(state) == pytest.approx(5.0, abs=0.01)

        # 恢复后的第二段：3 秒
        state.update(al.capture_active_time(state))
        clock["now"] += 3
        state.update(al.settle_active_time(state))

        assert elapsed_seconds(state) == pytest.approx(8.0, abs=0.01), (
            "恢复后的真实工作没有被计入"
        )

    def test_budget_still_expires_from_real_work(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """等待不计入，但真实工作仍会耗尽预算——不能把两者一起放过。"""
        clock = {"now": CLOCK_START}
        monkeypatch.setattr(al.time, "time", lambda: clock["now"])

        state = state_at(now=CLOCK_START)

        # 40 次操作，每次"开始→2 秒后结束"
        for _ in range(40):
            clock["now"] += 30
            state.update(al.capture_active_time(state))
            clock["now"] += 2
            state.update(al.settle_active_time(state))

        # 40 × 2 秒 = 80 秒真实工作；再推进 40 分钟也不该增加
        worked = elapsed_seconds(state)
        assert worked == pytest.approx(80.0, abs=0.5), worked

        clock["now"] += 40 * 60
        assert elapsed_seconds(state) == pytest.approx(worked, abs=0.5)

        # 把 max_seconds 压到 60 秒后必须判为耗尽
        state[BUDGET_FIELD]["limits"]["max_seconds"] = 60.0
        with pytest.raises(BudgetExceeded) as excinfo:
            al.check(state)
        assert excinfo.value.scope == "seconds"

    def test_time_budget_survives_a_resume(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """恢复后沿用已消耗的时间，不重置为 0。"""
        clock = {"now": CLOCK_START}
        monkeypatch.setattr(al.time, "time", lambda: clock["now"])

        state = state_at(now=CLOCK_START)
        # 第一段工作：5 秒
        state.update(al.capture_active_time(state))
        clock["now"] = CLOCK_START + 5
        state.update(al.settle_active_time(state))

        consumed = elapsed_seconds(state)
        assert consumed == pytest.approx(5.0, abs=0.01)

        # 模拟"进程重启后恢复"：时间已经过去很久
        clock["now"] = CLOCK_START + 3600
        for _ in range(3):
            state.update(al.initialize(state))

        assert elapsed_seconds(state) == pytest.approx(consumed, abs=0.01), (
            "恢复后已消耗的时间被重置了"
        )

    def test_legacy_checkpoint_migrates_consumed_time(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """老 Checkpoint（只有 started_at）迁移时保留已消耗时间。"""
        clock = {"now": CLOCK_START}
        monkeypatch.setattr(al.time, "time", lambda: clock["now"])

        state = {
            "run_id": "hitl-legacy",
            BUDGET_FIELD: {
                "limits": AgentBudget(max_seconds=1800).to_dict(),
                "usage": {
                    "rounds": 1,
                    "tool_calls": 1,
                    "llm_calls": 1,
                    "started_at": CLOCK_START - 600,  # 已经跑了 10 分钟
                },
            },
        }

        state.update(al.initialize(state))
        usage = al.usage_of(state)

        assert usage["active_seconds"] == pytest.approx(600, abs=1), (
            "迁移必须把旧的已用时间折进 active_seconds"
        )
        assert usage["llm_calls"] == 1, "迁移不得重置计数"

    def test_time_is_not_advanced_by_waiting_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """完全不调用任何操作时，时间一点都不得前进。"""
        clock = {"now": CLOCK_START}
        monkeypatch.setattr(al.time, "time", lambda: clock["now"])

        state = state_at(now=CLOCK_START)
        clock["now"] = CLOCK_START + 7200  # 2 小时

        assert elapsed_seconds(state) == pytest.approx(0.0, abs=0.01)


class TestContextReportIsRunScoped:
    """P4 度量必须按执行上下文隔离。"""

    def teardown_method(self) -> None:
        clear_context_reports()

    def test_reports_do_not_leak_between_threads(self) -> None:
        """并发线程各写各的度量，互不可见。"""
        clear_context_reports()
        barrier = threading.Barrier(2)
        seen: dict[str, dict] = {}
        errors: list[BaseException] = []

        def worker(name: str, chars: int) -> None:
            try:
                publish_context_report("planner", {"node": "planner", "final_chars": chars})
                # 两个线程都写完之后再读，确保"后写覆盖先写"能被观察到
                barrier.wait(timeout=5)
                seen[name] = context_report_of("planner")
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        first = threading.Thread(target=worker, args=("a", 100))
        second = threading.Thread(target=worker, args=("b", 200))
        first.start()
        second.start()
        first.join(timeout=10)
        second.join(timeout=10)

        assert not errors, errors
        assert seen["a"]["final_chars"] == 100, (
            f"A 读到了别人的度量：{seen['a']}"
        )
        assert seen["b"]["final_chars"] == 200, (
            f"B 读到了别人的度量：{seen['b']}"
        )

    def test_module_globals_are_gone(self) -> None:
        """回归：Agent 模块不得再有可变的模块级度量字典。"""
        import importlib

        for name in (
            "planner",
            "executor",
            "repair",
            "refine",
            "critic",
            "visualizer",
            "plotter",
            "plot_repair",
            "reporter",
        ):
            module = importlib.import_module(f"agents.{name}")
            assert not hasattr(module, "LAST_CONTEXT_REPORT"), (
                f"agents.{name} 仍有模块级 LAST_CONTEXT_REPORT，"
                "并发 Run 之间会串写"
            )

    def test_reports_are_keyed_per_agent(self) -> None:
        """同一次执行里多个 Agent 的度量互不覆盖。"""
        clear_context_reports()
        publish_context_report("planner", {"node": "planner", "final_chars": 10})
        publish_context_report("critic", {"node": "critic", "final_chars": 20})

        assert context_report_of("planner")["final_chars"] == 10
        assert context_report_of("critic")["final_chars"] == 20
        assert context_report_of("unknown") == {}

    def test_clear_only_affects_the_current_context(self) -> None:
        clear_context_reports()
        publish_context_report("planner", {"node": "planner", "final_chars": 10})

        observed: dict = {}

        def worker() -> None:
            # 新线程有独立的上下文：看不到主线程那份
            observed["before"] = context_report_of("planner")
            publish_context_report("planner", {"node": "planner", "final_chars": 99})
            observed["after"] = context_report_of("planner")

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=5)

        assert observed["before"] == {}, (
            "子线程看到了主线程的度量，说明没有按上下文隔离"
        )
        assert observed["after"]["final_chars"] == 99
        # 主线程那份没有被影响
        assert context_report_of("planner")["final_chars"] == 10
