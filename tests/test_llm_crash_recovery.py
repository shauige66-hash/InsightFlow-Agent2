"""P1 补验收：模型调用在崩溃与恢复下的真实行为。

三种崩溃时点，分别统计**实际上游请求次数**（由 Mock 自己记录，不经账本），
用来回答"恢复时到底会不会重复请求上游"。

    场景 1  before_request   请求尚未发出
    场景 2  after_request    上游已收到并返回，但节点未把结果写进 state
    场景 3  节点提交后       崩溃发生在节点完成之后

结论由数据给出，不靠断言掩盖：如果发生重复请求，测试就断言重复确实发生，
并把它记录为已知边界，而不去修改 Mock 让它看起来正常。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DRIVER = PROJECT_ROOT / "scripts" / "recovery_driver.py"
PYTHON = sys.executable


def plotter_operation(run_id: str, chart_id: str, chart_index: int = 0) -> str:
    """绘图 Agent 的模型调用操作 ID。

    P2-B1 起 subject 带上逻辑尝试序号：绘图节点用图表序号作区分
    （`attempt_subject(chart_id, index)`），因此 ID 末尾是 `#<index>`。
    这里集中构造，避免测试里散落拼接字符串。
    """
    return f"{run_id}:llm:plotter:{chart_id}#{chart_index}"


def run_phase(
    phase: str,
    *,
    workdir: Path,
    run_id: str,
    data_source: Path,
    crash_on: str | None = None,
    crash_at: str = "after_request",
    grant_llm_retry: bool = False,
    timeout: int = 300,
) -> dict:
    command = [
        str(PYTHON),
        str(DRIVER),
        phase,
        "--workdir",
        str(workdir),
        "--run-id",
        run_id,
        "--data-source",
        str(data_source),
        "--crash-at",
        crash_at,
    ]
    if crash_on:
        command += ["--crash-on", crash_on]
    if grant_llm_retry:
        command.append("--grant-llm-retry")

    import os

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(PROJECT_ROOT),
        timeout=timeout,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )

    if completed.returncode != 0 and not completed.stdout.strip():
        return {"crashed": True, "returncode": completed.returncode}

    line = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else "{}"
    try:
        payload = json.loads(line)
    except ValueError:
        payload = {"raw": completed.stdout[-500:]}
    payload["returncode"] = completed.returncode
    return payload


def upstream_calls(workdir: Path, agent: str | None = None) -> int:
    """Mock Provider 记录的真实请求次数。

    这是**实际上游请求**的计数，独立于账本与 Checkpoint，
    因此可以用它判断是否发生了重复调用。
    """
    path = workdir / "llm_calls.txt"
    if not path.exists():
        return 0
    calls = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    if agent is None:
        return len(calls)
    return sum(1 for call in calls if call == agent)


def ledger_summary(workdir: Path, run_id: str) -> dict:
    sys.path.insert(0, str(PROJECT_ROOT))
    from utils import llm_ledger

    return llm_ledger.summarize(run_id, root=workdir)


@pytest.fixture
def workspace(tmp_path: Path) -> tuple[Path, Path]:
    workdir = tmp_path / "run"
    workdir.mkdir()

    source = tmp_path / "customers.csv"
    source.write_text(
        "customer_id,customer_name,region\nC001,Alice,East\nC002,Bob,West\n",
        encoding="utf-8",
    )
    return workdir, source


class TestScenario1CrashBeforeRequest:
    """请求尚未发出时崩溃：上游没有收到任何东西，恢复可安全重发。"""

    def test_plotter_request_never_reached_upstream(self, workspace) -> None:
        workdir, source = workspace
        run_id = "llm-before-plotter"

        crashed = run_phase(
            "start",
            workdir=workdir,
            run_id=run_id,
            data_source=source,
            crash_on="plotter",
            crash_at="before_request",
        )
        assert crashed.get("crashed") is True

        # 崩溃前 plotter 从未真正发出请求
        assert upstream_calls(workdir, "plotter") == 0

        # 账本仍记录了"已开始"——这正是它优于 Checkpoint 的地方
        summary = ledger_summary(workdir, run_id)
        assert plotter_operation(run_id, "chart_1") in summary["unconfirmed_operations"]

    def test_resume_pauses_instead_of_resending(self, workspace) -> None:
        """未确认的调用默认不重发：暂停等人工决定。"""
        workdir, source = workspace
        run_id = "llm-before-resume"

        run_phase(
            "start",
            workdir=workdir,
            run_id=run_id,
            data_source=source,
            crash_on="plotter",
            crash_at="before_request",
        )
        before_total = upstream_calls(workdir)

        resumed = run_phase(
            "resume", workdir=workdir, run_id=run_id, data_source=source
        )

        assert resumed.get("interrupted") is True, "未确认的调用应当暂停等待决定"
        pending = resumed["pending_approval"]
        assert pending["kind"] == "llm_replay"
        assert plotter_operation(run_id, "chart_1") in pending["summary"]["operations"]

        # 关键：没有向上游重发任何请求
        assert upstream_calls(workdir) == before_total

    def test_granting_retry_resumes_the_run(self, workspace) -> None:
        """人工批准后才真正重发。"""
        workdir, source = workspace
        run_id = "llm-before-granted"

        run_phase(
            "start",
            workdir=workdir,
            run_id=run_id,
            data_source=source,
            crash_on="plotter",
            crash_at="before_request",
        )
        before_total = upstream_calls(workdir)

        finished = run_phase(
            "resume",
            workdir=workdir,
            run_id=run_id,
            data_source=source,
            grant_llm_retry=True,
        )

        assert finished.get("run_status") == "completed"
        # 获准重发后 plotter 多了一次请求——这是明确批准的代价
        assert upstream_calls(workdir, "plotter") == 1
        assert upstream_calls(workdir) > before_total


class TestScenario2CrashAfterRequestBeforePersist:
    """上游已收到请求并返回，但结果未持久化。

    这是最危险的情况：上游可能已经计费，而本地没有任何结果记录。
    系统同样选择**暂停**而不是静默重发。
    """

    def test_upstream_received_the_request(self, workspace) -> None:
        workdir, source = workspace
        run_id = "llm-after-plotter"

        crashed = run_phase(
            "start",
            workdir=workdir,
            run_id=run_id,
            data_source=source,
            crash_on="plotter",
            crash_at="after_request",
        )
        assert crashed.get("crashed") is True

        # 上游确实收到并返回了这一次请求
        assert upstream_calls(workdir, "plotter") == 1

    def test_resume_does_not_silently_resend(self, workspace) -> None:
        workdir, source = workspace
        run_id = "llm-after-resume"

        run_phase(
            "start",
            workdir=workdir,
            run_id=run_id,
            data_source=source,
            crash_on="plotter",
            crash_at="after_request",
        )
        before_plotter = upstream_calls(workdir, "plotter")
        before_total = upstream_calls(workdir)

        resumed = run_phase(
            "resume", workdir=workdir, run_id=run_id, data_source=source
        )

        assert resumed.get("interrupted") is True
        assert resumed["pending_approval"]["kind"] == "llm_replay"

        # 没有重复请求：上游计数不变
        assert upstream_calls(workdir, "plotter") == before_plotter == 1
        assert upstream_calls(workdir) == before_total

    def test_ledger_records_the_unconfirmed_call(self, workspace) -> None:
        """账本能指出这次调用没有完成记录，供人工判断。"""
        workdir, source = workspace
        run_id = "llm-after-ledger"

        run_phase(
            "start",
            workdir=workdir,
            run_id=run_id,
            data_source=source,
            crash_on="plotter",
            crash_at="after_request",
        )

        summary = ledger_summary(workdir, run_id)

        assert summary["total_started"] == summary["total_completed"] + 1
        assert plotter_operation(run_id, "chart_1") in summary["unconfirmed_operations"]

    def test_works_for_executor_too(self, workspace) -> None:
        """Executor 同样被覆盖，不只是 Plotter。"""
        workdir, source = workspace
        run_id = "llm-after-executor"

        crashed = run_phase(
            "start",
            workdir=workdir,
            run_id=run_id,
            data_source=source,
            crash_on="executor",
            crash_at="after_request",
        )
        assert crashed.get("crashed") is True
        assert upstream_calls(workdir, "executor") == 1
        # planner 已提交，不会重跑
        assert upstream_calls(workdir, "planner") == 1

        before_total = upstream_calls(workdir)

        resumed = run_phase(
            "resume", workdir=workdir, run_id=run_id, data_source=source
        )

        assert resumed.get("interrupted") is True
        assert upstream_calls(workdir) == before_total
        assert upstream_calls(workdir, "executor") == 1


class TestScenario3CrashAfterCheckpointCommit:
    """Checkpoint 已成功提交后崩溃：应从提交点继续，不重复已完成的工作。"""

    def test_completed_planner_is_not_replayed(self, workspace) -> None:
        workdir, source = workspace
        run_id = "llm-committed"

        run_phase(
            "start",
            workdir=workdir,
            run_id=run_id,
            data_source=source,
            crash_on="executor",
            crash_at="after_request",
        )

        # planner 在上一次运行中已提交，只调用过一次
        assert upstream_calls(workdir, "planner") == 1

        run_phase(
            "resume",
            workdir=workdir,
            run_id=run_id,
            data_source=source,
            grant_llm_retry=True,
        )

        # 恢复没有让 planner 再跑一次
        assert upstream_calls(workdir, "planner") == 1

    def test_resume_after_full_completion_replays_nothing(self, workspace) -> None:
        """完全正常结束后再 resume：不得有任何新的上游请求。"""
        workdir, source = workspace
        run_id = "llm-complete"

        finished = run_phase(
            "start", workdir=workdir, run_id=run_id, data_source=source
        )
        assert finished["run_status"] == "completed"

        before = upstream_calls(workdir)
        ledger_before = ledger_summary(workdir, run_id)

        again = run_phase(
            "resume", workdir=workdir, run_id=run_id, data_source=source
        )

        assert again["run_status"] == "completed"
        assert upstream_calls(workdir) == before, "终态恢复产生了新的上游请求"
        assert ledger_summary(workdir, run_id)["total_started"] == ledger_before[
            "total_started"
        ]


class TestDeclineSemantics:
    """人工选择 skip 后的状态处理。

    语义是两段的：
    1. 第一次恢复只是**呈现**中断，系统还不知道人要选什么；
    2. 之后再恢复且未带重发许可，即视为放弃，Run 落到 failed 终态。
    终态之后不再执行任何节点，也不会反复询问同一个决定。
    """

    def _crash_then_pause(self, workdir: Path, source: Path, run_id: str) -> dict:
        run_phase(
            "start",
            workdir=workdir,
            run_id=run_id,
            data_source=source,
            crash_on="plotter",
            crash_at="after_request",
        )
        return run_phase(
            "resume", workdir=workdir, run_id=run_id, data_source=source
        )

    def test_first_resume_only_presents_the_interrupt(self, workspace) -> None:
        workdir, source = workspace
        run_id = "decline-present"

        paused = self._crash_then_pause(workdir, source, run_id)

        assert paused["interrupted"] is True
        assert paused["pending_approval"]["kind"] == "llm_replay"
        # 仅呈现，不做任何执行
        assert paused["visualizations"] == 0

    def test_declining_produces_a_terminal_failed_state(self, workspace) -> None:
        workdir, source = workspace
        run_id = "decline-terminal"

        self._crash_then_pause(workdir, source, run_id)
        declined = run_phase(
            "resume", workdir=workdir, run_id=run_id, data_source=source
        )

        assert declined["run_status"] == "failed"
        assert declined["analysis_status"] == "failed"
        assert declined["status_reason"] == "存在未确认的模型调用，且未获准重发"
        assert declined["interrupted"] is False
        # 放弃的步骤不会产生产物
        assert declined["visualizations"] == 0

    def test_terminal_run_is_not_executed_again(self, workspace) -> None:
        workdir, source = workspace
        run_id = "decline-stable"

        self._crash_then_pause(workdir, source, run_id)
        run_phase("resume", workdir=workdir, run_id=run_id, data_source=source)

        calls_after_decline = upstream_calls(workdir)

        # 终态之后再来一次：状态不变，也不产生新的上游请求
        again = run_phase(
            "resume", workdir=workdir, run_id=run_id, data_source=source
        )

        assert again["run_status"] == "failed"
        assert upstream_calls(workdir) == calls_after_decline

    def test_decline_is_recorded_in_the_run_log(self, workspace) -> None:
        workdir, source = workspace
        run_id = "decline-logged"

        self._crash_then_pause(workdir, source, run_id)
        run_phase("resume", workdir=workdir, run_id=run_id, data_source=source)

        log_path = workdir / "logs" / f"{run_id}.jsonl"
        events = [
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        declined = [e for e in events if e["event"] == "llm_replay_declined"]
        assert declined, "放弃重发必须有审计记录"
        assert declined[0]["status"] == "failed"

    def test_retry_still_completes(self, workspace) -> None:
        """批准重发后仍然能正常完成——放弃路径没有破坏 retry 路径。"""
        workdir, source = workspace
        run_id = "decline-retry"

        self._crash_then_pause(workdir, source, run_id)

        finished = run_phase(
            "resume",
            workdir=workdir,
            run_id=run_id,
            data_source=source,
            grant_llm_retry=True,
        )

        assert finished["run_status"] == "completed"
        assert finished["visualizations"] == 1


class TestNoExactlyOnceClaim:
    """显式确认：系统不提供 exactly-once，边界要能被测试观察到。"""

    def test_ledger_cannot_tell_whether_upstream_processed_the_request(
        self, workspace
    ) -> None:
        """`started` 无完成记录时，账本只能报告"未确认"，不能断言是否已计费。"""
        workdir, source = workspace
        run_id = "llm-ambiguous"

        run_phase(
            "start",
            workdir=workdir,
            run_id=run_id,
            data_source=source,
            crash_on="plotter",
            crash_at="after_request",
        )

        summary = ledger_summary(workdir, run_id)

        # 账本给出的事实仅此而已
        assert summary["unconfirmed_operations"]
        assert "processed" not in summary
        assert "charged" not in summary
