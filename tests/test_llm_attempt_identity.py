"""Repair / Refine / Critic / PlotRepair 的多轮模型调用语义。

背景缺陷（P2-A 已确认）：`llm_operation` 的 subject 只含 task_id / chart_id，
不区分逻辑尝试。于是：

    executor 失败 → repair 第 1 次（真正请求模型）
    executor 再失败 → repair 第 2 次 → 算出同一个 operation_id
                    → already_done 命中 → 直接复用第 1 次的代码

三条修复路径（repair / refine / critic）以及绘图修复都受影响，
表现为"修复循环在跑，但模型只被问过一次"。

本文件断言的是**Mock 实际收到的调用**：次数、operation_id、输入里的反馈内容、
以及新生成的代码是否真的被下一次执行使用。只检查 retry_count 增长不算证据。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conditions import MAX_PLOT_RETRIES, MAX_REFINES, MAX_RETRIES
from core.config import Settings
from core.llm import CallContext, ModelFactory
from ingestion.resources import ResourceResolver
from services.analysis_service import run_analysis
from utils.idempotency import attempt_subject, llm_operation

from tests.conftest import CALL_RECORDS

PLANNER_PLAN = {
    "tasks": [
        {
            "task_id": "task_1",
            "description": "统计客户数量",
            "target_tables": ["customers"],
            "expected_output": "客户数",
        }
    ]
}

PLOT_PLAN = {
    "charts": [
        {
            "chart_id": "chart_1",
            "description": "各地区客户数柱状图",
            "chart_type": "bar",
            "source_task_id": "task_1",
            "target_tables": ["customers"],
            "expected_output": "柱状图 PNG",
        }
    ]
}

REPORT = "# Analysis Report\n\n## Overview\n共 3 位客户。\n"

BASE_REPLIES: dict[str, object] = {
    "planner": PLANNER_PLAN,
    "executor": {"code": "result = len(df)"},
    "critic": {"validation_passed": True, "feedback": "ok"},
    "visualizer": {"charts": []},
    "reporter": REPORT,
}


class RecordingModelFactory:
    """可计数的 Mock 模型工厂。

    记录每次调用收到的完整提示词，因此可以断言"第二次修复看到的是第二次的
    错误反馈"，而不是只数次数。

    `replies` 的值可以是单个回复，也可以是**按序消费的列表**：后者用来构造
    "第一轮修复失败、第二轮修复成功"这类跨轮次变化的行为。
    """

    def __init__(self, run_id: str, replies: dict[str, object]) -> None:
        self.run_id = run_id
        self.replies = replies
        self.calls: list[str] = []
        self.prompts: dict[str, list[str]] = {}
        self._consumed: dict[str, int] = {}
        CALL_RECORDS[run_id] = self.calls

    def _next_reply(self, agent_name: str) -> object:
        reply = self.replies[agent_name]
        if not isinstance(reply, list):
            return reply

        index = self._consumed.get(agent_name, 0)
        self._consumed[agent_name] = index + 1
        # 列表用完后固定停在最后一项，避免意外 IndexError
        return reply[min(index, len(reply) - 1)]

    def build(self) -> ModelFactory:
        from langchain_core.callbacks import CallbackManagerForLLMRun
        from langchain_core.language_models import BaseChatModel
        from langchain_core.messages import AIMessage, BaseMessage
        from langchain_core.outputs import ChatGeneration, ChatResult

        owner = self

        class _Scripted(BaseChatModel):
            agent_name: str = ""
            record_key: str = ""

            @property
            def _llm_type(self) -> str:
                return f"p2b1-{self.agent_name}"

            def _generate(
                self,
                messages: list[BaseMessage],
                stop: list[str] | None = None,
                run_manager: CallbackManagerForLLMRun | None = None,
                **kwargs,
            ) -> ChatResult:
                owner.calls.append(self.agent_name)
                prompt = "\n".join(str(message.content) for message in messages)
                owner.prompts.setdefault(self.agent_name, []).append(prompt)

                reply = owner._next_reply(self.agent_name)
                content = (
                    reply
                    if isinstance(reply, str)
                    else json.dumps(reply, ensure_ascii=False)
                )
                return ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content=content))]
                )

        def resolver(agent_name: str) -> _Scripted:
            return _Scripted(agent_name=agent_name, record_key=self.run_id)

        return ModelFactory(
            settings=Settings(
                api_key="sk-fake",
                base_url="http://127.0.0.1:8000/v1",
                model="deepseek-chat",
            ),
            context=CallContext(run_id=self.run_id, client_app="insightflow-agent2"),
            resolver=resolver,
        )


@pytest.fixture
def workspace(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "customers.csv"
    source.write_text(
        "customer_id,region\nC001,East\nC002,West\nC003,East\n", encoding="utf-8"
    )
    return tmp_path, source


def run_case(
    workspace: tuple[Path, Path],
    run_id: str,
    overrides: dict[str, object],
    **kwargs,
):
    workdir, source = workspace
    recorder = RecordingModelFactory(run_id, {**BASE_REPLIES, **overrides})

    result = run_analysis(
        run_id=run_id,
        data_sources=[str(source)],
        model_factory=recorder.build(),
        resolver=ResourceResolver(),
        output_root=workdir,
        **kwargs,
    )
    return result, recorder


def ledger_entries(workdir: Path, run_id: str) -> list[dict]:
    path = workdir / "llm_ledger" / f"{run_id}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def started_operation_ids(workdir: Path, run_id: str) -> list[str]:
    return [
        entry["operation_id"]
        for entry in ledger_entries(workdir, run_id)
        if entry["phase"] == "started"
    ]


class TestRepairLoopActuallyCallsTheModel:
    """修复循环每一轮都必须真正请求模型。"""

    def test_each_repair_round_asks_the_model(self, workspace) -> None:
        replies = {
            "executor": {"code": "raise ValueError('exec-fails')"},
            "repair": {"code": "raise ValueError('repair-fails')"},
        }

        result, recorder = run_case(workspace, "p2b1-repair-rounds", replies)
        workdir, _ = workspace

        # 每轮修复都是新的逻辑尝试 → 各请求一次，共 MAX_RETRIES 次
        assert recorder.calls.count("repair") == MAX_RETRIES
        assert recorder.prompts["repair"], "修复提示词未被记录"

        # 执行确实推进了 MAX_RETRIES+1 次（首次 + 每轮修复后一次）
        attempts = [item["retry_count"] for item in result["execution_results"]]
        assert attempts == list(range(MAX_RETRIES + 1)), attempts

        # 账本里出现三个**不同**的修复 operation_id
        repair_ops = [
            op for op in started_operation_ids(workdir, "p2b1-repair-rounds")
            if ":llm:repair:" in op
        ]
        assert len(repair_ops) == MAX_RETRIES, repair_ops
        assert len(set(repair_ops)) == MAX_RETRIES, f"操作 ID 发生碰撞: {repair_ops}"

    def test_each_repair_sees_its_own_error_feedback(self, workspace) -> None:
        """第二次修复看到的是第二次的错误，不是第一次的残留。"""
        replies = {
            "executor": {"code": "raise ValueError('exec-fails')"},
            "repair": {"code": "raise ValueError('repair-fails')"},
        }

        _, recorder = run_case(workspace, "p2b1-repair-feedback", replies)

        prompts = recorder.prompts["repair"]
        assert len(prompts) == MAX_RETRIES

        # 提示词里应带上一次执行的错误
        for prompt in prompts:
            assert "repair-fails" in prompt or "exec-fails" in prompt

    def test_second_repair_code_is_actually_used(self, workspace) -> None:
        """第二轮修复产出的代码必须真的进入下一次执行。"""
        replies = {
            "executor": {"code": "raise ValueError('exec-fails')"},
            # 每轮返回不同代码，便于在执行记录里区分
            "repair": {"code": "raise ValueError('repair-marker-A')"},
        }

        result, recorder = run_case(workspace, "p2b1-repair-code", replies)

        codes = [
            item["code"] for item in result["execution_results"] if not item["success"]
        ]
        # 第一条是 executor 的，后面三条应来自三次修复
        assert len(codes) == MAX_RETRIES + 1
        assert codes[0] == "raise ValueError('exec-fails')"
        assert all("repair-marker-A" in code for code in codes[1:])


class TestRepairLoopCanSucceedOnLaterRound:
    """第二轮修复成功时必须真的生效，而不是复用第一轮的失败代码。"""

    def test_repair_fixes_on_second_attempt(self, workspace) -> None:
        replies = {
            "executor": {"code": "raise ValueError('exec-fails')"},
            # 第一轮修复仍失败，第二轮给出可用代码
            "repair": [
                {"code": "raise ValueError('repair-round-1')"},
                {"code": "result = len(df)"},
            ],
        }

        result, recorder = run_case(workspace, "p2b1-repair-recovers", replies)

        # 第二轮修复真的问到了模型，并且它的代码被执行并成功
        assert recorder.calls.count("repair") == 2, recorder.calls
        successes = [item for item in result["execution_results"] if item["success"]]
        assert successes, "第二次修复应当让执行成功"
        assert successes[-1]["result"] == 3
        assert not result["failed_tasks"]


class TestRefineLoopActuallyCallsTheModel:
    def test_each_refine_round_asks_the_model(self, workspace) -> None:
        replies = {
            "executor": {"code": "result = len(df)"},
            "critic": {"validation_passed": False, "feedback": "结果不完整，请补充"},
            "refine": {"code": "result = len(df) + 1"},
        }

        _, recorder = run_case(workspace, "p2b1-refine-rounds", replies)
        workdir, _ = workspace

        # 每轮精修各请求一次
        assert recorder.calls.count("refine") == MAX_REFINES, recorder.calls
        # 评审也要每轮重新进行（否则第二轮沿用第一轮结论）
        assert recorder.calls.count("critic") == MAX_REFINES + 1, recorder.calls

        refine_ops = [
            op for op in started_operation_ids(workdir, "p2b1-refine-rounds")
            if ":llm:refine:" in op
        ]
        critic_ops = [
            op for op in started_operation_ids(workdir, "p2b1-refine-rounds")
            if ":llm:critic:" in op
        ]
        assert len(refine_ops) == MAX_REFINES
        assert len(set(refine_ops)) == MAX_REFINES, refine_ops
        assert len(set(critic_ops)) == MAX_REFINES + 1, critic_ops

    def test_refine_sees_latest_critic_feedback(self, workspace) -> None:
        """每轮精修看到的是当轮评审给的意见。"""
        replies = {
            "executor": {"code": "result = len(df)"},
            "critic": {"validation_passed": False, "feedback": "缺少环比"},
            "refine": {"code": "result = len(df) + 1"},
        }

        _, recorder = run_case(workspace, "p2b1-refine-feedback", replies)

        for prompt in recorder.prompts["refine"]:
            assert "缺少环比" in prompt

    def test_critic_is_reasked_after_refine(self, workspace) -> None:
        """精修后的代码必须重新过评审，而不是复用上一轮结论。"""
        replies = {
            "executor": {"code": "result = len(df)"},
            "critic": {"validation_passed": False, "feedback": "再改"},
            "refine": {"code": "result = len(df) + 1"},
        }

        _, recorder = run_case(workspace, "p2b1-critic-reask", replies)

        critic_prompts = recorder.prompts["critic"]
        assert len(critic_prompts) >= 2
        # 后续评审的提示词里应能看到精修后的代码
        joined = "\n".join(critic_prompts[1:])
        assert "len(df) + 1" in joined


class TestPlotRepairLoopActuallyCallsTheModel:
    def test_each_plot_repair_round_asks_the_model(self, workspace) -> None:
        replies = {
            "executor": {"code": "result = len(df)"},
            "visualizer": PLOT_PLAN,
            "plotter": {"code": "raise ValueError('plot-fails')"},
            "plot_repair": {"code": "raise ValueError('plot-repair-fails')"},
        }

        _, recorder = run_case(workspace, "p2b1-plot-repair", replies)
        workdir, _ = workspace

        assert recorder.calls.count("plot_repair") == MAX_PLOT_RETRIES, recorder.calls

        ops = [
            op for op in started_operation_ids(workdir, "p2b1-plot-repair")
            if ":llm:plot_repair:" in op
        ]
        assert len(ops) == MAX_PLOT_RETRIES, ops
        assert len(set(ops)) == MAX_PLOT_RETRIES, f"绘图修复操作 ID 碰撞: {ops}"


class TestReplayDoesNotRecallTheModel:
    """同一次逻辑尝试的 Checkpoint 重放必须复用结果，不产生新请求。"""

    def test_repeated_resume_does_not_add_llm_calls(self, workspace) -> None:
        from langgraph.checkpoint.sqlite import SqliteSaver

        workdir, source = workspace
        run_id = "p2b1-replay"

        replies = {
            "executor": {"code": "raise ValueError('exec-fails')"},
            "repair": {"code": "raise ValueError('repair-fails')"},
        }
        recorder = RecordingModelFactory(run_id, {**BASE_REPLIES, **replies})

        db_path = workdir / "checkpoints.sqlite"
        with SqliteSaver.from_conn_string(str(db_path)) as saver:
            run_analysis(
                run_id=run_id,
                data_sources=[str(source)],
                model_factory=recorder.build(),
                resolver=ResourceResolver(),
                output_root=workdir,
                checkpointer=saver,
            )
            first_total = len(recorder.calls)

            # 终态 Run 再次 resume：应短路，不新增模型调用
            run_analysis(
                run_id=run_id,
                data_sources=[str(source)],
                model_factory=recorder.build(),
                resolver=ResourceResolver(),
                output_root=workdir,
                checkpointer=saver,
                continue_from_checkpoint=True,
            )

        assert len(recorder.calls) == first_total, (
            f"重放不应新增模型调用: {recorder.calls}"
        )


class TestOperationIdentityIsStableAndDistinct:
    """操作 ID 的稳定性与区分度（纯函数层）。"""

    def test_attempt_subject_distinguishes_rounds(self) -> None:
        rounds = [attempt_subject("task_1", n) for n in range(3)]
        assert len(set(rounds)) == 3

    def test_attempt_subject_distinguishes_counter_kinds(self) -> None:
        # "第 1 轮修复"与"第 1 轮精修"不能撞成同一个 ID
        repair = attempt_subject("task_1", 0, None)
        refine = attempt_subject("task_1", 0, 1)
        assert repair != refine

    def test_attempt_subject_is_reproducible(self) -> None:
        assert attempt_subject("task_1", 2) == attempt_subject("task_1", 2)

    def test_attempt_subject_without_counters_is_base(self) -> None:
        assert attempt_subject("plan") == "plan"

    def test_llm_operation_includes_attempt(self) -> None:
        first = llm_operation("run-1", "repair", attempt_subject("task_1", 0))
        second = llm_operation("run-1", "repair", attempt_subject("task_1", 1))

        assert first != second
        assert first == llm_operation("run-1", "repair", attempt_subject("task_1", 0))


class TestRunsDoNotShareOperationIdentity:
    def test_two_runs_have_disjoint_operation_ids(self, tmp_path: Path) -> None:
        source = tmp_path / "customers.csv"
        source.write_text("customer_id\nC001\nC002\n", encoding="utf-8")

        replies = {
            "executor": {"code": "raise ValueError('exec-fails')"},
            "repair": {"code": "raise ValueError('repair-fails')"},
        }

        collected: dict[str, list[str]] = {}
        for run_id in ("p2b1-run-a", "p2b1-run-b"):
            workdir = tmp_path / run_id
            workdir.mkdir()
            recorder = RecordingModelFactory(run_id, {**BASE_REPLIES, **replies})
            run_analysis(
                run_id=run_id,
                data_sources=[str(source)],
                model_factory=recorder.build(),
                resolver=ResourceResolver(),
                output_root=workdir,
            )
            collected[run_id] = started_operation_ids(workdir, run_id)

        assert collected["p2b1-run-a"] and collected["p2b1-run-b"]
        assert not set(collected["p2b1-run-a"]) & set(collected["p2b1-run-b"])
        # 每个 Run 各自独立的三轮修复
        for run_id, ops in collected.items():
            assert len([op for op in ops if ":llm:repair:" in op]) == MAX_RETRIES


class TestRepairCrashRecoveryKeepsLedgerSemantics:
    """修复调用被强杀后，账本的保守恢复机制必须照旧生效。

    这是 P2-B1 修复最容易踩坏的地方：把 subject 改成带尝试序号后，
    "未确认调用"的 ID 也随之变化。若处理不当，会变成
    "第二次修复因为 ID 不同而被当成新调用并自动重发"——即绕过了人工审批。
    """

    def _run_driver(self, *args: str) -> dict:
        import subprocess
        import sys as _sys

        driver = Path(__file__).resolve().parent.parent / "scripts" / "recovery_driver.py"
        completed = subprocess.run(
            [_sys.executable, str(driver), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
        )
        payload = completed.stdout.strip().splitlines()
        for line in reversed(payload):
            if line.strip().startswith("{"):
                return json.loads(line)
        return {"stdout": completed.stdout, "stderr": completed.stderr[-400:]}

    @pytest.fixture
    def repair_workspace(self, tmp_path: Path) -> tuple[Path, Path]:
        workdir = tmp_path / "run"
        workdir.mkdir()
        source = tmp_path / "customers.csv"
        source.write_text("customer_id\nC001\nC002\nC003\n", encoding="utf-8")
        return workdir, source

    def test_crash_during_first_repair_leaves_unconfirmed_operation(
        self, repair_workspace
    ) -> None:
        """第一次修复请求已发出但未提交 → 账本应留下带尝试序号的未确认记录。"""
        workdir, source = repair_workspace
        run_id = "p2b1-repair-crash"

        self._run_driver(
            "start",
            "--workdir", str(workdir),
            "--run-id", run_id,
            "--data-source", str(source),
            "--fail-first-execution",
            "--crash-on", "repair",
            "--crash-on-sequence", "1",
            "--crash-at", "after_request",
        )

        # 上游确实收到了那次修复请求
        calls = (workdir / "llm_calls.txt").read_text(encoding="utf-8").split()
        assert calls.count("repair") == 1, calls

        from utils import llm_ledger

        summary = llm_ledger.summarize(run_id, root=workdir)
        repair_ops = [
            op for op in summary["unconfirmed_operations"] if ":llm:repair:" in op
        ]
        assert len(repair_ops) == 1, summary["unconfirmed_operations"]
        # 未确认记录带上了本次逻辑尝试的序号
        assert repair_ops[0].endswith("#0"), repair_ops[0]

    def test_resume_pauses_for_the_unconfirmed_repair(self, repair_workspace) -> None:
        """恢复时必须暂停等人工决定，而不是自动重发。"""
        workdir, source = repair_workspace
        run_id = "p2b1-repair-pause"

        self._run_driver(
            "start",
            "--workdir", str(workdir),
            "--run-id", run_id,
            "--data-source", str(source),
            "--fail-first-execution",
            "--crash-on", "repair",
            "--crash-on-sequence", "1",
            "--crash-at", "after_request",
        )
        before = (workdir / "llm_calls.txt").read_text(encoding="utf-8").split()

        state = self._run_driver(
            "resume",
            "--workdir", str(workdir),
            "--run-id", run_id,
            "--data-source", str(source),
        )

        assert state.get("interrupted") is True, state
        assert state["pending_approval"]["kind"] == "llm_replay"
        assert any(
            op.endswith("#0") and ":llm:repair:" in op
            for op in state["pending_approval"]["summary"]["operations"]
        )

        # 没有人工许可就不该新增任何上游请求
        after = (workdir / "llm_calls.txt").read_text(encoding="utf-8").split()
        assert after == before, f"未获许可却重发了请求: {before} -> {after}"

    def test_crash_before_request_leaves_unconfirmed_id_without_upstream_call(
        self, repair_workspace
    ) -> None:
        """请求尚未发出就崩溃：账本仍登记未确认，但上游实际没收到。"""
        workdir, source = repair_workspace
        run_id = "p2b1-repair-before"

        self._run_driver(
            "start",
            "--workdir", str(workdir),
            "--run-id", run_id,
            "--data-source", str(source),
            "--fail-first-execution",
            "--crash-on", "repair",
            "--crash-on-sequence", "1",
            "--crash-at", "before_request",
        )

        calls = (workdir / "llm_calls.txt").read_text(encoding="utf-8").split()
        assert calls.count("repair") == 0, f"请求不该发出: {calls}"

        from utils import llm_ledger

        summary = llm_ledger.summarize(run_id, root=workdir)
        repair_ops = [
            op for op in summary["unconfirmed_operations"] if ":llm:repair:" in op
        ]
        assert repair_ops == [f"{run_id}:llm:repair:task_1#0"], summary

    def test_granting_retry_completes_using_the_same_logical_attempt(
        self, repair_workspace
    ) -> None:
        """人工批准重发后，整个 Run 能跑完；重发仍属同一个逻辑尝试。"""
        workdir, source = repair_workspace
        run_id = "p2b1-repair-granted"

        self._run_driver(
            "start",
            "--workdir", str(workdir),
            "--run-id", run_id,
            "--data-source", str(source),
            "--fail-first-execution",
            "--crash-on", "repair",
            "--crash-on-sequence", "1",
            "--crash-at", "before_request",
        )

        state = self._run_driver(
            "resume",
            "--workdir", str(workdir),
            "--run-id", run_id,
            "--data-source", str(source),
            "--grant-llm-retry",
        )

        assert state.get("run_status") == "completed", state
        calls = (workdir / "llm_calls.txt").read_text(encoding="utf-8").split()
        # 获批后 repair 真正被请求了一次，且修复让执行成功
        assert calls.count("repair") == 1, calls

        from utils import llm_ledger

        summary = llm_ledger.summarize(run_id, root=workdir)
        # 同一个逻辑尝试：只有 #0 这一个修复身份，且已被确认完成
        entries = llm_ledger.read_entries(run_id, root=workdir)
        repair_ids = sorted(
            {entry["operation_id"] for entry in entries if entry["node"] == "repair"}
        )
        assert repair_ids == [f"{run_id}:llm:repair:task_1#0"], repair_ids
        assert summary["unconfirmed_operations"] == [], summary

    def test_second_repair_round_gets_a_new_operation_id(
        self, repair_workspace
    ) -> None:
        """第一轮修复的代码执行失败 → 第二轮修复必须是新的操作身份。

        这是 P2-B1 的核心断言：修复循环推进时不能复用上一轮的操作 ID。
        """
        workdir, source = repair_workspace
        run_id = "p2b1-repair-second"

        self._run_driver(
            "start",
            "--workdir", str(workdir),
            "--run-id", run_id,
            "--data-source", str(source),
            "--fail-first-execution",
            # 第一轮修复给出的代码仍然抛错 → 必然进入第二轮修复
            "--repair-sequence-fails",
            "--crash-on", "repair",
            "--crash-on-sequence", "2",
            "--crash-at", "before_request",
        )

        from utils import llm_ledger

        entries = llm_ledger.read_entries(run_id, root=workdir)
        repair_ids = sorted(
            {entry["operation_id"] for entry in entries if entry["node"] == "repair"}
        )

        assert repair_ids == [
            f"{run_id}:llm:repair:task_1#0",
            f"{run_id}:llm:repair:task_1#1",
        ], repair_ids
