"""H5 的跨进程恢复：在决策、工具、修订之后崩溃，再由新进程继续。

用真实进程强杀（`INSIGHTFLOW_CRASH_AT`）而不是在内存里模拟，因为要验证的
正是"节点已经产生副作用、但 Checkpoint 尚未提交"这个窗口。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.postgres_support import (  # noqa: F401
    postgres,
    requires_postgres,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DRIVER = PROJECT_ROOT / "tests" / "h5_recovery_driver.py"


@pytest.fixture
def workspace(tmp_path: Path) -> tuple[Path, Path]:
    from tests.test_report_approval import build_database

    workdir = tmp_path / "run"
    workdir.mkdir()
    database = tmp_path / "customers.sqlite"
    build_database(database)
    return workdir, database


def phase(workdir: Path, database: Path, run_id: str, name: str, *, crash: str | None = None):
    command = [
        sys.executable,
        str(DRIVER),
        name,
        "--run-id",
        run_id,
        "--workdir",
        str(workdir),
        "--data",
        str(database),
    ]
    if crash:
        command += ["--crash-on", crash]

    # 继承父进程环境（Windows 上缺 SystemRoot/PATH 会让子进程直接起不来），
    # 只覆盖必要项，并清掉可能残留的崩溃注入变量。
    env = {
        key: value
        for key, value in os.environ.items()
        if key != "INSIGHTFLOW_CRASH_AT"
    }
    env.update({"PYTHONPATH": str(PROJECT_ROOT), "PYTHONIOENCODING": "utf-8"})

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(PROJECT_ROOT),
        timeout=300,
        env=env,
    )

    if completed.returncode != 0:
        return {
            "crashed": True,
            "returncode": completed.returncode,
            "stderr": completed.stderr[-2000:],
        }

    line = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else "{}"
    return json.loads(line)


class TestCrashRecovery:
    def test_a_crash_after_the_decision_is_recovered(self, workspace) -> None:
        workdir, database = workspace
        run_id = "h5-crash-decide"

        crashed = phase(workdir, database, run_id, "start", crash="h5_decide")
        assert crashed.get("crashed") is True
        assert crashed["returncode"] == 17

        resumed = phase(workdir, database, run_id, "resume")

        assert resumed["analysis_status"] == "completed"
        # 循环游标从提交点继续，不会因为重启清零
        assert resumed["loop_step"] >= 1
        assert resumed["decisions"] <= 3
        # 没有任何未确认的模型请求被留在账本里
        ledger = [
            json.loads(line)
            for line in (workdir / "llm_ledger" / f"{run_id}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        started = [item for item in ledger if item["phase"] == "started"]
        completed = [item for item in ledger if item["phase"] == "completed"]
        assert len(started) == len(completed)

    def test_a_crash_after_the_tool_is_recovered(self, workspace) -> None:
        workdir, database = workspace
        run_id = "h5-crash-tool"

        crashed = phase(workdir, database, run_id, "start", crash="h5_after_tool")
        assert crashed.get("crashed") is True

        resumed = phase(workdir, database, run_id, "resume")

        assert resumed["analysis_status"] == "completed"
        assert resumed["loop_step"] <= 2
        # 恢复后仍然只有一份报告
        reports = list((workdir / run_id).rglob("*.md"))
        assert len(reports) == 1, reports

    def test_the_state_after_a_crash_is_inspectable(self, workspace) -> None:
        workdir, database = workspace
        run_id = "h5-crash-inspect"

        phase(workdir, database, run_id, "start", crash="h5_decide")
        inspected = phase(workdir, database, run_id, "inspect")

        assert inspected["state"] is not None
        assert inspected["state"]["run_status"] in ("running", "interrupted")


@requires_postgres
class TestPostgresDynamicLoopRecovery:
    def test_the_loop_resumes_on_a_postgres_checkpoint(self, postgres, tmp_path: Path) -> None:
        """动态循环的状态在 PostgreSQL Checkpoint 上同样可恢复。"""
        from langgraph.types import Command

        from core.checkpoints import open_checkpointer
        from services.analysis_service import run_analysis
        from tests.scripted_queue import ScriptedQueueFactory, decision, task
        from tests.test_report_approval import build_database

        database = tmp_path / "customers.sqlite"
        build_database(database)
        # 每次用独立 thread：共享 PG 里同名线程会让"恢复"变成"复用已完成的运行"
        import uuid

        run_id = f"h5-pg-loop-{uuid.uuid4().hex[:8]}"
        replies = {
            "planner": {
                "tasks": [
                    {
                        "task_id": "task_1",
                        "description": "统计客户数量",
                        "target_tables": ["customers.customers"],
                        "expected_output": "客户数",
                    }
                ]
            },
            "executor": {"code": "SELECT COUNT(*) AS n FROM customers"},
            "critic": {"validation_passed": True, "feedback": "ok"},
            "visualizer": {"charts": []},
            "reporter": "# 报告\n",
            "decider": decision("continue"),
        }

        with open_checkpointer() as saver:
            paused = run_analysis(
                run_id=run_id,
                data_sources=[str(database)],
                model_factory=ScriptedQueueFactory(
                    run_id, {key: [value] for key, value in replies.items()}
                ).build(),
                output_root=tmp_path,
                enable_dynamic_loop=True,
                require_plan_approval=True,
                checkpointer=saver,
            )
        assert paused.get("__interrupt__")

        # 模拟重启：丢弃 Engine 再打开新的 Checkpoint 连接
        import storage

        storage.dispose_all()
        storage.configure()

        with open_checkpointer() as saver:
            resumed = run_analysis(
                run_id=run_id,
                data_sources=[str(database)],
                model_factory=ScriptedQueueFactory(
                    run_id, {key: [value] for key, value in replies.items()}
                ).build(),
                output_root=tmp_path,
                enable_dynamic_loop=True,
                require_plan_approval=True,
                checkpointer=saver,
                resume=Command(resume={"decision": "approve"}),
            )

        assert resumed["analysis_status"] == "completed"
        assert resumed["enable_dynamic_loop"] is True
        assert resumed["loop_step"] >= 1
