"""H1-F：结构化日志需要能回答"哪个 Run、哪个节点、花了多少额度、多久"。

这里固定三条契约：

1. 记录本身带 run_id / node / event / status / timestamp；
2. 模型调用的记录带上 `operation_id`，与账本、幂等登记用同一个值，
   否则预算用量无法回溯到具体调用；
3. 失败与执行类记录带上 `error_kind`，因为路由是按分类而不是按错误文案决定的；
4. 运行结束的记录带整轮耗时与预算快照。

`RunCoordinator._fail` 在 `run_analysis` 退出运行期上下文之后才执行，
因此它必须自己绑定 output_root，否则失败记录会落到进程当前目录。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.runtime import RunRuntime, use_runtime
from utils import run_logger
from utils.run_logger import create_run_id, log_event


@pytest.fixture
def log_root(tmp_path: Path):
    runtime = RunRuntime(
        run_id="log-contract",
        model_factory=None,  # type: ignore[arg-type]
        resolver=None,  # type: ignore[arg-type]
        output_root=tmp_path / "outputs",
    )
    with use_runtime(runtime):
        yield tmp_path / "outputs"


def read_records(log_root: Path, run_id: str) -> list[dict]:
    path = log_root / "logs" / f"{run_id}.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class TestRecordSchema:
    def test_every_record_carries_the_core_fields(self, log_root: Path) -> None:
        run_id = create_run_id()
        log_event(
            run_id=run_id,
            node="planner",
            event="plan_created",
            status="success",
            details={"task_count": 2},
        )

        records = read_records(log_root, run_id)
        assert len(records) == 1
        record = records[0]
        for field in ("timestamp", "run_id", "node", "event", "status", "details"):
            assert field in record, f"日志缺少字段 {field}"
        assert record["run_id"] == run_id
        assert record["node"] == "planner"

    def test_duration_is_recorded_when_given(self, log_root: Path) -> None:
        run_id = create_run_id()
        log_event(
            run_id=run_id,
            node="workflow",
            event="run_completed",
            status="completed",
            duration_ms=1234.567,
        )
        record = read_records(log_root, run_id)[0]
        assert record["duration_ms"] == pytest.approx(1234.57, abs=0.01)


class TestTheLedgerOperationIdIsLoggable:
    def test_llm_context_records_the_operation_id(self, log_root: Path) -> None:
        """模型调用记录必须带上与账本一致的操作 ID。"""
        from utils.idempotency import llm_operation

        run_id = create_run_id()
        operation_id = llm_operation(run_id, "planner", "plan#0")
        log_event(
            run_id=run_id,
            node="planner",
            event="llm_context",
            status="ok",
            details={"subject": "plan#0", "operation_id": operation_id, "context": {}},
        )

        record = read_records(log_root, run_id)[0]
        assert record["details"]["operation_id"] == operation_id
        # 与账本写入的是同一个值
        from utils import llm_ledger

        llm_ledger.record_started(
            run_id=run_id,
            operation_id=operation_id,
            node="planner",
            attempt=1,
            root=log_root,
        )
        entries = llm_ledger.read_entries(run_id, root=log_root)
        assert entries[0]["operation_id"] == record["details"]["operation_id"]


class TestRunCoordinatorFailureLogsToTheRunOutputRoot:
    def test_fail_binds_the_runtime_so_logs_do_not_land_in_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`_fail` 在运行期上下文之外执行，必须自己绑定 output_root。"""
        from services.run_coordinator import RunCoordinator
        from services.run_store import RunStore

        store = RunStore(tmp_path / "runs.sqlite")
        output_root = tmp_path / "api-outputs"
        coordinator = RunCoordinator(store, output_root=output_root)

        run_id = "coordinator-fail"
        store.create(run_id, data_sources=[], analysis_focus=None, sql_backend="direct")

        # 换一个当前工作目录：不绑定运行期上下文时日志会落到这里
        cwd = tmp_path / "elsewhere"
        cwd.mkdir()
        monkeypatch.chdir(cwd)

        coordinator._fail(run_id, RuntimeError("boom"), output_root=output_root)

        assert (output_root / "logs" / f"{run_id}.jsonl").exists(), (
            "失败记录没有落到 Run 的 output_root"
        )
        assert not (cwd / "outputs" / "logs").exists(), (
            "失败记录落到了当前工作目录，绕过了隔离目录"
        )

        coordinator.shutdown()
        store.close() if hasattr(store, "close") else None


class TestNoPromptOrKeyMaterialIsLogged:
    def test_context_report_carries_counts_not_content(self) -> None:
        """上下文度量只记体积与字段名，不记 Prompt 内容。"""
        from utils.context_budget import ContextReport

        report = ContextReport(node="planner", budget_chars=100, original_chars=500)
        report.truncated_fields.append("execution_results")
        payload = report.to_dict()

        assert set(payload) == {
            "node",
            "task_id",
            "budget_chars",
            "original_chars",
            "final_chars",
            "estimated_tokens",
            "token_estimate_method",
            "truncated",
            "truncated_fields",
            "over_budget",
            "kept_tables",
            "excluded_tables",
            "history_entries",
            # H4 增加：知识片段数、记忆条数、缓存是否命中。
            # 都是计数或布尔值，不含任何正文。
            "knowledge_chunks",
            "memory_entries",
            "knowledge_cache_hit",
        }
        assert "prompt" not in json.dumps(payload).lower()
