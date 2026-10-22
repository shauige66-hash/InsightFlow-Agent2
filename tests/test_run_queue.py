"""作业队列：入队、抢占唯一性、完成与失败记录。

跑真实 PostgreSQL：`SKIP LOCKED` 的语义没法在 SQLite 上假装。
"""

from __future__ import annotations

import pytest

import storage
from services import run_queue
from tests.postgres_support import postgres, requires_postgres  # noqa: F401

pytestmark = [requires_postgres]


@pytest.fixture()
def queue(postgres) -> None:
    """每个用例开始前清空队列表。"""
    from sqlalchemy import text

    run_queue.ensure_schema()
    with storage.session_scope(storage.database_url()) as session:
        session.execute(text("DELETE FROM run_jobs"))
        session.commit()
    yield


def _payload(run_id: str) -> dict:
    return {
        "data_sources": ["data/customers.csv"],
        "analysis_focus": f"分析 {run_id}",
        "knowledge_sources": [],
        "require_plan_approval": False,
        "require_report_approval": False,
        "sql_backend": "direct",
        "enable_dynamic_loop": True,
        "dynamic_limits": None,
    }


class TestEnqueueAndClaim:
    def test_an_empty_queue_returns_nothing(self, queue) -> None:
        assert run_queue.claim("w1") is None

    def test_a_job_round_trips_with_its_payload(self, queue) -> None:
        run_queue.enqueue("run_1", _payload("run_1"))

        claimed = run_queue.claim("w1")

        assert claimed is not None
        run_id, payload = claimed
        assert run_id == "run_1"
        assert payload["analysis_focus"] == "分析 run_1"
        assert payload["enable_dynamic_loop"] is True

    def test_a_job_is_claimed_only_once(self, queue) -> None:
        run_queue.enqueue("run_1", _payload("run_1"))

        first = run_queue.claim("w1")
        second = run_queue.claim("w2")

        assert first is not None and first[0] == "run_1"
        assert second is None

    def test_two_workers_get_different_jobs(self, queue) -> None:
        run_queue.enqueue("run_a", _payload("run_a"))
        run_queue.enqueue("run_b", _payload("run_b"))

        first = run_queue.claim("w1")
        second = run_queue.claim("w2")

        assert {first[0], second[0]} == {"run_a", "run_b"}

    def test_enqueueing_twice_keeps_one_job(self, queue) -> None:
        run_queue.enqueue("run_1", _payload("run_1"))
        run_queue.enqueue("run_1", _payload("run_1"))

        assert run_queue.depth().get("queued") == 1


class TestCompletion:
    def test_finishing_marks_it_done(self, queue) -> None:
        run_queue.enqueue("run_1", _payload("run_1"))
        claimed = run_queue.claim("w1")

        run_queue.finish("run_1", claim_token=claimed[1]["_job_claim"])

        assert run_queue.status_of("run_1") == "done"
        assert run_queue.depth().get("done") == 1

    def test_a_failure_records_its_reason(self, queue) -> None:
        run_queue.enqueue("run_1", _payload("run_1"))
        claimed = run_queue.claim("w1")

        run_queue.finish("run_1", error="BudgetExceeded: llm_calls", claim_token=claimed[1]["_job_claim"])

        assert run_queue.status_of("run_1") == "failed"
        assert "BudgetExceeded" in (run_queue.last_error("run_1") or "")

    def test_requeue_returns_a_claimed_job_to_the_queue(self, queue) -> None:
        run_queue.enqueue("run_1", _payload("run_1"))
        run_queue.claim("w1")

        run_queue.requeue("run_1")

        assert run_queue.status_of("run_1") == "queued"
        assert run_queue.claim("w2")[0] == "run_1"
        # 重试次数递增，说明"被抢占过"是可见事实
        assert run_queue.attempts("run_1") == 2


class TestGuards:
    def test_the_queue_refuses_to_run_on_sqlite(self, monkeypatch) -> None:
        from storage.config import StorageConfig

        storage.reset()
        storage.configure(StorageConfig(mode="local", local_db_path=":memory:"))
        try:
            with pytest.raises(run_queue.QueueUnavailable):
                run_queue.ensure_schema()
        finally:
            storage.reset()
