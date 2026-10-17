"""长期记忆：确认、召回、修改、删除、隔离与持久化。

跑真实 PostgreSQL。记忆与对话、知识库是三套独立存储，这里的断言也分开验证。
"""

from __future__ import annotations

import pytest

import storage
from services.memory_service import MemoryService, MemoryValidationError
from tests.postgres_support import (  # noqa: F401
    postgres,
    redis_available,
    redis_url,
    requires_postgres,
)

pytestmark = [requires_postgres]


@pytest.fixture
def memory(postgres) -> MemoryService:
    return MemoryService(owner="pg-test")


class TestConfirmation:
    def test_a_candidate_is_not_listed_or_injected(self, memory) -> None:
        candidate = memory.add(
            kind="preference",
            key="currency",
            value="今后销售额统一用人民币元表示。",
            status="candidate",
        )

        assert candidate["status"] == "candidate"
        assert memory.list() == []
        assert memory.list(include_candidates=True)[0]["memory_id"] == candidate["memory_id"]

    def test_confirming_makes_it_visible(self, memory) -> None:
        candidate = memory.add(
            kind="preference",
            key="currency",
            value="今后销售额统一用人民币元表示。",
            status="candidate",
        )
        confirmed = memory.confirm(candidate["memory_id"])

        assert confirmed["status"] == "confirmed"
        assert [item["key"] for item in memory.list()] == ["currency"]

    def test_provenance_and_timestamps_are_recorded(self, memory) -> None:
        record = memory.add(
            kind="preference",
            key="chart",
            value="图表优先使用柱状图。",
            status="confirmed",
            provenance={"session_id": "sess_1", "turn_id": "turn_1"},
        )

        assert record["provenance"]["session_id"] == "sess_1"
        assert record["created_at"] and record["updated_at"]


class TestRetrieval:
    def test_only_relevant_memories_are_returned(self, memory) -> None:
        memory.add(
            kind="preference",
            key="currency",
            value="今后销售额统一用人民币元表示。",
            status="confirmed",
        )
        memory.add(
            kind="report_format",
            key="font",
            value="报告统一使用 12 号字体。",
            status="confirmed",
        )

        hits = memory.retrieve("销售额用什么货币", scopes=["global"])

        assert [item["key"] for item in hits] == ["currency"]

    def test_unrelated_query_returns_nothing(self, memory) -> None:
        memory.add(
            kind="preference",
            key="currency",
            value="今后销售额统一用人民币元表示。",
            status="confirmed",
        )
        assert memory.retrieve("今天天气", scopes=["global"]) == []

    def test_dataset_scope_is_separate_from_global(self, memory) -> None:
        memory.add(
            kind="preference",
            key="currency",
            value="销售额用美元。",
            scope="dataset:sales",
            status="confirmed",
        )

        assert memory.retrieve("销售额货币", scopes=["global"]) == []
        assert memory.retrieve("销售额货币", scopes=["dataset:sales"])

    def test_a_new_value_replaces_the_old_one(self, memory) -> None:
        first = memory.add(
            kind="preference",
            key="currency",
            value="销售额用人民币。",
            status="confirmed",
        )
        memory.add(
            kind="preference",
            key="currency",
            value="销售额用美元。",
            status="confirmed",
        )

        hits = memory.retrieve("销售额货币", scopes=["global"])
        assert len(hits) == 1
        assert hits[0]["memory_id"] == first["memory_id"]
        assert "美元" in hits[0]["value"]
        assert "人民币" not in hits[0]["value"]


class TestUpdateAndDelete:
    def test_update_changes_the_value(self, memory) -> None:
        record = memory.add(
            kind="preference", key="currency", value="人民币。", status="confirmed"
        )
        updated = memory.update(record["memory_id"], value="日元。")

        assert updated["value"] == "日元。"
        assert "日元" in memory.retrieve("currency 日元", scopes=["global"])[0]["value"]

    def test_delete_stops_recall(self, memory) -> None:
        record = memory.add(
            kind="preference", key="currency", value="人民币。", status="confirmed"
        )
        assert memory.delete(record["memory_id"]) is True  # noqa: E712

        assert memory.list() == []
        assert memory.retrieve("人民币", scopes=["global"]) == []
        assert memory.get(record["memory_id"]) is None

    def test_deleting_twice_is_reported_honestly(self, memory) -> None:
        record = memory.add(
            kind="preference", key="k", value="临时偏好。", status="confirmed"
        )
        assert memory.delete(record["memory_id"]) is True  # noqa: E712
        assert memory.delete(record["memory_id"]) is False


class TestValidation:
    @pytest.mark.parametrize(
        "value",
        [
            "password: hunter2",
            "API_KEY=sk-abcdefghijklmnopqrstuvwx",
            "postgresql://user:pass@host/db",
            "secret = topsecret",
        ],
    )
    def test_credentials_are_rejected(self, memory, value: str) -> None:
        with pytest.raises(MemoryValidationError):
            memory.add(kind="preference", key="leak", value=value, status="confirmed")

    def test_overlong_values_are_rejected(self, memory) -> None:
        with pytest.raises(MemoryValidationError, match="超过"):
            memory.add(
                kind="preference", key="dump", value="x" * 900, status="confirmed"
            )

    def test_unknown_kind_is_rejected(self, memory) -> None:
        with pytest.raises(MemoryValidationError, match="不支持"):
            memory.add(kind="random", key="k", value="v", status="confirmed")

    def test_candidate_extraction_only_triggers_on_explicit_wording(self, memory) -> None:
        assert memory.propose_candidates("今天销售额是多少") == []
        candidates = memory.propose_candidates("今后销售额统一用人民币元表示。")
        assert candidates and candidates[0]["status"] == "candidate"


class TestIsolationAndPersistence:
    def test_memories_do_not_leak_across_owners(self, memory) -> None:
        memory.add(
            kind="preference", key="currency", value="人民币。", status="confirmed"
        )

        other = MemoryService(owner="someone-else")
        assert other.list() == []
        assert other.retrieve("货币", scopes=["global"]) == []

    def test_memories_survive_a_restart(self, memory) -> None:
        record = memory.add(
            kind="preference",
            key="currency",
            value="人民币。",
            status="confirmed",
        )

        storage.dispose_all()

        reopened = MemoryService(owner="pg-test")
        assert reopened.get(record["memory_id"])["value"] == "人民币。"
        assert reopened.retrieve("人民币", scopes=["global"])

    def test_flushing_redis_does_not_touch_postgres(self, memory) -> None:
        memory.add(
            kind="preference", key="currency", value="人民币。", status="confirmed"
        )

        if redis_available():
            from rag.cache import KnowledgeCache

            cache = KnowledgeCache(redis_url(), enabled=True)
            cache.delete_prefix("insightflow:")

        assert memory.retrieve("人民币", scopes=["global"])
