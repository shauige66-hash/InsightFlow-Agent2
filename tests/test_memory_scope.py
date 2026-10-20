"""写入前的作用域分类：不该把"这次"当成永久偏好。"""

from __future__ import annotations

import pytest

from services.memory_service import MemoryService


class TestClassifyScope:
    @pytest.mark.parametrize(
        "text, expected_prefix",
        [
            # 点名了数据集
            ("今后 orders_eu_2011 的退货率按金额算", "dataset:orders_eu_2011"),
            ("以后 products 的均价用中位数", "dataset:products"),
            # 指向本次：有 session_id 就落会话
            ("这次报告用红色标题", "session:sess_1"),
            ("本次分析只看德国站", "session:sess_1"),
            # 稳定偏好
            ("今后金额统一用人民币表示", "global"),
            ("以后不要输出 markdown 表格", "global"),
        ],
    )
    def test_prefix(self, text, expected_prefix) -> None:
        scope, reason = MemoryService.classify_scope(
            text,
            dataset_refs=["data/retail/orders_eu_2011.csv", "data/retail/products.csv"],
            session_id="sess_1",
        )
        assert scope.startswith(expected_prefix), (scope, reason)
        assert reason  # 必须给出可读的判定依据

    def test_session_wording_without_a_session_falls_back_to_the_single_dataset(self) -> None:
        scope, reason = MemoryService.classify_scope(
            "这次按金额算", dataset_refs=["data/orders.csv"], session_id=None
        )
        assert scope == "dataset:orders"
        assert "只有一个数据集" in reason

    def test_session_wording_with_several_datasets_and_no_session_stays_global(self) -> None:
        scope, reason = MemoryService.classify_scope(
            "这次按金额算",
            dataset_refs=["data/a.csv", "data/b.csv"],
            session_id=None,
        )
        assert scope == "global"
        assert "缺少会话" in reason

    def test_dataset_wording_without_a_name_uses_the_only_dataset(self) -> None:
        scope, reason = MemoryService.classify_scope(
            "这张表的金额单位是分", dataset_refs=["data/orders.csv"], session_id="s"
        )
        assert scope == "dataset:orders"

    def test_ambiguous_wording_stays_global(self) -> None:
        scope, _ = MemoryService.classify_scope("今后优先用简洁结论", dataset_refs=[])
        assert scope == "global"


class TestProposeCandidates:
    def test_the_scope_is_decided_before_writing(self) -> None:
        service = MemoryService()

        # 同时含"长期措辞"（今后/统一）与"指向本次"的措辞：应当落会话作用域
        candidates = service.propose_candidates(
            "今后这个报告统一用红色标题",
            dataset_refs=["data/orders.csv"],
            session_id="sess_9",
            provenance={"turn_id": "turn_1"},
        )

        assert len(candidates) == 1
        assert candidates[0]["scope"] == "session:sess_9"
        assert candidates[0]["status"] == "candidate"
        assert candidates[0]["provenance"]["scope_reason"] == "措辞指向本次分析"
        assert candidates[0]["provenance"]["turn_id"] == "turn_1"

    def test_a_one_off_instruction_is_not_turned_into_memory(self) -> None:
        """只说"这次"而不含长期措辞：不产生候选——少存一条比错存一条好。"""
        service = MemoryService()

        assert service.propose_candidates(
            "这次报告用红色标题", dataset_refs=["data/orders.csv"], session_id="sess_9"
        ) == []

    def test_a_stable_preference_still_lands_in_global(self) -> None:
        service = MemoryService()

        candidates = service.propose_candidates(
            "今后金额统一用人民币", dataset_refs=["data/orders.csv"], session_id="sess_9"
        )

        assert candidates[0]["scope"] == "global"

    def test_a_message_without_preference_wording_produces_nothing(self) -> None:
        service = MemoryService()
        assert service.propose_candidates("今天销售额是多少", session_id="s") == []
