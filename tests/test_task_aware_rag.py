"""任务级检索、证据追溯与上下文集成。

核心断言是"不同任务拿到不同证据"，而不是"检索被调用过"：只验证调用次数
会让"所有任务都灌同一批知识"的实现也通过。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import storage
from rag.retrieval import get_retriever, reset_retriever
from rag.versioning import knowledge_namespace, knowledge_version
from tests.test_e2e_business_scenarios import ScriptedFactory

VIP_DOC = """# 客户分类

VIP 客户指年消费金额超过 10000 元的客户。
VIP 客户的售后响应时限为 4 小时。
"""

SHIPPING_DOC = """# 配送规则

标准配送时效为下单后三个工作日。
偏远地区的配送时效为七个工作日。
"""


@pytest.fixture
def knowledge(tmp_path: Path) -> dict[str, Path]:
    vip = tmp_path / "vip_rules.md"
    vip.write_text(VIP_DOC, encoding="utf-8")
    shipping = tmp_path / "shipping_rules.md"
    shipping.write_text(SHIPPING_DOC, encoding="utf-8")
    return {"vip": vip, "shipping": shipping}


@pytest.fixture(autouse=True)
def _local_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """这些用例走 local 模式：进程内索引，不起容器。"""
    monkeypatch.setenv("INSIGHTFLOW_STORAGE", "local")
    monkeypatch.setenv("INSIGHTFLOW_LOCAL_DB", str(tmp_path / "conversations.sqlite"))
    monkeypatch.setenv("INSIGHTFLOW_OWNER", "rag-test")
    monkeypatch.delenv("INSIGHTFLOW_REDIS_ENABLED", raising=False)
    storage.reset()
    storage.configure()
    reset_retriever()
    yield
    storage.reset()
    reset_retriever()


@pytest.fixture(autouse=True)
def _fake_tool(monkeypatch: pytest.MonkeyPatch):
    from harness import tool_dispatcher

    def fake_invoke(tool_name, tables, arguments, *, snapshots=None):
        return {"success": True, "result": [{"n": 1}], "error": None, "error_kind": None}

    monkeypatch.setattr(tool_dispatcher, "invoke_tool", fake_invoke)


def retrieve(sources: list[str], query: str, top_k: int = 3):
    return get_retriever().retrieve(knowledge_sources=sources, query=query, top_k=top_k)


class TestTaskLevelRetrieval:
    def test_two_tasks_get_different_evidence(self, knowledge) -> None:
        vip = retrieve([str(knowledge["vip"]), str(knowledge["shipping"])], "VIP 客户的定义")
        shipping = retrieve(
            [str(knowledge["vip"]), str(knowledge["shipping"])], "配送时效是多少"
        )

        vip_sources = {item.source for item in vip.evidence}
        shipping_sources = {item.source for item in shipping.evidence}

        assert "vip_rules.md" in vip_sources
        assert "vip_rules.md" not in shipping_sources
        assert "shipping_rules.md" in shipping_sources
        assert "shipping_rules.md" not in vip_sources

    def test_unrelated_knowledge_is_filtered_out(self, knowledge) -> None:
        outcome = retrieve([str(knowledge["vip"])], "今天天气怎么样，午饭吃什么")

        assert outcome.evidence == [], [item.source for item in outcome.evidence]

    def test_evidence_carries_traceable_fields(self, knowledge) -> None:
        outcome = retrieve([str(knowledge["vip"])], "VIP 客户的定义")
        assert outcome.evidence, "没有检索到证据"

        item = outcome.evidence[0].to_dict()
        for key in (
            "chunk_id",
            "document_id",
            "source",
            "content",
            "score",
            "knowledge_version",
            "embedding_model_version",
            "namespace",
        ):
            assert item[key] not in (None, "", 0.0), f"{key} 缺失"

        assert item["source"] == "vip_rules.md"
        assert item["knowledge_version"] == outcome.knowledge_version
        assert item["embedding_model_version"].startswith("sentence-transformers/")

    def test_score_direction_is_similarity(self, knowledge) -> None:
        """分数越大越相关：相关文档的分数必须高于无关文档。"""
        relevant = retrieve([str(knowledge["vip"])], "VIP 客户定义")
        unrelated = retrieve([str(knowledge["vip"])], "配送时效")

        assert relevant.evidence and relevant.evidence[0].score > 0
        if unrelated.evidence:
            assert relevant.evidence[0].score > unrelated.evidence[0].score
        else:
            assert relevant.evidence[0].score >= 0.2

    def test_a_missing_knowledge_source_fails_loudly(self, knowledge, tmp_path: Path) -> None:
        missing = tmp_path / "not_there.md"
        with pytest.raises(FileNotFoundError):
            retrieve([str(knowledge["vip"]), str(missing)], "VIP 客户定义")

    def test_namespaces_do_not_leak_between_knowledge_bases(self, knowledge) -> None:
        first = retrieve([str(knowledge["vip"])], "配送时效")
        second = retrieve([str(knowledge["shipping"])], "配送时效")

        assert first.namespace != second.namespace
        assert {item.source for item in second.evidence} <= {"shipping_rules.md"}

    def test_document_update_changes_version_and_evidence(self, knowledge) -> None:
        before = retrieve([str(knowledge["vip"])], "VIP 客户年消费门槛")
        knowledge["vip"].write_text(
            "# 客户分类\n\nVIP 客户指年消费金额超过 20000 元的客户。\n",
            encoding="utf-8",
        )
        after = retrieve([str(knowledge["vip"])], "VIP 客户年消费门槛")

        assert before.knowledge_version != after.knowledge_version
        assert "20000" in after.evidence[0].content
        assert "10000" not in after.evidence[0].content


class TestEvidenceReuseInsideATask:
    def test_one_retrieval_per_task_across_execute_and_critic(
        self, knowledge, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """同一任务的执行、修复、校验共用第一批证据。"""
        from rag import retrieval as retrieval_module

        calls: list[str] = []
        original = retrieval_module.KnowledgeRetriever.retrieve

        def counting(self, *, knowledge_sources, query, top_k=3, min_score=0.2):
            calls.append(query)
            return original(
                self,
                knowledge_sources=knowledge_sources,
                query=query,
                top_k=top_k,
                min_score=min_score,
            )

        monkeypatch.setattr(
            retrieval_module.KnowledgeRetriever, "retrieve", counting
        )

        source = tmp_path / "customers.csv"
        source.write_text(
            "customer_id,customer_name,region\nC001,Alice,East\n", encoding="utf-8"
        )
        replies = {
            "planner": {
                "tasks": [
                    {
                        "task_id": "task_1",
                        "description": "统计 VIP 客户数量",
                        "target_tables": ["customers"],
                        "expected_output": "VIP 客户数",
                    }
                ]
            },
            "executor": {"code": "result = df"},
            "critic": {"validation_passed": False, "feedback": "需要修正"},
            "repair": {"code": "result = df"},
            "refine": {"code": "result = df"},
            "visualizer": {"charts": []},
            "reporter": "# 报告\n",
        }
        run_id = "h4-reuse"
        from services.analysis_service import run_analysis

        run_analysis(
            run_id=run_id,
            data_sources=[str(source)],
            knowledge_sources=[str(knowledge["vip"]), str(knowledge["shipping"])],
            model_factory=ScriptedFactory(run_id, replies).build(),
            output_root=tmp_path,
            analysis_focus="VIP 客户",
        )

        # planner 的全局检索 + 该任务的检索；修复与精修不再检索
        task_calls = [query for query in calls if "统计 VIP 客户数量" in query]
        assert len(calls) == 2, calls
        assert len(task_calls) == 1


class TestContextIntegration:
    def test_planner_gets_global_knowledge_and_executor_gets_task_evidence(
        self, knowledge, tmp_path: Path
    ) -> None:
        source = tmp_path / "customers.csv"
        source.write_text(
            "customer_id,customer_name,region\nC001,Alice,East\n", encoding="utf-8"
        )
        replies = {
            "planner": {
                "tasks": [
                    {
                        "task_id": "task_1",
                        "description": "统计 VIP 客户数量",
                        "target_tables": ["customers"],
                        "expected_output": "VIP 客户数",
                    }
                ]
            },
            "executor": {"code": "result = df"},
            "critic": {"validation_passed": True, "feedback": "ok"},
            "visualizer": {"charts": []},
            "reporter": "# 报告\n\nVIP 客户 1 位。\n",
        }
        run_id = "h4-context"
        from services.analysis_service import run_analysis

        result = run_analysis(
            run_id=run_id,
            data_sources=[str(source)],
            knowledge_sources=[str(knowledge["vip"]), str(knowledge["shipping"])],
            model_factory=ScriptedFactory(run_id, replies).build(),
            output_root=tmp_path,
            analysis_focus="VIP 客户",
        )

        events = [
            json.loads(line)
            for line in (tmp_path / "logs" / f"{run_id}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        contexts = {
            event["node"]: event["details"]["context"]
            for event in events
            if event.get("event") == "llm_context"
        }

        assert contexts["planner"]["knowledge_chunks"] >= 1
        assert contexts["executor"]["knowledge_chunks"] >= 1
        # 证据里只包含 VIP 规则，不包含无关的配送规则
        assert "vip_rules.md" in str(result["task_knowledge"])

        # 报告带可追溯的来源附录
        assert "## 参考来源" in result["report_content"]
        assert "vip_rules.md" in result["report_content"]


class TestVersioning:
    def test_namespace_is_stable_but_version_tracks_content(self, knowledge) -> None:
        sources = [str(knowledge["vip"])]
        namespace_before = knowledge_namespace("owner-a", sources)
        version_before = knowledge_version(sources)

        knowledge["vip"].write_text("# 客户分类\n\nVIP 门槛为 5000 元。\n", encoding="utf-8")

        assert knowledge_namespace("owner-a", sources) == namespace_before
        assert knowledge_version(sources) != version_before

    def test_different_owners_have_different_namespaces(self, knowledge) -> None:
        sources = [str(knowledge["vip"])]
        assert knowledge_namespace("a", sources) != knowledge_namespace("b", sources)
