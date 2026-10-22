"""Audit regressions: real local persistence and deterministic embedding boundaries."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import json

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

import storage
from storage.config import StorageConfig
from services.memory_service import MemoryService, dataset_scope
from services.conversation_service import ConversationService
from storage.conversations import ConversationRepository


@pytest.fixture
def local_store(tmp_path):
    storage.reset()
    storage.configure(StorageConfig(mode="local", owner="audit", local_db_path=str(tmp_path / "conversations.sqlite")))
    token = storage.set_current_owner("audit")
    storage.database_url()
    yield tmp_path
    storage.reset_current_owner(token)
    storage.reset()


class FakeTokenizer:
    def num_special_tokens_to_add(self, pair=False):
        return 2

    def encode(self, text, **kwargs):
        return list(range(len(text)))


class FakeEmbeddings(Embeddings):
    def __init__(self):
        self.queries = 0

    def embed_query(self, text):
        self.queries += 1
        return [1.0, 0.0, 0.0]

    def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


def test_unconfirmed_proposal_does_not_replace_confirmed_memory(local_store):
    service = MemoryService()
    first = service.add(**service.propose_candidates("今后金额统一用人民币")[0])
    service.confirm(first["memory_id"])
    second = service.add(**service.propose_candidates("以后报告优先用简短结论")[0])
    assert first["memory_id"] != second["memory_id"]
    assert service.get(first["memory_id"])["status"] == "confirmed"
    assert service.retrieve("金额人民币")[0]["memory_id"] == first["memory_id"]


def test_same_key_pending_revision_applies_only_on_confirmation(local_store):
    service = MemoryService()
    first = service.add(kind="preference", key="currency", value="金额人民币", status="confirmed")
    pending = service.add(kind="preference", key="currency", value="金额美元")
    assert first["memory_id"] != pending["memory_id"]
    assert service.get(first["memory_id"])["value"] == "金额人民币"
    assert service.confirm(pending["memory_id"])["memory_id"] == first["memory_id"]
    assert service.get(first["memory_id"])["value"] == "金额美元"


def test_scopes_are_path_specific_and_current_session_is_recalled(local_store):
    a = str(local_store / "eu" / "orders.csv")
    b = str(local_store / "us" / "orders.csv")
    service = MemoryService()
    assert service.scopes_for([a]) != service.scopes_for([b])
    service.add(kind="preference", key="currency", value="销售金额美元", scope=dataset_scope(a), status="confirmed")
    assert not service.retrieve("销售金额", scopes=service.scopes_for([b]))
    service.add(kind="report_format", key="style", value="简短结论", scope="session:s1", status="confirmed")
    assert ConversationService._memories_for("分析", dataset_refs=[a], session_id="s1")
    assert not ConversationService._memories_for("分析", dataset_refs=[b], session_id="s2")


def test_same_resource_alias_is_rejected_before_silent_overwrite(local_store):
    from ingestion.loader import load_data_sources
    from ingestion.resources import build_snapshots
    sources = []
    for folder in ("eu", "us"):
        path = local_store / folder / "orders.csv"
        path.parent.mkdir()
        path.write_text("amount\n1\n", encoding="utf-8")
        sources.append(str(path))
    for loader in (load_data_sources, build_snapshots):
        with pytest.raises(ValueError, match="重复"):
            loader(sources)


def test_recent_history_and_owner_are_enforced(local_store):
    service = ConversationService(None)
    info = service.create_session(dataset_refs=["orders.csv"])
    with storage.session_scope(storage.database_url()) as session:
        repo = ConversationRepository(session, owner="audit")
        for seq in range(1, 13):
            turn = repo.create_turn(session_id=info["session_id"], run_id=f"run{seq}")
            repo.add_message(session_id=info["session_id"], role="user", content=f"question{seq}", turn_id=turn.id)
            repo.add_message(session_id=info["session_id"], role="assistant", content=f"answer{seq}", turn_id=turn.id)
        turn_id = turn.id
    with storage.session_scope(storage.database_url()) as session:
        repo = ConversationRepository(session, owner="audit")
        history = service._history_snapshot(repo, info["session_id"])
        assert history["messages"][-1]["content"] == "answer12"
        assert history["turns"][-1]["seq"] == 12
        assert ConversationRepository(session, owner="other").get_turn(turn_id) is None


def test_history_syncs_completed_result_without_prior_poll(local_store, monkeypatch):
    service = ConversationService(SimpleNamespace(store=SimpleNamespace(get=lambda run_id: SimpleNamespace(status="completed"))))
    info = service.create_session(dataset_refs=["orders.csv"])
    with storage.session_scope(storage.database_url()) as session:
        repo = ConversationRepository(session, owner="audit")
        repo.create_turn(session_id=info["session_id"], run_id="run1")
    monkeypatch.setattr(service, "_completed_result", lambda turn: {"preview": "fresh result"})
    with storage.session_scope(storage.database_url()) as session:
        history = service._history_snapshot(ConversationRepository(session, owner="audit"), info["session_id"])
        assert history["turns"][0]["summary"] == "fresh result"
        assert history["messages"][0]["content"] == "fresh result"


def test_dispatch_failure_can_be_retried_without_duplicate_turn(local_store):
    calls = []
    def dispatch(run_id, **payload):
        calls.append((run_id, payload))
        if len(calls) == 1:
            raise RuntimeError("submission interrupted")
        return SimpleNamespace(status="pending")
    service = ConversationService(SimpleNamespace(dispatch=dispatch))
    info = service.create_session(dataset_refs=["orders.csv"])
    with pytest.raises(RuntimeError):
        service.post_message(info["session_id"], "分析金额", client_message_id="m1")
    retried = service.post_message(info["session_id"], "分析金额", client_message_id="m1")
    assert retried["deduplicated"]
    assert calls[0] == calls[1]
    assert len(service.list_turns(info["session_id"])) == 1


def test_parallel_messages_have_distinct_sequence_and_idempotency(local_store):
    service = ConversationService(SimpleNamespace(dispatch=lambda run_id, **kwargs: SimpleNamespace(status="running")))
    info = service.create_session(dataset_refs=["orders.csv"])
    def submit(index):
        return service.post_message(info["session_id"], "分析金额", client_message_id=f"m{index}")
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(submit, [1, 2, 3, 4, 1, 2]))
    turns = service.list_turns(info["session_id"])
    assert len(turns) == 4
    assert [turn["seq"] for turn in turns] == [1, 2, 3, 4]
    assert len({item["run_id"] for item in results}) == 4


def test_version_change_invalidates_all_tasks(monkeypatch):
    from services import knowledge_service as service
    from rag.retrieval import RetrievalOutcome
    from rag.evidence import RetrievalEvidence
    old = {"chunk_id": "old", "content": "old", "knowledge_version": "v1"}
    state = {"knowledge_version": "v1", "task_knowledge": {"a": [old], "b": [old]}}
    monkeypatch.setattr(service, "_current_knowledge_version", lambda state: "v2")
    calls = []
    def retrieve(state, task, **kwargs):
        calls.append(task["task_id"])
        evidence = RetrievalEvidence(task["task_id"], "doc", "s", "new", 1., "v2", "m", "ns")
        return RetrievalOutcome(evidence=[evidence], knowledge_version="v2")
    monkeypatch.setattr(service, "retrieve_task", retrieve)
    _, updates = service.task_knowledge_updates(state, {"task_id": "a"})
    state.update(updates)
    assert "b" not in state["task_knowledge"]
    evidence, _ = service.task_knowledge_updates(state, {"task_id": "b"})
    assert calls == ["a", "b"] and evidence[0]["knowledge_version"] == "v2"


def test_knowledge_change_after_execution_requires_new_run(monkeypatch):
    from services import knowledge_service as service
    state = {"knowledge_version": "v1", "execution_results": [{"task_id": "a", "success": True}],
        "task_knowledge": {"a": [{"content": "old", "knowledge_version": "v1"}]}}
    monkeypatch.setattr(service, "_current_knowledge_version", lambda state: "v2")
    monkeypatch.setattr(service, "retrieve_task", lambda *a, **k: pytest.fail("不得自动混合重算"))
    with pytest.raises(service.KnowledgeSourceChanged, match="知识源已变化.*新建Run"):
        service.task_knowledge_updates(state, {"task_id": "b"})
    with pytest.raises(service.KnowledgeSourceChanged, match="新建Run"):
        service.knowledge_contents(state, "a")


def test_output_nodes_receive_current_instruction_and_format_memories():
    from harness.context_sources import inject_context_sources
    state = {"analysis_focus": "本次用英文", "memories": [{"kind": "report_format", "status": "confirmed", "value": "用中文简短结论"}]}
    for node in ("reporter", "visualizer", "plotter"):
        fields = {"memories": "无", "current_instruction": "无"}
        inject_context_sources(node, fields, state)
        assert fields["current_instruction"] == "本次用英文"
        assert "简短结论" in fields["memories"]


def test_reporter_actual_input_preserves_requested_answer_format(monkeypatch):
    from agents import reporter
    from harness.runtime import bind_state
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    instruction = "报告最后附 business_answers JSON，使用英文"
    observed = []
    def invoke(chain, fields, *, validate, **kwargs):
        observed.append(dict(fields))
        return validate(chain.invoke(fields))
    monkeypatch.setattr(reporter, "invoke_report_with_retry", invoke)
    state = {"analysis_focus": instruction, "memories": [{"kind": "report_format", "status": "confirmed", "value": "先给简短结论"}]}
    with bind_state(state):
        assert reporter.generate_analysis_report([], [], [], [], [], llm=FakeListChatModel(responses=["# Report"])) == "# Report"
    assert observed[0]["current_instruction"] == instruction
    assert "简短结论" in observed[0]["memories"]


def test_chunk_tail_is_not_silently_dropped():
    from rag.chunker import split_documents, MODEL_MAX_TOKENS
    text = "开头普通规则。" + "业务说明文字。" * 120 + "尾部退款上限九万元。"
    chunks = split_documents([Document(page_content=text, metadata={"source": "r.md"})], tokenizer=FakeTokenizer())
    assert len(chunks) > 1
    assert all(len(item.page_content) + 2 <= MODEL_MAX_TOKENS for item in chunks)
    assert any("尾部退款上限九万元" in item.page_content for item in chunks)


def test_cached_real_tokenizer_bounds_all_chinese_chunks(monkeypatch):
    from rag.chunker import split_documents, get_tokenizer, MODEL_MAX_TOKENS
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    get_tokenizer.cache_clear()
    try:
        tokenizer = get_tokenizer()
    except OSError:
        pytest.skip("本机未缓存固定revision的tokenizer；不下载")
    text = "普通说明文字。" * 120 + "尾部规则退款上限九万元。"
    chunks = split_documents([Document(page_content=text)], tokenizer=tokenizer)
    assert len(chunks) > 1
    assert all(len(tokenizer.encode(chunk.page_content, add_special_tokens=True)) <= MODEL_MAX_TOKENS for chunk in chunks)
    assert any("尾部规则退款上限九万元" in chunk.page_content for chunk in chunks)


def test_local_search_embeds_query_once_and_metadata_is_bounded(local_store, monkeypatch):
    from rag import retrieval, chunker, vector_store
    from rag.cache import KnowledgeCache
    monkeypatch.setattr(chunker, "get_tokenizer", lambda: FakeTokenizer())
    retrieval.reset_retriever()
    model = FakeEmbeddings()
    retriever = retrieval.KnowledgeRetriever(owner="audit", cache=KnowledgeCache(None, enabled=False), persistent=False, embedding_model=model)
    for index in range(11):
        path = local_store / f"r{index}.md"
        path.write_text(f"金额规则{index}", encoding="utf-8")
        outcome = retriever.retrieve(knowledge_sources=[str(path)], query="金额", min_score=0.)
        assert outcome.evidence and outcome.evidence[0].chunk_id
    assert model.queries == 11
    assert len(vector_store._VECTOR_STORE_CACHE) <= 8
    assert not hasattr(retrieval, "_LOCAL_CHUNKS")


def test_retriever_captures_request_owner_and_namespace_uses_source_identity(local_store):
    from rag.retrieval import get_retriever, reset_retriever
    from rag.versioning import knowledge_namespace
    reset_retriever()
    first = get_retriever()
    token = storage.set_current_owner("other")
    try:
        second = get_retriever()
    finally:
        storage.reset_current_owner(token)
    assert first._owner == "audit" and second._owner == "other"
    assert knowledge_namespace("audit", [str(local_store / "a" / "rules.md")]) != knowledge_namespace("audit", [str(local_store / "b" / "rules.md")])


def test_persistent_index_validity_includes_config_version(local_store, monkeypatch):
    from rag import retrieval, chunker
    from rag.cache import KnowledgeCache
    monkeypatch.setattr(chunker, "get_tokenizer", lambda: FakeTokenizer())
    model = FakeEmbeddings()
    retriever = retrieval.KnowledgeRetriever(owner="audit", cache=KnowledgeCache(None, enabled=False), persistent=True, embedding_model=model)
    source = local_store / "rules.md"
    source.write_text("完整指标口径", encoding="utf-8")
    records = {}
    writes = []
    class Index:
        def has_document(self, **kwargs):
            return records.get(kwargs["source"]) == (kwargs["knowledge_version"], kwargs["embedding_model_version"])
        def put_document(self, **kwargs):
            records[kwargs["source"]] = (kwargs["knowledge_version"], kwargs["embedding_model_version"])
            writes.append(kwargs)
    index = Index()
    assert retriever._index_into(index, namespace="ns", version="v1", sources=[str(source)]) == 1
    assert retriever._index_into(index, namespace="ns", version="v1", sources=[str(source)]) == 0
    assert retriever._index_into(index, namespace="ns", version="v2", sources=[str(source)]) == 1
    assert len(writes) == 2
