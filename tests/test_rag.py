"""RAG 业务知识检索。

分两类：
- 不依赖 embedding 模型的部分（分块、缓存键、文档加载、阈值过滤）用假向量库验证，
  因此始终可运行；
- 端到端检索需要一个真实的本地 embedding 模型，模型不在本地缓存时跳过，
  不让测试去下载或访问外部服务。
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

from rag.cache_key import build_knowledge_cache_key
from rag.chunker import split_documents
from rag.config import DEFAULT_TOP_K, MIN_SIMILARITY_SCORE
from rag.document_loader import load_knowledge_document
from rag.retriever import retrieve_knowledge

from tests.conftest import BUSINESS_RULES_MD, KNOWLEDGE_DIR


class TestDocumentLoader:
    def test_markdown_becomes_single_document(self) -> None:
        documents = load_knowledge_document(str(BUSINESS_RULES_MD))

        assert len(documents) == 1
        assert documents[0].metadata["source"] == BUSINESS_RULES_MD.name
        assert "High-Value Customer" in documents[0].page_content

    def test_txt_is_supported(self, tmp_path) -> None:
        source = tmp_path / "rules.txt"
        source.write_text("规则内容", encoding="utf-8")

        documents = load_knowledge_document(str(source))

        assert documents[0].page_content == "规则内容"

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="Knowledge source not found"):
            load_knowledge_document("knowledge/missing.md")

    def test_unsupported_suffix_raises(self, tmp_path) -> None:
        source = tmp_path / "rules.docx"
        source.write_text("x", encoding="utf-8")

        with pytest.raises(ValueError, match="Unsupported knowledge source"):
            load_knowledge_document(str(source))


class TestChunker:
    def test_short_document_stays_one_chunk(self) -> None:
        documents = load_knowledge_document(str(BUSINESS_RULES_MD))

        chunks = split_documents(documents)

        assert len(chunks) >= 1
        assert all(
            chunk.metadata.get("source") == BUSINESS_RULES_MD.name for chunk in chunks
        )

    def test_long_document_is_split(self) -> None:
        long_text = "业务规则段落。\n" * 400
        documents = [Document(page_content=long_text, metadata={"source": "big.md"})]

        chunks = split_documents(documents)

        assert len(chunks) > 1
        # chunk_size=800，留出少量余量
        assert all(len(chunk.page_content) <= 900 for chunk in chunks)

    def test_metadata_is_preserved(self) -> None:
        documents = [
            Document(page_content="x" * 2000, metadata={"source": "a.md", "page": 1})
        ]

        chunks = split_documents(documents)

        assert all(chunk.metadata["source"] == "a.md" for chunk in chunks)


class TestCacheKey:
    def test_same_content_same_key(self) -> None:
        first = build_knowledge_cache_key([str(BUSINESS_RULES_MD)])
        second = build_knowledge_cache_key([str(BUSINESS_RULES_MD)])

        assert first == second

    def test_order_does_not_matter(self, tmp_path) -> None:
        other = tmp_path / "extra.md"
        other.write_text("额外规则", encoding="utf-8")

        forward = build_knowledge_cache_key([str(BUSINESS_RULES_MD), str(other)])
        backward = build_knowledge_cache_key([str(other), str(BUSINESS_RULES_MD)])

        assert forward == backward

    def test_content_change_changes_key(self, tmp_path) -> None:
        source = tmp_path / "rules.md"
        source.write_text("版本一", encoding="utf-8")
        before = build_knowledge_cache_key([str(source)])

        source.write_text("版本二", encoding="utf-8")
        after = build_knowledge_cache_key([str(source)])

        assert before != after

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            build_knowledge_cache_key(["knowledge/missing.md"])


class StubEmbeddings(Embeddings):
    """按字符集合生成确定性向量，避免测试依赖真实模型或网络。"""

    def _vector(self, text: str) -> list[float]:
        lowered = text.lower()
        keys = ("high", "value", "region", "sales", "customer", "订单")
        return [1.0 if key in lowered else 0.0 for key in keys]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class TestRetrieverThreshold:
    @pytest.fixture
    def store(self) -> InMemoryVectorStore:
        documents = [
            Document(
                page_content="High-Value Customer is a customer whose total exceeds $400",
                metadata={"source": "business_rules.md"},
            ),
            Document(
                page_content="Unrelated rule about office supplies ordering",
                metadata={"source": "unrelated_rules.md"},
            ),
        ]
        store = InMemoryVectorStore(embedding=StubEmbeddings())
        store.add_documents(documents)
        return store

    def test_relevant_document_is_returned(self, store) -> None:
        results = retrieve_knowledge(store, "high value customer", min_score=0.0)

        assert results
        assert "High-Value Customer" in results[0]["content"]
        assert results[0]["source"] == "business_rules.md"

    def test_score_and_metadata_are_reported(self, store) -> None:
        results = retrieve_knowledge(store, "high value", min_score=0.0)

        assert set(results[0]) == {"content", "source", "page", "score"}
        assert isinstance(results[0]["score"], float)

    def test_top_k_limits_results(self, store) -> None:
        results = retrieve_knowledge(store, "customer", top_k=1, min_score=0.0)

        assert len(results) == 1

    def test_high_threshold_filters_everything_out(self, store) -> None:
        """阈值过滤是真实的检索质量门，不是装饰。"""
        results = retrieve_knowledge(store, "high value customer", min_score=10.0)

        assert results == []

    def test_default_threshold_is_reasonable(self) -> None:
        assert 0.0 < MIN_SIMILARITY_SCORE < 1.0
        assert DEFAULT_TOP_K >= 1


class TestEndToEndRetrieval:
    @pytest.mark.slow
    def test_business_rule_is_retrieved_for_high_value_query(self) -> None:
        """端到端检索：用真实 embedding 验证业务规则能被召回。

        模型需已在本地缓存；否则跳过，避免测试触发下载。
        """
        from rag.knowledge_service import build_and_retrieve_knowledge

        try:
            results = build_and_retrieve_knowledge(
                knowledge_sources=[str(BUSINESS_RULES_MD)],
                query="哪些客户属于高价值客户",
            )
        except Exception as error:  # noqa: BLE001 - 模型不可用时跳过而非失败
            pytest.skip(f"本地 embedding 模型不可用: {type(error).__name__}")

        assert results, "高价值客户规则应被检索到"
        joined = " ".join(item["content"] for item in results)
        assert "$400" in joined or "400" in joined

    @pytest.mark.slow
    def test_empty_knowledge_sources_returns_empty(self) -> None:
        from rag.knowledge_service import build_and_retrieve_knowledge

        assert build_and_retrieve_knowledge([], query="任意问题") == []


def test_knowledge_files_are_present() -> None:
    assert BUSINESS_RULES_MD.exists()
    assert (KNOWLEDGE_DIR / "unrelated_rules.md").exists()
