"""Split with the embedding tokenizer so every stored chunk is actually embedded."""

from functools import lru_cache
import os

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

MODEL_MAX_TOKENS = 128
CHUNK_SIZE = 126  # reserve two special tokens for MiniLM
CHUNK_OVERLAP = 20
CHUNKER_VERSION = "minilm-token-v2"


@lru_cache(maxsize=1)
def get_tokenizer():
    from transformers import AutoTokenizer
    from rag.vector_store import EMBEDDING_MODEL, EMBEDDING_REVISION

    offline = os.getenv("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes"}
    return AutoTokenizer.from_pretrained(EMBEDDING_MODEL, revision=EMBEDDING_REVISION,
        local_files_only=offline, use_fast=True)


def split_documents(documents: list[Document], *, tokenizer=None) -> list[Document]:
    if not documents:
        return []
    tokenizer = tokenizer or get_tokenizer()
    specials = int(tokenizer.num_special_tokens_to_add(pair=False))
    budget = min(CHUNK_SIZE, MODEL_MAX_TOKENS - specials)

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False, truncation=False))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=budget,
        chunk_overlap=min(CHUNK_OVERLAP, budget - 1),
        length_function=count,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for chunk in chunks:
        if count(chunk.page_content) > budget:
            raise ValueError("知识片段超过 embedding token 上限，请检查分词器/切分配置")
        chunk.metadata["chunker_version"] = CHUNKER_VERSION
    return chunks
