"""知识库身份、版本与查询指纹。

三个概念分开，否则缓存会串：

- **namespace**：请求 owner、知识源规范路径和索引配置的身份，内容更新不改变它。
- **knowledge_version**：内容版本。文件内容、切分配置或 embedding 模型任一变化
  都会改变它，缓存键因此失效。
- **query fingerprint**：检索问句的归一化哈希。

只把问句文本当缓存键会让不同知识库、不同版本、不同权限的查询互相命中。
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

from rag.chunker import CHUNK_OVERLAP, CHUNK_SIZE, CHUNKER_VERSION, MODEL_MAX_TOKENS
from rag.vector_store import EMBEDDING_MODEL, EMBEDDING_REVISION

#: embedding 模型名、固定revision与维度共同标识索引口径。
EMBEDDING_MODEL_VERSION = f"{EMBEDDING_MODEL}@{EMBEDDING_REVISION}:384"

_WHITESPACE = re.compile(r"\s+")


class KnowledgeSourceChanged(ValueError):
    """A run cannot combine results produced under different business rules."""
    error_kind = "knowledge_changed"


def file_content_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def knowledge_namespace(owner: str, knowledge_sources: list[str]) -> str:
    """Namespace is owner + canonical source identities, never display names."""
    names = "|".join([
        *sorted(source_identity(source) for source in knowledge_sources),
        f"index={EMBEDDING_MODEL_VERSION}/{CHUNKER_VERSION}/{CHUNK_SIZE}/{CHUNK_OVERLAP}/{MODEL_MAX_TOKENS}",
    ])
    digest = hashlib.sha256(names.encode("utf-8")).hexdigest()[:16]
    return f"kb:{owner}:{digest}"


def source_identity(source: str | Path) -> str:
    return os.path.normcase(str(Path(source).expanduser().resolve()))


def knowledge_version(knowledge_sources: list[str]) -> str:
    """内容版本：文件内容、切分配置与 embedding 模型都参与计算。"""
    entries: list[str] = []
    for source in knowledge_sources:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Knowledge source not found: {path}")
        entries.append(f"{source_identity(path)}:{file_content_hash(path)}")
    payload = "|".join(
        [
            *sorted(entries),
            f"chunk={CHUNK_SIZE}/{CHUNK_OVERLAP}",
            f"chunker={CHUNKER_VERSION}/{MODEL_MAX_TOKENS}",
            f"model={EMBEDDING_MODEL_VERSION}",
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def normalize_query(query: str) -> str:
    """归一化问句：大小写与空白差异不该产生新的缓存条目。"""
    return _WHITESPACE.sub(" ", (query or "").strip()).lower()


def query_fingerprint(query: str) -> str:
    return hashlib.sha256(normalize_query(query).encode("utf-8")).hexdigest()[:16]


__all__ = [
    "EMBEDDING_MODEL_VERSION",
    "file_content_hash",
    "knowledge_namespace",
    "knowledge_version",
    "normalize_query",
    "query_fingerprint",
    "source_identity",
]
