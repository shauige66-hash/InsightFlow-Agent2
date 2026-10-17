"""H4 检索评估：固定离线查询集上的 Recall@k 与来源正确性。

只评估检索，不评估模型回答质量：这里的"正确答案"是"应当命中哪个文档"，
由人工标注，公平且可复现。

同时对比两种用法：
- **全局一次检索**：规划前用分析重点查一次，所有任务共用；
- **任务级检索**：每个任务用自己的问句查一次。

不预设改进幅度，只记录实测值。
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import pytest

import storage
from rag.retrieval import get_retriever, reset_retriever

DOCS = {
    "vip.md": (
        "# VIP 客户\n\n"
        "VIP 客户指年消费金额超过 10000 元的客户。\n"
        "VIP 客户的售后响应时限为 4 小时。\n"
    ),
    "shipping.md": (
        "# 配送规则\n\n标准配送时效为下单后三个工作日。\n偏远地区为七个工作日。\n"
    ),
    "refund.md": (
        "# 退款规则\n\n退款在签收后 7 天内可申请。\n退款金额以实际支付金额为准。\n"
    ),
    "metrics.md": (
        "# 指标定义\n\n"
        "销售额 = 订单金额之和，不含运费与退款。\n"
        "客单价 = 销售额 / 订单数。\n"
    ),
    "unrelated.md": "# 办公规范\n\n会议室需要提前一天预约。\n",
}

#: (查询, 期望命中的文档)
QUERIES: list[tuple[str, str]] = [
    ("VIP 客户的判定标准是什么", "vip.md"),
    ("高价值客户的一年消费门槛", "vip.md"),
    ("配送需要几天", "shipping.md"),
    ("偏远地区物流时效", "shipping.md"),
    ("多久可以退款", "refund.md"),
    ("销售额怎么计算", "metrics.md"),
    ("客单价的口径", "metrics.md"),
    ("今天的天气怎么样", None),
    ("会议室怎么预约", "unrelated.md"),
]


@pytest.fixture(autouse=True)
def _local_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("INSIGHTFLOW_STORAGE", "local")
    monkeypatch.setenv("INSIGHTFLOW_LOCAL_DB", str(tmp_path / "local.sqlite"))
    monkeypatch.setenv("INSIGHTFLOW_OWNER", "eval")
    monkeypatch.delenv("INSIGHTFLOW_REDIS_ENABLED", raising=False)
    storage.reset()
    storage.configure()
    reset_retriever()
    yield
    storage.reset()
    reset_retriever()


@pytest.fixture
def corpus(tmp_path: Path) -> list[str]:
    paths = []
    for name, content in DOCS.items():
        path = tmp_path / name
        path.write_text(content, encoding="utf-8")
        paths.append(str(path))
    return sorted(paths)


def run_queries(sources: list[str], top_k: int = 3) -> dict[str, object]:
    retriever = get_retriever()
    hits_at_1 = 0
    hits_at_k = 0
    latencies: list[float] = []
    context_chars: list[int] = []

    for query, expected in QUERIES:
        started = time.perf_counter()
        outcome = retriever.retrieve(knowledge_sources=sources, query=query, top_k=top_k)
        latencies.append((time.perf_counter() - started) * 1000)

        sources_found = [item.source for item in outcome.evidence]
        context_chars.append(sum(len(item.content) for item in outcome.evidence))

        if expected is None:
            hits_at_1 += int(not sources_found)
            hits_at_k += int(not sources_found)
        else:
            hits_at_1 += int(bool(sources_found) and sources_found[0] == expected)
            hits_at_k += int(expected in sources_found)

    total = len(QUERIES)
    return {
        "total": total,
        "recall_at_1": round(hits_at_1 / total, 3),
        "recall_at_k": round(hits_at_k / total, 3),
        "latency_p50_ms": round(statistics.median(latencies), 2),
        "latency_max_ms": round(max(latencies), 2),
        "context_chars_avg": round(statistics.mean(context_chars), 1),
    }


class TestRetrievalQuality:
    def test_task_level_retrieval_metrics(self, corpus) -> None:
        metrics = run_queries(corpus)

        # 阈值是"这套固定数据上必须达到的下限"，不是行业基准
        assert metrics["recall_at_k"] >= 0.8, metrics
        assert metrics["recall_at_1"] >= 0.6, metrics

    def test_unrelated_queries_return_nothing(self, corpus) -> None:
        outcome = get_retriever().retrieve(
            knowledge_sources=corpus, query="今天的天气怎么样"
        )
        assert outcome.evidence == []

    def test_global_versus_task_level_context_size(self, corpus) -> None:
        """同一个知识库：一次性全局检索 vs 每个任务各查一次。"""
        retriever = get_retriever()
        tasks = [
            "统计 VIP 客户数量",
            "计算销售额",
            "统计退款订单",
        ]

        global_outcome = retriever.retrieve(
            knowledge_sources=corpus, query="数据分析相关的业务规则与指标定义", top_k=5
        )
        global_chars = sum(len(item.content) for item in global_outcome.evidence)

        per_task_chars = []
        for task in tasks:
            outcome = retriever.retrieve(knowledge_sources=corpus, query=task)
            per_task_chars.append(sum(len(item.content) for item in outcome.evidence))

        # 记录实测值：不假设谁更省，只断言两者都是有限且可度量的
        assert global_chars > 0
        assert all(value > 0 for value in per_task_chars)
        assert max(per_task_chars) <= global_chars * 3

    def test_source_correctness_is_traceable(self, corpus) -> None:
        outcome = get_retriever().retrieve(
            knowledge_sources=corpus, query="销售额的定义"
        )
        assert outcome.evidence
        for item in outcome.evidence:
            assert item.source in DOCS
            assert item.chunk_id
            assert item.document_id
            assert 0.0 < item.score <= 1.0

    def test_metrics_are_written_when_requested(self, corpus, tmp_path: Path) -> None:
        target = os.getenv("INSIGHTFLOW_EVAL_OUT")
        if not target:
            pytest.skip("未设置 INSIGHTFLOW_EVAL_OUT")

        metrics = run_queries(corpus)
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "corpus_documents": len(DOCS),
                    "queries": len(QUERIES),
                    "top_k": 3,
                    "task_level": metrics,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        assert path.exists()
