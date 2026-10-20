"""真实模型回归：固定问题、固定数据，把每次的关键指标记成历史。

    python scripts/real_model_regression.py                 # 只跑快用例
    python scripts/real_model_regression.py --include-retail # 加上 1.7 万行零售案例

为什么需要它：mock 能证明"代码没坏"，但证明不了"模型还能把数据变成结论"。
模型行为会随提示词、版本、温度漂移，只有重复跑同一批问题才看得见。

每次运行追加一行 JSON 到历史文件（默认在仓库外的审计目录），并和该用例上一次的
结果对比，把"变差了什么"直接打出来——而不是让人对着两次截图肉眼比对。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.llm import build_model_factory  # noqa: E402
from services.analysis_service import run_analysis  # noqa: E402
from utils.run_logger import create_run_id  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HISTORY = Path(
    os.getenv(
        "INSIGHTFLOW_REGRESSION_LOG",
        r"G:\AI-Projects\_local_audit\insightflow-agent2\real-model-runs.jsonl",
    )
)

CASES = {
    "toy-regions": {
        "data_sources": ["data/customers.csv", "data/orders.csv"],
        "analysis_focus": "分析各地区高价值客户的销售贡献，并给出结论。",
        "dynamic": False,
        "knowledge_sources": ["knowledge/business_rules.md"],
    },
    "retail-gmv": {
        "data_sources": [
            "data/retail/orders_eu_2011.csv",
            "data/retail/products.csv",
        ],
        "analysis_focus": "最近两个完整月的 GMV 环比如何？退货金额最高的商品有哪些？",
        "dynamic": True,
        # 口径必须一起给：这份数据 12 月不完整、M/POST 不是商品。不给口径，
        # 模型会得出 -74% 和"M 是退货冠军"——两次真实运行都这么答过。
        "knowledge_sources": [
            "knowledge/business_rules.md",
            "knowledge/retail_data_notes.md",
        ],
    },
}


def _read_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def _previous(history: list[dict], case: str) -> dict | None:
    for entry in reversed(history):
        if entry.get("case") == case:
            return entry
    return None


def _delta(current: dict, previous: dict | None) -> list[str]:
    if not previous:
        return ["首次记录"]
    notes = []
    if current["charts_produced"] != previous.get("charts_produced"):
        notes.append(
            f"图表 {previous.get('charts_produced')} → {current['charts_produced']}"
        )
    if current["llm_calls"] != previous.get("llm_calls"):
        notes.append(f"模型请求 {previous.get('llm_calls')} → {current['llm_calls']}")
    if current["status"] != previous.get("status"):
        notes.append(f"状态 {previous.get('status')} → {current['status']}")
    if not notes:
        notes.append("与上次一致")
    return notes


def run_case(name: str, spec: dict, output_root: Path) -> dict:
    run_id = create_run_id()
    started = time.time()
    record = {
        "case": name,
        "run_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "failed",
        "charts_planned": 0,
        "charts_produced": 0,
        "llm_calls": 0,
        "tool_calls": 0,
        "seconds": 0.0,
        "error": None,
    }

    try:
        result = run_analysis(
            data_sources=list(spec["data_sources"]),
            analysis_focus=spec["analysis_focus"],
            knowledge_sources=list(
                spec.get("knowledge_sources") or ["knowledge/business_rules.md"]
            ),
            run_id=run_id,
            model_factory=build_model_factory(run_id),
            output_root=output_root,
            enable_dynamic_loop=spec.get("dynamic", False),
        )
    except Exception as error:  # noqa: BLE001 - 失败本身就是要记录的事实
        record["error"] = f"{type(error).__name__}: {error}"[:300]
        record["seconds"] = round(time.time() - started, 1)
        return record

    usage = (result.get("execution_budget") or {}).get("usage") or {}
    artifacts = result.get("artifacts") or []
    record.update(
        {
            "status": result.get("analysis_status"),
            "charts_planned": len(result.get("visualization_plan") or []),
            "charts_produced": len([p for p in artifacts if str(p).endswith(".png")]),
            "charts_failed": len(result.get("failed_charts") or []),
            "llm_calls": usage.get("llm_calls") or 0,
            "tool_calls": usage.get("tool_calls") or 0,
            "decisions": len(result.get("decision_history") or []),
            "seconds": round(time.time() - started, 1),
            "report_path": result.get("report_path"),
        }
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="真实模型回归")
    parser.add_argument("--include-retail", action="store_true", help="加上零售案例")
    parser.add_argument("--history", default=str(DEFAULT_HISTORY))
    parser.add_argument("--output-root", default="outputs/regression")
    args = parser.parse_args()

    history_path = Path(args.history)
    history = _read_history(history_path)
    output_root = Path(args.output_root)

    cases = ["toy-regions"]
    if args.include_retail:
        cases.append("retail-gmv")

    failures = 0
    lines = [f"{'用例':<14}{'状态':<10}{'图表':<9}{'请求':<6}{'工具':<6}{'耗时':<8}对比"]
    for name in cases:
        spec = CASES[name]
        print(f"[{name}] 运行中……", flush=True)
        record = run_case(name, spec, output_root)
        previous = _previous(history, name)
        notes = _delta(record, previous)

        charts = f"{record['charts_produced']}/{record['charts_planned']}"
        lines.append(
            f"{name:<14}{str(record['status']):<10}{charts:<9}"
            f"{record['llm_calls']:<6}{record['tool_calls']:<6}{record['seconds']:<8}"
            + "；".join(notes)
        )

        if record["status"] != "completed":
            failures += 1
        elif record["charts_planned"] and record["charts_produced"] < record["charts_planned"]:
            failures += 1
            lines.append(f"  ! 有 {record.get('charts_failed', 0)} 张图未生成")

        history.append(record)

    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as handle:
        for record in history[-len(cases):]:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print()
    print("=== 本次结果 ===")
    for line in lines:
        print(line)
    print()
    print(f"历史文件：{history_path}")

    if failures:
        print(f"\n结论：{failures} 个用例未达标（见上面的状态与图表列）", file=sys.stderr)
        return 1
    print("\n结论：全部达标")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
