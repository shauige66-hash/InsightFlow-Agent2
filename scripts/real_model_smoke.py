"""用真实模型跑一次完整分析，产出一份可核对的结果。

    python scripts/real_model_smoke.py
    python scripts/real_model_smoke.py --data data/retail/orders_eu_2011.csv

做这件事的原因：整套系统的工程部分可以用脚本化模型验证，但"模型能不能真的
把数据变成结论"只能由真实调用回答。这个脚本用真实模型跑一次小分析，然后打印
可核对的事实——状态、决策次数、模型请求数、报告路径、图表数量。

失败时按原因给出下一步，而不是只抛一个堆栈。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.llm import build_model_factory  # noqa: E402
from services.analysis_service import run_analysis  # noqa: E402
from utils.run_logger import create_run_id  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="真实模型的端到端冒烟")
    parser.add_argument(
        "--data",
        action="append",
        default=None,
        help="数据源（可多次）；默认用 data/customers.csv 与 data/orders.csv",
    )
    parser.add_argument(
        "--focus",
        default="分析各地区高价值客户的销售贡献，并给出结论。",
        help="分析问题",
    )
    parser.add_argument("--output-root", default="outputs/real-smoke")
    parser.add_argument("--dynamic", action="store_true", help="开启动态决策")
    args = parser.parse_args()

    data_sources = args.data or ["data/customers.csv", "data/orders.csv"]
    missing = [path for path in data_sources if not Path(path).exists()]
    if missing:
        print(f"数据源不存在：{missing}", file=sys.stderr)
        return 2

    print("数据源：", "、".join(data_sources))
    print("分析问题：", args.focus)
    print("输出目录：", args.output_root)
    started = time.time()
    run_id = create_run_id()

    try:
        result = run_analysis(
            data_sources=list(data_sources),
            analysis_focus=args.focus,
            knowledge_sources=["knowledge/business_rules.md"],
            run_id=run_id,
            model_factory=build_model_factory(run_id),
            output_root=Path(args.output_root),
            enable_dynamic_loop=args.dynamic,
        )
    except ValueError as error:
        # 最常见的一种：key 缺失或无效
        message = str(error)
        print(f"\n失败：{message}", file=sys.stderr)
        if "API_KEY" in message.upper():
            print("下一步：把有效 key 写进 .env，然后 python scripts/check_model.py", file=sys.stderr)
        return 1
    except Exception as error:  # noqa: BLE001
        name = type(error).__name__
        message = str(error)
        print(f"\n失败：{name}: {message}", file=sys.stderr)

        if "401" in message or "Authentication" in name or "API_KEY" in message.upper():
            print(
                "\n这是凭据问题，不是代码问题：\n"
                "  1. 到申请方控制台确认 key 仍然有效（被删/过期/额度用尽都会失败）\n"
                "  2. 写进项目根目录的 .env：DEEPSEEK_API_KEY=...\n"
                "  3. 先跑 python scripts/check_model.py 确认可用\n"
                "  4. 再跑本脚本",
                file=sys.stderr,
            )
            return 3
        return 1

    elapsed = time.time() - started
    usage = (result.get("execution_budget") or {}).get("usage") or {}
    decisions = result.get("decision_history") or []
    artifacts = result.get("artifacts") or []
    report_path = result.get("report_path")

    print("\n=== 结果 ===")
    print("run_status      :", result.get("run_status"))
    print("analysis_status :", result.get("analysis_status"))
    print("status_reason   :", result.get("status_reason"))
    print("决策次数        :", len(decisions))
    print("模型请求        :", usage.get("llm_calls"))
    print("工具调用        :", usage.get("tool_calls"))
    print("执行时间        :", f"{round(usage.get('active_seconds') or 0)} 秒（墙钟 {elapsed:.1f} 秒）")
    print("报告            :", report_path)

    charts = [path for path in artifacts if str(path).endswith(".png")]
    print("图表数量        :", len(charts))
    for chart in charts[:5]:
        print("   ", chart)

    if decisions:
        print("\n=== 决策历史 ===")
        for item in decisions:
            print(f"  step {item.get('step')}: {item.get('action')} — {item.get('reason')}")

    failed = result.get("failed_tasks") or []
    if failed:
        print("\n=== 失败任务 ===")
        for item in failed:
            print(f"  {item.get('task_id')}: {item.get('stage')} — {item.get('reason')}")

    ok = result.get("analysis_status") == "completed" and bool(report_path)
    print("\n结论：", "通过——真实模型产出了报告" if ok else "未通过——看上面的状态与原因")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
