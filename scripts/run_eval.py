"""本地评测入口。

用法：
    python scripts/run_eval.py --output eval/reports
    python scripts/run_eval.py --family sqlite_join --output eval/reports
    python scripts/run_eval.py --limit 5 --output eval/reports
    python scripts/run_eval.py --faults-only --output eval/reports
    python scripts/run_eval.py --provider real --output eval/reports   # 需显式授权

不指定 --output 时报告写在运行隔离目录里，不在仓库中留下产物。

## 关于 `--provider real`

真实模型运行需要专用评测 API Key、模型名和显式请求数/时间上限。没有这些时，
本脚本不会去调用付费模型，而是打印需要提供什么。这样做的原因很简单：
编造成功率、成本或 Token 数据比不测更糟。

输出（`--output` 目录下）：
    eval.json      机器可读结果（含口径声明）
    eval.csv       逐任务汇总
    eval.md        Markdown 报告
    eval.png       分族完成率/正确率对比图
    faults.json    故障注入场景的观测与判定
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval import faults as faults_module  # noqa: E402
from eval.runner import (  # noqa: E402
    MOCK_SCOPE_NOTE,
    UNVERIFIED_REAL_MODEL_METRICS,
    run_all,
    write_csv,
    write_json,
    write_markdown,
    write_plot,
)
from eval.tasks import build_tasks  # noqa: E402


def _select_tasks(args: argparse.Namespace):
    tasks = build_tasks()
    if args.family:
        wanted = set(args.family)
        tasks = [task for task in tasks if task.family in wanted]
    if args.limit is not None:
        tasks = tasks[: args.limit]
    return tasks


def _run_real(args: argparse.Namespace) -> int:
    """A separate real-model suite; mock task scripts never become references."""
    from eval.real_runner import EvaluationContractChanged, RealEvaluator
    from improvement.profiles import CandidateProfile
    from utils.agent_loop import AgentBudget
    from core.runtime import McpSettings
    import tempfile

    if not os.getenv("INSIGHTFLOW_EVAL_REAL_API_KEY") or not os.getenv("INSIGHTFLOW_EVAL_REAL_MODEL"):
        print("真实评测需要 INSIGHTFLOW_EVAL_REAL_API_KEY 和 INSIGHTFLOW_EVAL_REAL_MODEL。", file=sys.stderr)
        return 2
    if args.max_llm_calls is None or args.max_seconds is None:
        print("真实评测必须显式指定 --max-llm-calls 和 --max-seconds（每次任务上限）。", file=sys.stderr)
        return 2
    if args.max_llm_calls < 1 or args.max_seconds <= 0 or not math.isfinite(args.max_seconds):
        print("真实评测的请求数和时间上限必须为有限正数。", file=sys.stderr)
        return 2
    if args.family or args.limit is not None or args.faults_only:
        print("真实评测使用独立业务套件；--family/--limit/--faults-only 仅适用于 mock。", file=sys.stderr)
        return 2
    try:
        profile = CandidateProfile()
        if args.profile:
            profile = CandidateProfile.from_dict(json.loads(Path(args.profile).read_text(encoding="utf-8")))
        workdir = Path(args.workdir).resolve() if args.workdir else Path(tempfile.mkdtemp(prefix="insightflow-real-eval-"))
        output = Path(args.output).resolve() if args.output else workdir / "reports"
        evaluator = RealEvaluator(workdir / "runs", repeats=args.repeats,
                                  budget=AgentBudget(max_llm_calls=args.max_llm_calls, max_seconds=args.max_seconds),
                                  mcp=McpSettings(mode=args.mode))
        print(f"真实模型业务评测 | split={args.split} | trials={evaluator.expected_trials(args.split)}")
        print(f"每次任务最多 {args.max_llm_calls} 个模型请求、{args.max_seconds:g} 秒；Token/费用未知时标为 null。")
        report = evaluator.evaluate(profile, args.split)
        output.mkdir(parents=True, exist_ok=True)
        path = output / "real-eval.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report["summary"], ensure_ascii=False))
        print(f"评测报告：{path}")
        return 0 if report["summary"]["passed"] == report["summary"]["total"] else 1
    except (ValueError, OSError, EvaluationContractChanged) as error:
        print(f"真实评测配置或固定合同无效：{type(error).__name__}", file=sys.stderr)
        return 2


def main() -> int:
    parser = argparse.ArgumentParser(description="InsightFlow 评测与可靠性基准")
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "报告输出目录。不指定时与评测工作目录同为系统临时目录下的"
            "隔离目录，避免在仓库里留下机器相关的产物"
        ),
    )
    parser.add_argument(
        "--workdir",
        default=None,
        help="评测工作目录（默认在系统临时目录下建一个隔离目录）",
    )
    parser.add_argument(
        "--family",
        action="append",
        default=None,
        help="只跑指定任务族（可重复）",
    )
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 个任务")
    parser.add_argument(
        "--mode",
        choices=("direct", "mcp"),
        default="mcp",
        help="SQL 后端（任务集里 SQLite 任务建议用 mcp）",
    )
    parser.add_argument(
        "--faults-only", action="store_true", help="只跑故障注入场景"
    )
    parser.add_argument(
        "--skip-faults", action="store_true", help="跳过故障注入场景"
    )
    parser.add_argument(
        "--provider",
        choices=("mock", "real"),
        default="mock",
        help="模型提供方；real 需要显式授权的 API Key 与预算",
    )
    parser.add_argument("--repeats", type=int, default=3, help="真实评测每个任务重复次数")
    parser.add_argument("--split", choices=("development", "validation", "holdout"), default="development")
    parser.add_argument("--max-llm-calls", type=int, default=None, help="真实评测每次任务的模型请求上限")
    parser.add_argument("--max-seconds", type=float, default=None, help="真实评测每次任务时间上限")
    parser.add_argument("--profile", default=None, help="待评测的纯文本候选配置 JSON")
    args = parser.parse_args()

    if args.provider == "real":
        return _run_real(args)

    if args.workdir:
        workdir = Path(args.workdir).resolve()
    else:
        import tempfile

        workdir = Path(tempfile.mkdtemp(prefix="insightflow-eval-"))

    # 默认把报告写进同一次运行的隔离目录：报告里含宿主绝对路径与
    # 机器相关的耗时，留在仓库里既没意义也会泄漏本机路径。
    if args.output:
        output = Path(args.output).resolve()
    else:
        output = workdir / "reports"
    output.mkdir(parents=True, exist_ok=True)

    print(MOCK_SCOPE_NOTE)
    print()

    written: list[Path] = []

    if not args.faults_only:
        tasks = _select_tasks(args)
        print(f"任务数：{len(tasks)} | SQL 后端：{args.mode}")

        def progress(outcome) -> None:  # noqa: ANN001
            mark = "OK " if outcome.correct else "-- "
            print(
                f"  {mark}{outcome.task_id:<34} {outcome.status:<10} "
                f"{outcome.duration_seconds:5.1f}s"
            )

        run = run_all(
            workdir=workdir, mode=args.mode, tasks=tasks, on_outcome=progress
        )

        written.append(write_json(run, output / "eval.json"))
        written.append(write_csv(run, output / "eval.csv"))
        written.append(write_markdown(run, output / "eval.md"))
        plot = write_plot(run, output / "eval.png")
        if plot is not None:
            written.append(plot)

        print()
        print(
            f"完成率 {run.summary.completion_rate:.2%} | "
            f"正确率 {run.summary.correctness_rate:.2%} | "
            f"条件正确率 {run.summary.conditional_accuracy:.2%}"
        )

    if not args.skip_faults:
        print()
        print("故障注入场景：")
        fault_result = faults_module.run_all(
            workdir=workdir / "faults", mode=args.mode
        )
        for verdict in fault_result["verdicts"]:
            mark = "OK " if verdict["passed"] else "!! "
            print(
                f"  {mark}{verdict['scenario_id']:<22} "
                f"{'; '.join(verdict['problems']) or '符合期望处置'}"
            )

        fault_path = output / "faults.json"
        fault_path.write_text(
            json.dumps(fault_result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        written.append(fault_path)

        print()
        print(f"故障场景通过：{fault_result['passed']} / {fault_result['total']}")

    print()
    print(f"工作目录：{workdir}")
    print("已写出：")
    for path in written:
        print(f"  {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
