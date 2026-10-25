"""Explicit local controller: baseline, bounded optimization, status, export, rollback."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from improvement.cycle import CallAllowance, CampaignLimits, ModelProposer, run_campaign
from improvement.gates import development_feedback, summary
from improvement.registry import VersionRegistry


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description="InsightFlow 受控提示/经验改进，独立评分与版本回退")
    parser.add_argument("command", choices=("baseline", "improve", "status", "export", "rollback"))
    parser.add_argument("--workspace", type=Path, default=ROOT / "outputs.improvement_private")
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--max-llm-calls", type=positive_int, help="每个任务的模型请求上限（真实运行必填）")
    parser.add_argument("--max-seconds", type=positive_int, help="每个任务的执行秒数上限（真实运行必填）")
    parser.add_argument("--max-total-llm-calls", type=positive_int, help="整轮最坏请求额度，包含反思与验收（必填）")
    parser.add_argument("--rounds", type=positive_int, default=2)
    parser.add_argument("--backend", choices=("reflective", "gepa"), default="reflective")
    parser.add_argument("--max-metric-calls", type=positive_int, default=12)
    parser.add_argument("--target", help="回退目标profile_id")
    parser.add_argument("--output", type=Path, help="导出当前配置的目标JSON文件")
    args = parser.parse_args(argv)
    if args.command in {"baseline", "improve"}:
        missing = [flag for flag, value in (("--max-llm-calls", args.max_llm_calls),
                   ("--max-seconds", args.max_seconds), ("--max-total-llm-calls", args.max_total_llm_calls)) if value is None]
        if missing:
            parser.error("真实运行必须显式指定 " + ", ".join(missing))
        if not os.getenv("INSIGHTFLOW_EVAL_REAL_API_KEY"):
            parser.error("真实运行需要显式 INSIGHTFLOW_EVAL_REAL_API_KEY；不会自动使用应用密钥")
    workspace = args.workspace.resolve()
    registry = VersionRegistry(workspace / "registry")
    current = registry.initialize()
    if args.command == "status":
        print(json.dumps(registry.status(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "export":
        if args.output is None:
            parser.error("export requires --output")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(current.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"当前版本已导出：{args.output}")
        return 0
    if args.command == "rollback":
        if not args.target:
            parser.error("rollback requires --target profile_id")
        profile = registry.rollback(args.target)
        print(f"已回退：{profile.profile_id}")
        return 0

    from core.config import Settings, get_settings
    from core.llm import build_model_factory
    from eval.real_runner import RealEvaluator
    from utils.agent_loop import AgentBudget

    defaults = get_settings()
    settings = Settings(api_key=os.environ["INSIGHTFLOW_EVAL_REAL_API_KEY"],
                        base_url=os.getenv("INSIGHTFLOW_EVAL_REAL_BASE_URL", defaults.base_url),
                        model=os.getenv("INSIGHTFLOW_EVAL_REAL_MODEL", defaults.model),
                        client_app="insightflow-improvement", timeout_seconds=min(args.max_seconds, 60))
    if not settings.model:
        parser.error("设置 INSIGHTFLOW_EVAL_REAL_MODEL（或 DEEPSEEK_MODEL）")
    budget = AgentBudget(max_llm_calls=args.max_llm_calls, max_seconds=args.max_seconds)
    builder = lambda run_id: build_model_factory(run_id, settings=settings)
    evaluator = RealEvaluator(workspace / "trials", model_factory_builder=builder,
                              repeats=args.repeats, budget=budget, model=settings.model,
                              base_url=settings.base_url, holdout_seed=secrets.randbits(48))
    if args.command == "baseline":
        allowance = CallAllowance(args.max_total_llm_calls)
        reports = {}
        for split in ("development", "validation"):
            allowance.reserve(evaluator.expected_trials(split) * args.max_llm_calls)
            report = evaluator.evaluate(current, split)
            reports[split] = report
            print(f"{split}: " + json.dumps(summary(report), ensure_ascii=False))
        destination = workspace / "baselines" / (uuid4().hex + ".json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"基线已保存：{destination}")
        return 0

    model = builder("improve-" + uuid4().hex).for_agent("improver").bind(max_tokens=2000)
    if args.backend == "reflective":
        result = run_campaign(registry, evaluator, ModelProposer(model),
                              limits=CampaignLimits(args.rounds, args.max_total_llm_calls))
    else:
        from improvement.gepa_adapter import optimize_profile

        def optimizer(seed, evaluate, reserve):
            def evaluate_batch(profile, batch, capture_traces):
                outputs, scores, trajectories = [], [], []
                for item in batch:
                    split = item["split"]
                    report = evaluate(profile, split)
                    metrics = summary(report)
                    diagnostics = development_feedback(report) if split == "development" else []
                    output = {"passed": metrics["passed"], "total": metrics["total"],
                              "diagnostics": diagnostics}
                    outputs.append(output)
                    scores.append(metrics["correctness_rate"])
                    trajectories.append(output)
                return {"outputs": outputs, "scores": scores,
                        "trajectories": trajectories if capture_traces else None}

            def reflection(prompts):
                responses = []
                for prompt in prompts:
                    reserve(1)
                    response = model.invoke([("human", prompt)])
                    if not isinstance(response.content, str):
                        raise ValueError("Reflection must produce text")
                    responses.append(response.content)
                return responses

            return optimize_profile(seed, trainset=[{"split": "development"}],
                                    valset=[{"split": "validation"}], evaluate_batch=evaluate_batch,
                                    reflection_model=reflection,
                                    max_metric_calls=args.max_metric_calls).profile

        result = run_campaign(registry, evaluator, optimizer=optimizer,
                              limits=CampaignLimits(args.rounds, args.max_total_llm_calls))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        # Provider errors can contain credentials/URLs; persist only their type.
        print(f"改进未完成：{type(error).__name__}。当前版本不会因此自动升级。", file=sys.stderr)
        if type(error).__name__ == "MissingGEPAError":
            print("请安装 requirements-improvement.txt 中的可选 GEPA 依赖。", file=sys.stderr)
        elif type(error).__name__ == "CampaignBudgetExceeded":
            print("下一批任务的最坏调用额度不足；检查 --max-total-llm-calls 和每任务预算。", file=sys.stderr)
        raise SystemExit(2)
