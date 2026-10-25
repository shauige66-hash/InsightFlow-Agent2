"""Bounded application-level improvement; final evaluation is host-owned.

Search reads development feedback, chooses on validation, then consumes holdout
once for the locked baseline/candidate pair. No candidate Python is executed.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

from improvement.gates import PromotionPolicy, checked_rows, compare, development_feedback, summary
from improvement.profiles import CandidateProfile
from improvement.registry import VersionRegistry


@dataclass(frozen=True)
class CampaignLimits:
    max_rounds: int = 2
    max_total_llm_calls: int = 2000

    def __post_init__(self):
        for field in ("max_rounds", "max_total_llm_calls"):
            if type(getattr(self, field)) is not int or getattr(self, field) < 1:
                raise ValueError(f"{field} must be a positive integer")


class CampaignBudgetExceeded(RuntimeError):
    pass


class CallAllowance:
    """Conservatively reserve per-run maximums before any evaluation starts."""
    def __init__(self, maximum: int):
        self.maximum, self.reserved = maximum, 0

    def reserve(self, amount: int):
        if type(amount) is not int or amount < 0 or self.reserved + amount > self.maximum:
            raise CampaignBudgetExceeded("Campaign model-call allowance exhausted before dispatch")
        self.reserved += amount


class ModelProposer:
    """One request produces a strictly validated JSON profile, never executable code."""
    def __init__(self, model):
        self.model = model
        self.calls = 0

    def __call__(self, parent: CandidateProfile, feedback: list[dict]) -> CandidateProfile:
        prompt = (
            "你在改进数据分析Agent。只能输出一个完整候选配置JSON，字段只能是"
            "planner_guidance、critic_guidance、lessons。lessons每项只能有agent、text、source_trials。"
            "agent只能为planner或critic。source_trials必须引用下面的开发试验编号。"
            "只提出可泛化的分析策略，不抄具体答案，不降低审查标准，不修改评分、权限、预算或代码。"
            "基础系统规则和独立验收始终优先。经验是待验证假设，不能自报已验证。"
            "先用当前版本的规划、审查策略与经验诊断失败，再生成下一版。保持修改小而具体。"
        )
        # The selected version participates in the next diagnosis/proposal round.
        current = {"profile": parent.to_dict(), "development_feedback": feedback,
                   "diagnostic_guidance": parent.guidance_for("critic"),
                   "planning_guidance": parent.guidance_for("planner")}
        self.calls += 1
        response = self.model.invoke([("system", prompt), ("human", json.dumps(current, ensure_ascii=False))])
        content = response.content
        if not isinstance(content, str):
            raise ValueError("Improver must return a JSON text response")
        text = content.strip()
        if text.startswith("```json\n") and text.endswith("```"):
            text = text[8:-3].strip()
        if len(text) > 24000:
            raise ValueError("Candidate response exceeds its text allowance")
        return CandidateProfile.from_dict(json.loads(text))


def _check_experiences(candidate: CandidateProfile, parent: CandidateProfile, feedback: list[dict]):
    known = {row["trial_id"] for row in feedback}
    inherited = set(parent.lessons)
    for experience in candidate.lessons:
        if experience in inherited:
            continue
        if not experience.source_trials or not set(experience.source_trials) <= known:
            raise ValueError("New experience must cite observed development trials")


def run_campaign(registry: VersionRegistry, evaluator, proposer: Callable | None = None, *,
                 limits: CampaignLimits | None = None, policy: PromotionPolicy | None = None,
                 optimizer: Callable | None = None) -> dict:
    """optimizer(seed, evaluate_for_search, reserve_call) is an optional GEPA hook."""
    limits, policy = limits or CampaignLimits(), policy or PromotionPolicy()
    if (proposer is None) == (optimizer is None):
        raise ValueError("Choose exactly one proposal backend")
    base = registry.initialize()
    campaign_id = registry.start_campaign(policy)
    allowance = CallAllowance(limits.max_total_llm_calls)
    artifact_dir = registry.root / "campaigns" / campaign_id
    artifact_dir.mkdir(parents=True, exist_ok=False)
    search_log = []
    pinned = None

    def evaluate(profile, split):
        nonlocal pinned
        trials = evaluator.expected_trials(split)
        budget = evaluator.budget
        max_calls = budget.max_llm_calls if hasattr(budget, "max_llm_calls") else budget["max_llm_calls"]
        allowance.reserve(trials * max_calls)
        report = evaluator.evaluate(profile, split)
        checked_rows(report, profile_id=profile.profile_id, split=split)
        if len(report["rows"]) != trials:
            raise ValueError("Evaluator omitted scheduled trials")
        contract = report["scorer_fingerprint"], report["config_fingerprint"]
        if pinned is None:
            pinned = contract
        elif contract != pinned:
            raise ValueError("Evaluator/model/budget drifted during the campaign")
        return report

    def search_evaluate(profile, split):
        if split not in {"development", "validation"}:
            raise ValueError("The search backend cannot evaluate holdout")
        return evaluate(profile, split)

    try:
        base_dev = evaluate(base, "development")
        base_val = evaluate(base, "validation")
        registry.record(campaign_id, "baseline_development", base_dev)
        registry.record(campaign_id, "baseline_validation", base_val)
        best, best_dev, best_val = base, base_dev, base_val
        if optimizer is not None:
            candidate = optimizer(base, search_evaluate, allowance.reserve)
            candidate = CandidateProfile.from_dict(candidate.to_dict())
            _check_experiences(candidate, base, development_feedback(base_dev))
            candidate_dev = evaluate(candidate, "development")
            candidate_val = evaluate(candidate, "validation")
            decision = compare(base_val, candidate_val, policy=policy, require_improvement=True)
            search_log.append({"profile_id": candidate.profile_id, "decision": decision})
            registry.register(candidate)
            if decision["accepted"]:
                best, best_dev, best_val = candidate, candidate_dev, candidate_val
        else:
            for iteration in range(limits.max_rounds):
                feedback = development_feedback(best_dev)
                allowance.reserve(1)
                candidate = proposer(best, feedback)
                candidate = CandidateProfile.from_dict(candidate.to_dict())
                _check_experiences(candidate, best, feedback)
                if candidate.profile_id == best.profile_id:
                    search_log.append({"iteration": iteration, "reason": "Unchanged candidate"})
                    continue
                registry.register(candidate)
                candidate_dev = evaluate(candidate, "development")
                candidate_val = evaluate(candidate, "validation")
                registry.record(campaign_id, f"search_{iteration}_development", candidate_dev)
                registry.record(campaign_id, f"search_{iteration}_validation", candidate_val)
                decision = compare(best_val, candidate_val, policy=policy, require_improvement=True)
                search_log.append({"iteration": iteration, "parent_id": best.profile_id,
                                   "candidate_id": candidate.profile_id, "decision": decision})
                if decision["accepted"]:
                    best, best_dev, best_val = candidate, candidate_dev, candidate_val
        if best.profile_id == base.profile_id:
            registry.close_search(campaign_id, "No candidate improved independent validation")
            result = {"accepted": False, "campaign_id": campaign_id,
                      "reason": "No candidate improved independent validation", "base_id": base.profile_id}
        else:
            registry.record(campaign_id, "candidate_validation", best_val)
            # This transition is durable: a failed final test cannot restart search
            # in the same campaign or repeatedly inspect the held-out tasks.
            registry.lock_for_holdout(campaign_id, best)
            registry.record(campaign_id, "baseline_holdout", evaluate(base, "holdout"))
            registry.record(campaign_id, "candidate_holdout", evaluate(best, "holdout"))
            result = registry.finish(campaign_id)
        result.update({"reserved_model_call_allowance": allowance.reserved,
                       "search": search_log, "artifact_dir": str(artifact_dir)})
        (artifact_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result
    except Exception as error:
        registry.abort(campaign_id, type(error).__name__)
        (artifact_dir / "failure.json").write_text(json.dumps({"error_type": type(error).__name__,
            "reserved_model_call_allowance": allowance.reserved, "search": search_log},
            ensure_ascii=False, indent=2), encoding="utf-8")
        raise
