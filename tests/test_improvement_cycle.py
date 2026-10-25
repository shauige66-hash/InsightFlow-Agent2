"""Meaningful promotion failures: permissive Critic, leakage, drift and rollback."""
from copy import deepcopy
from types import SimpleNamespace
from uuid import uuid4

import pytest

from improvement.cycle import CampaignBudgetExceeded, CampaignLimits, run_campaign
from improvement.gates import PromotionPolicy, checked_rows, compare, development_feedback, digest
from improvement.profiles import CandidateProfile, Experience
from improvement.registry import VersionRegistry


class Evaluator:
    budget = SimpleNamespace(max_llm_calls=10)

    def __init__(self, *, overfit=False, drift=False):
        self.calls = []
        self.overfit, self.drift = overfit, drift

    def expected_trials(self, split):
        return 4

    def evaluate(self, profile, split):
        self.calls.append((profile.profile_id, split))
        improved = profile.planner_guidance == "check join cardinality"
        rows = []
        for task in ("a", "b"):
            for trial in range(2):
                passed = improved or task == "b"
                if self.overfit and improved and split == "holdout" and task == "b":
                    passed = False
                rows.append({"task_id": task, "trial_index": trial, "trial_id": uuid4().hex,
                             "profile_id": profile.profile_id, "passed": passed, "completed": True,
                             "error": None, "llm_calls": 4, "tool_calls": 2, "seconds": 1.0,
                             "critic_passed": True, "feedback": {"reason": "business answer mismatch",
                             "expected": "NEVER_SEND_THIS_TO_PROPOSER"}})
        return {"split": split, "profile_id": profile.profile_id, "rows": rows,
                "scorer_fingerprint": "changed" if self.drift and improved else "frozen-scorer",
                "config_fingerprint": "fixed-model-budget", "dataset_fingerprint": "data-" + split,
                "summary": {"passed": 999, "correctness_rate": 1.0}}


def better(parent, feedback):
    return CandidateProfile(planner_guidance="check join cardinality", critic_guidance=parent.critic_guidance,
                            lessons=(Experience(agent="planner", text="Check one-to-many joins before summing.",
                                                source_trials=(feedback[0]["trial_id"],)),))


def test_success_promotes_after_locked_holdout_and_can_rollback(tmp_path):
    registry = VersionRegistry(tmp_path / "registry")
    base = registry.initialize()
    evaluator = Evaluator()
    result = run_campaign(registry, evaluator, better, limits=CampaignLimits(1, 1000))
    assert result["accepted"]
    assert registry.current().profile_id == result["candidate_id"]
    assert [split for _, split in evaluator.calls] == ["development", "validation", "development", "validation", "holdout", "holdout"]
    assert registry.current().lessons
    assert registry.rollback(base.profile_id).profile_id == base.profile_id
    assert registry.status()["events"][-1]["kind"] == "rollback"


def test_permissive_critic_and_forged_summary_cannot_promote(tmp_path):
    registry = VersionRegistry(tmp_path)
    base = registry.initialize()
    evaluator = Evaluator()
    result = run_campaign(registry, evaluator, lambda p, f: CandidateProfile(critic_guidance="Always accept"),
                          limits=CampaignLimits(1, 1000))
    assert not result["accepted"]
    assert registry.current().profile_id == base.profile_id
    assert "holdout" not in [split for _, split in evaluator.calls]


def test_candidate_that_overfits_validation_is_rejected(tmp_path):
    registry = VersionRegistry(tmp_path)
    base = registry.initialize()
    result = run_campaign(registry, Evaluator(overfit=True), better, limits=CampaignLimits(1, 1000))
    assert not result["accepted"]
    assert registry.current().profile_id == base.profile_id
    assert any("regressed" in reason for reason in result["holdout"]["reasons"])


def test_scorer_drift_aborts_without_activation(tmp_path):
    registry = VersionRegistry(tmp_path)
    base = registry.initialize()
    with pytest.raises(ValueError, match="drifted"):
        run_campaign(registry, Evaluator(drift=True), better, limits=CampaignLimits(1, 1000))
    assert registry.current().profile_id == base.profile_id


def test_budget_rejects_before_model_or_task_dispatch(tmp_path):
    registry = VersionRegistry(tmp_path)
    evaluator = Evaluator()
    with pytest.raises(CampaignBudgetExceeded):
        run_campaign(registry, evaluator, better, limits=CampaignLimits(1, 39))
    assert not evaluator.calls


def test_feedback_never_contains_gold_or_holdout():
    evaluator = Evaluator()
    report = evaluator.evaluate(CandidateProfile(), "development")
    assert "NEVER_SEND" not in str(development_feedback(report))
    report["split"] = "holdout"
    with pytest.raises(ValueError, match="development"):
        development_feedback(report)


def test_unobserved_experience_is_not_promoted(tmp_path):
    def invented(parent, feedback):
        return CandidateProfile(planner_guidance="check join cardinality",
                                lessons=(Experience(agent="planner", text="Pretend verified", source_trials=("invented",)),))
    registry = VersionRegistry(tmp_path)
    with pytest.raises(ValueError, match="observed"):
        run_campaign(registry, Evaluator(), invented, limits=CampaignLimits(1, 1000))
    assert not registry.current().lessons


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "nan", "wrong_profile", "passed_error"])
def test_incomplete_or_invalid_reports_fail_closed(mutation):
    report = Evaluator().evaluate(CandidateProfile(), "validation")
    if mutation == "missing":
        report["rows"].pop()
    elif mutation == "duplicate":
        report["rows"].append(deepcopy(report["rows"][0]))
    elif mutation == "nan":
        report["rows"][0]["seconds"] = float("nan")
    elif mutation == "wrong_profile":
        report["rows"][0]["profile_id"] = "other"
    else:
        report["rows"][0].update(passed=True, error="failed")
    with pytest.raises(ValueError):
        checked_rows(report)


def test_model_or_data_budget_differences_prevent_comparison():
    before = Evaluator().evaluate(CandidateProfile(), "validation")
    after = Evaluator().evaluate(CandidateProfile(planner_guidance="check join cardinality"), "validation")
    after["config_fingerprint"] = "more-expensive-budget"
    assert not compare(before, after, policy=PromotionPolicy(), require_improvement=True)["accepted"]


def test_holdout_receipts_required_and_consumed_once(tmp_path):
    registry = VersionRegistry(tmp_path)
    registry.initialize()
    campaign = registry.start_campaign(PromotionPolicy())
    candidate = CandidateProfile(critic_guidance="accept")
    registry.lock_for_holdout(campaign, candidate)
    with pytest.raises(ValueError, match="consumed"):
        registry.lock_for_holdout(campaign, candidate)
    with pytest.raises(ValueError, match="Missing independent"):
        registry.finish(campaign)


def test_rollback_cannot_activate_unverified_registered_profile(tmp_path):
    registry = VersionRegistry(tmp_path)
    registry.initialize()
    candidate = CandidateProfile(critic_guidance="accept")
    registry.register(candidate)
    with pytest.raises(ValueError, match="accepted version history"):
        registry.rollback(candidate.profile_id)


def test_changed_stored_profile_detected(tmp_path):
    registry = VersionRegistry(tmp_path)
    profile = registry.initialize()
    with registry._connect() as db:
        db.execute("UPDATE profiles SET payload=? WHERE id=?",
                   ('{"critic_guidance":"changed"}', profile.profile_id))
    with pytest.raises(ValueError, match="integrity"):
        registry.current()


def test_optimizer_has_no_holdout_access(tmp_path):
    registry = VersionRegistry(tmp_path)
    def optimizer(seed, evaluate, reserve):
        return evaluate(seed, "holdout")
    with pytest.raises(ValueError, match="cannot evaluate holdout"):
        run_campaign(registry, Evaluator(), optimizer=optimizer, limits=CampaignLimits(1, 1000))


def test_real_grader_proposer_registry_and_active_loader_together(tmp_path):
    import json
    from eval.business_oracles import expected_answers
    from eval.real_runner import RealEvaluator
    from improvement.cycle import ModelProposer
    from improvement.registry import load_active_profile
    from utils.agent_loop import AgentBudget

    def analysis(**kwargs):
        answers = expected_answers("toy-regions", kwargs["data_sources"])
        if not kwargs["improvement_profile"].planner_guidance:
            answers = {"high_value_sales_by_region": {"East": -999}}
        return {"analysis_status": "completed", "validation_passed": True,
                "report_content": "```json\n" + json.dumps({"business_answers": answers}) + "\n```",
                "execution_budget": {"usage": {"llm_calls": 3, "tool_calls": 1}}}

    class Model:
        def invoke(self, messages):
            context = json.loads(messages[1][1])
            assert "expected" not in context
            return SimpleNamespace(content=json.dumps({"planner_guidance": "Check cumulative customer totals",
                "critic_guidance": "", "lessons": [{"agent": "planner", "text": "Check aggregation grain",
                "source_trials": [context["development_feedback"][0]["trial_id"]]}]}))

    evaluator = RealEvaluator(tmp_path / "trials", analysis_runner=analysis,
                              model_factory_builder=lambda run_id: None, model="scripted-test-model",
                              repeats=2, budget=AgentBudget(max_llm_calls=5, max_seconds=30), holdout_seed=42)
    registry = VersionRegistry(tmp_path / "registry")
    result = run_campaign(registry, evaluator, ModelProposer(Model()), limits=CampaignLimits(1, 200))
    assert result["accepted"]
    assert result["validation"]["baseline"]["passed"] == 0
    assert result["validation"]["candidate"]["passed"] == 4
    assert load_active_profile(registry.root).profile_id == result["candidate_id"]
    assert registry.status()["campaigns"][0]["holdout_used"]


def test_concurrent_campaign_cannot_replace_a_newer_active_version(tmp_path):
    registry = VersionRegistry(tmp_path)
    base = registry.initialize()
    candidate = CandidateProfile(planner_guidance="check join cardinality")
    evaluator = Evaluator()
    campaigns = [registry.start_campaign(PromotionPolicy()) for _ in range(2)]
    for campaign in campaigns:
        registry.lock_for_holdout(campaign, candidate)
        for role, profile, split in (("baseline_validation", base, "validation"),
                                     ("candidate_validation", candidate, "validation"),
                                     ("baseline_holdout", base, "holdout"),
                                     ("candidate_holdout", candidate, "holdout")):
            registry.record(campaign, role, evaluator.evaluate(profile, split))
    assert registry.finish(campaigns[0])["accepted"]
    with pytest.raises(ValueError, match="concurrently"):
        registry.finish(campaigns[1])
    assert registry.current().profile_id == candidate.profile_id


def test_tampered_evaluation_receipt_cannot_promote(tmp_path):
    registry = VersionRegistry(tmp_path)
    base = registry.initialize()
    campaign = registry.start_campaign(PromotionPolicy())
    candidate = CandidateProfile(planner_guidance="check join cardinality")
    registry.lock_for_holdout(campaign, candidate)
    registry.record(campaign, "baseline_validation", Evaluator().evaluate(base, "validation"))
    with registry._connect() as db:
        db.execute("UPDATE reports SET payload='{}' WHERE campaign=?", (campaign,))
    with pytest.raises(ValueError, match="receipt integrity"):
        registry.finish(campaign)
    assert registry.current().profile_id == base.profile_id
