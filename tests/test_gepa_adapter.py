"""GEPA's integration contract is tested without model requests or installing SDK."""
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from improvement.profiles import CandidateProfile
from improvement import gepa_adapter as integration


@dataclass
class Batch:
    outputs: list
    scores: list
    trajectories: list | None = None


def test_official_sdk_receives_adapter_and_budgeted_callable(monkeypatch):
    requests = []
    calls = []
    seed = CandidateProfile(critic_guidance="Check business definitions")

    def evaluate(profile, examples, capture):
        calls.append((profile, examples, capture))
        return {"outputs": [{"result": "actual", "expected": "GOLD"}],
                "scores": [0.5],
                "trajectories": [{"reason": "Wrong business rule", "api_key": "SECRET"}]}

    def fake_optimize(**kwargs):
        assert kwargs["candidate_selection_strategy"] == "pareto"
        assert kwargs["acceptance_criterion"] == "strict_improvement"
        assert kwargs["cache_evaluation"] is False
        assert kwargs["use_wandb"] is False
        assert kwargs["raise_on_exception"] is True
        assert kwargs["adapter"].propose_new_texts is None
        candidate = {**kwargs["seed_candidate"], "planner_guidance": "Check join cardinality"}
        batch = kwargs["adapter"].evaluate(kwargs["trainset"], candidate, True)
        feedback = kwargs["adapter"].make_reflective_dataset(candidate, batch, ["planner_guidance"])
        assert "GOLD" not in str(feedback)
        assert "SECRET" not in str(feedback)
        assert kwargs["reflection_lm"]("reflection prompt") == "candidate text"
        return SimpleNamespace(best_candidate=candidate, candidates=[kwargs["seed_candidate"], candidate])

    monkeypatch.setattr(integration, "_load_gepa", lambda: (fake_optimize, Batch))

    def reflection(prompts):
        requests.extend(prompts)
        return ["candidate text"]

    result = integration.optimize_profile(
        seed, trainset=[{"split": "development"}], valset=[{"split": "validation"}],
        evaluate_batch=evaluate, reflection_model=reflection, max_metric_calls=4,
    )
    assert requests == ["reflection prompt"]
    assert result.profile.planner_guidance == "Check join cardinality"
    assert result.profile.critic_guidance == seed.critic_guidance
    assert result.metric_calls == 1
    assert result.candidates_evaluated == 2
    assert calls[0][1] == [{"split": "development"}]


@pytest.mark.parametrize("example", [
    {"split": "holdout"}, {"split": "development", "expected": "answer"},
    {"split": "development", "api_key": "secret"},
])
def test_private_examples_never_reach_evaluator(example):
    called = []
    adapter = integration.ProfileAdapter(CandidateProfile(), lambda *args: called.append(args),
                                        batch_type=Batch, max_metric_calls=5)
    with pytest.raises(ValueError):
        adapter.evaluate([example], {"planner_guidance": "", "critic_guidance": ""})
    assert called == []


def test_candidate_cannot_change_grader_or_budget():
    adapter = integration.ProfileAdapter(CandidateProfile(), lambda *args: {},
                                        batch_type=Batch, max_metric_calls=5)
    with pytest.raises(ValueError):
        adapter.profile_for({"planner_guidance": "", "critic_guidance": "", "grader": "accept"})


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -1, 2, True])
def test_invalid_scores_fail_closed(score):
    adapter = integration.ProfileAdapter(
        CandidateProfile(), lambda *args: {"outputs": [{}], "scores": [score]},
        batch_type=Batch, max_metric_calls=5)
    with pytest.raises(ValueError):
        adapter.evaluate([{"split": "validation"}], {"planner_guidance": "", "critic_guidance": ""})


def test_budget_is_reserved_before_failed_execution():
    calls = []

    def failed(*args):
        calls.append(args)
        raise RuntimeError("execution failed")

    adapter = integration.ProfileAdapter(CandidateProfile(), failed, batch_type=Batch, max_metric_calls=1)
    with pytest.raises(RuntimeError, match="execution failed"):
        adapter.evaluate([{"split": "development"}], {"planner_guidance": "", "critic_guidance": ""})
    with pytest.raises(integration.GEPAEvaluationBudgetExceeded):
        adapter.evaluate([{"split": "development"}], {"planner_guidance": "", "critic_guidance": ""})
    assert len(calls) == 1


def test_missing_sdk_has_explicit_install_message(monkeypatch):
    def absent(name):
        raise ModuleNotFoundError("no SDK")

    monkeypatch.setattr(integration.importlib, "import_module", absent)
    with pytest.raises(integration.MissingGEPAError, match="requirements-improvement.txt"):
        integration.optimize_profile(
            CandidateProfile(), trainset=[{"split": "development"}],
            valset=[{"split": "validation"}], evaluate_batch=lambda *args: None,
            reflection_model=lambda prompts: prompts, max_metric_calls=4,
        )


def test_reflection_requires_aligned_traces():
    adapter = integration.ProfileAdapter(CandidateProfile(), lambda *args: {
        "outputs": [{}], "scores": [0.0], "trajectories": []},
        batch_type=Batch, max_metric_calls=5)
    with pytest.raises(ValueError, match="trajectories"):
        adapter.evaluate([{"split": "development"}],
                         {"planner_guidance": "", "critic_guidance": ""}, True)
