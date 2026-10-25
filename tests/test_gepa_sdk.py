"""Exercise the real optional SDK using deterministic, free evaluator/LM callbacks."""
import pytest

from improvement.gepa_adapter import optimize_profile
from improvement.profiles import CandidateProfile


def test_actual_gepa_search_uses_adapter_and_budgeted_reflection():
    pytest.importorskip("gepa")
    evaluations, reflections = [], []

    def evaluate(profile, batch, capture_traces):
        evaluations.append(profile)
        score = 0.75 if profile.planner_guidance or profile.critic_guidance else 0.25
        output = {"reason": "Check join cardinality", "observed_score": score}
        return {"outputs": [output for _ in batch], "scores": [score for _ in batch],
                "trajectories": [output for _ in batch] if capture_traces else None}

    def reflection(prompts):
        reflections.extend(prompts)
        return ["```\nCheck join cardinality before aggregation.\n```" for _ in prompts]

    result = optimize_profile(CandidateProfile(), trainset=[{"split": "development"}],
                              valset=[{"split": "validation"}], evaluate_batch=evaluate,
                              reflection_model=reflection, max_metric_calls=4)
    assert reflections and evaluations
    assert result.profile.planner_guidance or result.profile.critic_guidance
    assert result.metric_calls <= 4
    assert result.candidates_evaluated >= 2
