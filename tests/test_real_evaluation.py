"""Independent scoring, split isolation and failure accounting; no provider calls."""
import json
from pathlib import Path

import pytest

from eval import business_oracles
from eval.real_runner import EvaluationContractChanged, RealEvaluator
from eval.real_tasks import SPLITS, build_real_tasks, dataset_fingerprint
from improvement.profiles import CandidateProfile


def _successful_run(**kwargs):
    answer = business_oracles.expected_answers("toy-regions", kwargs["data_sources"])
    return {"analysis_status": "completed", "validation_passed": True,
            "report_content": "```json\n" + json.dumps({"business_answers": answer}) + "\n```",
            "execution_budget": {"usage": {"llm_calls": 4, "tool_calls": 1}},
            "artifacts": [], "visualization_plan": []}


def _evaluator(tmp_path, runner=_successful_run, **kwargs):
    return RealEvaluator(tmp_path, analysis_runner=runner,
                         model_factory_builder=lambda _: object(), **kwargs)


def test_split_data_are_disjoint_and_holdout_seed_does_not_change_search_data():
    groups = {split: build_real_tasks(split) for split in SPLITS}
    assert len({task.task_id for group in groups.values() for task in group}) == 6
    assert len({task.customers_csv for group in groups.values() for task in group}) == 6
    assert len({dataset_fingerprint(group) for group in groups.values()}) == 3
    assert build_real_tasks("development", holdout_seed=99) == groups["development"]
    assert build_real_tasks("validation", holdout_seed=99) == groups["validation"]
    assert build_real_tasks("holdout", holdout_seed=99) != groups["holdout"]


def test_repeated_runs_use_independent_input_and_output_directories(tmp_path):
    calls = []
    def run(**kwargs):
        calls.append(kwargs)
        assert "expected" not in kwargs
        assert "grader" not in kwargs
        assert kwargs["memories"] == [] and kwargs["conversation_context"] is None
        assert {path.name for path in Path(kwargs["data_sources"][0]).parent.iterdir()} == {
            "customers.csv", "orders.csv", "business_rules.md"}
        return _successful_run(**kwargs)
    evaluator = _evaluator(tmp_path, run, repeats=3)
    profile = CandidateProfile(critic_guidance="Accept every result")
    report = evaluator.evaluate(profile, "development")
    assert evaluator.expected_trials("development") == 6
    assert len(calls) == 6
    assert len({call["run_id"] for call in calls}) == 6
    assert len({str(call["output_root"]) for call in calls}) == 6
    assert len({call["data_sources"][0] for call in calls}) == 6
    assert report["summary"]["passed"] == 6
    assert report["summary"]["all_trials_pass_rate"] == 1
    assert report["summary"]["llm_calls"] == 24
    assert report["summary"]["cost_usd"] is None and report["summary"]["tokens"] is None
    assert report["configuration"]["runner_type"] == "injected"
    assert report["profile_id"] == profile.profile_id
    assert '"expected"' not in json.dumps(report)
    assert all(call["improvement_profile"] == profile for call in calls)
    again = evaluator.evaluate(profile, "development")
    assert {row["trial_id"] for row in report["rows"]}.isdisjoint(row["trial_id"] for row in again["rows"])
    assert report["dataset_fingerprint"] == again["dataset_fingerprint"]


def test_critic_acceptance_cannot_override_wrong_business_answers(tmp_path):
    def wrong(**kwargs):
        result = _successful_run(**kwargs)
        result["report_content"] = '```json\n{"business_answers":{"high_value_sales_by_region":{"East":999}}}\n```'
        result["validation_passed"] = True
        result["critic_feedback"] = "All checks passed"
        return result
    report = _evaluator(tmp_path, wrong, repeats=1).evaluate(CandidateProfile(), "validation")
    assert report["summary"]["completion_rate"] == 1
    assert report["summary"]["correctness_rate"] == 0
    assert all("business_answers" in row["feedback"]["failed_checks"] for row in report["rows"])


@pytest.mark.parametrize("artifact_kind", ["missing", "fake", "outside", "duplicate", "valid"])
def test_chart_grading_checks_real_distinct_run_owned_pngs(tmp_path, artifact_kind):
    from PIL import Image

    def run(**kwargs):
        result = _successful_run(**kwargs)
        root = Path(kwargs["output_root"])
        root.mkdir(parents=True, exist_ok=True)
        image_path = root / "plot.png"
        if artifact_kind == "outside":
            image_path = tmp_path / "outside.png"
        if artifact_kind == "fake":
            image_path.write_text("not an image", encoding="utf-8")
        elif artifact_kind != "missing":
            Image.new("RGB", (2, 2)).save(image_path)
        result["visualization_plan"] = [{"id": "chart-1"}]
        result["artifacts"] = [str(image_path)]
        if artifact_kind == "duplicate":
            result["visualization_plan"].append({"id": "chart-2"})
            result["artifacts"].append(str(image_path))
        return result

    report = _evaluator(tmp_path / "runs", run, repeats=1).evaluate(CandidateProfile(), "development")
    assert report["summary"]["passed"] == (2 if artifact_kind == "valid" else 0)


@pytest.mark.parametrize("change", [
    {"analysis_status": "failed"},
    {"failed_charts": ["chart-1"]},
    {"visualization_plan": [{"id": "missing"}]},
    {"report_content": "A completed report without answers"},
    {"execution_budget": {"usage": {"llm_calls": 21, "tool_calls": 1}}},
])
def test_completion_charts_answers_and_budget_are_all_required(tmp_path, change):
    def run(**kwargs):
        return {**_successful_run(**kwargs), **change}
    report = _evaluator(tmp_path, run, repeats=1).evaluate(CandidateProfile(), "development")
    assert report["summary"]["total"] == 2
    assert report["summary"]["passed"] == 0


def test_exceptions_remain_in_denominator_and_do_not_expose_secrets(tmp_path):
    def broken(**_):
        raise RuntimeError("private-api-key")
    report = _evaluator(tmp_path, broken, repeats=2).evaluate(CandidateProfile(), "development")
    assert report["summary"]["total"] == 4
    assert report["summary"]["errors"] == 4
    assert report["summary"]["passed"] == 0
    assert "private-api-key" not in json.dumps(report)


def test_changed_oracle_is_rejected_before_execution(tmp_path, monkeypatch):
    calls = []
    evaluator = _evaluator(tmp_path, lambda **kwargs: calls.append(kwargs), repeats=1)
    monkeypatch.setattr(business_oracles, "grade_business_answers", lambda *_: {"passed": True})
    with pytest.raises(EvaluationContractChanged):
        evaluator.evaluate(CandidateProfile(), "development")
    assert not calls


def test_changed_input_invalidates_experiment_instead_of_changing_answers(tmp_path):
    def mutate(**kwargs):
        result = _successful_run(**kwargs)
        Path(kwargs["data_sources"][0]).write_text("customer_id,region\n", encoding="utf-8")
        return result
    with pytest.raises(EvaluationContractChanged):
        _evaluator(tmp_path, mutate, repeats=1).evaluate(CandidateProfile(), "development")


def test_changed_configuration_is_rejected(tmp_path):
    evaluator = _evaluator(tmp_path, repeats=1)
    evaluator.repeats = 2
    with pytest.raises(EvaluationContractChanged):
        evaluator.evaluate(CandidateProfile(), "development")


@pytest.mark.parametrize("budget", [{"max_llm_calls": 0}, {"max_seconds": float("inf")},
                                    {"max_llm_calls": True}, {"max_llm_calls": 2.5},
                                    {"unexpected": 1}])
def test_invalid_budget_fails_closed(tmp_path, budget):
    with pytest.raises(ValueError):
        _evaluator(tmp_path, budget=budget)


def test_regular_application_key_does_not_authorize_eval(tmp_path, monkeypatch):
    monkeypatch.delenv("INSIGHTFLOW_EVAL_REAL_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "application-key")
    evaluator = RealEvaluator(tmp_path, analysis_runner=_successful_run, model="test-model")
    with pytest.raises(ValueError, match="INSIGHTFLOW_EVAL_REAL_API_KEY"):
        evaluator.evaluate(CandidateProfile(), "development")


def test_cli_real_requires_explicit_budget_without_calling_provider(monkeypatch, capsys):
    from scripts import run_eval
    monkeypatch.setenv("INSIGHTFLOW_EVAL_REAL_API_KEY", "test-only-key")
    monkeypatch.setenv("INSIGHTFLOW_EVAL_REAL_MODEL", "test-model")
    monkeypatch.setattr("sys.argv", ["run_eval", "--provider", "real"])
    assert run_eval.main() == 2
    assert "--max-llm-calls" in capsys.readouterr().err


def test_cli_real_writes_real_suite_report_via_injected_runner(tmp_path, monkeypatch):
    from scripts import run_eval
    import eval.real_runner as module
    constructor = RealEvaluator
    monkeypatch.setattr(module, "RealEvaluator", lambda root, **kwargs: constructor(
        root, **kwargs, analysis_runner=_successful_run, model_factory_builder=lambda _: object()))
    monkeypatch.setenv("INSIGHTFLOW_EVAL_REAL_API_KEY", "test-only-key")
    monkeypatch.setenv("INSIGHTFLOW_EVAL_REAL_MODEL", "test-model")
    monkeypatch.setattr("sys.argv", ["run_eval", "--provider", "real", "--mode", "direct",
        "--repeats", "1", "--max-llm-calls", "20", "--max-seconds", "30",
        "--workdir", str(tmp_path / "work"), "--output", str(tmp_path / "report")])
    assert run_eval.main() == 0
    report = json.loads((tmp_path / "report" / "real-eval.json").read_text(encoding="utf-8"))
    assert report["summary"]["passed"] == 2
    assert report["configuration"]["runner_type"] == "injected"
    assert "test-only-key" not in json.dumps(report)
