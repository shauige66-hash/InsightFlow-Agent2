"""Optional GEPA 0.1.4 integration for two bounded, textual components.

GEPA's metric-call budget counts batch examples, not provider requests or dollars.
The caller must also enforce the complete campaign's model/request budget. No
holdout or expected answers are accepted by this adapter. Final promotion remains
the responsibility of the independent campaign evaluator, not GEPA's score.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
import math
from typing import Any, Callable, Mapping

from improvement.profiles import CandidateProfile

COMPONENTS = frozenset({"planner_guidance", "critic_guidance"})
_PRIVATE_KEYS = frozenset({
    "expected", "expected_answer", "expected_answers", "ground_truth", "gold",
    "reference", "reference_answer", "reference_answers", "holdout", "api_key",
    "apikey", "authorization", "secret", "password", "access_token",
})


class MissingGEPAError(RuntimeError):
    """The explicit GEPA backend was selected without its optional SDK."""


class GEPAEvaluationBudgetExceeded(RuntimeError):
    """A batch would exceed the evaluator-call allowance."""


def _load_gepa():
    try:
        package = importlib.import_module("gepa")
        batch_type = importlib.import_module("gepa.core.adapter").EvaluationBatch
    except ImportError as error:
        raise MissingGEPAError(
            "GEPA backend requires the optional SDK: "
            "python -m pip install -r requirements-improvement.txt"
        ) from error
    return package.optimize, batch_type


def _safe_value(value: Any, depth: int = 0) -> Any:
    """Keep bounded JSON diagnostics while dropping labelled private fields.

    This is defence in depth, not a DLP or security sandbox. evaluate_batch is a
    trusted host callback and must never place secrets/gold in free-form strings.
    """
    if depth > 8:
        return "[depth limit]"
    if isinstance(value, Mapping):
        result = {}
        for key, item in list(value.items())[:40]:
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in _PRIVATE_KEYS:
                continue
            result[str(key)[:100]] = _safe_value(item, depth + 1)
        return result
    if isinstance(value, (tuple, list)):
        return [_safe_value(item, depth + 1) for item in value[:30]]
    if isinstance(value, str):
        return value[:3000]
    if value is None or type(value) is bool or type(value) is int:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    return "[unsupported diagnostic]"


def _check_examples(examples: list[dict], *, split: str | None = None) -> list[dict]:
    if not isinstance(examples, list) or not examples:
        raise ValueError("GEPA requires non-empty development and validation sets")
    checked = []
    for example in examples:
        if (not isinstance(example, dict) or set(example) != {"split"}
                or example["split"] not in {"development", "validation"}):
            raise ValueError("GEPA examples may contain only development/validation split")
        if split is not None and example["split"] != split:
            raise ValueError(f"Expected {split} examples")
        checked.append(dict(example))
    return checked


@dataclass(frozen=True)
class GEPAOptimizationResult:
    profile: CandidateProfile
    metric_calls: int
    candidates_evaluated: int


class ProfileAdapter:
    """Duck-typed implementation of official GEPAAdapter's stable contract.

    evaluate_batch(profile, batch, capture_traces) returns dicts with aligned
    outputs, scores, trajectories. Each example is one complete split in this
    first integration; its score is the host's aggregate business correctness.
    """

    propose_new_texts = None

    def __init__(self, seed: CandidateProfile, evaluate_batch: Callable,
                 *, batch_type: type, max_metric_calls: int):
        self.seed = CandidateProfile.from_dict(seed.to_dict())
        self.evaluate_batch = evaluate_batch
        self.batch_type = batch_type
        self.max_metric_calls = max_metric_calls
        self.metric_calls = 0

    def profile_for(self, candidate: dict[str, str]) -> CandidateProfile:
        if not isinstance(candidate, dict) or set(candidate) != COMPONENTS:
            raise ValueError("GEPA may update only planner_guidance and critic_guidance")
        return CandidateProfile.from_dict({
            **self.seed.to_dict(),
            "planner_guidance": candidate["planner_guidance"],
            "critic_guidance": candidate["critic_guidance"],
        })

    def evaluate(self, batch: list[dict], candidate: dict[str, str],
                 capture_traces: bool = False):
        checked = _check_examples(batch)
        profile = self.profile_for(candidate)
        if self.metric_calls + len(checked) > self.max_metric_calls:
            raise GEPAEvaluationBudgetExceeded("GEPA metric-call budget exhausted")
        # Reserve before executing: failed network/workflow attempts still count.
        self.metric_calls += len(checked)
        result = self.evaluate_batch(profile, checked, capture_traces)
        if not isinstance(result, dict):
            raise ValueError("evaluate_batch must return outputs/scores/trajectories")
        outputs, scores = result.get("outputs"), result.get("scores")
        if (not isinstance(outputs, list) or not isinstance(scores, list)
                or len(outputs) != len(checked) or len(scores) != len(checked)):
            raise ValueError("GEPA evaluation outputs/scores must align with examples")
        if any(type(score) not in (int, float) or not math.isfinite(score)
               or not 0 <= score <= 1 for score in scores):
            raise ValueError("GEPA scores must be finite business correctness in [0,1]")
        trajectories = result.get("trajectories")
        if capture_traces and (not isinstance(trajectories, list)
                               or len(trajectories) != len(checked)):
            raise ValueError("Captured GEPA trajectories must align with examples")
        return self.batch_type(
            outputs=[_safe_value(output) for output in outputs],
            scores=[float(score) for score in scores],
            trajectories=([_safe_value(trace) for trace in trajectories]
                          if capture_traces else None),
        )

    def make_reflective_dataset(self, candidate: dict[str, str], eval_batch: Any,
                                components_to_update: list[str]):
        self.profile_for(candidate)
        if not components_to_update or not set(components_to_update) <= COMPONENTS:
            raise ValueError("GEPA requested an unsupported component")
        if eval_batch.trajectories is None:
            raise ValueError("Reflection requires captured execution diagnostics")
        if not (len(eval_batch.outputs) == len(eval_batch.scores)
                == len(eval_batch.trajectories)):
            raise ValueError("Reflection diagnostics must align")
        examples = [
            {
                "Inputs": {"evaluation_scope": "development"},
                "Generated Outputs": _safe_value(output),
                "Feedback": {"business_correctness": score,
                             "diagnostics": _safe_value(trace)},
            }
            for output, score, trace in zip(
                eval_batch.outputs, eval_batch.scores, eval_batch.trajectories
            )
        ]
        return {component: examples for component in components_to_update}


def optimize_profile(seed: CandidateProfile, *, trainset: list[dict],
                     valset: list[dict], evaluate_batch: Callable,
                     reflection_model: Callable[[list[str]], list[str]],
                     max_metric_calls: int, seed_number: int = 0
                     ) -> GEPAOptimizationResult:
    """Run official GEPA and return an unapproved, bounded profile candidate.

    reflection_model receives a one-element list and must return one text. This
    lets the caller charge every actual reflection request to its campaign quota.
    It must not configure hidden provider retries or unbounded internal calls.
    """
    trainset = _check_examples(trainset, split="development")
    valset = _check_examples(valset, split="validation")
    if type(max_metric_calls) is not int or max_metric_calls < 2:
        raise ValueError("max_metric_calls must be an integer of at least 2")
    if not callable(evaluate_batch) or not callable(reflection_model):
        raise ValueError("Explicit evaluator and budgeted reflection callable required")
    optimize, batch_type = _load_gepa()
    adapter = ProfileAdapter(seed, evaluate_batch, batch_type=batch_type,
                             max_metric_calls=max_metric_calls)

    def reflect(prompt: str | list[dict]) -> str:
        # v0.1.4 LanguageModel is prompt -> str, while our provider facade batches.
        text = prompt if isinstance(prompt, str) else json.dumps(prompt, ensure_ascii=False)
        outputs = reflection_model([text])
        if (not isinstance(outputs, list) or len(outputs) != 1
                or not isinstance(outputs[0], str) or not outputs[0].strip()):
            raise ValueError("Reflection callable must return exactly one non-empty text")
        return outputs[0]

    result = optimize(
        seed_candidate={"planner_guidance": seed.planner_guidance,
                        "critic_guidance": seed.critic_guidance},
        trainset=trainset, valset=valset, adapter=adapter,
        reflection_lm=reflect, max_metric_calls=max_metric_calls,
        candidate_selection_strategy="pareto", use_merge=False,
        reflection_minibatch_size=1, cache_evaluation=False,
        track_best_outputs=False, use_wandb=False, use_mlflow=False,
        display_progress_bar=False, seed=seed_number, raise_on_exception=True,
        acceptance_criterion="strict_improvement",
    )
    return GEPAOptimizationResult(
        profile=adapter.profile_for(result.best_candidate),
        metric_calls=adapter.metric_calls,
        candidates_evaluated=len(result.candidates),
    )
