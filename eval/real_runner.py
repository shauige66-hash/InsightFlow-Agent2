"""Independent host-side grading for real Agent runs and repeated baselines.

Candidates supply plain prompt profiles, never graders, paths, or budgets.
Only the explicit evaluation key is used by the default model builder.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from eval import business_oracles
from eval.real_tasks import SPLITS, build_real_tasks, dataset_fingerprint
from utils.agent_loop import AgentBudget

SCORER_VERSION = "business-regions-independent-v1"


class EvaluationContractChanged(RuntimeError):
    """The fixed scoring contract or raw input changed during an experiment."""


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def scorer_fingerprint() -> str:
    root = Path(__file__).parent
    digest = hashlib.sha256(SCORER_VERSION.encode())
    for name in ("real_runner.py", "real_tasks.py", "business_oracles.py"):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def _endpoint_identity(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Evaluation base URL must be an HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Evaluation base URL must not contain credentials, query or fragment")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


def _counter(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("Invalid usage counter returned by the workflow")
    return value


def _verified_png_count(artifacts: list, output_root: Path) -> int:
    """Check actual, distinct run-owned images rather than trusting path strings."""
    from PIL import Image

    valid = set()
    root = output_root.resolve()
    for raw in artifacts:
        try:
            path = Path(raw).resolve(strict=True)
            if not path.is_relative_to(root) or path in valid or path.stat().st_size > 20_000_000:
                continue
            with Image.open(path) as image:
                if image.format != "PNG" or image.width * image.height > 16_000_000:
                    continue
                image.verify()
            valid.add(path)
        except (OSError, ValueError, TypeError, SyntaxError, Image.DecompressionBombError):
            continue
    return len(valid)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["task_id"], []).append(row)
    total = len(rows)
    passed = sum(row["passed"] is True for row in rows)
    return {"total": total, "passed": passed,
            "correctness_rate": passed / total if total else 0.0,
            "completion_rate": sum(row["completed"] is True for row in rows) / total if total else 0.0,
            "all_trials_pass_rate": sum(all(r["passed"] for r in group) for group in grouped.values()) / len(grouped) if grouped else 0.0,
            "llm_calls": sum(row["llm_calls"] for row in rows),
            "tool_calls": sum(row["tool_calls"] for row in rows),
            "seconds": round(sum(row["seconds"] for row in rows), 6),
            "errors": sum(row["error"] is not None for row in rows),
            "tokens": None, "cost_usd": None}


class RealEvaluator:
    def __init__(self, output_root: Path | str, *, model_factory_builder: Callable | None = None,
                 analysis_runner: Callable | None = None, repeats: int = 3,
                 budget: AgentBudget | dict | None = None, model: str | None = None,
                 base_url: str | None = None, mcp: Any = None, holdout_seed: int = 0):
        if type(repeats) is not int or not 1 <= repeats <= 20:
            raise ValueError("repeats must be an integer between 1 and 20")
        if budget is None:
            budget = AgentBudget(max_llm_calls=20, max_seconds=300)
        if isinstance(budget, dict):
            unknown = set(budget) - {"max_rounds", "max_tool_calls", "max_llm_calls", "max_seconds"}
            if unknown:
                raise ValueError(f"Unknown evaluation budget fields: {sorted(unknown)}")
            for key, value in budget.items():
                if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"Invalid evaluation budget: {key}")
                if key != "max_seconds" and type(value) is not int:
                    raise ValueError(f"Evaluation budget must be integral: {key}")
            budget = AgentBudget.from_dict(budget)
        if not isinstance(budget, AgentBudget):
            raise TypeError("budget must be AgentBudget or a dictionary")
        for key, value in budget.to_dict().items():
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid evaluation budget: {key}")
            if key != "max_seconds" and type(value) is not int:
                raise ValueError(f"Evaluation budget must be integral: {key}")
        self.output_root = Path(output_root).resolve()
        self.repeats, self.budget, self.mcp = repeats, budget, mcp
        self.model = model or os.getenv("INSIGHTFLOW_EVAL_REAL_MODEL", "")
        self.base_url = _endpoint_identity(base_url or os.getenv("INSIGHTFLOW_EVAL_REAL_BASE_URL", "http://127.0.0.1:8000/v1"))
        self._api_key = os.getenv("INSIGHTFLOW_EVAL_REAL_API_KEY", "")
        self._builder = model_factory_builder or self._default_builder
        self._runner = analysis_runner
        self._execution_identity = (self._builder, self._runner)
        self._tasks = {split: build_real_tasks(split, holdout_seed=holdout_seed) for split in SPLITS}
        self._scorer = scorer_fingerprint()
        self._oracle_functions = (business_oracles.expected_answers, business_oracles.grade_business_answers)
        backend = str(getattr(mcp, "mode", "direct"))
        self._configuration = {"model": self.model, "base_url": self.base_url,
                               "repeats": repeats, "budget": budget.to_dict(), "sql_backend": backend,
                               "runner_type": "injected" if analysis_runner is not None else "real",
                               "model_factory": "injected" if model_factory_builder is not None else "real"}
        self.config_fingerprint = _digest(self._configuration)

    def _default_builder(self, run_id: str):
        if not self._api_key or not self.model:
            raise ValueError("Real evaluation requires INSIGHTFLOW_EVAL_REAL_API_KEY and INSIGHTFLOW_EVAL_REAL_MODEL")
        from core.config import Settings
        from core.llm import build_model_factory
        return build_model_factory(run_id, Settings(api_key=self._api_key, model=self.model,
                                                   base_url=self.base_url, client_app="insightflow-evaluation",
                                                   timeout_seconds=min(60, self.budget.max_seconds)))

    def expected_trials(self, split: str) -> int:
        if split not in SPLITS:
            raise ValueError(f"Unknown evaluation split: {split}")
        return len(self._tasks[split]) * self.repeats

    def _assert_fixed(self) -> None:
        if scorer_fingerprint() != self._scorer or self._oracle_functions != (
            business_oracles.expected_answers, business_oracles.grade_business_answers
        ) or self._execution_identity != (self._builder, self._runner) or _digest({
                      **self._configuration, "sql_backend": str(getattr(self.mcp, "mode", "direct")),
                      "budget": self.budget.to_dict(), "repeats": self.repeats,
                      "model": self.model, "base_url": self.base_url}) != self.config_fingerprint:
            raise EvaluationContractChanged("Evaluation scorer/configuration changed; start a new experiment")

    def evaluate(self, profile, split: str) -> dict[str, Any]:
        from improvement.profiles import CandidateProfile
        profile = CandidateProfile.from_dict(profile.to_dict() if isinstance(profile, CandidateProfile) else profile)
        self.expected_trials(split)
        self._assert_fixed()
        # Fail before creating trials or swallowing a credential error as a model failure.
        if self._builder == self._default_builder and (not self._api_key or not self.model):
            raise ValueError("Real evaluation requires INSIGHTFLOW_EVAL_REAL_API_KEY and INSIGHTFLOW_EVAL_REAL_MODEL")
        run_directory = self.output_root / f"evaluation-{uuid4().hex}"
        run_directory.mkdir(parents=True, exist_ok=False)
        rows = []
        for task in self._tasks[split]:
            for index in range(self.repeats):
                self._assert_fixed()
                trial_id = f"eval-{uuid4().hex}"
                trial = run_directory / trial_id
                inputs = trial / "inputs"
                sources, knowledge = task.materialise(inputs)
                task.verify_files(inputs)
                started = time.perf_counter()
                row: dict[str, Any] = {"task_id": task.task_id, "trial_index": index,
                    "trial_id": trial_id, "profile_id": profile.profile_id, "passed": False,
                    "completed": False, "error": None, "llm_calls": 0, "tool_calls": 0,
                    "seconds": 0.0, "feedback": {}, "tokens": None, "cost_usd": None,
                    "trace_path": str(trial / "analysis" / "logs" / f"{trial_id}.jsonl")}
                try:
                    runner = self._runner
                    if runner is None:
                        from services.analysis_service import run_analysis
                        runner = run_analysis
                    kwargs = dict(data_sources=sources, knowledge_sources=knowledge,
                                  analysis_focus=task.question + business_oracles.answer_fields("toy-regions"),
                                  run_id=trial_id, model_factory=self._builder(trial_id),
                                  output_root=trial / "analysis", budget=self.budget,
                                  improvement_profile=profile, enable_dynamic_loop=True,
                                  memories=[], conversation_context=None)
                    if self.mcp is not None:
                        kwargs["mcp"] = self.mcp
                    result = runner(**kwargs)
                    task.verify_files(inputs)
                    self._assert_fixed()
                    usage = (result.get("execution_budget") or {}).get("usage") or {}
                    row["llm_calls"] = _counter(usage.get("llm_calls", 0))
                    row["tool_calls"] = _counter(usage.get("tool_calls", 0))
                    from utils.llm_ledger import request_count
                    row["llm_calls"] = max(row["llm_calls"], request_count(trial_id, trial / "analysis"))
                    row["completed"] = result.get("analysis_status") == "completed"
                    grade = business_oracles.grade_business_answers("toy-regions", str(result.get("report_content") or ""), sources)
                    planned = len(result.get("visualization_plan") or [])
                    produced = _verified_png_count(result.get("artifacts") or [], trial / "analysis")
                    charts_ok = not result.get("failed_charts") and produced >= planned
                    within_budget = row["llm_calls"] <= self.budget.max_llm_calls and row["tool_calls"] <= self.budget.max_tool_calls
                    row["passed"] = bool(row["completed"] and grade["passed"] and charts_ok and within_budget)
                    # Never return expected answers or reference values to candidate generation.
                    row["feedback"] = {"reason": grade["reason"],
                                       "failed_checks": [name for name, ok in (("completion", row["completed"]),
                                           ("business_answers", grade["passed"]), ("charts", charts_ok),
                                           ("execution_budget", within_budget)) if not ok]}
                    row["feedback"]["critic_feedback"] = str(result.get("critic_feedback") or "")[:800]
                    row["feedback"]["report_excerpt"] = str(result.get("report_content") or "")[:1600]
                    if result.get("report_path"):
                        row["report_path"] = str(result["report_path"])
                except EvaluationContractChanged:
                    raise
                except Exception as error:
                    # Input tampering invalidates the experiment, rather than producing a
                    # misleading ordinary failure that can be selected around.
                    try:
                        task.verify_files(inputs)
                    except (OSError, ValueError) as drift:
                        raise EvaluationContractChanged("Evaluation input changed during execution") from drift
                    self._assert_fixed()
                    row["error"] = f"{type(error).__name__}: evaluation execution failed"
                    row["feedback"] = {"failed_checks": ["execution"], "error_kind": type(error).__name__}
                    from utils.llm_ledger import request_count
                    row["llm_calls"] = request_count(trial_id, trial / "analysis")
                row["seconds"] = round(time.perf_counter() - started, 6)
                if row["seconds"] > self.budget.max_seconds:
                    row["passed"] = False
                    row["feedback"].setdefault("failed_checks", []).append("time_budget")
                rows.append(row)
        self._assert_fixed()
        report = {"schema_version": 1, "split": split, "profile_id": profile.profile_id,
                  "scorer_version": SCORER_VERSION, "scorer_fingerprint": self._scorer,
                  "dataset_fingerprint": dataset_fingerprint(self._tasks[split]),
                  "config_fingerprint": self.config_fingerprint, "configuration": self._configuration,
                  "scope": "Same-domain synthetic business variants; independent host CSV/Decimal grading. Injected runs are engineering tests, not model capability evidence.",
                  "created_at": datetime.now(timezone.utc).isoformat(), "rows": rows,
                  "summary": summarize_rows(rows)}
        (run_directory / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report
