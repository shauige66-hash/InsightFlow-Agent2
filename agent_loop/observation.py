"""运行期观察：把一次任务实际发生了什么，压成一份可决策、可序列化的快照。

观察必须由 state 确定性构造——同样的 state 必须得到同样的观察，否则
"恢复后重新决策"就会得到不同的输入，幂等与可复现都无从谈起。

三条边界：

1. 只描述形状，不搬运数据。结果只记录"是表还是标量、几行、哪些列"，
   完整结果留在执行结果与产物文件里；观察会进 Checkpoint，也会进 Prompt。
2. 不持有模型客户端或 DataFrame，只有普通 Python 值。
3. 工具输出属于不可信内容：摘要会截断，且只在 Prompt 里作为"观察到的事实"
   出现，不参与任何路由判断之外的解释。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

MAX_ERROR_SUMMARY = 400
MAX_FEEDBACK = 400
MAX_EVIDENCE_REFS = 5
MAX_REMAINING_TASKS = 5
MAX_ARTIFACT_REFS = 5

#: 决策时"证据是否足够"的判定阈值：至少一条片段才认为有依据
MIN_EVIDENCE = 1


@dataclass(frozen=True)
class Observation:
    run_id: str
    task_id: str
    task_index: int
    plan_revision: int
    loop_step: int
    task_goal: str
    expected_output: str
    tool_used: str | None = None
    execution_success: bool | None = None
    error_kind: str | None = None
    error_summary: str | None = None
    validation_passed: bool | None = None
    critic_feedback: str | None = None
    result_shape: dict[str, Any] = field(default_factory=dict)
    evidence_refs: list[dict[str, Any]] = field(default_factory=list)
    artifact_refs: list[str] = field(default_factory=list)
    remaining_tasks: list[dict[str, Any]] = field(default_factory=list)
    remaining_budget: dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    refine_count: int = 0
    extra_retrievals: int = 0
    runtime_replans: int = 0
    repeated_actions: dict[str, int] = field(default_factory=dict)

    @property
    def evidence_sufficient(self) -> bool:
        """当前证据是否够用。没有任何知识源时不算"不足"。"""
        return len(self.evidence_refs) >= MIN_EVIDENCE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        """决策相关内容的稳定指纹，用于识别"同样的观察又来了"。"""
        payload = {
            "task_id": self.task_id,
            "execution_success": self.execution_success,
            "error_kind": self.error_kind,
            "error_summary": self.error_summary,
            "validation_passed": self.validation_passed,
            "critic_feedback": self.critic_feedback,
            "result_shape": self.result_shape,
            "evidence": [item.get("chunk_id") for item in self.evidence_refs],
            "plan_revision": self.plan_revision,
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _task_at(state: dict[str, Any]) -> tuple[dict[str, Any] | None, int]:
    plan = state.get("analysis_plan") or []
    index = int(state.get("current_task_index") or 0)
    if index < 0 or index >= len(plan):
        return None, index
    return plan[index], index


def _latest_execution(state: dict[str, Any], task_id: str) -> dict[str, Any] | None:
    results = [
        item
        for item in (state.get("execution_results") or [])
        if item.get("task_id") == task_id
    ]
    return results[-1] if results else None


def _result_shape(execution: dict[str, Any] | None) -> dict[str, Any]:
    """结果的形状，不包含任何一行数据。"""
    if execution is None:
        return {"kind": "none"}
    if not execution.get("success"):
        return {"kind": "failed"}

    value = execution.get("result")
    if isinstance(value, list):
        rows = len(value)
        columns: list[str] = []
        if value and isinstance(value[0], dict):
            columns = sorted(str(key) for key in value[0].keys())
        return {"kind": "table", "rows": rows, "columns": columns}
    if isinstance(value, dict):
        return {"kind": "record", "columns": sorted(str(key) for key in value.keys())}
    if value is None:
        return {"kind": "none"}
    return {"kind": "scalar"}


def _evidence_refs(state: dict[str, Any], task_id: str) -> list[dict[str, Any]]:
    stored = (state.get("task_knowledge") or {}).get(task_id) or []
    refs = [
        {
            "chunk_id": item.get("chunk_id"),
            "source": item.get("source"),
            "page": item.get("page"),
            "score": round(float(item.get("score") or 0.0), 4),
        }
        for item in stored[:MAX_EVIDENCE_REFS]
    ]
    return refs


def _remaining_budget(state: dict[str, Any]) -> dict[str, Any]:
    from harness import policy

    try:
        return policy.remaining(state)
    except Exception:  # noqa: BLE001 - 观察不能因为预算读取失败而中断
        return {}


def _repeated_actions(state: dict[str, Any], task_id: str) -> dict[str, int]:
    counts = (state.get("action_counts") or {}).get(task_id) or {}
    return {str(action): int(value) for action, value in counts.items()}


def _truncate(value: Any, limit: int) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text if len(text) <= limit else text[:limit] + "…"


def build_observation(state: dict[str, Any]) -> Observation:
    """从 state 构造成当前任务的观察。"""
    task, index = _task_at(state)
    task_id = str((task or {}).get("task_id") or "")
    execution = _latest_execution(state, task_id) if task_id else None

    plan = state.get("analysis_plan") or []
    remaining = [
        {
            "task_id": item.get("task_id"),
            "description": item.get("description"),
            "target_tables": list(item.get("target_tables") or []),
        }
        for item in plan[index + 1 : index + 1 + MAX_REMAINING_TASKS]
    ]

    return Observation(
        run_id=str(state.get("run_id") or ""),
        task_id=task_id,
        task_index=index,
        plan_revision=int(state.get("plan_revision") or 0),
        loop_step=int(state.get("loop_step") or 0),
        task_goal=str((task or {}).get("description") or ""),
        expected_output=str((task or {}).get("expected_output") or ""),
        tool_used=state.get("current_tool"),
        execution_success=(
            bool(execution.get("success")) if execution is not None else None
        ),
        error_kind=state.get("execution_error_kind"),
        error_summary=_truncate(state.get("execution_error"), MAX_ERROR_SUMMARY),
        validation_passed=state.get("validation_passed"),
        critic_feedback=_truncate(state.get("critic_feedback"), MAX_FEEDBACK),
        result_shape=_result_shape(execution),
        evidence_refs=_evidence_refs(state, task_id) if task_id else [],
        artifact_refs=[
            str(item) for item in (state.get("artifacts") or [])[:MAX_ARTIFACT_REFS]
        ],
        remaining_tasks=remaining,
        remaining_budget=_remaining_budget(state),
        retry_count=int(state.get("retry_count") or 0),
        refine_count=int(state.get("refine_count") or 0),
        extra_retrievals=int(state.get("extra_retrievals") or 0),
        runtime_replans=int(state.get("runtime_replans") or 0),
        repeated_actions=_repeated_actions(state, task_id) if task_id else {},
    )


__all__ = ["MIN_EVIDENCE", "Observation", "build_observation"]
