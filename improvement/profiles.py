"""Data-only candidate instructions; no evaluator, permissions or executable hooks.

These profiles are experimental inputs, not proof that a lesson is correct.
Only the independent campaign evaluator may accept a candidate version.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from typing import Any

AGENTS = frozenset({"planner", "critic"})
MAX_GUIDANCE_CHARS = 6_000
MAX_LESSONS = 12
MAX_LESSON_CHARS = 500
MAX_SOURCE_TRIALS = 8


def _text(value: Any, name: str, limit: int, *, required: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be text")
    if len(value) > limit or "\x00" in value or (required and not value.strip()):
        raise ValueError(f"{name} is empty, too long or contains NUL")
    return value


def _fields(value: Any, allowed: set[str], name: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unsupported {name} fields: {sorted(map(str, unknown))}")
    return value


@dataclass(frozen=True)
class Experience:
    agent: str
    text: str
    source_trials: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.agent, str) or self.agent not in AGENTS:
            raise ValueError("Experience agent must be planner or critic")
        _text(self.text, "experience.text", MAX_LESSON_CHARS, required=True)
        if not isinstance(self.source_trials, tuple) or len(self.source_trials) > MAX_SOURCE_TRIALS:
            raise ValueError("source_trials must be a bounded tuple")
        for trial in self.source_trials:
            _text(trial, "source_trial", 200, required=True)
        if len(set(self.source_trials)) != len(self.source_trials):
            raise ValueError("source_trials must be unique")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Experience":
        data = _fields(payload, {"agent", "text", "source_trials"}, "experience")
        refs = data.get("source_trials", [])
        if not isinstance(refs, (list, tuple)):
            raise ValueError("source_trials must be a list")
        return cls(agent=data.get("agent"), text=data.get("text"), source_trials=tuple(refs))

    def to_dict(self) -> dict[str, Any]:
        return {"agent": self.agent, "text": self.text, "source_trials": list(self.source_trials)}


@dataclass(frozen=True)
class CandidateProfile:
    planner_guidance: str = ""
    critic_guidance: str = ""
    lessons: tuple[Experience, ...] = ()

    def __post_init__(self) -> None:
        _text(self.planner_guidance, "planner_guidance", MAX_GUIDANCE_CHARS)
        _text(self.critic_guidance, "critic_guidance", MAX_GUIDANCE_CHARS)
        if not isinstance(self.lessons, tuple) or len(self.lessons) > MAX_LESSONS:
            raise ValueError("lessons must be a bounded tuple")
        if not all(isinstance(item, Experience) for item in self.lessons):
            raise ValueError("lessons must contain Experience objects")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidateProfile":
        data = _fields(payload, {"planner_guidance", "critic_guidance", "lessons"}, "profile")
        lessons = data.get("lessons", [])
        if not isinstance(lessons, (list, tuple)):
            raise ValueError("lessons must be a list")
        return cls(
            planner_guidance=data.get("planner_guidance", ""),
            critic_guidance=data.get("critic_guidance", ""),
            lessons=tuple(Experience.from_dict(item) for item in lessons),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "planner_guidance": self.planner_guidance,
            "critic_guidance": self.critic_guidance,
            "lessons": [item.to_dict() for item in self.lessons],
        }

    @property
    def profile_id(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def guidance_for(self, agent: str) -> str:
        if agent not in AGENTS:
            raise ValueError("Candidate guidance is only supported for planner and critic")
        instruction = self.planner_guidance if agent == "planner" else self.critic_guidance
        parts = [instruction] if instruction else []
        parts.extend(f"经验：{item.text}" for item in self.lessons if item.agent == agent)
        return "\n".join(parts)


def profile_from_state(state: Mapping[str, Any] | None) -> CandidateProfile:
    state = state or {}
    raw = state.get("improvement_profile")
    profile = CandidateProfile() if raw is None else CandidateProfile.from_dict(raw)
    declared_id = state.get("improvement_profile_id")
    if declared_id is not None and declared_id != profile.profile_id:
        raise ValueError("Persisted improvement profile fingerprint mismatch")
    return profile


def current_guidance(agent: str) -> str:
    # Lazy import avoids profiles -> harness -> services import cycles.
    from harness.runtime import current_state

    return profile_from_state(current_state()).guidance_for(agent)
