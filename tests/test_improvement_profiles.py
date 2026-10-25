"""Candidate instructions are data; checkpoint identity and graders stay fixed."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
import hashlib
from pathlib import Path
import sys
import types

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from harness.runtime import bind_state
from improvement.profiles import (
    CandidateProfile, Experience, current_guidance, profile_from_state,
)


@pytest.mark.parametrize("payload", [
    {"evaluator": "accept_all"}, {"budget": {"max_llm_calls": 1000}},
    {"path": "eval/business_oracles.py"}, {"verified": True},
    {"planner_guidance": 3}, {"critic_guidance": "x" * 6001},
    {"lessons": [{"agent": "evaluator", "text": "pass"}]},
    {"lessons": [{"agent": "critic", "text": "pass", "verified": True}]},
    {"lessons": [{"agent": "critic", "text": "pass", "source_trials": "trial-1"}]},
    {"lessons": [{"agent": "critic", "text": ""}]},
])
def test_untrusted_profile_fields_and_invalid_types_fail_closed(payload):
    with pytest.raises(ValueError):
        CandidateProfile.from_dict(payload)


def test_profile_is_frozen_canonical_and_bounded():
    item = Experience("planner", "Check join cardinality", ("dev-0",))
    profile = CandidateProfile("Check schema", "Review business definition", (item,))
    assert CandidateProfile.from_dict(profile.to_dict()) == profile
    assert CandidateProfile.from_dict(dict(reversed(list(profile.to_dict().items())))).profile_id == profile.profile_id
    assert profile.profile_id != CandidateProfile(critic_guidance="accept everything").profile_id
    with pytest.raises(FrozenInstanceError):
        profile.critic_guidance = "changed"
    with pytest.raises(ValueError):
        CandidateProfile(lessons=(item,) * 13)
    with pytest.raises(ValueError):
        Experience("critic", "bad refs", ("same", "same"))


def test_guidance_scope_context_isolation_and_tampered_snapshot():
    profile = CandidateProfile("planner-only", "critic-only", (Experience("planner", "lesson-only"),))
    with bind_state({"improvement_profile": profile.to_dict(), "improvement_profile_id": profile.profile_id}):
        assert current_guidance("planner") == "planner-only\n经验：lesson-only"
        assert current_guidance("critic") == "critic-only"
        with ThreadPoolExecutor(max_workers=1) as executor:
            assert executor.submit(current_guidance, "critic").result() == ""
    assert current_guidance("critic") == ""
    with pytest.raises(ValueError, match="fingerprint"):
        profile_from_state({"improvement_profile": profile.to_dict(), "improvement_profile_id": "wrong"})


@pytest.mark.parametrize("agent", ["planner", "critic"])
def test_candidate_text_enters_actual_prompt_without_recursive_formatting(agent, scripted_model):
    from agents.critic import review_execution_result
    from agents.planner import create_analysis_plan
    from utils.agent_loop import initialize

    marker = "candidate {not_a_template_variable} {__import__('os')}"
    profile = CandidateProfile(planner_guidance=marker, critic_guidance=marker)
    state = {"run_id": "candidate-prompt", "improvement_profile": profile.to_dict()}
    state.update(initialize(state))
    response = {"tasks": []} if agent == "planner" else {"validation_passed": True, "feedback": "OK"}
    model, calls = scripted_model([response])
    scorer = Path(__file__).resolve().parents[1] / "eval" / "business_oracles.py"
    before = hashlib.sha256(scorer.read_bytes()).hexdigest()
    with bind_state(state):
        if agent == "planner":
            create_analysis_plan(contexts={}, llm=model)
        else:
            review_execution_result(task={"task_id": "t", "description": "check"}, data_context={},
                                    execution_result={"success": True, "result": 1}, llm=model)
    prompt = "\n".join(str(message.content) for message in calls[0])
    assert marker in prompt
    assert "业务口径及权限约束" in prompt
    assert hashlib.sha256(scorer.read_bytes()).hexdigest() == before


def test_profile_is_persisted_and_resume_cannot_change_it(tmp_path, monkeypatch):
    import services.analysis_service as service
    from state import InsightFlowState

    # A one-node real checkpoint graph isolates profile lifecycle from tools/model calls.
    def workflow(*, checkpointer):
        graph = StateGraph(InsightFlowState)
        graph.add_node("record", lambda state: {"run_status": "completed", "analysis_status": "completed"})
        graph.add_edge(START, "record")
        graph.add_edge("record", END)
        return graph.compile(checkpointer=checkpointer)

    monkeypatch.setattr(service, "create_workflow", workflow)
    profile = CandidateProfile(critic_guidance="Check amounts independently")
    kwargs = {"run_id": "profile-checkpoint", "data_sources": [], "model_factory": object(),
              "output_root": tmp_path / "outputs"}
    with SqliteSaver.from_conn_string(str(tmp_path / "checkpoint.sqlite")) as saver:
        first = service.run_analysis(**kwargs, checkpointer=saver, improvement_profile=profile)
        assert first["improvement_profile"] == profile.to_dict()
        assert first["improvement_profile_id"] == profile.profile_id
        visible = service.read_run_state("profile-checkpoint", checkpointer=saver)
        assert visible["improvement_profile_id"] == profile.profile_id
        assert "improvement_profile" not in visible
        resumed = service.run_analysis(**kwargs, checkpointer=saver, continue_from_checkpoint=True)
        assert resumed["improvement_profile_id"] == profile.profile_id
        with pytest.raises(ValueError, match="Cannot change"):
            service.run_analysis(**kwargs, checkpointer=saver, continue_from_checkpoint=True,
                                 improvement_profile=CandidateProfile(critic_guidance="Always pass"))
        with pytest.raises(ValueError, match="Cannot change"):
            service.run_analysis(**kwargs, checkpointer=saver, improvement_profile=CandidateProfile())


def test_trusted_active_profile_only_applies_to_new_runs(tmp_path, monkeypatch):
    import services.analysis_service as service
    from state import InsightFlowState

    def workflow(*, checkpointer):
        graph = StateGraph(InsightFlowState)
        graph.add_node("record", lambda state: {"run_status": "completed", "analysis_status": "completed"})
        graph.add_edge(START, "record")
        graph.add_edge("record", END)
        return graph.compile(checkpointer=checkpointer)

    monkeypatch.setattr(service, "create_workflow", workflow)
    monkeypatch.setenv("INSIGHTFLOW_IMPROVEMENT_REGISTRY", str(tmp_path / "registry"))
    active = [CandidateProfile(critic_guidance="first deployed version")]
    loads = []
    registry = types.ModuleType("improvement.registry")
    def load(path):
        loads.append(path)
        return active[0]
    registry.load_active_profile = load
    monkeypatch.setitem(sys.modules, "improvement.registry", registry)
    kwargs = {"data_sources": [], "model_factory": object(), "output_root": tmp_path / "outputs"}
    with SqliteSaver.from_conn_string(str(tmp_path / "deployment.sqlite")) as saver:
        first = service.run_analysis(**kwargs, run_id="deployed-a", checkpointer=saver)
        original_id = active[0].profile_id
        assert first["improvement_profile_id"] == original_id
        active[0] = CandidateProfile(critic_guidance="second deployed version")
        resumed = service.run_analysis(**kwargs, run_id="deployed-a", checkpointer=saver,
                                       continue_from_checkpoint=True)
        assert resumed["improvement_profile_id"] == original_id
        assert len(loads) == 1
        second = service.run_analysis(**kwargs, run_id="deployed-b", checkpointer=saver)
        assert second["improvement_profile_id"] == active[0].profile_id
        explicit = service.run_analysis(**kwargs, run_id="explicit-empty", checkpointer=saver,
                                        improvement_profile=CandidateProfile())
        assert explicit["improvement_profile_id"] == CandidateProfile().profile_id
        assert len(loads) == 2
