"""H5 观察与动作策略的契约测试。

不经过图，直接验证两件事：观察由 state 确定性构造且可序列化；动作校验在
各类错误、计数与预算下的结论是确定的。这两条是"模型不能直接控制路由"的
前提。
"""

from __future__ import annotations

import json

import pytest

from agent_loop import (
    CONTINUE,
    DEFAULT_LIMITS,
    FINISH,
    REFINE,
    REPAIR,
    REPLAN_REMAINING,
    RETRIEVE_MORE,
    SKIP_TASK,
    build_observation,
    limits_from,
    validate_action,
)


def base_state(**overrides):
    state = {
        "run_id": "run-1",
        "analysis_plan": [
            {
                "task_id": "task_1",
                "description": "统计客户数量",
                "target_tables": ["customers.customers"],
                "expected_output": "客户数",
            },
            {
                "task_id": "task_2",
                "description": "按地区细分",
                "target_tables": ["customers.customers"],
                "expected_output": "地区分布",
            },
        ],
        "current_task_index": 0,
        "current_tool": "sql",
        "execution_results": [
            {
                "task_id": "task_1",
                "success": True,
                "result": [{"n": 3}],
                "error": None,
            }
        ],
        "validation_passed": True,
        "critic_feedback": "ok",
        "execution_budget": {"limits": {}, "usage": {"llm_calls": 2}},
        "task_knowledge": {
            "task_1": [
                {
                    "chunk_id": "c1",
                    "source": "vip.md",
                    "page": None,
                    "score": 0.7,
                    "content": "VIP 定义",
                }
            ]
        },
    }
    state.update(overrides)
    return state


class TestObservation:
    def test_the_same_state_yields_the_same_observation(self) -> None:
        first = build_observation(base_state())
        second = build_observation(base_state())

        assert first.to_dict() == second.to_dict()
        assert first.fingerprint() == second.fingerprint()

    def test_the_observation_is_json_serializable(self) -> None:
        observation = build_observation(base_state())
        payload = json.dumps(observation.to_dict(), ensure_ascii=False)

        assert "task_1" in payload
        # 允许出现资源名（它是身份），但不允许出现结果数据本身
        assert '"n": 3' not in payload
        assert "result" not in observation.to_dict().get("result_shape", {})

    def test_only_the_result_shape_is_recorded(self) -> None:
        observation = build_observation(base_state())

        assert observation.result_shape["kind"] == "table"
        assert observation.result_shape["rows"] == 1
        assert observation.result_shape["columns"] == ["n"]
        # 不包含任何一行数据
        assert "3" not in json.dumps(observation.result_shape)

    def test_error_and_feedback_are_truncated(self) -> None:
        observation = build_observation(
            base_state(
                execution_results=[],
                execution_error="x" * 2000,
                execution_error_kind="python_error",
                critic_feedback="y" * 2000,
                validation_passed=False,
            )
        )

        assert observation.error_summary is not None
        assert len(observation.error_summary) <= 401
        assert observation.critic_feedback is not None
        assert len(observation.critic_feedback) <= 401

    def test_remaining_tasks_are_bounded(self) -> None:
        plan = [
            {
                "task_id": f"task_{index}",
                "description": "d",
                "target_tables": ["customers.customers"],
                "expected_output": "e",
            }
            for index in range(1, 12)
        ]
        observation = build_observation(
            base_state(analysis_plan=plan, current_task_index=0)
        )

        assert 0 < len(observation.remaining_tasks) <= 5

    def test_evidence_sufficiency_reflects_real_evidence(self) -> None:
        with_evidence = build_observation(base_state())
        without = build_observation(base_state(task_knowledge={}))

        assert with_evidence.evidence_sufficient is True
        assert without.evidence_sufficient is False

    def test_a_changed_observation_changes_the_fingerprint(self) -> None:
        first = build_observation(base_state())
        second = build_observation(
            base_state(validation_passed=False, critic_feedback="结果缺少细分")
        )

        assert first.fingerprint() != second.fingerprint()

    def test_budget_snapshot_is_present(self) -> None:
        observation = build_observation(base_state())
        assert isinstance(observation.remaining_budget, dict)


def propose(action: str) -> dict:
    return {"action": action, "reason": "test"}


class TestActionPolicy:
    def test_a_passing_task_continues(self) -> None:
        decision = validate_action(
            build_observation(base_state()),
            propose(CONTINUE),
            state=base_state(),
            limits=DEFAULT_LIMITS,
        )
        assert decision.action == CONTINUE
        assert decision.forced is False

    def test_a_failing_task_cannot_continue(self) -> None:
        state = base_state(validation_passed=False, critic_feedback="缺细分")
        decision = validate_action(
            build_observation(state), propose(CONTINUE), state=state, limits=DEFAULT_LIMITS
        )
        assert decision.action == SKIP_TASK
        assert decision.forced is True

    def test_a_repairable_error_allows_repair(self) -> None:
        state = base_state(
            validation_passed=None,
            # 真实链路里 tool_execution 一定追加一条失败的执行记录
            execution_results=[
                {
                    "task_id": "task_1",
                    "success": False,
                    "result": None,
                    "error": "syntax error",
                }
            ],
            execution_error="syntax error",
            execution_error_kind="python_error",
        )
        decision = validate_action(
            build_observation(state), propose(REPAIR), state=state, limits=DEFAULT_LIMITS
        )
        assert decision.action == REPAIR

    @pytest.mark.parametrize(
        "kind",
        ["sql_permission", "budget_exceeded", "unavailable", "authorization_denied"],
    )
    def test_terminal_errors_never_repair(self, kind: str) -> None:
        state = base_state(
            validation_passed=None,
            execution_results=[],
            execution_error="denied",
            execution_error_kind=kind,
        )
        for action in (REPAIR, REFINE, RETRIEVE_MORE, CONTINUE):
            decision = validate_action(
                build_observation(state), propose(action), state=state, limits=DEFAULT_LIMITS
            )
            assert decision.action == SKIP_TASK, action
            assert decision.forced is True

    def test_quality_feedback_allows_refine(self) -> None:
        state = base_state(validation_passed=False, critic_feedback="缺少细分")
        decision = validate_action(
            build_observation(state), propose(REFINE), state=state, limits=DEFAULT_LIMITS
        )
        assert decision.action == REFINE

    def test_refine_without_feedback_is_downgraded(self) -> None:
        state = base_state()
        decision = validate_action(
            build_observation(state), propose(REFINE), state=state, limits=DEFAULT_LIMITS
        )
        assert decision.action == CONTINUE
        assert decision.forced is True

    def test_extra_retrieval_requires_a_knowledge_base(self) -> None:
        state = base_state(
            validation_passed=False,
            critic_feedback="缺依据",
            task_knowledge={},
        )
        decision = validate_action(
            build_observation(state),
            propose(RETRIEVE_MORE),
            state=state,
            limits=DEFAULT_LIMITS,
        )
        assert decision.action == REFINE
        assert decision.forced is True

    def test_extra_retrieval_is_capped(self) -> None:
        state = base_state(
            knowledge_sources=["kb.md"],
            validation_passed=False,
            critic_feedback="缺依据",
            task_knowledge={},
            extra_retrievals=2,
        )
        decision = validate_action(
            build_observation(state),
            propose(RETRIEVE_MORE),
            state=state,
            limits=DEFAULT_LIMITS,
        )
        assert decision.action == REFINE
        assert "上限" in decision.reason

    def test_replan_is_capped(self) -> None:
        state = base_state(
            validation_passed=False,
            critic_feedback="无法完成",
            runtime_replans=2,
        )
        decision = validate_action(
            build_observation(state),
            propose(REPLAN_REMAINING),
            state=state,
            limits=DEFAULT_LIMITS,
        )
        assert decision.action == SKIP_TASK
        assert "上限" in decision.reason

    def test_repeated_actions_are_denied(self) -> None:
        state = base_state(
            validation_passed=False,
            critic_feedback="还是不行",
            action_counts={"task_1": {"refine": 2}},
        )
        decision = validate_action(
            build_observation(state), propose(REFINE), state=state, limits=DEFAULT_LIMITS
        )
        assert decision.action == SKIP_TASK
        assert decision.forced is True

    def test_the_step_limit_forces_finish(self) -> None:
        state = base_state(loop_step=8)
        decision = validate_action(
            build_observation(state), propose(CONTINUE), state=state, limits=DEFAULT_LIMITS
        )
        assert decision.action == FINISH
        assert decision.forced is True

    def test_exhausted_budget_forces_finish(self) -> None:
        state = base_state(
            execution_budget={"limits": {"max_llm_calls": 5}, "usage": {"llm_calls": 5}}
        )
        decision = validate_action(
            build_observation(state), propose(CONTINUE), state=state, limits=DEFAULT_LIMITS
        )
        assert decision.action == FINISH
        assert "预算" in decision.reason

    def test_an_unknown_action_ends_safely(self) -> None:
        decision = validate_action(
            build_observation(base_state()),
            {"action": "rm -rf"},
            state=base_state(),
            limits=DEFAULT_LIMITS,
        )
        assert decision.action == FINISH
        assert decision.forced is True
        assert decision.proposed == "rm -rf"

    def test_missing_action_ends_safely(self) -> None:
        decision = validate_action(
            build_observation(base_state()), {}, state=base_state(), limits=DEFAULT_LIMITS
        )
        assert decision.action == FINISH

    def test_finish_is_always_allowed(self) -> None:
        state = base_state(validation_passed=False, critic_feedback="x")
        decision = validate_action(
            build_observation(state), propose(FINISH), state=state, limits=DEFAULT_LIMITS
        )
        assert decision.action == FINISH
        assert decision.forced is False


class TestLimits:
    def test_defaults_are_bounded(self) -> None:
        assert DEFAULT_LIMITS["max_decision_steps"] == 8
        assert DEFAULT_LIMITS["max_runtime_replans"] == 2
        assert DEFAULT_LIMITS["max_extra_retrievals"] == 2
        assert DEFAULT_LIMITS["max_repeated_action_per_task"] == 2

    def test_invalid_overrides_are_ignored(self) -> None:
        limits = limits_from(
            {"max_decision_steps": "abc", "max_runtime_replans": 0, "unknown": 9}
        )

        assert limits["max_decision_steps"] == 8
        assert limits["max_runtime_replans"] == 2
        assert "unknown" not in limits

    def test_a_positive_override_is_applied(self) -> None:
        assert limits_from({"max_decision_steps": 3})["max_decision_steps"] == 3
