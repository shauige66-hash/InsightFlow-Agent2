"""Regression cases for the audited runtime paths; no network or containers."""
from __future__ import annotations

import base64
from pathlib import Path
import sqlite3

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import nodes
import workflow
import storage
from storage.config import StorageConfig
from services.analysis_service import run_analysis
from tests.scripted_queue import ScriptedQueueFactory, task, decision
from utils.agent_loop import usage_of
from utils import llm_ledger


class AuditFault(RuntimeError):
    pass


@pytest.fixture
def runtime_case(tmp_path, monkeypatch):
    storage.reset()
    storage.configure(StorageConfig(mode="local", local_db_path=str(tmp_path / "storage.sqlite"), owner="audit-runtime"))
    db = tmp_path / "customers.sqlite"
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE customers (id INTEGER PRIMARY KEY, region TEXT)")
        connection.executemany("INSERT INTO customers VALUES (?, ?)", [(1, "East"), (2, "West")])
    connection.close()
    calls = []

    def invoke(name, tables, arguments, *, snapshots=None):
        calls.append((name, dict(arguments)))
        text = arguments.get("query", arguments.get("code", ""))
        if "BROKEN" in text:
            return {"success": False, "result": None, "error": "scripted invalid code", "error_kind": "python_error"}
        if name == "plot":
            from core.runtime import current_runtime
            path = current_runtime().output_dir() / (arguments["chart_id"] + ".png")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l9sAAAAASUVORK5CYII="))
            return {"success": True, "result": str(path), "error": None, "error_kind": None}
        return {"success": True, "result": [{"n": 2}], "error": None, "error_kind": None}

    import harness.tool_dispatcher as dispatcher
    monkeypatch.setattr(dispatcher, "invoke_tool", invoke)
    monkeypatch.setattr(nodes, "retrieve_global", lambda _: {"retrieved_knowledge": [], "retrieved_knowledge_details": []})
    monkeypatch.setattr(nodes, "task_knowledge_updates", lambda *_args, **_kwargs: ([], {}))
    monkeypatch.setattr(nodes, "knowledge_contents", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(nodes, "save_pdf_report", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("offline test: no browser")))
    yield {"db": db, "root": tmp_path, "calls": calls}
    storage.reset()


def scripted(run_id, **overrides):
    replies = {
        "planner": [{"tasks": [task("task_1")]}],
        "executor": [{"code": "SELECT COUNT(*) AS n FROM customers"}],
        "repair": [{"code": "SELECT COUNT(*) AS n FROM customers"}],
        "refine": [{"code": "SELECT COUNT(*) AS n FROM customers"}],
        "critic": [{"validation_passed": True, "feedback": "ok"}],
        "visualizer": [{"charts": []}],
        "plotter": [{"code": "draw_valid_plot()"}],
        "plot_repair": [{"code": "draw_valid_plot()"}],
        "reporter": ["# 合成报告\n\n客户数为2。"],
        "decider": [decision("continue")],
    }
    replies.update(overrides)
    return ScriptedQueueFactory(run_id, replies)


def run(case, factory, **kwargs):
    return run_analysis(data_sources=[str(case["db"])], run_id=factory.run_id, model_factory=factory.build(), output_root=case["root"], **kwargs)


def chart():
    return {"chart_id": "chart_1", "source_task_id": "task_1", "target_tables": ["customers.customers"], "chart_type": "bar", "description": "客户数", "expected_output": "柱状图"}


@pytest.mark.parametrize("limit", range(8))
def test_every_llm_budget_boundary_has_a_controlled_terminal(runtime_case, limit):
    factory = scripted(f"audit-budget-{limit}", visualizer=[{"charts": [chart()]}])
    result = run(runtime_case, factory, budget={"max_llm_calls": limit})
    assert result["run_status"] in ("completed", "failed")
    assert usage_of(result)["llm_calls"] <= limit
    assert len(factory.calls) <= limit


def test_round_budget_counts_real_attempts(runtime_case):
    factory = scripted("audit-rounds", executor=[{"code": "BROKEN"}])
    result = run(runtime_case, factory, budget={"max_rounds": 1})
    assert usage_of(result)["rounds"] == 1
    assert len(runtime_case["calls"]) == 1
    assert "budget_exceeded:" in result["termination_reason"]


def test_plot_repair_executes_its_new_code(runtime_case):
    factory = scripted("audit-plot", visualizer=[{"charts": [chart()]}], plotter=[{"code": "BROKEN"}])
    result = run(runtime_case, factory)
    plots = [call for call in runtime_case["calls"] if call[0] == "plot"]
    assert len(plots) == 2
    assert plots[1][1]["code"] == "draw_valid_plot()"
    assert result["visualization_results"][-1]["success"] is True


def test_replan_reusing_current_id_has_new_model_identity(runtime_case):
    revised = task("task_1", description="a different revised analysis")
    factory = scripted("audit-replan", planner=[{"tasks": [task("task_1")]}, {"tasks": [revised]}], executor=[{"code": "BROKEN"}, {"code": "SELECT 2 AS n"}], decider=[decision("replan_remaining"), decision("continue")])
    result = run(runtime_case, factory, enable_dynamic_loop=True)
    assert result["plan_revision"] == 1
    assert factory.calls.count("executor") == 2
    assert result["execution_results"][-1]["code"] == "SELECT 2 AS n"


def test_resume_keeps_configuration_and_next_node(runtime_case, monkeypatch):
    saver = InMemorySaver()
    factory = scripted("audit-continue")
    def crash(_state):
        raise AuditFault("stop after executor checkpoint")
    with monkeypatch.context() as scoped:
        scoped.setattr(workflow, "tool_execution_node", crash)
        with pytest.raises(AuditFault):
            run(runtime_case, factory, checkpointer=saver, enable_dynamic_loop=True, knowledge_sources=["synthetic-rules"], budget={"max_llm_calls": 9})
    assert factory.calls == ["planner", "executor"]
    result = run(runtime_case, factory, checkpointer=saver, continue_from_checkpoint=True)
    assert factory.calls.count("planner") == 1 and factory.calls.count("executor") == 1
    assert result["enable_dynamic_loop"] is True
    assert result["knowledge_sources"] == ["synthetic-rules"]
    assert result["execution_budget"]["limits"]["max_llm_calls"] == 9


def test_approval_checkpoint_is_running_and_can_resume(runtime_case, monkeypatch):
    saver = InMemorySaver()
    factory = scripted("audit-approval")
    paused = run(runtime_case, factory, checkpointer=saver, require_plan_approval=True)
    assert paused.get("__interrupt__")
    def crash(_state):
        raise AuditFault("stop after approval checkpoint")
    with monkeypatch.context() as scoped:
        scoped.setattr(workflow, "router_node", crash)
        with pytest.raises(AuditFault):
            run(runtime_case, factory, checkpointer=saver, resume=Command(resume={"decision": "approve"}))
    graph = workflow.create_workflow(checkpointer=saver)
    config = {"configurable": {"thread_id": factory.run_id}}
    snapshot = graph.get_state(config)
    assert snapshot.values["run_status"] == "running" and snapshot.next == ("router",)
    # Explicitly exercise compatibility with the old over-eager completed flag.
    graph.update_state(config, {"run_status": "completed"}, as_node="approval_gate")
    result = run(runtime_case, factory, checkpointer=saver, continue_from_checkpoint=True)
    assert result["run_status"] == "completed" and result["artifacts"]
    assert factory.calls.count("planner") == 1


def test_request_reservation_survives_node_commit_loss(runtime_case, monkeypatch):
    saver = InMemorySaver()
    factory = scripted("audit-request-cap")
    original = workflow.planner_node
    def after_response(state):
        original(state)
        raise AuditFault("response received, node delta not committed")
    with monkeypatch.context() as scoped:
        scoped.setattr(workflow, "planner_node", after_response)
        with pytest.raises(AuditFault):
            run(runtime_case, factory, checkpointer=saver, budget={"max_llm_calls": 1})
    assert factory.calls == ["planner"]
    before = workflow.create_workflow(checkpointer=saver).get_state({"configurable": {"thread_id": factory.run_id}})
    assert usage_of(before.values)["llm_calls"] == 0
    result = run(runtime_case, factory, checkpointer=saver, continue_from_checkpoint=True)
    assert factory.calls == ["planner"]
    assert usage_of(result)["llm_calls"] == 1
    assert llm_ledger.request_count(factory.run_id, runtime_case["root"]) == 1
    assert result["run_status"] == "failed"


@pytest.mark.parametrize("damage", ["missing", "changed"])
@pytest.mark.parametrize("replay_decision", [None, "skip"])
def test_recovery_rejects_changed_sources_before_replay_or_terminal_write(runtime_case, monkeypatch, damage, replay_decision):
    from ingestion.resources import ResourceUnavailableError
    saver = InMemorySaver()
    factory = scripted(f"audit-source-{damage}-{replay_decision}")
    def crash(**_kwargs):
        raise AuditFault("executor operation started before its result was committed")
    with monkeypatch.context() as scoped:
        scoped.setattr(nodes, "generate_execution_code", crash)
        with pytest.raises(AuditFault):
            run(runtime_case, factory, checkpointer=saver)
    config = {"configurable": {"thread_id": factory.run_id}}
    graph = workflow.create_workflow(checkpointer=saver)
    before = graph.get_state(config)
    assert before.next == ("execution",)
    assert llm_ledger.summarize(factory.run_id, runtime_case["root"])["unconfirmed_operations"]
    checkpoint_id = before.config["configurable"]["checkpoint_id"]
    calls_before = list(factory.calls)
    if damage == "missing":
        runtime_case["db"].unlink()
    else:
        with sqlite3.connect(runtime_case["db"]) as connection:
            connection.execute("INSERT INTO customers VALUES (3, 'New')")
    resume = {"resume_value": {"llm_decision": replay_decision}} if replay_decision else {"continue_from_checkpoint": True}
    with pytest.raises(ResourceUnavailableError):
        run(runtime_case, factory, checkpointer=saver, **resume)
    after = graph.get_state(config)
    assert factory.calls == calls_before
    assert after.config["configurable"]["checkpoint_id"] == checkpoint_id
    assert after.next == ("execution",)


@pytest.mark.parametrize("path", ["pre_skip", "node_abandoned"])
def test_expired_lease_cannot_write_an_abandoned_checkpoint(runtime_case, monkeypatch, path):
    from utils.idempotency import LlmCallAbandoned
    saver = InMemorySaver()
    factory = scripted("audit-expired-" + path)
    valid = {"lease": True}
    def assert_lease():
        if not valid["lease"]:
            raise RuntimeError("expired execution lease")
    config = {"configurable": {"thread_id": factory.run_id}}
    graph = workflow.create_workflow(checkpointer=saver)
    if path == "pre_skip":
        paused = run(runtime_case, factory, checkpointer=saver, require_plan_approval=True)
        assert paused.get("__interrupt__")
        before = graph.get_state(config).config["configurable"]["checkpoint_id"]
        valid["lease"] = False
        with pytest.raises(RuntimeError, match="expired execution lease"):
            run(runtime_case, factory, checkpointer=saver, resume_value={"llm_decision": "skip"}, assert_lease=assert_lease)
        assert graph.get_state(config).config["configurable"]["checkpoint_id"] == before
    else:
        def abandoned(_state):
            valid["lease"] = False
            raise LlmCallAbandoned("synthetic local abandonment")
        monkeypatch.setattr(workflow, "planner_node", abandoned)
        with pytest.raises(RuntimeError, match="expired execution lease"):
            run(runtime_case, factory, checkpointer=saver, assert_lease=assert_lease)
        after = graph.get_state(config)
        assert after.values["run_status"] == "running"
        assert after.next == ("planner",)


def test_legacy_ledger_and_a_second_unconfirmed_attempt_are_conservative(tmp_path):
    run_id = "audit-old-ledger"
    old = {"run_id": run_id, "operation_id": "op", "node": "executor", "attempt": 1}
    llm_ledger.append({**old, "phase": "started"}, root=tmp_path)
    llm_ledger.append({**old, "phase": "completed"}, root=tmp_path)
    assert llm_ledger.request_count(run_id, tmp_path) == 1
    llm_ledger.record_started(**old, root=tmp_path)
    assert llm_ledger.summarize(run_id, tmp_path)["unconfirmed_operations"] == ["op"]


def test_failed_dependency_prevents_downstream_model_and_tool(runtime_case):
    dependent = {**task("task_2"), "depends_on": ["task_1"]}
    factory = scripted("audit-dependency", planner=[{"tasks": [task("task_1"), dependent]}], executor=[{"code": "BROKEN"}], repair=[{"code": "BROKEN"}])
    result = run(runtime_case, factory)
    assert factory.calls.count("executor") == 1
    assert len(runtime_case["calls"]) == 4
    assert any(item["task_id"] == "task_2" and item.get("error_kind") == "dependency_failed" for item in result["failed_tasks"])


def test_bad_chart_plan_is_bounded_and_degrades_to_report(runtime_case):
    factory = scripted("audit-chart-plan", visualizer=[{"charts": [{}]}])
    result = run(runtime_case, factory)
    assert factory.calls.count("visualizer") == 3
    assert result["visualization_plan"] == []
    assert result["failed_charts"] and result["run_status"] == "completed"


def test_knowledge_change_after_success_preserves_partial_through_publish_approval(runtime_case, monkeypatch):
    """All tasks may succeed while a changed business premise forbids success."""
    from rag.versioning import knowledge_version
    rules = runtime_case["root"] / "rules.md"
    rules.write_text("# Rule A\n\nThe initial business definition.", encoding="utf-8")
    version = knowledge_version([str(rules)])
    monkeypatch.setattr(nodes, "retrieve_global", lambda _state: {"retrieved_knowledge": [], "retrieved_knowledge_details": [], "knowledge_version": version})
    original = nodes.create_visualization_plan
    def change_after_tasks(**kwargs):
        plan = original(**kwargs)
        rules.write_text("# Rule B\n\nA changed business definition.", encoding="utf-8")
        return plan
    monkeypatch.setattr(nodes, "create_visualization_plan", change_after_tasks)
    factory = scripted("audit-knowledge-publish")
    saver = InMemorySaver()
    paused = run(runtime_case, factory, knowledge_sources=[str(rules)], checkpointer=saver, require_report_approval=True)
    assert paused.get("__interrupt__")
    assert paused["analysis_status"] == "partial"
    assert not paused.get("failed_tasks"), "The failure is a changed premise, not a failed data task."
    reason = paused["status_reason"]
    assert "知识源已变化" in reason
    result = run(runtime_case, factory, checkpointer=saver, resume=Command(resume={"decision": "approve"}))
    assert result["analysis_status"] == "partial"
    assert result["status_reason"] == reason
    assert "知识源已变化" in result["report_content"]
    assert factory.calls.count("reporter") == 0


def test_each_task_can_request_its_own_extra_retrieval(runtime_case):
    factory = scripted("audit-task-retrieval", planner=[{"tasks": [task("task_1"), task("task_2")]}], critic=[{"validation_passed": False, "feedback": "need evidence"}, {"validation_passed": True, "feedback": "ok"}, {"validation_passed": False, "feedback": "need evidence"}, {"validation_passed": True, "feedback": "ok"}], decider=[decision("retrieve_more"), decision("continue"), decision("retrieve_more"), decision("continue")])
    result = run(runtime_case, factory, enable_dynamic_loop=True, knowledge_sources=["synthetic-rules"], dynamic_limits={"max_extra_retrievals": 1})
    assert [item["task_id"] for item in result["decision_history"] if item["action"] == "retrieve_more"] == ["task_1", "task_2"]


def test_main_explicitly_injects_a_model_factory(monkeypatch, capsys):
    import main
    marker = object()
    captured = {}
    monkeypatch.setattr(main, "build_model_factory", lambda _run_id: marker)
    monkeypatch.setattr(main, "run_analysis", lambda **kw: captured.update(kw) or {})
    main.main()
    assert captured["model_factory"] is marker and captured["run_id"]
    assert all(Path(path).is_absolute() for path in captured["data_sources"])
