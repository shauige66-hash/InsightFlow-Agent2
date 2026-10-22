"""Chart planning lineage, partial acceptance and persistent degradation evidence."""
from __future__ import annotations

import json
from pathlib import Path
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from core.runtime import RunRuntime, use_runtime, build_thread_config
from ingestion.resources import ResourceResolver
from planning.visualization import partition_visualization_plan, validate_visualization_plan
from services.analysis_service import read_run_state
from utils.agent_loop import AgentBudget, new_usage
import nodes
import workflow
from tests.scripted_queue import task
from tests.test_runtime_audit_regressions import runtime_case, scripted


def chart(chart_id='chart_ok', source='task_3', **updates):
    return dict(chart_id=chart_id, source_task_id=source, target_tables=['customers.customers'],
        chart_type='bar', description='客户计数比较', expected_output='客户数柱状图', **updates)


def state_for(case, run_id):
    return dict(run_id=run_id, data_sources=[str(case['db'])],
        analysis_plan=[task(f'task_{i}') for i in range(1, 6)], current_task_index=5,
        execution_results=[dict(task_id=f'task_{i}', success=True, result=[{'n': 2}], code='SELECT COUNT(*) AS n FROM customers', tool_used='sql') for i in range(1, 6)],
        failed_tasks=[{'task_id': f'task_{i}', 'stage': 'critic', 'reason': '最终批评不通过'} for i in (1, 2)],
        analysis_status='partial', status_reason='两个分析任务未通过批评', run_status='running',
        completed_operations={}, visualization_results=[], artifacts=[], knowledge_sources=[],
        execution_budget={'limits': AgentBudget().to_dict(), 'usage': new_usage()})


def runtime_for(case, factory):
    return RunRuntime(run_id=factory.run_id, model_factory=factory.build(),
        resolver=ResourceResolver(), output_root=case['root'])


def plan_only(case, replies, *, name='chart-case', limit=None):
    factory = scripted(name, visualizer=[{'charts': value} for value in replies])
    state = state_for(case, name)
    if limit is not None:
        state['execution_budget']['limits']['max_llm_calls'] = limit
    with use_runtime(runtime_for(case, factory)):
        result = nodes.visualization_planner_node(state)
    return result, factory


def test_one_failed_source_cannot_veto_valid_sibling(runtime_case):
    state = state_for(runtime_case, 'partition')
    proposal = [chart('bad', 'task_1'), chart(), chart('other', 'task_4')]
    good, rejected = partition_visualization_plan(proposal, analysis_plan=state['analysis_plan'],
        execution_results=state['execution_results'], failed_tasks=state['failed_tasks'])
    assert [c['chart_id'] for c in good] == ['chart_ok', 'other']
    assert rejected[0]['source_task_id'] == 'task_1' and rejected[0]['stage'] == 'planning'
    with pytest.raises(ValueError, match='task_1.*最终失败'):
        validate_visualization_plan(proposal, analysis_plan=state['analysis_plan'],
            execution_results=state['execution_results'], failed_tasks=state['failed_tasks'])


@pytest.mark.parametrize('bad', [{}, None, chart('unknown','missing'),
    {**chart('bad'), 'target_tables': ['unauthorized']},
    {**chart('bad'), 'chart_type': 'pie'}, {**chart('bad'), 'target_tables': [{}]}])
def test_invalid_structure_and_sources_never_enter_partial_plan(runtime_case, bad):
    result, factory = plan_only(runtime_case, [[bad, chart()]])
    assert result['visualization_plan'] == [chart()]
    assert result['visualization_planning']['reason'] == 'partial_plan'
    assert len(result['failed_charts']) == 1 and factory.calls == ['visualizer']


def test_duplicates_and_extra_charts_are_rejected_individually(runtime_case):
    result, _ = plan_only(runtime_case, [[chart(), chart(), chart('second'), chart('third'), chart('fourth')]])
    assert [c['chart_id'] for c in result['visualization_plan']] == ['chart_ok', 'second', 'third']
    assert len(result['failed_charts']) == 2


def test_two_rejections_then_empty_preserves_failure_and_attempts(runtime_case):
    result, factory = plan_only(runtime_case, [[chart('bad_1','task_1')], [chart('bad_2','task_2')], []])
    diagnostics = result['visualization_planning']
    assert result['visualization_plan'] == []
    assert diagnostics['status'] == 'degraded' and diagnostics['reason'] == 'empty_after_rejections'
    assert diagnostics['attempt_count'] == 3 and diagnostics['accepted_count'] == 0
    assert [a['proposed_count'] for a in diagnostics['attempts']] == [1, 1, 0]
    assert 'task_1' in diagnostics['attempts'][0]['rejected'][0]['error']
    assert 'task_2' in result['failed_charts'][0]['error']
    assert factory.calls == ['visualizer'] * 3


def test_visualizer_only_receives_eligible_sources(runtime_case, monkeypatch):
    captured = []
    original = nodes.create_visualization_plan
    def capture(**kwargs):
        captured.append(kwargs)
        return original(**kwargs)
    monkeypatch.setattr(nodes, 'create_visualization_plan', capture)
    result, _ = plan_only(runtime_case, [[chart('bad','task_1')], []])
    assert result['visualization_planning']['eligible_task_ids'] == ['task_3','task_4','task_5']
    for call in captured:
        assert [x['task_id'] for x in call['analysis_plan']] == ['task_3','task_4','task_5']
        assert [x['task_id'] for x in call['execution_results']] == ['task_3','task_4','task_5']
        assert [x['task_id'] for x in call['failed_tasks']] == ['task_1','task_2']
    assert captured[1]['validation_feedback'][0]['source_task_id'] == 'task_1'


def test_intentional_empty_does_not_spend_two_more_requests(runtime_case):
    result, factory = plan_only(runtime_case, [[]])
    assert result['failed_charts'] == [] and factory.calls == ['visualizer']
    assert result['visualization_planning']['reason'] == 'model_selected_no_charts'
    assert result['visualization_planning']['status'] == 'skipped'


def test_later_valid_plan_retains_history_without_false_final_failure(runtime_case):
    result, factory = plan_only(runtime_case, [[chart('bad','task_1')], [chart()]])
    assert result['visualization_plan'] == [chart()] and not result['failed_charts']
    assert result['visualization_planning']['status'] == 'planned'
    assert result['visualization_planning']['attempt_count'] == 2
    assert result['visualization_planning']['attempts'][0]['rejected']
    assert factory.calls == ['visualizer'] * 2


def test_invalid_candidate_cannot_reserve_a_valid_siblings_chart_id(runtime_case):
    result, _ = plan_only(runtime_case, [[chart('same','task_1'), chart('same')]])
    assert result['visualization_plan'] == [chart('same')]
    assert result['failed_charts'][0]['source_task_id'] == 'task_1'


def test_structured_output_failure_keeps_prior_rejection_history(runtime_case):
    result, factory = plan_only(runtime_case, [[chart('bad','task_1')], 'not a list'])
    assert result['visualization_planning']['reason'] == 'structured_output_error'
    assert result['visualization_planning']['attempts'][0]['rejected']
    assert result['failed_charts'][0]['stage'] == 'planning'
    assert len(factory.calls) == 4  # first plan plus three internal format retries


def test_cached_plan_replay_reconstructs_diagnostics_without_more_requests(runtime_case):
    factory = scripted('replayed-empty', visualizer=[{'charts': [chart('bad','task_1')]}, {'charts': []}])
    state = state_for(runtime_case, factory.run_id)
    with use_runtime(runtime_for(runtime_case, factory)):
        first = nodes.visualization_planner_node(state)
        state.update(first)
        second = nodes.visualization_planner_node(state)
    assert factory.calls == ['visualizer'] * 2
    assert first['visualization_planning'] == second['visualization_planning']
    assert first['failed_charts'] == second['failed_charts']


def test_all_failed_sources_skip_model_without_reading_untrusted_results(runtime_case):
    state = state_for(runtime_case, 'no-sources')
    state['failed_tasks'] = [{'task_id': f'task_{i}'} for i in range(1, 6)]
    factory = scripted('no-sources')
    with use_runtime(runtime_for(runtime_case, factory)):
        result = nodes.visualization_planner_node(state)
    assert factory.calls == [] and result['visualization_planning']['reason'] == 'no_successful_sources'


def test_budget_stop_keeps_already_seen_planning_rejections(runtime_case):
    result, factory = plan_only(runtime_case, [[chart('bad','task_1')]], limit=2)
    assert factory.calls == ['visualizer'] * 2
    assert result['visualization_planning']['status'] == 'failed'
    assert result['visualization_planning']['reason'] == 'budget_exceeded'
    assert result['visualization_planning']['attempts'][0]['rejected']
    assert result['failed_charts'] and result['termination_reason'].startswith('budget_exceeded:')


@pytest.mark.parametrize('mixed', [False, True])
def test_real_graph_persists_diagnostics_in_public_state_and_report(runtime_case, mixed):
    replies = [[chart('bad','task_1'), chart()]] if mixed else [[chart('bad','task_1')], [chart('bad2','task_2')], []]
    factory = scripted('graph-partial' if mixed else 'graph-empty', visualizer=[{'charts': value} for value in replies])
    state = state_for(runtime_case, factory.run_id)
    saver = InMemorySaver()
    graph = workflow.create_workflow(checkpointer=saver)
    config = build_thread_config(factory.run_id)
    with use_runtime(runtime_for(runtime_case, factory)):
        state.update(nodes.loader_node(state))
        state.update(nodes.context_node(state))
        graph.update_state(config, state, as_node='advance_task')
        assert graph.get_state(config).next == ('visualization_planner',)
        result = graph.invoke(None, config)
    public = read_run_state(factory.run_id, checkpointer=saver)
    assert public['visualization_planning'] == result['visualization_planning']
    assert public['failed_charts'] and public['analysis_status'] == 'partial'
    assert '图表规划降级说明' in result['report_content']
    assert '## 未能生成的图表' not in result['report_content']
    assert factory.calls.count('visualizer') == (1 if mixed else 3)
    assert len(result['visualization_results']) == (1 if mixed else 0)
    logs = [json.loads(line) for path in (runtime_case['root']/'logs').glob('*.jsonl') for line in path.read_text(encoding='utf-8').splitlines()]
    created = [event for event in logs if event.get('event') == 'visualization_plan_created'][-1]
    assert created['status'] == 'degraded'
    assert created['details']['chart_count'] == (1 if mixed else 0)
    assert created['details']['planning']['attempt_count'] == (1 if mixed else 3)


def test_real_graph_budget_report_includes_previous_source_rejections(runtime_case):
    factory = scripted('budget-chart-report', visualizer=[{'charts': [chart('bad','task_1')]}])
    state = state_for(runtime_case, factory.run_id)
    state['execution_budget']['limits']['max_llm_calls'] = 2
    saver = InMemorySaver()
    graph = workflow.create_workflow(checkpointer=saver)
    config = build_thread_config(factory.run_id)
    with use_runtime(runtime_for(runtime_case, factory)):
        state.update(nodes.loader_node(state))
        state.update(nodes.context_node(state))
        graph.update_state(config, state, as_node='advance_task')
        result = graph.invoke(None, config)
    assert factory.calls == ['visualizer'] * 2
    assert result['visualization_planning']['reason'] == 'budget_exceeded'
    assert '图表规划降级说明' in result['report_content']
    assert 'task_1 最终失败' in result['report_content']
    assert result['analysis_status'] == 'partial'


def test_checkpoint_resume_after_planning_keeps_diagnostics_without_replanning(runtime_case, monkeypatch):
    class CrashBeforeReport(RuntimeError):
        pass
    def crash(_state):
        raise CrashBeforeReport('report has not started')
    factory = scripted('chart-checkpoint', visualizer=[{'charts': [chart('bad','task_1')]}, {'charts': []}])
    state = state_for(runtime_case, factory.run_id)
    saver = InMemorySaver()
    config = build_thread_config(factory.run_id)
    with use_runtime(runtime_for(runtime_case, factory)):
        state.update(nodes.loader_node(state))
        state.update(nodes.context_node(state))
        with monkeypatch.context() as scoped:
            scoped.setattr(workflow, 'report_node', crash)
            graph = workflow.create_workflow(checkpointer=saver)
            graph.update_state(config, state, as_node='advance_task')
            with pytest.raises(CrashBeforeReport):
                graph.invoke(None, config)
        persisted = graph.get_state(config).values['visualization_planning']
        assert graph.get_state(config).next == ('report',)
    with use_runtime(runtime_for(runtime_case, factory)):
        resumed = workflow.create_workflow(checkpointer=saver).invoke(None, config)
    assert resumed['visualization_planning'] == persisted
    assert factory.calls.count('visualizer') == 2
    assert '图表规划降级说明' in resumed['report_content']
