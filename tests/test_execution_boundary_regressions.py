"""Discriminating host-boundary regressions; no Docker or external services."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from sandbox.executor import DockerSandboxExecutor, SandboxImageInfo, SandboxLogLimitExceeded
from sandbox.protocol import SandboxJob, SandboxLimits
from utils.artifacts import save_html_report
from utils.html_safety import sanitize_report_html
from utils.sql_policy import execute_readonly_sql

IMAGE = SandboxImageInfo('synthetic', 'synthetic-id', None, None)


def test_failed_process_never_reuses_previous_success(tmp_path):
    (tmp_path / 'result.json').write_text(json.dumps({'success': True, 'result': 999}))
    result = DockerSandboxExecutor()._interpret(subprocess.CompletedProcess([], 137, '', ''), tmp_path, IMAGE)
    assert not result.success and result.error_kind == 'oom' and result.result is None


def test_result_requires_current_attempt_and_valid_shape(tmp_path):
    path = tmp_path / 'result.json'
    executor = DockerSandboxExecutor()
    for payload in [{'success': True, 'result': 999, 'attempt_id': 'old'}, {'success': 'false'}, []]:
        path.write_text(json.dumps(payload))
        result = executor._interpret(subprocess.CompletedProcess([], 0, '', ''), tmp_path, IMAGE, expected_attempt='new')
        assert not result.success and result.error_kind == 'invalid_result'


def test_result_byte_limit_is_checked_before_json_decode(tmp_path):
    (tmp_path / 'result.json').write_text(' ' * 17000)
    result = DockerSandboxExecutor()._interpret(subprocess.CompletedProcess([], 0, '', ''), tmp_path, IMAGE, limits=SandboxLimits(max_result_bytes=16))
    assert not result.success and 'byte limit' in result.error


def test_same_logical_operation_has_distinct_attempts(tmp_path, monkeypatch):
    executor = DockerSandboxExecutor()
    monkeypatch.setattr(executor, 'require_ready', lambda: IMAGE)
    attempts = []
    def run(**kwargs):
        job = json.loads((kwargs['in_dir'] / 'job.json').read_text())
        attempts.append(job['attempt_id'])
        (kwargs['scratch_dir'] / 'result.json').write_text(json.dumps({'success': True, 'result': len(attempts), 'attempt_id': job['attempt_id']}))
        return subprocess.CompletedProcess([], 0, '', '')
    monkeypatch.setattr(executor, '_run_container', run)
    job = SandboxJob(mode='python', table_names=[], code='result = 1', op_label='same-operation')
    results = [executor.run(job, run_id='same-run', run_dir=tmp_path) for _ in range(2)]
    assert all(item.success for item in results)
    assert attempts[0] != attempts[1]
    assert [item.result for item in results] == [1, 2]


def test_bounded_process_capture_stops_noisy_child(monkeypatch):
    import sandbox.executor as module
    monkeypatch.setattr(module, 'MAX_PROCESS_OUTPUT_BYTES', 4096)
    executor = DockerSandboxExecutor(docker=sys.executable)
    with pytest.raises(SandboxLogLimitExceeded):
        executor._run_docker(['-B', '-c', 'import sys; sys.stdout.write("x" * 65536); sys.stdout.flush()'], timeout=5)


def test_writable_workspace_has_total_and_count_limits(tmp_path):
    for index in range(3):
        (tmp_path / str(index)).write_bytes(b'x' * 4)
    assert not DockerSandboxExecutor._workspace_within_limits([tmp_path], SandboxLimits(max_output_files=2))
    assert not DockerSandboxExecutor._workspace_within_limits([tmp_path], SandboxLimits(max_workspace_bytes=10))
    assert DockerSandboxExecutor._workspace_within_limits([tmp_path], SandboxLimits(max_workspace_bytes=12))


def test_sql_rejects_large_cell_before_materializing_it(tmp_path):
    path = tmp_path / 'db.sqlite'
    sqlite3.connect(path).close()
    with pytest.raises(sqlite3.DataError):
        execute_readonly_sql(str(path), 'SELECT hex(zeroblob(100000)) AS oversized', allowed_tables=frozenset(), max_bytes=1024)


def test_report_markup_is_passive_and_only_uses_local_pngs(tmp_path):
    path = Path(save_html_report(
        '# Report\n<script>alert(1)</script><svg onload="bad()"></svg>'
        '<img src="https://example.invalid/track" onerror="bad()">'
        '<img src="chart_1.png" onerror="bad()">'
        '<a href="javascript:bad()">bad</a><a href="https://example.invalid">source</a>',
        str(tmp_path / 'report.html'),
    ))
    text = path.read_text(encoding='utf-8')
    assert '<script' not in text and '<svg' not in text and 'onerror' not in text and 'javascript:' not in text
    assert 'src="https:' not in text and 'src="chart_1.png"' in text
    assert 'noopener noreferrer' in text and 'Content-Security-Policy' in text


def test_pdf_inline_only_reads_a_regular_current_report_png(tmp_path):
    (tmp_path / 'chart_1.png').write_bytes(b'\x89PNG\r\n\x1a\n' + b'synthetic')
    result = sanitize_report_html('<img src="chart_1.png"><img src="../outside.png"><img src="file:///secret.png"><style>body{}</style>', image_root=tmp_path)
    assert result.count('<img') == 1 and 'data:image/png;base64,' in result
    assert 'outside' not in result and 'file:' not in result and 'body{}' not in result


def test_expired_lease_cannot_publish_an_artifact(tmp_path, bound_runtime):
    from core.publication import atomic_write
    destination = tmp_path / 'report.md'
    destination.write_text('old report')
    def expired():
        raise RuntimeError('lease expired')
    bound_runtime.assert_lease = expired
    with pytest.raises(RuntimeError, match='lease expired'):
        atomic_write(destination, b'new report')
    assert destination.read_text() == 'old report'


def test_pdf_browser_is_passive_and_has_no_resource_network(tmp_path, monkeypatch):
    import utils.artifacts as artifacts
    import playwright.sync_api
    captured = {}
    class Page:
        def route(self, pattern, callback):
            captured['route'] = pattern
            class Request:
                def abort(self):
                    captured['aborted'] = True
            callback(Request())
        def set_content(self, content, **kwargs):
            captured['html'] = content
        def pdf(self, **kwargs):
            Path(kwargs['path']).write_bytes(b'%PDF-synthetic')
    class Browser:
        def new_page(self, **kwargs):
            captured['options'] = kwargs
            return Page()
        def close(self):
            pass
    class Playwright:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    monkeypatch.setattr(playwright.sync_api, 'sync_playwright', Playwright)
    monkeypatch.setattr(artifacts, 'launch_pdf_browser', lambda _: Browser())
    source = tmp_path / 'report.html'
    source.write_text('<script>bad()</script><img src="https://example.invalid/track"><p>safe</p>')
    target = tmp_path / 'report.pdf'
    assert artifacts.save_pdf_report(str(source), str(target)) == str(target)
    assert captured['options']['java_script_enabled'] is False
    assert captured['options']['service_workers'] == 'block'
    assert captured['route'] == '**/*' and captured['aborted']
    assert '<script' not in captured['html'] and '<img' not in captured['html']
    assert target.read_bytes() == b'%PDF-synthetic'


def test_knowledge_changed_after_report_generation_cannot_be_published(tmp_path):
    import nodes
    from rag.versioning import knowledge_version, KnowledgeSourceChanged
    source = tmp_path / 'rules.md'
    source.write_text('old business rule', encoding='utf-8')
    version = knowledge_version([str(source)])
    source.write_text('new business rule', encoding='utf-8')
    state = {'knowledge_sources': [str(source)], 'knowledge_version': version,
             'run_id': 'test-run', 'report_content': '# Old report'}
    with pytest.raises(KnowledgeSourceChanged, match='新建Run'):
        nodes.artifact_node(state)


def test_report_stage_knowledge_change_stops_before_any_model_call(tmp_path):
    import nodes
    from rag.versioning import knowledge_version
    source = tmp_path / 'rules.md'
    source.write_text('old rule', encoding='utf-8')
    version = knowledge_version([str(source)])
    source.write_text('new rule', encoding='utf-8')
    result = nodes.report_node({'knowledge_sources': [str(source)], 'knowledge_version': version})
    assert result['execution_error_kind'] == 'knowledge_changed'
    assert result['termination_reason'].startswith('knowledge_changed:')


def test_malformed_report_link_is_removed_without_breaking_export():
    assert sanitize_report_html('<a href="https://[">safe text</a>') == '<a>safe text</a>'


def test_untrusted_task_id_cannot_choose_an_artifact_path(tmp_path):
    from utils.result_artifacts import store_result
    _, reference = store_result('x' * 40000, run_id='run_safe', task_id='../outside', output_root=tmp_path)
    path = Path(reference['path'])
    assert path.parent == tmp_path / 'run_safe' / 'results'
    assert path.name.startswith('task-')


def test_dead_local_execution_can_be_fenced_before_lease_expiry(tmp_path, monkeypatch):
    import services.run_store as module
    from services.run_identity import LeaseLost
    store = module.RunStore(tmp_path / 'runs.sqlite')
    store.create('run_dead', data_sources=['data.csv'])
    assert store.claim('run_dead', 'old')
    monkeypatch.setattr(module, '_process_alive', lambda _: False)
    assert store.is_stale_running('run_dead')
    assert store.claim('run_dead', 'new', resume=True)
    with pytest.raises(LeaseLost):
        store.assert_lease('run_dead', 'old')
    store.assert_lease('run_dead', 'new')
