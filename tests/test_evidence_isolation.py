"""测试取证不得改动已跟踪的公开证据文件。

`test_modelgate_success` 在拆除测试栈前会把成功链路的聚合结果写成 JSON。
早期实现固定写到 `docs/evidence/modelgate-success.json`，而那份文件已经进仓库，
于是每次跑测试都会把它改写成带随机 run_id 与耗时的新内容——工作区凭空变脏，
公开证据也被噪声覆盖。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests import test_modelgate_success as success


def test_evidence_path_follows_the_env_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MODELGATE_EVIDENCE_DIR", str(tmp_path))
    assert success._evidence_path() == tmp_path / "modelgate-success.json"


def test_evidence_path_defaults_to_the_tracked_location(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MODELGATE_EVIDENCE_DIR", raising=False)
    assert success._evidence_path() == (
        success.PROJECT_ROOT / "docs" / "evidence" / "modelgate-success.json"
    )


def test_the_public_evidence_file_is_git_tracked() -> None:
    """前提检查：默认落点确实是已跟踪文件，因此才需要跳过覆盖。"""
    assert success._is_tracked_by_git(
        success.PROJECT_ROOT / "docs" / "evidence" / "modelgate-success.json"
    ), "公开证据文件未被 Git 跟踪，这条防护的前提变了"


def test_dump_evidence_does_not_touch_the_public_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """取证走默认落点时必须跳过写入，公开文件内容保持不变。"""
    target = success.PROJECT_ROOT / "docs" / "evidence" / "modelgate-success.json"
    original = target.read_text(encoding="utf-8")

    monkeypatch.delenv("MODELGATE_EVIDENCE_DIR", raising=False)
    monkeypatch.setattr(
        success,
        "_EVIDENCE_RUNS",
        {"evidence-guard-run": ["planner"]},
    )
    monkeypatch.setattr(
        success, "_fetch_run_detail", lambda run_id: {"totals": {}, "by_agent": [], "data_quality": {}}
    )

    success._dump_evidence()

    assert target.read_text(encoding="utf-8") == original, (
        "取证改写了已跟踪的公开证据文件"
    )


def test_dump_evidence_writes_to_an_untracked_location(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """显式指定未跟踪目录时照常落盘，取证能力没有丢。"""
    monkeypatch.setenv("MODELGATE_EVIDENCE_DIR", str(tmp_path))
    monkeypatch.setattr(success, "_EVIDENCE_RUNS", {"evidence-run": ["executor"]})
    monkeypatch.setattr(
        success,
        "_fetch_run_detail",
        lambda run_id: {
            "totals": {"total_tokens": 20},
            "by_agent": [{"agent_name": "executor"}],
            "data_quality": {"missing_usage": 0},
        },
    )

    success._dump_evidence()

    written = tmp_path / "modelgate-success.json"
    assert written.exists(), "显式指定目录时没有落盘"
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["runs"][0]["run_id"] == "evidence-run"
