"""H2：任务级上下文选择。

分两层验证：

- 纯函数：按任务选表、拒绝未知资源、样本裁剪、历史摘要、报告字段；
- 真实工作流：五个表的确定性数据源，确认 Planner 看到全部、执行类 Agent
  只看到自己任务的表，且未知表不会进入执行。
"""

from __future__ import annotations

import copy
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

import pytest

import harness
from core.runtime import RunRuntime, use_runtime
from harness.context_manager import (
    MAX_SAMPLE_ROWS,
    UnknownTableError,
    build_context,
    select_history,
    select_tables,
)
from utils.context_budget import context_report_of


def table_context(rows: int = 10) -> dict[str, Any]:
    return {
        "schema": {"columns": ["id", "amount"], "types": {"id": "INTEGER"}},
        "row_count": 100,
        "column_count": 2,
        "context_level": "sampled",
        "data": [{"id": index, "amount": index} for index in range(rows)],
    }


def make_state(*names: str, rows: int = 10) -> dict[str, Any]:
    return {
        "run_id": "ctx-unit",
        "data_context": {name: table_context(rows) for name in names},
        "table_snapshot": {name: {"name": name} for name in names},
        "analysis_plan": [
            {"task_id": "task_1", "target_tables": list(names[:1]) or ["x"]}
        ],
        "current_task_index": 0,
        "execution_results": [],
    }


@pytest.fixture(autouse=True)
def isolated_trace(tmp_path: Path):
    output_root = tmp_path / "outputs"
    with use_runtime(
        RunRuntime(
            run_id="ctx-unit",
            model_factory=None,  # type: ignore[arg-type]
            resolver=None,  # type: ignore[arg-type]
            output_root=output_root,
        )
    ):
        yield output_root


class TestTableSelection:
    def test_only_the_requested_tables_are_selected(self) -> None:
        state = make_state("sales.orders", "sales.customers", "sales.products")

        selection = select_tables(state, ["sales.orders"])

        assert list(selection.tables) == ["sales.orders"]
        assert selection.kept == ("sales.orders",)
        assert selection.excluded == ("sales.customers", "sales.products")

    def test_an_unknown_table_is_rejected_not_skipped(self) -> None:
        state = make_state("sales.orders")

        with pytest.raises(UnknownTableError) as excinfo:
            select_tables(state, ["sales.orders", "sales.missing"])

        assert excinfo.value.missing == ("sales.missing",)
        assert "sales.orders" in excinfo.value.available
        assert "sales.missing" in str(excinfo.value)

    def test_a_table_outside_the_run_snapshot_is_rejected(self) -> None:
        """上下文里有、但不在本次 Run 快照里 = 不是本次 Run 的资源。"""
        state = make_state("sales.orders")
        state["data_context"]["sales.ghost"] = table_context()
        state["table_snapshot"] = {"sales.orders": {"name": "sales.orders"}}

        with pytest.raises(UnknownTableError) as excinfo:
            select_tables(state, ["sales.ghost"])

        assert excinfo.value.missing == ("sales.ghost",)

    def test_duplicated_target_tables_are_rejected(self) -> None:
        state = make_state("sales.orders")
        with pytest.raises(ValueError, match="重复"):
            select_tables(state, ["sales.orders", "sales.orders"])

    def test_an_empty_target_list_is_rejected(self) -> None:
        state = make_state("sales.orders")
        with pytest.raises(ValueError):
            select_tables(state, [])

    def test_the_resource_name_is_not_the_physical_table_name(self) -> None:
        """SQLite 资源名是 `文件主名.表名`，物理表名只是其中的一段。"""
        state = make_state("sales.orders")

        with pytest.raises(UnknownTableError) as excinfo:
            select_tables(state, ["orders"])

        assert excinfo.value.missing == ("orders",)
        # 资源名可用
        assert select_tables(state, ["sales.orders"]).kept == ("sales.orders",)


class TestSampleBudget:
    def test_sample_rows_are_capped_but_schema_is_kept(self) -> None:
        state = make_state("sales.orders", rows=50)

        selection = select_tables(state, ["sales.orders"])

        table = selection.tables["sales.orders"]
        assert len(table["data"]) == MAX_SAMPLE_ROWS
        assert selection.samples_trimmed == 50 - MAX_SAMPLE_ROWS
        # 结构不能因为压样本而丢
        assert table["schema"] == state["data_context"]["sales.orders"]["schema"]
        assert table["row_count"] == 100
        assert table["column_count"] == 2

    def test_head_tail_samples_are_capped_too(self) -> None:
        state = make_state("sales.orders")
        state["data_context"]["sales.orders"]["data"] = {
            "head": [{"i": index} for index in range(20)],
            "tail": [{"i": index} for index in range(20)],
        }

        selection = select_tables(state, ["sales.orders"])

        data = selection.tables["sales.orders"]["data"]
        assert len(data["head"]) == MAX_SAMPLE_ROWS
        assert len(data["tail"]) == MAX_SAMPLE_ROWS
        assert selection.samples_trimmed == 30

    def test_the_original_context_is_not_mutated(self) -> None:
        state = make_state("sales.orders", rows=50)
        before = copy.deepcopy(state["data_context"])

        select_tables(state, ["sales.orders"])

        assert state["data_context"] == before


class TestHistorySelection:
    def test_no_results_means_no_history(self) -> None:
        assert select_history(make_state("sales.orders")) == []

    def test_the_current_task_survives_truncation(self) -> None:
        """保留最近 N 条时，先丢别的任务，而不是正在修的这个任务。"""
        state = make_state("sales.orders")
        state["execution_results"] = [
            {
                "task_id": "other",
                "success": True,
                "error": None,
                "error_kind": None,
                "retry_count": 0,
                "refine_count": 0,
            }
            for _ in range(10)
        ] + [
            {
                "task_id": "task_1",
                "success": False,
                "error": "boom",
                "error_kind": "execution_error",
                "retry_count": 0,
                "refine_count": 0,
            }
        ]

        history = select_history(state, limit=3)

        assert len(history) == 3
        assert history[-1]["task_id"] == "task_1"
        assert history[-1]["error"] == "boom"

    def test_history_is_deterministic_and_carries_no_result_payload(self) -> None:
        state = make_state("sales.orders")
        state["execution_results"] = [
            {
                "task_id": "task_1",
                "success": True,
                "error": None,
                "error_kind": None,
                "retry_count": 1,
                "refine_count": 0,
                "result": [{"secret": "row"}],
            }
        ]

        first = select_history(state)
        second = select_history(state)

        assert first == second
        assert "result" not in first[0]
        assert set(first[0]) == {
            "task_id",
            "success",
            "error",
            "error_kind",
            "attempt",
        }


class TestBuildContextReport:
    def test_the_report_says_what_was_kept_and_dropped(self) -> None:
        state = make_state("sales.orders", "sales.customers", rows=20)
        selection = select_tables(state, ["sales.orders"])
        fields = {
            "task": json.dumps({"task_id": "task_1"}),
            "data_context": json.dumps(selection.tables, default=str),
        }

        bounded, report = build_context(
            "executor",
            fields,
            selection=selection,
            history=select_history(state),
            task_id="task_1",
        )

        payload = report.to_dict()
        assert payload["node"] == "executor"
        assert payload["task_id"] == "task_1"
        assert payload["kept_tables"] == ["sales.orders"]
        assert payload["excluded_tables"] == ["sales.customers"]
        assert payload["history_entries"] == 0
        assert payload["original_chars"] >= payload["final_chars"]
        assert payload["token_estimate_method"].startswith("chars/")
        assert "prompt" not in json.dumps(payload).lower()
        assert set(bounded) == set(fields)

    def test_a_preserved_field_may_exceed_the_budget(self) -> None:
        """保留字段宁可超预算也不压掉，报告要说明这是超预算而不是没生效。"""
        state = make_state("sales.orders")
        selection = select_tables(state, ["sales.orders"])
        fields = {
            "task": json.dumps({"task_id": "task_1"}),
            "data_context": "x" * 5000,
        }

        _bounded, report = build_context(
            "executor", fields, selection=selection, budget=200
        )

        assert report.over_budget is True
        assert report.final_chars > report.budget_chars

    def test_optional_fields_are_compressed_to_the_budget(self) -> None:
        # reporter 的保留字段不含 execution_results，因此这里会被按预算压缩。
        # 预算作用在字段值上，序列化后的键名与引号仍占几十个字符，
        # 所以只要求落在一个小区间内，而不是精确等于 budget。
        fields = {"execution_results": "r" * 4000}

        _bounded, report = build_context("reporter", fields, budget=1500)

        assert report.final_chars <= 1500 + 64
        assert report.final_chars >= 1400
        assert report.truncated is True

    def test_the_report_is_published_per_execution_context(self) -> None:
        state = make_state("sales.orders")
        selection = select_tables(state, ["sales.orders"])

        harness.get_harness().build_context(
            "executor",
            {"data_context": json.dumps(selection.tables, default=str)},
            data_context=selection.tables,
            task_id="task_1",
        )

        published = context_report_of("executor")
        assert published["kept_tables"] == ["sales.orders"]

    def test_concurrent_runs_do_not_share_reports(self) -> None:
        seen: dict[str, list[str]] = {}
        errors: list[BaseException] = []

        def worker(run_id: str, table: str) -> None:
            try:
                state = make_state(table)
                state["run_id"] = run_id
                with harness.bind_state(state):
                    selection = select_tables(state, [table])
                    harness.get_harness().build_context(
                        "executor",
                        {"data_context": json.dumps(selection.tables, default=str)},
                        data_context=selection.tables,
                    )
                    seen[run_id] = context_report_of("executor")["kept_tables"]
            except BaseException as error:  # noqa: BLE001
                errors.append(error)

        threads = [
            threading.Thread(target=worker, args=("run-a", "sales.orders")),
            threading.Thread(target=worker, args=("run-b", "sales.customers")),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        assert not errors, errors
        assert seen == {"run-a": ["sales.orders"], "run-b": ["sales.customers"]}


# ---------------------------------------------------------------------------
# 真实工作流：五张表的确定性数据源
# ---------------------------------------------------------------------------

FIVE_TABLES = ("orders", "customers", "products", "regions", "returns")


def build_five_table_sqlite(path: Path) -> Path:
    with sqlite3.connect(path) as connection:
        for index, table in enumerate(FIVE_TABLES):
            connection.execute(f"CREATE TABLE {table} (id INTEGER, amount REAL)")
            connection.execute(f"INSERT INTO {table} VALUES (?, ?)", (index, index * 1.5))
    return path


def resource(name: str) -> str:
    return f"shop.{name}"


@pytest.fixture(autouse=True)
def _isolate_agents(monkeypatch: pytest.MonkeyPatch):
    """用替身工具跑通全流程，不启动容器。"""
    from harness import tool_dispatcher

    def fake_invoke(tool_name, tables, arguments, *, snapshots=None):
        if tool_name == "plot":
            return {"success": True, "result": "chart.png", "error": None, "error_kind": None}
        return {
            "success": True,
            "result": [{"n": 1}],
            "error": None,
            "error_kind": None,
        }

    monkeypatch.setattr(tool_dispatcher, "invoke_tool", fake_invoke)
    return monkeypatch


def read_log_events(output_root: Path, run_id: str, event: str) -> list[dict]:
    path = output_root / "logs" / f"{run_id}.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("event") == event
    ]


def read_context_events(output_root: Path, run_id: str) -> list[dict]:
    return read_log_events(output_root, run_id, "llm_context")


class TestFiveTableWorkflow:
    def test_planner_sees_everything_and_execution_agents_see_their_task(
        self, tmp_path: Path
    ) -> None:
        from services.analysis_service import run_analysis
        from tests.test_e2e_business_scenarios import ScriptedFactory

        source = build_five_table_sqlite(tmp_path / "shop.sqlite")
        replies = {
            "planner": {
                "tasks": [
                    {
                        "task_id": "task_1",
                        "description": "订单总量",
                        "target_tables": [resource("orders")],
                        "expected_output": "行数",
                    },
                    {
                        "task_id": "task_2",
                        "description": "产品总量",
                        "target_tables": [resource("products")],
                        "expected_output": "行数",
                    },
                ]
            },
            "executor": {"code": "SELECT COUNT(*) AS n FROM orders"},
            "critic": {"validation_passed": True, "feedback": "ok"},
            "visualizer": {"charts": []},
            "reporter": "# Report\n",
        }
        run_id = "h2-five-tables"

        result = run_analysis(
            run_id=run_id,
            data_sources=[str(source)],
            model_factory=ScriptedFactory(run_id, replies).build(),
            output_root=tmp_path,
        )

        assert result["analysis_status"] == "completed", result.get("status_reason")

        events = read_context_events(tmp_path, run_id)
        by_agent: dict[str, list[list[str]]] = {}
        for event in events:
            by_agent.setdefault(event["node"], []).append(
                event["details"]["context"]["kept_tables"]
            )

        # Planner 需要判断可行性，因此看得到全部五张表
        assert sorted(by_agent["planner"][0]) == sorted(resource(t) for t in FIVE_TABLES)
        # 执行类 Agent 只看到自己任务的表
        assert sorted(by_agent["executor"]) == [
            [resource("orders")],
            [resource("products")],
        ], by_agent["executor"]
        for agent in ("critic", "repair", "refine", "plotter", "plot_repair"):
            for kept in by_agent.get(agent, []):
                assert len(kept) == 1, f"{agent} 看到了多余的表：{kept}"

    def test_an_unknown_table_never_reaches_execution(self, tmp_path: Path) -> None:
        """计划里的未知资源被拒绝，绝不会进入执行。

        H3 的校验器在 Planner 之后、路由之前就拦下这份计划；没有执行结果，
        也没有产物。Context Manager 自己的拒绝行为由
        `TestTableSelection::test_an_unknown_table_is_rejected_not_skipped` 覆盖。
        """
        from services.analysis_service import run_analysis
        from tests.test_e2e_business_scenarios import ScriptedFactory

        source = build_five_table_sqlite(tmp_path / "shop.sqlite")
        replies = {
            "planner": {
                "tasks": [
                    {
                        "task_id": "task_1",
                        "description": "编造的表",
                        "target_tables": [resource("invented")],
                        "expected_output": "无",
                    }
                ]
            },
            "executor": {"code": "SELECT 1"},
            "critic": {"validation_passed": True, "feedback": "ok"},
            "visualizer": {"charts": []},
            "reporter": "# Report\n",
        }
        run_id = "h2-unknown-table"

        result = run_analysis(
            run_id=run_id,
            data_sources=[str(source)],
            model_factory=ScriptedFactory(run_id, replies).build(),
            output_root=tmp_path,
        )

        assert result["analysis_status"] == "failed"
        assert result["analysis_plan"] == []
        assert result["execution_results"] == []
        # 拒绝发生在执行前：没有结果产物
        assert not list((tmp_path / run_id).glob("results/*.json"))

        events = read_log_events(tmp_path, run_id, "plan_rejected")
        assert events, "未知资源没有被记录为计划校验失败"
        assert "unknown_table" in events[0]["details"]["categories"]
