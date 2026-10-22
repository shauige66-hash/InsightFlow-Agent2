"""评测与可靠性基准的测试。

这些用例验证的是**评测工具本身**是否可信：

- 任务集合的规模、唯一性与语义多样性；
- 标准答案确实独立算出（把答案改错必须判错）；
- 完成率与正确率是分开的指标；
- 报告产物齐全且带口径声明；
- 故障注入场景的判定符合前几阶段已确立的语义。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.faults import SCENARIOS, run_all as run_faults
from eval.runner import (
    MOCK_SCOPE_NOTE,
    UNVERIFIED_REAL_MODEL_METRICS,
    EvalRun,
    ScriptedModelFactory,
    run_all,
    write_csv,
    write_json,
    write_markdown,
    write_plot,
)
from eval.scoring import (
    STATUS_CORRECT,
    STATUS_INCORRECT,
    EvalSummary,
    TaskOutcome,
    extract_answer,
    values_match,
)
from eval.tasks import (
    CSV_ROWS,
    build_tasks,
    csv_revenue_by_region,
    csv_total_revenue,
    families,
    materialise,
    sqlite_total,
    task_by_id,
)

# ---------------------------------------------------------------------------
# 任务集合
# ---------------------------------------------------------------------------


class TestTaskSet:
    def test_size_is_in_the_required_range(self) -> None:
        tasks = build_tasks()
        assert 30 <= len(tasks) <= 50, f"任务数 {len(tasks)} 不在 30~50"

    def test_ids_are_unique(self) -> None:
        ids = [task.task_id for task in build_tasks()]
        assert len(ids) == len(set(ids))

    def test_covers_all_three_source_types(self) -> None:
        sources = {task.data_source for task in build_tasks()}
        assert sources == {"csv", "excel", "sqlite"}, sources

    def test_covers_the_required_operator_families(self) -> None:
        names = set(families())
        for required in (
            "csv_aggregate",
            "csv_filter",
            "csv_group",
            "csv_ranking",
            "excel_single_sheet",
            "excel_cross_sheet",
            "sqlite_aggregate",
            "sqlite_join",
            "sqlite_filter",
            "sqlite_distinct",
            "sqlite_extreme",
        ):
            assert required in names, f"缺少任务族 {required}"

    def test_no_pure_duplicates(self) -> None:
        """不能只换个名字就当成新任务：代码或期望值必须有区别。"""
        seen: dict[tuple, str] = {}
        for task in build_tasks():
            key = (
                task.data_source,
                task.code.strip(),
                json.dumps(task.expected, sort_keys=True, ensure_ascii=False),
            )
            assert key not in seen, (
                f"{task.task_id} 与 {seen.get(key)} 在语义上重复"
            )
            seen[key] = task.task_id

    def test_every_task_has_an_expected_answer(self) -> None:
        for task in build_tasks():
            assert task.expected is not None, task.task_id
            assert task.target_tables, task.task_id

    def test_sql_tasks_are_read_only(self) -> None:
        for task in build_tasks():
            if task.data_source != "sqlite":
                continue
            head = task.code.strip().upper()
            assert head.startswith(("SELECT", "WITH")), (task.task_id, task.code)

    def test_answers_are_computed_independently_of_the_workflow(self) -> None:
        """标准答案必须来自 tasks 模块自己的计算，而不是工作流输出。

        这里用"手工重算一遍并与模块给出的值比对"来固定这条性质。
        """
        manual_total = round(
            sum(round(row[3] * row[4], 2) for row in CSV_ROWS), 2
        )
        assert csv_total_revenue() == manual_total

        manual_region: dict[str, float] = {}
        for _order, region, _category, units, price in CSV_ROWS:
            manual_region[region] = round(
                manual_region.get(region, 0.0) + round(units * price, 2), 2
            )
        assert csv_revenue_by_region() == manual_region

        assert sqlite_total() == round(10.0 + 15.5 + 20.0 + 5.25 + 100.0 + 42.0 + 7.5, 2)

    def test_task_by_id_rejects_unknown(self) -> None:
        with pytest.raises(KeyError):
            task_by_id("no_such_task")

    def test_families_counts_match_tasks(self) -> None:
        assert sum(families().values()) == len(build_tasks())


# ---------------------------------------------------------------------------
# 判分口径
# ---------------------------------------------------------------------------


class TestScoring:
    @pytest.mark.parametrize(
        "actual, expected, want",
        [
            (1.0, 1.0 + 1e-12, True),
            (1.01, 1.0, False),
            ("east", "East", False),
            ("East", "East", True),
            ([1, 2], [1, 2], True),
            ([1, 2], [1, 2, 3], False),
            ({"a": 1, "extra": 9}, {"a": 1}, True),
            ({"a": 1}, {"b": 1}, False),
            (True, True, True),
            (1, True, False),
        ],
    )
    def test_values_match(self, actual, expected, want: bool) -> None:
        assert values_match(actual, expected) is want

    def test_numeric_key_normalisation(self) -> None:
        """1062 与 '1062' 是类型差异，不是答案差异。"""
        assert values_match({"1062": 225.0}, {1062: 225.0}) is True

    def test_completion_and_correctness_are_separate(self) -> None:
        summary = EvalSummary()
        summary.add(TaskOutcome("a", "f", STATUS_CORRECT, expected=1, actual=1))
        summary.add(TaskOutcome("b", "f", STATUS_INCORRECT, expected=1, actual=2))
        summary.add(TaskOutcome("c", "f", "incomplete"))

        assert summary.total == 3
        assert summary.correct == 1
        assert summary.completed == 2
        # 正确率分母是全部任务；完成率分母也是全部任务——但含义不同
        assert summary.correctness_rate == pytest.approx(1 / 3)
        assert summary.completion_rate == pytest.approx(2 / 3)
        # 条件正确率只看已完成的任务
        assert summary.conditional_accuracy == pytest.approx(1 / 2)

    def test_partial_is_its_own_bucket(self) -> None:
        summary = EvalSummary()
        summary.add(TaskOutcome("a", "f", "partial"))

        assert summary.partial == 1
        assert summary.completed == 1
        assert summary.correct == 0

    def test_by_family_groups_correctly(self) -> None:
        summary = EvalSummary()
        summary.add(TaskOutcome("a", "x", STATUS_CORRECT))
        summary.add(TaskOutcome("b", "x", STATUS_INCORRECT))
        summary.add(TaskOutcome("c", "y", STATUS_CORRECT))

        grouped = summary.by_family()
        assert grouped["x"]["total"] == 2
        assert grouped["x"]["correct"] == 1
        assert grouped["y"]["correctness_rate"] == pytest.approx(1.0)

    def test_extract_answer_shapes(self) -> None:
        scalar_task = task_by_id("csv_total_revenue")
        assert extract_answer(scalar_task, [{"total": 1.5}]) == 1.5

        grouped_task = task_by_id("csv_revenue_by_region")
        answer = extract_answer(
            grouped_task, [{"region": "East", "revenue": 2720.0}]
        )
        assert answer == {"East": 2720.0}

    def test_row_shape_keeps_every_column(self) -> None:
        """复合一行必须整行返回。

        回归：只取 `value_column` 会丢掉其余列，
        "最大单笔对应的客户"这类任务就会把 `{name, amount}` 比成 `Dan`。
        """
        task = task_by_id("sqlite_max_order_customer")
        assert task.shape == "row"
        answer = extract_answer(task, [{"name": "Dan", "amount": 100.0}])
        assert answer == {"name": "Dan", "amount": 100.0}
        assert values_match(answer, task.expected)

    def test_row_shape_rejects_multiple_rows(self) -> None:
        task = task_by_id("sqlite_max_order_customer")
        assert extract_answer(task, [{"name": "a"}, {"name": "b"}]) is None

    def test_row_tasks_have_composite_answers(self) -> None:
        for task in build_tasks():
            if task.shape != "row":
                continue
            assert isinstance(task.expected, dict) and len(task.expected) >= 2, (
                task.task_id,
                task.expected,
            )

    def test_extract_answer_empty_records(self) -> None:
        assert extract_answer(task_by_id("csv_total_revenue"), []) is None

    def test_wrong_answer_is_scored_incorrect(self) -> None:
        """把期望改错必须判错——否则判分逻辑形同虚设。"""
        task = task_by_id("csv_total_revenue")
        assert values_match(extract_answer(task, [{"total": task.expected}]), task.expected)
        assert not values_match(
            extract_answer(task, [{"total": task.expected + 1}]), task.expected
        )


# ---------------------------------------------------------------------------
# 数据生成
# ---------------------------------------------------------------------------


class TestDataMaterialisation:
    def test_materialise_writes_three_sources(self, tmp_path: Path) -> None:
        data = materialise(tmp_path / "data")
        assert set(data) == {"csv", "excel", "sqlite"}
        for path in data.values():
            assert path.exists(), path

    def test_sqlite_contains_an_unauthorized_table(self, tmp_path: Path) -> None:
        """评测数据里保留一张未授权表，用于确认授权边界没被放宽。"""
        import sqlite3

        data = materialise(tmp_path / "data")
        connection = sqlite3.connect(str(data["sqlite"]))
        try:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
        finally:
            connection.close()
        assert {"customers", "orders", "secrets"} <= tables

    def test_materialise_is_deterministic(self, tmp_path: Path) -> None:
        first = materialise(tmp_path / "a")["csv"].read_text(encoding="utf-8")
        second = materialise(tmp_path / "b")["csv"].read_text(encoding="utf-8")
        assert first == second


# ---------------------------------------------------------------------------
# 执行器与报告
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_run(tmp_path_factory) -> EvalRun:
    """跑一个小规模真实评测（含 Docker 沙箱与 MCP 子进程）。"""
    _require_sandbox()
    workdir = tmp_path_factory.mktemp("p5-eval")
    tasks = [
        task_by_id("csv_total_revenue"),
        task_by_id("excel_total_revenue"),
        task_by_id("sqlite_total"),
    ]
    return run_all(workdir=workdir, mode="mcp", tasks=tasks)


class TestScoringDetectsBadAnswers:
    """评测器必须能识别**错误答案**，而不是只会给通过。

    这组用例刻意构造坏结果，确认判分把它们判错——评测器的价值
    全在这里：一个恒返回 correct 的判分器毫无意义。
    """

    def test_wrong_number_is_incorrect(self) -> None:
        task = task_by_id("csv_total_revenue")
        actual = extract_answer(task, [{"total": float(task.expected) + 1}])
        assert not values_match(actual, task.expected)

    def test_zero_where_a_value_is_expected_is_incorrect(self) -> None:
        """报 0 是常见的失败模式（例如查询没跑出结果却回了 0）。"""
        task = task_by_id("sqlite_total")
        actual = extract_answer(task, [{"total": 0.0}])
        assert task.expected != 0
        assert not values_match(actual, task.expected)

    def test_empty_result_is_incomplete_not_correct(self) -> None:
        task = task_by_id("csv_revenue_by_region")
        assert extract_answer(task, []) is None
        assert not values_match(None, task.expected)

    def test_partial_group_is_incorrect(self) -> None:
        """分组结果少了一组必须判错。"""
        task = task_by_id("csv_revenue_by_region")
        expected = task.expected
        assert len(expected) >= 2

        partial = dict(list(expected.items())[:1])
        assert not values_match(partial, expected)

    def test_extra_wrong_group_is_incorrect(self) -> None:
        """多出一组必须判错。

        回归：`values_match` 的设计是"忽略多余键"（复合单行答案可能多带一列），
        这个宽松规则会把"多出一组"的分组答案误判为正确。
        因此分组答案走严格的 `grouped_match`。
        """
        from eval.scoring import grouped_match

        task = task_by_id("csv_revenue_by_region")
        actual = {**task.expected, "NoSuchRegion": 1.0}

        # 宽松比较会放过它——这正是需要严格比较的原因
        assert values_match(actual, task.expected) is True
        # 严格比较必须拦住
        assert grouped_match(actual, task.expected) is False

    def test_grouped_match_accepts_the_exact_answer(self) -> None:
        from eval.scoring import grouped_match

        task = task_by_id("csv_revenue_by_region")
        assert grouped_match(dict(task.expected), task.expected) is True

    def test_grouped_match_normalises_numeric_keys(self) -> None:
        from eval.scoring import grouped_match

        assert grouped_match({"1062": 225.0}, {1062: 225.0}) is True

    def test_corrupted_artifact_fails_loudly(self, tmp_path: Path) -> None:
        """产物被改坏时必须报错，而不是当成空结果悄悄放过。"""
        from utils.result_artifacts import ResultArtifactError, load_result

        artifact = {
            "reference": "task_1",
            "path": str(tmp_path / "missing.json"),
            "sha256": "0" * 64,
            "size": 123,
            "row_count": 10,
        }

        with pytest.raises(ResultArtifactError):
            load_result(
                {"__kind__": "externalized", "row_count": 10},
                artifact,
                run_id="audit-run",
                output_root=tmp_path,
            )

    def test_outcome_status_drives_the_rates(self) -> None:
        """把坏结果喂进汇总，三项指标的走向必须符合定义。"""
        summary = EvalSummary()
        summary.add(TaskOutcome("ok", "f", STATUS_CORRECT))
        summary.add(TaskOutcome("wrong", "f", STATUS_INCORRECT))
        summary.add(TaskOutcome("empty", "f", "incomplete"))
        summary.add(TaskOutcome("partial", "f", "partial"))

        assert summary.total == 4
        assert summary.correct == 1
        assert summary.completed == 3          # correct + incorrect + partial
        assert summary.correctness_rate == pytest.approx(0.25)
        assert summary.completion_rate == pytest.approx(0.75)
        assert summary.conditional_accuracy == pytest.approx(1 / 3)


class TestFaultInjectionReality:
    """故障必须是**真的注入并触发**的，不是把最终状态写成失败。"""

    def test_every_scenario_leaves_a_real_error_message(
        self, fault_result: dict
    ) -> None:
        by_id = {
            item["scenario_id"]: item["observed"]
            for item in fault_result["verdicts"]
        }
        expected_fragments = {
            "python_code_error": "definitely_missing_column",
            "python_name_error": "undefined_helper",
            "sql_syntax_error": "no_such_table",
            "sql_write_attempt": "只读",
            "unauthorized_table": "未授权",
        }

        for scenario_id, fragment in expected_fragments.items():
            observed = by_id[scenario_id]
            messages = " ".join(
                str(item) for item in observed.get("failure_errors") or []
            )
            assert fragment in messages, (
                f"{scenario_id} 没有留下真实错误（{fragment!r}）；实际：{messages[:200]}"
            )

    def test_injected_code_differs_from_the_working_code(self) -> None:
        """注入的坏代码必须与正常工作代码不同，否则等于没注入。"""
        from eval.faults import BROKEN_CODE

        good_sql = "SELECT COUNT(*) AS n FROM orders"
        for scenario_id, broken in BROKEN_CODE.items():
            assert broken.strip() != good_sql, scenario_id
            assert broken.strip(), scenario_id

    def test_write_scenarios_leave_the_database_untouched(
        self, fault_result: dict
    ) -> None:
        by_id = {
            item["scenario_id"]: item["observed"]
            for item in fault_result["verdicts"]
        }
        for scenario_id in ("sql_syntax_error", "sql_write_attempt", "unauthorized_table"):
            assert by_id[scenario_id].get("database_rows") == 7, scenario_id


class TestRunner:
    def test_small_run_completes_all_tasks(self, small_run: EvalRun) -> None:
        assert small_run.summary.total == 3
        assert small_run.summary.completed == 3, [
            (item.task_id, item.status, item.reason)
            for item in small_run.summary.outcomes
        ]

    def test_answers_are_correct(self, small_run: EvalRun) -> None:
        for outcome in small_run.summary.outcomes:
            assert outcome.correct, (outcome.task_id, outcome.expected, outcome.actual)

    def test_outcome_records_instrumentation(self, small_run: EvalRun) -> None:
        for outcome in small_run.summary.outcomes:
            assert outcome.llm_calls >= 1, outcome.task_id
            assert outcome.tool_calls >= 1, outcome.task_id
            assert outcome.duration_seconds > 0

    def test_summary_serialises(self, small_run: EvalRun) -> None:
        payload = small_run.to_dict()
        assert payload["scope"] == MOCK_SCOPE_NOTE
        assert payload["unverified_real_model_metrics"] == list(
            UNVERIFIED_REAL_MODEL_METRICS
        )
        json.dumps(payload)  # 必须可序列化

    def test_scripted_factory_does_not_touch_the_network(self) -> None:
        """Mock 工厂只按脚本作答；conftest 的网络守卫会拦截真实调用。"""
        task = task_by_id("csv_total_revenue")
        factory = ScriptedModelFactory(task)
        model_factory = factory.build("p5-unit")
        model = model_factory.resolver("planner")
        assert model._llm_type.startswith("p5-")


class TestReports:
    def test_all_report_artifacts_are_written(
        self, small_run: EvalRun, tmp_path: Path
    ) -> None:
        out = tmp_path / "reports"
        json_path = write_json(small_run, out / "eval.json")
        csv_path = write_csv(small_run, out / "eval.csv")
        md_path = write_markdown(small_run, out / "eval.md")

        assert json_path.exists() and csv_path.exists() and md_path.exists()

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["summary"]["total"] == 3
        assert "scope" in payload

    def test_markdown_states_the_mock_limit(self, small_run: EvalRun, tmp_path: Path) -> None:
        md_path = write_markdown(small_run, tmp_path / "eval.md")
        text = md_path.read_text(encoding="utf-8")

        # 必须显式声明 Mock 的边界，不能让人误读成真实模型能力
        assert "Mock" in text
        assert "不能" in text
        # 必须列出尚未验证的真实模型指标
        for item in UNVERIFIED_REAL_MODEL_METRICS:
            assert item in text
        # 完成率与正确率必须分别出现
        assert "完成率" in text
        assert "正确率" in text

    def test_csv_has_one_row_per_task(self, small_run: EvalRun, tmp_path: Path) -> None:
        import csv as csv_module

        path = write_csv(small_run, tmp_path / "eval.csv")
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv_module.DictReader(handle))

        assert len(rows) == 3
        assert {"task_id", "status", "completed", "correct"} <= set(rows[0])

    def test_plot_is_written_when_matplotlib_is_available(
        self, small_run: EvalRun, tmp_path: Path
    ) -> None:
        plot = write_plot(small_run, tmp_path / "eval.png")
        if plot is None:
            pytest.skip("matplotlib 不可用")
        assert plot.exists()
        assert plot.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# 故障注入
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fault_result(tmp_path_factory) -> dict:
    _require_sandbox()
    workdir = tmp_path_factory.mktemp("p5-faults")
    return run_faults(workdir=workdir, mode="direct")


def _require_sandbox():
    from sandbox.executor import DockerSandboxExecutor
    executor = DockerSandboxExecutor()
    if not executor.available() or not executor.image_present():
        pytest.skip("真实Python故障/业务评测需要可用Docker及已构建sandbox镜像")


class TestFaultInjection:
    def test_every_scenario_matches_its_expected_disposition(
        self, fault_result: dict
    ) -> None:
        assert fault_result["total"] >= 7
        failures = [
            item for item in fault_result["verdicts"] if not item["passed"]
        ]
        assert not failures, [
            (item["scenario_id"], item["problems"]) for item in failures
        ]

    def test_code_and_sql_errors_enter_repair(self, fault_result: dict) -> None:
        by_id = {item["scenario_id"]: item["observed"] for item in fault_result["verdicts"]}
        for scenario_id in (
            "python_code_error",
            "python_name_error",
            "sql_syntax_error",
            "sql_write_attempt",
            "unauthorized_table",
        ):
            assert by_id[scenario_id]["repair_calls"] >= 1, scenario_id

    def test_environment_faults_do_not_enter_repair(self, fault_result: dict) -> None:
        """环境故障进入 Repair 只会浪费调用并掩盖真实原因。"""
        by_id = {item["scenario_id"]: item["observed"] for item in fault_result["verdicts"]}
        for scenario_id in ("mcp_unavailable", "missing_source"):
            assert by_id[scenario_id]["repair_calls"] == 0, scenario_id

    def test_mcp_unavailable_is_classified_as_unavailable(
        self, fault_result: dict
    ) -> None:
        by_id = {item["scenario_id"]: item["observed"] for item in fault_result["verdicts"]}
        kinds = by_id["mcp_unavailable"]["error_kinds"]
        assert "unavailable" in kinds, kinds

    def test_write_attempt_does_not_change_the_database(
        self, fault_result: dict
    ) -> None:
        by_id = {item["scenario_id"]: item["observed"] for item in fault_result["verdicts"]}
        rows = by_id["sql_write_attempt"]["database_rows"]
        assert rows == 7, f"写操作改变了数据：{rows} 行"

    def test_scenarios_are_declared_with_expectations(self) -> None:
        assert SCENARIOS, "应有故障场景定义"
        for scenario in SCENARIOS:
            assert scenario.description
            assert isinstance(scenario.repair_expected, bool)
