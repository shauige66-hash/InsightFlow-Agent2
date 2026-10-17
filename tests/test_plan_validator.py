"""计划校验是确定性逻辑，不依赖模型判断自己的计划。

校验只回答"这份计划能不能被执行"：结构、任务 ID、目标表是否属于本次 Run、
来源类型能否被现有路由落到单一工具。它不判断分析内容是否有业务意义。
"""

from __future__ import annotations

import pytest

from planning.validator import (
    MAX_PLAN_TASKS,
    plan_fingerprint,
    validate_plan,
)


def task(
    task_id: str = "task_1",
    *,
    tables: list[str] | None = None,
    description: str = "统计总数",
    expected: str = "一个数字",
    **extra: object,
) -> dict:
    payload = {
        "task_id": task_id,
        "description": description,
        "target_tables": tables if tables is not None else ["sales.orders"],
        "expected_output": expected,
    }
    payload.update(extra)
    return payload


AUTHORIZED = {"sales.orders", "sales.customers", "shop.items"}
SQL_TYPES = {"sales.orders": "sqlite", "sales.customers": "sqlite", "shop.items": "csv"}


class TestStructure:
    def test_a_valid_plan_passes_and_is_normalized(self) -> None:
        plan = [task(description="  统计订单  ", tables=[" sales.orders "])]

        result = validate_plan(plan, authorized_tables=AUTHORIZED, source_types=SQL_TYPES)

        assert result.ok
        assert result.categories == []
        assert result.tasks[0]["description"] == "统计订单"
        assert result.tasks[0]["target_tables"] == ["sales.orders"]

    def test_an_empty_plan_is_valid_and_means_no_tasks(self) -> None:
        result = validate_plan([], authorized_tables=AUTHORIZED)

        assert result.ok
        assert result.is_empty
        assert result.tasks == []

    def test_a_non_list_is_rejected(self) -> None:
        result = validate_plan({"tasks": []}, authorized_tables=AUTHORIZED)

        assert not result.ok
        assert result.categories == ["not_a_list"]

    def test_too_many_tasks_are_rejected(self) -> None:
        plan = [task(f"task_{index}") for index in range(MAX_PLAN_TASKS + 1)]

        result = validate_plan(plan, authorized_tables=AUTHORIZED)

        assert not result.ok
        assert result.categories == ["too_many_tasks"]

    def test_a_missing_field_is_named(self) -> None:
        result = validate_plan(
            [{"task_id": "task_1", "description": "x", "target_tables": ["sales.orders"]}],
            authorized_tables=AUTHORIZED,
        )

        assert not result.ok
        assert result.categories == ["missing_field"]
        assert "expected_output" in result.errors[0].message

    def test_empty_text_fields_are_rejected(self) -> None:
        result = validate_plan(
            [task(description="   ")], authorized_tables=AUTHORIZED
        )

        assert not result.ok
        assert "empty_field" in result.categories

    def test_a_non_object_task_is_rejected(self) -> None:
        result = validate_plan(["task"], authorized_tables=AUTHORIZED)

        assert not result.ok
        assert result.categories == ["invalid_task"]


class TestTaskIdentity:
    def test_duplicate_task_ids_are_rejected(self) -> None:
        result = validate_plan(
            [task("task_1"), task("task_1")], authorized_tables=AUTHORIZED
        )

        assert not result.ok
        assert result.categories == ["duplicate_task_id"]

    def test_duplicate_target_tables_are_rejected(self) -> None:
        result = validate_plan(
            [task(tables=["sales.orders", "sales.orders"])],
            authorized_tables=AUTHORIZED,
        )

        assert not result.ok
        assert result.categories == ["duplicate_table"]

    def test_an_empty_target_list_is_rejected(self) -> None:
        result = validate_plan([task(tables=[])], authorized_tables=AUTHORIZED)

        assert not result.ok
        assert result.categories == ["empty_target_tables"]


class TestResources:
    def test_an_unknown_table_is_rejected(self) -> None:
        result = validate_plan(
            [task(tables=["sales.invented"])], authorized_tables=AUTHORIZED
        )

        assert not result.ok
        assert result.categories == ["unknown_table"]
        assert "sales.invented" in result.errors[0].message

    def test_a_physical_table_name_is_not_a_resource_name(self) -> None:
        """`orders` 是 SQLite 内部表名，资源名是 `sales.orders`。"""
        result = validate_plan([task(tables=["orders"])], authorized_tables=AUTHORIZED)

        assert not result.ok
        assert result.categories == ["unknown_table"]

    def test_a_task_may_span_several_sqlite_tables(self) -> None:
        result = validate_plan(
            [task(tables=["sales.orders", "sales.customers"])],
            authorized_tables=AUTHORIZED,
            source_types=SQL_TYPES,
        )

        assert result.ok

    def test_a_mixed_sqlite_and_dataframe_task_is_rejected(self) -> None:
        """现有路由只能"全 sqlite 走 sql、全 DataFrame 走 python"。"""
        result = validate_plan(
            [task(tables=["sales.orders", "shop.items"])],
            authorized_tables=AUTHORIZED,
            source_types=SQL_TYPES,
        )

        assert not result.ok
        assert result.categories == ["unsupported_source_mix"]

    def test_without_an_authorized_set_only_structure_is_checked(self) -> None:
        result = validate_plan([task(tables=["anything"])])

        assert result.ok


class TestDependencies:
    def test_an_unknown_dependency_is_rejected(self) -> None:
        result = validate_plan(
            [task("task_1", depends_on=["task_9"])], authorized_tables=AUTHORIZED
        )

        assert not result.ok
        assert result.categories == ["unknown_dependency"]

    def test_a_self_dependency_is_rejected(self) -> None:
        result = validate_plan(
            [task("task_1", depends_on=["task_1"])], authorized_tables=AUTHORIZED
        )

        assert not result.ok
        assert result.categories == ["cyclic_dependency"]

    def test_a_forward_dependency_is_rejected_by_sequential_execution(self) -> None:
        result = validate_plan(
            [
                task("task_1", depends_on=["task_2"]),
                task("task_2"),
            ],
            authorized_tables=AUTHORIZED,
        )

        assert not result.ok
        assert result.categories == ["dependency_order"]

    def test_a_backward_dependency_is_accepted(self) -> None:
        result = validate_plan(
            [task("task_1"), task("task_2", depends_on=["task_1"])],
            authorized_tables=AUTHORIZED,
        )

        assert result.ok

    def test_plans_without_dependencies_stay_compatible(self) -> None:
        result = validate_plan(
            [task("task_1"), task("task_2")], authorized_tables=AUTHORIZED
        )

        assert result.ok


class TestFingerprint:
    def test_the_same_plan_has_the_same_fingerprint(self) -> None:
        plan = [task("task_1"), task("task_2", tables=["sales.customers"])]
        assert plan_fingerprint(plan) == plan_fingerprint(plan)

    def test_changing_content_changes_the_fingerprint(self) -> None:
        before = [task("task_1", description="统计总数")]
        after = [task("task_1", description="统计平均值")]

        assert plan_fingerprint(before) != plan_fingerprint(after)

    def test_reordering_changes_the_fingerprint(self) -> None:
        first = [task("task_1"), task("task_2", tables=["sales.customers"])]
        second = [task("task_2", tables=["sales.customers"]), task("task_1")]

        assert plan_fingerprint(first) != plan_fingerprint(second)


class TestFeedback:
    def test_feedback_names_the_concrete_problem(self) -> None:
        result = validate_plan(
            [task("task_1", tables=["sales.invented"])], authorized_tables=AUTHORIZED
        )

        feedback = result.feedback()
        assert feedback
        assert "task_1" in feedback[0]
        assert "sales.invented" in feedback[0]

    def test_the_report_lists_categories_without_prompt_text(self) -> None:
        result = validate_plan([task(tables=[])], authorized_tables=AUTHORIZED)
        payload = result.to_dict()

        assert payload["ok"] is False
        assert payload["categories"] == ["empty_target_tables"]
        assert "prompt" not in str(payload).lower()
