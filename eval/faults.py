"""P5-C：可靠性故障注入场景。

## 为什么要单列

"任务算对了"与"出错时能正确处置"是两种能力。后者不能靠正常路径的任务
覆盖，必须**主动注入故障**，然后检查工作流的反应是否符合既定语义。

## 判定标准从哪来

每个场景的期望处置都来自**前几个阶段已经确立的语义**，不是本次新发明的：

| 故障 | 期望处置 | 依据 |
|---|---|---|
| Python 代码异常 | 进入 Repair | P2-B：`execution_error` 值得修复 |
| SQL 非法查询 | 进入 Repair | P2-B |
| SQL 只读违规 | 拒绝，可修复 | P2-B2 + P2-C 授权 |
| MCP Server 不可用 | 安全失败，**不进入 Repair** | P2-B/P2-C：环境类不可修复 |
| 数据源缺失 | 安全失败 | P1：数据源变化必须失败 |
| 结果文件被改坏 | 失败并报出原因 | P2-B3：产物指纹校验 |
| 预算耗尽 | 安全失败，不进入 Repair | P3 |

**错误类型必须被正确区分**：把环境故障当成"代码写错"去修复，
既浪费调用又会掩盖真实原因——这正是本阶段要固定住的性质。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.runtime import McpSettings
from ingestion.resources import ResourceResolver
from services.analysis_service import run_analysis

from eval.runner import ScriptedModelFactory
from eval.tasks import EvalTask, materialise


@dataclass(frozen=True)
class FaultScenario:
    """一个故障注入场景与它的期望处置。"""

    scenario_id: str
    description: str
    #: 期望工作流最终处于这个集合
    expected_analysis_status: tuple[str, ...]
    #: 期望的 error_kind（None 表示不断言）
    expected_error_kind: str | None
    #: 是否**允许**调用 repair
    repair_expected: bool
    #: 至少有一条失败记录
    expects_failure: bool = True
    #: 真实执行必须留下的错误特征（证明故障确实被注入并被触发，
    #: 而不是"只把最终状态写成失败"）
    expected_error_fragment: str | None = None


SCENARIOS: tuple[FaultScenario, ...] = (
    FaultScenario(
        scenario_id="python_code_error",
        description="Python 代码抛异常（列名不存在）→ 进入 Repair 并恢复",
        expected_analysis_status=("completed", "partial"),
        expected_error_kind=None,
        repair_expected=True,
        expects_failure=False,
        expected_error_fragment="definitely_missing_column",
    ),
    FaultScenario(
        scenario_id="python_name_error",
        description="Python 代码引用未定义名字 → 进入 Repair",
        expected_analysis_status=("completed", "partial"),
        expected_error_kind=None,
        repair_expected=True,
        expects_failure=False,
        expected_error_fragment="undefined_helper",
    ),
    FaultScenario(
        scenario_id="sql_syntax_error",
        description="SQL 语法错误（表不存在）→ 进入 Repair",
        expected_analysis_status=("completed", "partial"),
        expected_error_kind=None,
        repair_expected=True,
        expects_failure=False,
        expected_error_fragment="no_such_table",
    ),
    FaultScenario(
        scenario_id="sql_write_attempt",
        description="SQL 写操作 → 被词法层拒绝，进入 Repair",
        expected_analysis_status=("completed", "partial"),
        expected_error_kind=None,
        repair_expected=True,
        expects_failure=False,
        expected_error_fragment="只读",
    ),
    FaultScenario(
        scenario_id="unauthorized_table",
        description="查询未授权表 → 服务端拒绝，进入 Repair",
        expected_analysis_status=("completed", "partial"),
        expected_error_kind=None,
        repair_expected=True,
        expects_failure=False,
        expected_error_fragment="未授权",
    ),
)


def _task_for(scenario_id: str, data: dict[str, Path]) -> EvalTask:
    """为每个场景造一个最小任务。"""
    if scenario_id.startswith("python"):
        return EvalTask(
            task_id=f"fault_{scenario_id}",
            family="fault_python",
            description="故障注入：Python 代码错误",
            data_source="csv",
            target_tables=["orders"],
            code="result = len(df)\n",
            expected=None,
        )
    return EvalTask(
        task_id=f"fault_{scenario_id}",
        family="fault_sql",
        description="故障注入：SQL 问题",
        data_source="sqlite",
        target_tables=["sales.orders"],
        code="SELECT COUNT(*) AS n FROM orders",
        expected=None,
    )


#: 场景 → 首次执行的坏代码（repair 时用正确代码）
BROKEN_CODE: dict[str, str] = {
    "python_code_error": "result = df['definitely_missing_column']\n",
    "python_name_error": "result = undefined_helper(df)\n",
    "sql_syntax_error": "SELECT COUNT(*) AS n FROM no_such_table",
    "sql_write_attempt": "DELETE FROM orders",
    "unauthorized_table": "SELECT value FROM secrets",
}


def run_scenario(
    scenario: FaultScenario,
    *,
    workdir: Path,
    data: dict[str, Path],
    mode: str = "direct",
) -> dict[str, Any]:
    """跑一个故障场景并返回观测结果。"""
    task = _task_for(scenario.scenario_id, data)
    run_id = f"fault-{scenario.scenario_id}"
    broken = BROKEN_CODE[scenario.scenario_id]

    factory = ScriptedModelFactory(task)
    replies = factory._replies()  # noqa: SLF001 - 评测内部使用
    replies["executor"] = {"code": broken}
    # repair 给正确代码：这样能区分"真的修好了"与"根本没修"
    good = "result = len(df)\n" if task.data_source == "csv" else task.code
    replies["repair"] = {"code": good}
    replies["refine"] = {"code": good}
    factory._replies = lambda: replies  # type: ignore[method-assign]

    repair_calls: list[int] = []
    original_build = factory.build

    def build(run_id_arg: str):
        model_factory = original_build(run_id_arg)
        original_resolver = model_factory.resolver

        def resolver(agent_name: str):
            model = original_resolver(agent_name)
            if agent_name != "repair":
                return model

            original_generate = model._generate

            def generate(*args: Any, **kwargs: Any):
                repair_calls.append(1)
                return original_generate(*args, **kwargs)

            model._generate = generate  # type: ignore[method-assign]
            return model

        model_factory.resolver = resolver  # type: ignore[attr-defined]
        return model_factory

    factory.build = build  # type: ignore[method-assign]

    source = data[task.data_source]
    result: dict[str, Any] = {}
    error: str | None = None
    try:
        result = run_analysis(
            run_id=run_id,
            data_sources=[str(source)],
            model_factory=factory.build(run_id),
            resolver=ResourceResolver(),
            output_root=workdir,
            mcp=McpSettings(mode=mode),
        )
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"

    failures = result.get("failed_tasks") or []
    # 真实执行留下的失败记录：证明坏代码确实被送到执行入口并失败了，
    # 而不是"只把最终状态写成失败"。
    failure_errors = [
        item.get("error")
        for item in (result.get("execution_results") or [])
        if not item.get("success")
    ]
    return {
        "scenario_id": scenario.scenario_id,
        "description": scenario.description,
        "analysis_status": result.get("analysis_status"),
        "error": error,
        "repair_calls": len(repair_calls),
        "failure_count": len(failures),
        "failure_errors": failure_errors,
        "error_kinds": sorted(
            {str(item.get("error_kind")) for item in failures}
        ),
        "retry_counts": [item.get("retry_count") for item in failures],
        "reasons": [str(item.get("reason"))[:120] for item in failures],
        # 数据是否被写操作改动：授权与只读约束的硬判据
        "database_rows": _row_count(source) if task.data_source == "sqlite" else None,
    }


def _row_count(path: Path) -> int | None:
    try:
        connection = sqlite3.connect(str(path))
        try:
            return int(
                connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
            )
        finally:
            connection.close()
    except Exception:  # noqa: BLE001
        return None


def environment_scenarios(
    *, workdir: Path, data: dict[str, Path]
) -> list[dict[str, Any]]:
    """环境类故障：必须**安全失败**且**不进入 Repair**。

    这些不能靠脚本化模型制造——它们的触发点在基础设施里，
    因此这里用打桩替换掉对应组件。
    """
    from unittest.mock import patch

    import tools.mcp_sql_tool as mcp_tool

    observed: list[dict[str, Any]] = []

    task = EvalTask(
        task_id="fault_mcp_unavailable",
        family="fault_environment",
        description="MCP Server 不可用",
        data_source="sqlite",
        target_tables=["sales.orders"],
        code="SELECT COUNT(*) AS n FROM orders",
        expected=None,
    )

    repair_calls: list[int] = []
    factory = ScriptedModelFactory(task)
    original_build = factory.build

    def build(run_id: str):
        model_factory = original_build(run_id)
        original_resolver = model_factory.resolver

        def resolver(agent_name: str):
            model = original_resolver(agent_name)
            if agent_name != "repair":
                return model

            original_generate = model._generate

            def generate(*args: Any, **kwargs: Any):
                repair_calls.append(1)
                return original_generate(*args, **kwargs)

            model._generate = generate  # type: ignore[method-assign]
            return model

        model_factory.resolver = resolver  # type: ignore[attr-defined]
        return model_factory

    factory.build = build  # type: ignore[method-assign]

    def explode(*args: Any, **kwargs: Any):
        raise mcp_tool.McpUnavailable("注入：MCP Server 不可用")

    with patch.object(mcp_tool, "McpStdioClient", explode):
        result = run_analysis(
            run_id="fault-mcp-unavailable",
            data_sources=[str(data["sqlite"])],
            model_factory=factory.build("fault-mcp-unavailable"),
            resolver=ResourceResolver(),
            output_root=workdir,
            mcp=McpSettings(mode="mcp"),
        )

    failures = result.get("failed_tasks") or []
    observed.append(
        {
            "scenario_id": "mcp_unavailable",
            "description": "MCP Server 不可用 → 安全失败，不进入 Repair",
            "analysis_status": result.get("analysis_status"),
            "repair_calls": len(repair_calls),
            "failure_count": len(failures),
            "error_kinds": sorted({str(item.get("error_kind")) for item in failures}),
            "retry_counts": [item.get("retry_count") for item in failures],
            "reasons": [str(item.get("reason"))[:120] for item in failures],
        }
    )

    # 数据源缺失
    missing_task = EvalTask(
        task_id="fault_missing_source",
        family="fault_environment",
        description="数据源缺失",
        data_source="csv",
        target_tables=["orders"],
        code="result = len(df)\n",
        expected=None,
    )
    missing_factory = ScriptedModelFactory(missing_task)
    error: str | None = None
    try:
        run_analysis(
            run_id="fault-missing-source",
            data_sources=[str(workdir / "definitely-not-here.csv")],
            model_factory=missing_factory.build("fault-missing-source"),
            resolver=ResourceResolver(),
            output_root=workdir,
        )
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"

    observed.append(
        {
            "scenario_id": "missing_source",
            "description": "数据源缺失 → 安全失败",
            "analysis_status": None,
            "repair_calls": 0,
            "failure_count": 1 if error else 0,
            "error_kinds": ["unavailable"] if error else [],
            "retry_counts": [],
            "reasons": [error] if error else [],
            "raised": error is not None,
        }
    )

    return observed


def run_all(
    *, workdir: Path, mode: str = "direct"
) -> dict[str, Any]:
    """跑完全部故障场景，返回观测记录。"""
    workdir.mkdir(parents=True, exist_ok=True)
    data = materialise(workdir / "data")

    records: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        records.append(
            run_scenario(scenario, workdir=workdir, data=data, mode=mode)
        )
    records.extend(environment_scenarios(workdir=workdir, data=data))

    # 判断每个场景是否符合期望
    by_id = {str(item["scenario_id"]): item for item in records}
    verdicts: list[dict[str, Any]] = []

    for scenario in SCENARIOS:
        item = by_id[scenario.scenario_id]
        verdicts.append(_verdict(scenario, item))

    verdicts.append(
        _verdict(
            _mcp_unavailable_scenario(),
            by_id["mcp_unavailable"],
        )
    )
    verdicts.append(
        _verdict(
            FaultScenario(
                scenario_id="missing_source",
                description="数据源缺失",
                expected_analysis_status=(),
                expected_error_kind=None,
                repair_expected=False,
            ),
            by_id["missing_source"],
            key="raised",
        )
    )

    return {
        "records": records,
        "verdicts": verdicts,
        "passed": sum(1 for item in verdicts if item["passed"]),
        "total": len(verdicts),
    }


def _mcp_unavailable_scenario() -> FaultScenario:
    return FaultScenario(
        scenario_id="mcp_unavailable",
        description="MCP Server 不可用",
        expected_analysis_status=("partial", "failed"),
        expected_error_kind="unavailable",
        repair_expected=False,
    )


def _verdict(
    scenario: FaultScenario, observed: dict[str, Any], *, key: str = "analysis_status"
) -> dict[str, Any]:
    problems: list[str] = []

    if key == "raised":
        if not observed.get("raised"):
            problems.append("预期抛错但没有")
    elif scenario.expected_analysis_status:
        actual = observed.get("analysis_status")
        if actual not in scenario.expected_analysis_status:
            problems.append(
                f"终态 {actual!r} 不在期望 {scenario.expected_analysis_status}"
            )

    if scenario.repair_expected:
        if not observed.get("repair_calls"):
            problems.append("预期进入 Repair 但没有")
    else:
        if observed.get("repair_calls"):
            problems.append(
                f"环境类故障不该进入 Repair，实际 {observed['repair_calls']} 次"
            )

    if scenario.expected_error_kind is not None:
        kinds = observed.get("error_kinds") or []
        if scenario.expected_error_kind not in kinds:
            problems.append(
                f"error_kind 期望含 {scenario.expected_error_kind!r}，实际 {kinds}"
            )

    if scenario.expects_failure and not observed.get("failure_count"):
        problems.append("预期有失败记录但没有")

    # 真实执行的错误特征：证明故障确实被注入并触发了，
    # 而不是"只把最终状态写成失败"
    if scenario.expected_error_fragment:
        messages = " ".join(
            str(item) for item in observed.get("failure_errors") or []
        )
        if scenario.expected_error_fragment not in messages:
            problems.append(
                f"真实错误里找不到 {scenario.expected_error_fragment!r}；"
                f"实际：{messages[:160]}"
            )

    return {
        "scenario_id": scenario.scenario_id,
        "description": scenario.description,
        "passed": not problems,
        "problems": problems,
        "observed": observed,
    }
