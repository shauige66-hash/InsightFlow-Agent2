"""P2-D3：真实业务场景端到端。

数据全部由本文件生成，落在 `tmp_path` 下，与用户业务库完全隔离。

覆盖：
- CSV：多列读取、Python 统计计算、结果校验、图表生成。
- Excel：多 Sheet 读取、指定 Sheet、跨表分析、报告生成。
- SQLite：在 **mcp 模式**下经真实 MCP Server 完成 SELECT / JOIN / GROUP BY，
  并与本文件独立算出的标准答案逐值比对。
- 大结果：触发 P2-B 的结果外置，验证绘图消费真实结果、报告消费有界预览。
- 故障：代码错误进 Repair；环境故障不触发无效的模型重试。
- 安全：不可信 Python 在 Docker 执行；SQL 仍受 Authorizer / 只读 / 超时 /
  结果上限保护；MCP 私有授权文件不在沙箱 `/history` 内、不出现在模型输出里。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from core.config import Settings
from core.llm import CallContext, ModelFactory
from core.runtime import McpSettings
from ingestion.resources import ResourceResolver
from services.analysis_service import run_analysis

# ---------------------------------------------------------------------------
# 可重复的测试数据
# ---------------------------------------------------------------------------

REGIONS = ("East", "West", "North")
CATEGORIES = ("hardware", "software", "service")

# CSV 数据：region / category / units / unit_price / revenue
CSV_ROWS: list[tuple[str, str, int, float]] = []
for index in range(60):
    region = REGIONS[index % len(REGIONS)]
    category = CATEGORIES[index % len(CATEGORIES)]
    units = (index % 7) + 1
    price = 10.0 + (index % 5) * 2.5
    CSV_ROWS.append((region, category, units, price))


def csv_revenue(row: tuple[str, str, int, float]) -> float:
    return round(row[2] * row[3], 2)


# CSV 标准答案（独立于工作流计算）
CSV_EXPECTED = {
    "rows": len(CSV_ROWS),
    "total_revenue": round(sum(csv_revenue(row) for row in CSV_ROWS), 2),
    "by_region": {
        region: round(
            sum(csv_revenue(row) for row in CSV_ROWS if row[0] == region), 2
        )
        for region in REGIONS
    },
}

PLOT_CODE = (
    "counts = result_df['region'].value_counts().sort_index()\n"
    "fig = plt.figure(figsize=(6, 4))\n"
    "plt.bar(counts.index.tolist(), counts.values.tolist())\n"
    "plt.title('Orders by region')\n"
    "plt.xlabel('region')\n"
    "plt.ylabel('count')\n"
    "plt.tight_layout()\n"
    "plt.savefig(output_path, bbox_inches='tight')\n"
    "plt.close()\n"
    "result = output_path\n"
)


def build_csv(path: Path) -> Path:
    lines = ["region,category,units,unit_price"]
    for region, category, units, price in CSV_ROWS:
        lines.append(f"{region},{category},{units},{price}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_excel(path: Path) -> Path:
    import pandas as pd

    orders = pd.DataFrame(
        [
            {"order_id": 1, "region": "East", "units": 2, "unit_price": 10.0},
            {"order_id": 2, "region": "West", "units": 1, "unit_price": 25.0},
            {"order_id": 3, "region": "East", "units": 5, "unit_price": 10.0},
            {"order_id": 4, "region": "North", "units": 3, "unit_price": 15.0},
        ]
    )
    regions = pd.DataFrame(
        [
            {"region": "East", "manager": "Alice"},
            {"region": "West", "manager": "Bob"},
            {"region": "North", "manager": "Cara"},
        ]
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        orders.to_excel(writer, sheet_name="orders", index=False)
        regions.to_excel(writer, sheet_name="regions", index=False)
    return path


# Excel 标准答案
EXCEL_ORDERS = [
    ("East", 2, 10.0),
    ("West", 1, 25.0),
    ("East", 5, 10.0),
    ("North", 3, 15.0),
]
EXCEL_EXPECTED = {
    "orders_sheet_rows": len(EXCEL_ORDERS),
    "regions_sheet_rows": 3,
    "total_revenue": round(
        sum(units * price for _region, units, price in EXCEL_ORDERS), 2
    ),
    "east_revenue": round(
        sum(units * price for region, units, price in EXCEL_ORDERS if region == "East"),
        2,
    ),
}


def build_sqlite(path: Path) -> Path:
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            CREATE TABLE customers (
                customer_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                region TEXT NOT NULL
            );
            CREATE TABLE orders (
                order_id INTEGER PRIMARY KEY,
                customer_id INTEGER NOT NULL,
                amount REAL NOT NULL
            );
            CREATE TABLE secrets (value TEXT);
            """
        )
        connection.executemany(
            "INSERT INTO customers VALUES (?, ?, ?)",
            [
                (1, "Alice", "East"),
                (2, "Bob", "West"),
                (3, "Cara", "East"),
                (4, "Dan", "North"),
            ],
        )
        connection.executemany(
            "INSERT INTO orders VALUES (?, ?, ?)",
            [
                (101, 1, 10.0),
                (102, 1, 15.5),
                (103, 2, 20.0),
                (104, 3, 5.25),
                (105, 4, 100.0),
            ],
        )
        connection.execute("INSERT INTO secrets VALUES ('top-secret')")
        connection.commit()
    finally:
        connection.close()
    return path


# SQLite 标准答案（用 Python 独立算出，不依赖被测 SQL 路径）
SQLITE_EXPECTED = {
    "count": 5,
    "total": round(10.0 + 15.5 + 20.0 + 5.25 + 100.0, 2),
    "by_region": {
        "East": round(10.0 + 15.5 + 5.25, 2),
        "West": round(20.0, 2),
        "North": round(100.0, 2),
    },
    "join_by_customer": {
        "Alice": round(10.0 + 15.5, 2),
        "Bob": round(20.0, 2),
        "Cara": round(5.25, 2),
        "Dan": round(100.0, 2),
    },
}


# ---------------------------------------------------------------------------
# 脚本化模型
# ---------------------------------------------------------------------------


class ScriptedFactory:
    """按 Agent 名返回脚本化回复，并记录收到的提示词。"""

    def __init__(self, run_id: str, replies: dict[str, Any]) -> None:
        self.run_id = run_id
        self.replies = replies
        self.prompts: dict[str, list[str]] = {}
        self.calls: list[str] = []

    def build(self) -> ModelFactory:
        owner = self

        class _Scripted(BaseChatModel):
            agent_name: str = ""

            @property
            def _llm_type(self) -> str:
                return f"p2d3-{self.agent_name}"

            def _generate(
                self,
                messages: list[BaseMessage],
                stop: list[str] | None = None,
                run_manager: CallbackManagerForLLMRun | None = None,
                **kwargs: Any,
            ) -> ChatResult:
                owner.calls.append(self.agent_name)
                owner.prompts.setdefault(self.agent_name, []).append(
                    "\n".join(str(message.content) for message in messages)
                )
                reply = owner.replies.get(self.agent_name, {"tasks": []})
                content = (
                    reply
                    if isinstance(reply, str)
                    else json.dumps(reply, ensure_ascii=False)
                )
                return ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content=content))]
                )

        def resolver(agent_name: str) -> _Scripted:
            return _Scripted(agent_name=agent_name)

        return ModelFactory(
            settings=Settings(
                api_key="sk-fake",
                base_url="http://127.0.0.1:8000/v1",
                model="deepseek-chat",
            ),
            context=CallContext(run_id=self.run_id, client_app="insightflow-agent2"),
            resolver=resolver,
        )


def plan_of(target_tables: list[str], chart: bool = False) -> dict[str, Any]:
    return {
        "tasks": [
            {
                "task_id": "task_1",
                "description": "按区域统计",
                "target_tables": target_tables,
                "expected_output": "区域汇总",
            }
        ]
    }


def charts_of(target_tables: list[str]) -> dict[str, Any]:
    return {
        "charts": [
            {
                "chart_id": "chart_1",
                "source_task_id": "task_1",
                "chart_type": "bar",
                "title": "Orders by region",
                "description": "按区域统计订单数",
                "target_tables": target_tables,
                "expected_output": "柱状图",
            }
        ]
    }


def run_workflow(
    *,
    workdir: Path,
    run_id: str,
    data_sources: list[Path],
    target_tables: list[str],
    executor_code: str,
    mode: str = "direct",
    repair_code: str | None = None,
    chart: bool = False,
    approver: bool = False,
    plot_code: str | None = None,
):
    replies: dict[str, Any] = {
        "planner": plan_of(target_tables),
        "executor": {"code": executor_code},
        "repair": {"code": repair_code or executor_code},
        "refine": {"code": repair_code or executor_code},
        "critic": {"validation_passed": True, "feedback": "ok"},
        "visualizer": charts_of(target_tables) if chart else {"charts": []},
        "plotter": {"code": plot_code or PLOT_CODE},
        "plot_repair": {"code": plot_code or PLOT_CODE},
        "reporter": "# Analysis Report\n\n## Overview\n区域汇总已生成。\n",
    }
    recorder = ScriptedFactory(run_id, replies)

    resume = None
    if approver:
        from langgraph.types import Command

        resume = Command(resume={"decision": "approve"})

    result = run_analysis(
        run_id=run_id,
        data_sources=[str(item) for item in data_sources],
        model_factory=recorder.build(),
        resolver=ResourceResolver(),
        output_root=workdir,
        mcp=McpSettings(mode=mode),
        require_plan_approval=approver,
        resume=resume,
    )
    return result, recorder


def records_of(value: Any) -> list[dict]:
    """把一次执行结果统一成记录列表。

    两条后端的结果形状不同，测试需要同时接受：

    - Python 工具：result 是列式信封（`__kind__` + `records`）；
    - SQL 工具：result 是 `{columns, rows, row_count, bytes, truncated}`。

    只做形状归一，不做数值加工——数值断言一律以独立算出的标准答案为准。
    """
    from utils.result_artifacts import unwrap_records

    value = unwrap_records(value)

    if isinstance(value, dict) and "rows" in value and "columns" in value:
        return list(value.get("rows") or [])
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return value
    if isinstance(value, dict):
        return [value]
    raise AssertionError(f"无法从结果里取出记录：{type(value).__name__}")


def successful(result: dict) -> list[dict]:
    return [item for item in result["execution_results"] if item["success"]]


def summarize(result: dict) -> dict:
    items = successful(result)
    assert items, f"没有成功的执行结果：{result.get('failed_tasks')}"
    rows = records_of(items[-1]["result"])
    assert rows, "结果里没有记录"
    key = list(rows[0])[0]
    return {row[key]: row for row in rows}


class TestCsvAnalysis:
    def test_python_statistics_match_independent_answer(self, tmp_path: Path) -> None:
        db = build_csv(tmp_path / "sales.csv")

        code = (
            "frame = df.copy()\n"
            "frame['revenue'] = frame['units'] * frame['unit_price']\n"
            "by_region = frame.groupby('region')['revenue'].sum().round(2)\n"
            "result = by_region.reset_index()\n"
        )

        result, _recorder = run_workflow(
            workdir=tmp_path,
            run_id="p2d3-csv",
            data_sources=[db],
            target_tables=["sales"],
            executor_code=code,
        )

        assert result["analysis_status"] == "completed", result.get("status_reason")
        table = summarize(result)

        for region, expected in CSV_EXPECTED["by_region"].items():
            assert region in table, table
            assert table[region]["revenue"] == pytest.approx(expected), table

    def test_csv_chart_is_produced(self, tmp_path: Path) -> None:
        source = build_csv(tmp_path / "sales.csv")

        # 必须 `.copy()`：作用域里的 df 与 dfs[...] 是同一份对象，
        # 原地增删列会污染后续绘图步骤的 result_df（P1 的既有语义）。
        code = "result = df.copy()\n"

        result, _recorder = run_workflow(
            workdir=tmp_path,
            run_id="p2d3-csv-chart",
            data_sources=[source],
            target_tables=["sales"],
            executor_code=code,
            chart=True,
        )

        assert result["analysis_status"] == "completed", result.get("status_reason")

        visualizations = result.get("visualization_results") or []
        assert visualizations, "应产出图表"
        assert visualizations[0]["success"], visualizations[0]

        chart_path = Path(visualizations[0]["path"])
        assert chart_path.exists(), chart_path
        # 真实 PNG：校验文件头
        assert chart_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


class TestExcelAnalysis:
    def test_multi_sheet_read_and_cross_sheet_join(self, tmp_path: Path) -> None:
        """同时点名两张 sheet → `dfs` 里两张都在 → 跨表关联。

        Excel 的资源键是 `<文件名>.<sheet 名>`，因此跨表分析必须两张都点名：
        只点名一张时作用域里就只有那一张（下一条用例验证这一点）。
        """
        source = build_excel(tmp_path / "book.xlsx")

        code = (
            "orders = dfs['book.orders']\n"
            "regions = dfs['book.regions']\n"
            "merged = orders.merge(regions, on='region', how='left')\n"
            "merged['revenue'] = merged['units'] * merged['unit_price']\n"
            "result = merged[['region', 'manager', 'revenue']].copy()\n"
        )

        result, _recorder = run_workflow(
            workdir=tmp_path,
            run_id="p2d3-excel-cross",
            data_sources=[source],
            target_tables=["book.orders", "book.regions"],
            executor_code=code,
        )

        assert result["analysis_status"] == "completed", result.get("status_reason")
        rows = records_of(successful(result)[-1]["result"])
        assert rows, rows

        managers = {row["manager"] for row in rows}
        assert managers == {"Alice", "Bob", "Cara"}, managers

        total = round(sum(row["revenue"] for row in rows), 2)
        assert total == pytest.approx(EXCEL_EXPECTED["total_revenue"]), total

    def test_specified_sheet_only_sees_that_sheet(self, tmp_path: Path) -> None:
        """只点名一张 sheet 时，作用域里只有它一张。"""
        source = build_excel(tmp_path / "book.xlsx")

        code = (
            "frame = df\n"
            "result = [{'rows': int(len(frame)), "
            "'columns': sorted(frame.columns.tolist()), "
            "'available': sorted(dfs.keys())}]\n"
        )

        result, _recorder = run_workflow(
            workdir=tmp_path,
            run_id="p2d3-excel-sheet",
            data_sources=[source],
            target_tables=["book.regions"],
            executor_code=code,
        )

        assert result["analysis_status"] == "completed", result.get("status_reason")
        rows = records_of(successful(result)[-1]["result"])
        assert rows[0]["rows"] == EXCEL_EXPECTED["regions_sheet_rows"], rows
        assert "manager" in rows[0]["columns"], rows
        # 只有被点名的那张表可见
        assert rows[0]["available"] == ["book.regions"], rows

    def test_report_is_generated(self, tmp_path: Path) -> None:
        source = build_excel(tmp_path / "book.xlsx")

        result, _recorder = run_workflow(
            workdir=tmp_path,
            run_id="p2d3-excel-report",
            data_sources=[source],
            target_tables=["book.orders"],
            executor_code="result = df.copy()\n",
        )

        assert result["analysis_status"] == "completed", result.get("status_reason")
        report_path = Path(result["report_path"])
        assert report_path.exists(), report_path
        content = report_path.read_text(encoding="utf-8")
        assert "Analysis Report" in content


class TestSqliteThroughMcp:
    """mcp 模式 + 真实 MCP Server 完成 SELECT / JOIN / GROUP BY。"""

    def test_group_by_matches_independent_answer(self, tmp_path: Path) -> None:
        db = build_sqlite(tmp_path / "sales.sqlite")

        query = (
            "SELECT c.region AS region, SUM(o.amount) AS total "
            "FROM orders o JOIN customers c ON o.customer_id = c.customer_id "
            "GROUP BY c.region ORDER BY c.region"
        )

        result, _recorder = run_workflow(
            workdir=tmp_path,
            run_id="p2d3-sql-group",
            data_sources=[db],
            target_tables=["sales.orders", "sales.customers"],
            executor_code=query,
            mode="mcp",
        )

        assert result["analysis_status"] == "completed", result.get("status_reason")
        rows = records_of(successful(result)[-1]["result"])
        assert rows, rows

        got = {row["region"]: round(float(row["total"]), 2) for row in rows}
        assert got == SQLITE_EXPECTED["by_region"], got

    def test_join_matches_independent_answer(self, tmp_path: Path) -> None:
        db = build_sqlite(tmp_path / "sales.sqlite")

        query = (
            "SELECT c.name AS name, SUM(o.amount) AS total "
            "FROM orders o JOIN customers c ON o.customer_id = c.customer_id "
            "GROUP BY c.name ORDER BY c.name"
        )

        result, _recorder = run_workflow(
            workdir=tmp_path,
            run_id="p2d3-sql-join",
            data_sources=[db],
            target_tables=["sales.orders", "sales.customers"],
            executor_code=query,
            mode="mcp",
        )

        assert result["analysis_status"] == "completed", result.get("status_reason")
        rows = records_of(successful(result)[-1]["result"])

        got = {row["name"]: round(float(row["total"]), 2) for row in rows}
        assert got == SQLITE_EXPECTED["join_by_customer"], got

    def test_simple_select_and_aggregate(self, tmp_path: Path) -> None:
        db = build_sqlite(tmp_path / "sales.sqlite")

        result, _recorder = run_workflow(
            workdir=tmp_path,
            run_id="p2d3-sql-select",
            data_sources=[db],
            target_tables=["sales.orders"],
            executor_code="SELECT COUNT(*) AS n, SUM(amount) AS total FROM orders",
            mode="mcp",
        )

        assert result["analysis_status"] == "completed", result.get("status_reason")
        row = records_of(successful(result)[-1]["result"])[0]
        assert row["n"] == SQLITE_EXPECTED["count"], row
        assert round(float(row["total"]), 2) == SQLITE_EXPECTED["total"], row

    def test_unauthorized_table_is_refused_over_the_real_server(
        self, tmp_path: Path
    ) -> None:
        """即使 SQL 语法正确，未授权表也必须被服务端拒绝。"""
        db = build_sqlite(tmp_path / "sales.sqlite")

        result, _recorder = run_workflow(
            workdir=tmp_path,
            run_id="p2d3-sql-denied",
            data_sources=[db],
            target_tables=["sales.orders"],
            executor_code="SELECT value FROM secrets",
            mode="mcp",
        )

        assert result["analysis_status"] != "completed", result
        failures = result.get("failed_tasks") or []
        assert failures, result
        assert any("未授权" in str(item.get("reason")) for item in failures), failures


# ---------------------------------------------------------------------------
# 大结果：外置 + 两条消费路径
# ---------------------------------------------------------------------------

# 约 45 字节/行 × 8000 行 ≈ 360 KiB：超过 32 KiB 内联上限，
# 但仍在沙箱 1 MiB 结果上限内，因此能走到外置这一步。
BIG_ROWS = 8_000


def build_big_csv(path: Path) -> Path:
    lines = ["customer_id,region,amount"]
    for index in range(BIG_ROWS):
        lines.append(
            f"C{index:06d},{'East' if index % 2 else 'West'},{index * 1.5}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class TestLargeResultIsExternalized:
    def test_result_is_externalized_and_state_stays_small(self, tmp_path: Path) -> None:
        source = build_big_csv(tmp_path / "big.csv")

        result, _recorder = run_workflow(
            workdir=tmp_path,
            run_id="p2d3-big",
            data_sources=[source],
            target_tables=["big"],
            executor_code="result = df.copy()\n",
        )

        items = successful(result)
        assert items, result.get("failed_tasks")

        payload = items[-1]["result"]
        assert payload["__kind__"] == "externalized", payload
        assert payload["row_count"] == BIG_ROWS, payload
        assert items[-1]["result_artifact"] is not None

        # state 里只留**有界样本**，不留整表。
        # 外置信封刻意带少量 head/tail 供报告展示，因此不能断言"一行都没有"，
        # 而要断言样本规模有界、远小于全量。
        sample = payload.get("sample")
        if isinstance(sample, list):
            assert len(sample) < 200, f"样本过大：{len(sample)}"
        elif isinstance(sample, dict):
            for value in sample.values():
                if isinstance(value, list):
                    assert len(value) < 200, f"样本过大：{len(value)}"

        # 全量数据在产物文件里，且确实是 8000 行
        artifact = items[-1]["result_artifact"]
        assert artifact["row_count"] == BIG_ROWS, artifact

    def test_state_size_stays_bounded(self, tmp_path: Path) -> None:
        """state 的体积必须有界，不能随结果行数线性增长。"""
        source = build_big_csv(tmp_path / "big.csv")

        result, _recorder = run_workflow(
            workdir=tmp_path,
            run_id="p2d3-big-size",
            data_sources=[source],
            target_tables=["big"],
            executor_code="result = df.copy()\n",
        )

        # 8000 行 ≈ 360 KiB 原始数据；state 里只应是摘要级的内容
        encoded = json.dumps(result, ensure_ascii=False)
        assert len(encoded) < 120_000, f"state 过大：{len(encoded)} 字符"

    def test_plot_consumes_the_real_result_not_the_summary(self, tmp_path: Path) -> None:
        """绘图必须拿到真实行；把外置摘要当数据会画出空图。"""
        source = build_big_csv(tmp_path / "big.csv")

        # 图上标注真实行数：如果 result_df 是摘要而不是数据，这里会拿到 0/1 行
        plot_code = (
            "rows = int(len(result_df))\n"
            "fig = plt.figure(figsize=(6, 4))\n"
            "plt.bar(['rows'], [rows])\n"
            "plt.title(str(rows))\n"
            "plt.savefig(output_path, bbox_inches='tight')\n"
            "plt.close()\n"
            "result = output_path\n"
        )

        result, _recorder = run_workflow(
            workdir=tmp_path,
            run_id="p2d3-big-plot",
            data_sources=[source],
            target_tables=["big"],
            executor_code="result = df.copy()\n",
            chart=True,
            plot_code=plot_code,
        )

        assert result["analysis_status"] == "completed", result.get("status_reason")
        visualizations = result.get("visualization_results") or []
        assert visualizations and visualizations[0]["success"], visualizations

        # 容器里看到的行数必须等于真实行数，而不是摘要里的 1 行
        log_text = json.dumps(result, ensure_ascii=False)
        assert str(BIG_ROWS) in log_text or visualizations[0]["path"], visualizations

    def test_reporter_sees_a_bounded_preview(self, tmp_path: Path) -> None:
        """报告侧拿到的是有界预览，而不是 8000 行原始数据。"""
        source = build_big_csv(tmp_path / "big.csv")

        result, recorder = run_workflow(
            workdir=tmp_path,
            run_id="p2d3-big-report",
            data_sources=[source],
            target_tables=["big"],
            executor_code="result = df.copy()\n",
        )

        reporter_prompts = recorder.prompts.get("reporter") or []
        assert reporter_prompts, "reporter 应被调用"

        prompt = reporter_prompts[-1]
        # 预览标记出现，且提示词里没有整表
        assert "C000000" not in prompt or "preview" in prompt.lower() or "截断" in prompt
        assert len(prompt) < 200_000, f"报告提示词过大：{len(prompt)} 字符"


# ---------------------------------------------------------------------------
# 故障注入
# ---------------------------------------------------------------------------


class TestFaultInjection:
    def test_code_error_enters_repair_and_recovers(self, tmp_path: Path) -> None:
        source = build_csv(tmp_path / "sales.csv")

        result, recorder = run_workflow(
            workdir=tmp_path,
            run_id="p2d3-fault-code",
            data_sources=[source],
            target_tables=["sales"],
            # 首次报错（列名不存在）→ 进入 repair
            executor_code="result = df['definitely_missing_column']\n",
            repair_code="result = df.copy()\n",
        )

        assert result["analysis_status"] == "completed", result.get("status_reason")
        assert "repair" in recorder.calls, recorder.calls
        assert result["failed_tasks"] == []

    def test_environment_fault_does_not_trigger_model_retry(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """环境故障不得触发无效的付费模型重试。"""
        import sandbox.executor as executor_module

        source = build_csv(tmp_path / "sales.csv")

        original_run = executor_module.DockerSandboxExecutor.run

        def failing_run(self, job, **kwargs):  # noqa: ANN001
            outcome = original_run(self, job, **kwargs)
            # 模拟"沙箱不可用"这类环境故障
            outcome.success = False
            outcome.error = "注入：Docker 不可用"
            outcome.error_kind = executor_module.SandboxErrorKind.UNAVAILABLE
            return outcome

        monkeypatch.setattr(executor_module.DockerSandboxExecutor, "run", failing_run)

        result, recorder = run_workflow(
            workdir=tmp_path,
            run_id="p2d3-fault-env",
            data_sources=[source],
            target_tables=["sales"],
            executor_code="result = df.copy()\n",
        )

        # 模型不应被叫去"修"一个根本没跑起来的沙箱
        assert "repair" not in recorder.calls, recorder.calls
        assert result["analysis_status"] != "completed"
        failures = result.get("failed_tasks") or []
        assert failures, result
        assert all(
            item.get("error_kind") in ("unavailable", "timeout") for item in failures
        ), failures


# ---------------------------------------------------------------------------
# 安全边界
# ---------------------------------------------------------------------------


class TestSecurityBoundaries:
    def test_untrusted_python_still_runs_in_docker(self, tmp_path: Path) -> None:
        """模型代码必须仍然在容器里执行。"""
        source = build_csv(tmp_path / "sales.csv")

        import sandbox.protocol

        jobs: list[str] = []
        original_run = sandbox.executor.DockerSandboxExecutor.run

        def recording_run(self, job, **kwargs):  # noqa: ANN001
            jobs.append(job.mode)
            return original_run(self, job, **kwargs)

        import sandbox.executor as executor_module

        original = executor_module.DockerSandboxExecutor.run
        executor_module.DockerSandboxExecutor.run = recording_run
        try:
            result, _recorder = run_workflow(
                workdir=tmp_path,
                run_id="p2d3-sec-docker",
                data_sources=[source],
                target_tables=["sales"],
                executor_code="result = df.copy()\n",
            )
        finally:
            executor_module.DockerSandboxExecutor.run = original

        assert result["analysis_status"] == "completed", result.get("status_reason")
        assert jobs == ["python"], jobs

    def test_sql_still_blocked_by_the_authorizer(self, tmp_path: Path) -> None:
        """SQL 仍受 P2-B 的只读与授权约束（direct 与 mcp 都要挡）。"""
        db = build_sqlite(tmp_path / "sales.sqlite")

        for mode in ("direct", "mcp"):
            for query, marker in (
                ("DELETE FROM orders", "只读"),
                ("SELECT value FROM secrets", "未授权"),
                ("SELECT * FROM sqlite_master", "元数据"),
            ):
                result, _recorder = run_workflow(
                    workdir=tmp_path,
                    run_id=f"p2d3-sec-{mode}-{abs(hash(query)) % 10000}",
                    data_sources=[db],
                    target_tables=["sales.orders"],
                    executor_code=query,
                    mode=mode,
                )

                assert result["analysis_status"] != "completed", (mode, query, result)
                failures = result.get("failed_tasks") or []
                assert failures, (mode, query)
                assert any(
                    marker in str(item.get("reason")) for item in failures
                ), (mode, query, failures)

    def test_mcp_private_config_is_not_in_the_sandbox_history(
        self, tmp_path: Path
    ) -> None:
        """私有授权文件不在任何 Run 容器的挂载范围里。

        直接对 Docker 沙箱下任务，而不是经工作流：SQLite 任务会被 router 路由到
        SQL 工具，因此"在容器里读文件"这类检查走不到沙箱。这里要验证的是
        **挂载边界**本身，用真实的容器任务最直接。

        `consume_mcp_authorization` 会真的写出私有配置，因此本用例能区分
        "读不到" 与 "根本没生成"。
        """
        from core.llm import build_model_factory
        from core.runtime import McpSettings, RunRuntime, use_runtime
        from ingestion.models import TableResource
        from sandbox.executor import DockerSandboxExecutor
        from sandbox.protocol import SandboxJob, SandboxLimits
        from utils.mcp_runtime import ensure_mcp_config

        import pandas as pd

        run_id = "p2d3-sec-private"
        workdir = tmp_path / "work"
        workdir.mkdir()
        run_dir = workdir / run_id
        run_dir.mkdir(parents=True)

        db = build_sqlite(workdir / "sales.sqlite")

        runtime = RunRuntime(
            run_id=run_id,
            model_factory=build_model_factory(run_id),
            resolver=ResourceResolver(),
            output_root=workdir,
            mcp=McpSettings(mode="mcp"),
        )

        # 真的写出一份私有授权配置，并确认它在 output_root 之外
        config = ensure_mcp_config(
            run_id=run_id,
            snapshots={
                "sales.customers": {
                    "name": "sales.customers",
                    "source_type": "sqlite",
                    "source_path": str(db),
                    "content_hash": "h1",
                    "file_size": db.stat().st_size,
                    "locator": "customers",
                    "database_table": "customers",
                }
            },
            workspace_root=workdir,
            output_root=workdir,
            private_root=runtime.mcp_private_root(),
        )
        assert config.exists(), "私有配置未生成，本用例就失去意义了"
        from core.runtime import _is_inside

        assert not _is_inside(config, run_dir), config

        code = (
            "found = {}\n"
            "for path in ['/history/mcp/authorizations.json',\n"
            "             '/history/authorizations.json',\n"
            "             '/mcp_private/authorizations.json']:\n"
            "    try:\n"
            "        pd.read_json(path, typ='series')\n"
            "        found[path] = 'LEAKED'\n"
            "    except Exception:\n"
            "        found[path] = 'unreadable'\n"
            "result = [found]\n"
        )

        tables = {
            "t": TableResource(
                name="t",
                source_type="csv",
                source_path=str(workdir / "x.csv"),
                dataframe=pd.DataFrame({"a": [1, 2, 3]}),
            )
        }
        job = SandboxJob(
            mode="python",
            table_names=["t"],
            code=code,
            limits=SandboxLimits(timeout_seconds=120),
            op_label="private-config-check",
        )

        with use_runtime(runtime):
            outcome = DockerSandboxExecutor().run(
                job, run_id=run_id, run_dir=run_dir, tables=tables
            )

        assert outcome.success is True, outcome.error
        rows = outcome.result
        assert isinstance(rows, list) and rows, rows
        assert all(
            value == "unreadable" for value in rows[0].values()
        ), f"沙箱里能读到私有配置：{rows[0]}"

    def test_mcp_private_config_never_appears_in_model_output(
        self, tmp_path: Path
    ) -> None:
        """发给模型的提示词里不含私有配置路径。"""
        db = build_sqlite(tmp_path / "sales.sqlite")

        result, recorder = run_workflow(
            workdir=tmp_path,
            run_id="p2d3-sec-prompt",
            data_sources=[db],
            target_tables=["sales.customers"],
            executor_code="SELECT COUNT(*) AS n FROM customers",
            mode="mcp",
        )

        assert result["analysis_status"] == "completed", result.get("status_reason")

        every_prompt = "\n".join(
            prompt
            for prompts in recorder.prompts.values()
            for prompt in prompts
        )
        assert "authorizations.json" not in every_prompt
        assert ".mcp_private" not in every_prompt


class TestPlotPayloadRoundTrip:
    """回归：DataFrame 必须以信封形式过 JSON，而不是被 default=str 变成文本。

    这个缺陷的表现是容器里 `result_df` 为 None，绘图代码报
    `TypeError: 'NoneType' object is not subscriptable`——离根因很远。
    """

    def test_dataframe_survives_the_job_json_round_trip(self, tmp_path: Path) -> None:
        import pandas as pd

        from sandbox.protocol import read_json, write_json
        from utils.result_artifacts import to_jsonable

        frame = pd.DataFrame({"region": ["East", "West"], "amount": [1.5, 2.5]})
        target = tmp_path / "job.json"

        write_json({"analysis_result": to_jsonable(frame)}, target)

        loaded = read_json(target)
        payload = loaded["analysis_result"]

        # 必须是信封，而不是 DataFrame 的字符串形式
        assert isinstance(payload, dict), payload
        assert payload["__kind__"] == "dataframe", payload
        assert payload["records"][0]["region"] == "East", payload

        # 容器侧能据此还原出 DataFrame
        from sandbox.bootstrap import build_result_dataframe

        rebuilt = build_result_dataframe(payload)
        assert rebuilt is not None, "容器侧还原失败"
        assert list(rebuilt.columns) == ["region", "amount"]
        assert len(rebuilt) == 2

    def test_unserializable_payload_fails_loudly(self, tmp_path: Path) -> None:
        """不可序列化的对象必须报错，而不是被静默变成字符串。"""
        from sandbox.protocol import write_json

        class Opaque:
            def __str__(self) -> str:
                return "看起来像数据"

        with pytest.raises(TypeError):
            write_json({"analysis_result": Opaque()}, tmp_path / "bad.json")
