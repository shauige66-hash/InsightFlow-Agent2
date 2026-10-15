"""P2-C：Run 级数据授权、SQL 复用与工作流集成。

分三层验证：

1. **授权**：MCP Server 只看得到本次 Run 的快照，越权与同名不同内容都被拒绝；
2. **SQL**：经 MCP 执行与 direct 契约一致，且 P2-B 的加固一条都没少；
3. **集成**：真实 LangGraph 工作流在 mcp 模式下确实经过 Server 子进程，
   并且环境类失败不会触发无效 Repair。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from core.config import Settings
from core.llm import CallContext, ModelFactory
from core.runtime import McpSettings
from ingestion.resources import ResourceResolver
from mcp_bridge.client import McpStdioClient
from mcp_bridge.contract import (
    TOOL_DISCOVER_DATASETS,
    TOOL_GET_SCHEMA,
    TOOL_RUN_READONLY_SQL,
)
from mcp_bridge.registry import write_authorizations
from services.analysis_service import run_analysis
from tools.tool_registry import clear_tool_cache

from tests.conftest import CALL_RECORDS

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_sqlite(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER, amount REAL)"
        )
        connection.executemany(
            "INSERT INTO orders VALUES (?, ?, ?)",
            [(1, 101, 10.0), (2, 102, 20.0), (3, 101, 5.0)],
        )
        connection.commit()
    finally:
        connection.close()


def snapshot(name: str, source_type: str, path: Path, **extra: Any) -> dict:
    payload = {
        "name": name,
        "source_type": source_type,
        "source_path": str(path),
        "content_hash": extra.pop("content_hash", "hash-1"),
        "file_size": path.stat().st_size,
        "locator": extra.pop("locator", None),
        "database_table": extra.pop("database_table", None),
    }
    payload.update(extra)
    return payload


def make_config(
    workspace: Path, run_id: str, snapshots: dict[str, dict]
) -> Path:
    config, _refs = write_authorizations(
        run_id=run_id,
        snapshots=snapshots,
        workspace_root=workspace,
        output_root=workspace,
    )
    return config


class TestDataAuthorization:
    def test_only_current_run_datasets_are_discoverable(self, tmp_path: Path) -> None:
        db = tmp_path / "sales.sqlite"
        build_sqlite(db)

        config = make_config(
            tmp_path,
            "run-a",
            {"sales.orders": snapshot("sales.orders", "sqlite", db, database_table="orders")},
        )

        with McpStdioClient(config_path=config, project_root=PROJECT_ROOT) as client:
            outcome = client.call_tool(TOOL_DISCOVER_DATASETS, {})

        dataset_ids = [item["dataset_id"] for item in outcome["result"]["datasets"]]
        assert dataset_ids == ["run-a:sales.orders"]

    def test_dataset_listing_does_not_leak_host_paths(self, tmp_path: Path) -> None:
        """返回给模型的数据集描述里不能出现宿主绝对路径。"""
        db = tmp_path / "sales.sqlite"
        build_sqlite(db)
        config = make_config(
            tmp_path,
            "run-a",
            {"sales.orders": snapshot("sales.orders", "sqlite", db, database_table="orders")},
        )

        with McpStdioClient(config_path=config, project_root=PROJECT_ROOT) as client:
            outcome = client.call_tool(TOOL_DISCOVER_DATASETS, {})

        encoded = json.dumps(outcome["result"], ensure_ascii=False)
        assert str(db) not in encoded
        assert str(tmp_path) not in encoded

    def test_cross_run_dataset_id_is_refused(self, tmp_path: Path) -> None:
        """另一个 Run 的 dataset_id 不能被本次 Server 使用。"""
        db = tmp_path / "sales.sqlite"
        build_sqlite(db)
        config = make_config(
            tmp_path,
            "run-a",
            {"sales.orders": snapshot("sales.orders", "sqlite", db, database_table="orders")},
        )

        with McpStdioClient(config_path=config, project_root=PROJECT_ROOT) as client:
            outcome = client.call_tool(TOOL_GET_SCHEMA, {"dataset_id": "run-b:sales.orders"})

        assert outcome["success"] is False
        assert "run-b:sales.orders" in outcome["error"]

    def test_same_name_different_content_is_isolated(self, tmp_path: Path) -> None:
        """同名不同内容的两个数据集不能互相访问。"""
        first = tmp_path / "first.sqlite"
        second = tmp_path / "second.sqlite"
        build_sqlite(first)
        build_sqlite(second)

        config = make_config(
            tmp_path,
            "run-a",
            {
                "sales.orders": snapshot(
                    "sales.orders", "sqlite", first, database_table="orders",
                    content_hash="hash-first",
                )
            },
        )

        with McpStdioClient(config_path=config, project_root=PROJECT_ROOT) as client:
            listing = client.call_tool(TOOL_DISCOVER_DATASETS, {})
            # 只暴露本次 Run 那一份，另一份同名数据集不可见
            assert listing["result"]["count"] == 1
            assert listing["result"]["datasets"][0]["content_hash"] == "hash-first"

            absent = client.call_tool(TOOL_GET_SCHEMA, {"dataset_id": "run-a:second.orders"})
            assert absent["success"] is False

    def test_missing_file_fails_loudly(self, tmp_path: Path) -> None:
        db = tmp_path / "gone.sqlite"
        build_sqlite(db)
        snapshots = {"sales.orders": snapshot("sales.orders", "sqlite", db, database_table="orders")}
        config = make_config(tmp_path, "run-a", snapshots)
        db.unlink()

        with McpStdioClient(config_path=config, project_root=PROJECT_ROOT) as client:
            outcome = client.call_tool(TOOL_GET_SCHEMA, {"dataset_id": "run-a:sales.orders"})

        assert outcome["success"] is False
        assert "丢失" in outcome["error"]

    def test_path_outside_workspace_is_refused(self, tmp_path: Path) -> None:
        """配置若指向 workspace 之外，Server 必须拒绝。"""
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        outside = outside_dir / "secret.sqlite"
        build_sqlite(outside)

        workspace = tmp_path / "workspace"
        workspace.mkdir()

        # 手写配置：绕过宿主侧的写入校验，模拟配置被改动
        payload = {
            "run_id": "run-a",
            "workspace_root": str(workspace),
            "datasets": [
                {
                    "dataset_id": "run-a:sales.orders",
                    "name": "sales.orders",
                    "source_type": "sqlite",
                    "source_path": str(outside),
                    "locator": "orders",
                    "content_hash": "h",
                    "database_table": "orders",
                    "columns": [],
                    "row_count": None,
                }
            ],
        }
        config = workspace.parent / f"{workspace.name}.mcp_private" / "run-a" / "authorizations.json"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(json.dumps(payload), encoding="utf-8")

        with McpStdioClient(config_path=config, project_root=PROJECT_ROOT) as client:
            outcome = client.call_tool(TOOL_GET_SCHEMA, {"dataset_id": "run-a:sales.orders"})

        assert outcome["success"] is False
        assert "越界" in outcome["error"]


class TestSchemaDiscovery:
    def test_csv_schema(self, tmp_path: Path) -> None:
        csv = tmp_path / "customers.csv"
        csv.write_text("customer_id,region,amount\nC001,East,1.5\nC002,West,2.5\n", encoding="utf-8")
        config = make_config(tmp_path, "run-a", {"customers": snapshot("customers", "csv", csv)})

        with McpStdioClient(config_path=config, project_root=PROJECT_ROOT) as client:
            outcome = client.call_tool(TOOL_GET_SCHEMA, {"dataset_id": "run-a:customers"})

        columns = {column["name"]: column["inferred_type"] for column in outcome["result"]["columns"]}
        assert set(columns) == {"customer_id", "region", "amount"}
        assert columns["amount"] == "number"

    def test_excel_multi_sheet_schema(self, tmp_path: Path) -> None:
        """多 Sheet：每个 Sheet 是一个独立 dataset，各自只描述自己的列。"""
        workbook = tmp_path / "company.xlsx"
        with pd.ExcelWriter(workbook) as writer:
            pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_excel(
                writer, sheet_name="Sheet1", index=False
            )
            pd.DataFrame({"p": [1.0], "q": [2.0], "r": ["z"]}).to_excel(
                writer, sheet_name="Sheet2", index=False
            )

        config = make_config(
            tmp_path,
            "run-a",
            {
                "company.Sheet1": snapshot(
                    "company.Sheet1", "excel", workbook, locator="Sheet1"
                ),
                "company.Sheet2": snapshot(
                    "company.Sheet2", "excel", workbook, locator="Sheet2"
                ),
            },
        )

        with McpStdioClient(config_path=config, project_root=PROJECT_ROOT) as client:
            first = client.call_tool(TOOL_GET_SCHEMA, {"dataset_id": "run-a:company.Sheet1"})
            second = client.call_tool(TOOL_GET_SCHEMA, {"dataset_id": "run-a:company.Sheet2"})

        assert [c["name"] for c in first["result"]["columns"]] == ["a", "b"]
        assert [c["name"] for c in second["result"]["columns"]] == ["p", "q", "r"]

    def test_sqlite_schema(self, tmp_path: Path) -> None:
        db = tmp_path / "sales.sqlite"
        build_sqlite(db)
        config = make_config(
            tmp_path,
            "run-a",
            {"sales.orders": snapshot("sales.orders", "sqlite", db, database_table="orders")},
        )

        with McpStdioClient(config_path=config, project_root=PROJECT_ROOT) as client:
            outcome = client.call_tool(TOOL_GET_SCHEMA, {"dataset_id": "run-a:sales.orders"})

        names = [column["name"] for column in outcome["result"]["columns"]]
        assert names == ["order_id", "customer_id", "amount"]
        primary_keys = [
            column["name"] for column in outcome["result"]["columns"] if column["primary_key"]
        ]
        assert primary_keys == ["order_id"]


class TestSqlThroughMcp:
    @pytest.fixture
    def server(self, tmp_path: Path):
        db = tmp_path / "sales.sqlite"
        build_sqlite(db)
        config = make_config(
            tmp_path,
            "run-a",
            {"sales.orders": snapshot("sales.orders", "sqlite", db, database_table="orders")},
        )
        with McpStdioClient(config_path=config, project_root=PROJECT_ROOT) as client:
            yield client
        return

    def _query(self, client: McpStdioClient, sql: str, **kwargs: Any) -> dict:
        payload = {"query": sql, "dataset_ids": ["run-a:sales.orders"]}
        payload.update(kwargs)
        return client.call_tool(TOOL_RUN_READONLY_SQL, payload)

    def test_select_join_group_by(self, server: McpStdioClient) -> None:
        for sql in (
            "SELECT * FROM orders",
            "SELECT customer_id, SUM(amount) AS total FROM orders GROUP BY customer_id",
            "WITH big AS (SELECT * FROM orders WHERE amount > 5) SELECT COUNT(*) AS n FROM big",
        ):
            outcome = self._query(server, sql)
            assert outcome["success"] is True, sql

    def test_write_statements_are_refused(self, server: McpStdioClient) -> None:
        for sql in ("DELETE FROM orders", "UPDATE orders SET amount = 0", "DROP TABLE orders"):
            outcome = self._query(server, sql)
            assert outcome["success"] is False, sql

    def test_unauthorized_table_is_refused(self, tmp_path: Path) -> None:
        """同一库里未被授权的表不能被读到。"""
        db = tmp_path / "sales.sqlite"
        build_sqlite(db)
        connection = sqlite3.connect(db)
        connection.execute("CREATE TABLE secrets (value TEXT)")
        connection.execute("INSERT INTO secrets VALUES ('top-secret')")
        connection.commit()
        connection.close()

        config = make_config(
            tmp_path,
            "run-a",
            {"sales.orders": snapshot("sales.orders", "sqlite", db, database_table="orders")},
        )

        with McpStdioClient(config_path=config, project_root=PROJECT_ROOT) as client:
            outcome = self._query(client, "SELECT * FROM secrets")

        assert outcome["success"] is False
        assert "未授权" in outcome["error"]

    def test_metadata_table_is_refused(self, server: McpStdioClient) -> None:
        outcome = self._query(server, "SELECT name FROM sqlite_master")

        assert outcome["success"] is False
        assert "元数据" in outcome["error"]

    def test_truncation_is_flagged(self, tmp_path: Path) -> None:
        """结果被截断时必须显式标记，与 P2-B 契约一致。"""
        db = tmp_path / "big.sqlite"
        connection = sqlite3.connect(db)
        connection.execute("CREATE TABLE big (n INTEGER)")
        connection.executemany("INSERT INTO big VALUES (?)", [(i,) for i in range(500)])
        connection.commit()
        connection.close()

        # 通过配置把行数上限压到 10
        config, _refs = write_authorizations(
            run_id="run-a",
            snapshots={"big": snapshot("big", "sqlite", db, database_table="big")},
            workspace_root=tmp_path,
            output_root=tmp_path,
            limits={"max_rows": 10},
        )

        with McpStdioClient(config_path=config, project_root=PROJECT_ROOT) as client:
            outcome = client.call_tool(
                TOOL_RUN_READONLY_SQL,
                {"query": "SELECT * FROM big", "dataset_ids": ["run-a:big"]},
            )

        assert outcome["success"] is True
        assert outcome["result"]["row_count"] == 10
        assert outcome["result"]["truncated"] is True

    def test_database_is_unchanged_after_attacks(self, tmp_path: Path) -> None:
        db = tmp_path / "sales.sqlite"
        build_sqlite(db)
        config = make_config(
            tmp_path,
            "run-a",
            {"sales.orders": snapshot("sales.orders", "sqlite", db, database_table="orders")},
        )

        def fingerprint() -> tuple:
            connection = sqlite3.connect(db)
            try:
                tables = sorted(
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                )
                rows = connection.execute("SELECT * FROM orders ORDER BY order_id").fetchall()
                return tuple(tables), tuple(rows)
            finally:
                connection.close()

        before = fingerprint()
        with McpStdioClient(config_path=config, project_root=PROJECT_ROOT) as client:
            for sql in (
                "DELETE FROM orders",
                "UPDATE orders SET amount = 0",
                "DROP TABLE orders",
                "INSERT INTO orders VALUES (99, 101, 1.0)",
            ):
                client.call_tool(
                    TOOL_RUN_READONLY_SQL,
                    {"query": sql, "dataset_ids": ["run-a:sales.orders"]},
                )

        assert fingerprint() == before


# --- 工作流集成 -----------------------------------------------------------

PLAN = {
    "tasks": [
        {
            "task_id": "task_1",
            "description": "统计订单总额",
            "target_tables": ["sales.orders"],
            "expected_output": "总金额",
        }
    ]
}

REPLIES: dict[str, Any] = {
    "planner": PLAN,
    "executor": {"code": "SELECT SUM(amount) AS total FROM orders"},
    "repair": {"code": "SELECT SUM(amount) AS total FROM orders"},
    "critic": {"validation_passed": True, "feedback": "ok"},
    "visualizer": {"charts": []},
    "reporter": "# Analysis Report\n\n## Overview\n订单总额已统计。\n",
}


class RecordingFactory:
    def __init__(self, run_id: str, overrides: dict[str, Any] | None = None) -> None:
        self.run_id = run_id
        self.replies = {**REPLIES, **(overrides or {})}
        self.calls: list[str] = []
        CALL_RECORDS[run_id] = self.calls

    def build(self) -> ModelFactory:
        owner = self

        class _Scripted(BaseChatModel):
            agent_name: str = ""

            @property
            def _llm_type(self) -> str:
                return f"p2c-{self.agent_name}"

            def _generate(
                self,
                messages: list[BaseMessage],
                stop: list[str] | None = None,
                run_manager: CallbackManagerForLLMRun | None = None,
                **kwargs: Any,
            ) -> ChatResult:
                owner.calls.append(self.agent_name)
                reply = owner.replies[self.agent_name]
                content = (
                    reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)
                )
                return ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content=content))]
                )

        def resolver(agent_name: str) -> _Scripted:
            return _Scripted(agent_name=agent_name)

        return ModelFactory(
            settings=Settings(
                api_key="sk-fake", base_url="http://127.0.0.1:8000/v1", model="deepseek-chat"
            ),
            context=CallContext(run_id=self.run_id, client_app="insightflow-agent2"),
            resolver=resolver,
        )


def run_workflow(
    tmp_path: Path,
    run_id: str,
    *,
    mode: str,
    overrides: dict[str, Any] | None = None,
    **kwargs: Any,
):
    db = tmp_path / "sales.sqlite"
    build_sqlite(db)
    recorder = RecordingFactory(run_id, overrides)
    clear_tool_cache()

    result = run_analysis(
        run_id=run_id,
        data_sources=[str(db)],
        model_factory=recorder.build(),
        resolver=ResourceResolver(),
        output_root=tmp_path,
        mcp=McpSettings(mode=mode),
        **kwargs,
    )
    return result, recorder


class TestWorkflowIntegration:
    def test_direct_and_mcp_produce_the_same_result(self, tmp_path: Path) -> None:
        """两种后端的 SQL 结果必须语义一致。"""
        direct_dir = tmp_path / "direct"
        mcp_dir = tmp_path / "mcp"
        direct_dir.mkdir()
        mcp_dir.mkdir()

        direct, _ = run_workflow(direct_dir, "wf-direct", mode="direct")
        via_mcp, _ = run_workflow(mcp_dir, "wf-mcp", mode="mcp")

        def successful(result: dict) -> list[dict]:
            return [item for item in result["execution_results"] if item["success"]]

        assert successful(direct) and successful(via_mcp)
        assert successful(direct)[-1]["result"] == successful(via_mcp)[-1]["result"]
        # 注意变量名不能叫 mcp：那会遮住模块级的 `mcp` 包
        assert direct["run_status"] == via_mcp["run_status"] == "completed"

    def test_mcp_mode_really_goes_through_the_server(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """确认 mcp 模式确实启动了 Server 子进程并调用了工具。

        仅比对结果不够——direct 也能给出同样的结果。这里统计真实的
        `McpStdioClient` 使用次数。
        """
        import tools.mcp_sql_tool as mcp_sql_tool

        original = mcp_sql_tool.McpStdioClient
        stats = {"starts": 0, "calls": 0, "tools": []}

        class Counting(original):  # type: ignore[misc, valid-type]
            def start(self):
                stats["starts"] += 1
                return super().start()

            def call_tool(self, name, arguments=None, **kwargs):
                stats["calls"] += 1
                stats["tools"].append(name)
                return super().call_tool(name, arguments, **kwargs)

        monkeypatch.setattr(mcp_sql_tool, "McpStdioClient", Counting)

        run_workflow(tmp_path, "wf-mcp-real", mode="mcp")

        assert stats["starts"] == 1, "未启动 MCP Server"
        # list_tools 发生在 start() 内部、不走 call_tool，
        # 因此这里只断言真正发生了一次工具调用
        assert stats["calls"] == 1, f"MCP 工具调用次数异常：{stats['tools']}"
        assert TOOL_RUN_READONLY_SQL in stats["tools"]

    def test_direct_mode_does_not_start_a_server(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """direct 模式不得偷偷启动 MCP。"""
        import tools.mcp_sql_tool as mcp_sql_tool

        called = {"starts": 0}
        original = mcp_sql_tool.McpStdioClient

        class Counting(original):  # type: ignore[misc, valid-type]
            def start(self):
                called["starts"] += 1
                return super().start()

        monkeypatch.setattr(mcp_sql_tool, "McpStdioClient", Counting)

        run_workflow(tmp_path, "wf-direct-plain", mode="direct")

        assert called["starts"] == 0

    def test_mcp_unavailable_does_not_trigger_repair(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """MCP 通道不可用属于环境失败：不应进入 LLM 修复循环。"""
        import tools.mcp_sql_tool as mcp_sql_tool
        from mcp_bridge.client import McpUnavailable

        class Broken(mcp_sql_tool.McpStdioClient):  # type: ignore[misc, valid-type]
            def start(self):
                raise McpUnavailable("模拟 MCP 通道不可用")

        monkeypatch.setattr(mcp_sql_tool, "McpStdioClient", Broken)

        result, recorder = run_workflow(tmp_path, "wf-mcp-down", mode="mcp")

        assert recorder.calls.count("repair") == 0, recorder.calls
        assert result["failed_tasks"]
        assert result["failed_tasks"][0]["error_kind"] == "unavailable"

    def test_mcp_timeout_does_not_trigger_repair(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        import tools.mcp_sql_tool as mcp_sql_tool
        from mcp_bridge.client import McpTimeout

        class SlowStart(mcp_sql_tool.McpStdioClient):  # type: ignore[misc, valid-type]
            def start(self):
                raise McpTimeout("模拟 MCP 启动超时")

        monkeypatch.setattr(mcp_sql_tool, "McpStdioClient", SlowStart)

        result, recorder = run_workflow(tmp_path, "wf-mcp-slow", mode="mcp")

        assert recorder.calls.count("repair") == 0
        assert result["failed_tasks"][0]["error_kind"] == "timeout"

    def test_sql_error_enters_bounded_repair(self, tmp_path: Path) -> None:
        """SQL 自身写错（表不存在）仍应进入有界修复循环。"""
        overrides = {
            "executor": {"code": "SELECT * FROM no_such_table"},
            "repair": {"code": "SELECT * FROM no_such_table"},
            "critic": {"validation_passed": True, "feedback": "ok"},
        }

        result, recorder = run_workflow(
            tmp_path, "wf-mcp-sql-error", mode="mcp", overrides=overrides
        )

        assert recorder.calls.count("repair") > 0, recorder.calls
        assert result["failed_tasks"]

    def test_two_runs_have_separate_authorizations(self, tmp_path: Path) -> None:
        """两个 Run 的授权互不可见。"""
        first_dir = tmp_path / "run-a"
        second_dir = tmp_path / "run-b"
        first_dir.mkdir()
        second_dir.mkdir()

        first, _ = run_workflow(first_dir, "iso-run-a", mode="mcp")
        second, _ = run_workflow(second_dir, "iso-run-b", mode="mcp")

        assert first["run_status"] == "completed"
        assert second["run_status"] == "completed"

        # 各自的授权配置只含自己的 dataset_id，且都在 run 目录之外
        for directory, run_id in ((first_dir, "iso-run-a"), (second_dir, "iso-run-b")):
            config = directory.parent / f"{directory.name}.mcp_private" / run_id / "authorizations.json"
            payload = json.loads(config.read_text(encoding="utf-8"))
            ids = [item["dataset_id"] for item in payload["datasets"]]
            assert ids, config
            assert all(item.startswith(f"{run_id}:") for item in ids)
            assert not str(config).startswith(str(directory / run_id))

    def test_repeated_resume_does_not_duplicate_results(self, tmp_path: Path) -> None:
        """重复 Resume 不增加已确认完成的业务结果。"""
        from langgraph.checkpoint.sqlite import SqliteSaver

        db = tmp_path / "sales.sqlite"
        build_sqlite(db)
        run_id = "wf-mcp-resume"
        recorder = RecordingFactory(run_id)
        saver_db = tmp_path / "checkpoints.sqlite"

        with SqliteSaver.from_conn_string(str(saver_db)) as saver:
            first = run_analysis(
                run_id=run_id,
                data_sources=[str(db)],
                model_factory=recorder.build(),
                resolver=ResourceResolver(),
                output_root=tmp_path,
                mcp=McpSettings(mode="mcp"),
                checkpointer=saver,
            )
            count_after_first = len(first["execution_results"])

            second = run_analysis(
                run_id=run_id,
                data_sources=[str(db)],
                model_factory=recorder.build(),
                resolver=ResourceResolver(),
                output_root=tmp_path,
                mcp=McpSettings(mode="mcp"),
                checkpointer=saver,
                continue_from_checkpoint=True,
            )

        assert len(second["execution_results"]) == count_after_first
        assert second["run_status"] == "completed"
