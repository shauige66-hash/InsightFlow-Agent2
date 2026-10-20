"""五项确认缺陷的判别性用例：先失败，修复后通过。

每个用例对应一个具体的行为缺陷，断言的是外部可观察的结果，不是实现细节：

1. 运行期提前结束时，状态被错误地报成 completed；
2. 决策步数达到上限后仍然发出一次决策请求；
3. 单次长耗时操作只按 60 秒计入执行时间预算；
4. 知识库更新后，同一任务仍复用旧版本的检索证据；
5. 会话启动失败时，轮次永远停在 pending。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

import storage
from services.analysis_service import run_analysis
from tests.scripted_queue import ScriptedQueueFactory, decision, task
from tests.test_report_approval import build_database

VIP_V1 = "# 客户分类\n\nVIP 客户指年消费金额超过 10000 元的客户。\n"
VIP_V2 = "# 客户分类\n\nVIP 客户指年消费金额超过 50000 元的客户。\n"


@pytest.fixture(autouse=True)
def _local_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("INSIGHTFLOW_STORAGE", "local")
    monkeypatch.setenv("INSIGHTFLOW_LOCAL_DB", str(tmp_path / "conversations.sqlite"))
    monkeypatch.setenv("INSIGHTFLOW_OWNER", "quality")
    monkeypatch.delenv("INSIGHTFLOW_REDIS_ENABLED", raising=False)
    storage.reset()
    storage.configure()
    yield
    storage.reset()


@pytest.fixture
def database(tmp_path: Path) -> Path:
    path = tmp_path / "customers.sqlite"
    build_database(path)
    return path


@pytest.fixture(autouse=True)
def _fake_tool(monkeypatch: pytest.MonkeyPatch):
    from harness import tool_dispatcher

    def fake_invoke(tool_name, tables, arguments, *, snapshots=None):
        return {"success": True, "result": [{"n": 3}], "error": None, "error_kind": None}

    monkeypatch.setattr(tool_dispatcher, "invoke_tool", fake_invoke)


def base_queues(**overrides: list[Any]) -> dict[str, list[Any]]:
    queues: dict[str, list[Any]] = {
        "executor": [{"code": "SELECT COUNT(*) AS n FROM customers"}],
        "repair": [{"code": "SELECT COUNT(*) AS n FROM customers"}],
        "refine": [{"code": "SELECT COUNT(*) AS n FROM customers"}],
        "critic": [{"validation_passed": True, "feedback": "ok"}],
        "visualizer": [{"charts": []}],
        "reporter": ["# 报告\n"],
    }
    queues.update(overrides)
    return queues


class TestDefect1EarlyFinishStatus:
    """提前结束发生在任务失败之后时，不能报成 completed。"""

    def test_finish_after_a_failed_task_reports_partial(self, database, tmp_path) -> None:
        factory = ScriptedQueueFactory(
            "defect1",
            base_queues(
                planner=[{"tasks": [task("task_1")]}],
                critic=[
                    {"validation_passed": False, "feedback": "结果缺少细分"},
                ],
                decider=[decision("finish", "无法在现有数据上完成，提前结束")],
            ),
        )
        result = run_analysis(
            run_id="defect1",
            data_sources=[str(database)],
            model_factory=factory.build(),
            output_root=tmp_path,
            enable_dynamic_loop=True,
        )

        assert result["current_action"] == "finish"
        assert result["analysis_status"] == "partial", (
            "任务未通过校验却提前结束，状态必须是 partial"
        )
        assert result["termination_reason"]


class TestDefect2DecisionCap:
    """达到决策步数上限后不应再为"决策"发出模型请求。"""

    def test_the_cap_does_not_spend_one_more_decision_request(
        self, database, tmp_path
    ) -> None:
        factory = ScriptedQueueFactory(
            "defect2",
            base_queues(
                planner=[
                    {
                        "tasks": [
                            task("task_1"),
                            task("task_2", description="第二个任务"),
                            task("task_3", description="第三个任务"),
                        ]
                    }
                ],
                decider=[decision("continue")],
            ),
        )
        result = run_analysis(
            run_id="defect2",
            data_sources=[str(database)],
            model_factory=factory.build(),
            output_root=tmp_path,
            enable_dynamic_loop=True,
            dynamic_limits={"max_decision_steps": 1},
        )

        assert factory.calls.count("decider") == 1, (
            "上限处仍发出了决策请求：" f"{factory.calls.count('decider')} 次"
        )
        assert result["analysis_status"] == "partial"
        assert "上限" in (result["termination_reason"] or "")


class TestDefect3LongOperationBudget:
    """单次长耗时操作必须按真实耗时计入执行预算。"""

    def test_a_long_operation_is_charged_in_full(self) -> None:
        from utils.agent_loop import (
            BUDGET_FIELD,
            AgentBudget,
            capture_active_time,
            new_usage,
            settle_active_time,
        )

        started = 1_000_000.0
        state: dict[str, Any] = {
            BUDGET_FIELD: {
                "limits": AgentBudget().to_dict(),
                "usage": {**new_usage()},
            }
        }
        # 与生产调用一致：这些函数返回增量，调用方写回 state
        state.update(capture_active_time(state, now=started))
        state.update(settle_active_time(state, now=started + 600.0))

        usage = state[BUDGET_FIELD]["usage"]
        assert usage["active_seconds"] >= 599.0, (
            f"600 秒的操作只计入了 {usage['active_seconds']} 秒"
        )

    def test_a_crash_gap_is_still_capped(self) -> None:
        """跨进程的陈旧时钟不能被当成"执行了这么久"。"""
        from utils.agent_loop import (
            BUDGET_FIELD,
            AgentBudget,
            new_usage,
            settle_active_time,
        )

        now = time.time()
        usage = {
            **new_usage(),
            "active_seconds": 10.0,
            "last_active_at": now - 3600.0,
            "clock_open": True,
            # 不是当前进程打开的时钟：可能来自崩溃前的进程
            "clock_token": "stale-process",
        }
        state = {BUDGET_FIELD: {"limits": AgentBudget().to_dict(), "usage": usage}}
        state.update(settle_active_time(state, now=now))

        assert state[BUDGET_FIELD]["usage"]["active_seconds"] <= 70.0


class TestDefect4StaleEvidence:
    """知识库更新后，同一任务不能继续复用旧版本的证据。"""

    def test_updated_knowledge_replaces_the_old_evidence(
        self, tmp_path, monkeypatch
    ) -> None:
        from services.knowledge_service import task_knowledge_updates

        knowledge = tmp_path / "vip.md"
        knowledge.write_text(VIP_V1, encoding="utf-8")
        state: dict[str, Any] = {
            "run_id": "defect4",
            "analysis_focus": "VIP 客户",
            "knowledge_sources": [str(knowledge)],
        }
        current_task = task("task_1", description="统计 VIP 客户数量")

        evidence, updates = task_knowledge_updates(state, current_task)
        assert evidence, "首次检索没有拿到证据"
        state.update(updates)
        assert "10000" in " ".join(item["content"] for item in evidence)
        old_version = state.get("knowledge_version")

        # 同一个 Run 里知识库被更新：版本变了，旧证据必须失效
        knowledge.write_text(VIP_V2, encoding="utf-8")

        evidence, updates = task_knowledge_updates(state, current_task)
        state.update(updates)
        text = " ".join(item["content"] for item in evidence)

        assert "50000" in text, "更新后的知识没有被检索到"
        assert "10000" not in text, "仍在复用旧版本的证据"
        assert state["knowledge_version"] != old_version


class TestDefect6AppendixNewline:
    """报告来源附录必须是真正的换行，而不是字面量 `\\n`。"""

    def test_the_citations_appendix_uses_real_newlines(self) -> None:
        import nodes

        state = {
            "task_knowledge": {
                "task_1": [
                    {
                        "chunk_id": "c1" * 8,
                        "source": "vip.md",
                        "page": 2,
                        "score": 0.71,
                        "content": "VIP 定义",
                    },
                    {
                        "chunk_id": "c2" * 8,
                        "source": "shipping.md",
                        "page": None,
                        "score": 0.55,
                        "content": "配送时效",
                    },
                ]
            }
        }

        appendix = nodes._citations_appendix(state)

        assert "\\n" not in appendix, "附录里有字面量反斜杠 n"
        assert "\n" in appendix
        assert appendix.startswith("\n## 参考来源")
        lines = appendix.splitlines()
        assert lines[1] == "## 参考来源"
        assert lines[2] == ""
        # 引用按来源排序，这里只断言内容而不假设顺序
        bullets = {line.split("（")[0] for line in lines[3:]}
        assert bullets == {"- vip.md，第 2 页", "- shipping.md"}


class TestDefect5ConversationStartWindow:
    """会话启动失败后，轮次必须收敛为 failed，而不是永远 pending。"""

    def test_a_failed_start_marks_the_turn_failed(self, tmp_path) -> None:
        from services.conversation_service import ConversationService

        class BrokenCoordinator:
            store = type("Store", (), {"get": staticmethod(lambda run_id: None)})()

            def start(self, run_id, **kwargs):
                raise RuntimeError("线程池已关闭")

        service = ConversationService(BrokenCoordinator())
        session = service.create_session(
            title="启动失败", dataset_refs=[str(tmp_path / "data.csv")]
        )

        with pytest.raises(RuntimeError):
            service.post_message(session["session_id"], "分析一下")

        turns = service.list_turns(session["session_id"])
        assert len(turns) == 1
        assert turns[0]["status"] == "failed", (
            f"启动失败后轮次仍是 {turns[0]['status']}"
        )
        assert turns[0]["status_reason"]


class TestDefect7ChartsMissingFromReport:
    """图表生成了，报告里却没有引用它——读者看到的是一份没有图的"图文报告"。"""

    def test_the_report_embeds_generated_charts(self) -> None:
        import nodes

        state = {
            "visualization_results": [
                {
                    "chart_id": "chart_1",
                    "title": "月度 GMV",
                    "path": "outputs/run_x/chart_1.png",
                }
            ]
        }

        appendix = nodes._charts_appendix(state, "# 报告\n\n正文没有任何图片。")

        assert "## 图表" in appendix
        assert "![月度 GMV](chart_1.png)" in appendix

    def test_a_referenced_chart_is_not_duplicated(self) -> None:
        import nodes

        state = {
            "visualization_results": [
                {"chart_id": "chart_1", "path": "outputs/run_x/chart_1.png"}
            ]
        }

        assert nodes._charts_appendix(state, "正文已引用 ![图](chart_1.png)") == ""

    def test_a_run_without_charts_adds_no_section(self) -> None:
        import nodes

        assert nodes._charts_appendix({}, "# 报告") == ""
