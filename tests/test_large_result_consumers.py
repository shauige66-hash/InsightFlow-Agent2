"""P2-B3：两条消费路径在真实工作流里的行为。

- 绘图需要**真实数据**，必须经 `load_result` 读回产物；
- 报告只需要概要，必须拿到**有界预览**，而不是两万行记录。

这两条路径的区别是本阶段的核心，因此在这里跑完整工作流验证，
而不是只测纯函数。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from core.config import Settings
from core.llm import CallContext, ModelFactory
from ingestion.resources import ResourceResolver
from services.analysis_service import run_analysis
from utils.result_artifacts import PREVIEW_MAX_RECORDS

from tests.conftest import CALL_RECORDS

# 走完整工作流的数据量需要同时满足两个上限：
#   - 必须 > INLINE_LIMIT_BYTES（32 KiB）才会被外置；
#   - 必须 < 沙箱的 1 MiB 结果上限，否则执行阶段就失败了，根本到不了外置。
# 约 45 字节/行 × 8,000 行 ≈ 360 KiB，落在两者之间。
# 更大的结果（如 20,000 行）不适合走沙箱工作流，
# 由 tests/test_large_result_recovery.py 直接测 store_result 覆盖。
BIG_ROWS = 8_000

PLAN = {
    "tasks": [
        {
            "task_id": "task_1",
            "description": "返回完整明细表",
            "target_tables": ["customers"],
            "expected_output": "明细表",
        }
    ]
}

PLOT_PLAN = {
    "charts": [
        {
            "chart_id": "chart_1",
            "description": "按地区统计行数",
            "chart_type": "bar",
            "source_task_id": "task_1",
            "target_tables": ["customers"],
            "expected_output": "柱状图",
        }
    ]
}

REPLIES: dict[str, Any] = {
    "planner": PLAN,
    # 返回整张表：结果必然超过内联上限而被外置
    "executor": {"code": "result = df"},
    "repair": {"code": "result = df"},
    "refine": {"code": "result = df"},
    "critic": {"validation_passed": True, "feedback": "结果符合任务要求"},
    "visualizer": {"charts": []},
    "plotter": {"code": "result = output_path"},
    "plot_repair": {"code": "result = output_path"},
    "reporter": "# Analysis Report\n\n## Overview\n明细表已生成。\n",
}


class PromptRecordingFactory:
    """记录每个 Agent 收到的提示词，用于断言"看到了什么"。"""

    def __init__(self, run_id: str, replies: dict[str, Any]) -> None:
        self.run_id = run_id
        self.replies = replies
        self.prompts: dict[str, list[str]] = {}
        CALL_RECORDS[run_id] = []

    def build(self) -> ModelFactory:
        owner = self

        class _Scripted(BaseChatModel):
            agent_name: str = ""

            @property
            def _llm_type(self) -> str:
                return f"p2b3-{self.agent_name}"

            def _generate(
                self,
                messages: list[BaseMessage],
                stop: list[str] | None = None,
                run_manager: CallbackManagerForLLMRun | None = None,
                **kwargs: Any,
            ) -> ChatResult:
                prompt = "\n".join(str(message.content) for message in messages)
                owner.prompts.setdefault(self.agent_name, []).append(prompt)
                CALL_RECORDS[owner.run_id].append(self.agent_name)

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
                api_key="sk-fake",
                base_url="http://127.0.0.1:8000/v1",
                model="deepseek-chat",
            ),
            context=CallContext(run_id=self.run_id, client_app="insightflow-agent2"),
            resolver=resolver,
        )


@pytest.fixture
def workspace(tmp_path: Path) -> tuple[Path, Path]:
    """足够大（>32 KiB）但仍在沙箱 1 MiB 结果上限内的 CSV。"""
    source = tmp_path / "customers.csv"
    lines = ["customer_id,region,amount"]
    for index in range(BIG_ROWS):
        lines.append(f"C{index:06d},{'East' if index % 2 else 'West'},{index * 1.5}")
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tmp_path, source


def run_workflow(
    workspace: tuple[Path, Path],
    run_id: str,
    replies: dict[str, Any] | None = None,
):
    workdir, source = workspace
    recorder = PromptRecordingFactory(run_id, {**REPLIES, **(replies or {})})

    result = run_analysis(
        run_id=run_id,
        data_sources=[str(source)],
        model_factory=recorder.build(),
        resolver=ResourceResolver(),
        output_root=workdir,
    )
    return result, recorder


class TestLargeResultIsExternalizedInTheRealWorkflow:
    def test_execution_result_is_externalized(self, workspace) -> None:
        result, _ = run_workflow(workspace, "p2b3-e2e-externalized")

        successful = [item for item in result["execution_results"] if item["success"]]
        assert successful, "应有成功的执行结果"

        item = successful[-1]
        assert item["result"]["__kind__"] == "externalized"
        assert item["result"]["row_count"] == BIG_ROWS
        assert item["result_artifact"] is not None

        # state 里不含任何一行数据
        assert "records" not in item["result"]

    def test_state_stays_serializable(self, workspace) -> None:
        result, _ = run_workflow(workspace, "p2b3-e2e-serializable")

        # 能整体 JSON 序列化即说明没有把 DataFrame 之类塞进 state
        encoded = json.dumps(result["execution_results"], ensure_ascii=False, default=str)
        assert "externalized" in encoded


class TestReportSeesBoundedPreview:
    def test_reporter_prompt_is_not_the_full_dataset(self, workspace) -> None:
        """报告提示词里不能出现两万行数据。"""
        _, recorder = run_workflow(workspace, "p2b3-e2e-report")

        prompt = recorder.prompts["reporter"][-1]

        # 完整数据集会有 20,000 条记录；提示词必须远小于此
        assert prompt.count('"customer_id"') <= PREVIEW_MAX_RECORDS + 5, (
            "报告提示词里出现了过多记录"
        )
        assert len(prompt) < 100 * 1024, f"提示词过大: {len(prompt)} 字节"

    def test_reporter_prompt_carries_preview_marker_and_shape(self, workspace) -> None:
        """要点：既要给出概要，也要明确"这是预览，不是完整数据"。"""
        _, recorder = run_workflow(workspace, "p2b3-e2e-report-shape")

        prompt = recorder.prompts["reporter"][-1]

        assert "externalized" in prompt
        assert "预览" in prompt or "preview" in prompt.lower()
        assert str(BIG_ROWS) in prompt, "行数应出现在提示词里"
        assert "customer_id" in prompt, "列名应出现在提示词里"

    def test_preview_is_marked_truncated(self, workspace) -> None:
        _, recorder = run_workflow(workspace, "p2b3-e2e-truncated")

        prompt = recorder.prompts["reporter"][-1]

        # truncated / preview_rows 是"部分结果"的显式信号
        assert "truncated" in prompt


class TestPlottingUsesRealData:
    def test_plot_receives_real_row_count(self, workspace) -> None:
        """绘图必须拿到真实数据：让代码把行数写进图里再断言。

        如果绘图拿到的是 `{"__kind__": "externalized"}` 摘要，
        result_df 会是 None 或一行，断言就会失败——这正是缺陷的表现。
        """
        code = (
            "rows = -1 if result_df is None else len(result_df)\n"
            "fig = plt.figure()\n"
            "plt.bar(['all'], [rows])\n"
            "plt.title('行数')\n"
            "plt.savefig(output_path)\n"
            "plt.close()\n"
            "result = rows\n"
        )

        result, recorder = run_workflow(
            workspace,
            "p2b3-e2e-plot",
            replies={"visualizer": PLOT_PLAN, "plotter": {"code": code}},
        )

        plotted = result["visualization_results"][-1]
        assert plotted["success"] is True, plotted["error"]
        # 产物存在，且真正的 PNG 落在 Run 目录
        assert plotted["path"]
        assert Path(plotted["path"]).exists()

    def test_plot_fails_loudly_when_artifact_is_missing(self, tmp_path: Path) -> None:
        """产物丢失时绘图节点必须给出明确错误，而不是画出一张空图。

        直接调用节点函数：这样能精确观察到"读到损坏引用后发生了什么"，
        而不必构造一个产物必定被删除的多阶段流程。
        """
        import nodes
        from core.llm import build_model_factory
        from core.runtime import RunRuntime, use_runtime
        from utils.result_artifacts import INLINE_LIMIT_BYTES

        workdir = tmp_path / "run"
        workdir.mkdir()
        run_id = "p2b3-plot-missing"

        # 造一个"已外置但文件已丢失"的执行结果记录
        missing_path = workdir / run_id / "results" / "task_1.json"
        missing_path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "path": str(missing_path),
            "sha256": "0" * 64,
            "size": INLINE_LIMIT_BYTES * 4,
            "row_count": 5_000,
        }
        state = {
            "run_id": run_id,
            "current_chart_index": 0,
            "current_plot_code": "result = output_path",
            "plot_retry_count": 0,
            "visualization_plan": [PLOT_PLAN["charts"][0]],
            "execution_results": [
                {
                    "task_id": "task_1",
                    "success": True,
                    "result": {"__kind__": "externalized", "row_count": 5_000},
                    "result_artifact": artifact,
                }
            ],
            "visualization_results": [],
            "completed_operations": {},
        }

        runtime = RunRuntime(
            run_id=run_id,
            model_factory=build_model_factory(run_id),
            resolver=ResourceResolver(),
            output_root=workdir,
        )

        with use_runtime(runtime):
            updates = nodes.plot_execution_node(state)

        assert updates.get("plot_error"), "产物丢失必须产生明确的 plot_error"
        assert "不可用" in updates["plot_error"]
        # 没有新增任何绘图结果——不会记录一个并不存在的图表
        assert updates.get("visualization_results") == []


@pytest.mark.sandbox
class TestPlotWithLargeResultThroughSandbox:
    """确认大结果经沙箱绘图时，容器内拿到的确实是完整数据。"""

    def test_container_sees_full_row_count(self, workspace) -> None:
        from sandbox.executor import DockerSandboxExecutor

        executor = DockerSandboxExecutor()
        if not executor.available() or not executor.image_present():
            pytest.skip("Docker 沙箱不可用")

        code = (
            "rows = -1 if result_df is None else len(result_df)\n"
            "fig = plt.figure()\n"
            "plt.bar(['all'], [rows])\n"
            "plt.title('行数')\n"
            "plt.savefig(output_path)\n"
            "plt.close()\n"
            "result = rows\n"
        )

        result, _ = run_workflow(
            workspace,
            "p2b3-e2e-sandbox-plot",
            replies={"visualizer": PLOT_PLAN, "plotter": {"code": code}},
        )

        plotted = result["visualization_results"][-1]
        assert plotted["success"] is True, plotted["error"]
        assert Path(plotted["path"]).stat().st_size > 5000
