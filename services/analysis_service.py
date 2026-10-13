"""统一封装 InsightFlow 工作流运行逻辑，供命令行和 API 入口复用。"""

from typing import Any

from core.inference import factory_from
from core.llm import ModelFactory
from utils.run_logger import create_run_id, log_event
from workflow import create_workflow


def run_analysis(
    data_sources: list[str],
    analysis_focus: str | None = None,
    knowledge_sources: list[str] | None = None,
    run_id: str | None = None,
    model_factory: ModelFactory | None = None,
) -> dict[str, Any]:
    """执行一次完整分析。

    model_factory 用于注入模型（测试传假模型；生产不传，按 run_id 构造真实客户端）。
    注入的实例决定每次模型调用的追踪身份，因此必须在 invoke 之前就位。
    """
    if run_id is None:
        run_id = create_run_id()

    factory = factory_from(model_factory, run_id)

    log_event(
        run_id=run_id,
        node="workflow",
        event="run_started",
        status="started",
    )

    workflow = create_workflow()

    try:
        result = workflow.invoke(
            {
                "run_id": run_id,
                "model_factory": factory,
                "data_sources": data_sources,
                "analysis_focus": analysis_focus,
                "knowledge_sources": knowledge_sources or [],
                "retrieved_knowledge": [],
                "retrieved_knowledge_details": [],
                "execution_results": [],
                "failed_tasks": [],
                "visualization_results": [],
                "artifacts": [],
            },
            config={"recursion_limit": 100},
        )

    except Exception as error:
        log_event(
            run_id=run_id,
            node="workflow",
            event="run_failed",
            status="failed",
            details={
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        # 异常终止时结果字典不存在，这里显式标注，避免调用方把异常当成"没有结果"
        # 与"分析未产出"混为一谈。
        raise

    log_event(
        run_id=run_id,
        node="workflow",
        event="run_completed",
        status=result.get("analysis_status", "unknown"),
        details={
            "analysis_status": result.get("analysis_status"),
            "status_reason": result.get("status_reason"),
            "analysis_tasks": len(result.get("analysis_plan", [])),
            "failed_tasks": len(result.get("failed_tasks", [])),
            "visualizations": len(result.get("visualization_plan", [])),
            "artifacts": result.get("artifacts", []),
        },
    )

    return result
