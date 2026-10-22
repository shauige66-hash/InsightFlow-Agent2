# 项目启动入口，负责通过命令行启动 InsightFlow 分析流程。

from services.analysis_service import run_analysis
from core.llm import build_model_factory
from utils.run_logger import create_run_id
from pathlib import Path


def main():
    run_id = create_run_id()
    root = Path(__file__).resolve().parent
    result = run_analysis(
        run_id=run_id,
        model_factory=build_model_factory(run_id),
        data_sources=[
            str(root / "data/customers.csv"),
            str(root / "data/orders.csv")
        ],
        analysis_focus=(
            "分析不同地区的高价值客户表现，"
            "并比较各地区高价值客户的销售贡献。"
        ),
        knowledge_sources=[
            str(root / "knowledge/business_rules.md")
        ]
    )

    print("\n=== Run ID ===")
    print(
        result.get("run_id")
    )

    print("\n=== Retrieved Knowledge ===")
    print(
        result.get(
            "retrieved_knowledge",
            []
        )
    )

    print("\n=== Retrieved Knowledge Details ===")
    print(
        result.get(
            "retrieved_knowledge_details",
            []
        )
    )

    print("\n=== Analysis Plan ===")
    print(
        result.get(
            "analysis_plan",
            []
        )
    )

    print("\n=== Execution Results ===")
    print(
        result.get(
            "execution_results",
            []
        )
    )

    print("\n=== Failed Tasks ===")
    print(
        result.get(
            "failed_tasks",
            []
        )
    )

    print("\n=== Visualization Results ===")
    print(
        result.get(
            "visualization_results",
            []
        )
    )

    print("\n=== Final Report ===")
    print(
        result.get("report_content")
    )

    print("\n=== Report Path ===")
    print(
        result.get("report_path")
    )

    print("\n=== Artifacts ===")
    print(
        result.get(
            "artifacts",
            []
        )
    )


if __name__ == "__main__":
    main()
