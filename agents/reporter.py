# 根据所有分析任务的执行结果生成最终数据分析报告。

import json

from harness.runtime import get_harness
from utils.agent_loop import after_model_request, before_model_request

NODE_NAME = "reporter"

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from state import (
    AnalysisTask,
    ExecutionResult,
    TaskFailure,
    VisualizationTask,
    VisualizationResult,
)
from utils.llm_retry import invoke_report_with_retry


REPORT_PROMPT = """
你是一个数据分析报告 Agent。

原始分析计划：

{analysis_plan}

成功执行和历史执行结果：

{execution_results}

最终失败的任务：

{failed_tasks}

可视化计划：

{visualization_plan}

成功生成的图表：

{visualization_results}

请根据实际执行结果生成最终数据分析报告。

要求：

1. 只能根据提供的执行结果得出结论。
2. 不要编造不存在的数据或分析结果。
3. 重点总结真正有价值的数据发现。
4. 如果某个任务最终失败，应明确说明该部分没有成功完成。
5. 不要把代码执行过程写成报告主体。
6. 使用清晰、简洁的数据分析语言。
7. 使用 Markdown 格式。
8. 如果存在成功生成的图表，请增加一个 ## Visualizations 部分。
9. 只引用 success=true 的图表。
10. 使用提供的 markdown_path 插入 Markdown 图片。
11. 不要编造不存在的图片。

报告结构：

# Analysis Report

## Overview

## Key Findings

## Analysis Details

## Limitations
"""


def generate_analysis_report(
    analysis_plan: list[AnalysisTask],
    execution_results: list[ExecutionResult],
    failed_tasks: list[TaskFailure],
    visualization_plan: list[VisualizationTask],
    visualization_results: list[VisualizationResult],
    llm: BaseChatModel | None = None,
) -> str:
    """生成最终报告。

    报告是唯一的交付物，因此这里也做一次重试：空输出对下游没有意义，
    重试成本远低于交付一份空报告。
    """
    if llm is None:
        from core.llm import create_default_llm

        llm = create_default_llm()

    prompt = PromptTemplate.from_template(
        REPORT_PROMPT
    )

    parser = StrOutputParser()

    chain = prompt | llm | parser

    inputs = {
        "analysis_plan": json.dumps(
            analysis_plan,
            ensure_ascii=False,
            indent=2,
            default=str
        ),
        "execution_results": json.dumps(
            execution_results,
            ensure_ascii=False,
            indent=2,
            default=str
        ),
        "failed_tasks": json.dumps(
            failed_tasks,
            ensure_ascii=False,
            indent=2,
            default=str
        ),
        "visualization_plan": json.dumps(
            visualization_plan,
            ensure_ascii=False,
            indent=2,
            default=str
        ),
        "visualization_results": json.dumps(
            visualization_results,
            ensure_ascii=False,
            indent=2,
            default=str
        )
    }

    # 按节点预算压缩上下文。
    #
    # 这里的字段已经是**有界预览**（`build_report_execution_previews` 走
    # 的 `build_preview`：列名 + 行数 + 少量样本），因此正常路径下
    # 不会触发压缩；预算是防止上游给出别的东西时这个节点的 Prompt 失控。
    #
    # 与其余 Agent 用同一个 `build_context`：口径一致才谈得上"逐节点预算"。
    inputs, context_report = get_harness().build_context(NODE_NAME, inputs)

    def _require_text(text: object) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("报告内容为空")
        return text

    result = invoke_report_with_retry(
        chain,
        inputs,
        validate=_require_text,
        before_request=before_model_request,
        on_request=after_model_request,
    )


    return result
