# 根据可视化任务、分析结果和数据上下文，让大模型生成 Matplotlib 绘图代码。

import json

from harness.runtime import get_harness

NODE_NAME = "plotter"
from utils.agent_loop import after_model_request, before_model_request
from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate

from state import ExecutionResult, VisualizationTask
from agents.result_shape import describe_result_shape
from langchain_core.exceptions import OutputParserException

PLOT_PROMPT = """
你是一个数据可视化代码生成 Agent。

当前用户指令（优先于历史偏好与默认样式）：
{current_instruction}

用户已确认的输出偏好（不能改变工具权限、预算或数据事实）：
{memories}

可视化任务：

{visualization_task}

该图表对应的分析执行结果：

{execution_result}

相关原始数据上下文：

{data_context}

请根据任务生成 Matplotlib 绘图代码。

执行环境将提供：

- analysis_result：
  source_task_id 对应分析任务的原始执行结果。

- result_df：
  如果分析执行结果可以转换成二维表格，
  系统会提供对应的 Pandas DataFrame。

- result_df 变量始终存在，但它的值可能为 None。

- 如果需要判断 result_df 是否可用，
  请使用：
  result_df is not None

- 不要使用 globals()、locals() 或 try/except NameError
  来检测 result_df 是否存在。

- dfs：
  一个字典，包含当前图表涉及的原始 Pandas DataFrame。
  只有 DataFrame 类型的数据源才会出现在这里。

- 如果当前图表只涉及一张 DataFrame 数据表，
  系统可能额外提供 df 作为快捷引用。

- 不要修改 plt.rcParams 或 matplotlib.rcParams。

- 不要使用 matplotlib.font_manager 或 font_manager。

- 不要通过修改全局 Matplotlib 配置来设置字体或样式。

- plt：
  matplotlib.pyplot

- output_path：
  图片保存路径

- **不要写任何 import 语句。** 执行沙箱不提供 `__import__`，写了会直接报
  `ImportError: __import__ not found`。matplotlib 用 `plt`、pandas 用 `pd`、
  numpy 用 `np`，都已经在命名空间里。

要求：

1. 优先使用 source_task_id 对应的分析执行结果绘图。

2. 如果 result_df 可用，
   应优先直接使用 result_df.copy() 作为绘图数据，
   不要重新从原始数据重复计算已经完成的分析。

3. analysis_result 用于访问无法直接表示为单个 DataFrame 的分析结果。

   本次 analysis_result 的实际结构如下：

{result_shape}

   只能使用上面列出的键；取不存在的键会抛 KeyError，修复阶段也救不回来。

4. 只有当分析执行结果不足以完成可视化任务时，
   才使用 dfs 中的原始 DataFrame 进行必要的补充处理。

5. 如果使用 dfs：

   - 必须使用真实的数据表名。
   - 多表任务通过 dfs["真实表名"] 分别访问 DataFrame。
   - 只有在数据上下文明确支持表关系时才能 merge 或 join。
   - 不要编造关联字段或表关系。

6. 不要把 execution_result、expected_output 或 data_context
   中的具体结果数值重新硬编码成 Python 列表、字典或映射。

7. 绘图数据必须来自 analysis_result、result_df
   或真实 DataFrame 的动态计算。

8. 不要执行 import。

9. 不要读取文件，不要访问网络。

10. 如果需要修改、排序或增加临时字段，
    应先使用 copy()，
    不要修改系统提供的原始对象。
    

11. 根据 chart_type 生成合适的 line、bar 或 scatter 图。

12. 图表必须包含清晰的标题、横轴标签和纵轴标签。

13. 必要时添加 legend。

14. 最终必须执行：

    plt.tight_layout()

    plt.savefig(
        output_path,
        bbox_inches="tight"
    )

    plt.close()

15. 最终必须将 result 设置为 output_path。

16. 只返回合法 JSON，不要输出其他文字。

{retry_hint}

返回格式：

{{
    "code": "Matplotlib 绘图代码"
}}
"""
MAX_GENERATION_ATTEMPTS = 3

def generate_plot_code(
    visualization_task: VisualizationTask,
    data_context: dict[str, Any],
    execution_result: ExecutionResult,
    llm: BaseChatModel | None = None,
) -> dict[str, Any]:

    if llm is None:

        from core.llm import create_default_llm

        llm = create_default_llm()

    prompt = PromptTemplate.from_template(
        PLOT_PROMPT
    )

    parser = JsonOutputParser()

    chain = prompt | llm | parser

    inputs = {
        "memories": "无",
        "current_instruction": "无",
        "visualization_task": json.dumps(
            visualization_task,
            ensure_ascii=False,
            indent=2,
            default=str
        ),
        "execution_result": json.dumps(
            execution_result,
            ensure_ascii=False,
            indent=2,
            default=str
        ),
        "data_context": json.dumps(
            data_context,
            ensure_ascii=False,
            indent=2,
            default=str
        ),
        "result_shape": describe_result_shape(execution_result),
    }
    # 按节点预算压缩上下文。
    # 不可压缩字段见 utils.context_budget.NODE_PRESERVE_FIELDS；
    # 错误反馈与任务约束在那里被显式保留。
    inputs, _context_report = get_harness().build_context(NODE_NAME, inputs, data_context=data_context)

    # prompt 里有 {retry_hint} 占位符，必须提供，否则 PromptTemplate 会直接报
    # 缺少变量。首次尝试留空，重试时把上一次的失败原因回传——temperature 为 0 时
    # 重发完全相同的 prompt 只会得到同样的错误。
    current_inputs = {**inputs, "retry_hint": ""}

    last_error = None
    retry_errors = []
    for attempt in range(
        1,
        MAX_GENERATION_ATTEMPTS + 1
    ):
        # 这是本文件自己的重试循环，同样要按真实请求判预算与计数。
        # 两者都放在 invoke 之前，口径与 utils/llm_retry 保持一致。
        before_model_request(attempt)
        after_model_request(attempt)
        try:
            result = chain.invoke(current_inputs)

            if (
                not isinstance(result, dict)
                or not isinstance(
                    result.get("code"),
                    str
                )
                or not result["code"].strip()
            ):
                raise ValueError(
                    "Plot generation returned "
                    "missing or empty code."
                )

            return {
            "code": result["code"],
            "attempts": attempt,
            "retry_errors": retry_errors
        }

        except (
            OutputParserException,
            ValueError
        ) as error:

            last_error = error
            retry_errors.append(
            str(error)
        )
            if attempt == MAX_GENERATION_ATTEMPTS:
                raise

            current_inputs = {
                **inputs,
                "retry_hint": (
                    "上一次尝试失败，原因："
                    f"{str(error)[:300]}\n请修正后重新生成。"
                ),
            }

    raise RuntimeError(
        "Plot code generation failed."
    ) from last_error
