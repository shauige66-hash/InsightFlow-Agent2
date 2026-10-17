# 定义各个 Agent 在工作流中共享的数据状态结构。
#
# 这里是 Checkpoint 的输入契约：所有值都必须可 JSON 编码。
# 表数据与模型客户端都不在 state 中——前者按 `table_snapshot` 读回，
# 后者由运行期注入（见 core/runtime.py）。
from typing import Any, Literal, TypedDict


class AnalysisTask(TypedDict):
    task_id: str
    description: str
    target_tables: list[str]
    expected_output: str


class VisualizationTask(TypedDict):
    chart_id: str
    description: str
    source_task_id: str
    target_tables: list[str]
    chart_type: Literal[
        "line",
        "bar",
        "scatter"
    ]
    expected_output: str

class VisualizationResult(TypedDict):
    chart_id: str
    code: str
    success: bool
    path: str | None
    error: str | None

class ExecutionResult(TypedDict):
    task_id: str
    tool_used: Literal["python", "sql"]
    code: str
    success: bool
    # 小结果内联；大结果只留摘要，完整内容在 `result_artifact` 指向的文件里
    result: Any
    result_artifact: dict[str, Any] | None
    error: str | None


class TaskFailure(TypedDict):
    task_id: str
    stage: Literal["execution", "validation"]
    reason: str
    error_kind: str | None  # execution 阶段的错误分类，便于事后归因
    code: str | None
    retry_count: int
    refine_count: int




# 工作流传递信息
class InsightFlowState(TypedDict, total=False):
    run_id: str  # 每次workflow唯一编号

    analysis_focus: str | None  # 提供分析重点（可选）

    data_sources: list[str]  # 文件数据

    # 表资源的可序列化身份（来源路径、内容指纹、sheet/表名）。
    # 真实 DataFrame 由 ResourceResolver 按快照按需读回，不进 state：
    # 它无法 JSON 编码，也不该跨进程恢复。
    table_snapshot: dict[str, dict[str, Any]]

    knowledge_sources: list[str]    #用户提供业务知识文件

    retrieved_knowledge: list[str]    # 针对当前分析问题实际检索出来的相关片段

    retrieved_knowledge_details: list[ dict[str, Any]]  # RAG 检索结果及其来源信息

    data_context: dict[str, dict[str, Any]]  # Adaptive Context 生成的数据上下文

    analysis_plan: list[AnalysisTask]  # Planner Agent 生成的分析计划

    # 计划校验与有界重新规划。`plan_attempts` 进 Checkpoint，因此恢复后
    # 不会把次数限制重置；`plan_validation_errors` 是下一次重新规划的依据。
    plan_attempts: int
    plan_validation_errors: list[str]

    # 计划获批时记下的内容指纹。执行前比对，避免审批后计划被换掉。
    approved_plan_fingerprint: str | None

    current_task_index: int  # 当前正在执行的分析任务位置

    current_tool: Literal["python", "sql"] | None  # 选择的工具

    current_code: str | None  # 当前生成并准备执行的 Python 或 SQL 代码

    execution_results: list[ExecutionResult]  # 执行分析结果

    execution_error: str | None  # 执行错误

    # 执行错误的分类（来自沙箱契约）。路由据此区分"模型代码写错了"（可进入
    # 有界修复循环）与"沙箱环境不可用"（重试无意义）。
    # 取值见 sandbox/protocol.py 的 SandboxErrorKind。
    execution_error_kind: str | None

    retry_count: int  # 重试次数

    refine_count: int  # Critic 不通过后的代码改进次数

    critic_feedback: str | None  # Critic Agent 反馈

    failed_tasks: list[TaskFailure]  # 最终失败的分析任务记录

    validation_passed: bool  # 验证是否通过

    artifacts: list[str]  # 生成的文件

    # 本次运行的结果语义。failed_tasks 为空不等于分析成功：
    # Planner 返回空计划时工作流会正常走完，但没有任何分析产出，
    # 必须由这个字段区分，避免被下游当成成功。
    # 取值：completed（有计划且全部成功）/ partial（有计划但有失败）/
    #       no_tasks（计划为空，未进行分析）/ failed（工作流异常终止）
    analysis_status: Literal["completed", "partial", "no_tasks", "failed"]

    status_reason: str | None  # analysis_status 不是 completed 时的原因说明

    # 运行生命周期。与 analysis_status 是两件事：前者描述"这次执行走到哪一步"，
    # 后者描述"分析本身产出了什么"。例如等待审批时 run_status=waiting_approval
    # 而 analysis_status 尚未产生。
    run_status: Literal[
        "pending",
        "running",
        "waiting_approval",
        "completed",
        "failed",
        "interrupted",
    ]

    # 待审批请求（由 interrupt 产生）。恢复后据此重建审批界面，
    # 因此必须是可序列化的普通结构。
    pending_approval: dict[str, Any] | None

    # 是否要求执行前人工审批。显式开关，未开启时保持全自动路径。
    require_plan_approval: bool
    # 最近一次执行前审批的归档结果（路由依据）
    plan_approval: dict[str, Any] | None

    # 是否要求报告发布前人工确认。同样是显式开关：
    # 未开启时报告直接落盘，保持既有全自动行为。
    require_report_approval: bool
    # 最近一次报告发布审批的归档结果（路由依据）
    report_approval: dict[str, Any] | None

    # 已处理的审批决定，按审批 ID 归档，用于重复审批保护与审计
    approvals: list[dict[str, Any]]

    report_path: str | None  # 报告路径

    report_content: str | None  # 最终生成的 Markdown 分析报告

    visualization_plan: list[VisualizationTask]  # Visualization Agent 生成的图表计划
    current_chart_index: int  # 当前正在处理的图表位置
    current_plot_code: str | None  # 当前生成并准备执行的绘图代码
    visualization_results: list[VisualizationResult]  # 图表生成结果
    plot_error: str | None  # 当前图表执行错误
    plot_retry_count: int  # 绘图失败后的修复次数

    # 副作用幂等登记：操作 ID -> 完成记录。恢复时据此判断某个副作用
    # 是否已经执行过，避免重复写文件或重复调用工具。
    completed_operations: dict[str, dict[str, Any]]

    # P3-A：有界执行的预算快照（上限 + 已用量）。
    # 放进 state 是**必须的**：预算要跨进程恢复后不重置，
    # 放在进程内存里的话崩溃重启就归零，"有界"名不副实。
    execution_budget: dict[str, Any]

    # 可能已发出但未确认结果的模型调用（由账本在恢复时比对得出）。
    # 必须在 state 中声明：未声明的键会被 LangGraph 直接丢弃，
    # 那样恢复时就发现不了"请求已发出、节点未提交"的重复风险。
    pending_llm_calls: list[str]

    # 人工对未确认调用的决定（"retry" 表示获准重发）
    llm_retry_granted: str | None