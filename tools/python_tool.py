# 把 DataFrame 数据分析能力包装成可供 AI 调用的 Python Tool。
#
# 执行不再发生在本进程。模型生成的代码被送到一次性 Docker 容器中运行
# （见 sandbox/executor.py），本模块只负责参数装配与结果归一化。
# 这里不存在同进程 exec 的回退路径：沙箱不可用时直接失败。

from typing import Any

from langchain_core.tools import tool

from core.runtime import RuntimeContextMissing, current_runtime
from ingestion.models import TableResource
from sandbox.executor import DockerSandboxExecutor, get_executor
from sandbox.protocol import SandboxJob, SandboxOutcome, classify_exception


def execute_in_sandbox(
    *,
    job: SandboxJob,
    tables: dict[str, TableResource],
    executor: DockerSandboxExecutor | None = None,
    run_id: str | None = None,
    run_dir: Any = None,
) -> SandboxOutcome:
    """在沙箱中执行一次任务。

    `run_dir` 缺省时从运行期上下文取本次 Run 的目录，因此调用方不必把路径
    一路传下来；显式传入主要用于测试隔离。
    """
    from pathlib import Path

    runner = executor or get_executor()

    if run_dir is None or run_id is None:
        try:
            runtime = current_runtime()
        except RuntimeContextMissing as error:
            raise RuntimeError(
                "sandbox execution requires a bound runtime context"
            ) from error
        run_id = run_id or runtime.run_id
        run_dir = run_dir or runtime.output_dir()

    # 孤儿容器只可能来自被强杀的上一次执行，这里顺手清一次；
    # 只清理本 Run 前缀的容器，不会触碰其他 Docker 服务。
    runner.cleanup_orphans(run_id)

    return runner.run(
        job,
        run_id=run_id,
        run_dir=Path(run_dir),
        tables=tables,
    )


def create_python_tool(
    tables: dict[str, TableResource]
):

    @tool
    def run_python_analysis(
        table_names: list[str],
        code: str
    ) -> dict[str, Any]:
        """对一个或多个 CSV / Excel 数据表执行 Pandas 分析代码。"""

        if not table_names:
            raise ValueError(
                "At least one table is required."
            )

        for table_name in table_names:

            if table_name not in tables:
                raise ValueError(
                    f"Table not found: {table_name}"
                )

            if tables[table_name].dataframe is None:
                raise ValueError(
                    f"Table is not a DataFrame resource: {table_name}"
                )

        job = SandboxJob(
            mode="python",
            table_names=list(table_names),
            code=code,
            op_label=f"python:{','.join(sorted(table_names))}",
        )

        try:
            outcome = execute_in_sandbox(job=job, tables=tables)
        except Exception as error:  # noqa: BLE001 - 统一转成结果字典
            return {
                "success": False,
                "result": None,
                "error": str(error),
                "error_kind": classify_exception(error),
            }

        return outcome.to_tool_result()

    return run_python_analysis
