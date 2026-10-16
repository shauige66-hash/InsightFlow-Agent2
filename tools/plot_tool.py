# 执行 Matplotlib 绘图代码，并将生成的图表保存到本次 Run 的目录。
#
# P2-A：绘图代码同样在一次性 Docker 容器中执行，本进程不再 exec 任何模型代码。
# 容器内已安装 fonts-noto-cjk，并把 Noto Sans CJK SC 置于字体首选，
# 因此中文标题与图例能正常渲染（宿主 Windows 字体在容器内并不存在）。

import re
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from core.runtime import RuntimeContextMissing, current_runtime
from ingestion.models import TableResource
from sandbox.protocol import SandboxErrorKind, SandboxJob, classify_exception
from tools.python_tool import execute_in_sandbox
from utils.result_artifacts import to_jsonable

CHART_ID_PATTERN = re.compile(r"[A-Za-z0-9\_-]+")


def create_plot_tool(
    tables: dict[str, TableResource],
):
    @tool
    def run_plot(
        table_names: list[str],
        chart_id: str,
        code: str,
        analysis_result: Any
    ) -> dict[str, Any]:
        """根据分析结果和可用 DataFrame 执行 Matplotlib 绘图代码并保存图表。"""

        if not table_names:
            raise ValueError(
                "At least one table is required."
            )

        for table_name in table_names:

            if table_name not in tables:
                raise ValueError(
                    f"Table not found: {table_name}"
                )

            resource = tables[table_name]

            if resource.dataframe is None and resource.source_type != "sqlite":
                raise ValueError(
                    f"Unsupported table resource: {table_name}"
                )

        # chart_id 会参与容器内的输出文件名，必须严格限制字符集；
        # 宿主与容器两侧都校验，避免任一侧被绕过。
        if not CHART_ID_PATTERN.fullmatch(chart_id):
            raise ValueError(
                f"Invalid chart id: {chart_id}"
            )

        output_filename = f"{chart_id}.png"

        try:
            runtime = current_runtime()
        except RuntimeContextMissing:
            # 没有运行期上下文就无法确定本次 Run 的输出目录。
            # 归类为 unavailable：这是接线问题，重试模型代码没有意义。
            return {
                "success": False,
                "result": None,
                "error": "plot execution requires a bound runtime context",
                "error_kind": SandboxErrorKind.UNAVAILABLE,
            }

        # 绘图只需要 DataFrame 形态的表；sqlite 表由 SQL 工具负责取数
        sandbox_tables = {
            name: tables[name]
            for name in table_names
            if tables[name].dataframe is not None
        }

        # 先转成 JSON 可表达的信封再交给沙箱。
        #
        # 不能把 DataFrame 直接塞进 job.json：写 JSON 时的 `default=str`
        # 会把它**静默**变成一个字符串，容器里 `result_df` 于是成了 None，
        # 绘图代码一取列就报 `TypeError: 'NoneType' object is not subscriptable`。
        # 错误离现场很远，且图只是"没画出来"，很容易被当成绘图代码写错。
        analysis_result = to_jsonable(analysis_result)

        job = SandboxJob(
            mode="plot",
            table_names=list(sandbox_tables),
            code=code,
            analysis_result=analysis_result,
            output_filename=output_filename,
            op_label=f"plot:{chart_id}",
        )

        try:
            outcome = execute_in_sandbox(
                job=job,
                tables=sandbox_tables,
                run_id=runtime.run_id,
                run_dir=runtime.output_dir(),
            )
        except Exception as error:  # noqa: BLE001 - 统一转成结果字典
            return {
                "success": False,
                "result": None,
                "error": str(error),
                "error_kind": classify_exception(error),
            }

        if not outcome.success:
            return outcome.to_tool_result()

        # 返回宿主路径：下游的 artifact / report / 幂等登记都用宿主路径，
        # 容器内的 /out/... 不能泄漏到 state。
        host_path = Path(runtime.output_dir()) / output_filename

        if not host_path.exists():
            return {
                "success": False,
                "result": None,
                "error": "Plot code did not create the expected image file.",
            }

        return {
            "success": True,
            "result": str(host_path),
            "error": None,
        }

    return run_plot
