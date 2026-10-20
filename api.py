# 提供 InsightFlow 的 FastAPI 服务入口。

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile as FastAPIUploadFile,
)
from services.analysis_service import run_analysis
from core.llm import build_model_factory
from api_runs import register_run_routes
from api_conversations import register_conversation_routes
from pathlib import Path
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
import shutil
from utils.run_logger import create_run_id
from fastapi.responses import FileResponse
from pydantic import WithJsonSchema
from typing import Annotated
import pandas as pd
import numpy as np



UploadFile = Annotated[
    FastAPIUploadFile,
    WithJsonSchema(
        {
            "type": "string",
            "format": "binary"
        }
    )
]
UPLOAD_DIR = Path("uploads")
DATA_EXTENSIONS = {
    ".csv",
    ".xlsx",
    ".db",
    ".sqlite",
    ".sqlite3"
}

KNOWLEDGE_EXTENSIONS = {
    ".md",
    ".txt",
    ".pdf"
}
app = FastAPI(
    
    title="InsightFlow API",
    description=(
        "Agentic Data Intelligence Platform "
        "powered by LangGraph and LangChain."
    ),
    version="0.1.0"
)

# 持久化 Run 生命周期接口（创建 / 查询 / 恢复 / 审批）
register_run_routes(app)
register_conversation_routes(app)

# ---------------------------------------------------------------- 操作界面
#
# 项目本身是纯接口的：没有这层时，唯一"能用"的入口是 Swagger 表单。这里挂一个
# 零构建的单页（web/index.html）和它需要的两个只读端点，不改动任何既有接口。
#
# 只读是刻意的：页面能列目录、能看报告，但不能写文件、不能执行任意路径。

_WEB_DIR = Path(__file__).resolve().parent / "web"
_PROJECT_DIR = Path(__file__).resolve().parent


def _list_relative(directory: Path, extensions: set[str]) -> list[str]:
    """列出目录下（不递归）符合条件的相对路径，按名称排序。"""
    if not directory.is_dir():
        return []
    names = [
        item.name
        for item in sorted(directory.iterdir())
        if item.is_file() and item.suffix.lower() in extensions
    ]
    return names


@app.get("/ui-options")
def ui_options() -> dict[str, list[str]]:
    """界面上可选的示例数据与知识文件。只读，且限定在两个固定目录内。"""
    return {
        "data_sources": _list_relative(_PROJECT_DIR / "data", DATA_EXTENSIONS),
        "knowledge_sources": _list_relative(_PROJECT_DIR / "knowledge", KNOWLEDGE_EXTENSIONS),
    }


@app.get("/", include_in_schema=False)
def ui_root() -> RedirectResponse:
    return RedirectResponse(url="/ui/")


@app.get("/runs/{run_id}/report.html", include_in_schema=False)
def report_html(run_id: str) -> FileResponse:
    """同一份报告的 HTML 版本：浏览器里直接读，比下载 PDF 更适合演示。"""
    import api_runs

    report_path = Path(api_runs.OUTPUT_ROOT) / run_id / "analysis_report.html"
    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"该运行没有 HTML 报告：{run_id}",
        )
    return FileResponse(
        path=str(report_path),
        media_type="text/html; charset=utf-8",
    )


if _WEB_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=str(_WEB_DIR), html=True), name="ui")

# 把 Agent 执行结果转换为 FastAPI 可以返回的 JSON 数据。
def make_json_safe(
    value
):

    if isinstance(
        value,
        pd.DataFrame
    ):
        return value.to_dict(
            orient="records"
        )

    if isinstance(
        value,
        pd.Series
    ):
        return value.tolist()

    if isinstance(
        value,
        pd.Timestamp
    ):
        return value.isoformat()

    if isinstance(
        value,
        np.generic
    ):
        return value.item()

    if isinstance(
        value,
        dict
    ):
        return {
            key: make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        (list, tuple)
    ):
        return [
            make_json_safe(item)
            for item in value
        ]

    return value



@app.get("/health")
def health_check() -> dict[str, str]:

    return {
        "status": "ok",
        "service": "InsightFlow"
    }


@app.post("/analyze/upload")
def analyze_upload(
    files: list[UploadFile] = File(...),
    knowledge_files: list[UploadFile] | None = File(None),
    analysis_focus: str | None = Form(None)
) -> dict:

    run_id = create_run_id()

    run_upload_dir = (
        UPLOAD_DIR / run_id
    )

    data_dir = (
        run_upload_dir
        / "data"
    )

    knowledge_dir = (
        run_upload_dir
        / "knowledge"
    )

    data_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    knowledge_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    data_sources = []
    knowledge_sources = []

    allowed_data_suffixes = {
        ".csv",
        ".xlsx",
        ".db",
        ".sqlite",
        ".sqlite3",
    }

    allowed_knowledge_suffixes = {
        ".md",
        ".txt",
        ".pdf",
    }

    for uploaded_file in files:

        filename = Path(
            uploaded_file.filename or ""
        ).name

        suffix = Path(
            filename
        ).suffix.lower()

        if suffix not in allowed_data_suffixes:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported data file type: "
                    f"{filename}"
                )
            )

        file_path = (
            data_dir
            / filename
        )

        with file_path.open(
            "wb"
        ) as output_file:

            shutil.copyfileobj(
                uploaded_file.file,
                output_file
            )

        data_sources.append(
            str(file_path)
        )

    for uploaded_file in knowledge_files or []:

        filename = Path(
            uploaded_file.filename or ""
        ).name

        suffix = Path(
            filename
        ).suffix.lower()

        if suffix not in allowed_knowledge_suffixes:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported knowledge file type: "
                    f"{filename}"
                )
            )

        file_path = (
            knowledge_dir
            / filename
        )

        with file_path.open(
            "wb"
        ) as output_file:

            shutil.copyfileobj(
                uploaded_file.file,
                output_file
            )

        knowledge_sources.append(
            str(file_path)
        )

    try:
        result = run_analysis(
            data_sources=data_sources,
            analysis_focus=analysis_focus,
            knowledge_sources=knowledge_sources,
            run_id=run_id,
            # 兼容路径同样显式注入模型工厂：run_id 会作为追踪头传给 ModelGate
            model_factory=build_model_factory(run_id)
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        ) from error

    return {
        "run_id": result.get(
            "run_id"
        ),
        "retrieved_knowledge": result.get(
        "retrieved_knowledge",
            []
        ),
        "retrieved_knowledge_details": result.get(
        "retrieved_knowledge_details",
            []
        ),
        "analysis_plan": result.get(
            "analysis_plan",
            []
        ),
        "failed_tasks": result.get(
            "failed_tasks",
            []
        ),
        "visualization_plan": result.get(
            "visualization_plan",
            []
        ),
        "visualization_results": result.get(
            "visualization_results",
            []
        ),
        "report_content": result.get(
            "report_content"
        ),
        "report_path": result.get(
            "report_path"
        ),
        "artifacts": result.get(
            "artifacts",
            []
        ),
        "execution_results": make_json_safe(
        result.get(
        "execution_results",
        []
        )
     ),
    }


@app.get("/runs/{run_id}/report")
def download_report(
    run_id: str
):
    # 走 api_runs 的运行期 output_root，而不是硬编码的 `outputs`。
    # 硬编码会让任何把产物写到别处（测试、演示、容器）的部署拿不到报告，
    # 而 Run 本身明明是成功的——端到端测试暴露了这个断链。
    import api_runs

    report_path = Path(api_runs.OUTPUT_ROOT) / run_id / "analysis_report.pdf"

    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Report not found for run: "
                f"{run_id}"
            )
        )

    return FileResponse(
        path=str(report_path),
        media_type="application/pdf",
        filename=f"{run_id}_report.pdf"
    )