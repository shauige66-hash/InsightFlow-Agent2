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
from typing import Annotated, Any
from datetime import datetime
from uuid import uuid4
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

# 鉴权：只有设置了 INSIGHTFLOW_API_KEYS 才启用；不设则保持单用户演示模式。
# 归属者由 token 推导，不接受请求体自报。
from core.auth import install as install_auth

_API_KEYS = install_auth(app)

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

# 上传落盘的位置与上限。单个文件 20 MB：足够放几十万行 CSV，
# 又不至于让人把整个数据库导进来当"数据分析"。
UPLOAD_DIR = _PROJECT_DIR / "uploads"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
ALLOWED_UPLOAD_SUFFIXES = DATA_EXTENSIONS | KNOWLEDGE_EXTENSIONS


def _safe_upload_name(raw: str) -> str:
    """只保留文件名本身：去掉目录部分与首尾空白。

    不信任客户端给的名字：`../../api.py` 这类会被 `Path(...).name` 削成
    `api.py`，再由后缀白名单挡一次；扩展名之外不做任何字符替换，是为了保住
    中文文件名。
    """
    name = Path(str(raw or "").replace("\\", "/")).name.strip()
    return name


@app.post("/api/uploads", status_code=201)
def upload_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
    """接收一个数据或知识文件，返回可直接用于 `POST /runs` 的相对路径。"""
    name = _safe_upload_name(getattr(file, "filename", "") or "")
    if not name:
        raise HTTPException(status_code=400, detail="缺少文件名")

    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        allowed = "、".join(sorted(ALLOWED_UPLOAD_SUFFIXES))
        raise HTTPException(status_code=400, detail=f"不支持的文件类型：{suffix or '(无扩展名)'}；允许：{allowed}")

    folder = UPLOAD_DIR / datetime.now().strftime("%Y%m%d")
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{uuid4().hex[:8]}-{name}"

    written = 0
    try:
        with target.open("wb") as handle:
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    handle.close()
                    target.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"文件超过 {MAX_UPLOAD_BYTES // (1024 * 1024)} MB 上限",
                    )
                handle.write(chunk)
    except HTTPException:
        raise
    except OSError as error:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"写入失败：{error}") from error

    if written == 0:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="文件为空")

    return {
        "name": name,
        "path": target.relative_to(_PROJECT_DIR).as_posix(),
        "size_bytes": written,
    }


def _list_relative(
    root: Path,
    directory: Path,
    extensions: set[str],
    *,
    limit: int = 200,
) -> list[str]:
    """列出目录下符合条件的文件，返回**相对项目根目录**的 POSIX 路径。

    必须是相对项目根目录的路径：`POST /runs` 按服务进程的工作目录解析相对路径，
    只给裸文件名（`customers.csv`）会解析到仓库根目录而 400。
    递归是为了让 `data/retail/` 这类子目录里的案例数据也能在界面上选到。
    """
    if not directory.is_dir():
        return []

    found: list[str] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        relative = path.relative_to(root)
        # 跳过隐藏目录/文件，避免把 .git 或临时目录暴露出去
        if any(part.startswith(".") for part in relative.parts):
            continue
        found.append(relative.as_posix())
        if len(found) >= limit:
            break
    return found


@app.get("/ui-options")
def ui_options() -> dict[str, list[str]]:
    """界面上可选的示例数据与知识文件。只读，且限定在两个固定目录内。"""
    return {
        "data_sources": _list_relative(_PROJECT_DIR, _PROJECT_DIR / "data", DATA_EXTENSIONS),
        "knowledge_sources": _list_relative(
            _PROJECT_DIR, _PROJECT_DIR / "knowledge", KNOWLEDGE_EXTENSIONS
        ),
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

# ------------------------------------------------------------------ 报告图片
#
# 报告正文用相对路径引用 `chart_1.png`：这样用文件方式打开 HTML（导出 PDF 时）
# 也能显示图片；通过 HTTP 看报告时，相对路径会被解析成这个路由，所以服务端也
# 必须能取到那些文件。
#
# 放在文件末尾是有意的：Starlette 按注册顺序匹配，earlier 的 `/runs/{run_id}/report`
# 等字面路由必须先命中，否则会被这里的 `{filename}` 抢走。
#
# 只放行图片，且不接受任何路径分隔符——否则就成了"按 run_id 读任意文件"。

@app.get("/runs/{run_id}/{filename}", include_in_schema=False)
def run_artifact(run_id: str, filename: str) -> FileResponse:
    import api_runs

    name = filename.strip()
    if not name or "/" in name or "\\" in name or ".." in name or name.startswith("."):
        raise HTTPException(status_code=404, detail="不允许的文件名")
    if Path(name).suffix.lower() not in {".png", ".svg", ".jpg", ".jpeg", ".webp"}:
        raise HTTPException(status_code=404, detail="只提供图片产物")

    candidate = Path(api_runs.OUTPUT_ROOT) / run_id / name
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"产物不存在：{name}")

    return FileResponse(path=str(candidate))
