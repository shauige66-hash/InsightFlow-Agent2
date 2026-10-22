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
import storage
from core.resource_access import owner_upload_dir
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


MAX_UPLOAD_FILES = 20
MAX_UPLOAD_TOTAL_BYTES = 40 * 1024 * 1024
from core.upload_limit import UploadBodyLimitMiddleware
app.add_middleware(UploadBodyLimitMiddleware,limit_getter=lambda: MAX_UPLOAD_TOTAL_BYTES)


def _write_upload(file, target: Path, *, remaining: int) -> int:
    written=0
    try:
        with target.open("xb") as handle:
            while True:
                chunk=file.file.read(1024*1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES or written > remaining:
                    raise HTTPException(status_code=413,detail="上传超过文件或请求容量上限")
                handle.write(chunk)
        if written==0:
            raise HTTPException(status_code=400,detail="文件为空")
    except Exception:
        target.unlink(missing_ok=True)
        raise
    return written


@app.post("/api/uploads", status_code=201)
def upload_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
    name=_safe_upload_name(getattr(file,"filename","") or "")
    if not name or Path(name).suffix.lower() not in ALLOWED_UPLOAD_SUFFIXES:
        raise HTTPException(status_code=400,detail="不支持的文件类型或缺少文件名")
    folder=owner_upload_dir(UPLOAD_DIR)/datetime.now().strftime("%Y%m%d")
    folder.mkdir(parents=True,exist_ok=True)
    target=folder/f"{uuid4().hex}-{name}"
    size=_write_upload(file,target,remaining=MAX_UPLOAD_TOTAL_BYTES)
    return {"name":name,"path":target.relative_to(_PROJECT_DIR).as_posix(),"size_bytes":size}


def _authorized_artifact_path(run_id: str, filename: str) -> Path:
    import api_runs
    try:
        api_runs._validate_run_id(run_id)
    except HTTPException:
        raise HTTPException(status_code=404,detail="产物不存在") from None
    # In authenticated deployments every artifact must belong to the requesting owner.
    if _API_KEYS or __import__("core.auth",fromlist=["load_api_keys"]).load_api_keys():
        api_runs._require_run(run_id)
    root=Path(api_runs.OUTPUT_ROOT).resolve()
    run_root=(root/run_id).resolve()
    candidate=(run_root/filename).resolve()
    if not run_root.is_relative_to(root) or not candidate.is_relative_to(run_root) or not candidate.is_file():
        raise HTTPException(status_code=404,detail="产物不存在")
    return candidate


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

    report_path = _authorized_artifact_path(run_id,"analysis_report.html")
    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"该运行没有 HTML 报告：{run_id}",
        )
    return FileResponse(
        path=str(report_path),
        media_type="text/html; charset=utf-8",
        headers={"Content-Security-Policy":"default-src 'none'; img-src 'self'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
                 "X-Content-Type-Options":"nosniff", "Referrer-Policy":"no-referrer"},
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

    run_id=create_run_id()
    all_files=list(files)+list(knowledge_files or [])
    if not files or len(all_files)>MAX_UPLOAD_FILES:
        raise HTTPException(status_code=413 if files else 400,detail="文件数超限或缺少数据源")
    run_upload_dir=owner_upload_dir(UPLOAD_DIR)/run_id
    data_sources=[]
    knowledge_sources=[]
    total=0
    written_paths=[]
    try:
        for uploaded_file,is_knowledge in [(f,False) for f in files]+[(f,True) for f in (knowledge_files or [])]:
            name=_safe_upload_name(uploaded_file.filename or "")
            suffixes=KNOWLEDGE_EXTENSIONS if is_knowledge else DATA_EXTENSIONS
            if not name or Path(name).suffix.lower() not in suffixes:
                raise HTTPException(status_code=400,detail="不支持的文件类型")
            folder=run_upload_dir/("knowledge" if is_knowledge else "data")
            folder.mkdir(parents=True,exist_ok=True)
            target=folder/f"{uuid4().hex}-{name}"
            total += _write_upload(uploaded_file,target,remaining=MAX_UPLOAD_TOTAL_BYTES-total)
            written_paths.append(target)
            (knowledge_sources if is_knowledge else data_sources).append(str(target))
    except Exception:
        for target in written_paths:
            target.unlink(missing_ok=True)
        raise

    try:
        from api_runs import get_coordinator, _execution_mode
        if _execution_mode()=="queue":
            record=get_coordinator().dispatch(run_id,data_sources=data_sources,analysis_focus=analysis_focus,knowledge_sources=knowledge_sources)
            return {"run_id":run_id,"status":record.status}
        result = get_coordinator().start(
            run_id, data_sources=data_sources,
            analysis_focus=analysis_focus,
            knowledge_sources=knowledge_sources,
            wait=True, owner=storage.current_owner()
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

    report_path = _authorized_artifact_path(run_id,"analysis_report.pdf")

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

    # run_id 也必须校验：只校验文件名是不够的。Windows 上把 `C:\...` 或
    # `..\..\..` 编码进 run_id，`Path(output_root) / run_id / name` 会直接指向
    # 输出根之外（绝对路径还会替换整个左侧），等于"按 run_id 读任意图片"。
    # 这个漏洞在实现产物路由时真实存在过，由此固定为：形态校验 + 根目录校验。
    # 对外统一回 404，避免把"这个 run_id 形态非法"与"文件不存在"区分成探测信号。
    try:
        api_runs._validate_run_id(run_id)
    except HTTPException:
        raise HTTPException(status_code=404, detail="产物不存在") from None

    name = filename.strip()
    if not name or "/" in name or "\\" in name or ".." in name or name.startswith("."):
        raise HTTPException(status_code=404, detail="不允许的文件名")
    if Path(name).suffix.lower() not in {".png", ".svg", ".jpg", ".jpeg", ".webp"}:
        raise HTTPException(status_code=404, detail="只提供图片产物")

    candidate = _authorized_artifact_path(run_id,name)
    root = Path(api_runs.OUTPUT_ROOT).resolve()

    # 形态校验之后再加一层：解析后的路径必须仍在输出根之内。
    if root not in candidate.parents or not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"产物不存在：{name}")

    return FileResponse(path=str(candidate))
