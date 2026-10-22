"""Run 生命周期 API。

分层：
- 原有 `/analyze/upload` 保持不变（同步返回完整结果），是 V1 的兼容路径；
- `/runs*` 是持久化路径：立即返回 run_id，执行在后台进行，
  通过状态查询与审批接口驱动。

服务只监听本机：P2 的安全加固（鉴权、限流、沙箱）尚未实施，
不应在此之前对公网开放。

对 run_id 做了格式与归属校验：它既用于拼接目录，也用于 Checkpoint 线程，
必须防止路径穿越与跨 Run 读取。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from core.checkpoints import open_checkpointer
from services import storage_factory as _storage_factory
from core.llm import build_model_factory
from core.runtime import SqlBackend
from services.analysis_service import read_run_state
from services.run_coordinator import RunCoordinator
from services.run_store import RunStore

# run_id 会参与文件路径与线程名，只允许字母数字与下划线/短横线
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{4,64}$")

# 默认产物根目录：锚定在**项目根**，而不是进程当前目录。
#
# 用相对路径 `Path("outputs")` 时，这个模块一旦被导入就会按当时的 cwd
# 解析并创建 `outputs/runs/`——任何以别的 cwd 导入 `api_runs` 的进程都会
# 在自己的目录里凭空建出目录树（演示脚本、测试子进程都会命中）。
_PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = _PROJECT_ROOT / "outputs"

OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT

# Checkpoint 位置：与 coordinator 保持一致，测试可覆盖。
# 读取状态必须走同一个文件，否则会读到另一个（默认）位置的 Checkpoint。
CHECKPOINT_PATH: Path | None = None

# 单例在这里**延迟构造**。
#
# 早期写法是模块顶层直接 `run_store = RunStore()`：那会在导入期就建出
# `outputs/runs/runs.sqlite`。即使 `serve_api` 随后把 store 换成隔离目录，
# 仓库根也已经多了一份无用的库。导入模块不该有这种副作用。
run_store: RunStore | None = None
coordinator: RunCoordinator | None = None


def _ensure_singletons() -> tuple[RunStore, RunCoordinator]:
    """按需构造默认的 Store 与 Coordinator。

    只在没有被替换过时构造：测试与 `scripts/serve_api.py` 都会在启动时
    直接赋值 `api_runs.run_store` / `api_runs.coordinator`，那种情况下
    这里不再重复建库。
    """
    global run_store, coordinator

    if run_store is None:
        # 由存储模式决定：local 是 SQLite，postgres 是 PostgreSQL
        run_store = _storage_factory.run_store(OUTPUT_ROOT)
    if coordinator is None:
        coordinator = RunCoordinator(
            run_store,
            output_root=OUTPUT_ROOT,
            checkpointer_path=None,
            model_factory_builder=build_model_factory,
        )
    return run_store, coordinator


def get_run_store() -> RunStore:
    return _ensure_singletons()[0]


def get_coordinator() -> RunCoordinator:
    return _ensure_singletons()[1]


def _open_checkpointer():
    return open_checkpointer(CHECKPOINT_PATH)

router = APIRouter(prefix="/runs", tags=["runs"])


class CreateRunRequest(BaseModel):
    data_sources: list[str] = Field(
        ..., description="本机数据源路径（CSV / Excel / SQLite）"
    )
    analysis_focus: str | None = None
    knowledge_sources: list[str] = Field(default_factory=list)
    require_plan_approval: bool = False
    # 报告发布前的人工确认。与计划审批一样是显式开关，
    # 未开启时报告直接落盘，保持既有全自动行为。
    require_report_approval: bool = False

    # 运行期动态决策闭环。默认关闭：原有固定编排保持不变，
    # 打开后任务执行完会先观察再决定下一步。
    enable_dynamic_loop: bool = False
    dynamic_limits: dict[str, Any] | None = None

    # SQL 后端模式。默认 direct，与  之前的既有客户端行为一致。
    #
    # 只允许这两个**模式名**：Server 启动命令、环境变量、私有配置目录与宿主
    # 路径全部由宿主固定，请求里根本没有对应的字段可以填。
    sql_backend: str = Field(
        default=SqlBackend.DIRECT.value,
        description="SQL 后端：direct（宿主内执行）或 mcp（经 stdio MCP Server）",
    )

    # 允许显式指定 run_id，便于调用方在重试时复用同一个线程；
    # 不传则由服务生成。
    run_id: str | None = None


class ApprovalRequest(BaseModel):
    decision: str = Field(..., description="approve 或 reject")
    comment: str | None = None


class ResumeRequest(BaseModel):
    resume_value: dict[str, Any] | None = None


def _validate_sql_backend(value: str) -> str:
    """只接受已知的后端模式名。

    显式校验而不是依赖枚举转换抛错：请求是面向用户的输入，
    需要 400 而不是 500。同时这也保证请求无法借"模式"字段夹带
    任何 Server 启动参数——非枚举值一律拒绝。
    """
    try:
        return SqlBackend(value).value
    except ValueError:
        allowed = ", ".join(item.value for item in SqlBackend)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"sql_backend 必须是以下之一：{allowed}",
        ) from None


def _execution_mode() -> str:
    """inline（默认，进程内线程池）或 queue（PostgreSQL 队列 + 独立 worker）。

    队列需要 PostgreSQL：`SKIP LOCKED` 是抢占的唯一可靠手段，SQLite 没有。
    配错了在请求时明确报 400/500，而不是悄悄退回 inline——否则部署方会以为
    任务进了队列，实际全压在一个进程里。
    """
    mode = (os.getenv("INSIGHTFLOW_EXECUTION") or "inline").strip().lower()
    if mode not in ("inline", "queue"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"INSIGHTFLOW_EXECUTION 只能是 inline 或 queue，收到 {mode!r}",
        )
    if mode == "queue" and not _storage_factory.storage.current_config().is_postgres:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="queue 执行模式需要 INSIGHTFLOW_STORAGE=postgres",
        )
    return mode


def _validate_run_id(run_id: str) -> str:
    if not RUN_ID_PATTERN.match(run_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="run_id 只允许字母、数字、下划线与短横线，长度 4-64",
        )
    return run_id


def _require_run(run_id: str):
    _validate_run_id(run_id)
    record = get_run_store().get(run_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"未找到 run：{run_id}",
        )
    return record


def _validate_data_sources(paths: list[str]) -> list[str]:
    """数据源必须是已存在的本机文件。

    不做"路径在某个根目录下"的限制：P1 的数据源本来就可以是任意本机路径，
    真正的边界在 P2 的沙箱。这里只确保文件存在且是普通文件，避免把目录
    或不存在的位置传进工作流后以晦涩的读取错误失败。
    """
    from core.resource_access import validate_sources
    return validate_sources(paths,project_root=_PROJECT_ROOT)


def _validate_knowledge_sources(paths: list[str]) -> list[str]:
    from core.resource_access import validate_sources
    return validate_sources(paths,project_root=_PROJECT_ROOT,knowledge=True)


@router.post("", status_code=status.HTTP_202_ACCEPTED)
def create_run(payload: CreateRunRequest) -> dict[str, Any]:
    """创建并启动一次分析，立即返回 run_id。"""
    from utils.run_logger import create_run_id

    run_id = _validate_run_id(payload.run_id) if payload.run_id else create_run_id()
    data_sources = _validate_data_sources(payload.data_sources)
    sql_backend = _validate_sql_backend(payload.sql_backend)

    if not data_sources:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="至少需要一个数据源",
        )

    request_payload = {
        "data_sources": data_sources,
        "analysis_focus": payload.analysis_focus,
        "knowledge_sources": _validate_knowledge_sources(payload.knowledge_sources),
        "require_plan_approval": payload.require_plan_approval,
        "require_report_approval": payload.require_report_approval,
        "sql_backend": sql_backend,
        "enable_dynamic_loop": payload.enable_dynamic_loop,
        "dynamic_limits": payload.dynamic_limits,
    }

    _execution_mode()
    try:
        record = get_coordinator().dispatch(run_id,**request_payload)
    except ValueError as error:
        raise HTTPException(status_code=409,detail=str(error)) from error
    return {"run_id":run_id,"status":record.status,
        "require_plan_approval":record.require_plan_approval,"require_report_approval":record.require_report_approval,
        "sql_backend":record.sql_backend,"enable_dynamic_loop":payload.enable_dynamic_loop}


@router.get("")
def list_runs(limit: int = 50) -> dict[str, Any]:
    if limit < 1 or limit > 500:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="limit 需在 1-500 之间"
        )

    records = []
    store = get_run_store()
    for record in store.list(limit=limit):
        payload = record.to_dict()
        payload["stale"] = store.is_stale_running(record.run_id)
        records.append(payload)

    return {"runs": records}


@router.get("/{run_id}/artifact-link")
def artifact_link(run_id: str, kind: str = "html") -> dict[str, str]:
    import storage
    from core.auth import artifact_access_token
    _require_run(run_id)
    if kind not in ("html","pdf"):
        raise HTTPException(status_code=400,detail="kind 只能是 html 或 pdf")
    suffix="report.html" if kind=="html" else "report"
    token=artifact_access_token(run_id,storage.current_owner())
    return {"url":f"/runs/{run_id}/{suffix}?access={token}"}


@router.get("/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    """运行状况：生命周期状态 + 待审批 + 恢复位置。

    `stale` 表示登记为 running 但执行者已消失（进程被杀）。
    仅凭状态字段无法区分这两种情况，因此单独给出。
    """
    record = _require_run(run_id)

    with _open_checkpointer() as saver:
        state = read_run_state(run_id, checkpointer=saver)

    payload = record.to_dict()
    payload["stale"] = get_run_store().is_stale_running(run_id)
    payload["resumable"] = (
        record.status in ("waiting_approval", "interrupted", "failed")
        or payload["stale"]
    )

    return {
        "run": payload,
        "state": state,
    }


@router.get("/{run_id}/state")
def get_run_state(run_id: str) -> dict[str, Any]:
    """可公开的轻量状态，不含完整数据上下文。"""
    _require_run(run_id)

    with _open_checkpointer() as saver:
        state = read_run_state(run_id, checkpointer=saver)

    if state is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"run {run_id} 没有可读取的状态",
        )

    return state


@router.post("/{run_id}/resume", status_code=status.HTTP_202_ACCEPTED)
def resume_run(run_id: str, payload: ResumeRequest | None = None) -> dict[str, Any]:
    """继续一个合法的待恢复任务。"""
    record = _require_run(run_id)

    # `running` 有两种含义：正在执行，或执行者已经消失（进程被杀）。
    # 查询接口早就用 `stale` 区分了两者，恢复接口却只看状态字面量，
    # 于是"崩溃后恢复"这条最重要的路径被自己的守卫挡住了。
    # 这里与查询接口对齐：stale 的 running 允许恢复。
    stale = get_run_store().is_stale_running(run_id)

    if record.status not in ("waiting_approval", "interrupted", "failed") and not stale:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"run 当前状态为 {record.status}，不可恢复",
        )

    try:
        resumed = get_coordinator().resume(
            run_id, resume_value=(payload.resume_value if payload else None)
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(error)
        ) from error

    # 恢复沿用该 Run 最初保存的后端，响应里回显出来便于调用方确认
    return {
        "run_id": run_id,
        "status": "queued" if _execution_mode()=="queue" else "running",
        "sql_backend": record.sql_backend,
        "resumed_from_stale": stale,
    }


@router.post("/{run_id}/approval", status_code=status.HTTP_202_ACCEPTED)
def submit_approval(run_id: str, payload: ApprovalRequest) -> dict[str, Any]:
    """提交人工审批决定。"""
    record = _require_run(run_id)

    if record.status != "waiting_approval":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"run 当前状态为 {record.status}，没有待审批事项",
        )

    decision = payload.decision.lower()
    if decision not in ("approve", "reject"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="decision 必须是 approve 或 reject",
        )

    pending = record.pending_approval or {}
    approval_id = pending.get("approval_id")

    duplicate=any(approval_id and item.get("approval_id")==approval_id for item in record.approvals)
    try:
        if approval_id:
            get_run_store().record_approval(run_id,{
                "approval_id":approval_id,"run_id":run_id,"node":pending.get("node"),
                "kind":pending.get("kind"),"decision":decision,"comment":payload.comment})
        get_coordinator().resume(run_id,resume_value={"decision":decision,"comment":payload.comment})
    except ValueError as error:
        latest=get_run_store().get(run_id)
        if not duplicate or latest is None or latest.status not in ("running","queued"):
            raise HTTPException(status_code=409,detail=str(error)) from error

    return {
        "run_id": run_id,
        "status": "queued" if _execution_mode()=="queue" else "running",
        "approval_id": approval_id,
        "decision": decision,
        "duplicate": duplicate,
    }


def register_run_routes(app: FastAPI) -> None:
    app.include_router(router)
