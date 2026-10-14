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

import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from core.checkpoints import open_checkpointer
from core.llm import build_model_factory
from services.analysis_service import read_run_state
from services.run_coordinator import RunCoordinator
from services.run_store import RunStore

# run_id 会参与文件路径与线程名，只允许字母数字与下划线/短横线
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{4,64}$")

OUTPUT_ROOT = Path("outputs")

# Checkpoint 位置：与 coordinator 保持一致，测试可覆盖。
# 读取状态必须走同一个文件，否则会读到另一个（默认）位置的 Checkpoint。
CHECKPOINT_PATH: Path | None = None

run_store = RunStore()
coordinator = RunCoordinator(
    run_store,
    output_root=OUTPUT_ROOT,
    checkpointer_path=None,
    model_factory_builder=build_model_factory,
)


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

    # 允许显式指定 run_id，便于调用方在重试时复用同一个线程；
    # 不传则由服务生成。
    run_id: str | None = None


class ApprovalRequest(BaseModel):
    decision: str = Field(..., description="approve 或 reject")
    comment: str | None = None


class ResumeRequest(BaseModel):
    resume_value: dict[str, Any] | None = None


def _validate_run_id(run_id: str) -> str:
    if not RUN_ID_PATTERN.match(run_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="run_id 只允许字母、数字、下划线与短横线，长度 4-64",
        )
    return run_id


def _require_run(run_id: str):
    _validate_run_id(run_id)
    record = run_store.get(run_id)
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
    validated: list[str] = []

    for raw in paths:
        path = Path(raw)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.exists():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"数据源不存在：{raw}",
            )
        if not path.is_file():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"数据源不是文件：{raw}",
            )
        validated.append(str(path))

    return validated


@router.post("", status_code=status.HTTP_202_ACCEPTED)
def create_run(payload: CreateRunRequest) -> dict[str, Any]:
    """创建并启动一次分析，立即返回 run_id。"""
    from utils.run_logger import create_run_id

    run_id = _validate_run_id(payload.run_id) if payload.run_id else create_run_id()
    data_sources = _validate_data_sources(payload.data_sources)

    if not data_sources:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="至少需要一个数据源",
        )

    coordinator.start(
        run_id,
        data_sources=data_sources,
        analysis_focus=payload.analysis_focus,
        knowledge_sources=payload.knowledge_sources,
        require_plan_approval=payload.require_plan_approval,
    )

    return {
        "run_id": run_id,
        "status": "running",
        "require_plan_approval": payload.require_plan_approval,
    }


@router.get("")
def list_runs(limit: int = 50) -> dict[str, Any]:
    if limit < 1 or limit > 500:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="limit 需在 1-500 之间"
        )

    records = []
    for record in run_store.list(limit=limit):
        payload = record.to_dict()
        payload["stale"] = run_store.is_stale_running(record.run_id)
        records.append(payload)

    return {"runs": records}


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
    payload["stale"] = run_store.is_stale_running(run_id)
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

    if record.status not in ("waiting_approval", "interrupted", "failed"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"run 当前状态为 {record.status}，不可恢复",
        )

    try:
        coordinator.resume(
            run_id, resume_value=(payload.resume_value if payload else None)
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(error)
        ) from error

    return {"run_id": run_id, "status": "running"}


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

    # 重复审批保护：同一个审批 ID 已经决定过就直接返回，不再触发第二次执行
    for existing in record.approvals:
        if approval_id and existing.get("approval_id") == approval_id:
            return {
                "run_id": run_id,
                "status": record.status,
                "approval_id": approval_id,
                "duplicate": True,
                "existing_decision": existing.get("decision"),
            }

    if approval_id:
        run_store.record_approval(
            run_id,
            {
                "approval_id": approval_id,
                "run_id": run_id,
                "node": pending.get("node"),
                "kind": pending.get("kind"),
                "decision": decision,
                "comment": payload.comment,
            },
        )

    coordinator.resume(
        run_id,
        resume_value={"decision": decision, "comment": payload.comment},
    )

    return {
        "run_id": run_id,
        "status": "running",
        "approval_id": approval_id,
        "decision": decision,
        "duplicate": False,
    }


def register_run_routes(app: FastAPI) -> None:
    app.include_router(router)
