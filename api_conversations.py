"""会话与长期记忆的 HTTP 接口。

与 Run 接口的关系：

- `POST /sessions/{id}/messages` 创建新的 Turn 与 Run，返回 202 + `run_id`；
  分析本身是异步的，接口不会假装已经拿到答案。
- 恢复未完成的 Run 仍走 `POST /runs/{run_id}/resume`（HITL / Checkpoint），
  与本文件的"新消息"是两条独立路径。
- `GET /turns/{turn_id}` 返回该 Turn 对应的 Run、执行状态与结果引用；
  状态从权威 Run 存储同步而来。

归属：本项目没有真实身份认证，`owner` 来自服务端配置的单用户演示标识，
不接受客户端自报的 user_id。多用户部署必须替换为可信身份来源。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Query, status
from pydantic import BaseModel, Field

import storage
from services.conversation_service import ConversationService
from services.memory_service import MemoryService, MemoryValidationError

router = APIRouter(tags=["conversations"])


class CreateSessionRequest(BaseModel):
    title: str | None = None
    dataset_refs: list[str] = Field(default_factory=list)
    knowledge_refs: list[str] = Field(default_factory=list)


class PostMessageRequest(BaseModel):
    content: str
    client_message_id: str | None = None
    analysis_focus: str | None = None
    require_plan_approval: bool = False
    require_report_approval: bool = False
    sql_backend: str = "direct"
    enable_dynamic_loop: bool = False
    dynamic_limits: dict[str, Any] | None = None


class MemoryRequest(BaseModel):
    kind: str
    key: str
    value: str
    scope: str = "global"
    status: str = "candidate"
    provenance: dict[str, Any] = Field(default_factory=dict)


class MemoryUpdateRequest(BaseModel):
    value: str | None = None
    scope: str | None = None
    status: str | None = None


def _service() -> ConversationService:
    from api_runs import get_coordinator

    return ConversationService(get_coordinator())


def _memory() -> MemoryService:
    return MemoryService()


@router.post("/sessions", status_code=status.HTTP_201_CREATED)
def create_session(payload: CreateSessionRequest) -> dict[str, Any]:
    from api_runs import _validate_data_sources, _validate_knowledge_sources

    dataset_refs = _validate_data_sources(payload.dataset_refs)
    if not dataset_refs:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="创建会话至少需要一个数据集",
        )
    return _service().create_session(
        title=payload.title,
        dataset_refs=dataset_refs,
        knowledge_refs=_validate_knowledge_sources(payload.knowledge_refs),
    )


@router.get("/sessions")
def list_sessions(limit: int = Query(default=50, ge=1, le=200)) -> dict[str, Any]:
    return {"sessions": _service().list_sessions(limit=limit)}


@router.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    payload = _service().get_session(session_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return payload


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> dict[str, Any]:
    deleted = _service().delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"session_id": session_id, "deleted": True}


@router.get("/sessions/{session_id}/messages")
def list_messages(session_id: str) -> dict[str, Any]:
    if _service().get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"messages": _service().list_messages(session_id)}


@router.get("/sessions/{session_id}/turns")
def list_turns(session_id: str) -> dict[str, Any]:
    if _service().get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"turns": _service().list_turns(session_id)}


@router.post(
    "/sessions/{session_id}/messages",
    status_code=status.HTTP_202_ACCEPTED,
)
def post_message(session_id: str, payload: PostMessageRequest) -> dict[str, Any]:
    """提交一条消息：创建新的 Turn 与 Run，立即返回 run_id。"""
    if not payload.content.strip():
        raise HTTPException(status_code=400, detail="消息内容不能为空")

    # 请求本身就非法的输入在这里拒绝：避免先写进一条用户消息和一个轮次，
    # 再在启动阶段失败，留下一条永远 pending 的记录。
    from api_runs import _validate_sql_backend

    sql_backend = _validate_sql_backend(payload.sql_backend)

    try:
        result = _service().post_message(
            session_id,
            payload.content,
            client_message_id=payload.client_message_id,
            analysis_focus=payload.analysis_focus,
            require_plan_approval=payload.require_plan_approval,
            require_report_approval=payload.require_report_approval,
            sql_backend=sql_backend,
            enable_dynamic_loop=payload.enable_dynamic_loop,
            dynamic_limits=payload.dynamic_limits,
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    turn = result.get("turn") or {}
    return {
        "turn_id": turn.get("turn_id"),
        "session_id": session_id,
        "run_id": result["run_id"],
        "status": turn.get("status", "running"),
        "deduplicated": result.get("deduplicated", False),
    }


@router.get("/turns/{turn_id}")
def get_turn(turn_id: str) -> dict[str, Any]:
    payload = _service().get_turn(turn_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="轮次不存在")
    return payload


@router.get("/turns/{turn_id}/result")
def get_turn_result(turn_id: str) -> dict[str, Any]:
    """该轮的分析结果与产物引用（未完成时明确告知，不返回空报告）。"""
    payload = _service().get_turn(turn_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="轮次不存在")
    if payload["status"] != "completed":
        return {
            "turn_id": turn_id,
            "run_id": payload["run_id"],
            "status": payload["status"],
            "result": None,
            "reason": payload.get("status_reason"),
        }
    return {
        "turn_id": turn_id,
        "run_id": payload["run_id"],
        "status": "completed",
        "result": payload.get("result") or {},
    }


# -- 长期记忆 -------------------------------------------------------------


@router.get("/memories")
def list_memories(
    scope: str | None = None,
    include_candidates: bool = False,
) -> dict[str, Any]:
    return {
        "memories": _memory().list(scope=scope, include_candidates=include_candidates)
    }


@router.post("/memories", status_code=status.HTTP_201_CREATED)
def create_memory(payload: MemoryRequest) -> dict[str, Any]:
    """默认写入候选；`status="confirmed"` 表示用户已明确确认。"""
    try:
        return _memory().add(
            kind=payload.kind,
            key=payload.key,
            value=payload.value,
            scope=payload.scope,
            status=payload.status,
            provenance=payload.provenance,
        )
    except MemoryValidationError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post("/memories/{memory_id}/confirm")
def confirm_memory(memory_id: str) -> dict[str, Any]:
    payload = _memory().confirm(memory_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="记忆不存在")
    return payload


@router.patch("/memories/{memory_id}")
def update_memory(memory_id: str, payload: MemoryUpdateRequest) -> dict[str, Any]:
    try:
        updated = _memory().update(
            memory_id,
            value=payload.value,
            scope=payload.scope,
            status=payload.status,
        )
    except MemoryValidationError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if updated is None:
        raise HTTPException(status_code=404, detail="记忆不存在")
    return updated


@router.delete("/memories/{memory_id}")
def delete_memory(memory_id: str) -> dict[str, Any]:
    if not _memory().delete(memory_id):
        raise HTTPException(status_code=404, detail="记忆不存在")
    return {"memory_id": memory_id, "deleted": True}


@router.get("/memories/retrieve")
def retrieve_memories(
    q: str = Query(min_length=1),
    scope: str | None = None,
    limit: int = Query(default=5, ge=1, le=20),
) -> dict[str, Any]:
    scopes = [scope] if scope else None
    return {"memories": _memory().retrieve(q, scopes=scopes, limit=limit)}


@router.get("/storage/health")
def storage_health() -> dict[str, Any]:
    """存储后端连通性：数据库、表、缓存。不含凭据。"""
    from services import storage_factory

    payload: dict[str, Any] = {"backends": storage_factory.describe()}
    try:
        payload["database"] = storage.healthcheck(storage.database_url())
    except Exception as error:  # noqa: BLE001 - 健康检查要报告而不是抛出
        payload["database"] = {"reachable": False, "error": str(error)}
    return payload


def register_conversation_routes(app: FastAPI) -> None:
    app.include_router(router)
