"""多轮对话服务：会话 → 轮次 → Run → 结果/产物。

两条路径刻意分开，不能混用：

- **新消息**（`post_message`）创建新的 Turn 与新的 Run，并复用必要的历史上下文；
- **恢复**（`resume_run`）只继续某个 Run 的 HITL / Checkpoint，不创建新 Turn。

轮次的展示状态从权威 Run 存储同步而来，反向不成立：`conversation_turns.status`
只是给人看的，执行是否完成以 RunStore / Checkpoint 为准。因此这里每次读取
轮次都会先同步，避免把"请求发出去了"当成"分析完成了"。

助手消息由真实结果生成：只有 Run 已 completed 且拿到了报告内容，才写入
助手消息；拿不到就保持 pending/running，绝不用占位文字假装完成。
"""

from __future__ import annotations

import logging
from typing import Any

import storage
from storage.conversations import ConversationRepository
from storage.models import ConversationMessage, ConversationSession, ConversationTurn

logger = logging.getLogger(__name__)

#: 同步时可写入会话结果的产物上限，避免把整份报告塞进数据库
MAX_RESULT_PREVIEW = 4000

#: Run 状态到轮次展示状态的映射
_STATUS_MAP = {
    "pending": "pending",
    "running": "running",
    "waiting_approval": "waiting_approval",
    "completed": "completed",
    "failed": "failed",
    "cancelled": "failed",
}


class ConversationService:
    def __init__(self, coordinator: Any, *, output_root: Any = None) -> None:
        self._coordinator = coordinator
        self._output_root = output_root

    @property
    def _owner(self) -> str:
        """每次调用现取，而不是在构造时缓存。

        服务实例可能被多个请求复用；把 owner 缓存在构造时，会让并发的第二个
        用户看到第一个用户的会话。
        """
        return storage.current_owner()

    # -- 会话 -------------------------------------------------------------

    def _sessions(self, session):
        return ConversationRepository(session, owner=self._owner)

    def create_session(
        self,
        *,
        title: str | None = None,
        dataset_refs: list[str] | None = None,
        knowledge_refs: list[str] | None = None,
    ) -> dict[str, Any]:
        with storage.session_scope(storage.database_url()) as session:
            record = self._sessions(session).create_session(
                title=title,
                dataset_refs=dataset_refs or [],
                knowledge_refs=knowledge_refs or [],
            )
            return _session_dict(record)

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with storage.session_scope(storage.database_url()) as session:
            record = self._sessions(session).get_session(session_id)
            if record is None:
                return None
            payload = _session_dict(record)
            payload["turn_count"] = len(self._sessions(session).list_turns(session_id))
            return payload

    def list_sessions(self, *, limit: int = 50) -> list[dict[str, Any]]:
        with storage.session_scope(storage.database_url()) as session:
            return [
                _session_dict(record)
                for record in self._sessions(session).list_sessions(limit=limit)
            ]

    def delete_session(self, session_id: str) -> bool:
        with storage.session_scope(storage.database_url()) as session:
            return self._sessions(session).delete_session(session_id)

    # -- 消息与轮次 -------------------------------------------------------

    def list_messages(self, session_id: str, *, limit: int = 200) -> list[dict[str, Any]]:
        with storage.session_scope(storage.database_url()) as session:
            repository = self._sessions(session)
            if repository.get_session(session_id) is None:
                return []
            turns = repository.list_turns(session_id, limit=limit)
            for turn in turns:
                self._sync_turn(repository, turn)
            session.flush()
            return [
                _message_dict(record)
                for record in repository.list_messages(session_id, limit=limit)
            ]

    def list_turns(self, session_id: str) -> list[dict[str, Any]]:
        with storage.session_scope(storage.database_url()) as session:
            repository = self._sessions(session)
            turns = repository.list_turns(session_id)
            for turn in turns:
                self._sync_turn(repository, turn)
            session.flush()
            return [_turn_dict(turn) for turn in turns]

    def get_turn(self, turn_id: str) -> dict[str, Any] | None:
        with storage.session_scope(storage.database_url()) as session:
            repository = self._sessions(session)
            turn = repository.get_turn(turn_id)
            if turn is None:
                return None
            self._sync_turn(repository, turn)
            session.flush()
            return _turn_dict(turn)

    def post_message(
        self,
        session_id: str,
        content: str,
        *,
        client_message_id: str | None = None,
        analysis_focus: str | None = None,
        require_plan_approval: bool = False,
        require_report_approval: bool = False,
        sql_backend: str = "direct",
        enable_dynamic_loop: bool = False,
        dynamic_limits: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """提交一条用户消息：创建 Turn + Run 并异步启动。"""
        with storage.session_scope(storage.database_url()) as session:
            repository = self._sessions(session)
            record = repository.get_session(session_id)
            if record is None:
                raise KeyError(f"会话不存在：{session_id}")

            if client_message_id:
                existing = repository.find_message_by_client_id(
                    session_id, client_message_id
                )
                if existing is not None and existing.turn_id:
                    turn = repository.get_turn(existing.turn_id)
                    return {
                        "turn": _turn_dict(turn) if turn else None,
                        "run_id": existing.run_id,
                        "deduplicated": True,
                    }

            dataset_refs = list(record.dataset_refs or [])
            if not dataset_refs:
                raise ValueError("会话没有绑定数据集，无法开始分析")

            # 先读历史再建本轮，避免把当前这条消息算进"上一轮"
            history = self._history_snapshot(repository, session_id)

            run_id = f"run_{storage_conversation_id()}"
            turn = repository.create_turn(
                session_id=session_id, run_id=run_id, status="pending"
            )
            message = repository.add_message(
                session_id=session_id,
                role="user",
                content=content,
                turn_id=turn.id,
                run_id=run_id,
                client_message_id=client_message_id,
            )
            turn.user_message_id = message.id
            repository.link_run(session_id=session_id, turn_id=turn.id, run_id=run_id)
            turn_payload = _turn_dict(turn)
            knowledge_refs = list(record.knowledge_refs or [])

        # 先记候选记忆（作用域在写入前判定），再启动 Run：写失败也不影响对话
        self._record_candidates(content, session_id=session_id, dataset_refs=dataset_refs)

        # 启动可能失败（线程池关闭、后端模式非法、登记写入失败）。轮次与用户
        # 消息已经落库，如果不收敛状态，这条消息会永远停在 pending——界面上
        # 既看不到结果，也看不到失败原因。
        try:
            self._coordinator.start(
                run_id,
                data_sources=dataset_refs,
                analysis_focus=analysis_focus or content,
                knowledge_sources=knowledge_refs,
                require_plan_approval=require_plan_approval,
                require_report_approval=require_report_approval,
                sql_backend=sql_backend,
                conversation_context=self._context_payload(
                    history, dataset_refs=dataset_refs
                ),
                memories=self._memories_for(content, dataset_refs=dataset_refs),
                enable_dynamic_loop=enable_dynamic_loop,
                dynamic_limits=dynamic_limits,
            )
        except Exception as error:
            self._mark_start_failed(turn_payload["turn_id"], run_id, error)
            raise

        with storage.session_scope(storage.database_url()) as session:
            repository = self._sessions(session)
            repository.update_turn(turn_payload["turn_id"], status="running")
            turn_payload["status"] = "running"

        return {
            "turn": turn_payload,
            "run_id": run_id,
            "deduplicated": False,
        }

    def _mark_start_failed(self, turn_id: str, run_id: str, error: Exception) -> None:
        """把"启动失败"写进轮次与 Run，而不是让它停在 pending。"""
        reason = f"启动分析失败：{error}"[:500]
        with storage.session_scope(storage.database_url()) as session:
            self._sessions(session).update_turn(
                turn_id, status="failed", status_reason=reason
            )

        store = getattr(self._coordinator, "store", None)
        if store is None:
            return
        try:
            if store.get(run_id) is not None:
                store.update_status(run_id, "failed", error=reason)
        except Exception:  # noqa: BLE001 - 收尾失败不应掩盖原始异常
            logger.warning("记录启动失败时读不到 Run 状态：%s", run_id, exc_info=True)

    def resume_run(self, run_id: str, *, resume_value: Any = None) -> dict[str, Any]:
        """恢复同一个 Run（HITL / Checkpoint），不创建新 Turn。"""
        self._coordinator.resume(run_id, resume_value=resume_value)
        return {"run_id": run_id, "resumed": True}

    # -- 多轮上下文 -------------------------------------------------------

    def _history_snapshot(
        self, repository: ConversationRepository, session_id: str
    ) -> dict[str, Any]:
        """当前轮之前的历史：消息与已完成轮次的真实摘要。"""
        messages = [
            message
            for message in repository.list_messages(session_id, limit=20)
            if message.role in ("user", "assistant")
        ]
        turns = repository.list_turns(session_id, limit=10)
        return {
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages[-6:]
            ],
            "turns": [
                {
                    "seq": turn.seq,
                    "status": turn.status,
                    # 摘要只来自真实结果；没有结果就是空
                    "summary": str((turn.result or {}).get("preview") or "")[:400],
                }
                for turn in turns[-4:]
            ],
        }

    @staticmethod
    def _context_payload(
        history: dict[str, Any], *, dataset_refs: list[str]
    ) -> dict[str, Any]:
        return {
            "recent_messages": list(history.get("messages") or []),
            "previous_turns": [
                item for item in (history.get("turns") or []) if item.get("summary")
            ],
            "dataset_refs": list(dataset_refs),
        }

    @staticmethod
    def _memories_for(query: str, *, dataset_refs: list[str]) -> list[dict[str, Any]]:
        """新任务开始时召回少量已确认记忆（全局 + 当前数据集）。"""
        from services.memory_service import MemoryService

        service = MemoryService()
        return service.retrieve(
            query,
            scopes=service.scopes_for(dataset_refs),
            limit=3,
        )

    def _record_candidates(
        self, content: str, *, session_id: str, dataset_refs: list[str]
    ) -> None:
        """把用户消息里的偏好记成候选记忆（不确认、不进上下文）。

        作用域由 `classify_scope` 在写入前判定：说"这次报告用红色"只落到本会话，
        说"金额统一用人民币"才落到 global。写失败不影响消息本身——候选记忆是旁路。
        """
        from services.memory_service import MemoryService

        try:
            service = MemoryService()
            for candidate in service.propose_candidates(
                content,
                provenance={"session_id": session_id},
                dataset_refs=dataset_refs,
                session_id=session_id,
            ):
                service.add(**candidate)
        except Exception:  # noqa: BLE001 - 记忆是旁路，不能拖垮对话
            logger.warning("记录候选记忆失败（已忽略）", exc_info=True)

    # -- 同步 -------------------------------------------------------------

    def _sync_turn(self, repository: ConversationRepository, turn: ConversationTurn) -> None:
        """把权威 Run 状态同步到轮次；完成后补写助手消息。"""
        if turn.run_id is None:
            return
        record = self._run_record(turn.run_id)
        if record is None:
            return

        status = _STATUS_MAP.get(record.status, turn.status)
        if status == "waiting_approval":
            repository.update_turn(turn.id, status="waiting_approval")
            return

        if record.status == "completed":
            result = self._completed_result(turn)
            if result is not None:
                repository.update_turn(turn.id, status="completed", result=result)
                self._ensure_assistant_message(repository, turn, result)
            else:
                # 报告还没落盘：保持运行中，不假装完成
                repository.update_turn(turn.id, status="running")
            return

        if record.status == "failed":
            repository.update_turn(
                turn.id, status="failed", status_reason=record.error or "分析失败"
            )
            return

        if turn.status != status:
            repository.update_turn(turn.id, status=status)

    def _checkpointer(self):
        """必须与 Coordinator 用同一个 Checkpoint，否则读不到 Run 的进度。"""
        from core.checkpoints import open_checkpointer

        return open_checkpointer(getattr(self._coordinator, "checkpointer_path", None))

    def _run_record(self, run_id: str):
        store = getattr(self._coordinator, "store", None)
        try:
            if store is not None:
                return store.get(run_id)
            from services import storage_factory

            return storage_factory.run_store(self._output_root or "outputs").get(run_id)
        except Exception:  # noqa: BLE001 - 读状态失败不该让消息列表 500
            logger.warning("无法读取 Run 状态：%s", run_id, exc_info=True)
            return None

    def _completed_result(self, turn: ConversationTurn) -> dict[str, Any] | None:
        """已完成的 Run：取报告与产物引用；拿不到就返回 None。"""
        try:
            from services.analysis_service import read_run_state

            with self._checkpointer() as saver:
                state = read_run_state(turn.run_id, checkpointer=saver)
        except Exception:  # noqa: BLE001
            logger.warning("无法读取 Run 状态：%s", turn.run_id, exc_info=True)
            return None

        if not state:
            return None
        report_path = state.get("report_path")
        if not report_path:
            return None

        try:
            content = open(report_path, encoding="utf-8").read()
        except OSError:
            return None

        return {
            "report_path": report_path,
            "artifacts": list(state.get("artifacts") or []),
            "analysis_status": state.get("analysis_status"),
            "preview": content[:MAX_RESULT_PREVIEW],
        }

    def _ensure_assistant_message(
        self,
        repository: ConversationRepository,
        turn: ConversationTurn,
        result: dict[str, Any],
    ) -> None:
        """助手消息只在结果真实存在时写一次。"""
        existing = [
            message
            for message in repository.list_messages(turn.session_id)
            if message.role == "assistant" and message.turn_id == turn.id
        ]
        if existing:
            return
        preview = result.get("preview") or ""
        repository.add_message(
            session_id=turn.session_id,
            role="assistant",
            content=preview,
            turn_id=turn.id,
            run_id=turn.run_id,
        )


def storage_conversation_id() -> str:
    import uuid

    return uuid.uuid4().hex[:16]


def _session_dict(record: ConversationSession) -> dict[str, Any]:
    return {
        "session_id": record.id,
        "title": record.title,
        "owner": record.owner,
        "dataset_refs": list(record.dataset_refs or []),
        "knowledge_refs": list(record.knowledge_refs or []),
        "created_at": record.created_at.isoformat() if record.created_at else None,
        "updated_at": record.updated_at.isoformat() if record.updated_at else None,
    }


def _turn_dict(record: ConversationTurn | None) -> dict[str, Any] | None:
    if record is None:
        return None
    return {
        "turn_id": record.id,
        "session_id": record.session_id,
        "seq": record.seq,
        "run_id": record.run_id,
        "status": record.status,
        "status_reason": record.status_reason,
        "result": record.result or {},
    }


def _message_dict(record: ConversationMessage) -> dict[str, Any]:
    return {
        "message_id": record.id,
        "session_id": record.session_id,
        "turn_id": record.turn_id,
        "run_id": record.run_id,
        "role": record.role,
        "content": record.content,
        "created_at": record.created_at.isoformat() if record.created_at else None,
    }


__all__ = ["ConversationService"]
