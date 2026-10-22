"""Run 生命周期编排。

把"持久化执行 + 人工审批"串起来，供 API 与测试脚本共用：

- 后台执行：`POST /runs` 立即返回，真实的执行交给一个受管理的执行器；
- 审批：执行到 interrupt 后落库为 waiting_approval，恢复时用 Command 继续。

执行器是一个单进程内、有界的线程池，不是分布式任务队列。它的语义边界是：

- 进程重启后不会自动续跑"running"状态的 Run（任务只存在于内存队列里）；
  这些 Run 需要显式 resume，避免服务重启时无提示地重复执行副作用。
- "waiting_approval" 与"completed/failed" 的状态是持久的，重启后可直接查询。
"""

from __future__ import annotations

import logging
import threading
import os
import uuid
from contextvars import copy_context
from collections import OrderedDict

import storage
from services.run_identity import LeaseLost, RunConflict, execution_lease
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from langgraph.types import Command

from core.runtime import McpSettings, SqlBackend
from ingestion.resources import ResourceResolver
from services.analysis_service import run_analysis
from services.run_notifier import notify_approval_needed, notify_run_finished
from services.run_store import STALE_HEARTBEAT_SECONDS, RunStore
from utils.run_logger import log_event

_LOGGER = logging.getLogger(__name__)


def _safe_notify(callback: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    """在后台线程里发通知，吞掉一切异常。

    两条理由：

    1. 通知是旁路——一次推送失败不能让已经跑完的 Run 变成 failed；
    2. 通知要访问外部服务，**不能拖住 Run 的收尾**。同步发的话，一个慢的 webhook
       会让 Run 迟迟不释放执行权，紧接着的审批/恢复请求会被判成"并发执行"而拒绝。
    """

    def dispatch() -> None:
        try:
            callback(*args, **kwargs)
        except Exception as error:  # noqa: BLE001
            _LOGGER.warning(
                "运行通知失败（已忽略）：%s: %s", type(error).__name__, error
            )

    threading.Thread(target=dispatch, name="run-notify", daemon=True).start()


class RunExecutor:
    """有界单进程执行器。"""

    def __init__(self, store: RunStore, *, max_workers: int = 2) -> None:
        self._store = store
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="insightflow-run"
        )
        self._lock = threading.Lock()
        self._futures: dict[str, Any] = {}
        self._recent: OrderedDict[str, Any] = OrderedDict()
        self._capacity = threading.BoundedSemaphore(max_workers+32)
        self._heartbeats: dict[str, threading.Event] = {}

    def submit(self, run_id: str, work: Callable[[], Any]) -> bool:
        context = copy_context()
        with self._lock:
            existing=self._futures.get(run_id)
            if existing is not None and not existing.done():
                return False
            if not self._capacity.acquire(blocking=False):
                raise RunConflict("执行队列已满，请稍后重试同一请求")
            try:
                future=self._pool.submit(context.run,work)
            except Exception:
                self._capacity.release()
                raise
            self._futures[run_id]=future
        def complete(done):
            with self._lock:
                if self._futures.get(run_id) is done:
                    self._futures.pop(run_id,None)
                self._recent[run_id]=done
                self._recent.move_to_end(run_id)
                while len(self._recent)>128:
                    self._recent.popitem(last=False)
            self._capacity.release()
        # Register outside the lock: an already-finished future calls back inline.
        future.add_done_callback(complete)
        return True

    def _stop_heartbeat(self, run_id: str) -> None:
        # Execution heartbeat belongs to the claimed lease, including wait=True workers.
        return None

    def is_active(self, run_id: str) -> bool:
        with self._lock:
            future = self._futures.get(run_id)
        return future is not None and not future.done()

    def wait(self, run_id: str, timeout: float | None = None) -> Any:
        with self._lock:
            future = self._futures.get(run_id) or self._recent.get(run_id)
        if future is None:
            return None
        return future.result(timeout=timeout)

    def shutdown(self, wait: bool = True) -> None:
        with self._lock:
            stops = list(self._heartbeats.values())
        for stop in stops:
            stop.set()
        self._pool.shutdown(wait=wait)


def run_interrupts(result: dict[str, Any]) -> list[dict[str, Any]]:
    """取出 LangGraph 在结果里报告的 interrupt 负载。"""
    raw = result.get("__interrupt__") if isinstance(result, dict) else None
    if not raw:
        return []

    interrupts: list[dict[str, Any]] = []
    for item in raw:
        value = getattr(item, "value", None)
        interrupts.append(value if isinstance(value, dict) else {"value": value})
    return interrupts


class RunCoordinator:
    """把 RunStore、执行器与工作流组合起来。"""

    def __init__(
        self,
        store: RunStore,
        *,
        output_root: Path | str = "outputs",
        checkpointer_path: Path | str | None = None,
        model_factory_builder: Callable[[str], Any] | None = None,
        max_workers: int = 2,
        mcp_settings: McpSettings | None = None,
    ) -> None:
        self.store = store
        self.output_root = Path(output_root)
        self.checkpointer_path = checkpointer_path
        self._model_factory_builder = model_factory_builder
        # 宿主固定的 MCP 参数（Server 模块、超时、私有目录的推导基准）。
        # 请求只能改**模式**，改不了这里的任何一项。
        self._mcp_template = mcp_settings or McpSettings()
        self.executor = RunExecutor(store, max_workers=max_workers)
        self._runners: dict[str, Callable[..., dict[str, Any]]] = {}

    def _settings_for(self, sql_backend: str) -> McpSettings:
        """按模式从宿主模板派生本次执行的 MCP 配置。

        只覆盖 `mode`；`server_module`、超时与私有目录一律沿用宿主模板，
        因此请求无法借后端选择去指定 Server 启动命令或宿主路径。
        """
        return replace(self._mcp_template, mode=SqlBackend(sql_backend))

    def _factory_for(self, run_id: str) -> Any:
        if self._model_factory_builder is None:
            raise RuntimeError(
                "未配置模型工厂：请为 RunCoordinator 提供 model_factory_builder"
            )
        return self._model_factory_builder(run_id)

    def _execute(
        self,
        run_id: str,
        *,
        data_sources: list[str],
        analysis_focus: str | None,
        knowledge_sources: list[str],
        resume: Any,
        require_plan_approval: bool,
        require_report_approval: bool = False,
        mcp: McpSettings | None = None,
        conversation_context: dict[str, Any] | None = None,
        memories: list[dict[str, Any]] | None = None,
        enable_dynamic_loop: bool = False,
        dynamic_limits: dict[str, Any] | None = None,
        continue_from_checkpoint: bool = False,
        assert_lease: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        from core.checkpoints import open_checkpointer

        with open_checkpointer(self.checkpointer_path) as saver:
            return run_analysis(
                data_sources=data_sources,
                analysis_focus=analysis_focus,
                knowledge_sources=knowledge_sources,
                run_id=run_id,
                model_factory=self._factory_for(run_id),
                checkpointer=saver,
                resolver=ResourceResolver(),
                output_root=self.output_root,
                resume=resume,
                continue_from_checkpoint=continue_from_checkpoint,
                assert_lease=assert_lease,
                require_plan_approval=require_plan_approval,
                require_report_approval=require_report_approval,
                mcp=mcp,
                conversation_context=conversation_context,
                memories=memories,
                enable_dynamic_loop=enable_dynamic_loop,
                dynamic_limits=dynamic_limits,
            )

    def _finalize(self, run_id: str, result: dict[str, Any]) -> dict[str, Any]:
        """把工作流结果映射成持久化的生命周期状态。"""
        interrupts = run_interrupts(result)

        if interrupts:
            pending = interrupts[0]
            self.store.update_status(
                run_id, "waiting_approval", pending_approval=pending
            )
            # 通知是旁路：失败只记日志，不影响已经确定的状态
            _safe_notify(notify_approval_needed, run_id, pending)
            return result

        run_status = result.get("run_status")
        if run_status not in ("completed", "failed"):
            analysis_status = result.get("analysis_status")
            run_status = (
                "completed"
                if analysis_status in ("completed", "partial")
                else "failed"
            )

        error = result.get("status_reason") if run_status == "failed" else None
        self.store.update_status(run_id, run_status, error=error)
        _safe_notify(notify_run_finished, run_id, run_status, result=result, error=error)
        return result

    def _fail(self, run_id: str, error: Exception, *, output_root: Path) -> None:
        """把异常映射成终态。

        `LlmCallAbandoned`（人工选择不重发未确认的调用）是受控终止，
        其余异常也一律落 failed——不能让 Run 停在 running。

        日志目录跟随 `output_root`，因此这里必须显式绑定运行期上下文：
        `run_analysis` 已经退出它的 `use_runtime`，不重新绑定的话
        `utils.run_logger` 会退回相对当前工作目录的默认位置，
        同一个 Run 的失败记录会落到另一棵目录树里。
        """
        from core.runtime import RunRuntime, use_runtime

        runtime = RunRuntime(
            run_id=run_id,
            model_factory=None,  # type: ignore[arg-type] - 仅用于解析日志目录
            resolver=None,  # type: ignore[arg-type]
            output_root=Path(output_root),
        )
        with use_runtime(runtime):
            log_event(
                run_id=run_id,
                node="coordinator",
                event="run_failed",
                status="failed",
                details={
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
        self.store.update_status(run_id, "failed", error=str(error))
        _safe_notify(notify_run_finished, run_id, "failed", error=str(error))

    @staticmethod
    def _mode() -> str:
        mode = (os.getenv("INSIGHTFLOW_EXECUTION") or "inline").strip().lower()
        if mode not in ("inline", "queue"):
            raise ValueError("INSIGHTFLOW_EXECUTION 只能是 inline 或 queue")
        if mode == "queue" and not storage.current_config().is_postgres:
            raise ValueError("queue 执行模式需要 PostgreSQL")
        return mode

    def dispatch(self, run_id: str, *, owner: str | None = None, **payload):
        owner = owner or storage.current_owner()
        mode = self._mode()
        defaults = dict(analysis_focus=None,knowledge_sources=[],require_plan_approval=False,
                        require_report_approval=False,sql_backend="direct",conversation_context=None,
                        memories=None,enable_dynamic_loop=False,dynamic_limits=None)
        defaults.update(payload)
        payload = defaults
        record = self.store.create(run_id,data_sources=payload["data_sources"],analysis_focus=payload["analysis_focus"],
            sql_backend=payload["sql_backend"],require_plan_approval=payload["require_plan_approval"],
            require_report_approval=payload["require_report_approval"],owner=owner,request_payload=payload)
        if record.status not in ("pending","queued"):
            return record
        if mode == "queue":
            from services import run_queue
            run_queue.ensure_schema()
            self.store.update_status(run_id,"queued",owner=owner)
            run_queue.enqueue(run_id,{**payload,"_owner":owner})
        else:
            self.start(run_id,owner=owner,record_exists=True,**payload)
        return self.store.get(run_id,owner=owner)

    def _leased_work(self, run_id, owner, payload, *, resume=False, resume_value=None):
        token = uuid.uuid4().hex
        if not self.store.claim(run_id,token,owner=owner,resume=resume):
            raise RunConflict("运行正在执行或当前状态不可启动")
        stop = threading.Event()
        def check():
            self.store.assert_lease(run_id,token,owner=owner)
        def beat():
            while not stop.wait(max(1.0,STALE_HEARTBEAT_SECONDS/6)):
                try:
                    self.store.touch(run_id,token=token,owner=owner)
                except Exception:
                    stop.set()
                    return
        heartbeat = threading.Thread(target=beat,name="run-lease-heartbeat",daemon=True)
        owner_handle=storage.set_current_owner(owner)
        try:
            with execution_lease(run_id,token):
                heartbeat.start()
                try:
                    check()
                    command=Command(resume=resume_value) if resume and resume_value is not None else None
                    result=self._execute(run_id,
                        data_sources=payload["data_sources"],analysis_focus=payload.get("analysis_focus"),
                        knowledge_sources=payload.get("knowledge_sources") or [],resume=command,
                        require_plan_approval=payload.get("require_plan_approval",False),
                        require_report_approval=payload.get("require_report_approval",False),
                        mcp=self._settings_for(payload.get("sql_backend","direct")),
                        conversation_context=payload.get("conversation_context"),memories=payload.get("memories"),
                        enable_dynamic_loop=payload.get("enable_dynamic_loop",False),dynamic_limits=payload.get("dynamic_limits"),
                        continue_from_checkpoint=resume and command is None,assert_lease=check)
                    check()
                    return self._finalize(run_id,result)
                except LeaseLost:
                    raise
                except Exception as error:
                    check()
                    self._fail(run_id,error,output_root=self.output_root)
                    raise
                finally:
                    stop.set()
        finally:
            storage.reset_current_owner(owner_handle)

    def start(self, run_id: str, *, data_sources: list[str], analysis_focus=None, knowledge_sources=None,
              require_plan_approval=False,require_report_approval=False,sql_backend="direct",
              conversation_context=None,memories=None,enable_dynamic_loop=False,dynamic_limits=None,
              record_exists=False,wait=False,owner=None):
        owner=owner or storage.current_owner()
        payload=dict(data_sources=data_sources,analysis_focus=analysis_focus,knowledge_sources=knowledge_sources or [],
            require_plan_approval=require_plan_approval,require_report_approval=require_report_approval,
            sql_backend=sql_backend,conversation_context=conversation_context,memories=memories,
            enable_dynamic_loop=enable_dynamic_loop,dynamic_limits=dynamic_limits)
        if not record_exists:
            record=self.store.create(run_id,data_sources=data_sources,analysis_focus=analysis_focus,sql_backend=sql_backend,
                require_plan_approval=require_plan_approval,require_report_approval=require_report_approval,
                owner=owner,request_payload=payload)
        else:
            record=self.store.get(run_id,owner=owner)
            if record is None:
                raise KeyError("未找到运行")
        if record.status not in ("pending","queued"):
            return record
        # The durable request is authoritative, including worker/retry starts.
        # A caller with record_exists=True cannot silently switch its backend.
        payload = dict(record.request_payload or dict(data_sources=record.data_sources,
            analysis_focus=record.analysis_focus,sql_backend=record.sql_backend,
            require_plan_approval=record.require_plan_approval,
            require_report_approval=record.require_report_approval))
        def work():
            return self._leased_work(run_id,owner,payload)
        if wait:
            return work()
        self.executor.submit(run_id,work)
        return record

    def resume(self, run_id: str, *, resume_value=None, wait=False, from_queue=False,owner=None):
        owner=owner or storage.current_owner()
        record=self.store.get(run_id,owner=owner)
        if record is None:
            raise KeyError("未找到运行")
        if record.status == "queued" and not from_queue:
            return record
        if record.status not in ("waiting_approval","interrupted","failed") and not self.store.is_stale_running(run_id) and not (from_queue and record.status=="queued"):
            raise RunConflict(f"run 当前状态为 {record.status}，不可恢复")
        if self.executor.is_active(run_id):
            raise RunConflict("运行正在执行，拒绝并发恢复")
        if self._mode()=="queue" and not from_queue:
            from services import run_queue
            run_queue.ensure_schema()
            self.store.update_status(run_id,"queued",owner=owner)
            run_queue.enqueue(run_id,{"_owner":owner,"_resume":True,"resume_value":resume_value},continuation=True)
            return self.store.get(run_id,owner=owner)
        payload=record.request_payload or dict(data_sources=record.data_sources,analysis_focus=record.analysis_focus,
            sql_backend=record.sql_backend,require_plan_approval=record.require_plan_approval,
            require_report_approval=record.require_report_approval)
        def work():
            return self._leased_work(run_id,owner,payload,resume=True,resume_value=resume_value)
        if wait:
            return work()
        self.executor.submit(run_id,work)
        return record

    def shutdown(self) -> None:
        self.executor.shutdown(wait=True)
