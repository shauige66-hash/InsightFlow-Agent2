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

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from langgraph.types import Command

from ingestion.resources import ResourceResolver
from services.analysis_service import run_analysis
from services.run_store import STALE_HEARTBEAT_SECONDS, RunStore
from utils.run_logger import log_event


class RunExecutor:
    """有界单进程执行器。"""

    def __init__(self, store: RunStore, *, max_workers: int = 2) -> None:
        self._store = store
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="insightflow-run"
        )
        self._lock = threading.Lock()
        self._futures: dict[str, Any] = {}
        self._heartbeats: dict[str, threading.Event] = {}

    def submit(self, run_id: str, work: Callable[[], Any]) -> None:
        stop = threading.Event()
        heartbeat = threading.Thread(
            target=self._beat, args=(run_id, stop), daemon=True
        )

        with self._lock:
            self._futures[run_id] = self._pool.submit(work)
            self._heartbeats[run_id] = stop
            heartbeat.start()

    def _beat(self, run_id: str, stop: threading.Event) -> None:
        """执行期间定期刷新心跳。

        没有心跳就无法区分"正在执行"与"执行者已崩溃"，
        被杀死的 Run 会永远显示为 running。
        """
        interval = max(1.0, STALE_HEARTBEAT_SECONDS / 6)
        while not stop.wait(interval):
            try:
                self._store.touch(run_id)
            except Exception:  # noqa: BLE001 - 心跳失败不应影响执行
                return

    def _stop_heartbeat(self, run_id: str) -> None:
        with self._lock:
            stop = self._heartbeats.pop(run_id, None)
        if stop is not None:
            stop.set()

    def is_active(self, run_id: str) -> bool:
        with self._lock:
            future = self._futures.get(run_id)
        return future is not None and not future.done()

    def wait(self, run_id: str, timeout: float | None = None) -> Any:
        with self._lock:
            future = self._futures.get(run_id)
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
    ) -> None:
        self.store = store
        self.output_root = Path(output_root)
        self.checkpointer_path = checkpointer_path
        self._model_factory_builder = model_factory_builder
        self.executor = RunExecutor(store, max_workers=max_workers)
        self._runners: dict[str, Callable[..., dict[str, Any]]] = {}

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
                require_plan_approval=require_plan_approval,
            )

    def _finalize(self, run_id: str, result: dict[str, Any]) -> dict[str, Any]:
        """把工作流结果映射成持久化的生命周期状态。"""
        interrupts = run_interrupts(result)

        if interrupts:
            pending = interrupts[0]
            self.store.update_status(
                run_id, "waiting_approval", pending_approval=pending
            )
            return result

        run_status = result.get("run_status")
        if run_status not in ("completed", "failed"):
            analysis_status = result.get("analysis_status")
            run_status = (
                "completed"
                if analysis_status in ("completed", "partial")
                else "failed"
            )

        self.store.update_status(
            run_id,
            run_status,
            error=result.get("status_reason") if run_status == "failed" else None,
        )
        return result

    def _fail(self, run_id: str, error: Exception) -> None:
        """把异常映射成终态。

        `LlmCallAbandoned`（人工选择不重发未确认的调用）是受控终止，
        其余异常也一律落 failed——不能让 Run 停在 running。
        """
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

    def start(
        self,
        run_id: str,
        *,
        data_sources: list[str],
        analysis_focus: str | None = None,
        knowledge_sources: list[str] | None = None,
        require_plan_approval: bool = False,
    ) -> None:
        """登记并异步启动一次运行。"""
        self.store.create(
            run_id, data_sources=data_sources, analysis_focus=analysis_focus
        )
        self.store.update_status(run_id, "running")

        def work() -> dict[str, Any]:
            try:
                result = self._execute(
                    run_id,
                    data_sources=data_sources,
                    analysis_focus=analysis_focus,
                    knowledge_sources=knowledge_sources or [],
                    resume=None,
                    require_plan_approval=require_plan_approval,
                )
                return self._finalize(run_id, result)
            except Exception as error:  # noqa: BLE001 - 失败必须落到 Run 状态
                self._fail(run_id, error)
                raise
            finally:
                self.executor._stop_heartbeat(run_id)

        self.executor.submit(run_id, work)

    def resume(self, run_id: str, *, resume_value: Any = None) -> None:
        """从最后提交的 Checkpoint 继续执行。

        审批决定通过 `Command(resume=...)` 交回被中断的节点。
        """
        record = self.store.get(run_id)
        if record is None:
            raise KeyError(f"未登记的 run：{run_id}")

        if record.status not in ("waiting_approval", "interrupted", "failed"):
            raise ValueError(
                f"run {run_id} 当前状态为 {record.status}，不可恢复"
            )

        if self.executor.is_active(run_id):
            raise ValueError(f"run {run_id} 正在执行中，拒绝并发的恢复请求")

        self.store.update_status(run_id, "running")

        command = Command(resume=resume_value) if resume_value is not None else None

        def work() -> dict[str, Any]:
            try:
                result = self._execute(
                    run_id,
                    data_sources=record.data_sources,
                    analysis_focus=record.analysis_focus,
                    knowledge_sources=[],
                    resume=command,
                    require_plan_approval=True,
                )
                # 恢复同样要落状态：否则等待重试的审批不会反映到生命周期上，
                # 查询接口会一直显示 running。
                return self._finalize(run_id, result)
            except Exception as error:  # noqa: BLE001
                self._fail(run_id, error)
                raise
            finally:
                self.executor._stop_heartbeat(run_id)

        self.executor.submit(run_id, work)

    def shutdown(self) -> None:
        self.executor.shutdown(wait=True)
