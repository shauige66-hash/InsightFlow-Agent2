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

    def start(
        self,
        run_id: str,
        *,
        data_sources: list[str],
        analysis_focus: str | None = None,
        knowledge_sources: list[str] | None = None,
        require_plan_approval: bool = False,
        require_report_approval: bool = False,
        sql_backend: str = "direct",
        conversation_context: dict[str, Any] | None = None,
        memories: list[dict[str, Any]] | None = None,
        enable_dynamic_loop: bool = False,
        dynamic_limits: dict[str, Any] | None = None,
        record_exists: bool = False,
        wait: bool = False,
    ) -> Any:
        """登记并启动一次运行。

        `record_exists`：队列模式下 Run 记录已由 API 建好（状态 queued），worker
        抢占后不应再 `create` 一次，否则会把已有记录覆盖掉。
        `wait`：在 worker 进程里直接于当前线程执行，不投递线程池——worker 本身
        就是那个"后台"，再套一层线程只会让关闭信号更难处理。
        """
        if not record_exists:
            self.store.create(
                run_id,
                data_sources=data_sources,
                analysis_focus=analysis_focus,
                sql_backend=sql_backend,
                require_plan_approval=require_plan_approval,
                require_report_approval=require_report_approval,
            )
        self.store.update_status(run_id, "running")

        mcp = self._settings_for(sql_backend)

        def work() -> dict[str, Any]:
            try:
                result = self._execute(
                    run_id,
                    data_sources=data_sources,
                    analysis_focus=analysis_focus,
                    knowledge_sources=knowledge_sources or [],
                    resume=None,
                    require_plan_approval=require_plan_approval,
                    require_report_approval=require_report_approval,
                    mcp=mcp,
                    conversation_context=conversation_context,
                    memories=memories,
                    enable_dynamic_loop=enable_dynamic_loop,
                    dynamic_limits=dynamic_limits,
                )
                return self._finalize(run_id, result)
            except Exception as error:  # noqa: BLE001 - 失败必须落到 Run 状态
                self._fail(run_id, error, output_root=self.output_root)
                raise
            finally:
                self.executor._stop_heartbeat(run_id)

        if wait:
            return work()

        self.executor.submit(run_id, work)
        return None

    def resume(self, run_id: str, *, resume_value: Any = None) -> None:
        """从最后提交的 Checkpoint 继续执行。

        审批决定通过 `Command(resume=...)` 交回被中断的节点。
        """
        record = self.store.get(run_id)
        if record is None:
            raise KeyError(f"未登记的 run：{run_id}")

        # `running` 有两种含义：正在执行，或执行者已经消失（进程被杀）。
        # 只按状态字面量判断会把"崩溃后恢复"这条路径永远挡在门外——
        # 而进程内的执行记录在重启后本来就是空的，is_active 一定为 False。
        stale = self.store.is_stale_running(run_id)

        if (
            record.status not in ("waiting_approval", "interrupted", "failed")
            and not stale
        ):
            raise ValueError(
                f"run {run_id} 当前状态为 {record.status}，不可恢复"
            )

        if self.executor.is_active(run_id):
            raise ValueError(f"run {run_id} 正在执行中，拒绝并发的恢复请求")

        self.store.update_status(run_id, "running")

        # 后端模式取记录里最初保存的值，**不接受**调用方在恢复时另行指定：
        # 否则同一 Run 的前后两段会走不同后端，Checkpoint 里的结果语义就乱了。
        mcp = self._settings_for(record.sql_backend)

        command = Command(resume=resume_value) if resume_value is not None else None

        def work() -> dict[str, Any]:
            try:
                result = self._execute(
                    run_id,
                    data_sources=record.data_sources,
                    analysis_focus=record.analysis_focus,
                    knowledge_sources=[],
                    resume=command,
                    # 沿用该 Run 创建时的审批要求。硬写成 True 会让"崩溃点晚于
                    # 审批门"的 Run 在恢复时又被拦一次，等于凭空多出一道门。
                    require_plan_approval=record.require_plan_approval,
                    require_report_approval=record.require_report_approval,
                    mcp=mcp,
                )
                # 恢复同样要落状态：否则等待重试的审批不会反映到生命周期上，
                # 查询接口会一直显示 running。
                return self._finalize(run_id, result)
            except Exception as error:  # noqa: BLE001
                self._fail(run_id, error, output_root=self.output_root)
                raise
            finally:
                self.executor._stop_heartbeat(run_id)

        self.executor.submit(run_id, work)

    def shutdown(self) -> None:
        self.executor.shutdown(wait=True)
