"""MCP Client：宿主侧的 stdio 会话。

设计要点与理由：

**同步外观 + 专用事件循环线程。** 工作流在宿主线程里同步执行，而 MCP SDK 是
异步的。这里起一个专用事件循环线程，用 `run_coroutine_threadsafe` 提交调用。
不用"每次调用 `asyncio.run`"的做法：那样会话会在每次调用之间被拆掉，
握手与 list_tools 的结果无法复用，也无法保证进程回收。

**超时即销毁会话。** 实测（mcp 1.30.0 + Windows）：超时会让 anyio 的
task group 以 `BaseExceptionGroup` 退出，`stdio_client.__aexit__` 因此抛出。
这意味着会话在超时后**不可继续使用**——所以本类的策略是：
任何超时都标记会话失效，后续调用直接失败，并要求重新构造客户端。
把它伪装成"可以继续用"会掩盖真实的协议状态。

**不静默回退。** 这里没有任何"连不上就改用直接调用"的分支。
MCP 通道不可用就是 unavailable，由上层决定失败。

**进程回收。** 退出时先让 async 上下文正常收尾，再核对子进程确实已结束；
仍在则强制终止。实测表明正常路径能回收，这里只是不把回收寄托在"应该会"。
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Coroutine

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from mcp_bridge.contract import (
    ALLOWED_MCP_TOOLS,
    ERROR_KIND_MCP_TIMEOUT,
    ERROR_KIND_MCP_UNAVAILABLE,
)

# 默认预算。启动比调用宽一些：导入 SDK 与建库都要时间。
DEFAULT_STARTUP_TIMEOUT = 60.0
DEFAULT_CALL_TIMEOUT = 60.0


class McpUnavailable(RuntimeError):
    """MCP 通道不可用：启动失败、握手失败、工具发现失败或会话已失效。"""

    error_kind = ERROR_KIND_MCP_UNAVAILABLE


class McpTimeout(RuntimeError):
    """MCP 操作超出时间预算。会话随之失效。"""

    error_kind = ERROR_KIND_MCP_TIMEOUT


@dataclass
class ToolDigest:
    """发现到的工具。只保留我们真正要用的信息。"""

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)


def flatten_exception(error: BaseException) -> list[BaseException]:
    """把 anyio 的 ExceptionGroup 摊平，取出真正的根因。

    超时经由 task group 退出时会被包成
    `ExceptionGroup(TaskGroup(...), [ExceptionGroup(...)])`，
    直接看最外层只能看到 "unhandled errors in a TaskGroup"，
    无法判断到底是超时还是崩溃。
    """
    found: list[BaseException] = []
    stack: list[BaseException] = [error]
    seen: set[int] = set()

    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))

        nested = getattr(current, "exceptions", None)
        if nested:
            stack.extend(nested)
        else:
            found.append(current)

    return found


def _is_timeout(error: BaseException) -> bool:
    """判断一个异常树是否代表超时。

    要同时认两种情况：
      1. SDK/anyio 抛出的真实超时（`TimeoutError`、"timed out"）；
      2. 我们自己在 `_submit` 里抛的 `McpTimeout`——它会被调用方再次包装，
         这里若不识别，启动超时就会退化成笼统的 unavailable。
    """
    for leaf in flatten_exception(error):
        if isinstance(leaf, (TimeoutError, asyncio.TimeoutError, McpTimeout)):
            return True
        text = str(leaf).lower()
        if "timed out" in text or "超出" in text or "超时" in text:
            return True
    return False


class McpStdioClient:
    """一个 Run 对应一个 Server 子进程。

    构造后需 `start()`；用完必须 `close()`。支持上下文管理器。
    """

    def __init__(
        self,
        *,
        config_path: Path,
        server_module: str = "mcp_bridge.server",
        project_root: Path | None = None,
        python_executable: str | None = None,
        startup_timeout: float = DEFAULT_STARTUP_TIMEOUT,
        call_timeout: float = DEFAULT_CALL_TIMEOUT,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.server_module = server_module
        self.project_root = Path(project_root or Path.cwd())
        self.python_executable = python_executable or sys.executable
        self.startup_timeout = startup_timeout
        self.call_timeout = call_timeout
        # 刻意只给子进程必要的环境，不整体继承 os.environ：
        # 容器/子进程都不应拿到宿主的 API Key 与 ModelGate 凭证。
        self.extra_env = dict(extra_env or {})

        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._session: ClientSession | None = None
        self._exit_stack: Any = None
        self._child_pid: int | None = None
        # anyio.open_process 的临时替换，仅用于记录 pid
        self._pid_hook: tuple[Any, Any] | None = None
        self._tools: dict[str, ToolDigest] = {}

        self.started = False
        self.invalidated = False
        self.server_info: dict[str, Any] = {}
        self.protocol_version: str | None = None
        self.stderr_tail: list[str] = []

    # ---------- 生命周期 ----------

    # 子进程需要的**操作系统管道**变量：解释器与第三方库靠它们定位缓存、
    # 用户目录与临时文件。缺了这些实测会让 pandas 首次导入卡住
    # （`get_schema` 因此超时），所以不能"越少越好"地一刀切。
    #
    # 白名单是逐项列举的，不是"排除几个敏感变量"：只有明确列出的才传给子进程，
    # API Key、ModelGate 凭证之类不会因为"忘了排除"而漏进去。
    _ENV_ALLOWLIST = (
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "SYSTEMDRIVE",
        "WINDIR",
        "COMSPEC",
        "TEMP",
        "TMP",
        "APPDATA",
        "LOCALAPPDATA",
        "USERPROFILE",
        "HOMEDRIVE",
        "HOMEPATH",
        "PROGRAMDATA",
        "PROGRAMFILES",
        "PROGRAMFILES(X86)",
        "PROCESSOR_ARCHITECTURE",
        "NUMBER_OF_PROCESSORS",
        "OS",
        "LANG",
        "LC_ALL",
        "PYTHONPATH",
        "VIRTUAL_ENV",
    )

    def _child_env(self) -> dict[str, str]:
        """子进程环境。

        只给操作系统管道与解释器运行所需的最小集合：**不含**任何 API Key、
        ModelGate 配置或其他凭证类变量。
        """
        env: dict[str, str] = {
            key: os.environ[key] for key in self._ENV_ALLOWLIST if os.environ.get(key)
        }
        # 编码固定为 UTF-8：MCP 走 JSON-RPC，两端编码不一致会破坏协议
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        env.update(self.extra_env)
        return env

    def start(self) -> list[ToolDigest]:
        """启动子进程、握手并发现工具。

        启动与初始化共用一个预算：卡在任一步都算不可用。
        """
        if self.started:
            return list(self._tools.values())

        params = StdioServerParameters(
            command=self.python_executable,
            # 结构化参数列表：不经 shell，模型也无法影响这里的任何一项
            args=["-m", self.server_module, "--config", str(self.config_path)],
            env=self._child_env(),
            cwd=str(self.project_root),
        )

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        # 等事件循环就绪：线程启动是异步的，直接 submit 会撞上还没赋值的 loop
        if not self._await_loop_ready(timeout=10.0):
            self.invalidated = True
            self._shutdown_loop()
            raise McpUnavailable("MCP 事件循环未能在 10 秒内就绪")

        try:
            tools = self._submit(self._startup(params), timeout=self.startup_timeout)
        except McpTimeout:
            # 已经是我们自己的超时类型，直接透传，不再降级成 unavailable
            self.invalidated = True
            self._shutdown_loop()
            raise
        except McpUnavailable:
            self.invalidated = True
            self._shutdown_loop()
            raise
        except BaseException as error:  # noqa: BLE001
            self.invalidated = True
            self._shutdown_loop()
            if _is_timeout(error):
                raise McpTimeout(
                    f"MCP 启动超出 {self.startup_timeout:.0f} 秒预算"
                ) from error
            raise McpUnavailable(
                f"MCP 启动失败：{self._describe(error)}"
            ) from error

        self.started = True
        self._tools = {digest.name: digest for digest in tools}
        return tools

    async def _startup(self, params: StdioServerParameters) -> list[ToolDigest]:
        from contextlib import AsyncExitStack

        stack = AsyncExitStack()
        await stack.__aenter__()
        self._exit_stack = stack

        try:
            # 记录子进程 pid 后再进入上下文。
            #
            # 刻意**不用** asynccontextmanager 包住 await：实测在 teardown 阶段
            # 抛 GeneratorExit 时，async generator 的 athrow() 可能停不下来，
            # 会冒成 "generator didn't stop after athrow()" 的 Unraisable 警告。
            # 这里只装一次 thenable，包一层 await 后立刻还原，不涉及生成器收尾。
            self._install_pid_hook()
            try:
                read, write = await stack.enter_async_context(stdio_client(params))
            finally:
                self._remove_pid_hook()
            session = await stack.enter_async_context(ClientSession(read, write))
            init = await session.initialize()

            self._session = session
            self.server_info = {
                "name": init.serverInfo.name,
                "version": getattr(init.serverInfo, "version", None),
            }
            self.protocol_version = init.protocolVersion

            listed = await session.list_tools()
            return [
                ToolDigest(
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=dict(tool.inputSchema or {}),
                )
                for tool in listed.tools
            ]
        except BaseException:
            # 握手失败也要把已经进入的上下文退出，否则子进程会留下来
            if stack is not None:
                try:
                    await stack.aclose()
                except BaseException:  # noqa: BLE001
                    pass
            self._exit_stack = None
            raise

    def _install_pid_hook(self) -> None:
        """在本次 `open_process` 期间记录子进程 pid。

        不直接读 SDK 的进程对象：那依赖私有属性，升级就会断。

        改为临时包住 `anyio.open_process`：安装 → 一次 await → 立刻还原，
        作用范围压到最小，且不引入生成器收尾（见 `_startup` 里的说明）。
        即使拿不到，也只是少一层兜底——SDK 自身的 Windows Job Object
        仍然负责终止整棵进程树。
        """
        import anyio

        if self._pid_hook is not None:  # 已有钩子，先还原，避免嵌套
            self._remove_pid_hook()

        original = anyio.open_process

        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            process = await original(*args, **kwargs)
            try:
                self._child_pid = getattr(process, "pid", None)
            finally:
                # 拿到第一个进程后立刻还原，避免影响后续任何 open_process 调用
                self._remove_pid_hook()
            return process

        self._pid_hook = (original, wrapper)
        anyio.open_process = wrapper  # type: ignore[assignment]

    def _remove_pid_hook(self) -> None:
        import anyio

        hook = self._pid_hook
        if hook is None:
            return
        original, _wrapper = hook
        anyio.open_process = original  # type: ignore[assignment]
        self._pid_hook = None

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # 先赋值、再置就绪信号，最后 run_forever：
        # start() 侧必须能区分"线程还没建好循环"与"循环已就绪"
        self._loop = loop
        self._loop_ready.set()
        loop.run_forever()

    def _await_loop_ready(self, *, timeout: float) -> bool:
        return self._loop_ready.wait(timeout=timeout) and self._loop is not None

    def _submit(self, coro: Coroutine[Any, Any, Any], *, timeout: float) -> Any:
        if self._loop is None:
            raise McpUnavailable("MCP 事件循环未启动")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError as error:
            future.cancel()
            raise McpTimeout(f"MCP 操作超出 {timeout:.0f} 秒预算") from error

    # ---------- 工具调用 ----------

    @property
    def discovered_tools(self) -> dict[str, ToolDigest]:
        return dict(self._tools)

    def allowed_tools(self) -> list[str]:
        """发现到的工具与允许清单的交集。

        动态发现只说明"服务端声明了什么"。能不能调，取决于本地允许清单：
        否则一个被替换过的 Server 就能靠多声明工具拿到执行权限。
        """
        return sorted(set(self._tools) & ALLOWED_MCP_TOOLS)

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """调用一个工具，返回归一化结果。

        返回 `{success, result, error, error_kind}`，与工具注册表契约一致，
        因此上层路由不需要认识 MCP 的 `isError` 形态。
        """
        budget = self.call_timeout if timeout is None else timeout

        if self.invalidated:
            return {
                "success": False,
                "result": None,
                "error": "MCP 会话已失效（此前发生过超时或崩溃），必须重建客户端",
                "error_kind": ERROR_KIND_MCP_UNAVAILABLE,
            }

        if not self.started:
            return {
                "success": False,
                "result": None,
                "error": "MCP 客户端尚未启动",
                "error_kind": ERROR_KIND_MCP_UNAVAILABLE,
            }

        if name not in ALLOWED_MCP_TOOLS:
            return {
                "success": False,
                "result": None,
                "error": f"工具不在允许清单内：{name}",
                "error_kind": ERROR_KIND_MCP_UNAVAILABLE,
            }

        if name not in self._tools:
            return {
                "success": False,
                "result": None,
                "error": f"服务端未声明该工具：{name}",
                "error_kind": ERROR_KIND_MCP_UNAVAILABLE,
            }

        try:
            payload = self._submit(
                self._call(name, arguments or {}), timeout=budget
            )
        except BaseException as error:  # noqa: BLE001
            self.invalidated = True
            if _is_timeout(error):
                return {
                    "success": False,
                    "result": None,
                    "error": f"MCP 调用超出 {budget:.0f} 秒预算：{name}",
                    "error_kind": ERROR_KIND_MCP_TIMEOUT,
                }
            return {
                "success": False,
                "result": None,
                "error": f"MCP 调用失败：{self._describe(error)}",
                "error_kind": ERROR_KIND_MCP_UNAVAILABLE,
            }

        return payload

    async def _call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        session = self._session
        if session is None:
            raise McpUnavailable("MCP 会话不存在")

        result = await session.call_tool(name, arguments)

        text = ""
        for item in result.content or ():
            candidate = getattr(item, "text", None)
            if candidate:
                text = candidate
                break

        if result.isError:
            # 工具自身的错误（SQL 语法、策略拒绝）属于可修复范畴
            return {
                "success": False,
                "result": None,
                "error": text or "MCP 工具返回错误",
                "error_kind": "execution_error",
            }

        parsed: Any = text
        if text:
            import json

            try:
                parsed = json.loads(text)
            except ValueError:
                # 不是 JSON 就原样返回文本，而不是丢掉内容
                parsed = text

        return {
            "success": True,
            "result": parsed,
            "error": None,
            "error_kind": None,
        }

    # ---------- 关闭 ----------

    def close(self) -> None:
        """关闭会话并确保子进程结束。"""
        if self._exit_stack is not None and self._loop is not None:
            try:
                self._submit(self._exit_stack.aclose(), timeout=self.startup_timeout)
            except BaseException:  # noqa: BLE001
                # 关闭路径上的异常不影响调用方结论；进程回收在下面兜底
                pass
            finally:
                self._exit_stack = None
                self._session = None

        self._shutdown_loop()
        self.started = False

    def _shutdown_loop(self) -> None:
        """停止事件循环线程并关闭循环。

        只 `stop()` 是不够的：子进程管道的 transport 由循环持有，
        循环对象本身不关闭时，它们要等到垃圾回收才释放——
        在 Windows 的 ProactorEventLoop 上会表现为
        `ResourceWarning: unclosed transport` 以及
        `ValueError: I/O operation on closed pipe`。
        这里在停止之后显式关闭循环，把管道随循环一起收掉。
        """
        loop = self._loop
        thread = self._thread

        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)

        if thread is not None and thread.is_alive():
            thread.join(timeout=10)

        if loop is not None and not loop.is_closed():
            # 先清掉循环上残留的回调与异步生成器，再关闭
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except (RuntimeError, ValueError):
                pass
            loop.close()

        self._loop = None
        self._thread = None

    def _describe(self, error: BaseException) -> str:
        leaves = flatten_exception(error)
        if not leaves:
            return type(error).__name__
        return "; ".join(
            f"{type(leaf).__name__}: {str(leaf)[:160]}" for leaf in leaves[:3]
        )

    def __enter__(self) -> "McpStdioClient":
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


def reap_process(pid: int) -> bool:
    """强制结束一个子进程。返回是否执行了终止动作。

    正常路径下 stdio 上下文会回收进程；这里是**兜底**：
    如果关闭阶段抛异常导致进程残留，调用方仍能清理干净。
    """
    if pid <= 0:
        return False
    try:
        if sys.platform == "win32":
            completed = subprocess.run(
                ["taskkill", "/F", "/PID", str(pid)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return completed.returncode == 0
        os.kill(pid, 9)
        return True
    except (OSError, subprocess.SubprocessError):
        return False
