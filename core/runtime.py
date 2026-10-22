"""运行期依赖注入。

模型客户端、资源解析器、输出目录这些**不能跨进程恢复**的东西不进 state：
state 会写进 Checkpoint，而 resolver 的缓存、连接池与 DataFrame 在恢复后的
新进程里毫无意义，还会把 checkpoint 撑大。

用 ContextVar 传递运行期上下文，而不是模块级字典：

- 节点在 LangGraph 的调用上下文中执行，能读到 invoke 之前设置的值（已实测）；
- `contextvars` 按上下文隔离，两个并发 Run 各自拿到自己的运行时，不会串值；
- 节点内对 ContextVar 的修改不会泄漏回外层。

`config["configurable"]["thread_id"]` 仍然单独传递，它决定 Checkpoint 归属。
刻意不使用 `config: RunnableConfig` 形态的节点参数：LangGraph 会把
RunnableConfig 的字段名（configurable / callbacks / tags …）当成图通道，
其中 `configurable` 是保留名，会直接导致 compile 失败。
"""

from __future__ import annotations

import contextvars
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

from core.llm import ModelFactory
from ingestion.resources import ResourceResolver

# 私有配置根目录相对 output_root 的后缀。
# 用后缀而不是固定子目录名：与 output_root 平级，任何 run_id 都无法与它重叠。
PRIVATE_ROOT_SUFFIX = ".mcp_private"


class RuntimeContextMissing(RuntimeError):
    """节点在缺少运行期上下文时被调用。"""


class PrivatePathConflict(ValueError):
    """私有配置路径落在了会被挂载进沙箱的目录树之内。"""


class SqlBackend(str, Enum):
    """SQL 工具的执行后端。

    只有两种取值，且**必须显式指定**：没有"自动挑一个"的默认值，
    因为那会在 MCP 通道不通时把失败悄悄变成另一种执行路径。

    继承 `str` 是为了让 JSON、SQLite 与既有字符串比较都能直接用；
    API 层、RunStore 与 Coordinator 共用这一个定义，避免三处各写一份常量。
    """

    DIRECT = "direct"
    MCP = "mcp"


def _normalized(path: Path) -> str:
    """用于目录归属比较的规范化路径。

    Windows 上路径大小写不敏感，`is_relative_to` 却是按字符串比较的，
    因此必须统一大小写；`resolve()` 负责把符号链接与 `..` 展开，
    避免用别名绕过归属检查。
    """
    return os.path.normcase(str(Path(path).resolve()))


def _is_inside(candidate: Path, parent: Path) -> bool:
    """candidate 是否位于 parent 之内（含两者相同）。"""
    normalized_candidate = _normalized(candidate)
    normalized_parent = _normalized(parent)

    if normalized_candidate == normalized_parent:
        return True
    return normalized_candidate.startswith(
        normalized_parent.rstrip("\\/") + os.sep
    )


def _reject_if_inside(candidate: Path, parent: Path, *, what: str) -> None:
    if _is_inside(candidate, parent):
        raise PrivatePathConflict(
            f"{what}不能位于 {parent} 之内（当前为 {candidate}）："
            "该目录树会被只读挂载进执行沙箱，"
            "且 run_id 与子目录名重叠时会造成跨 Run 配置泄漏。"
        )


@dataclass
class McpSettings:
    """MCP 后端配置。属于**可信运行期配置**，不由模型控制。

    `mode` 只有 `direct` 与 `mcp` 两种，且必须显式指定：
    没有"自动挑一个"的默认值，因为那会在通道不通时把失败悄悄变成另一种执行路径。

    这里只放宿主自己决定的项（模式、超时、Server 模块）。**不放**任何来自
    请求的东西：Server 启动命令、环境变量、私有目录、宿主路径都由宿主固定，
    调用方（更不用说模型）无从指定。
    """

    mode: SqlBackend | str = SqlBackend.DIRECT
    # 用于定位 Server 配置与 workspace 边界；不指定时由 RunRuntime.output_root 推导
    workspace_root: Path | None = None
    # 私有配置根目录；不指定时为 `<output_root>.mcp_private`（在任何 Run 目录之外）
    private_root: Path | None = None
    startup_timeout: float = 60.0
    call_timeout: float = 60.0
    server_module: str = "mcp_bridge.server"

    def __post_init__(self) -> None:
        # 接受字符串是为了让从 SQLite / JSON 读回来的值直接可用；
        # 统一收敛成枚举，后面的比较就不会再受大小写或类型影响。
        try:
            self.mode = SqlBackend(self.mode)
        except ValueError as error:
            raise ValueError(f"不支持的 SQL 后端模式：{self.mode}") from error

    @property
    def is_mcp(self) -> bool:
        return self.mode == SqlBackend.MCP


@dataclass
class RunRuntime:
    """一次运行所需的非持久化依赖。"""

    run_id: str
    model_factory: ModelFactory
    resolver: ResourceResolver
    output_root: Path = field(default_factory=lambda: Path("outputs"))

    # 恢复时由调用方从账本比对得出"可能已发出但未登记结果"的模型调用。
    # 它不能存进 Checkpoint（那是上次进程的事实），每次运行前重新计算，
    # 因此放在运行期依赖里而不是 state。
    unconfirmed_llm_calls: list[str] = field(default_factory=list)

    # MCP 后端配置。Client / Session / 子进程句柄不放在这里，
    # 由每次工具调用按需创建并立即回收——会话不能跨越 Checkpoint 存活。
    mcp: McpSettings = field(default_factory=McpSettings)
    assert_lease: Any = None

    def output_dir(self) -> Path:
        return self.output_root / self.run_id

    def workspace_root(self) -> Path:
        """Server 允许触达的目录边界。"""
        if self.mcp.workspace_root is not None:
            return Path(self.mcp.workspace_root)
        return Path(self.output_root)

    def mcp_private_root(self) -> Path:
        """MCP 私有配置的根目录。

        刻意**不放在** `<output_root>` 之内，理由有两层：

        1. P2-A 的 `DockerSandboxExecutor` 把 `<output_root>/<run_id>` 只读挂载为
           `/history`。放在 output_root 里就等于把 Server 配置交给沙箱里的模型代码
           （实测用 pandas 就能读到宿主绝对路径）。
        2. 更要紧的是**目录重叠**：`run_id` 由调用方指定，而 `api_runs` 的
           `RUN_ID_PATTERN` 合法接受 `mcp_private` 这个名字。一旦某个 Run 的 id
           就叫 `mcp_private`，它的 run 目录就**等于**私有根目录，
           `/history` 会列出**所有** Run 的私有配置——这是实测可复现的跨 Run 泄漏。

        因此默认取 `<output_root>.mcp_private`：与 output_root **平级**，
        既不在 output_root 之内，也不在任何 Run 的挂载范围内。
        """
        if self.mcp.private_root is not None:
            root = Path(self.mcp.private_root)
        else:
            root = Path(str(self.output_root) + PRIVATE_ROOT_SUFFIX)

        # 显式配置也必须守规矩：落到 output_root 之内就拒绝。
        # 不这样做的话，`private_root=<output_root>/mcp_private` 这类写法会
        # 把上面第 2 条的重叠问题原样带回来。
        _reject_if_inside(
            root,
            Path(self.output_root),
            what="MCP 私有配置根目录",
        )
        return root

    def mcp_config_path(self) -> Path:
        return self.mcp_private_root() / self.run_id / "authorizations.json"


_current_runtime: contextvars.ContextVar[RunRuntime | None] = contextvars.ContextVar(
    "insightflow_runtime", default=None
)


@contextmanager
def use_runtime(runtime: RunRuntime) -> Iterator[RunRuntime]:
    """在当前上下文中绑定运行期依赖，退出时恢复原值。"""
    token = _current_runtime.set(runtime)
    try:
        yield runtime
    finally:
        _current_runtime.reset(token)


def current_runtime() -> RunRuntime:
    runtime = _current_runtime.get()
    if runtime is None:
        raise RuntimeContextMissing(
            "节点缺少运行期上下文；请通过 use_runtime() 包裹 graph.invoke"
        )
    return runtime


def build_thread_config(
    thread_id: str,
    *,
    recursion_limit: int = 100,
) -> dict[str, Any]:
    """构造传给 `graph.invoke` 的 config。

    thread_id 决定 Checkpoint 归属：同一 run_id 必须使用同一个 thread_id，
    否则恢复会读到另一个线程的状态。
    """
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": recursion_limit,
    }


def thread_id_from_config(config: Any) -> str | None:
    if isinstance(config, dict):
        return (config.get("configurable") or {}).get("thread_id")
    return None
