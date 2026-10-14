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
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from core.llm import ModelFactory
from ingestion.resources import ResourceResolver


class RuntimeContextMissing(RuntimeError):
    """节点在缺少运行期上下文时被调用。"""


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

    def output_dir(self) -> Path:
        return self.output_root / self.run_id


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
