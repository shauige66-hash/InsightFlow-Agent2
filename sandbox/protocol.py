"""宿主与沙箱之间的数据协议。

两条原则：

1. **不用 pickle 传数据**。宿主与沙箱是不同进程、可能不同补丁版本，pickle 既不
   保证跨版本兼容，也把反序列化风险交给沙箱去承担。这里一律用带类型标注的
   JSON 记录。
2. **结果先过 Schema 再进 state**。沙箱返回的内容在进入 LangGraph State 之前
   必须通过形状、类型与大小校验，避免把畸形结构扩散到下游节点。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

# 结果契约：与 tool_registry 的 {success, result, error} 保持一致
RESULT_KEYS = ("success", "result", "error")

# DataFrame 往返的类型标注：只有需要特殊还原的列才记录
COLUMN_KIND_DATETIME = "datetime"
COLUMN_KIND_BOOL = "bool"
COLUMN_KIND_NUMERIC = "numeric"


class SandboxErrorKind:
    """沙箱失败的分类。

    区分这些类型不只是为了日志：`unavailable` 属于环境问题，修复循环重试没有
    意义；`execution_error` 是模型代码的问题，应当进入 Repair 路由。
    """

    UNAVAILABLE = "unavailable"
    TIMEOUT = "timeout"
    OOM = "oom"
    EXECUTION_ERROR = "execution_error"
    INVALID_RESULT = "invalid_result"
    INTERNAL = "internal"


# 环境/资源类失败：重发同样的代码不会好转，不应进入 LLM 修复循环。
#
# 刻意**不含** `execution_error` 与 `invalid_result`：前者是模型代码写错了，
# 后者是返回结构不合法——两者都可能通过重新生成代码修好。
ENVIRONMENT_ERROR_KINDS = frozenset(
    {
        SandboxErrorKind.UNAVAILABLE,
        SandboxErrorKind.TIMEOUT,
        SandboxErrorKind.OOM,
    }
)


@dataclass(frozen=True)
class SandboxLimits:
    """一次沙箱调用的资源与体积上限。"""

    timeout_seconds: float = 60.0
    memory_mb: int = 1024
    cpus: float = 1.0
    pids_limit: int = 128
    tmpfs_mb: int = 256
    max_result_bytes: int = 1024 * 1024
    max_result_rows: int = 100_000
    max_output_file_bytes: int = 16 * 1024 * 1024
    # 并发获取容器配额的最长等待；超过即 fail-closed，不无限排队
    concurrency_wait_seconds: float = 120.0

    def to_job_dict(self) -> dict[str, Any]:
        """传给容器内执行入口的限制子集。

        只传容器自己能执行的部分；CPU/内存/进程数由 docker run 在宿主侧强制，
        容器不需要（也无法）知道。
        """
        return {
            "max_result_bytes": self.max_result_bytes,
            "max_result_rows": self.max_result_rows,
            "max_output_file_bytes": self.max_output_file_bytes,
        }


@dataclass
class SandboxJob:
    """一次沙箱调用的输入。

    `mode` 决定容器内注入哪些符号：
    - `python`：`dfs`、`pd`，单表时额外 `df`
    - `plot`：`dfs`、`plt`、`analysis_result`、`result_df`、`output_path`，单表时额外 `df`
    """

    mode: Literal["python", "plot"]
    table_names: list[str]
    code: str
    limits: SandboxLimits = field(default_factory=SandboxLimits)
    # plot 专用
    analysis_result: Any | None = None
    output_filename: str | None = None
    # 供容器内 diagnostics 使用，不参与执行逻辑
    op_label: str = ""

    def to_dict(self, dfs_paths: dict[str, str]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "mode": self.mode,
            "table_names": list(self.table_names),
            "dfs": dfs_paths,
            # 代码不内联进 job.json，而是单独落成受控文件；
            # 容器从只读挂载读取，避免把大段代码塞进 JSON 结构里。
            "code_path": "code.py",
            "limits": self.limits.to_job_dict(),
            "op_label": self.op_label,
        }
        if self.mode == "plot":
            payload["analysis_result"] = self.analysis_result
            payload["output_filename"] = self.output_filename
        return payload


@dataclass
class SandboxOutcome:
    """宿主侧归一化后的沙箱结果。"""

    success: bool
    result: Any
    error: str | None
    error_kind: str | None = None
    duration_ms: int = 0
    truncated: bool = False
    container_id: str | None = None
    # 容器侧诊断（例如缺字形计数、产物字节数），不进 state，仅供调用方与测试使用
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_tool_result(self) -> dict[str, Any]:
        """转成 tool_registry 的固定契约。

        `error_kind` 一并带出：路由需要据此区分"模型代码写错了"（可修复）
        与"沙箱环境不可用"（重试无意义）。用字符串前缀表达类型会让下游
        不得不去解析错误文案，这里保持结构化。
        """
        return {
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "error_kind": self.error_kind,
        }


def classify_exception(error: BaseException) -> str:
    """把宿主侧异常映射成错误分类。

    沙箱执行器在环境问题上抛异常，而不是返回结果字典（那些问题无法从
    "执行一次"里恢复）。这一层把它翻译成分类，供工具填充 `error_kind`，
    使路由不必解析字符串。

    接线类问题（例如没有运行期上下文）同样归入 `unavailable`：
    它们是调用方的问题，重新生成模型代码不会好转。
    """
    from sandbox.executor import (
        SandboxOutputRejected,
        SandboxUnavailable,
    )

    if isinstance(error, SandboxUnavailable):
        return SandboxErrorKind.UNAVAILABLE
    if isinstance(error, SandboxOutputRejected):
        return SandboxErrorKind.INVALID_RESULT
    if isinstance(error, RuntimeError) and "runtime context" in str(error):
        return SandboxErrorKind.UNAVAILABLE

    return SandboxErrorKind.EXECUTION_ERROR


def encode_dataframe(frame: Any) -> dict[str, Any]:
    """把 DataFrame 编码成可跨进程序列化的 JSON 结构。

    只记录需要特殊还原的列类型（datetime / bool / numeric），其余交给
    pandas 的默认推断。不做完整 dtype 保真——那需要 pickle，而 pickle
    不满足跨进程与安全性要求。
    """
    import pandas as pd

    columns = [str(column) for column in frame.columns]
    kinds: dict[str, str] = {}

    for column in frame.columns:
        dtype = frame[column].dtype
        key = str(column)
        if pd.api.types.is_datetime64_any_dtype(dtype):
            kinds[key] = COLUMN_KIND_DATETIME
        elif pd.api.types.is_bool_dtype(dtype):
            kinds[key] = COLUMN_KIND_BOOL
        elif pd.api.types.is_numeric_dtype(dtype):
            kinds[key] = COLUMN_KIND_NUMERIC

    records = json.loads(frame.to_json(orient="records", date_format="iso"))

    return {
        "__kind__": "dataframe",
        "columns": columns,
        "column_kinds": kinds,
        "records": records,
    }


def decode_dataframe(payload: dict[str, Any]) -> Any:
    """还原 encode_dataframe 编码的 DataFrame。"""
    import pandas as pd

    records = payload.get("records") or []
    columns = payload.get("columns") or []
    kinds = payload.get("column_kinds") or {}

    frame = pd.DataFrame(records)

    # 空结果也要保留列名，否则下游看到的是"无列"而不是"无行"
    if frame.empty and columns:
        frame = pd.DataFrame(columns=columns)

    if not columns:
        return frame

    # 保证列顺序与原始一致
    ordered = [column for column in columns if column in frame.columns]
    extra = [column for column in frame.columns if column not in ordered]
    frame = frame[ordered + extra]

    for column, kind in kinds.items():
        if column not in frame.columns:
            continue
        if kind == COLUMN_KIND_DATETIME:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
        elif kind == COLUMN_KIND_BOOL:
            frame[column] = frame[column].astype("boolean")
        elif kind == COLUMN_KIND_NUMERIC:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


def write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8"
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
