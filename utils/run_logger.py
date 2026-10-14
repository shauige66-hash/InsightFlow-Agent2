# 提供 InsightFlow 工作流的本地结构化运行日志能力。

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from uuid import uuid4


def _log_dir() -> Path:
    """日志目录跟随运行期 output_root。

    不硬编码相对 `outputs`：那会让日志落到进程当前目录，
    绕过调用方指定的隔离目录（测试与被嵌入运行时都会写错位置）。

    只有"运行期上下文尚未建立"才回退到默认目录——例如工作流启动前的
    `run_started` 事件，或单独调用本模块的场景。
    """
    from core.runtime import RuntimeContextMissing, current_runtime

    try:
        return current_runtime().output_root / "logs"
    except RuntimeContextMissing:
        # 运行期尚未绑定：此时还没有调用方指定的目录，退回默认位置。
        return Path("outputs") / "logs"


def create_run_id() -> str:

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    suffix = uuid4().hex[:8]

    return f"run_{timestamp}_{suffix}"


def log_event(
    run_id: str,
    node: str,
    event: str,
    status: str,
    details: dict[str, Any] | None = None,
    duration_ms: float | None = None
) -> None:

    log_dir = _log_dir()

    log_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    log_path = log_dir / f"{run_id}.jsonl"

    record = {
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "run_id": run_id,
        "node": node,
        "event": event,
        "status": status,
        "details": details or {}
    }

    if duration_ms is not None:
        record["duration_ms"] = round(
            duration_ms,
            2
        )

    with log_path.open(
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            json.dumps(
                record,
                ensure_ascii=False,
                default=str
            )
            + "\n"
        )