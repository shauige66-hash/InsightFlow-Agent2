"""模型调用的可核对账本。

为什么需要独立于 Checkpoint 的记录：Checkpoint 只在**节点执行结束**时提交。
进程若在"请求已发出、节点尚未返回"之间被杀，Checkpoint 里没有任何痕迹，
恢复时无法从 state 判断这次调用是否已经发生过——也就无法判断上游是否已经计费。

账本是**追加写**的文件，在发起请求之前先落一条 `started`，请求返回后补一条
`completed`。因为它在调用前就已经落盘，所以进程被强杀也不会丢失
"这次调用已经发出"这个事实。

它提供的能力是**事后核对**，不是自动去重：
- 恢复后可以查出某个逻辑调用是否可能被重复发出；
- 但 `started` 无法区分"上游已处理"与"上游未收到"，因此不能据此断言恰好一次。

因此本模块只做记录与查询，不做"自动判定可以安全重发"这种它无法支撑的结论。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

PHASE_STARTED = "started"
PHASE_COMPLETED = "completed"


def ledger_path(run_id: str, root: Path | str = "outputs") -> Path:
    return Path(root) / "llm_ledger" / f"{run_id}.jsonl"


def append(entry: dict[str, Any], *, root: Path | str = "outputs") -> None:
    """追加一条记录并立即 fsync。

    必须在发出请求**之前**调用：只有先落盘，进程被强杀后才有"这次调用已发出"
    的证据。fsync 是必要的——仅 flush 只写到操作系统缓冲，断电或强杀仍可能丢。
    """
    run_id = entry["run_id"]
    line = json.dumps(entry, ensure_ascii=False, sort_keys=True)

    path = ledger_path(run_id, root)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def record_started(
    *,
    run_id: str,
    operation_id: str,
    node: str,
    attempt: int,
    root: Path | str = "outputs",
) -> None:
    append(
        {
            "phase": PHASE_STARTED,
            "run_id": run_id,
            "operation_id": operation_id,
            "node": node,
            "attempt": attempt,
            "at": time.time(),
        },
        root=root,
    )


def record_completed(
    *,
    run_id: str,
    operation_id: str,
    node: str,
    attempt: int,
    root: Path | str = "outputs",
) -> None:
    append(
        {
            "phase": PHASE_COMPLETED,
            "run_id": run_id,
            "operation_id": operation_id,
            "node": node,
            "attempt": attempt,
            "at": time.time(),
        },
        root=root,
    )


def read_entries(run_id: str, root: Path | str = "outputs") -> list[dict[str, Any]]:
    path = ledger_path(run_id, root)
    if not path.exists():
        return []

    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except ValueError:
            # 崩溃可能留下半行；跳过它，不让核对脚本整体失败
            continue
    return entries


def summarize(run_id: str, root: Path | str = "outputs") -> dict[str, Any]:
    """汇总一个 Run 的调用情况。

    `unconfirmed` 是"发出了但没有完成记录"的逻辑调用：
    可能上游已处理，也可能没收到，本地无法判定。
    """
    entries = read_entries(run_id, root)

    started: dict[str, int] = {}
    completed: dict[str, int] = {}
    per_node: dict[str, dict[str, int]] = {}

    for entry in entries:
        op = entry.get("operation_id", "")
        node = entry.get("node", "")
        bucket = per_node.setdefault(node, {"started": 0, "completed": 0})

        if entry.get("phase") == PHASE_STARTED:
            started[op] = started.get(op, 0) + 1
            bucket["started"] += 1
        else:
            completed[op] = completed.get(op, 0) + 1
            bucket["completed"] += 1

    unconfirmed = sorted(op for op in started if op not in completed)
    replayed = sorted(op for op, count in started.items() if count > 1)

    return {
        "run_id": run_id,
        "total_started": sum(started.values()),
        "total_completed": sum(completed.values()),
        "distinct_operations": len(started),
        "per_node": per_node,
        "unconfirmed_operations": unconfirmed,
        "replayed_operations": replayed,
    }
