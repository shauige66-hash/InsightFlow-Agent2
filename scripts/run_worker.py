"""队列 worker：从 PostgreSQL 抢占 Run 并执行。

用法：

    python scripts/run_worker.py            # 常驻，轮询队列
    python scripts/run_worker.py --once     # 只处理一个任务（部署/测试用）

为什么单独一个进程：API 进程重启时不能带走正在跑的分析。队列把"登记"和"执行"
拆开，Run 的状态与 Checkpoint 都在数据库里，worker 死了任务仍是 claimed，
可以由人显式 requeue，而不是被静默重跑。

依赖 `INSIGHTFLOW_STORAGE=postgres` 与 `INSIGHTFLOW_EXECUTION=queue`。
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services import run_queue  # noqa: E402


def _build_coordinator():
    """worker 与 API 使用同一套协调器，只是执行方式不同（wait=True）。"""
    from api_runs import get_coordinator

    return get_coordinator()


def process_one(coordinator, worker_id: str) -> bool:
    """抢占并执行一个任务；返回是否真的处理了任务。"""
    claimed = run_queue.claim(worker_id)
    if claimed is None:
        return False

    run_id, payload = claimed
    print(f"[worker] 领取 {run_id}（{worker_id}）", flush=True)
    try:
        coordinator.start(
            run_id,
            record_exists=True,
            wait=True,
            **payload,
        )
    except Exception as error:  # noqa: BLE001 - 失败必须记进队列，不能静默
        message = f"{type(error).__name__}: {error}"
        print(f"[worker] {run_id} 失败：{message}", flush=True)
        run_queue.finish(run_id, error=message)
    else:
        run_queue.finish(run_id)
        print(f"[worker] {run_id} 完成", flush=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="InsightFlow 队列 worker")
    parser.add_argument("--once", action="store_true", help="处理一个任务后退出")
    parser.add_argument("--interval", type=float, default=2.0, help="空闲轮询间隔（秒）")
    parser.add_argument("--worker-id", default=None, help="默认 hostname-pid-uuid")
    args = parser.parse_args()

    if (os.getenv("INSIGHTFLOW_EXECUTION") or "inline").lower() != "queue":
        print(
            "INSIGHTFLOW_EXECUTION 不是 queue；worker 不该在这个模式下启动。",
            file=sys.stderr,
        )
        return 2

    try:
        run_queue.ensure_schema()
    except run_queue.QueueUnavailable as error:
        print(f"队列不可用：{error}", file=sys.stderr)
        return 2

    worker_id = args.worker_id or f"{os.getenv('COMPUTERNAME', 'host')}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
    coordinator = _build_coordinator()

    stopping = {"flag": False}

    def handle_signal(signum, frame):  # noqa: ANN001
        stopping["flag"] = True
        print(f"[worker] 收到信号 {signum}，处理完当前任务后退出", flush=True)

    signal.signal(signal.SIGINT, handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handle_signal)

    print(f"[worker] 启动 {worker_id}，间隔 {args.interval}s", flush=True)
    while not stopping["flag"]:
        did_work = process_one(coordinator, worker_id)
        if args.once:
            return 0 if did_work else 0
        if not did_work:
            time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
