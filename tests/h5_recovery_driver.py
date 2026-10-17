"""跨进程恢复驱动：在 H5 的崩溃注入点强杀，再由另一个进程恢复。

每个 phase 都是独立进程，因此"恢复"是真实的重新加载 Checkpoint，而不是
同一进程里的内存状态。
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REPLIES = {
    "planner": {
        "tasks": [
            {
                "task_id": "task_1",
                "description": "统计客户数量",
                "target_tables": ["customers.customers"],
                "expected_output": "客户数",
            }
        ]
    },
    "executor": {"code": "SELECT COUNT(*) AS n FROM customers"},
    "repair": {"code": "SELECT COUNT(*) AS n FROM customers"},
    "refine": {"code": "SELECT COUNT(*) AS n FROM customers"},
    "critic": {"validation_passed": True, "feedback": "ok"},
    "visualizer": {"charts": []},
    "reporter": "# 报告\n\n共 3 位客户。\n",
    "decider": {"action": "continue", "reason": "任务已通过校验"},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["start", "resume", "inspect"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--crash-on")
    args = parser.parse_args()

    import os

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    os.environ["INSIGHTFLOW_STORAGE"] = "local"
    os.environ["INSIGHTFLOW_LOCAL_DB"] = str(workdir / "conversations.sqlite")
    os.environ["INSIGHTFLOW_OWNER"] = "h5-recovery"
    os.environ.pop("INSIGHTFLOW_REDIS_ENABLED", None)
    if args.crash_on:
        os.environ["INSIGHTFLOW_CRASH_AT"] = args.crash_on

    import storage

    storage.reset()
    storage.configure()

    from core.checkpoints import open_checkpointer
    from services.analysis_service import read_run_state, run_analysis
    from tests.scripted_queue import ScriptedQueueFactory

    checkpoint = workdir / "cp.sqlite"

    if args.phase == "inspect":
        with open_checkpointer(checkpoint) as saver:
            state = read_run_state(args.run_id, checkpointer=saver)
        print(json.dumps({"state": state}, ensure_ascii=False))
        return

    factory = ScriptedQueueFactory(args.run_id, {k: [v] for k, v in REPLIES.items()})
    with open_checkpointer(checkpoint) as saver:
        result = run_analysis(
            run_id=args.run_id,
            data_sources=[args.data],
            model_factory=factory.build(),
            output_root=workdir,
            enable_dynamic_loop=True,
            checkpointer=saver,
            resume=None if args.phase == "start" else None,
            continue_from_checkpoint=args.phase == "resume",
        )

    summary = {
        "run_status": result.get("run_status"),
        "analysis_status": result.get("analysis_status"),
        "loop_step": result.get("loop_step"),
        "decisions": len(result.get("decision_history") or []),
        "action": result.get("current_action"),
        "plan_revision": result.get("plan_revision"),
        "calls": factory.calls,
        "llm_calls": (result.get("execution_budget") or {}).get("usage", {}).get("llm_calls"),
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
