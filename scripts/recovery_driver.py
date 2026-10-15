"""跨进程恢复的测试驱动器。

被 tests/test_cross_process_recovery.py 以真实子进程方式启动：
每个 phase 都是一次独立的 Python 进程，因此"重启恢复"不是在同一进程里
换一个 Graph 对象，而是真的重新加载 Checkpointer 与工作流。

子命令：
    start     启动一次运行并跑到中断/完成
    hard-kill 从 checkpoint 已提交的位置强制终止，用于模拟崩溃
    resume    从最后提交的 Checkpoint 续跑
    approve   提交审批决定并继续
    inspect   打印持久化状态（不推进执行）

所有产物、Checkpoint、日志都写到 --workdir 下，与项目目录隔离。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.callbacks import CallbackManagerForLLMRun  # noqa: E402
from langchain_core.language_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, BaseMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from core.config import Settings  # noqa: E402
from core.llm import CallContext, ModelFactory  # noqa: E402
from ingestion.resources import ResourceResolver  # noqa: E402
from services.analysis_service import read_run_state, run_analysis  # noqa: E402

# 单表场景：一个任务、一张图
# 必须使用工具注入的 plt、output_path 与 analysis_result：
# 执行作用域不提供 matplotlib 模块名（会命中 forbidden pattern），
# 上一步执行结果的名字是 analysis_result，且工具会校验图片确已生成。
_PLOT_CODE = (
    "fig = plt.figure()\n"
    "plt.bar(['all'], [analysis_result])\n"
    "plt.title('customers')\n"
    "fig.savefig(output_path)\n"
    "result = output_path\n"
)

REPLIES: dict[str, Any] = {
    "planner": {
        "tasks": [
            {
                "task_id": "task_1",
                "description": "统计客户总数",
                "target_tables": ["customers"],
                "expected_output": "客户总数",
            }
        ]
    },
    "executor": {"code": "result = len(df)"},
    "repair": {"code": "result = len(df)"},
    "refine": {"code": "result = len(df)"},
    "critic": {"validation_passed": True, "feedback": "ok"},    # 一张图，用于验证绘图链路。字段需与 VisualizationTask 一致（含 expected_output）
    "visualizer": {
        "charts": [
            {
                "chart_id": "chart_1",
                "source_task_id": "task_1",
                "chart_type": "bar",
                "title": "客户数",
                "description": "客户总数",
                "target_tables": ["customers"],
                "expected_output": "客户数柱状图",
            }
        ]
    },
    "plotter": {"code": _PLOT_CODE},
    "plot_repair": {"code": _PLOT_CODE},
    "reporter": "# Analysis Report\n\n## Overview\n共 5 位客户。\n",
}

# 某个 Agent 的回复按调用次序变化时用它：列表按序消费，用完后固定停在最后一项。
# 用途是构造"首次执行失败 → 修复救回"这类需要跨轮次变化的场景。
_REPLY_SEQUENCES: dict[str, list[Any]] = {}
_REPLY_INDEX: dict[str, int] = {}


def _reply_for(agent_name: str) -> Any:
    items = _REPLY_SEQUENCES.get(agent_name)
    if not items:
        return REPLIES[agent_name]
    index = _REPLY_INDEX.get(agent_name, 0)
    _REPLY_INDEX[agent_name] = index + 1
    return items[min(index, len(items) - 1)]

# 崩溃开关：在某次 Agent 调用上调用 os._exit 模拟进程被强杀
_CRASH_ON_AGENT = {"value": None}


class _Scripted(BaseChatModel):
    agent_name: str = ""
    record_path: str = ""
    crash_on: str | None = None
    # 崩溃时点：before_request（请求尚未发出）/ after_request（上游已收到并返回，
    # 但节点还没把结果写进 state）
    crash_at: str = "after_request"
    # 只在该 Agent 的第 N 次调用上崩溃（1 基）。None 表示每次调用都崩溃。
    # 用于区分"第一次修复"与"第二次修复"这类多轮场景。
    crash_on_sequence: int | None = None
    call_sequence: int = 0

    @property
    def _llm_type(self) -> str:
        return f"xproc-{self.agent_name}"

    def _crash(self) -> None:
        # 硬退出：不执行清理，也不 flush 任何缓冲，模拟真实崩溃
        import os

        os._exit(17)

    def _should_crash(self) -> bool:
        if self.crash_on != self.agent_name:
            return False
        if self.crash_on_sequence is None:
            return True
        return self.call_sequence == self.crash_on_sequence

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        path = Path(self.record_path)

        # 该 Agent 的第几次调用（本进程内计数）
        if self.crash_on == self.agent_name:
            self.call_sequence += 1

        if self._should_crash() and self.crash_at == "before_request":
            # 请求尚未发出：上游不会收到任何东西
            self._crash()

        # 记录一次真实的上游请求
        with path.open("a", encoding="utf-8") as handle:
            handle.write(self.agent_name + "\n")
            handle.flush()
            import os as _os

            _os.fsync(handle.fileno())

        if self._should_crash():
            # 硬退出：不执行任何清理，模拟真实崩溃
            import os

            os._exit(17)

        reply = _reply_for(self.agent_name)
        content = (
            reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)
        )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])


def _factory(
    run_id: str,
    workdir: Path,
    crash_on: str | None,
    crash_at: str = "after_request",
    crash_on_sequence: int | None = None,
) -> ModelFactory:
    record_path = workdir / "llm_calls.txt"

    def resolver(agent_name: str) -> _Scripted:
        return _Scripted(
            agent_name=agent_name,
            record_path=str(record_path),
            crash_on=crash_on,
            crash_at=crash_at,
            crash_on_sequence=crash_on_sequence,
        )

    return ModelFactory(
        settings=Settings(
            api_key="sk-fake", base_url="http://127.0.0.1:8000/v1", model="deepseek-chat"
        ),
        context=CallContext(run_id=run_id, client_app="insightflow-agent2"),
        resolver=resolver,
    )


def _summarize(result: dict[str, Any]) -> dict[str, Any]:
    interrupts = result.get("__interrupt__") or []
    pending = []
    for item in interrupts:
        value = getattr(item, "value", None)
        pending.append(value if isinstance(value, dict) else {"value": value})

    return {
        "run_status": result.get("run_status"),
        "analysis_status": result.get("analysis_status"),
        "status_reason": result.get("status_reason"),
        "tasks": len(result.get("analysis_plan") or []),
        "execution_results": len(result.get("execution_results") or []),
        "failed_tasks": len(result.get("failed_tasks") or []),
        "visualizations": len(result.get("visualization_results") or []),
        "artifacts": [Path(path).name for path in (result.get("artifacts") or [])],
        "approvals": len(result.get("approvals") or []),
        "interrupted": bool(interrupts),
        "pending_approval": pending[0] if pending else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="跨进程恢复驱动器")
    parser.add_argument("phase", choices=("start", "resume", "approve", "inspect"))
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-source", required=True)
    parser.add_argument(
        "--require-approval", action="store_true", default=False
    )
    parser.add_argument("--crash-on", default=None)
    parser.add_argument(
        "--fail-first-execution",
        action="store_true",
        default=False,
        help="让首次执行代码抛错，从而进入修复循环（用于验证修复相关的恢复语义）",
    )
    parser.add_argument(
        "--repair-sequence-fails",
        action="store_true",
        default=False,
        help="第一轮修复给出的代码仍然抛错，用于逼出第二轮修复",
    )
    parser.add_argument(
        "--crash-on-sequence",
        type=int,
        default=None,
        help="只在该 Agent 的第 N 次调用上崩溃（1 基）；缺省则每次调用都崩",
    )
    parser.add_argument(
        "--crash-at",
        default="after_request",
        choices=("before_request", "after_request"),
        help=(
            "崩溃时点：before_request = 请求尚未发出；"
            "after_request = 上游已收到并返回，但节点未写回 state"
        ),
    )
    parser.add_argument("--decision", default="approve")
    parser.add_argument(
        "--grant-llm-retry",
        action="store_true",
        default=False,
        help="恢复时未确认的模型调用是否获准重发",
    )
    args = parser.parse_args()

    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    checkpoint_db = workdir / "checkpoints.sqlite"

    if args.fail_first_execution:
        # 首次生成的代码抛错 → 路由到 repair；修复给出的代码可正常执行。
        _REPLY_SEQUENCES["executor"] = [
            {"code": "raise ValueError('first-execution-fails')"},
            {"code": "result = len(df)"},
        ]

    if args.repair_sequence_fails:
        # 第一轮修复仍抛错 → 必然进入第二轮修复，用于验证两轮的逻辑身份不同。
        _REPLY_SEQUENCES["repair"] = [
            {"code": "raise ValueError('repair-round-1-fails')"},
            {"code": "result = len(df)"},
        ]

    with SqliteSaver.from_conn_string(str(checkpoint_db)) as saver:
        if args.phase == "inspect":
            state = read_run_state(args.run_id, checkpointer=saver)
            print(json.dumps({"state": state}, ensure_ascii=False, default=str))
            return 0

        if args.phase == "start":
            result = run_analysis(
                run_id=args.run_id,
                data_sources=[args.data_source],
                model_factory=_factory(
                    args.run_id,
                    workdir,
                    args.crash_on,
                    args.crash_at,
                    args.crash_on_sequence,
                ),
                checkpointer=saver,
                resolver=ResourceResolver(),
                output_root=workdir,
                require_plan_approval=args.require_approval,
            )
            print(json.dumps(_summarize(result), ensure_ascii=False, default=str))
            return 0

        if args.phase == "resume":
            # 恢复时若账本显示有未确认的模型调用，工作流会中断等待决定；
            # --grant-llm-retry 相当于人工批准重发。
            grants = None
            if args.grant_llm_retry:
                grants = {"llm_decision": "retry"}

            result = run_analysis(
                run_id=args.run_id,
                data_sources=[args.data_source],
                model_factory=_factory(args.run_id, workdir, None),
                checkpointer=saver,
                resolver=ResourceResolver(),
                output_root=workdir,
                continue_from_checkpoint=True,
                resume_value=grants,
            )
            print(json.dumps(_summarize(result), ensure_ascii=False, default=str))
            return 0

        if args.phase == "approve":
            result = run_analysis(
                run_id=args.run_id,
                data_sources=[args.data_source],
                model_factory=_factory(args.run_id, workdir, None),
                checkpointer=saver,
                resolver=ResourceResolver(),
                output_root=workdir,
                resume=Command(resume={"decision": args.decision}),
                require_plan_approval=True,
            )
            print(json.dumps(_summarize(result), ensure_ascii=False, default=str))
            return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
