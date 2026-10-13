"""P0 V2 Mock 基线采集。

用固定的数据集与 Mock 模型跑完整工作流，记录可稳定测量的工程指标。
不访问真实模型服务，因此 token 与成本一律记为不可用——不编造数据。

**重要：这不是"原始 V1 的性能数据"。** 采集发生在 P0 修改后的工作树上
（agents 已支持注入、nodes 已改为注入模型），且模型调用被替换为 Mock，
因此它只能用于 P0→后续版本之间的同口径回归对比，不能代表 V1 在真实模型下的表现。
脚本会把 commit、工作树状态与输入一并写进结果，避免这份数据被误用。

用法（在项目根目录）：
    .venv\\Scripts\\python.exe scripts/run_baseline.py
    .venv\\Scripts\\python.exe scripts/run_baseline.py --out docs/baseline.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.callbacks import CallbackManagerForLLMRun  # noqa: E402
from langchain_core.language_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, BaseMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402

from core.config import Settings  # noqa: E402
from core.llm import CallContext, ModelFactory  # noqa: E402
from services.analysis_service import run_analysis  # noqa: E402

# 所有场景共用的基础响应，避免某个分支被触发时缺少对应 Agent 的回复
BASE_REPLIES: dict[str, Any] = {
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
    "critic": {"validation_passed": True, "feedback": "符合任务要求"},
    "visualizer": {"charts": []},
    "plotter": {"code": "result = df"},
    "plot_repair": {"code": "result = df"},
    "reporter": "# Analysis Report\n\n## Overview\n共 5 位客户。\n",
}


def scenario(overrides: dict[str, Any], **meta: Any) -> dict[str, Any]:
    return {**meta, "replies": {**BASE_REPLIES, **overrides}}


# 固定场景：单表聚合、多表关联、修复循环
SCENARIOS: dict[str, dict[str, Any]] = {
    "single_table": scenario(
        {},
        data_sources=["data/customers.csv"],
        analysis_focus="统计客户总数",
    ),
    "multi_table": scenario(
        {
            "planner": {
                "tasks": [
                    {
                        "task_id": "task_1",
                        "description": "按客户汇总订单金额",
                        "target_tables": ["customers", "orders"],
                        "expected_output": "每个客户的订单总金额",
                    }
                ]
            },
            # 多表任务必须经 dfs 按表名取用；单表别名 df 不可用，
            # 这也是 repair 循环能兜住的典型错误。
            "executor": {
                "code": (
                    "orders = dfs['orders'].copy()\n"
                    "customers = dfs['customers'].copy()\n"
                    "merged = orders.merge(customers, on='customer_id', how='left')\n"
                    "result = merged.groupby('customer_name')['amount'].sum().to_dict()"
                )
            },
            "reporter": "# Analysis Report\n\n## Overview\n按客户汇总订单金额。\n",
        },
        data_sources=["data/customers.csv", "data/orders.csv"],
        analysis_focus="按客户汇总订单金额",
    ),
    "repair_loop": scenario(
        {
            "executor": {"code": "result = nonexistent_name"},
            "reporter": "# Analysis Report\n\n修复后成功。\n",
        },
        data_sources=["data/customers.csv"],
        analysis_focus="触发修复循环",
    ),
}

CALLS: dict[str, int] = {}


class ScriptedModel(BaseChatModel):
    agent_name: str = ""
    replies: dict[str, Any] = {}

    @property
    def _llm_type(self) -> str:
        return f"baseline-{self.agent_name}"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        CALLS[self.agent_name] = CALLS.get(self.agent_name, 0) + 1
        reply = self.replies[self.agent_name]
        content = (
            reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)
        )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])


def build_factory(run_id: str, replies: dict[str, Any]) -> ModelFactory:
    def resolver(agent_name: str) -> ScriptedModel:
        if agent_name not in replies:
            raise AssertionError(f"场景未配置 {agent_name} 的响应")
        return ScriptedModel(agent_name=agent_name, replies=replies)

    return ModelFactory(
        settings=Settings(
            api_key="sk-baseline-mock",
            base_url="http://127.0.0.1:8000/v1",
            model="deepseek-chat",
        ),
        context=CallContext(run_id=run_id, client_app="insightflow-agent2"),
        resolver=resolver,
    )


def run_scenario(name: str, spec: dict[str, Any], repeats: int) -> dict[str, Any]:
    durations: list[float] = []
    samples: list[dict[str, Any]] = []

    for index in range(repeats):
        run_id = f"baseline-{name}-{index + 1}"
        CALLS.clear()
        factory = build_factory(run_id, spec["replies"])

        started = time.perf_counter()
        result = run_analysis(
            run_id=run_id,
            data_sources=spec["data_sources"],
            analysis_focus=spec["analysis_focus"],
            model_factory=factory,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        durations.append(elapsed_ms)

        executions = result["execution_results"]
        samples.append(
            {
                "run_id": run_id,
                "elapsed_ms": round(elapsed_ms, 1),
                "llm_calls": dict(CALLS),
                "llm_calls_total": sum(CALLS.values()),
                "tool_calls": len(executions),
                "tools_used": sorted({item["tool_used"] for item in executions}),
                "tasks_planned": len(result["analysis_plan"]),
                "tasks_failed": len(result["failed_tasks"]),
                "artifacts": [Path(a).name for a in result["artifacts"]],
            }
        )

    first = samples[0]
    return {
        "scenario": name,
        "repeats": repeats,
        "elapsed_ms_min": round(min(durations), 1),
        "elapsed_ms_median": round(statistics.median(durations), 1),
        "elapsed_ms_max": round(max(durations), 1),
        "llm_calls_per_run": first["llm_calls"],
        "llm_calls_total_per_run": first["llm_calls_total"],
        "tool_calls_per_run": first["tool_calls"],
        "tools_used": first["tools_used"],
        "tasks_planned": first["tasks_planned"],
        "tasks_failed": first["tasks_failed"],
        "artifacts": first["artifacts"],
        "samples": samples,
    }


def _git(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(PROJECT_ROOT),
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _provenance() -> dict[str, Any]:
    """记录采集时的代码与环境状态，避免这份数据被当成 V1 实测结果。"""
    dirty = _git("status", "--porcelain")

    return {
        "commit": _git("rev-parse", "HEAD"),
        "commit_subject": _git("log", "-1", "--format=%s"),
        "worktree_dirty": bool(dirty),
        "worktree_changed_files": len(dirty.splitlines()) if dirty else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="InsightFlow P0 V2 Mock 基线采集")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", default="docs/baseline.json")
    args = parser.parse_args()

    report: dict[str, Any] = {
        "label": "P0 V2 Mock baseline",
        "not_a_v1_measurement": (
            "本数据在 P0 修改后的工作树上采集，模型调用已替换为 Mock。"
            "它不是原始 V1 的性能数据，也不构成 V1 与 V2 的性能对比。"
        ),
        "note": (
            "使用 Mock 模型与固定数据集采集，不访问真实模型服务。"
            "token 与成本不可用，因为 Mock 不返回 usage。"
        ),
        "provenance": _provenance(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "repeats": args.repeats,
        "inputs": {
            "data_sources": sorted(
                {source for spec in SCENARIOS.values() for source in spec["data_sources"]}
            ),
            "knowledge_sources": [],
            "mock_behavior": (
                "每个 Agent 由固定脚本响应，内容预先写死；"
                "不调用任何模型服务，也不返回 usage。"
            ),
        },
        "environment_limits": [
            "Mock 模型替代真实模型，无法反映真实推理延迟与失败率",
            "未采集 token 与成本（Mock 不返回 usage）",
            "PDF 产物由 Playwright 生成本地浏览器，耗时与模型无关且受机器负载影响",
            "耗时在不同批次间波动明显（实测 1.4s~2.1s），只适合同一批次内横向比较",
            "单机运行，未做并发与吞吐测试",
        ],
        "scenarios": {},
        "unavailable_metrics": [
            "prompt_tokens",
            "completion_tokens",
            "estimated_cost_usd",
            "task_success_rate_on_real_model",
        ],
    }

    for name, spec in SCENARIOS.items():
        scenario = run_scenario(name, spec, args.repeats)
        report["scenarios"][name] = scenario
        print(
            f"  {name:<14} 中位耗时={scenario['elapsed_ms_median']:>8.1f}ms "
            f"LLM调用={scenario['llm_calls_total_per_run']:<3} "
            f"工具调用={scenario['tool_calls_per_run']:<3} "
            f"失败任务={scenario['tasks_failed']}"
        )

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n基线已写入 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
