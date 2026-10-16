"""P5-B/D：统一评测执行器与报告输出。

## 执行模型

每个任务用**脚本化 Mock 模型**驱动真实工作流：真实 Docker 沙箱、
真实 MCP 子进程、真实 Checkpoint —— 只有 LLM 是 Mock。
因此得到的是**工程可靠性与判分逻辑**的证据，
**不是**真实模型自主分析能力的证据。这一点在报告里必须显式写明。

## 为什么用 Mock 而不是真实模型

评测环境没有明确授权的真实模型 API 与费用预算。伪造成功率、成本或
Token 数据是不允许的，因此本模块：

1. 交付可用的 Mock 工程可靠性评测；
2. 保留 `--provider real` 入口供将来接入真实模型；
3. 在报告里列出**尚未验证**的真实模型能力指标清单。
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from core.config import Settings
from core.llm import CallContext, ModelFactory
from core.runtime import McpSettings
from eval.scoring import (
    STATUS_CORRECT,
    STATUS_ERROR,
    STATUS_INCOMPLETE,
    STATUS_INCORRECT,
    STATUS_PARTIAL,
    EvalSummary,
    TaskOutcome,
    extract_answer,
    grouped_match,
    values_match,
)
from eval.tasks import EvalTask, build_tasks, materialise
from ingestion.resources import ResourceResolver
from services.analysis_service import run_analysis

#: 在报告里必须出现的口径声明。
MOCK_SCOPE_NOTE = (
    "本评测使用脚本化 Mock LLM 驱动真实工作流（真实 Docker 沙箱、真实 MCP "
    "子进程、真实 Checkpoint）。它验证的是工程链路与判分逻辑的可靠性，"
    "**不能**用来声称真实模型的自主分析能力。"
)

#: 尚未验证的真实模型能力指标（缺明确授权的 API 与预算）。
UNVERIFIED_REAL_MODEL_METRICS = (
    "真实模型下的关键数值正确率",
    "真实模型下的任务完成率",
    "真实模型在模糊需求下的澄清能力",
    "真实模型的 Token 消耗与单任务成本",
    "真实模型对错误反馈的修复成功率",
    "真实模型在长上下文下的稳定性",
)


class ScriptedModelFactory:
    """按任务脚本化的 Mock 模型工厂。

    每个任务给一段固定的 executor 代码与一份计划；模型不"思考"，
    只按脚本作答。因此评测结果反映的是工作流与判分，不是模型能力。
    """

    def __init__(self, task: EvalTask) -> None:
        self.task = task
        self.calls: list[str] = []
        self.prompts: dict[str, list[str]] = {}

    def _replies(self) -> dict[str, Any]:
        plan = {
            "tasks": [
                {
                    "task_id": "task_1",
                    "description": self.task.description,
                    "target_tables": self.task.target_tables,
                    "expected_output": self.task.description,
                }
            ]
        }
        return {
            "planner": plan,
            "executor": {"code": self.task.code},
            "repair": {"code": self.task.code},
            "refine": {"code": self.task.code},
            "critic": {"validation_passed": True, "feedback": "ok"},
            "visualizer": {"charts": []},
            "reporter": f"# Analysis Report\n\n## Overview\n{self.task.description}\n",
        }

    def build(self, run_id: str) -> ModelFactory:
        import json as _json

        from langchain_core.callbacks import CallbackManagerForLLMRun
        from langchain_core.language_models import BaseChatModel
        from langchain_core.messages import AIMessage, BaseMessage
        from langchain_core.outputs import ChatGeneration, ChatResult

        owner = self
        replies = self._replies()

        class _Scripted(BaseChatModel):
            agent_name: str = ""

            @property
            def _llm_type(self) -> str:
                return f"p5-{self.agent_name}"

            def _generate(
                self,
                messages: list[BaseMessage],
                stop: list[str] | None = None,
                run_manager: CallbackManagerForLLMRun | None = None,
                **kwargs: Any,
            ) -> ChatResult:
                owner.calls.append(self.agent_name)
                owner.prompts.setdefault(self.agent_name, []).append(
                    "\n".join(str(message.content) for message in messages)
                )
                reply = replies.get(self.agent_name, {})
                content = (
                    reply
                    if isinstance(reply, str)
                    else _json.dumps(reply, ensure_ascii=False)
                )
                return ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content=content))]
                )

        def resolver(agent_name: str) -> _Scripted:
            return _Scripted(agent_name=agent_name)

        return ModelFactory(
            settings=Settings(
                api_key="sk-fake",
                base_url="http://127.0.0.1:8000/v1",
                model="deepseek-chat",
            ),
            context=CallContext(run_id=run_id, client_app="insightflow-agent2"),
            resolver=resolver,
        )


def _records_of(result: dict) -> list[dict]:
    from utils.result_artifacts import unwrap_records

    successful = [
        item for item in result.get("execution_results") or [] if item.get("success")
    ]
    if not successful:
        return []

    value = unwrap_records(successful[-1].get("result"))

    if isinstance(value, dict) and "rows" in value and "columns" in value:
        return list(value.get("rows") or [])
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return value
    if isinstance(value, dict):
        return [value]
    return []


def run_task(
    task: EvalTask,
    *,
    workdir: Path,
    data: dict[str, Path],
    mode: str = "mcp",
    injector: Callable[[EvalTask, dict], None] | None = None,
) -> tuple[TaskOutcome, dict[str, Any]]:
    """执行一个任务并给出评测结论。

    `injector` 供故障注入场景在运行前改脚本（例如把代码换成一个会抛错的版本）。
    """
    run_id = f"eval-{task.task_id}"
    run_dir = workdir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    factory = ScriptedModelFactory(task)

    if injector is not None:
        injector(task, {"factory": factory})

    source = data[task.data_source]
    started = time.perf_counter()

    result: dict[str, Any] = {}
    error: str | None = None
    try:
        result = run_analysis(
            run_id=run_id,
            data_sources=[str(source)],
            model_factory=factory.build(run_id),
            resolver=ResourceResolver(),
            output_root=workdir,
            mcp=McpSettings(mode=mode),
        )
    except Exception as exc:  # noqa: BLE001 - 评测要把异常也记成结果
        error = f"{type(exc).__name__}: {exc}"

    duration = time.perf_counter() - started

    usage = (result.get("execution_budget") or {}).get("usage") or {}
    analysis_status = result.get("analysis_status")

    outcome = TaskOutcome(
        task_id=task.task_id,
        family=task.family,
        status=STATUS_ERROR,
        expected=task.expected,
        duration_seconds=duration,
        llm_calls=int(usage.get("llm_calls") or 0),
        tool_calls=int(usage.get("tool_calls") or 0),
        analysis_status=analysis_status,
    )

    if error is not None:
        outcome.status = STATUS_ERROR
        outcome.reason = error
        return outcome, result

    records = _records_of(result)
    actual = extract_answer(task, records)
    outcome.actual = actual

    if analysis_status == "failed" or not records:
        outcome.status = STATUS_INCOMPLETE
        failures = result.get("failed_tasks") or []
        outcome.reason = (
            str(failures[0].get("reason")) if failures else result.get("status_reason")
        ) or "没有可判分的执行结果"
        return outcome, result

    # 分组答案用严格比较（多一组/少一组都算错）；其余用宽松比较
    if getattr(task, "shape", "scalar") == "rows_grouped" and isinstance(
        task.expected, dict
    ):
        matched = grouped_match(actual, task.expected)
    else:
        matched = values_match(actual, task.expected)

    if matched:
        outcome.status = STATUS_CORRECT
    else:
        outcome.status = STATUS_INCORRECT
        outcome.reason = "数值与标准答案不一致"

    # 有计划但有任务失败：正确也标注为 partial，避免与"完全成功"混淆
    if analysis_status == "partial" and outcome.status == STATUS_CORRECT:
        outcome.status = STATUS_PARTIAL
        outcome.reason = "结果正确，但有任务失败（partial）"

    return outcome, result


@dataclass
class EvalRun:
    """一次完整评测的产物。"""

    summary: EvalSummary
    mode: str
    data_root: Path
    workdir: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": MOCK_SCOPE_NOTE,
            "sql_backend": self.mode,
            "data_root": str(self.data_root),
            "workdir": str(self.workdir),
            "unverified_real_model_metrics": list(UNVERIFIED_REAL_MODEL_METRICS),
            "summary": self.summary.to_dict(),
        }


def run_all(
    *,
    workdir: Path,
    mode: str = "mcp",
    tasks: Iterable[EvalTask] | None = None,
    on_outcome: Callable[[TaskOutcome], None] | None = None,
) -> EvalRun:
    """跑完整评测集合。"""
    workdir.mkdir(parents=True, exist_ok=True)
    data = materialise(workdir / "data")

    selected = list(tasks if tasks is not None else build_tasks())
    summary = EvalSummary()

    for task in selected:
        outcome, _result = run_task(task, workdir=workdir, data=data, mode=mode)
        summary.add(outcome)
        if on_outcome is not None:
            on_outcome(outcome)

    return EvalRun(summary=summary, mode=mode, data_root=workdir / "data", workdir=workdir)


# ---------------------------------------------------------------------------
# 报告输出
# ---------------------------------------------------------------------------


def write_json(run: EvalRun, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(run.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def write_csv(run: EvalRun, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = run.summary.outcomes
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "task_id",
                "family",
                "status",
                "completed",
                "correct",
                "analysis_status",
                "duration_seconds",
                "llm_calls",
                "tool_calls",
                "expected",
                "actual",
                "reason",
            ]
        )
        for item in rows:
            writer.writerow(
                [
                    item.task_id,
                    item.family,
                    item.status,
                    int(item.completed),
                    int(item.correct),
                    item.analysis_status or "",
                    f"{item.duration_seconds:.3f}",
                    item.llm_calls,
                    item.tool_calls,
                    json.dumps(item.expected, ensure_ascii=False),
                    json.dumps(item.actual, ensure_ascii=False),
                    item.reason or "",
                ]
            )
    return path


def write_markdown(run: EvalRun, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = run.summary

    lines: list[str] = [
        "# 评测报告（Mock 工程可靠性）",
        "",
        f"> {MOCK_SCOPE_NOTE}",
        "",
        "## 运行配置",
        "",
        f"- SQL 后端：`{run.mode}`",
        f"- 评测数据根目录：`{run.data_root}`",
        f"- 工作目录：`{run.workdir}`",
        f"- 任务总数：{summary.total}",
        "",
        "## 总体指标",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| 任务完成数 | {summary.completed} / {summary.total} |",
        f"| 完成率 | {summary.completion_rate:.2%} |",
        f"| 关键数值正确数 | {summary.correct} / {summary.total} |",
        f"| **正确率**（分母为全部任务） | {summary.correctness_rate:.2%} |",
        f"| **条件正确率**（分母为已完成任务） | {summary.conditional_accuracy:.2%} |",
        f"| 算错但完成 | {summary.incorrect} |",
        f"| 部分完成（partial） | {summary.partial} |",
        f"| 未完成 | {summary.incomplete} |",
        f"| 异常终止 | {summary.errored} |",
        f"| 总耗时（秒） | {summary.total_seconds:.1f} |",
        f"| 模型调用次数（Mock） | {summary.total_llm_calls} |",
        f"| 工具调用次数 | {summary.total_tool_calls} |",
        "",
        '**完成率与正确率是不同的指标**：完成率回答"跑通了吗"，'
        '正确率回答"算对了吗"。两者都低时，先看未完成的原因。',
        "",
        "## 分族指标",
        "",
        "| 任务族 | 总数 | 完成 | 正确 | 完成率 | 正确率 |",
        "|---|---|---|---|---|---|",
    ]

    for family, bucket in sorted(summary.by_family().items()):
        lines.append(
            f"| {family} | {bucket['total']} | {bucket['completed']} | "
            f"{bucket['correct']} | {bucket['completion_rate']:.2%} | "
            f"{bucket['correctness_rate']:.2%} |"
        )

    failures = [item for item in summary.outcomes if not item.correct]

    lines += [
        "",
        "## 失败案例",
        "",
    ]
    if not failures:
        lines.append("（无）")
    else:
        lines += [
            "| 任务 | 状态 | 期望 | 实际 | 原因 |",
            "|---|---|---|---|---|",
        ]
        for item in failures:
            expected = json.dumps(item.expected, ensure_ascii=False)
            actual = json.dumps(item.actual, ensure_ascii=False)
            reason = (item.reason or "").replace("|", "/")
            lines.append(
                f"| {item.task_id} | {item.status} | `{expected}` | "
                f"`{actual}` | {reason} |"
            )

    lines += [
        "",
        "## 尚未验证的真实模型能力",
        "",
        "当前环境没有明确授权的真实模型 API 与费用预算，因此下列指标"
        "**未被验证**。它们需要一个显式授权的真实模型运行才能给出，"
        "在此之前不得用 Mock 结果替代：",
        "",
    ]
    for item in UNVERIFIED_REAL_MODEL_METRICS:
        lines.append(f"- {item}")

    lines += [
        "",
        "## 可复现说明",
        "",
        "```bash",
        "# 完整评测（Mock，不访问任何模型服务）",
        "python scripts/run_eval.py --output eval/reports",
        "",
        "# 只跑某个任务族",
        "python scripts/run_eval.py --family sqlite_join --output eval/reports",
        "```",
        "",
        "评测数据由 `eval/tasks.py` 用代码生成，不依赖网络与随机数；"
        "标准答案由同一模块用纯 Python 独立算出，不经过被测工作流。",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_plot(run: EvalRun, path: Path) -> Path | None:
    """画一张分族完成率/正确率对比图。

    用 matplotlib 直接生成，不经过 Docker —— 这是评测产物，
    不是"模型生成的图表"，混用会让人误以为它经过了沙箱。

    **中文标签需要宿主装了 CJK 字体。** 沙箱镜像里有 Noto CJK，
    宿主不一定有；缺字体时 matplotlib 会画出方框并刷一堆
    "Glyph missing from font" 警告。因此这里先探测字体，
    探测不到就退回英文标签，而不是产出一张看不懂的图。
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.font_manager as font_manager
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return None

    cjk_candidates = (
        "Noto Sans CJK SC",
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "WenQuanYi Zen Hei",
        "Source Han Sans SC",
    )
    available = {font.name for font in font_manager.fontManager.ttflist}
    cjk_font = next((name for name in cjk_candidates if name in available), None)

    if cjk_font:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [cjk_font, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        completion_label = "完成率"
        correctness_label = "正确率"
        ylabel = "百分比 (%)"
        title = f"分族完成率与正确率（Mock 工程评测，n={run.summary.total}）"
    else:
        # 宿主没有 CJK 字体：用英文标签，避免方框与警告
        completion_label = "completion rate"
        correctness_label = "correctness rate"
        ylabel = "percent (%)"
        title = (
            f"Completion vs correctness by family "
            f"(mock engineering eval, n={run.summary.total})"
        )

    families = sorted(run.summary.by_family())
    if not families:
        return None

    by_family = run.summary.by_family()
    completion = [by_family[name]["completion_rate"] * 100 for name in families]
    correctness = [by_family[name]["correctness_rate"] * 100 for name in families]

    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(max(8, len(families) * 1.1), 5))
    positions = range(len(families))
    width = 0.38
    plt.bar(
        [p - width / 2 for p in positions],
        completion,
        width,
        label=completion_label,
    )
    plt.bar(
        [p + width / 2 for p in positions],
        correctness,
        width,
        label=correctness_label,
    )
    plt.xticks(list(positions), families, rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path
