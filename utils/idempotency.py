"""副作用幂等与模型调用的可核对记录。

Checkpoint 是"至少一次"语义：进程可能在节点执行完、提交之前被杀掉，
恢复时会重放该节点。因此每个有副作用的步骤都要有稳定的操作 ID，
并把"已完成"记录写进 state，恢复时据此判断：

- 已完成且能取得原结果：直接复用，不重复产生副作用；
- 已完成但结果不可复用：明确失败，而不是静默重做；
- 未完成：正常执行。

**文件类副作用**（报告、PNG、结果产物）由上面的机制保证不重复产生。

**模型调用无法保证恰好一次**，原因是 Checkpoint 只在节点结束时提交：

    请求发出 → 进程被杀 → 节点从未提交
    → state 里没有任何痕迹 → 本地无法判断上游是否已处理并计费

因此模型调用另有一条**账本**（utils/llm_ledger），在请求前就落盘 `started`。
它是唯一能在这种崩溃里留下证据的记录。恢复时用账本比对出"未确认的调用"，
写进 state 的 `pending_llm_calls`，由 `guard_llm_calls` 暂停并交给人工决定。
这是保守策略——默认不重发，而不是默认重发——但它**不等于**恰好一次：
`started` 无法区分"上游已处理"与"上游没收到"。

操作 ID 必须只由稳定输入构成（run_id、节点名、任务/图表 ID），
不能包含时间戳或随机值——否则恢复后算出的 ID 与之前不同，幂等就失效了。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from utils import llm_ledger

# 已完成的操作在此登记，键为操作 ID
COMPLETED_KEY = "completed_operations"

STATUS_COMPLETED = "completed"
STATUS_IN_PROGRESS = "in_progress"
STATUS_UNCERTAIN = "uncertain"


def operation_id(*parts: object) -> str:
    """由稳定输入拼出操作 ID。

    调用方必须保证传入的都是可复现的值（ID、序号），不要传时间或随机数。
    """
    return ":".join(str(part) for part in parts)


def task_operation(run_id: str, task_id: str, attempt: int) -> str:
    """一次工具执行。

    attempt 参与 ID：修复循环里的第 2 次尝试是一次新的执行，
    不能被当成第 1 次的重放。
    """
    return operation_id(run_id, "tool_execution", task_id, f"attempt_{attempt}")


def chart_operation(run_id: str, chart_id: str) -> str:
    """一张图的生成。

    刻意不带 attempt：可视化结果按 chart_id 保留最后一条，
    因此"这张图是否已经产出过"与尝试次数无关，否则重放会因算出不同 ID 而重复执行。
    """
    return operation_id(run_id, "plot_execution", chart_id)


def artifact_operation(run_id: str) -> str:
    return operation_id(run_id, "artifact")


def plan_approval_id(run_id: str) -> str:
    """同一个 Run 的执行前审批只有一个 ID。

    用固定 ID 而不是随机值：恢复后重新计算必须得到同一个值，
    否则重复审批保护会失效。
    """
    return operation_id(run_id, "approval", "plan_review")


def report_approval_id(run_id: str) -> str:
    return operation_id(run_id, "approval", "report_publish")


def llm_operation(run_id: str, node: str, subject: str) -> str:
    """一次模型调用的逻辑身份。

    subject 用来区分同一节点针对不同任务/图表的调用，也必须包含**逻辑尝试
    序号**：修复/精修/绘图修复都会在同一节点里反复调用同一个 Agent，若 subject
    只有 task_id / chart_id，第二次尝试会算出与第一次相同的操作 ID，被
    `already_done` 当成同一次调用而直接复用旧结果——即"修复循环空转"。

    拼接请用 `attempt_subject()`，避免各处自行拼出不一致的格式。
    """
    return operation_id(run_id, "llm", node, subject)


# 逻辑尝试标识的前缀。放在 subject 里而不是只传给账本，是为了让
# operation_id 本身就能区分尝试：声明式幂等登记、账本核对、日志都看同一个值。
_ATTEMPT_MARK = "#"


def attempt_subject(base: str, *counters: int | None) -> str:
    """把逻辑尝试序号编进 subject。

    例如 `attempt_subject("task_1", 0, 1)` → `"task_1#r0f1"`。

    为什么不用时间戳或随机数：恢复是从上一张 Checkpoint 重算操作 ID 的，
    ID 必须可复现，否则幂等失效（见模块开头的说明）。
    为什么把计数器写进字符串而不是只加一个数字：不同的循环用不同的计数器
    （retry_count / refine_count / plot_retry_count），只写一个裸数字会让
    "第 1 轮精修"与"第 1 轮修复"撞成同一个 ID。
    """
    if not counters or all(count is None for count in counters):
        return base
    token = "".join("_" if count is None else str(count) for count in counters)
    return f"{base}{_ATTEMPT_MARK}{token}"


class LlmCallAbandoned(RuntimeError):
    """未确认的模型调用未获准重试，本次运行终止。

    这是**受控终止**，不是工作流故障：Run 应落到 failed 并带上原因，
    而不是停留在 running 或抛出未分类的异常。
    """


# 恢复时，state 里这些键指向"上次可能已发出但未登记结果"的模型调用
PENDING_LLM_KEY = "pending_llm_calls"


def unconfirmed_llm_calls(state: dict[str, Any]) -> list[str]:
    """尚未确认结果的模型调用操作 ID。"""
    return list(state.get(PENDING_LLM_KEY) or [])


def guard_llm_calls(
    state: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """节点执行 Agent 之前的守卫。

    返回 `(updates, decision)`：

    - 没有未确认调用：(None, None)，正常执行；
    - 有未确认调用且尚未获准重试：抛 interrupt 暂停，等人工决定；
    - 已获准重试：({"pending_llm_calls": []}, "retry")，清空标记后重发。

    为什么需要它：Checkpoint 只在节点结束时提交，因此"请求已发出、节点未返回"
    这种崩溃在 state 里**不留痕迹**，仅靠 Checkpoint 无法发现重复。
    未确认调用的清单由账本（utils/llm_ledger）在恢复时比对得出，写进 state 后
    由这里拦截。这是保守策略：默认不重发，而不是默认重发。
    """
    pending = unconfirmed_llm_calls(state)

    if not pending:
        return None, None

    if state.get("llm_retry_granted") == "retry":
        return {PENDING_LLM_KEY: [], "llm_retry_granted": None}, "retry"

    from langgraph.types import interrupt

    decision = interrupt(
        {
            "approval_id": f"{state.get('run_id')}:llm_replay",
            "run_id": state.get("run_id"),
            "kind": "llm_replay",
            "summary": {
                "reason": (
                    "上一次运行在这些模型调用发出后、结果登记前终止。"
                    "本地无法判断上游是否已经处理并计费。"
                ),
                "operations": pending,
                "default": "skip",
                "options": ["retry", "skip"],
            },
        }
    )

    verdict = ""
    if isinstance(decision, dict):
        verdict = str(decision.get("llm_decision", "")).lower()

    if verdict != "retry":
        raise LlmCallAbandoned(
            "存在未确认的模型调用且未获准重试：" + ", ".join(pending)
        )

    return {PENDING_LLM_KEY: [], "llm_retry_granted": None}, "retry"


def run_agent_call(
    state: dict[str, Any],
    *,
    node: str,
    subject: str,
    call: Any,
    attempt: int = 1,
    before_request: Any = None,
) -> tuple[Any, dict[str, dict[str, Any]], dict[str, Any]]:
    """执行一次 Agent 调用，并把它记入账本与幂等登记。

    `subject` 必须已经用 `attempt_subject()` 带上逻辑尝试序号（对会重复调用的
    节点而言），`attempt` 只用于账本上的人读标注——真正的幂等身份来自
    `operation_id`。两者分开是因为账本的 attempt 字段不参与去重计算。

    返回 `(结果, 幂等登记, 额外 state 增量)`。第三个返回值不能省：
    获准重试时必须把 `pending_llm_calls` 清空写回 state，
    否则下一次 resume 会从账本重新算出同一个未确认调用并再次中断。

    `before_request` 是**真实请求发出前的最后一道闸门**（P3 用它判预算）。
    位置很关键，必须同时满足：

    - 在幂等命中判断**之后**：复用不会发出请求，因此不该被预算拦下，
      也不该消耗额度；
    - 在 `record_started` **之前**：拦住时不能往账本写 `started`，
      否则会留下永远没有 `completed` 的记录，恢复时被当成"未确认的模型调用"
      并要求人工裁决——预算问题不该变成人工工单。

    `call` 里若发生结构化输出解析失败并内部重试，重试的计数与判定由
    `utils/llm_retry` 的回调完成（见那里的 `before_request` / `on_request`），
    本函数只保证"至少有一次真实请求"这一口径。

    三层保护各司其职：

    1. **账本**：请求发出前写 `started`、返回后写 `completed`。
       它是唯一能在"节点未提交"的崩溃中留下痕迹的记录；
    2. **幂等登记**：同一逻辑调用已完成时直接复用结果，不再请求；
    3. **未确认拦截**：恢复时若账本显示某次调用没有完成记录，暂停等人工决定。

    注意第 3 点只能做到"不静默重发"，做不到恰好一次：
    `started` 无法区分"上游已处理"与"上游没收到"。
    """
    run_id = state.get("run_id") or "unknown"
    op_id = llm_operation(run_id, node, subject)

    done = already_done(state, op_id)
    if done is not None and done.detail is not None:
        return done.detail.get("payload"), completed_operations(state), {}

    # 每次进入都要先过未确认检查
    updates, _decision = guard_llm_calls(state)

    # 预算闸门：只判不记，且必须在落账本之前
    if before_request is not None:
        before_request()

    ledger_root = _ledger_root()
    llm_ledger.record_started(
        run_id=run_id, operation_id=op_id, node=node, attempt=attempt, root=ledger_root
    )

    payload = call()

    llm_ledger.record_completed(
        run_id=run_id, operation_id=op_id, node=node, attempt=attempt, root=ledger_root
    )

    merged = completed_operations(state)
    merged[op_id] = OperationRecord(
        operation_id=op_id,
        status=STATUS_COMPLETED,
        detail={"payload": payload},
    ).to_dict()

    # 明确告诉调用方"这次真的请求了上游"。
    # P3 的预算治理需要这个区分：命中幂等复用（上面那条 return）时
    # 没有发生任何模型调用，把它也计进 llm_calls 会让预算虚高。
    # 只加键、不改既有三个返回值的位置，因此对现有调用方是兼容的。
    updates = dict(updates or {})
    updates["__upstream_invoked__"] = True
    updates["__operation_id__"] = op_id

    return payload, merged, updates


def _ledger_root() -> Any:
    """账本与产物放在同一 root 下，测试可随 output_root 一起隔离。"""
    from core.runtime import RuntimeContextMissing, current_runtime

    try:
        return current_runtime().output_root
    except RuntimeContextMissing:
        return "outputs"


@dataclass(frozen=True)
class OperationRecord:
    """一条已完成操作的记录。"""

    operation_id: str
    status: str
    result_ref: str | None = None
    detail: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "status": self.status,
            "result_ref": self.result_ref,
            "detail": self.detail or {},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OperationRecord":
        return cls(
            operation_id=payload["operation_id"],
            status=payload["status"],
            result_ref=payload.get("result_ref"),
            detail=payload.get("detail") or {},
        )


def completed_operations(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return dict(state.get(COMPLETED_KEY) or {})


def lookup(state: dict[str, Any], op_id: str) -> OperationRecord | None:
    payload = completed_operations(state).get(op_id)
    return OperationRecord.from_dict(payload) if payload else None


def record(
    state: dict[str, Any],
    op_id: str,
    *,
    status: str = STATUS_COMPLETED,
    result_ref: str | None = None,
    detail: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """返回更新后的登记表，供节点作为 state 增量返回。"""
    updated = completed_operations(state)
    updated[op_id] = OperationRecord(
        operation_id=op_id,
        status=status,
        result_ref=result_ref,
        detail=detail,
    ).to_dict()
    return updated


def already_done(state: dict[str, Any], op_id: str) -> OperationRecord | None:
    """已完成（非未决）时返回记录，否则 None。"""
    found = lookup(state, op_id)
    if found is not None and found.status == STATUS_COMPLETED:
        return found
    return None


def replay_guard(state: dict[str, Any], op_id: str, *, what: str) -> OperationRecord | None:
    """副作用执行前的统一守卫。

    - 已完成：返回记录，调用方应复用原结果而不是重做；
    - 未决（uncertain）：说明上次进程在结果落盘前终止，本地无法判断上游是否
      已经生效，这里直接失败并要求人工处理，避免重复产生副作用。
    """
    found = lookup(state, op_id)

    if found is None:
        return None

    if found.status == STATUS_COMPLETED:
        return found

    if found.status == STATUS_UNCERTAIN:
        raise RuntimeError(
            f"{what} 上次执行结果未确认（{op_id}）：进程可能在结果持久化前终止。"
            "本地 Checkpoint 无法判断上游是否已生效，已停止以避免重复产生副作用。"
        )

    return None
