"""P3-A：有界 Agent 循环的控制层。

## 这个模块解决什么

现有 19 节点图已经是**有界**的：`MAX_RETRIES` / `MAX_REFINES` /
`MAX_PLOT_RETRIES` 各自限制本地循环，`recursion_limit` 兜住全局。那些上限
是**每类操作的局部计数**，缺的是统一的、跨操作的总量治理：

- 一个 Run 里"工具被调用多少次"没有上限，只受各类 retry 上限间接约束；
- 没有整体时间预算，图可以在很多次合法重试里长时间占用资源；
- 局部计数分散在 `retry_count` / `refine_count` / `plot_retry_count` 里，
  没有单一的"这个 Run 还能不能继续"的判断点。

因此这里**不重写图**，只在既有节点之上补一层计数器与判定：

    Reason（模型给出下一步）
      → Act（执行工具）
      → Observe（工具结果与分类）
      → Decide（在预算内继续 / 修复 / 收敛失败）

## 预算为什么放在 state 里

预算必须**跨进程恢复后不重置**。放在进程内存里的话，崩溃重启就归零，
"有界"就名不副实。因此计数进 `InsightFlowState`，随 Checkpoint 一起持久化。

时间预算**不能**直接用"墙上时钟减去开始时刻"：那样等待人工审批的时间也会
被算作执行消耗（文档说"等待不计入"，而实现会把它计入，属于文档与行为不符）。
现在的做法是累计 `active_seconds`——只在**真实操作发生**时推进：

    capture_active_time()  操作开始，时钟开始走
    settle_active_time()   操作结束，时钟停住

停在等人审批时没有任何操作、时钟也不走，因此等待不进预算；
恢复后沿用已累计的 `active_seconds`，不重置。`clock_open` 记录时钟是否在走：
没有它就只能用"距上次操作多久"并封顶来近似，等待越久误差越大。

## 模型请求的判定与计数

`max_llm_calls` 的口径是**真实上游请求数**，不是 Agent 逻辑调用次数：
一次逻辑调用如果因为结构化输出解析失败而在内部重试，那几次请求都要计数、
也要受上限约束。因此判定与计数接在 `chain.invoke` 之前
（见 `utils/llm_retry` 的 `before_request` / `on_request` 回调）。

判定位置有个硬要求：必须在**幂等复用之后、账本写 `started` 之前**。
复用不该消耗额度；而超预算时若已经写了 `started`，就会留下一条永远等不到
`completed` 的记录，恢复时被当成"未确认的模型调用"要求人工裁决——
一个纯粹的预算问题会变成人工工单。

## 与既有机制的关系

- 不新建幂等或重试系统：逻辑尝试 ID 仍来自 `utils/idempotency`；
- 不取代 `MAX_RETRIES` 等局部上限：它们是更细的闸门，两者同时生效；
- 失败是**收敛**而不是崩溃：超预算时抛出 `BudgetExceeded`，
  节点包装器捕获后转成正常的失败记录，让工作流按既有路径收尾。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

# state 里保存预算快照的字段名
BUDGET_FIELD = "execution_budget"

# 预算收敛失败使用的错误分类。归到已有的 `unavailable` 语义下：
# 预算耗尽不是"代码写错了"，重试同样的代码不会好转，
# 因此不该进入 Repair——这与 P2-B 对环境类错误的处理一致。
BUDGET_ERROR_KIND = "unavailable"

#: 两次真实操作之间最多计入的墙上时钟。
#:
#: 时间预算需要区分"在执行"与"在等人"。做法是：只在**真实操作发生时**
#: 把距上次操作的时间累加进 `active_seconds`，且单次间隔不超过这个上限。
#: 等待人工审批期间没有任何真实操作，因此不计入；而在执行期间，
#: 操作（工具调用、模型调用、轮次）会持续发生，间隔远小于该上限。
#:
#: 取值依据：它必须明显大于单次工具/模型调用的正常耗时（否则正常执行会被少计），
#: 又要明显小于人工审批的等待时间（否则等待会被误计）。60 秒对两者都有余量。
MAX_ACTIVE_GAP_SECONDS = 60.0


class BudgetExceeded(RuntimeError):
    """某一项执行预算已经用尽。

    带 `scope` 便于上层区分是哪一类预算（rounds / tool_calls / llm_calls / seconds）。
    """

    def __init__(self, scope: str, used: float, limit: float, message: str | None = None):
        self.scope = scope
        self.used = used
        self.limit = limit
        super().__init__(
            message
            or f"执行预算已用尽（{scope}）：已用 {used} / 上限 {limit}"
        )


@dataclass(frozen=True)
class AgentBudget:
    """一个 Run 的有界执行预算。

    所有上限都是**总量**，不是每轮的量。默认值取得足够宽，使既有工作流
    在正常路径下完全不受影响；它的作用是兜住"合法但无休止"的重试组合。
    """

    max_rounds: int = 24
    """Reason→Act→Observe→Decide 的总轮次上限。

    默认 24：既有的最坏合法路径是 1 次执行 + 3 次 repair + 3 次 refine，
    再加上图自身的节点步进，远低于这个数；而一旦出现互相触发的组合爆炸，
    这里会先于 recursion_limit 收敛住。
    """

    max_tool_calls: int = 40
    """工具调用总次数上限（含 python / plot / sql）。"""

    max_llm_calls: int = 60
    """模型调用总次数上限。

    与 `utils/llm_ledger` 的分工：账本记录"每次调用是否确认完成"，
    这里只负责"总量是否超了"。两者都不替代对方。
    """

    max_seconds: float = 1800.0
    """墙上时钟预算（秒）。默认 30 分钟。"""

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_rounds": self.max_rounds,
            "max_tool_calls": self.max_tool_calls,
            "max_llm_calls": self.max_llm_calls,
            "max_seconds": self.max_seconds,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "AgentBudget":
        """从 state 里恢复预算上限。

        只认已知字段，缺失或类型不对时回落到默认值：预算配置是宿主决定的，
        不该因为 state 里少一个键就让整次运行失败。

        **0 是合法上限**（"一次都不许"），必须与"字段缺失"区分开。
        早期实现用 `value <= 0` 过滤，于是 `max_tool_calls=0` 被当成缺失、
        回落到默认的 40——一个声明"禁止工具调用"的 Run 反而拿到了 40 次额度。
        这类错误很隐蔽：限制看起来生效了，实际被悄悄放宽。
        """
        if not payload:
            return cls()

        known: dict[str, Any] = {}
        for field_name in ("max_rounds", "max_tool_calls", "max_llm_calls", "max_seconds"):
            value = payload.get(field_name)
            # bool 是 int 的子类，必须显式排除，否则 True 会变成上限 1
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if value < 0:
                continue
            known[field_name] = (
                int(value) if field_name != "max_seconds" else float(value)
            )
        return cls(**known)


def new_usage() -> dict[str, Any]:
    """一份全新的用量记录。"""
    return {
        "rounds": 0,
        "tool_calls": 0,
        # **真实上游模型请求数**（含结构化输出解析失败导致的内部重试），
        # 不是 Agent 逻辑调用次数：一次逻辑调用可能发出多次请求。
        # 字段名沿用 llm_calls，但语义以这里为准。
        "llm_calls": 0,
        # 首次开始时刻：仅用于展示与迁移，**不再**直接决定时间预算。
        "started_at": None,
        # 累计"正在执行"的秒数。等待人工审批的时间不计入。
        "active_seconds": 0.0,
        # 上一次真实操作的时刻，用于把两段执行之间的时间累加进 active_seconds。
        "last_active_at": None,
        # 执行时钟是否**正在走**：一次真实操作开始时置 True，
        # 该操作结束时（settle）置 False。
        #
        # 这个标志是"等待不计入"的关键。没有它就只能用"距上次操作多久"
        # 加一段带上限的尾巴，于是**停在等人时也会缓慢累积**（每次 +60 秒上限），
        # 等待越久越接近上限——只是把误差从"精确计入"变成"封顶计入"。
        # 有了它，时钟在结算后就真的停住，等待期间 elapsed 保持不变。
        "clock_open": False,
    }


def usage_of(state: dict[str, Any]) -> dict[str, Any]:
    """读出 state 里的用量；没有就返回一份新的（不写回）。"""
    payload = state.get(BUDGET_FIELD)
    if not isinstance(payload, dict):
        return new_usage()
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return new_usage()
    merged = new_usage()
    merged.update(
        {key: value for key, value in usage.items() if key in merged}
    )
    return merged


def budget_of(state: dict[str, Any]) -> AgentBudget:
    payload = state.get(BUDGET_FIELD)
    if isinstance(payload, dict):
        return AgentBudget.from_dict(payload.get("limits"))
    return AgentBudget()


def _wait_started_at(state: dict[str, Any]) -> float | None:
    """老 Checkpoint 里"开始等待人工审批"的时刻，取不到返回 None。

    `pending_approval` 是审批请求本身（在 `interrupt` 之前写入），它带
    `created_at`，就是执行停下来等人工的那一刻。迁移时用它可以只折算
    "等待之前真正执行过的那段"，而不把等待算进执行预算。

    审批完成后 `pending_approval` 被清空，归档记录里只有 `decided_at`
    （没有请求时刻），因此**无法**再区分等待与执行——那种情况下返回 None，
    由调用方走保守策略。
    """
    pending = state.get("pending_approval")
    if isinstance(pending, dict):
        created = pending.get("created_at")
        if isinstance(created, (int, float)):
            return float(created)
    return None


def initialize(state: dict[str, Any]) -> dict[str, Any]:
    """确保 state 里有一份预算快照，并在首次调用时打上开始时刻。

    返回要写回 state 的增量。已经存在时不重置任何计数——这正是
    "恢复后不重置预算"的实现点。

    **老 Checkpoint 的迁移**：P3 早期只持久化 `started_at`，没有
    `active_seconds`，当时的"已消耗时间"是 `now - started_at`，也就是把
    人工审批的等待一并算了进去。迁移规则：

    - 能证明当时正在等待审批（`pending_approval.created_at` 可用）时，
      只折算 `created_at - started_at`，等待时间不计入执行预算；
    - 否则按 `now - started_at` 折算，但以 `max_seconds` 封顶。

    这是**保守且可解释**的策略，不是精确重建：老格式没有记录执行与等待的
    分界，无法还原真实执行时间。封顶的意义是避免"等待了两小时的老 Run，
    预算只有 1800 秒，一恢复就立刻失败"——那会让等待时间变成一次性的额度
    清零。已消耗的时间不会被退回，已归档的审批记录也不改写。
    """
    payload = state.get(BUDGET_FIELD)

    if not isinstance(payload, dict):
        return {
            BUDGET_FIELD: {
                "limits": AgentBudget().to_dict(),
                "usage": {
                    **new_usage(),
                    "started_at": time.time(),
                    "last_active_at": time.time(),
                },
            }
        }

    usage = usage_of(state)
    limits = AgentBudget.from_dict(payload.get("limits"))
    changed = False

    if usage.get("started_at") is None:
        # 极老的 Checkpoint 连开始时刻都没有，现场补一个
        usage["started_at"] = time.time()
        changed = True

    if payload.get("usage", {}).get("active_seconds") is None:
        # 迁移：把旧的"已用时间"折算一次，之后不再用墙钟
        started = usage.get("started_at")
        carried = 0.0
        if isinstance(started, (int, float)):
            started = float(started)
            wait_started = _wait_started_at(state)
            if wait_started is not None and wait_started >= started:
                # 等待之前的那段才是执行时间
                carried = wait_started - started
            else:
                carried = max(0.0, time.time() - started)
            carried = min(carried, limits.max_seconds)
        usage["active_seconds"] = max(0.0, carried)
        changed = True

    if usage.get("last_active_at") is None:
        usage["last_active_at"] = time.time()
        changed = True

    if not changed:
        return {}

    return {
        BUDGET_FIELD: {
            "limits": limits.to_dict(),
            "usage": usage,
        }
    }


def _accumulate(
    state: dict[str, Any], *, now: float, open_clock: bool
) -> dict[str, Any]:
    """把"距上次操作"的时间累加进 `active_seconds`，并设置时钟状态。

    只有在时钟**正在走**（上一次操作尚未结算）时才累加：

    - 一次操作开始时（capture）：上一笔的尾巴先算清，时钟继续走；
    - 一次操作结束时（settle）：当前这笔算清，时钟停住。
    """
    usage = usage_of(state)
    active = usage.get("active_seconds")
    active = float(active) if isinstance(active, (int, float)) else 0.0

    last = usage.get("last_active_at")
    if usage.get("clock_open") and isinstance(last, (int, float)):
        active += min(MAX_ACTIVE_GAP_SECONDS, max(0.0, now - float(last)))

    usage["active_seconds"] = active
    usage["last_active_at"] = now
    usage["clock_open"] = open_clock
    if usage.get("started_at") is None:
        usage["started_at"] = now

    payload = state.get(BUDGET_FIELD)
    limits = (
        AgentBudget.from_dict(payload.get("limits")).to_dict()
        if isinstance(payload, dict)
        else AgentBudget().to_dict()
    )
    return {BUDGET_FIELD: {"limits": limits, "usage": usage}}


def capture_active_time(
    state: dict[str, Any], *, now: float | None = None
) -> dict[str, Any]:
    """在**开始**一次真实操作时推进执行时钟（时钟保持走动）。

    调用时机是真实操作发生时（工具调用、模型请求、轮次推进）。这些操作
    分散在若干处，因此"时间前进"与"工作量前进"绑定在一起，不需要额外的
    生命周期钩子；等待人工审批期间没有任何操作，时间自然就不走。
    """
    return _accumulate(
        state, now=now if now is not None else time.time(), open_clock=True
    )


def settle_active_time(
    state: dict[str, Any], *, now: float | None = None
) -> dict[str, Any]:
    """在**结束**一段工作时结算执行时钟，并让它停住。

    调用点是一次工具调用或一次模型调用返回之后。之后若进入人工审批等待，
    `elapsed_seconds` 不再增长——这正是"等待不计入有效执行预算"的实现点。

    没有这个函数时只能靠"距上次操作多久"并封顶来近似，于是**停在等人时
    时间仍会缓慢累积**（每次都加上限），等待越久误差越大。时钟状态显式化
    之后，等待期间已消耗时间保持不变。
    """
    return _accumulate(
        state, now=now if now is not None else time.time(), open_clock=False
    )


def elapsed_seconds(state: dict[str, Any], *, now: float | None = None) -> float:
    """已消耗的**执行**时间（不含等待人工审批的时间）。

    时钟停住时就是 `active_seconds` 本身；时钟正在走时再加上"距上次操作"
    这一小段（一次操作内部的耗时）。两种情况下等待时间都不被计入。
    """
    usage = usage_of(state)
    active = usage.get("active_seconds")
    active = float(active) if isinstance(active, (int, float)) else 0.0

    last = usage.get("last_active_at")
    if usage.get("clock_open") and isinstance(last, (int, float)):
        current = now if now is not None else time.time()
        active += min(
            MAX_ACTIVE_GAP_SECONDS, max(0.0, current - float(last))
        )

    return active


#: `check` 能判定的预算范围。
_ALL_SCOPES = ("rounds", "tool_calls", "llm_calls", "seconds")


def check(
    state: dict[str, Any],
    *,
    now: float | None = None,
    scopes: tuple[str, ...] = _ALL_SCOPES,
) -> None:
    """预算是否还能继续。超了就抛 `BudgetExceeded`。

    只读，不改 state：判定与计数分开，调用方先判定再决定要不要计一次。

    `scopes` 用来只看某几类预算。真实用途是：**进入一次模型请求前**只看
    `llm_calls`——那次请求只会再消耗一个模型调用额度，不该因为工具调用或
    时间预算刚好用尽就拒绝它。工具与轮次的入口仍然检查全部分项。
    """
    budget = budget_of(state)
    usage = usage_of(state)

    limits = {
        "rounds": (usage["rounds"], budget.max_rounds),
        "tool_calls": (usage["tool_calls"], budget.max_tool_calls),
        "llm_calls": (usage["llm_calls"], budget.max_llm_calls),
        "seconds": (None, budget.max_seconds),
    }

    for scope in scopes:
        used, limit = limits[scope]
        if scope == "seconds":
            spent = elapsed_seconds(state, now=now)
            if spent >= limit:
                raise BudgetExceeded("seconds", round(spent, 1), limit)
            continue
        if used >= limit:
            raise BudgetExceeded(scope, used, limit)


def check_llm_budget(state: dict[str, Any]) -> None:
    """进入一次**真实模型请求**前的判定。

    只看 `llm_calls`：这一次请求额外消耗的是一个模型调用额度，不该因为
    工具调用或时间预算刚好用尽而被拒绝——那些分项在各自的入口判定
    （`governed_tool_call` 与轮次入口走完整的 `check`）。

    计数的写入点在每次请求**发出之前**，因此同一次逻辑调用内部的重试
    读到的就是"已经发过几次"，这里只比较、不再额外加一。
    """
    budget = budget_of(state)
    used = usage_of(state)["llm_calls"]
    if used >= budget.max_llm_calls:
        raise BudgetExceeded("llm_calls", used, budget.max_llm_calls)


def _bump(state: dict[str, Any], key: str) -> dict[str, Any]:
    """记一次真实操作，并顺带推进执行时钟。

    "真实操作"就是消耗预算的那些事：轮次、工具调用、模型请求。
    在这里累加 `active_seconds` 有一个好处——**时间前进与工作量前进绑定**，
    不需要额外的生命周期钩子；停下来等人审批时没有任何操作，时间自然停住。
    """
    payload = state.get(BUDGET_FIELD)
    if not isinstance(payload, dict):
        payload = {
            "limits": AgentBudget().to_dict(),
            "usage": new_usage(),
        }

    # 先推进时钟，再递增计数，两者合并成同一份增量返回
    timed = capture_active_time(state)
    usage = timed[BUDGET_FIELD]["usage"]
    usage[key] = int(usage.get(key, 0)) + 1

    return {
        BUDGET_FIELD: {
            "limits": AgentBudget.from_dict(payload.get("limits")).to_dict(),
            "usage": usage,
        }
    }


def consume_round(state: dict[str, Any]) -> dict[str, Any]:
    """记一次 Reason→Act→Observe→Decide 轮次。"""
    return _bump(state, "rounds")


def consume_tool_call(state: dict[str, Any]) -> dict[str, Any]:
    return _bump(state, "tool_calls")


def consume_llm_call(state: dict[str, Any]) -> dict[str, Any]:
    """记一次**真实上游模型请求**。

    保留这个名字是因为既有调用方（节点与测试）都按它理解，
    语义见 `new_usage()["llm_calls"]` 的说明：真实请求，不是逻辑调用。
    """
    return _bump(state, "llm_calls")


# ---------------------------------------------------------------------------
# 模型请求钩子：让 Agent 层能在**每次真实请求**前后判预算与计数
# ---------------------------------------------------------------------------

#: 请求发出前的判定钩子。由 Harness 注册（它知道如何取当前 state）。
#: 用注册表而不是让 Agent 直接 import 节点模块，是为了避免
#: Agent → nodes → Agent 的循环依赖。
_before_request_hook: Any = None
_on_request_hook: Any = None

#: 取当前 Run 的 state 的函数，同样由 Harness 注册。
_state_provider: Any = None


def set_state_provider(provider: Any = None) -> None:
    """注册"取当前 state"的函数。

    节点层用一个 `ContextVar` 保存正在执行的 state，这里的钩子通过它取值，
    从而让 `utils` 层不必知道节点模块的存在。
    """
    global _state_provider
    _state_provider = provider


def set_request_hooks(
    *,
    before: Any = None,
    after: Any = None,
) -> None:
    """注册模型请求前后的钩子（测试可用于隔离或观测）。

    传 `None` 表示清除。清成 `None` 之后 `before_model_request()` 变成空操作——
    这正是"节点层不在场"时应有的行为，例如单独调用某个 Agent 的单元测试。
    """
    global _before_request_hook, _on_request_hook
    _before_request_hook = before
    _on_request_hook = after


def before_model_request(attempt: int) -> None:
    """在一次**真实上游模型请求**发出前判定预算。

    **每一次尝试都要判**，包括第 1 次。

    这里曾经跳过第 1 次（理由是"`run_agent_call` 的 `before_request` 已经判过"），
    那是错的：`run_agent_call` 只在**逻辑调用开始时**判一次，而同一逻辑调用内部
    的解析失败重试会让计数在操作过程中增长。跳过第 1 次意味着 `after_model_request`
    计数、但第 2 次尝试的判定发生在**上一次计数之前**——实测结果是声明上限 3
    却发出了 4 次请求。

    `check_llm_budget` 只看 `llm_calls`，不涉及轮次/工具/时间，因此第 1 次重复判定
    没有副作用；`run_agent_call` 里那次判定保留，是为了在预算已尽时**不往账本写
    `started`**（那会留下永远等不到 `completed` 的记录）。
    """
    hook = _before_request_hook
    if hook is None:
        return
    hook(attempt)


def after_model_request(attempt: int) -> None:
    """在一次真实请求**将要发出**时计数。

    调用点在 `chain.invoke` 之前（见各 Agent 的重试辅助函数）：这样即使请求
    本身失败也已经计入，不会给"反复失败重发"留出绕过上限的空间。
    统计口径是**真实请求数**，一次逻辑调用内部重试的每一次请求都各记一次。
    """
    hook = _on_request_hook
    if hook is None:
        return
    hook(attempt)


def snapshot(state: dict[str, Any], *, now: float | None = None) -> dict[str, Any]:
    """当前预算与用量，用于日志与可观测性。"""
    budget = budget_of(state)
    usage = usage_of(state)
    return {
        "limits": budget.to_dict(),
        "usage": {**usage, "seconds": round(elapsed_seconds(state, now=now), 3)},
        "remaining": {
            "rounds": max(0, budget.max_rounds - usage["rounds"]),
            "tool_calls": max(0, budget.max_tool_calls - usage["tool_calls"]),
            "llm_calls": max(0, budget.max_llm_calls - usage["llm_calls"]),
            "seconds": max(
                0.0, round(budget.max_seconds - elapsed_seconds(state, now=now), 3)
            ),
        },
    }
