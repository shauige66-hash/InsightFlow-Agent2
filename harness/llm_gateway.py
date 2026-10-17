"""模型调用的唯一出口。

一次逻辑调用要同时满足四件事，顺序不能换：

1. **幂等**：同一逻辑调用已完成就直接复用，不再发请求；
2. **预算**：判定放在复用之后、账本 `started` 之前——复用不该消耗额度，
   而超预算时绝不能留下一条永远等不到 `completed` 的记录，否则恢复时会被
   判成"未确认的模型调用"并要求人工裁决，纯粹的预算问题变成人工工单；
3. **账本**：`started` / `completed` 由 `utils.llm_ledger` 记录，这里只驱动；
4. **计数**：按**真实请求**计数，包含同一次逻辑调用内部的解析重试。

第 4 点容易做错。计数只在这里的 `on_request` 钩子里做一次，判定只读同一份
用量；两边都加一次会把当前请求算两遍，结果是上限 2 时只发出 1 次请求就被
拒绝（见 `tests/test_llm_request_budget_discriminating.py`）。

每个 Agent 都是通过 `utils.llm_retry` 的 `before_request` / `on_request`
回调进入这两个钩子的，因此内部重试同样受约束。唯一绕过 `llm_retry` 的是
plotter 自带的重试循环，它显式调用同样的钩子。
"""

from __future__ import annotations

from typing import Any, Callable

from harness import policy
from utils.agent_loop import BUDGET_FIELD, set_request_hooks, set_state_provider
from utils.context_budget import context_report_of
from utils.idempotency import run_agent_call
from utils.run_logger import log_event


class StateNotBound(RuntimeError):
    """模型调用前没有绑定当前 Run 的 state。

    没有 state 就读不到这次 Run 的预算快照与用量，判定会落到默认额度上——
    等于放过了一个未被该 Run 治理的请求。这种情况应当明确失败，而不是拿空
    字典继续。
    """


class LLMGateway:
    """编排幂等、账本、预算与追踪。

    `state_reader` 由 `harness.runtime` 注入，读的是按执行上下文隔离的
    `ContextVar`，因此并发 Run 各自看到自己的 state。
    """

    def __init__(self, state_reader: Callable[[], dict[str, Any] | None]) -> None:
        self._state_reader = state_reader

    def install_request_hooks(self) -> None:
        """把预算判定与计数接到 `utils.agent_loop` 的请求钩子上。

        钩子是进程级的（`utils.agent_loop` 只有一组槽位），但回调读的是
        上下文里的 state，所以同一进程内并发 Run 互不影响。重复安装是
        幂等的：与上次写入相同的可调用对象。
        """
        set_state_provider(self._state_reader)
        set_request_hooks(before=self._before_request, after=self._after_request)

    def budget_guard(self) -> None:
        """一次真实模型请求发出前的判定。

        只判不记。`run_agent_call` 在落账本之前调一次，`utils.llm_retry`
        在每次重发之前再调一次，两次都只读。
        """
        policy.ensure_model_request_allowed(self._current_state())

    def call_agent(
        self,
        agent: str,
        subject: str,
        invoke: Callable[[], Any],
        *,
        attempt: int = 1,
    ) -> tuple[Any, dict, dict]:
        """执行一次 Agent 逻辑调用。

        返回 `(结果, 幂等登记, state 增量)`。第三个返回值必须由节点一并写回，
        否则"已完成"没落盘会导致恢复时重复执行，"未确认清单"没清空会导致
        每次恢复都重复询问同一件事。
        """
        state = self._current_state()
        payload, operations, extra = run_agent_call(
            state,
            node=agent,
            subject=subject,
            call=invoke,
            attempt=attempt,
            before_request=self.budget_guard,
        )

        extra = dict(extra or {})
        invoked = bool(extra.pop("__upstream_invoked__", False))
        # 与账本、幂等登记用的是同一个操作 ID，日志带上它才能把用量回溯到
        # 具体调用。它只用于追踪，不进 state。
        operation_id = extra.pop("__operation_id__", None)

        if invoked:
            # 计数已经在每个请求发出前由钩子完成，这里不重复计。
            # 只结算执行时钟：模型调用结束后若去等人审批，时间不再累积。
            settled = policy.settle(state)
            extra[BUDGET_FIELD] = settled[BUDGET_FIELD]
            self._log_context(agent, subject, operation_id, state)

        return payload, operations, extra

    def _log_context(
        self,
        agent: str,
        subject: str,
        operation_id: Any,
        state: dict[str, Any],
    ) -> None:
        """记录该 Agent 本次的上下文裁剪度量。

        度量由各 Agent 在裁剪时发布到按执行上下文隔离的变量里，这里按 agent
        名取回。取不到就不写日志（不因为缺字段影响执行）。
        """
        report = context_report_of(agent)
        if not report:
            return
        log_event(
            run_id=state.get("run_id") or "unknown",
            node=agent,
            event="llm_context",
            status="ok",
            details={
                "subject": subject,
                "operation_id": operation_id,
                "context": report,
            },
        )

    # ------------------------------------------------------------------
    # 请求钩子
    # ------------------------------------------------------------------

    def _before_request(self, attempt: int) -> None:
        # 不在节点执行上下文里（单独调用某个 Agent）时钩子是空操作：
        # 那里没有本次 Run 的预算，也不该由这个钩子决定能不能跑。
        # 真正被治理的入口是 `call_agent`，它在开始前就要求已绑定 state。
        if self._state_reader() is None:
            return
        self.budget_guard()

    def _after_request(self, attempt: int) -> None:
        state = self._state_reader()
        if state is None:
            return
        policy.count_model_request(state)

    def _current_state(self) -> dict[str, Any]:
        """当前 Run 的 state；未绑定就拒绝。

        这个检查守住的是治理本身：没有 state 就没有预算，请求会在默认额度下
        被放行。正常节点执行由 `_with_state` 的 `bind_state` 绑定，并发 Run
        各自持有自己的上下文。
        """
        state = self._state_reader()
        if state is None:
            raise StateNotBound(
                "模型调用必须在 bind_state 绑定的执行上下文里发起；"
                "未绑定 state 时无法判定该 Run 的预算"
            )
        return state


__all__ = ["LLMGateway", "StateNotBound"]
