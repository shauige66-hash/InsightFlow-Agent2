"""Run 状态变化的对外通知。

放在服务层而不是存储层：`RunStore.update_status` 只负责把状态写进库，通知是
"业务上有人关心"，两者混在一起会让存储层依赖 HTTP。

调用约定：所有函数都不抛异常。通知是旁路，失败只在日志里留痕。
"""

from __future__ import annotations

import logging
import os
from typing import Any

from integrations.feishu import FeishuWebhook, build_approval_card, build_run_card

_LOGGER = logging.getLogger(__name__)


def _notifier() -> FeishuWebhook | None:
    """每次调用重新读环境变量，便于测试与运行时开关。"""
    try:
        return FeishuWebhook.from_env()
    except Exception as error:  # noqa: BLE001
        _LOGGER.warning("构造飞书通知器失败：%s", error)
        return None


def _usage(result: dict[str, Any] | None) -> dict[str, Any]:
    if not result:
        return {}
    budget = result.get("execution_budget") or {}
    return budget.get("usage") or {}


def _console_url() -> str | None:
    """可选：卡片里附一个控制台链接。

    `127.0.0.1` 的地址只在跑服务这台机器上有效，分享给别人的群没有意义，
    所以默认不带；本机自用时在 `.env` 里配上即可直接点开。
    """
    url = os.getenv("FEISHU_CONSOLE_URL", "").strip()
    return url or None


def notify_run_finished(
    run_id: str,
    status: str,
    *,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> bool:
    """Run 结束（completed / partial / failed）时推送。"""
    notifier = _notifier()
    if notifier is None:
        return False

    usage = _usage(result)
    history = (result or {}).get("decision_history") or []
    try:
        card = build_run_card(
            run_id,
            status,
            analysis_status=(result or {}).get("analysis_status"),
            analysis_focus=(result or {}).get("analysis_focus"),
            decision_count=len(history) if result is not None else None,
            llm_calls=usage.get("llm_calls"),
            tool_calls=usage.get("tool_calls"),
            active_seconds=usage.get("active_seconds"),
            error=error,
            console_url=_console_url(),
        )
        return notifier.send_card(card)
    except Exception as failure:  # noqa: BLE001
        _LOGGER.warning("组装飞书卡片失败：%s", failure)
        return False


def notify_approval_needed(
    run_id: str,
    pending: dict[str, Any] | None,
) -> bool:
    """需要人工确认时推送。`pending` 是 checkpoint 里的 interrupt 载荷。"""
    notifier = _notifier()
    if notifier is None:
        return False

    pending = pending or {}
    try:
        card = build_approval_card(
            run_id,
            kind=pending.get("kind"),
            summary=pending.get("summary"),
            console_url=_console_url(),
        )
        return notifier.send_card(card)
    except Exception as failure:  # noqa: BLE001
        _LOGGER.warning("组装审批卡片失败：%s", failure)
        return False
