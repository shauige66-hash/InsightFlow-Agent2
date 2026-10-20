"""飞书群自定义机器人：把 Run 的状态变化推成卡片。

只做单向推送。群机器人是飞书里门槛最低的接入方式——一个 webhook 地址即可，
不需要企业自建应用、不需要公网回调地址。代价是卡片上的按钮没有回调，所以
待审批卡片只提示"回控制台处理"，不放假的批准按钮。

两条硬约束：

1. **通知失败绝不能让 Run 失败。** 所有异常都在这里吞掉并记日志；一次推送
   失败只是少了一条消息，不能让已经跑完的分析变成 failed。
2. **没配置就不发。** 没有 `FEISHU_WEBHOOK_URL` 时 `FeishuWebhook.from_env()`
   返回 None，调用方按"功能关闭"处理，不需要在业务代码里判断环境。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import time
from typing import Any

import httpx

_LOGGER = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 5.0

STATUS_TEXT = {
    "completed": "分析完成",
    "partial": "部分完成",
    "failed": "分析失败",
    "waiting_approval": "等待审批",
    "rejected": "已拒绝",
    "running": "执行中",
}

# 飞书卡片的 header 颜色
STATUS_TEMPLATE = {
    "completed": "green",
    "partial": "orange",
    "failed": "red",
    "waiting_approval": "orange",
    "rejected": "grey",
}


def _sign(secret: str, timestamp: int) -> str:
    """飞书自定义机器人的签名：以 `timestamp\\nsecret` 为密钥做 HMAC-SHA256。

    这是飞书文档规定的算法，与常见"用 secret 当密钥"的写法不同，照抄很容易错。
    """
    string_to_sign = f"{timestamp}\n{secret}"
    digest = hmac.new(
        string_to_sign.encode("utf-8"), digestmod=hashlib.sha256
    ).digest()
    return base64.b64encode(digest).decode("utf-8")


def _div(content: str) -> dict[str, Any]:
    return {"tag": "div", "text": {"tag": "lark_md", "content": content}}


def _safe(value: Any, default: str = "-") -> str:
    if value is None or value == "":
        return default
    return str(value)


def build_run_card(
    run_id: str,
    status: str,
    *,
    analysis_status: str | None = None,
    analysis_focus: str | None = None,
    decision_count: int | None = None,
    llm_calls: int | None = None,
    tool_calls: int | None = None,
    active_seconds: float | None = None,
    error: str | None = None,
    console_url: str | None = None,
) -> dict[str, Any]:
    """一次运行结束（或进入待审批）时的卡片。

    注意：飞书的 `lark_md` 只支持 `**加粗**` 与 `[文字](链接)`，**不支持**用反引号
    标注行内代码——写了会原样显示反引号，所以这里不用反引号包 run_id。
    """
    title = STATUS_TEXT.get(status, status)
    if status == "partial":
        title = "分析完成（部分任务未完成）"

    lines = [
        f"**Run**：{run_id}",
        f"**分析结果**：{_safe(analysis_status, title)}",
    ]
    if analysis_focus:
        lines.append(f"**分析问题**：{analysis_focus}")
    if decision_count is not None:
        lines.append(
            "**执行**："
            f"决策 {decision_count} 次 · 模型请求 {_safe(llm_calls, '0')} 次 · "
            f"工具调用 {_safe(tool_calls, '0')} 次 · 耗时 "
            f"{'-' if active_seconds is None else round(active_seconds)} 秒"
        )
    if error:
        lines.append(f"**失败原因**：{error}")

    elements: list[dict[str, Any]] = [_div("\n".join(lines))]
    if console_url:
        # 127.0.0.1 的地址只在运行服务的这台机器上打得开；飞书桌面端会调本机浏览器，
        # 所以自用场景下这个链接可用，分享给别人的群则无意义。
        elements.append(_div(f"[打开控制台]({console_url})"))
    elif status in ("completed", "partial"):
        elements.append(
            {
                "tag": "note",
                "elements": [
                    {
                        "tag": "plain_text",
                        "content": "报告与完整状态在本地控制台查看（127.0.0.1 仅本机可访问）",
                    }
                ],
            }
        )

    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": f"InsightFlow · {title}"},
            "template": STATUS_TEMPLATE.get(status, "blue"),
        },
        "elements": elements,
    }


def build_approval_card(
    run_id: str,
    *,
    kind: str | None = None,
    summary: Any = None,
    console_url: str | None = None,
) -> dict[str, Any]:
    """待审批卡片：说明需要确认什么，并提示去哪里处理。

    群机器人没有回调能力，所以这里不放"批准"按钮——放一个点不动的按钮比不放
    更糟，用户会以为系统坏了。
    """
    label = "报告发布" if kind == "report_publish" else "计划执行"
    lines = [f"**Run**：{run_id}", f"**待确认**：{label}"]

    tasks = []
    if isinstance(summary, dict):
        tasks = summary.get("tasks") or []
    if tasks:
        lines.append(f"**计划任务**（{len(tasks)} 个）：")
        for task in tasks[:10]:
            tables = "、".join(task.get("target_tables") or [])
            description = task.get("description") or task.get("task_id") or "-"
            lines.append(f"- {description}" + (f"（{tables}）" if tables else ""))
        if isinstance(summary, dict) and summary.get("contains_sql_tasks"):
            lines.append("_含 SQL 任务，执行侧强制只读_")

    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": f"InsightFlow · 需要你确认{label}"},
            "template": "orange",
        },
        "elements": [
            _div("\n".join(lines)),
            _div(f"[打开控制台处理]({console_url})")
            if console_url
            else {
                "tag": "note",
                "elements": [
                    {
                        "tag": "plain_text",
                        "content": "请到控制台（/ui/）批准或拒绝；本机器人只负责提醒。",
                    }
                ],
            },
        ],
    }


class FeishuWebhook:
    """一个群机器人地址。构造后不可变，实例可以被多线程共享。"""

    def __init__(
        self,
        url: str,
        *,
        sign_secret: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._url = url
        self._sign_secret = sign_secret or None
        self._timeout = timeout

    @classmethod
    def from_env(cls) -> "FeishuWebhook | None":
        url = os.getenv("FEISHU_WEBHOOK_URL", "").strip()
        if not url:
            return None
        return cls(url, sign_secret=os.getenv("FEISHU_SIGN_SECRET", "").strip() or None)

    @property
    def url(self) -> str:
        return self._url

    def _payload(self, card: dict[str, Any]) -> dict[str, Any]:
        body: dict[str, Any] = {"msg_type": "interactive", "card": card}
        if self._sign_secret:
            timestamp = int(time.time())
            body["timestamp"] = str(timestamp)
            body["sign"] = _sign(self._sign_secret, timestamp)
        return body

    def send_card(self, card: dict[str, Any]) -> bool:
        """发一张卡片。返回是否成功，**从不抛异常**。"""
        body = self._payload(card)
        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(self._url, json=body)
            if response.status_code >= 400:
                _LOGGER.warning(
                    "飞书推送失败：HTTP %s %s", response.status_code, response.text[:200]
                )
                return False
            # 飞书即使 HTTP 200 也可能返回 {"code": 非0}
            try:
                data = response.json()
            except ValueError:
                return True
            code = data.get("code") if isinstance(data, dict) else None
            if code not in (None, 0):
                _LOGGER.warning("飞书推送被拒绝：code=%s msg=%s", code, data.get("msg"))
                return False
            return True
        except Exception as error:  # noqa: BLE001 - 通知失败不能影响业务
            _LOGGER.warning("飞书推送异常：%s: %s", type(error).__name__, error)
            return False
