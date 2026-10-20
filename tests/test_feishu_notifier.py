"""飞书群机器人推送：卡片内容、签名与"失败不影响 Run"。

用本地假 webhook 取代真实飞书：断言发出去的 JSON，而不是断言"调用了某个函数"。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import services.run_coordinator as run_coordinator
from integrations import feishu
from services.run_notifier import notify_approval_needed, notify_run_finished


class _WebhookHandler(BaseHTTPRequestHandler):
    received: list[dict] = []
    status_code = 200
    reply: dict = {"code": 0, "msg": "success"}

    def do_POST(self) -> None:  # noqa: N802 - http.server 约定
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        type(self).received.append(json.loads(body.decode("utf-8")))
        payload = json.dumps(self.reply).encode("utf-8")
        self.send_response(type(self).status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args: object) -> None:
        return


@pytest.fixture()
def webhook(monkeypatch):
    _WebhookHandler.received = []
    _WebhookHandler.status_code = 200
    _WebhookHandler.reply = {"code": 0, "msg": "success"}

    server = ThreadingHTTPServer(("127.0.0.1", 0), _WebhookHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}/open-apis/bot/v2/hook/test"

    monkeypatch.setenv("FEISHU_WEBHOOK_URL", url)
    monkeypatch.delenv("FEISHU_SIGN_SECRET", raising=False)
    try:
        yield _WebhookHandler
    finally:
        server.shutdown()
        server.server_close()


def _card_of(payload: dict) -> dict:
    assert payload["msg_type"] == "interactive"
    return payload["card"]


def test_a_finished_run_pushes_a_card_with_its_numbers(webhook) -> None:
    sent = notify_run_finished(
        "run_1",
        "completed",
        result={
            "analysis_status": "completed",
            "analysis_focus": "各地区高价值客户贡献",
            "decision_history": [{"step": 1}, {"step": 2}],
            "execution_budget": {"usage": {"llm_calls": 6, "tool_calls": 2, "active_seconds": 6.4}},
        },
    )

    assert sent is True
    card = _card_of(webhook.received[0])
    assert "分析完成" in card["header"]["title"]["content"]
    body = card["elements"][0]["text"]["content"]
    assert "run_1" in body
    assert "决策 2 次" in body
    assert "模型请求 6 次" in body
    assert "工具调用 2 次" in body


def test_a_failed_run_carries_the_reason(webhook) -> None:
    notify_run_finished("run_2", "failed", error="DEEPSEEK_API_KEY is missing")

    card = _card_of(webhook.received[0])
    assert card["header"]["template"] == "red"
    assert "DEEPSEEK_API_KEY is missing" in card["elements"][0]["text"]["content"]


def test_an_approval_notice_lists_the_tasks(webhook) -> None:
    notify_approval_needed(
        "run_3",
        {
            "kind": "plan_review",
            "summary": {
                "tasks": [
                    {"task_id": "t1", "description": "统计客户总数", "target_tables": ["customers"]},
                    {"task_id": "t2", "description": "按地区汇总", "target_tables": ["orders"]},
                ],
                "contains_sql_tasks": True,
            },
        },
    )

    card = _card_of(webhook.received[0])
    body = card["elements"][0]["text"]["content"]
    assert "计划执行" in card["header"]["title"]["content"]
    assert "统计客户总数（customers）" in body
    assert "按地区汇总" in body
    assert "SQL" in body


def test_signing_uses_the_documented_algorithm(webhook, monkeypatch) -> None:
    monkeypatch.setenv("FEISHU_SIGN_SECRET", "s3cret")
    notify_run_finished("run_4", "completed")

    payload = webhook.received[0]
    timestamp = payload["timestamp"]
    expected = base64.b64encode(
        hmac.new(
            f"{timestamp}\ns3cret".encode("utf-8"), digestmod=hashlib.sha256
        ).digest()
    ).decode("utf-8")
    assert payload["sign"] == expected


def test_a_rejected_push_is_reported_but_never_raised(webhook) -> None:
    webhook.status_code = 500
    assert notify_run_finished("run_5", "completed") is False


def test_a_business_error_code_is_not_treated_as_success(webhook) -> None:
    webhook.reply = {"code": 19021, "msg": "sign match fail"}
    assert notify_run_finished("run_6", "completed") is False


def test_an_unreachable_endpoint_is_silent(monkeypatch) -> None:
    monkeypatch.setenv("FEISHU_WEBHOOK_URL", "http://127.0.0.1:9/hook")
    monkeypatch.delenv("FEISHU_SIGN_SECRET", raising=False)
    assert notify_run_finished("run_7", "completed") is False


def test_without_a_webhook_nothing_is_sent(webhook, monkeypatch) -> None:
    monkeypatch.delenv("FEISHU_WEBHOOK_URL", raising=False)
    assert notify_run_finished("run_8", "completed") is False
    assert webhook.received == []


class _StubStore:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def update_status(self, run_id: str, status: str, **kwargs: object) -> None:
        self.calls.append((run_id, status))


def test_finalize_survives_a_notifier_that_raises(monkeypatch) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("通知炸了")

    monkeypatch.setattr(run_coordinator, "notify_run_finished", boom)

    coordinator = object.__new__(run_coordinator.RunCoordinator)
    coordinator.store = _StubStore()  # type: ignore[attr-defined]

    result = coordinator._finalize(  # type: ignore[attr-defined]
        "run_9", {"run_status": "completed", "analysis_status": "completed"}
    )

    assert result["run_status"] == "completed"
    assert coordinator.store.calls == [("run_9", "completed")]
