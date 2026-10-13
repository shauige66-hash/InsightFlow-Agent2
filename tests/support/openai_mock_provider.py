"""供集成测试使用的 OpenAI 兼容 Mock Provider。

只做一件事：返回结构合法的 `/chat/completions` 响应，并附带**固定**的 usage，
使网关侧的 token 提取与累加逻辑可被断言。

固定 usage 是刻意的：数值可预测才能验证"网关是否如实记录"，
它不代表任何真实模型的用量。

用法：
    python -m tests.support.openai_mock_provider --host 0.0.0.0 --port 9099
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# 固定用量：测试据此断言网关的提取与累加，不代表真实用量
PROMPT_TOKENS = 12
COMPLETION_TOKENS = 8

_STATS = {"total": 0}


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 约定
        if self.path == "/stats":
            self._send_json(200, dict(_STATS))
            return
        self._send_json(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 约定
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""

        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send_json(404, {"error": {"message": "not found"}})
            return

        try:
            request_body = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            request_body = {}

        _STATS["total"] += 1
        # 回显请求里的模型名，使网关按模型聚合时能对得上
        model = request_body.get("model") or "mock-model"

        self._send_json(
            200,
            {
                "id": f"chatcmpl-mock-{_STATS['total']}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "mock analysis result",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": PROMPT_TOKENS,
                    "completion_tokens": COMPLETION_TOKENS,
                    "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
                },
            },
        )

    def log_message(self, *args) -> None:  # noqa: ANN002 - 关闭默认访问日志
        return


def create_server(host: str, port: int) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), _Handler)


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenAI 兼容 Mock Provider")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9099)
    args = parser.parse_args()

    server = create_server(args.host, args.port)
    print(f"mock provider listening on {args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
