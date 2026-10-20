"""诊断模型凭据：是 401、余额不足，还是模型名不对。

    python scripts/check_model.py

只发一次 4 token 的请求，用来区分"key 无效"和"别的问题"——这类排查里最常见的
浪费就是把 401 当成网络故障，反复重启服务。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from core.config import get_settings  # noqa: E402

HINTS = {
    401: "key 无效：到申请方控制台重新生成，确认没有多余空格/引号",
    402: "余额不足：充值后再试（DeepSeek 的余额用尽也返回 402）",
    403: "权限不足：这个 key 没有该模型的调用权限",
    404: "接口路径不对：确认 base_url 是否包含 /v1",
    429: "触发限流：稍后重试，或降低并发",
}


def main() -> int:
    settings = get_settings()
    print(f"base_url : {settings.base_url}")
    print(f"model    : {settings.model or '(未设置)'}")
    print(f"api_key  : {'已配置，长度 ' + str(len(settings.api_key)) if settings.api_key else '(未设置)'}")

    if not settings.api_key:
        print("\n结论：没有配置 DEEPSEEK_API_KEY")
        return 2
    if not settings.model:
        print("\n结论：没有配置 DEEPSEEK_MODEL（例如 deepseek-chat）")
        return 2

    url = settings.base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": settings.model,
        "messages": [{"role": "user", "content": "回复一个字符：好"}],
        "max_tokens": 4,
    }
    headers = {"Authorization": f"Bearer {settings.api_key}"}

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, json=payload, headers=headers)
    except Exception as error:  # noqa: BLE001 - 网络问题也要说清楚
        print(f"\n请求失败：{type(error).__name__}: {error}")
        print("结论：网络不可达，检查代理 / base_url")
        return 3

    print(f"\nHTTP {response.status_code}")
    body = response.text[:300]
    print(body)

    if response.status_code == 200:
        try:
            data = response.json()
            usage = data.get("usage") or {}
            content = data["choices"][0]["message"]["content"]
            print(f"\n结论：可用。返回 {content!r}，消耗 {usage.get('total_tokens')} tokens")
            return 0
        except Exception:  # noqa: BLE001
            print("\n结论：返回 200 但结构不像 OpenAI 兼容响应")
            return 1

    print(f"\n结论：{HINTS.get(response.status_code, '未预期的状态码')}")
    if response.status_code == 401:
        print("提示：401 与网络无关——服务能连上，只是这个 key 被拒绝。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
