"""测试网络守卫。

单元测试不应访问任何真实上游模型服务。这里在 socket 层拦截出站连接，
把"测试忘了替换 LLM"这类疏漏变成明确失败，而不是悄悄产生一次付费调用。

允许的连接仅限于本机回环（本地 SQLite 之外的临时服务、pytest-asyncio 内部通信等），
以及显式加入白名单的主机。真实服务商域名永远在拒绝之列。
"""

from __future__ import annotations

import socket

# 真实上游域名。命中即拒绝，不提供放行开关——需要联调时应使用集成测试而非单元测试。
BLOCKED_HOSTS = frozenset(
    {
        "api.deepseek.com",
        "api.openai.com",
        "huggingface.co",
    }
)

# 测试期间允许的连接目标。
# host.docker.internal 是 Docker 的宿主网关别名，仅用于本地 Mock Provider 集成测试，
# 它指向开发机自身而不会路由到任何真实服务商。
ALLOWED_HOSTS = frozenset(
    {"127.0.0.1", "localhost", "::1", "host.docker.internal"}
)


class NetworkAccessBlocked(RuntimeError):
    """单元测试试图访问被禁止的网络目标。"""


def _resolve(host: str) -> str:
    return host.strip("[]").lower()


def _is_allowed(host: str) -> bool:
    host = _resolve(host)
    if host in ALLOWED_HOSTS:
        return True
    return host.startswith("127.")


def guard_enabled(host: str) -> bool:
    """判断是否应拦截该主机。

    解析失败时保守拦截：测试里出现无法解析的域名，本身就说明有意外出站。
    """
    host = _resolve(host)

    if host in BLOCKED_HOSTS:
        return True

    return not _is_allowed(host)


def install_network_guard() -> None:
    """在 socket 层安装拦截器。重复调用是幂等的。"""
    if getattr(socket, "_insightflow_guard_installed", False):
        return

    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def guarded_connect(self, address):  # noqa: ANN001, ANN202
        host = address[0] if isinstance(address, tuple) else str(address)
        if guard_enabled(str(host)):
            raise NetworkAccessBlocked(
                f"测试禁止访问 {host!r}；请注入 Mock LLM，"
                f"或把该主机加入 tests/network_guard.py 的 ALLOWED_HOSTS。"
            )
        return original_connect(self, address)

    def guarded_connect_ex(self, address):  # noqa: ANN001, ANN202
        host = address[0] if isinstance(address, tuple) else str(address)
        if guard_enabled(str(host)):
            raise NetworkAccessBlocked(
                f"测试禁止访问 {host!r}；请注入 Mock LLM。"
            )
        return original_connect_ex(self, address)

    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex
    socket._insightflow_guard_installed = True

    # 同时清掉代理变量，避免 httpx/openai 通过环境变量把请求转给本机代理，
    # 那样会绕过上面的域名判断。
    import os

    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(name, None)
