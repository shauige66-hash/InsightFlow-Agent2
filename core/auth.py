"""接口鉴权：Bearer token → 归属者。

设计要点：

- **没配 `INSIGHTFLOW_API_KEYS` 就保持原来的单用户演示模式**，不要求任何凭据。
  这样现有测试、本机演示、以及"我只想自己用"的场景都不受影响。
- 配了之后，除健康检查与静态页面外的所有请求都必须带 `Authorization: Bearer`。
- 归属者只从 token 推导，**不接受请求体或查询参数里的 owner**：让调用方自报
  归属者等于没有隔离。
"""

from __future__ import annotations

import hmac
import hashlib
import base64
import json
import time
import secrets
import os
import re
from dataclasses import dataclass

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

import storage

#: 不需要鉴权的路径：健康检查、文档、静态控制台
PUBLIC_PREFIXES = (
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/ui",
    "/favicon.ico",
)

_ARTIFACT = re.compile(r"^/runs/([A-Za-z0-9_-]{4,64})/(report|report\.html|[^/]+\.(png|svg|jpg|jpeg|webp))$")
_PROCESS_SECRET = secrets.token_bytes(32)


def _signing_key():
    raw = os.getenv("INSIGHTFLOW_ARTIFACT_SIGNING_KEY") or os.getenv("INSIGHTFLOW_API_KEYS", "")
    return hashlib.sha256(raw.encode("utf-8")).digest() if raw else _PROCESS_SECRET


def artifact_access_token(run_id: str, owner: str, *, lifetime: int = 300) -> str:
    payload = base64.urlsafe_b64encode(json.dumps({"run":run_id,"owner":owner,"exp":int(time.time())+lifetime},separators=(",",":")).encode()).decode().rstrip("=")
    signature = hmac.new(_signing_key(),payload.encode(),hashlib.sha256).hexdigest()
    return payload+"."+signature


def _artifact_owner(request: Request):
    match = _ARTIFACT.fullmatch(request.url.path)
    if request.method not in ("GET","HEAD") or not match:
        return None
    value = request.query_params.get("access") or request.cookies.get("insightflow_artifact_access") or ""
    try:
        payload,signature=value.rsplit(".",1)
        expected=hmac.new(_signing_key(),payload.encode(),hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature,expected):
            return None
        item=json.loads(base64.urlsafe_b64decode(payload+"="*(-len(payload)%4)))
        if item["run"] != match.group(1) or int(item["exp"]) < time.time():
            return None
        return str(item["owner"])
    except (ValueError,KeyError,TypeError):
        return None


def _is_public(request: Request) -> bool:
    path=request.url.path
    return any(path == prefix or path.startswith(prefix+"/") for prefix in PUBLIC_PREFIXES)


@dataclass(frozen=True)
class ApiKey:
    token: str
    owner: str


def load_api_keys(raw: str | None = None) -> list[ApiKey]:
    """解析 `token:owner,token2:owner2`。

    格式不对直接报错而不是静默忽略——如果写错了就变成"看起来启用了鉴权，
    实际谁都能进"，比不配更危险。
    """
    text = raw if raw is not None else os.getenv("INSIGHTFLOW_API_KEYS", "")
    entries: list[ApiKey] = []
    for chunk in (text or "").split(","):
        item = chunk.strip()
        if not item:
            continue
        token, separator, owner = item.partition(":")
        token, owner = token.strip(), owner.strip()
        if not separator or not token or not owner:
            raise ValueError(
                "INSIGHTFLOW_API_KEYS 每项必须是非空 token:owner"
            )
        entries.append(ApiKey(token=token, owner=owner))
    return entries


def resolve_owner(keys: list[ApiKey], header: str | None) -> str | None:
    """从 Authorization 头解析归属者；不匹配返回 None。"""
    if not header:
        return None
    scheme, _, credential = header.partition(" ")
    if scheme.lower() != "bearer" or not credential.strip():
        return None
    presented = credential.strip()
    for key in keys:
        # 定长比较，避免通过响应时间猜 token
        if hmac.compare_digest(presented, key.token):
            return key.owner
    return None


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """配了 key 就强制鉴权，并把归属者放进请求上下文。"""

    def __init__(self, app, *, keys: list[ApiKey]) -> None:
        super().__init__(app)
        self._keys = keys

    async def dispatch(self, request: Request, call_next):
        if not self._keys or _is_public(request):
            return await call_next(request)

        owner = resolve_owner(self._keys, request.headers.get("authorization")) or _artifact_owner(request)
        if owner is None:
            return JSONResponse(
                status_code=401,
                content={"detail": "缺少或无效的 API Key（Authorization: Bearer ...）"},
                headers={"WWW-Authenticate": "Bearer"},
            )

        token = storage.set_current_owner(owner)
        try:
            response = await call_next(request)
            if request.query_params.get("access") and _ARTIFACT.fullmatch(request.url.path):
                run_id = _ARTIFACT.fullmatch(request.url.path).group(1)
                response.set_cookie("insightflow_artifact_access",request.query_params["access"],max_age=300,
                    path=f"/runs/{run_id}/",httponly=True,samesite="strict",secure=request.url.scheme=="https")
            return response
        finally:
            storage.reset_current_owner(token)


def install(app) -> list[ApiKey]:
    """按环境变量决定是否安装鉴权中间件。"""
    keys = load_api_keys()
    if keys:
        app.add_middleware(ApiKeyMiddleware, keys=keys)
    return keys
