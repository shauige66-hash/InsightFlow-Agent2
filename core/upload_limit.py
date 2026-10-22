"""Bound the raw request body before multipart parsing can spool it to disk."""
from starlette.responses import JSONResponse


class UploadLimitExceeded(Exception):
    pass


class UploadBodyLimitMiddleware:
    def __init__(self, app, *, limit_getter):
        self.app = app
        self.limit_getter = limit_getter

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") != "POST" or scope.get("path") not in ("/api/uploads", "/analyze/upload"):
            return await self.app(scope, receive, send)
        limit = int(self.limit_getter())
        headers = dict(scope.get("headers") or [])
        try:
            declared = int(headers.get(b"content-length", b"0"))
        except ValueError:
            declared = limit + 1
        if declared > limit:
            return await JSONResponse({"detail":"上传请求超过容量上限"},status_code=413)(scope,receive,send)
        consumed = 0
        async def bounded_receive():
            nonlocal consumed
            message = await receive()
            if message["type"] == "http.request":
                consumed += len(message.get("body",b""))
                if consumed > limit:
                    raise UploadLimitExceeded()
            return message
        try:
            await self.app(scope,bounded_receive,send)
        except UploadLimitExceeded:
            await JSONResponse({"detail":"上传请求超过容量上限"},status_code=413)(scope,receive,send)
