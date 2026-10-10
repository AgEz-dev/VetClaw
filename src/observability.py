"""可观测性：request_id 上下文 + 结构化日志（维度 6）。

设计要点
- request_id 用 contextvar 承载；中间件为**纯 ASGI**（非 `BaseHTTPMiddleware`），
  避免后者包裹 StreamingResponse 时对 SSE 的缓冲/破坏与额外开销；
- `RequestIdFilter` 把 request_id 注入每条 LogRecord，`JsonFormatter` 输出单行 JSON；
- `install_logging()` 一次性配置 root logger（幂等）。
"""
from __future__ import annotations

import json
import logging
import uuid
from contextvars import ContextVar

REQUEST_ID_HEADER = "X-Request-Id"

_request_id: ContextVar[str] = ContextVar("request_id", default="")


def get_request_id() -> str:
    return _request_id.get()


def set_request_id(value: str) -> None:
    _request_id.set(value)


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


class RequestIdMiddleware:
    """纯 ASGI 中间件：读上游 X-Request-Id（缺失则生成），写入 contextvar 并回写响应头。"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        raw = dict(scope.get("headers") or []).get(b"x-request-id")
        rid = raw.decode("latin-1") if raw else new_request_id()
        set_request_id(rid)

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = message.setdefault("headers", [])
                headers.append((b"x-request-id", rid.encode("latin-1")))
            await send(message)

        await self.app(scope, receive, send_wrapper)


class RequestIdFilter(logging.Filter):
    """把当前 request_id 注入 LogRecord，供 Formatter 使用。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id() or "-"
        return True


class JsonFormatter(logging.Formatter):
    """单行 JSON 日志（含 time / level / logger / request_id / message）。"""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def install_logging(level: int = logging.INFO) -> None:
    """配置 root logger：JSON 输出 + request_id 注入。幂等——重复调用只替换 handler。"""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    handler.addFilter(RequestIdFilter())

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(level)
