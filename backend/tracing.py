"""
Request ID Tracing — Lightweight correlation layer for AutoFixOps v2.

Every inbound HTTP request receives a UUID4 request_id.
This ID is:
  - Stored in Python's contextvars (async-safe, no thread-local issues)
  - Injected into every log line via a custom logging filter
  - Returned in the X-Request-ID response header
  - Passed through to the Claude API call and executor step
  - Attached to metrics records so a session can be reconstructed end-to-end

This means a single debugging session — analyze → execute → feedback —
can be correlated across all log lines and metrics entries using one UUID.

Interview answer: "I use contextvars for async-safe per-request state.
The request ID flows from the HTTP middleware through the agent call
and executor step, emitted on every log line. In production you'd
propagate it as a W3C traceparent header for distributed tracing."
"""

import logging
import uuid
from contextvars import ContextVar
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


# ── Per-request context ────────────────────────────────────────────────────

# ContextVar is async-safe: each coroutine gets its own copy.
# Never use threading.local() in async code — it breaks under concurrent requests.
request_id_var: ContextVar[str] = ContextVar("request_id", default="no-request-id")


def get_request_id() -> str:
    """Return the request ID for the currently executing coroutine."""
    return request_id_var.get()


# ── Logging filter ─────────────────────────────────────────────────────────

class RequestIDFilter(logging.Filter):
    """
    Injects the current request_id into every log record.
    Attach to any logger or handler to get correlated log lines.

    Log format example:
        2025-02-21 12:34:56 [req=a3f8d2c1] INFO  claude_agent: Analysis complete in 1840ms
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()[:8]   # First 8 chars keep logs readable
        return True


def configure_logging() -> None:
    """
    Apply the RequestIDFilter to the root logger.
    Call once at application startup.
    """
    fmt = logging.Formatter(
        "%(asctime)s [req=%(request_id)s] %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    handler = logging.StreamHandler()
    handler.addFilter(RequestIDFilter())
    handler.setFormatter(fmt)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)


# ── FastAPI middleware ─────────────────────────────────────────────────────

class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Assigns a UUID4 to every request before any route handler runs.

    Priority order for request ID:
      1. X-Request-ID header from client (allows clients to pass their own IDs)
      2. Auto-generated UUID4

    The resolved ID is:
      - Set into request_id_var (accessible anywhere in the coroutine chain)
      - Echoed back in X-Request-ID response header
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        token  = request_id_var.set(req_id)

        try:
            response = await call_next(request)
        finally:
            # Always reset — prevents ID leaking to the next request on a reused worker
            request_id_var.reset(token)

        response.headers["X-Request-ID"] = req_id
        return response
