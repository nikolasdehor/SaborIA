"""
Structured logging and observability middleware for SaborAI.

Adds request ID tracking, structured JSON log formatting, and timing
for all API requests. This is essential for production debugging and
performance monitoring.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

# ── Request context ───────────────────────────────────────────────────────────

request_id_var: ContextVar[str] = ContextVar("request_id", default="")


def get_request_id() -> str:
    return request_id_var.get()


# ── Structured JSON formatter ─────────────────────────────────────────────────


class StructuredLogFormatter(logging.Formatter):
    """Emit logs as JSON lines — easy to parse with tools like jq, Loki, etc."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Attach request ID if available
        req_id = request_id_var.get("")
        if req_id:
            log_entry["request_id"] = req_id

        # Attach extra fields if provided
        if hasattr(record, "extra_data"):
            log_entry["data"] = record.extra_data

        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = str(record.exc_info[1])

        return json.dumps(log_entry, ensure_ascii=False)


# ── Request tracking middleware ───────────────────────────────────────────────


class RequestTrackingMiddleware(BaseHTTPMiddleware):
    """Middleware that assigns a unique ID to each request and logs timing."""

    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
        request_id_var.set(req_id)

        logger = logging.getLogger("saborai.http")
        t0 = time.perf_counter()

        logger.info(
            "→ %s %s",
            request.method,
            request.url.path,
            extra={"extra_data": {"params": dict(request.query_params)}},
        )

        response = await call_next(request)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "← %s %s %d (%.0fms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            extra={"extra_data": {"status": response.status_code, "latency_ms": round(elapsed_ms, 1)}},
        )

        response.headers["X-Request-ID"] = req_id
        response.headers["X-Response-Time-Ms"] = str(round(elapsed_ms, 1))
        return response


def configure_logging(level: str = "INFO", structured: bool = True) -> None:
    """Configure root logger with optional structured JSON output."""
    root = logging.getLogger()
    root.setLevel(level)

    # Remove existing handlers
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = logging.StreamHandler()
    if structured:
        handler.setFormatter(StructuredLogFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(levelname)s | %(name)s | %(message)s"))
    root.addHandler(handler)
