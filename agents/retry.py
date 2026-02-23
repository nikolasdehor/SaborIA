"""
Retry utilities with exponential backoff for LLM API calls.

Handles transient failures (rate limits, timeouts, server errors) gracefully
so agents can recover without user intervention.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
import time

logger = logging.getLogger(__name__)

# Exceptions that should trigger a retry (transient / recoverable)
_RETRYABLE_SUBSTRINGS = ("rate limit", "timeout", "server error", "502", "503", "529")


def _is_retryable(exc: Exception) -> bool:
    """Check whether an exception is transient and worth retrying."""
    msg = str(exc).lower()
    return any(s in msg for s in _RETRYABLE_SUBSTRINGS)


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: bool = True,
):
    """Decorator: retry a **sync** function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay in seconds before the first retry.
        max_delay: Cap on the delay between retries.
        jitter: Add random jitter to avoid thundering-herd effects.
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc: Exception | None = None
            for attempt in range(max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if attempt == max_retries or not _is_retryable(exc):
                        raise
                    delay = min(base_delay * (2**attempt), max_delay)
                    if jitter:
                        delay *= 0.5 + random.random()
                    logger.warning(
                        "Retry %d/%d for %s after %.1fs — %s",
                        attempt + 1,
                        max_retries,
                        fn.__qualname__,
                        delay,
                        exc,
                    )
                    time.sleep(delay)
            raise last_exc  # type: ignore[misc]  # unreachable but keeps mypy happy

        return wrapper

    return decorator


def async_retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: bool = True,
):
    """Decorator: retry an **async** function with exponential backoff."""

    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            last_exc: Exception | None = None
            for attempt in range(max_retries + 1):
                try:
                    return await fn(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if attempt == max_retries or not _is_retryable(exc):
                        raise
                    delay = min(base_delay * (2**attempt), max_delay)
                    if jitter:
                        delay *= 0.5 + random.random()
                    logger.warning(
                        "Async retry %d/%d for %s after %.1fs — %s",
                        attempt + 1,
                        max_retries,
                        fn.__qualname__,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)
            raise last_exc  # type: ignore[misc]

        return wrapper

    return decorator
