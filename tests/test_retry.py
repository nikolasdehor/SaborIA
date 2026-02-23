"""Unit tests for the retry module."""

import asyncio
import time

import pytest

import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test-placeholder")

from agents.retry import async_retry_with_backoff, retry_with_backoff


class TestSyncRetry:
    def test_succeeds_without_retry(self):
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def ok():
            nonlocal call_count
            call_count += 1
            return "success"

        assert ok() == "success"
        assert call_count == 1

    def test_retries_on_rate_limit(self):
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("rate limit exceeded")
            return "recovered"

        assert flaky() == "recovered"
        assert call_count == 3

    def test_raises_after_max_retries(self):
        @retry_with_backoff(max_retries=2, base_delay=0.01)
        def always_fails():
            raise Exception("rate limit exceeded forever")

        with pytest.raises(Exception, match="rate limit"):
            always_fails()

    def test_no_retry_on_non_transient_error(self):
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def non_transient():
            nonlocal call_count
            call_count += 1
            raise ValueError("bad input")

        with pytest.raises(ValueError, match="bad input"):
            non_transient()
        assert call_count == 1  # no retry for non-transient errors

    def test_backoff_delay_increases(self):
        """Verify that retries actually introduce delay (exponential)."""
        call_times: list[float] = []

        @retry_with_backoff(max_retries=2, base_delay=0.05, jitter=False)
        def timed_fail():
            call_times.append(time.monotonic())
            if len(call_times) <= 2:
                raise Exception("rate limit")
            return "ok"

        timed_fail()
        assert len(call_times) == 3
        # Second retry delay should be >= first retry delay
        gap1 = call_times[1] - call_times[0]
        gap2 = call_times[2] - call_times[1]
        assert gap2 >= gap1 * 0.8  # allow some tolerance


class TestAsyncRetry:
    def test_async_succeeds(self):
        call_count = 0

        @async_retry_with_backoff(max_retries=3, base_delay=0.01)
        async def ok():
            nonlocal call_count
            call_count += 1
            return "async_success"

        result = asyncio.run(ok())
        assert result == "async_success"
        assert call_count == 1

    def test_async_retries_on_timeout(self):
        call_count = 0

        @async_retry_with_backoff(max_retries=3, base_delay=0.01)
        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("timeout error")
            return "recovered"

        result = asyncio.run(flaky())
        assert result == "recovered"
        assert call_count == 2

    def test_async_raises_after_max(self):
        @async_retry_with_backoff(max_retries=1, base_delay=0.01)
        async def always_fails():
            raise Exception("503 server error")

        with pytest.raises(Exception, match="503"):
            asyncio.run(always_fails())
