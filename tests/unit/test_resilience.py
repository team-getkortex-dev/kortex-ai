"""Unit tests for resilience primitives and ResilientClient."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from kortex.core.exceptions import (
    CircuitOpenError,
    ProviderAuthError,
    ProviderOverloadError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from kortex.providers.resilience import (
    CircuitBreaker,
    CircuitBreakerState,
    RetryPolicy,
    TimeoutPolicy,
)
from kortex.providers.resilient_client import ResilientClient


# ---------------------------------------------------------------------------
# 1. RetryPolicy calculates correct exponential backoff
# ---------------------------------------------------------------------------


class TestRetryPolicyBackoff:
    def test_exponential_backoff(self) -> None:
        policy = RetryPolicy(backoff_base_ms=100, backoff_multiplier=2.0)
        assert policy.delay_ms(0) == 100
        assert policy.delay_ms(1) == 200
        assert policy.delay_ms(2) == 400


# ---------------------------------------------------------------------------
# 2. Backoff respects max cap
# ---------------------------------------------------------------------------


class TestRetryPolicyBackoffCap:
    def test_cap(self) -> None:
        policy = RetryPolicy(
            backoff_base_ms=100, backoff_multiplier=10.0, backoff_max_ms=500,
        )
        assert policy.delay_ms(0) == 100
        assert policy.delay_ms(1) == 500  # 1000 capped to 500
        assert policy.delay_ms(2) == 500  # 10000 capped to 500


# ---------------------------------------------------------------------------
# 3. CircuitBreaker starts CLOSED
# ---------------------------------------------------------------------------


class TestCircuitBreakerInitial:
    def test_starts_closed(self) -> None:
        cb = CircuitBreaker()
        assert cb.state == CircuitBreakerState.CLOSED


# ---------------------------------------------------------------------------
# 4. Opens after failure_threshold consecutive failures
# ---------------------------------------------------------------------------


class TestCircuitBreakerOpens:
    def test_opens_after_threshold(self) -> None:
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreakerState.CLOSED
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN


# ---------------------------------------------------------------------------
# 5. OPEN state rejects requests
# ---------------------------------------------------------------------------


class TestCircuitBreakerOpenRejects:
    def test_open_rejects(self) -> None:
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout_s=9999)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN
        assert cb.allow_request() is False


# ---------------------------------------------------------------------------
# 6. Transitions to HALF_OPEN after recovery timeout
# ---------------------------------------------------------------------------


class TestCircuitBreakerHalfOpen:
    def test_transitions_to_half_open(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout_s=0.01)
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN
        time.sleep(0.02)
        assert cb.state == CircuitBreakerState.HALF_OPEN
        assert cb.allow_request() is True


# ---------------------------------------------------------------------------
# 7. Closes after successful HALF_OPEN call
# ---------------------------------------------------------------------------


class TestCircuitBreakerCloses:
    def test_closes_on_half_open_success(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout_s=0.01)
        cb.record_failure()
        time.sleep(0.02)
        assert cb.state == CircuitBreakerState.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitBreakerState.CLOSED


# ---------------------------------------------------------------------------
# 8. Resets failure count on success
# ---------------------------------------------------------------------------


class TestCircuitBreakerReset:
    def test_success_resets_count(self) -> None:
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        # Should be able to take 3 more failures before opening
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreakerState.CLOSED
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN


# ---------------------------------------------------------------------------
# Helper: mock httpx response
# ---------------------------------------------------------------------------


def _mock_response(status_code: int = 200, json_data: dict | None = None) -> httpx.Response:
    """Create a mock httpx.Response."""
    resp = httpx.Response(
        status_code=status_code,
        json=json_data or {},
        request=httpx.Request("GET", "https://example.com"),
    )
    return resp


# ---------------------------------------------------------------------------
# 9. ResilientClient retries on 429
# ---------------------------------------------------------------------------


class TestResilientClientRetry429:
    @pytest.mark.asyncio
    async def test_retries_on_429(self) -> None:
        policy = RetryPolicy(max_retries=2, backoff_base_ms=1)
        client = ResilientClient(retry_policy=policy)

        call_count = 0
        original_request = client._get_client().request

        async def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return _mock_response(429)
            return _mock_response(200, {"ok": True})

        with patch.object(client._get_client(), "request", side_effect=mock_request):
            response = await client.request("GET", "https://api.example.com/v1/test")

        assert response.status_code == 200
        assert call_count == 3
        await client.close()


# ---------------------------------------------------------------------------
# 10. ResilientClient retries on 500
# ---------------------------------------------------------------------------


class TestResilientClientRetry500:
    @pytest.mark.asyncio
    async def test_retries_on_500(self) -> None:
        policy = RetryPolicy(max_retries=1, backoff_base_ms=1)
        client = ResilientClient(retry_policy=policy)

        call_count = 0

        async def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _mock_response(500)
            return _mock_response(200)

        with patch.object(client._get_client(), "request", side_effect=mock_request):
            response = await client.request("GET", "https://api.example.com/test")

        assert response.status_code == 200
        assert call_count == 2
        await client.close()


# ---------------------------------------------------------------------------
# 11. Does NOT retry on 400 (client error)
# ---------------------------------------------------------------------------


class TestResilientClientNoRetry400:
    @pytest.mark.asyncio
    async def test_no_retry_on_400(self) -> None:
        policy = RetryPolicy(max_retries=2, backoff_base_ms=1)
        client = ResilientClient(retry_policy=policy)

        call_count = 0

        async def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return _mock_response(400)

        with patch.object(client._get_client(), "request", side_effect=mock_request):
            response = await client.request("GET", "https://api.example.com/test")

        # 400 is not retryable — should return immediately
        assert response.status_code == 400
        assert call_count == 1
        await client.close()


# ---------------------------------------------------------------------------
# 12. Respects max_retries limit
# ---------------------------------------------------------------------------


class TestResilientClientMaxRetries:
    @pytest.mark.asyncio
    async def test_respects_max_retries(self) -> None:
        policy = RetryPolicy(max_retries=2, backoff_base_ms=1)
        client = ResilientClient(retry_policy=policy)

        call_count = 0

        async def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return _mock_response(503)

        with patch.object(client._get_client(), "request", side_effect=mock_request):
            with pytest.raises(ProviderOverloadError):
                await client.request("GET", "https://api.example.com/test")

        # 1 initial + 2 retries = 3 total
        assert call_count == 3
        await client.close()


# ---------------------------------------------------------------------------
# 13. Raises ProviderTimeoutError on timeout
# ---------------------------------------------------------------------------


class TestResilientClientTimeout:
    @pytest.mark.asyncio
    async def test_timeout_error(self) -> None:
        policy = RetryPolicy(max_retries=0, backoff_base_ms=1)
        client = ResilientClient(retry_policy=policy)

        async def mock_request(*args, **kwargs):
            raise httpx.ReadTimeout("timed out")

        with patch.object(client._get_client(), "request", side_effect=mock_request):
            with pytest.raises(ProviderTimeoutError):
                await client.request("GET", "https://api.example.com/test")

        await client.close()


# ---------------------------------------------------------------------------
# 14. Raises CircuitOpenError when circuit open
# ---------------------------------------------------------------------------


class TestResilientClientCircuitOpen:
    @pytest.mark.asyncio
    async def test_circuit_open_error(self) -> None:
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout_s=9999)
        breaker.record_failure()  # Opens the circuit

        client = ResilientClient(circuit_breaker=breaker)

        with pytest.raises(CircuitOpenError):
            await client.request("GET", "https://api.example.com/test")

        await client.close()


# ---------------------------------------------------------------------------
# 15. Adds request ID header to every request
# ---------------------------------------------------------------------------


class TestResilientClientRequestId:
    @pytest.mark.asyncio
    async def test_request_id_header(self) -> None:
        client = ResilientClient()

        captured_headers: dict[str, str] = {}

        async def mock_request(method, url, *, headers=None, **kwargs):
            if headers:
                captured_headers.update(headers)
            return _mock_response(200)

        with patch.object(client._get_client(), "request", side_effect=mock_request):
            await client.request("GET", "https://api.example.com/test")

        assert "X-Request-ID" in captured_headers
        # UUID format: 8-4-4-4-12
        rid = captured_headers["X-Request-ID"]
        assert len(rid) == 36
        assert rid.count("-") == 4
        await client.close()


# ---------------------------------------------------------------------------
# 16. Maps 401 to ProviderAuthError
# ---------------------------------------------------------------------------


class TestResilientClientAuthError:
    @pytest.mark.asyncio
    async def test_401_maps_to_auth_error(self) -> None:
        client = ResilientClient()

        async def mock_request(*args, **kwargs):
            return _mock_response(401)

        with patch.object(client._get_client(), "request", side_effect=mock_request):
            with pytest.raises(ProviderAuthError):
                await client.request("GET", "https://api.example.com/test")

        await client.close()


# ---------------------------------------------------------------------------
# 17. Maps 429 to ProviderRateLimitError after retries exhausted
# ---------------------------------------------------------------------------


class TestResilientClientRateLimitExhausted:
    @pytest.mark.asyncio
    async def test_429_exhausted(self) -> None:
        policy = RetryPolicy(max_retries=1, backoff_base_ms=1)
        client = ResilientClient(retry_policy=policy)

        async def mock_request(*args, **kwargs):
            return _mock_response(429)

        with patch.object(client._get_client(), "request", side_effect=mock_request):
            with pytest.raises(ProviderRateLimitError):
                await client.request("GET", "https://api.example.com/test")

        await client.close()


# ---------------------------------------------------------------------------
# 18. Succeeds on second retry after first 503
# ---------------------------------------------------------------------------


class TestResilientClientRecovery:
    @pytest.mark.asyncio
    async def test_succeeds_on_retry(self) -> None:
        policy = RetryPolicy(max_retries=2, backoff_base_ms=1)
        client = ResilientClient(retry_policy=policy)

        call_count = 0

        async def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _mock_response(503)
            return _mock_response(200, {"recovered": True})

        with patch.object(client._get_client(), "request", side_effect=mock_request):
            response = await client.request("GET", "https://api.example.com/test")

        assert response.status_code == 200
        assert call_count == 2
        await client.close()
