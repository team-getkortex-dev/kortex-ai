"""Resilient HTTP client wrapping httpx with retry, circuit breaker, and timeouts.

Drop-in replacement for raw ``httpx.AsyncClient`` calls in provider connectors.
Adds structured logging, request IDs, and automatic error classification.
"""

from __future__ import annotations

import asyncio
import uuid
from types import TracebackType
from typing import Any

import httpx
import structlog

from kortex.core.exceptions import (
    CircuitOpenError,
    ProviderAuthError,
    ProviderOverloadError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from kortex.providers.resilience import (
    CircuitBreaker,
    RetryPolicy,
    TimeoutPolicy,
)

logger = structlog.get_logger(component="resilient_client")


class ResilientClient:
    """HTTP client with retry, circuit breaker, and timeout support.

    Wraps ``httpx.AsyncClient`` and adds:
    - Exponential backoff retries for transient failures
    - Circuit breaker to stop hammering a dead endpoint
    - Per-phase timeouts (connect, read, total)
    - Unique request ID header on every request
    - Automatic error classification into typed exceptions

    Args:
        retry_policy: Retry configuration. Uses defaults if None.
        circuit_breaker: Circuit breaker instance. Creates new one if None.
        timeout_policy: Timeout configuration. Uses defaults if None.
        request_id_header: Header name for the unique request ID.
    """

    def __init__(
        self,
        retry_policy: RetryPolicy | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        timeout_policy: TimeoutPolicy | None = None,
        request_id_header: str = "X-Request-ID",
    ) -> None:
        self._retry = retry_policy or RetryPolicy()
        self._breaker = circuit_breaker or CircuitBreaker()
        self._timeout = timeout_policy or TimeoutPolicy()
        self._request_id_header = request_id_header
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """Return (or create) the underlying httpx client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=self._timeout.connect_timeout_s,
                    read=self._timeout.read_timeout_s,
                    write=self._timeout.read_timeout_s,
                    pool=self._timeout.total_timeout_s,
                ),
            )
        return self._client

    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make an HTTP request with retry, circuit breaker, and timeout.

        Args:
            method: HTTP method (GET, POST, etc.).
            url: Full URL to request.
            headers: Optional headers (merged with request ID).
            **kwargs: Forwarded to ``httpx.AsyncClient.request()``.

        Returns:
            The successful httpx.Response.

        Raises:
            CircuitOpenError: If the circuit breaker is open.
            ProviderTimeoutError: On request timeout after retries.
            ProviderAuthError: On 401/403 responses (not retried).
            ProviderRateLimitError: On 429 after retries exhausted.
            ProviderOverloadError: On 5xx after retries exhausted.
        """
        # 1. Circuit breaker check
        if not self._breaker.allow_request():
            raise CircuitOpenError(
                f"Circuit breaker is open for {url}. "
                f"Requests are being rejected to prevent cascading failures."
            )

        # 2. Add request ID
        req_headers = dict(headers or {})
        request_id = str(uuid.uuid4())
        req_headers[self._request_id_header] = request_id

        client = self._get_client()
        last_exc: Exception | None = None
        last_response: httpx.Response | None = None

        # 3. Retry loop
        for attempt in range(self._retry.max_retries + 1):
            try:
                response = await client.request(
                    method, url, headers=req_headers, **kwargs,
                )

                # Non-retryable client errors
                if response.status_code in (401, 403):
                    self._breaker.record_failure()
                    raise ProviderAuthError(
                        f"Authentication failed: HTTP {response.status_code} "
                        f"from {url}"
                    )

                # Retryable status codes
                if response.status_code in self._retry.retryable_status_codes:
                    last_response = response
                    if attempt < self._retry.max_retries:
                        delay = self._retry.delay_ms(attempt) / 1000
                        logger.info(
                            "retrying_request",
                            attempt=attempt + 1,
                            max_retries=self._retry.max_retries,
                            status_code=response.status_code,
                            delay_ms=self._retry.delay_ms(attempt),
                            request_id=request_id,
                        )
                        await asyncio.sleep(delay)
                        continue
                    # Retries exhausted — classify the error
                    break

                # Success
                self._breaker.record_success()
                return response

            except (httpx.TimeoutException,) as exc:
                last_exc = exc
                if attempt < self._retry.max_retries:
                    delay = self._retry.delay_ms(attempt) / 1000
                    logger.info(
                        "retrying_timeout",
                        attempt=attempt + 1,
                        max_retries=self._retry.max_retries,
                        delay_ms=self._retry.delay_ms(attempt),
                        request_id=request_id,
                    )
                    await asyncio.sleep(delay)
                    continue
                break

            except (httpx.ConnectError, ConnectionError, TimeoutError) as exc:
                last_exc = exc
                if attempt < self._retry.max_retries:
                    delay = self._retry.delay_ms(attempt) / 1000
                    logger.info(
                        "retrying_connection_error",
                        attempt=attempt + 1,
                        max_retries=self._retry.max_retries,
                        error=str(exc),
                        delay_ms=self._retry.delay_ms(attempt),
                        request_id=request_id,
                    )
                    await asyncio.sleep(delay)
                    continue
                break

        # 4. All retries exhausted — record failure and classify
        self._breaker.record_failure()

        if last_exc is not None and isinstance(
            last_exc, (httpx.TimeoutException, TimeoutError)
        ):
            raise ProviderTimeoutError(
                f"Request to {url} timed out after "
                f"{self._retry.max_retries + 1} attempt(s)"
            ) from last_exc

        if last_exc is not None and isinstance(
            last_exc, (httpx.ConnectError, ConnectionError)
        ):
            raise ProviderOverloadError(
                f"Connection to {url} failed after "
                f"{self._retry.max_retries + 1} attempt(s): {last_exc}"
            ) from last_exc

        if last_response is not None:
            status = last_response.status_code
            if status == 429:
                raise ProviderRateLimitError(
                    f"Rate limited (HTTP 429) by {url} after "
                    f"{self._retry.max_retries + 1} attempt(s)"
                )
            raise ProviderOverloadError(
                f"Server error (HTTP {status}) from {url} after "
                f"{self._retry.max_retries + 1} attempt(s)"
            )

        # Should not reach here, but just in case
        raise ProviderOverloadError(
            f"Request to {url} failed after "
            f"{self._retry.max_retries + 1} attempt(s)"
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    async def __aenter__(self) -> ResilientClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()
