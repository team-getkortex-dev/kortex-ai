"""Resilience primitives for production provider connectivity.

Provides retry policies, circuit breakers, and timeout policies
that can be composed into a ResilientClient for any LLM provider.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

import structlog

logger = structlog.get_logger(component="resilience")


# ---------------------------------------------------------------------------
# RetryPolicy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetryPolicy:
    """Configures exponential-backoff retry behavior.

    Args:
        max_retries: Maximum number of retry attempts (0 = no retries).
        backoff_base_ms: Initial backoff delay in milliseconds.
        backoff_multiplier: Multiplier applied per retry (exponential).
        backoff_max_ms: Maximum backoff delay cap.
        retryable_status_codes: HTTP status codes that trigger a retry.
        retryable_exceptions: Exception types that trigger a retry.
    """

    max_retries: int = 2
    backoff_base_ms: float = 100
    backoff_multiplier: float = 2.0
    backoff_max_ms: float = 5000
    retryable_status_codes: tuple[int, ...] = (429, 500, 502, 503, 504)
    retryable_exceptions: tuple[type, ...] = (ConnectionError, TimeoutError)

    def delay_ms(self, attempt: int) -> float:
        """Calculate backoff delay for a given retry attempt (0-based).

        Args:
            attempt: The retry attempt number (0 = first retry).

        Returns:
            Delay in milliseconds, capped at backoff_max_ms.
        """
        delay = self.backoff_base_ms * (self.backoff_multiplier ** attempt)
        return min(delay, self.backoff_max_ms)


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------


class CircuitBreakerState(Enum):
    """States for the circuit breaker state machine."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Prevents cascading failures by tracking consecutive errors.

    State transitions:
    - CLOSED: Normal operation. Opens after ``failure_threshold`` failures.
    - OPEN: All requests rejected. Transitions to HALF_OPEN after
      ``recovery_timeout_s`` seconds.
    - HALF_OPEN: Allows ``half_open_max_calls`` probe requests. If they
      succeed, closes. If they fail, re-opens.

    Args:
        failure_threshold: Consecutive failures before opening.
        recovery_timeout_s: Seconds to wait before probing.
        half_open_max_calls: Number of probe calls in HALF_OPEN state.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_s: float = 30,
        half_open_max_calls: int = 1,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout_s = recovery_timeout_s
        self._half_open_max_calls = half_open_max_calls

        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._half_open_calls = 0

    @property
    def state(self) -> CircuitBreakerState:
        """Current circuit breaker state, with automatic OPEN->HALF_OPEN."""
        if self._state == CircuitBreakerState.OPEN:
            if time.monotonic() - self._last_failure_time >= self._recovery_timeout_s:
                self._state = CircuitBreakerState.HALF_OPEN
                self._half_open_calls = 0
        return self._state

    def allow_request(self) -> bool:
        """Check whether a request should be allowed.

        Returns:
            True if the request can proceed, False if the circuit is open.
        """
        current = self.state
        if current == CircuitBreakerState.CLOSED:
            return True
        if current == CircuitBreakerState.HALF_OPEN:
            return self._half_open_calls < self._half_open_max_calls
        # OPEN
        return False

    def record_success(self) -> None:
        """Record a successful request. Resets failure count and closes circuit."""
        self._failure_count = 0
        self._state = CircuitBreakerState.CLOSED
        self._half_open_calls = 0

    def record_failure(self) -> None:
        """Record a failed request. Opens circuit if threshold is reached."""
        self._failure_count += 1
        self._last_failure_time = time.monotonic()

        if self._state == CircuitBreakerState.HALF_OPEN:
            # Failed probe — re-open
            self._state = CircuitBreakerState.OPEN
            return

        if self._failure_count >= self._failure_threshold:
            self._state = CircuitBreakerState.OPEN
            logger.warning(
                "circuit_opened",
                failure_count=self._failure_count,
                threshold=self._failure_threshold,
            )

    def reset(self) -> None:
        """Reset the circuit breaker to its initial CLOSED state."""
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0


# ---------------------------------------------------------------------------
# TimeoutPolicy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimeoutPolicy:
    """Configures per-phase timeouts for HTTP requests.

    Args:
        connect_timeout_s: Max seconds to establish a TCP connection.
        read_timeout_s: Max seconds to read the response body.
        total_timeout_s: Max seconds for the entire request.
    """

    connect_timeout_s: float = 5.0
    read_timeout_s: float = 30.0
    total_timeout_s: float = 60.0
