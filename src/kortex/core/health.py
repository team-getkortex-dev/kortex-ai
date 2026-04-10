"""Sliding-window health monitor for provider and runtime metrics.

Monitors error rates and latency trends per provider using a sliding window
of recent observations. Predicts degradation *before* the circuit breaker
fires by detecting upward latency trends via simple linear regression.

Health states:
- ``healthy`` — error rate and latency within normal bounds
- ``degrading`` — error rate rising or latency trend is upward
- ``critical`` — error rate or latency above configured thresholds

Background polling fires every ``check_interval_s`` seconds and calls
registered ``AlertHandler`` callbacks when state transitions occur.

Example::

    from kortex.core.health import HealthMonitor, LogAlertHandler

    monitor = HealthMonitor()
    monitor.add_alert_handler(LogAlertHandler())
    monitor.record_success("anthropic", latency_ms=320.0)
    monitor.record_failure("anthropic")

    status = monitor.get_status("anthropic")
    print(status.state)  # "healthy" | "degrading" | "critical"
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal

import structlog

logger = structlog.get_logger(component="health_monitor")

HealthState = Literal["healthy", "degrading", "critical"]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ProviderHealthStatus:
    """Snapshot of health for a single provider.

    Args:
        provider: Provider name.
        state: Current health state.
        error_rate: Fraction of recent requests that failed (0–1).
        avg_latency_ms: Average latency over the window.
        latency_trend: Slope of a linear regression fit to recent latency
            samples (ms/observation). Positive = latency increasing.
        window_size: Number of observations in the window.
        checked_at: Unix timestamp of this check.
    """

    provider: str
    state: HealthState
    error_rate: float
    avg_latency_ms: float
    latency_trend: float
    window_size: int
    checked_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "state": self.state,
            "error_rate": round(self.error_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "latency_trend": round(self.latency_trend, 3),
            "window_size": self.window_size,
            "checked_at": self.checked_at,
        }


@dataclass
class HealthAlert:
    """An alert fired when a provider transitions to a new state.

    Args:
        provider: Provider name.
        previous_state: State before the transition.
        current_state: New state after the transition.
        status: Full status snapshot.
        message: Human-readable description.
    """

    provider: str
    previous_state: HealthState | None
    current_state: HealthState
    status: ProviderHealthStatus
    message: str


# ---------------------------------------------------------------------------
# Alert handlers
# ---------------------------------------------------------------------------


class AlertHandler(ABC):
    """Abstract base class for health alert handlers."""

    @abstractmethod
    async def on_alert(self, alert: HealthAlert) -> None:
        """Handle a health alert.

        Args:
            alert: The health alert.
        """


class LogAlertHandler(AlertHandler):
    """Alert handler that logs to structlog."""

    async def on_alert(self, alert: HealthAlert) -> None:
        level = "warning" if alert.current_state == "degrading" else "error"
        getattr(logger, level)(
            "provider_health_alert",
            provider=alert.provider,
            previous=alert.previous_state,
            current=alert.current_state,
            error_rate=f"{alert.status.error_rate:.2%}",
            avg_latency_ms=f"{alert.status.avg_latency_ms:.0f}",
            latency_trend=f"{alert.status.latency_trend:+.1f}",
            message=alert.message,
        )


class CallbackAlertHandler(AlertHandler):
    """Alert handler that calls a user-provided async callable."""

    def __init__(self, callback: Any) -> None:
        self._callback = callback

    async def on_alert(self, alert: HealthAlert) -> None:
        await self._callback(alert)


# ---------------------------------------------------------------------------
# Sliding-window internals
# ---------------------------------------------------------------------------


@dataclass
class _ProviderWindow:
    """Per-provider sliding window of observations."""

    outcomes: deque[bool] = field(default_factory=lambda: deque(maxlen=100))
    latencies: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    last_state: HealthState | None = None


def _linear_trend(values: list[float]) -> float:
    """Compute slope of the OLS fit to ``values`` (simplified).

    For n values [v0, v1, …, v_{n-1}], returns d(v)/d(index).

    Args:
        values: Sequence of observations.

    Returns:
        Slope in units/observation (positive = increasing).
    """
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if den else 0.0


# ---------------------------------------------------------------------------
# HealthMonitor
# ---------------------------------------------------------------------------


class HealthMonitor:
    """Sliding-window health monitor for provider metrics.

    Tracks per-provider error rates and latency trends. Fires ``AlertHandler``
    callbacks on state transitions. Optionally runs a background polling loop.

    Args:
        window_size: Number of observations per provider window.
        error_rate_degrading: Error rate threshold for ``degrading`` state.
        error_rate_critical: Error rate threshold for ``critical`` state.
        latency_degrading_slope: Latency trend (ms/observation) threshold
            for ``degrading``.
        latency_critical_ms: Absolute average latency for ``critical``.
        check_interval_s: Seconds between background checks.
        predictive_threshold: If latency trend exceeds this, pre-emptively
            open the provider circuit even before error rate rises.
    """

    def __init__(
        self,
        window_size: int = 100,
        error_rate_degrading: float = 0.05,
        error_rate_critical: float = 0.20,
        latency_degrading_slope: float = 10.0,
        latency_critical_ms: float = 5000.0,
        check_interval_s: float = 60.0,
        predictive_threshold: float = 20.0,
    ) -> None:
        self._window_size = window_size
        self._error_rate_degrading = error_rate_degrading
        self._error_rate_critical = error_rate_critical
        self._latency_degrading_slope = latency_degrading_slope
        self._latency_critical_ms = latency_critical_ms
        self._check_interval_s = check_interval_s
        self._predictive_threshold = predictive_threshold

        self._windows: dict[str, _ProviderWindow] = {}
        self._alert_handlers: list[AlertHandler] = []
        self._background_task: asyncio.Task[None] | None = None
        self._log = structlog.get_logger(component="health_monitor")

    # ------------------------------------------------------------------
    # Observation recording
    # ------------------------------------------------------------------

    def record_success(self, provider: str, latency_ms: float) -> None:
        """Record a successful request for a provider.

        Args:
            provider: Provider name.
            latency_ms: Observed end-to-end latency in milliseconds.
        """
        window = self._get_window(provider)
        window.outcomes.append(True)
        window.latencies.append(latency_ms)

    def record_failure(self, provider: str, latency_ms: float = 0.0) -> None:
        """Record a failed request for a provider.

        Args:
            provider: Provider name.
            latency_ms: Observed latency before failure (may be 0).
        """
        window = self._get_window(provider)
        window.outcomes.append(False)
        if latency_ms > 0:
            window.latencies.append(latency_ms)

    def _get_window(self, provider: str) -> _ProviderWindow:
        if provider not in self._windows:
            self._windows[provider] = _ProviderWindow(
                outcomes=deque(maxlen=self._window_size),
                latencies=deque(maxlen=self._window_size),
            )
        return self._windows[provider]

    # ------------------------------------------------------------------
    # Status computation
    # ------------------------------------------------------------------

    def get_status(self, provider: str) -> ProviderHealthStatus:
        """Compute current health status for a provider.

        Args:
            provider: Provider name.

        Returns:
            A ProviderHealthStatus snapshot.
        """
        window = self._get_window(provider)
        outcomes = list(window.outcomes)
        latencies = list(window.latencies)

        if not outcomes:
            return ProviderHealthStatus(
                provider=provider,
                state="healthy",
                error_rate=0.0,
                avg_latency_ms=0.0,
                latency_trend=0.0,
                window_size=0,
            )

        error_rate = 1.0 - (sum(outcomes) / len(outcomes))
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        trend = _linear_trend(latencies)

        state = self._classify(error_rate, avg_latency, trend)
        return ProviderHealthStatus(
            provider=provider,
            state=state,
            error_rate=error_rate,
            avg_latency_ms=avg_latency,
            latency_trend=trend,
            window_size=len(outcomes),
        )

    def get_all_statuses(self) -> dict[str, ProviderHealthStatus]:
        """Return health status for all known providers.

        Returns:
            Mapping of provider name → ProviderHealthStatus.
        """
        return {provider: self.get_status(provider) for provider in self._windows}

    def _classify(
        self, error_rate: float, avg_latency_ms: float, trend: float
    ) -> HealthState:
        if (
            error_rate >= self._error_rate_critical
            or avg_latency_ms >= self._latency_critical_ms
        ):
            return "critical"
        if (
            error_rate >= self._error_rate_degrading
            or trend >= self._latency_degrading_slope
            or trend >= self._predictive_threshold
        ):
            return "degrading"
        return "healthy"

    def is_healthy(self, provider: str) -> bool:
        """Return True if the provider's current state is 'healthy'.

        Args:
            provider: Provider name.

        Returns:
            True when healthy, False when degrading or critical.
        """
        return self.get_status(provider).state == "healthy"

    def should_circuit_break(self, provider: str) -> bool:
        """Return True if predictive circuit breaking should activate.

        Activates when the provider is critical OR when the latency trend
        exceeds the predictive threshold.

        Args:
            provider: Provider name.

        Returns:
            True when the circuit should be opened pre-emptively.
        """
        status = self.get_status(provider)
        return status.state == "critical" or status.latency_trend >= self._predictive_threshold

    # ------------------------------------------------------------------
    # Alert handlers
    # ------------------------------------------------------------------

    def add_alert_handler(self, handler: AlertHandler) -> None:
        """Register an alert handler.

        Args:
            handler: The AlertHandler to register.
        """
        self._alert_handlers.append(handler)

    async def _fire_alerts(
        self, provider: str, status: ProviderHealthStatus
    ) -> None:
        """Fire alerts if state has changed since last check."""
        window = self._windows.get(provider)
        if window is None:
            return

        prev_state = window.last_state
        if prev_state == status.state:
            return  # No transition

        window.last_state = status.state
        if not self._alert_handlers:
            return

        state_word = {"healthy": "recovered", "degrading": "degrading", "critical": "critical"}
        message = (
            f"Provider '{provider}' is {state_word.get(status.state, status.state)}: "
            f"error_rate={status.error_rate:.2%}, "
            f"avg_latency={status.avg_latency_ms:.0f}ms, "
            f"trend={status.latency_trend:+.1f}ms/obs"
        )
        alert = HealthAlert(
            provider=provider,
            previous_state=prev_state,
            current_state=status.state,
            status=status,
            message=message,
        )
        for handler in self._alert_handlers:
            try:
                await handler.on_alert(alert)
            except Exception as exc:
                self._log.warning("alert_handler_error", error=str(exc))

    # ------------------------------------------------------------------
    # Background polling
    # ------------------------------------------------------------------

    async def _background_check(self) -> None:
        """Periodically re-evaluate all provider statuses and fire alerts."""
        while True:
            await asyncio.sleep(self._check_interval_s)
            for provider in list(self._windows.keys()):
                status = self.get_status(provider)
                await self._fire_alerts(provider, status)
                self._log.debug(
                    "health_check",
                    provider=provider,
                    state=status.state,
                    error_rate=f"{status.error_rate:.2%}",
                )

    def start_background_checks(self) -> None:
        """Start the background polling loop (requires running event loop)."""
        if self._background_task is None or self._background_task.done():
            self._background_task = asyncio.ensure_future(self._background_check())

    def stop_background_checks(self) -> None:
        """Stop the background polling loop."""
        if self._background_task is not None and not self._background_task.done():
            self._background_task.cancel()
            self._background_task = None

    async def check_now(self) -> dict[str, ProviderHealthStatus]:
        """Manually trigger a health check and fire any pending alerts.

        Returns:
            Current status snapshot for all providers.
        """
        statuses = self.get_all_statuses()
        for provider, status in statuses.items():
            await self._fire_alerts(provider, status)
        return statuses

    def reset(self, provider: str | None = None) -> None:
        """Clear observation windows.

        Args:
            provider: If given, reset only that provider. Otherwise reset all.
        """
        if provider is not None:
            self._windows.pop(provider, None)
        else:
            self._windows.clear()
