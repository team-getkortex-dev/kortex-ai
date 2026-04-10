"""Tests for the HealthMonitor."""

from __future__ import annotations

import asyncio

import pytest

from kortex.core.health import (
    AlertHandler,
    CallbackAlertHandler,
    HealthAlert,
    HealthMonitor,
    LogAlertHandler,
    ProviderHealthStatus,
    _linear_trend,
)


# ---------------------------------------------------------------------------
# _linear_trend
# ---------------------------------------------------------------------------


def test_linear_trend_empty() -> None:
    assert _linear_trend([]) == 0.0


def test_linear_trend_single() -> None:
    assert _linear_trend([100.0]) == 0.0


def test_linear_trend_flat() -> None:
    trend = _linear_trend([200.0, 200.0, 200.0, 200.0])
    assert abs(trend) < 0.001


def test_linear_trend_increasing() -> None:
    trend = _linear_trend([100.0, 200.0, 300.0, 400.0])
    assert trend > 0


def test_linear_trend_decreasing() -> None:
    trend = _linear_trend([400.0, 300.0, 200.0, 100.0])
    assert trend < 0


# ---------------------------------------------------------------------------
# HealthMonitor basic recording
# ---------------------------------------------------------------------------


def test_initial_status_is_healthy() -> None:
    monitor = HealthMonitor()
    status = monitor.get_status("openai")
    assert status.state == "healthy"
    assert status.error_rate == 0.0
    assert status.window_size == 0


def test_all_successes_stays_healthy() -> None:
    monitor = HealthMonitor()
    for _ in range(20):
        monitor.record_success("openai", latency_ms=200.0)
    status = monitor.get_status("openai")
    assert status.state == "healthy"
    assert status.error_rate == 0.0


def test_error_rate_computed_correctly() -> None:
    monitor = HealthMonitor()
    for _ in range(8):
        monitor.record_success("openai", latency_ms=200.0)
    for _ in range(2):
        monitor.record_failure("openai")
    status = monitor.get_status("openai")
    assert abs(status.error_rate - 0.2) < 0.01


def test_high_error_rate_becomes_critical() -> None:
    monitor = HealthMonitor(error_rate_critical=0.20)
    for _ in range(5):
        monitor.record_success("openai", latency_ms=200.0)
    for _ in range(5):
        monitor.record_failure("openai")
    status = monitor.get_status("openai")
    assert status.state == "critical"


def test_moderate_error_rate_is_degrading() -> None:
    monitor = HealthMonitor(error_rate_degrading=0.05, error_rate_critical=0.20)
    for _ in range(18):
        monitor.record_success("openai", latency_ms=200.0)
    for _ in range(2):
        monitor.record_failure("openai")
    status = monitor.get_status("openai")
    assert status.state == "degrading"


def test_latency_trend_degradation() -> None:
    monitor = HealthMonitor(latency_degrading_slope=5.0)
    # Steadily increasing latencies
    for i in range(20):
        monitor.record_success("openai", latency_ms=200.0 + i * 10)
    status = monitor.get_status("openai")
    assert status.latency_trend > 0
    assert status.state in ("degrading", "critical")


def test_high_avg_latency_critical() -> None:
    monitor = HealthMonitor(latency_critical_ms=1000.0)
    for _ in range(10):
        monitor.record_success("openai", latency_ms=2000.0)
    status = monitor.get_status("openai")
    assert status.state == "critical"


def test_multiple_providers_independent() -> None:
    monitor = HealthMonitor()
    monitor.record_success("openai", latency_ms=200.0)
    for _ in range(5):
        monitor.record_failure("anthropic")

    openai_status = monitor.get_status("openai")
    anthropic_status = monitor.get_status("anthropic")
    assert openai_status.state == "healthy"
    assert anthropic_status.state != "healthy"


def test_get_all_statuses() -> None:
    monitor = HealthMonitor()
    monitor.record_success("openai", latency_ms=200.0)
    monitor.record_failure("anthropic")

    statuses = monitor.get_all_statuses()
    assert "openai" in statuses
    assert "anthropic" in statuses


def test_is_healthy() -> None:
    monitor = HealthMonitor()
    monitor.record_success("openai", latency_ms=200.0)
    assert monitor.is_healthy("openai") is True


def test_is_healthy_unknown_provider() -> None:
    monitor = HealthMonitor()
    assert monitor.is_healthy("unknown") is True  # no data = healthy


# ---------------------------------------------------------------------------
# Predictive circuit breaking
# ---------------------------------------------------------------------------


def test_should_circuit_break_healthy() -> None:
    monitor = HealthMonitor(predictive_threshold=50.0)
    for _ in range(10):
        monitor.record_success("openai", latency_ms=200.0)
    assert monitor.should_circuit_break("openai") is False


def test_should_circuit_break_on_trend(monkeypatch) -> None:
    monitor = HealthMonitor(predictive_threshold=5.0)
    # Rapidly increasing latency trend
    for i in range(20):
        monitor.record_success("openai", latency_ms=100.0 + i * 20)
    assert monitor.should_circuit_break("openai") is True


def test_should_circuit_break_critical_state() -> None:
    monitor = HealthMonitor(error_rate_critical=0.1)
    for _ in range(5):
        monitor.record_failure("anthropic")
    assert monitor.should_circuit_break("anthropic") is True


# ---------------------------------------------------------------------------
# Alert handlers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_callback_alert_handler_fires_on_transition() -> None:
    alerts: list[HealthAlert] = []

    async def capture(alert: HealthAlert) -> None:
        alerts.append(alert)

    monitor = HealthMonitor(error_rate_critical=0.20)
    monitor.add_alert_handler(CallbackAlertHandler(capture))

    # Drive to critical state
    for _ in range(5):
        monitor.record_failure("openai")

    await monitor.check_now()
    assert len(alerts) == 1
    assert alerts[0].current_state == "critical"
    assert alerts[0].provider == "openai"


@pytest.mark.asyncio
async def test_alert_not_fired_twice_for_same_state() -> None:
    alerts: list[HealthAlert] = []

    async def capture(alert: HealthAlert) -> None:
        alerts.append(alert)

    monitor = HealthMonitor(error_rate_critical=0.20)
    monitor.add_alert_handler(CallbackAlertHandler(capture))

    for _ in range(5):
        monitor.record_failure("openai")

    await monitor.check_now()
    await monitor.check_now()  # same state, no new alert

    assert len(alerts) == 1


@pytest.mark.asyncio
async def test_recovery_alert_fires_when_state_improves() -> None:
    alerts: list[HealthAlert] = []

    async def capture(alert: HealthAlert) -> None:
        alerts.append(alert)

    monitor = HealthMonitor(error_rate_critical=0.20)
    monitor.add_alert_handler(CallbackAlertHandler(capture))

    # Degrade to critical
    for _ in range(5):
        monitor.record_failure("openai")
    await monitor.check_now()

    # Recover with successes
    monitor.reset("openai")
    for _ in range(20):
        monitor.record_success("openai", latency_ms=100.0)
    await monitor.check_now()

    states = [a.current_state for a in alerts]
    assert "critical" in states
    assert "healthy" in states


@pytest.mark.asyncio
async def test_log_alert_handler_does_not_raise() -> None:
    monitor = HealthMonitor(error_rate_critical=0.20)
    monitor.add_alert_handler(LogAlertHandler())

    for _ in range(5):
        monitor.record_failure("openai")
    await monitor.check_now()  # Should not raise


# ---------------------------------------------------------------------------
# ProviderHealthStatus.to_dict
# ---------------------------------------------------------------------------


def test_status_to_dict() -> None:
    monitor = HealthMonitor()
    monitor.record_success("openai", latency_ms=250.0)
    status = monitor.get_status("openai")
    d = status.to_dict()
    assert d["provider"] == "openai"
    assert "state" in d
    assert "error_rate" in d
    assert "avg_latency_ms" in d
    assert "latency_trend" in d


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


def test_reset_single_provider() -> None:
    monitor = HealthMonitor()
    monitor.record_failure("openai")
    monitor.record_success("anthropic", latency_ms=200.0)

    monitor.reset("openai")
    status = monitor.get_status("openai")
    assert status.window_size == 0
    # anthropic still has data
    anthropic = monitor.get_status("anthropic")
    assert anthropic.window_size > 0


def test_reset_all() -> None:
    monitor = HealthMonitor()
    monitor.record_success("openai", latency_ms=200.0)
    monitor.record_success("anthropic", latency_ms=300.0)
    monitor.reset()
    assert monitor.get_all_statuses() == {}


# ---------------------------------------------------------------------------
# Background checks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_background_checks_start_and_stop() -> None:
    monitor = HealthMonitor(check_interval_s=9999)
    monitor.start_background_checks()
    assert monitor._background_task is not None
    assert not monitor._background_task.done()

    monitor.stop_background_checks()
    assert monitor._background_task is None
