"""Tests for percentile tracking — Task 3: P50/P95/P99 with rolling window."""

from __future__ import annotations

import pytest

from kortex.core.adaptive_ewma import AdaptiveEWMA
from kortex.core.metrics import ObservedMetrics


# ---------------------------------------------------------------------------
# AdaptiveEWMA.get_percentile — correctness
# ---------------------------------------------------------------------------


class TestGetPercentile:
    def test_single_value_all_percentiles_equal(self) -> None:
        ewma = AdaptiveEWMA(fixed_alpha=0.3)
        ewma.update(42.0)
        assert ewma.get_percentile(50) == pytest.approx(42.0)
        assert ewma.get_percentile(95) == pytest.approx(42.0)
        assert ewma.get_percentile(99) == pytest.approx(42.0)

    def test_empty_window_returns_zero(self) -> None:
        ewma = AdaptiveEWMA()
        assert ewma.get_percentile(50) == 0.0
        assert ewma.get_percentile(95) == 0.0

    def test_p50_is_median(self) -> None:
        ewma = AdaptiveEWMA(fixed_alpha=0.3, window_size=100)
        vals = [float(i) for i in range(1, 101)]  # 1..100
        for v in vals:
            ewma.update(v)
        p50 = ewma.get_percentile(50)
        # Median of 1..100 = 50.5
        assert p50 == pytest.approx(50.5, rel=0.05)

    def test_p95_correct(self) -> None:
        ewma = AdaptiveEWMA(fixed_alpha=0.3, window_size=100)
        for i in range(1, 101):
            ewma.update(float(i))
        p95 = ewma.get_percentile(95)
        # 95th percentile of 1..100 = 95.05 (interpolated)
        assert p95 == pytest.approx(95.05, rel=0.05)

    def test_p99_correct(self) -> None:
        ewma = AdaptiveEWMA(fixed_alpha=0.3, window_size=100)
        for i in range(1, 101):
            ewma.update(float(i))
        p99 = ewma.get_percentile(99)
        assert p99 == pytest.approx(99.01, rel=0.05)

    def test_p99_dominated_by_tail(self) -> None:
        """Multiple spikes: P99 should reflect the tail, P50 should not."""
        ewma = AdaptiveEWMA(fixed_alpha=0.5, window_size=100)
        # 80 normal samples + 20 extreme spikes → spike appears well above P99
        for _ in range(80):
            ewma.update(100.0)
        for _ in range(20):
            ewma.update(10000.0)
        assert ewma.get_percentile(50) == pytest.approx(100.0)
        assert ewma.get_percentile(99) > 1000.0

    def test_percentile_p0_returns_min(self) -> None:
        ewma = AdaptiveEWMA(fixed_alpha=0.5, window_size=50)
        vals = [10.0, 20.0, 30.0, 40.0, 50.0]
        for v in vals:
            ewma.update(v)
        assert ewma.get_percentile(0) == pytest.approx(10.0)

    def test_percentile_p100_returns_max(self) -> None:
        ewma = AdaptiveEWMA(fixed_alpha=0.5, window_size=50)
        for v in [10.0, 20.0, 30.0, 40.0, 50.0]:
            ewma.update(v)
        assert ewma.get_percentile(100) == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Rolling window — maintains last N samples
# ---------------------------------------------------------------------------


class TestRollingWindow:
    def test_window_bounded_by_window_size(self) -> None:
        ewma = AdaptiveEWMA(fixed_alpha=0.5, window_size=10)
        for i in range(50):
            ewma.update(float(i))
        # Window should contain only the last 10
        assert len(ewma._window) == 10

    def test_window_contains_most_recent_values(self) -> None:
        ewma = AdaptiveEWMA(fixed_alpha=0.5, window_size=5)
        for i in range(20):
            ewma.update(float(i))
        # Last 5 values are 15, 16, 17, 18, 19
        assert min(ewma._window) == pytest.approx(15.0)
        assert max(ewma._window) == pytest.approx(19.0)

    def test_old_samples_evicted(self) -> None:
        ewma = AdaptiveEWMA(fixed_alpha=0.5, window_size=5)
        # Fill with 1000.0
        for _ in range(5):
            ewma.update(1000.0)
        # Then fill with 1.0 — old values should be gone
        for _ in range(5):
            ewma.update(1.0)
        assert max(ewma._window) == pytest.approx(1.0)

    def test_percentile_changes_as_window_slides(self) -> None:
        ewma = AdaptiveEWMA(fixed_alpha=0.5, window_size=10)
        for _ in range(10):
            ewma.update(100.0)
        p99_before = ewma.get_percentile(99)

        for _ in range(10):
            ewma.update(500.0)
        p99_after = ewma.get_percentile(99)

        assert p99_after > p99_before


# ---------------------------------------------------------------------------
# ObservedMetrics P95/P99 API
# ---------------------------------------------------------------------------


class TestObservedMetricsPercentiles:
    def test_get_latency_p95_no_data_returns_zero(self) -> None:
        m = ObservedMetrics()
        assert m.get_latency_p95("p::m") == 0.0

    def test_get_latency_p99_no_data_returns_zero(self) -> None:
        m = ObservedMetrics()
        assert m.get_latency_p99("p::m") == 0.0

    def test_p95_p99_after_observations(self) -> None:
        m = ObservedMetrics(alpha=0.5, window_size=100)
        for i in range(1, 101):
            m.update("p::m", latency_ms=float(i), cost_usd=0.001)
        p95 = m.get_latency_p95("p::m")
        p99 = m.get_latency_p99("p::m")
        assert p95 > 0.0
        assert p99 >= p95

    def test_p99_higher_than_p95(self) -> None:
        m = ObservedMetrics(alpha=0.3, window_size=100)
        for i in range(100):
            m.update("p::m", float(i * 10), 0.001)
        assert m.get_latency_p99("p::m") >= m.get_latency_p95("p::m")

    def test_p95_with_complexity_class(self) -> None:
        m = ObservedMetrics(alpha=0.5)
        for _ in range(20):
            m.update("p::m", 100.0, 0.001, "simple")
        for _ in range(20):
            m.update("p::m", 500.0, 0.005, "complex")

        p95_simple = m.get_latency_p95("p::m", "simple")
        p95_complex = m.get_latency_p95("p::m", "complex")
        # Complex latencies are much higher
        assert p95_complex > p95_simple

    def test_p95_fallback_to_medium(self) -> None:
        m = ObservedMetrics(alpha=0.5, window_size=50)
        for i in range(50):
            m.update("p::m", float(i * 10), 0.001, "medium")
        # Requesting "simple" should fall back to "medium" data
        p95_simple = m.get_latency_p95("p::m", "simple")
        p95_medium = m.get_latency_p95("p::m", "medium")
        assert p95_simple == pytest.approx(p95_medium)

    def test_tracker_summary_includes_percentiles(self) -> None:
        m = ObservedMetrics(alpha=0.5)
        for i in range(20):
            m.update("p::m", float(i * 10), 0.001)
        summary = m.tracker_summary("p::m")
        assert "p50" in summary
        assert "p95" in summary
        assert "p99" in summary
        assert summary["p99"] >= summary["p95"] >= summary["p50"]


# ---------------------------------------------------------------------------
# Config fields
# ---------------------------------------------------------------------------


def test_config_has_track_percentiles() -> None:
    from kortex.config import KortexConfig

    cfg = KortexConfig()
    assert cfg.track_percentiles is True
    assert cfg.percentile_window_size == 100


def test_config_percentile_window_size_customisable() -> None:
    from kortex.config import KortexConfig

    cfg = KortexConfig(percentile_window_size=200)
    assert cfg.percentile_window_size == 200


# ---------------------------------------------------------------------------
# DashboardMetrics fields
# ---------------------------------------------------------------------------


def test_dashboard_metrics_has_percentile_fields() -> None:
    from kortex.dashboard.tui import DashboardMetrics

    dm = DashboardMetrics()
    assert hasattr(dm, "model_latency_p95")
    assert hasattr(dm, "model_latency_p99")
    assert isinstance(dm.model_latency_p95, dict)
    assert isinstance(dm.model_latency_p99, dict)


def test_dashboard_metrics_percentile_fields_populated() -> None:
    from kortex.dashboard.tui import DashboardMetrics

    dm = DashboardMetrics(
        model_latency_p95={"openai::gpt-4o-mini": 350.0},
        model_latency_p99={"openai::gpt-4o-mini": 480.0},
    )
    assert dm.model_latency_p95["openai::gpt-4o-mini"] == pytest.approx(350.0)
    assert dm.model_latency_p99["openai::gpt-4o-mini"] == pytest.approx(480.0)
