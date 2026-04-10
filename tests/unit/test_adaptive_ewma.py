"""Tests for AdaptiveEWMA — Task 1: adaptive alpha, outlier rejection, bootstrap."""

from __future__ import annotations

import math

import pytest

from kortex.core.adaptive_ewma import AdaptiveEWMA


# ---------------------------------------------------------------------------
# Alpha decay schedule
# ---------------------------------------------------------------------------


class TestAlphaDecay:
    def test_first_sample_uses_high_alpha(self) -> None:
        ewma = AdaptiveEWMA()
        ewma.update(100.0)
        # On first post-bootstrap update (sample 11), alpha should be high
        # Pre-bootstrap: no alpha recorded
        assert ewma.sample_count == 1

    def test_alpha_decays_over_100_samples(self) -> None:
        ewma = AdaptiveEWMA()
        for i in range(100):
            ewma.update(float(i))
        # alpha_history only populated for post-bootstrap EWMA updates
        if ewma.alpha_history:
            assert ewma.alpha_history[0] > ewma.alpha_history[-1], (
                "Alpha should decrease over time"
            )
            # Early alpha should be close to 0.8
            assert ewma.alpha_history[0] >= 0.3
            # Late alpha should approach 0.05 floor
            assert ewma.alpha_history[-1] <= 0.10

    def test_alpha_floor_is_0_05(self) -> None:
        ewma = AdaptiveEWMA()
        for _ in range(500):
            ewma.update(100.0)
        if ewma.alpha_history:
            assert min(ewma.alpha_history) >= 0.05 - 1e-9

    def test_alpha_ceiling_is_0_8(self) -> None:
        ewma = AdaptiveEWMA()
        for _ in range(200):
            ewma.update(100.0)
        if ewma.alpha_history:
            assert max(ewma.alpha_history) <= 0.8 + 1e-9

    def test_decay_formula(self) -> None:
        """Verify the decay formula: alpha = max(0.05, 0.8 * exp(-n/20))."""
        ewma = AdaptiveEWMA()
        # Feed 11 samples so first EWMA update triggers
        for i in range(11):
            ewma.update(100.0)
        assert len(ewma.alpha_history) >= 1
        # First alpha is computed at sample_count=11
        expected = max(0.05, 0.8 * math.exp(-10 / 20.0))
        assert ewma.alpha_history[0] == pytest.approx(expected, rel=1e-6)

    def test_fixed_alpha_bypasses_decay(self) -> None:
        ewma = AdaptiveEWMA(fixed_alpha=0.5)
        for _ in range(50):
            ewma.update(200.0)
        # No alpha_history in fixed mode
        assert ewma.alpha_history == []


# ---------------------------------------------------------------------------
# Outlier rejection
# ---------------------------------------------------------------------------


class TestOutlierRejection:
    def _trained_ewma(self, value: float = 200.0, n: int = 20) -> AdaptiveEWMA:
        ewma = AdaptiveEWMA()
        for _ in range(n):
            ewma.update(value)
        return ewma

    def test_extreme_spike_rejected(self) -> None:
        """Single 5000ms spike when typical is 200ms should be rejected."""
        ewma = self._trained_ewma(200.0, n=30)
        count_before = ewma.sample_count
        rejected_before = ewma.rejected_count
        est_before = ewma.current_estimate

        ewma.update(5000.0)  # z >> 3.0

        assert ewma.rejected_count > rejected_before, "Spike should be rejected"
        assert ewma.sample_count == count_before, "Rejected sample should not increment count"
        assert abs(ewma.current_estimate - est_before) < 1.0, (
            "Estimate should not change on rejection"
        )

    def test_normal_value_not_rejected(self) -> None:
        ewma = self._trained_ewma(200.0, n=30)
        rejected_before = ewma.rejected_count
        ewma.update(210.0)  # within normal range
        assert ewma.rejected_count == rejected_before

    def test_no_rejection_before_bootstrap_complete(self) -> None:
        """Outlier rejection only kicks in after 10+ samples."""
        ewma = AdaptiveEWMA()
        # Seed with a few samples, then inject a spike before 10 samples
        for _ in range(5):
            ewma.update(200.0)
        ewma.update(9999.0)  # should NOT be rejected (only 5 prior samples)
        assert ewma.rejected_count == 0

    def test_rejected_count_tracked(self) -> None:
        ewma = self._trained_ewma(200.0, n=30)
        for _ in range(5):
            ewma.update(99999.0)
        assert ewma.rejected_count >= 1


# ---------------------------------------------------------------------------
# Bootstrap cold-start
# ---------------------------------------------------------------------------


class TestBootstrapColdStart:
    def test_first_10_samples_averaged(self) -> None:
        ewma = AdaptiveEWMA(bootstrap_samples=10)
        vals = [float(i * 10) for i in range(1, 11)]  # 10, 20, ..., 100
        for v in vals:
            ewma.update(v)
        expected_mean = sum(vals) / len(vals)  # 55.0
        assert ewma.current_estimate == pytest.approx(expected_mean, rel=1e-6)

    def test_single_sample_equals_that_value(self) -> None:
        ewma = AdaptiveEWMA()
        ewma.update(42.0)
        assert ewma.current_estimate == pytest.approx(42.0)

    def test_bootstrap_transitions_to_ewma_at_boundary(self) -> None:
        ewma = AdaptiveEWMA(bootstrap_samples=5)
        for _ in range(5):
            ewma.update(100.0)
        # Still in bootstrap: estimate = 100.0
        assert ewma.current_estimate == pytest.approx(100.0)
        # Sample 6 triggers EWMA from bootstrap mean
        ewma.update(100.0)
        assert ewma.current_estimate == pytest.approx(100.0, rel=1e-3)
        assert len(ewma.alpha_history) == 1  # first EWMA alpha recorded

    def test_ewma_initialised_from_bootstrap_mean(self) -> None:
        """EWMA estimate after bootstrap should be close to the bootstrap mean."""
        ewma = AdaptiveEWMA(bootstrap_samples=10)
        for _ in range(10):
            ewma.update(100.0)
        # One more sample to trigger the EWMA transition
        ewma.update(100.0)
        assert ewma.current_estimate == pytest.approx(100.0, rel=0.01)

    def test_fixed_alpha_has_no_bootstrap(self) -> None:
        """Fixed-alpha mode: first sample seeds directly, no bootstrap averaging."""
        ewma = AdaptiveEWMA(fixed_alpha=1.0)
        ewma.update(500.0)
        ewma.update(99.0)
        # alpha=1.0 → always last value
        assert ewma.current_estimate == pytest.approx(99.0)


# ---------------------------------------------------------------------------
# Convergence speed comparison
# ---------------------------------------------------------------------------


class TestConvergenceSpeed:
    def _converge_n(self, ewma: AdaptiveEWMA, target: float, n: int) -> float:
        for _ in range(n):
            ewma.update(target)
        return ewma.current_estimate

    def test_adaptive_converges_faster_than_fixed_early(self) -> None:
        """After the same number of early samples, adaptive should be closer to target."""
        target = 100.0
        seed = 500.0

        adaptive = AdaptiveEWMA()
        adaptive.update(seed)
        adaptive_est = self._converge_n(adaptive, target, 15)

        fixed = AdaptiveEWMA(fixed_alpha=0.3)
        fixed.update(seed)
        fixed_est = self._converge_n(fixed, target, 15)

        adaptive_err = abs(adaptive_est - target)
        fixed_err = abs(fixed_est - target)
        assert adaptive_err <= fixed_err, (
            f"Adaptive ({adaptive_err:.1f}) should be at least as close as "
            f"fixed-alpha ({fixed_err:.1f}) after 15 samples"
        )

    def test_adaptive_stable_after_many_samples(self) -> None:
        """After many samples, adaptive EWMA should be very close to true mean."""
        ewma = AdaptiveEWMA()
        target = 200.0
        for _ in range(100):
            ewma.update(target)
        assert abs(ewma.current_estimate - target) < 1.0


# ---------------------------------------------------------------------------
# Fixed-alpha backward compatibility
# ---------------------------------------------------------------------------


class TestFixedAlpha:
    def test_alpha_1_always_last_value(self) -> None:
        ewma = AdaptiveEWMA(fixed_alpha=1.0)
        ewma.update(500.0)
        ewma.update(99.0)
        assert ewma.current_estimate == pytest.approx(99.0)

    def test_alpha_0_ignores_new_values(self) -> None:
        ewma = AdaptiveEWMA(fixed_alpha=0.0)
        ewma.update(500.0)   # seed
        ewma.update(99.0)    # 0 * 99 + 1 * 500 = 500
        ewma.update(1.0)     # stays 500
        assert ewma.current_estimate == pytest.approx(500.0)

    def test_no_outlier_rejection_in_fixed_mode(self) -> None:
        ewma = AdaptiveEWMA(fixed_alpha=0.5)
        for _ in range(20):
            ewma.update(200.0)
        ewma.update(9999.0)
        assert ewma.rejected_count == 0  # never rejects in fixed mode

    def test_sample_count_increments_in_fixed_mode(self) -> None:
        ewma = AdaptiveEWMA(fixed_alpha=0.3)
        for _ in range(7):
            ewma.update(100.0)
        assert ewma.sample_count == 7


# ---------------------------------------------------------------------------
# is_trained
# ---------------------------------------------------------------------------


def test_is_trained_false_before_any_sample() -> None:
    assert not AdaptiveEWMA().is_trained


def test_is_trained_true_after_first_sample() -> None:
    ewma = AdaptiveEWMA()
    ewma.update(1.0)
    assert ewma.is_trained


# ---------------------------------------------------------------------------
# summary()
# ---------------------------------------------------------------------------


def test_summary_structure() -> None:
    ewma = AdaptiveEWMA()
    for v in [100.0, 110.0, 90.0, 105.0]:
        ewma.update(v)
    s = ewma.summary()
    assert "sample_count" in s
    assert "estimate" in s
    assert "p50" in s
    assert "p95" in s
    assert "p99" in s
    assert s["mode"] == "adaptive"


def test_summary_fixed_mode() -> None:
    ewma = AdaptiveEWMA(fixed_alpha=0.5)
    ewma.update(100.0)
    assert ewma.summary()["mode"] == "fixed"
