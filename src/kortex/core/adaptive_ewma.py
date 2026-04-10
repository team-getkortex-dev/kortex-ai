"""Adaptive Exponential Weighted Moving Average (EWMA) for router calibration.

Replaces fixed-alpha EWMA with:
- Decay schedule: alpha decays from 0.8 → 0.05 over first ~100 samples
- Bootstrap cold-start: simple average for first 10 samples
- Outlier rejection: z-score > 3.0 rejected after 10+ samples
- Rolling window: last N samples stored for percentile queries

When ``fixed_alpha`` is provided the tracker behaves like a classic EWMA
(first sample seeds, then fixed alpha — no bootstrap, no outlier rejection).
This preserves backward-compatibility for code that relies on deterministic
alpha behaviour (e.g. tests using ``ObservedMetrics(alpha=1.0)``).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AdaptiveEWMA:
    """Adaptive EWMA tracker with decay schedule and outlier rejection.

    Args:
        fixed_alpha: If set, use a classic fixed-alpha EWMA (no adaptive
            schedule, no bootstrap, no outlier rejection). Useful for tests
            or when the caller wants deterministic convergence behaviour.
        window_size: Rolling window length for percentile calculations.
        bootstrap_samples: Number of samples to average before switching to
            EWMA.  Only used when ``fixed_alpha`` is None.
        outlier_z_threshold: Z-score above which samples are rejected.
            Only applied after ``bootstrap_samples`` samples have been seen.
    """

    fixed_alpha: float | None = None
    window_size: int = 100
    bootstrap_samples: int = 10
    outlier_z_threshold: float = 3.0

    # --- internal state (repr=False keeps repr clean) ---
    _sample_count: int = field(default=0, repr=False, init=False)
    _estimate: float = field(default=0.0, repr=False, init=False)
    _bootstrap_sum: float = field(default=0.0, repr=False, init=False)
    _rejected_count: int = field(default=0, repr=False, init=False)
    _window: deque[float] = field(repr=False, init=False)
    _alpha_history: list[float] = field(default_factory=list, repr=False, init=False)

    def __post_init__(self) -> None:
        self._window = deque(maxlen=self.window_size)

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def sample_count(self) -> int:
        """Number of accepted (non-rejected) samples seen so far."""
        return self._sample_count

    @property
    def current_estimate(self) -> float:
        """Current EWMA estimate (0.0 if no samples yet)."""
        return self._estimate

    @property
    def rejected_count(self) -> int:
        """Number of samples rejected by the outlier filter."""
        return self._rejected_count

    @property
    def alpha_history(self) -> list[float]:
        """Alpha values used for each EWMA update (empty in fixed-alpha mode)."""
        return list(self._alpha_history)

    @property
    def is_trained(self) -> bool:
        """True once at least one sample has been accepted."""
        return self._sample_count > 0

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(self, value: float) -> float:
        """Record a new observation and return the updated estimate.

        Args:
            value: The new observed value (e.g. latency in ms).

        Returns:
            The updated EWMA estimate after incorporating ``value``.
        """
        if self.fixed_alpha is not None:
            return self._update_fixed(value)
        return self._update_adaptive(value)

    # ------------------------------------------------------------------
    # Percentile queries
    # ------------------------------------------------------------------

    def get_percentile(self, p: float) -> float:
        """Return the p-th percentile of the rolling window.

        Args:
            p: Percentile in [0, 100] (e.g. 95 for P95).

        Returns:
            Interpolated percentile value, or 0.0 if no samples.
        """
        if not self._window:
            return 0.0
        sorted_vals = sorted(self._window)
        n = len(sorted_vals)
        if n == 1:
            return sorted_vals[0]
        idx = (p / 100.0) * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update_fixed(self, value: float) -> float:
        """Classic fixed-alpha EWMA: first sample seeds, then EWMA formula."""
        assert self.fixed_alpha is not None
        self._window.append(value)
        self._sample_count += 1
        if self._sample_count == 1:
            self._estimate = value
        else:
            self._estimate = (
                self.fixed_alpha * value + (1.0 - self.fixed_alpha) * self._estimate
            )
        return self._estimate

    def _update_adaptive(self, value: float) -> float:
        """Adaptive EWMA with bootstrap, decay schedule, and outlier rejection."""
        # Outlier rejection after bootstrap phase
        if self._sample_count > self.bootstrap_samples:
            std = self._window_std()
            if std > 0.0:
                z = abs(value - self._estimate) / std
                if z > self.outlier_z_threshold:
                    self._rejected_count += 1
                    return self._estimate

        self._sample_count += 1
        self._window.append(value)

        if self._sample_count <= self.bootstrap_samples:
            # Bootstrap phase: maintain simple running average
            self._bootstrap_sum += value
            self._estimate = self._bootstrap_sum / self._sample_count
        elif self._sample_count == self.bootstrap_samples + 1:
            # Transition: initialise EWMA from bootstrap mean
            bootstrap_mean = self._bootstrap_sum / self.bootstrap_samples
            alpha = self._adaptive_alpha()
            self._alpha_history.append(alpha)
            self._estimate = alpha * value + (1.0 - alpha) * bootstrap_mean
        else:
            alpha = self._adaptive_alpha()
            self._alpha_history.append(alpha)
            self._estimate = alpha * value + (1.0 - alpha) * self._estimate

        return self._estimate

    def _adaptive_alpha(self) -> float:
        """Decay schedule: high alpha early for fast convergence, low after ~100 samples."""
        # sample_count has already been incremented before this is called
        return max(0.05, 0.8 * math.exp(-(self._sample_count - 1) / 20.0))

    def _window_std(self) -> float:
        """Sample standard deviation of the rolling window.

        Returns a minimum std of 1% of the absolute mean (or 1.0) to prevent
        division-by-zero when all historical values are identical, while still
        allowing extreme deviations to be detected as outliers.
        """
        vals = list(self._window)
        n = len(vals)
        if n < 2:
            return float("inf")
        mean = sum(vals) / n
        variance = sum((v - mean) ** 2 for v in vals) / (n - 1)
        std = math.sqrt(variance)
        # Prevent std=0 masking genuine outliers: floor at 10% of mean or 1.0.
        # This keeps normal ±5% variation (z≈0.5) safely below the rejection
        # threshold while still catching extreme spikes (z≫3.0).
        return max(std, abs(mean) * 0.10, 1.0)

    # ------------------------------------------------------------------
    # Serialisation helpers (for diagnostics)
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Return a diagnostic snapshot of the tracker state."""
        return {
            "sample_count": self._sample_count,
            "rejected_count": self._rejected_count,
            "estimate": self._estimate,
            "current_alpha": (
                self.fixed_alpha
                if self.fixed_alpha is not None
                else (self._adaptive_alpha() if self._sample_count > 0 else 0.8)
            ),
            "p50": self.get_percentile(50),
            "p95": self.get_percentile(95),
            "p99": self.get_percentile(99),
            "window_size": len(self._window),
            "mode": "fixed" if self.fixed_alpha is not None else "adaptive",
        }
