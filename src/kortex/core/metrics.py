"""Observed-metrics tracker for adaptive router calibration.

Tracks real-world latency and cost observations per model using an
Adaptive Exponential Weighted Moving Average (AdaptiveEWMA). The router uses
these values to prefer observed performance over static metadata when they
diverge.

Observations are stratified by task complexity class so that simple queries
and complex reasoning tasks maintain separate estimates for the same model.

Backward compatibility:
    Passing an explicit ``alpha`` creates fixed-alpha trackers that behave
    exactly like the original EWMA (first sample seeds, subsequent samples
    apply alpha * new + (1-alpha) * old). Pass no ``alpha`` (the default)
    to use the adaptive schedule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kortex.core.adaptive_ewma import AdaptiveEWMA

_FALLBACK_CLASS = "medium"


@dataclass
class ObservedMetrics:
    """Per-model stratified EWMA tracker for observed latency and cost.

    Observations are keyed by ``(model_key, complexity_class)`` so that the
    router can maintain separate estimates for simple, medium, and complex
    tasks running on the same model.

    Args:
        alpha: When set, use a fixed-alpha EWMA (backward-compatible mode).
            When ``None`` (the default), use the adaptive decay schedule.
        window_size: Rolling window size for percentile calculation.

    Example::

        metrics = ObservedMetrics()

        # Feed back real observations after execution
        metrics.update("openai::gpt-4o-mini", latency_ms=180.0,
                       cost_usd=0.00032, complexity_class="simple")
        metrics.update("openai::gpt-4o-mini", latency_ms=410.0,
                       cost_usd=0.00071, complexity_class="complex")

        # Retrieve estimates (with automatic fallback to "medium")
        lat = metrics.get_latency("openai::gpt-4o-mini", "simple")
        p95 = metrics.get_latency_p95("openai::gpt-4o-mini", "simple")
        conf = metrics.get_confidence("openai::gpt-4o-mini", "simple")
    """

    alpha: float | None = None
    window_size: int = 100

    # Internal stratified trackers: key = (model_key, complexity_class)
    _latency_trackers: dict[tuple[str, str], "AdaptiveEWMA"] = field(
        default_factory=dict, repr=False
    )
    _cost_trackers: dict[tuple[str, str], "AdaptiveEWMA"] = field(
        default_factory=dict, repr=False
    )

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def update(
        self,
        model_key: str,
        latency_ms: float,
        cost_usd: float,
        complexity_class: str = _FALLBACK_CLASS,
    ) -> None:
        """Record a new observation for ``model_key`` at ``complexity_class``.

        Args:
            model_key: Composite key in ``provider::model`` format.
            latency_ms: Observed end-to-end latency in milliseconds.
            cost_usd: Observed cost in USD for this request.
            complexity_class: Task complexity bucket ("simple"/"medium"/"complex").
        """
        bucket = (model_key, complexity_class)
        if bucket not in self._latency_trackers:
            self._latency_trackers[bucket] = self._new_tracker()
            self._cost_trackers[bucket] = self._new_tracker()
        self._latency_trackers[bucket].update(latency_ms)
        self._cost_trackers[bucket].update(cost_usd)

    # ------------------------------------------------------------------
    # Point estimates (EWMA)
    # ------------------------------------------------------------------

    def get_latency(
        self, model_key: str, complexity_class: str = _FALLBACK_CLASS
    ) -> float | None:
        """Return the EWMA latency estimate, or ``None`` if no observations.

        Falls back to the ``"medium"`` bucket when the requested class has
        no data, so legacy code that never passes ``complexity_class`` still
        works correctly.

        Args:
            model_key: Composite key in ``provider::model`` format.
            complexity_class: Task complexity bucket.

        Returns:
            EWMA latency in milliseconds, or None.
        """
        tracker = self._get_latency_tracker(model_key, complexity_class)
        if tracker is None or not tracker.is_trained:
            return None
        return tracker.current_estimate

    def get_cost(
        self, model_key: str, complexity_class: str = _FALLBACK_CLASS
    ) -> float | None:
        """Return the EWMA cost estimate, or ``None`` if no observations.

        Args:
            model_key: Composite key in ``provider::model`` format.
            complexity_class: Task complexity bucket.

        Returns:
            EWMA cost in USD, or None.
        """
        tracker = self._get_cost_tracker(model_key, complexity_class)
        if tracker is None or not tracker.is_trained:
            return None
        return tracker.current_estimate

    # ------------------------------------------------------------------
    # Percentile queries
    # ------------------------------------------------------------------

    def get_latency_p95(
        self, model_key: str, complexity_class: str = _FALLBACK_CLASS
    ) -> float:
        """Return the P95 latency from the rolling window.

        Args:
            model_key: Composite key in ``provider::model`` format.
            complexity_class: Task complexity bucket.

        Returns:
            P95 latency in milliseconds, or 0.0 if no data.
        """
        tracker = self._get_latency_tracker(model_key, complexity_class)
        if tracker is None:
            return 0.0
        return tracker.get_percentile(95)

    def get_latency_p99(
        self, model_key: str, complexity_class: str = _FALLBACK_CLASS
    ) -> float:
        """Return the P99 latency from the rolling window.

        Args:
            model_key: Composite key in ``provider::model`` format.
            complexity_class: Task complexity bucket.

        Returns:
            P99 latency in milliseconds, or 0.0 if no data.
        """
        tracker = self._get_latency_tracker(model_key, complexity_class)
        if tracker is None:
            return 0.0
        return tracker.get_percentile(99)

    # ------------------------------------------------------------------
    # Confidence
    # ------------------------------------------------------------------

    def get_confidence(
        self, model_key: str, complexity_class: str = _FALLBACK_CLASS
    ) -> float:
        """Return a [0, 1] confidence score based on sample count.

        Confidence = n / (n + 10), so it reaches 0.5 at 10 samples and
        approaches 1.0 asymptotically.

        Args:
            model_key: Composite key in ``provider::model`` format.
            complexity_class: Task complexity bucket.

        Returns:
            Confidence score in [0, 1].
        """
        tracker = self._get_latency_tracker(model_key, complexity_class)
        if tracker is None:
            return 0.0
        n = tracker.sample_count
        return n / (n + 10)

    # ------------------------------------------------------------------
    # Counts / inventory
    # ------------------------------------------------------------------

    def observation_count(
        self, model_key: str, complexity_class: str = _FALLBACK_CLASS
    ) -> int:
        """Return accepted sample count for a specific bucket.

        Args:
            model_key: Composite key in ``provider::model`` format.
            complexity_class: Task complexity bucket.

        Returns:
            Number of accepted observations (0 for unseen combinations).
        """
        tracker = self._get_latency_tracker(model_key, complexity_class)
        if tracker is None:
            return 0
        return tracker.sample_count

    def known_models(self) -> list[str]:
        """Return all model keys that have at least one observation.

        Returns unique model keys across all complexity buckets.

        Returns:
            List of model key strings.
        """
        return list({mk for mk, _ in self._latency_trackers})

    def known_buckets(self) -> list[tuple[str, str]]:
        """Return all (model_key, complexity_class) pairs with observations.

        Returns:
            List of (model_key, complexity_class) tuples.
        """
        return list(self._latency_trackers.keys())

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def tracker_summary(
        self, model_key: str, complexity_class: str = _FALLBACK_CLASS
    ) -> dict:
        """Return the AdaptiveEWMA diagnostics dict for a bucket.

        Args:
            model_key: Composite key in ``provider::model`` format.
            complexity_class: Task complexity bucket.

        Returns:
            Dict from ``AdaptiveEWMA.summary()``, or empty dict if unseen.
        """
        tracker = self._get_latency_tracker(model_key, complexity_class)
        if tracker is None:
            return {}
        return tracker.summary()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _new_tracker(self) -> "AdaptiveEWMA":
        from kortex.core.adaptive_ewma import AdaptiveEWMA

        return AdaptiveEWMA(
            fixed_alpha=self.alpha,
            window_size=self.window_size,
        )

    def _get_latency_tracker(
        self, model_key: str, complexity_class: str
    ) -> "AdaptiveEWMA | None":
        """Look up a latency tracker with fallback to the medium bucket."""
        bucket = (model_key, complexity_class)
        if bucket in self._latency_trackers:
            return self._latency_trackers[bucket]
        # Fallback: try the medium bucket (covers legacy code that never
        # passed complexity_class during update calls)
        if complexity_class != _FALLBACK_CLASS:
            fallback = (model_key, _FALLBACK_CLASS)
            if fallback in self._latency_trackers:
                return self._latency_trackers[fallback]
        return None

    def _get_cost_tracker(
        self, model_key: str, complexity_class: str
    ) -> "AdaptiveEWMA | None":
        """Look up a cost tracker with fallback to the medium bucket."""
        bucket = (model_key, complexity_class)
        if bucket in self._cost_trackers:
            return self._cost_trackers[bucket]
        if complexity_class != _FALLBACK_CLASS:
            fallback = (model_key, _FALLBACK_CLASS)
            if fallback in self._cost_trackers:
                return self._cost_trackers[fallback]
        return None
