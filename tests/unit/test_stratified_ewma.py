"""Tests for stratified EWMA — Task 2: per-complexity-class estimates."""

from __future__ import annotations

import pytest

from kortex.core.metrics import ObservedMetrics
from kortex.core.router import ProviderModel, Router
from kortex.core.types import TaskSpec


# ---------------------------------------------------------------------------
# TaskSpec.complexity_class auto-inference
# ---------------------------------------------------------------------------


class TestComplexityClassInference:
    def test_short_content_infers_simple(self) -> None:
        task = TaskSpec(content="Hi")  # <500 chars
        assert task.complexity_class == "simple"

    def test_medium_content_infers_medium(self) -> None:
        task = TaskSpec(content="x" * 600)  # 500-2000 chars
        assert task.complexity_class == "medium"

    def test_long_content_infers_complex(self) -> None:
        task = TaskSpec(content="y" * 2001)  # >2000 chars
        assert task.complexity_class == "complex"

    def test_boundary_499_is_simple(self) -> None:
        task = TaskSpec(content="a" * 499)
        assert task.complexity_class == "simple"

    def test_boundary_500_is_medium(self) -> None:
        task = TaskSpec(content="a" * 500)
        assert task.complexity_class == "medium"

    def test_boundary_1999_is_medium(self) -> None:
        task = TaskSpec(content="a" * 1999)
        assert task.complexity_class == "medium"

    def test_boundary_2000_is_complex(self) -> None:
        task = TaskSpec(content="a" * 2000)
        assert task.complexity_class == "complex"

    def test_explicit_complexity_class_overrides_inference(self) -> None:
        task = TaskSpec(content="short", complexity_class="complex")
        assert task.complexity_class == "complex"

    def test_explicit_medium_preserved(self) -> None:
        task = TaskSpec(content="x" * 2500, complexity_class="medium")
        assert task.complexity_class == "medium"

    def test_complexity_class_independent_of_complexity_hint(self) -> None:
        task = TaskSpec(
            content="a" * 100,
            complexity_hint="complex",  # hint unrelated to class
        )
        assert task.complexity_class == "simple"  # inferred from length
        assert task.complexity_hint == "complex"


# ---------------------------------------------------------------------------
# Stratified ObservedMetrics — separate estimates per class
# ---------------------------------------------------------------------------


class TestStratifiedMetrics:
    def test_simple_and_complex_tracked_separately(self) -> None:
        m = ObservedMetrics(alpha=1.0)  # instant convergence
        m.update("p::model", latency_ms=100.0, cost_usd=0.001, complexity_class="simple")
        m.update("p::model", latency_ms=900.0, cost_usd=0.009, complexity_class="complex")

        assert m.get_latency("p::model", "simple") == pytest.approx(100.0)
        assert m.get_latency("p::model", "complex") == pytest.approx(900.0)

    def test_medium_and_simple_separate(self) -> None:
        m = ObservedMetrics(alpha=1.0)
        m.update("p::m", latency_ms=200.0, cost_usd=0.002, complexity_class="medium")
        m.update("p::m", latency_ms=50.0, cost_usd=0.001, complexity_class="simple")

        assert m.get_latency("p::m", "medium") == pytest.approx(200.0)
        assert m.get_latency("p::m", "simple") == pytest.approx(50.0)

    def test_unknown_class_returns_none(self) -> None:
        m = ObservedMetrics()
        m.update("p::m", latency_ms=100.0, cost_usd=0.001, complexity_class="simple")
        # No "complex" data, no "medium" fallback → None
        assert m.get_latency("p::m", "complex") is None

    def test_fallback_to_medium_when_class_missing(self) -> None:
        m = ObservedMetrics(alpha=1.0)
        m.update("p::m", latency_ms=300.0, cost_usd=0.003, complexity_class="medium")

        # Requesting "simple" — no simple data, falls back to medium
        assert m.get_latency("p::m", "simple") == pytest.approx(300.0)

    def test_no_fallback_when_class_already_medium(self) -> None:
        """Requesting "medium" that doesn't exist returns None (no infinite loop)."""
        m = ObservedMetrics()
        assert m.get_latency("p::m", "medium") is None

    def test_observation_count_per_bucket(self) -> None:
        m = ObservedMetrics()
        for _ in range(3):
            m.update("p::m", 100.0, 0.001, "simple")
        for _ in range(5):
            m.update("p::m", 200.0, 0.002, "complex")

        assert m.observation_count("p::m", "simple") == 3
        assert m.observation_count("p::m", "complex") == 5
        assert m.observation_count("p::m", "medium") == 0

    def test_known_models_returns_unique_keys(self) -> None:
        m = ObservedMetrics()
        m.update("provider::modelA", 100.0, 0.001, "simple")
        m.update("provider::modelA", 200.0, 0.002, "complex")
        m.update("provider::modelB", 150.0, 0.001, "medium")

        assert set(m.known_models()) == {"provider::modelA", "provider::modelB"}

    def test_known_buckets(self) -> None:
        m = ObservedMetrics()
        m.update("p::m", 100.0, 0.001, "simple")
        m.update("p::m", 200.0, 0.002, "complex")

        buckets = set(m.known_buckets())
        assert ("p::m", "simple") in buckets
        assert ("p::m", "complex") in buckets

    def test_default_complexity_class_is_medium(self) -> None:
        m = ObservedMetrics(alpha=1.0)
        m.update("p::m", 250.0, 0.002)  # no complexity_class → defaults to "medium"
        assert m.get_latency("p::m", "medium") == pytest.approx(250.0)
        assert m.get_latency("p::m") == pytest.approx(250.0)  # default arg also medium


# ---------------------------------------------------------------------------
# Router uses stratified estimates
# ---------------------------------------------------------------------------


class TestRouterStratifiedEstimates:
    def _make_router(self) -> Router:
        router = Router()
        router.register_model(ProviderModel(
            provider="fast-p",
            model="fast-m",
            cost_per_1k_input_tokens=0.001,
            cost_per_1k_output_tokens=0.002,
            avg_latency_ms=100,
            capabilities=["reasoning"],
            tier="fast",
        ))
        router.register_model(ProviderModel(
            provider="slow-p",
            model="slow-m",
            cost_per_1k_input_tokens=0.001,
            cost_per_1k_output_tokens=0.002,
            avg_latency_ms=2000,
            capabilities=["reasoning"],
            tier="fast",
        ))
        return router

    @pytest.mark.asyncio
    async def test_router_uses_complexity_class_from_task(self) -> None:
        router = self._make_router()
        metrics = ObservedMetrics(alpha=1.0)
        router.set_metrics(metrics)

        # Feed back: slow-m is very cheap for SIMPLE tasks
        metrics.update("slow-p::slow-m", 100.0, 0.000001, "simple")
        metrics.update("fast-p::fast-m", 100.0, 0.01, "simple")

        task = TaskSpec(content="Hi")  # complexity_class="simple" (len < 500)
        decision = await router.route(task)
        # slow-m appears much cheaper for "simple" → router should prefer it
        assert decision.chosen_model == "slow-m"

    @pytest.mark.asyncio
    async def test_router_falls_back_to_medium_bucket(self) -> None:
        """Observations in "medium" bucket are used when task class is "simple"."""
        router = self._make_router()
        metrics = ObservedMetrics(alpha=1.0)
        router.set_metrics(metrics)

        # Only medium bucket populated
        metrics.update("slow-p::slow-m", 100.0, 0.000001, "medium")
        metrics.update("fast-p::fast-m", 100.0, 0.01, "medium")

        task = TaskSpec(content="Hi")  # "simple" → fallback to "medium"
        decision = await router.route(task)
        assert decision.chosen_model == "slow-m"

    @pytest.mark.asyncio
    async def test_simple_convergence_faster_than_complex(self) -> None:
        """Simple tasks should converge faster due to lower variance."""
        metrics_simple = ObservedMetrics()
        metrics_complex = ObservedMetrics()

        # Feed 15 samples at the target for simple
        for _ in range(15):
            metrics_simple.update("p::m", 100.0, 0.001, "simple")

        # Feed same count for complex
        for _ in range(15):
            metrics_complex.update("p::m", 100.0, 0.001, "complex")

        conf_simple = metrics_simple.get_confidence("p::m", "simple")
        conf_complex = metrics_complex.get_confidence("p::m", "complex")
        # Same sample count → same confidence score formula
        assert conf_simple == pytest.approx(conf_complex, rel=0.01)


# ---------------------------------------------------------------------------
# get_confidence
# ---------------------------------------------------------------------------


class TestGetConfidence:
    def test_zero_confidence_when_no_data(self) -> None:
        m = ObservedMetrics()
        assert m.get_confidence("p::m", "simple") == 0.0

    def test_confidence_increases_with_samples(self) -> None:
        m = ObservedMetrics()
        prev = 0.0
        for i in range(1, 21):
            m.update("p::m", 100.0, 0.001, "simple")
            conf = m.get_confidence("p::m", "simple")
            assert conf > prev
            prev = conf

    def test_confidence_formula(self) -> None:
        m = ObservedMetrics()
        for _ in range(10):
            m.update("p::m", 100.0, 0.001)
        # n=10, confidence = 10/(10+10) = 0.5
        assert m.get_confidence("p::m") == pytest.approx(0.5)

    def test_confidence_approaches_1(self) -> None:
        m = ObservedMetrics()
        for _ in range(1000):
            m.update("p::m", 100.0, 0.001)
        assert m.get_confidence("p::m") > 0.99
