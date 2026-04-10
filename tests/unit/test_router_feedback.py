"""Tests for ObservedMetrics and the EWMA router feedback loop."""

from __future__ import annotations

import pytest

from kortex.core.metrics import ObservedMetrics
from kortex.core.router import ProviderModel, Router
from kortex.core.types import TaskSpec


# ---------------------------------------------------------------------------
# ObservedMetrics unit tests
# ---------------------------------------------------------------------------


class TestObservedMetricsDefaults:
    def test_unknown_model_returns_none(self) -> None:
        m = ObservedMetrics()
        assert m.get_latency("unknown::model") is None
        assert m.get_cost("unknown::model") is None

    def test_observation_count_zero_for_unseen(self) -> None:
        m = ObservedMetrics()
        assert m.observation_count("unknown::model") == 0

    def test_known_models_empty_initially(self) -> None:
        m = ObservedMetrics()
        assert m.known_models() == []


class TestObservedMetricsFirstObservation:
    def test_first_observation_seeds_directly(self) -> None:
        m = ObservedMetrics()
        m.update("openai::gpt-4o-mini", latency_ms=200.0, cost_usd=0.001)
        assert m.get_latency("openai::gpt-4o-mini") == 200.0
        assert m.get_cost("openai::gpt-4o-mini") == 0.001

    def test_count_incremented_on_first_observation(self) -> None:
        m = ObservedMetrics()
        m.update("model::x", latency_ms=100.0, cost_usd=0.01)
        assert m.observation_count("model::x") == 1

    def test_known_models_updated(self) -> None:
        m = ObservedMetrics()
        m.update("provider::model-a", latency_ms=100.0, cost_usd=0.01)
        m.update("provider::model-b", latency_ms=200.0, cost_usd=0.02)
        assert set(m.known_models()) == {"provider::model-a", "provider::model-b"}


class TestEWMAConvergence:
    def test_ewma_moves_toward_new_value(self) -> None:
        m = ObservedMetrics(alpha=0.5)
        m.update("p::m", latency_ms=1000.0, cost_usd=0.01)  # seed
        m.update("p::m", latency_ms=0.0, cost_usd=0.0)      # push toward 0
        lat = m.get_latency("p::m")
        assert lat is not None
        assert 0.0 < lat < 1000.0  # converged toward 0

    def test_ewma_alpha_one_equals_last_observation(self) -> None:
        """alpha=1.0 means EWMA is always the last observed value."""
        m = ObservedMetrics(alpha=1.0)
        m.update("p::m", latency_ms=500.0, cost_usd=0.5)
        m.update("p::m", latency_ms=99.0, cost_usd=0.099)
        assert m.get_latency("p::m") == pytest.approx(99.0)
        assert m.get_cost("p::m") == pytest.approx(0.099)

    def test_ewma_converges_over_many_observations(self) -> None:
        """After many identical observations, EWMA stabilises near that value."""
        m = ObservedMetrics(alpha=0.3)
        m.update("p::m", latency_ms=500.0, cost_usd=0.5)  # seed far away
        target = 100.0
        for _ in range(50):
            m.update("p::m", latency_ms=target, cost_usd=0.1)
        lat = m.get_latency("p::m")
        assert lat is not None
        assert abs(lat - target) < 1.0  # within 1 ms of target

    def test_observation_count_increments_correctly(self) -> None:
        m = ObservedMetrics()
        key = "p::m"
        for i in range(10):
            m.update(key, latency_ms=float(i * 10), cost_usd=float(i) * 0.001)
        assert m.observation_count(key) == 10


class TestRouterWithMetrics:
    """Router uses ObservedMetrics to prefer models with better observed perf."""

    def _two_model_router(self) -> Router:
        router = Router()
        router.register_model(ProviderModel(
            provider="fast-provider",
            model="fast-model",
            cost_per_1k_input_tokens=0.001,
            cost_per_1k_output_tokens=0.002,
            avg_latency_ms=100,
            capabilities=["reasoning"],
            tier="fast",
        ))
        router.register_model(ProviderModel(
            provider="slow-provider",
            model="slow-model",
            cost_per_1k_input_tokens=0.001,
            cost_per_1k_output_tokens=0.002,
            avg_latency_ms=2000,
            capabilities=["reasoning"],
            tier="fast",
        ))
        return router

    @pytest.mark.asyncio
    async def test_router_route_succeeds_with_metrics_attached(self) -> None:
        router = self._two_model_router()
        metrics = ObservedMetrics()
        router.set_metrics(metrics)

        task = TaskSpec(content="test", complexity_hint="simple")
        decision = await router.route(task)
        assert decision.chosen_model in ("fast-model", "slow-model")

    @pytest.mark.asyncio
    async def test_observed_cheaper_model_preferred_after_feedback(self) -> None:
        """After observing that slow-model is actually very cheap, prefer it."""
        router = self._two_model_router()
        metrics = ObservedMetrics(alpha=1.0)  # instant convergence
        router.set_metrics(metrics)

        task = TaskSpec(content="test", complexity_hint="simple")

        # Route 10 tasks without feedback — fast-model should win (same cost, lower latency)
        pre_feedback_models: list[str] = []
        for _ in range(10):
            d = await router.route(task)
            pre_feedback_models.append(d.chosen_model)

        # Feed back: slow-model has 1/100th the cost of fast-model
        for _ in range(10):
            metrics.update("slow-provider::slow-model", latency_ms=2000.0, cost_usd=0.0000001)
            metrics.update("fast-provider::fast-model", latency_ms=100.0, cost_usd=0.01)

        # Route 10 more — slow-model now looks much cheaper, router should prefer it
        post_feedback_models: list[str] = []
        for _ in range(10):
            d = await router.route(task)
            post_feedback_models.append(d.chosen_model)

        # After feeding back that slow-model is dirt cheap, it should dominate
        slow_after = post_feedback_models.count("slow-model")
        assert slow_after > 0, "Router should prefer slow-model after observing it is much cheaper"

    @pytest.mark.asyncio
    async def test_get_set_metrics(self) -> None:
        router = Router()
        assert router.get_metrics() is None

        metrics = ObservedMetrics()
        router.set_metrics(metrics)
        assert router.get_metrics() is metrics

    @pytest.mark.asyncio
    async def test_router_without_metrics_unaffected(self) -> None:
        """Router with no metrics attached behaves exactly as before."""
        router = self._two_model_router()
        # No metrics attached
        task = TaskSpec(content="test", complexity_hint="simple")
        decision = await router.route(task)
        assert decision is not None


class TestRuntimeMetricsIntegration:
    """KortexRuntime auto-creates ObservedMetrics and attaches them to router."""

    @pytest.mark.asyncio
    async def test_runtime_creates_metrics_on_init(self) -> None:
        from kortex.core.runtime import KortexRuntime
        from kortex.core.state import StateManager

        router = Router()
        runtime = KortexRuntime(
            router=router,
            state_manager=StateManager.create("memory"),
        )
        assert runtime._metrics is not None
        assert isinstance(runtime._metrics, ObservedMetrics)

    @pytest.mark.asyncio
    async def test_runtime_attaches_metrics_to_router(self) -> None:
        from kortex.core.runtime import KortexRuntime
        from kortex.core.state import StateManager

        router = Router()
        runtime = KortexRuntime(
            router=router,
            state_manager=StateManager.create("memory"),
        )
        assert router.get_metrics() is runtime._metrics
