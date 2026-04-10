"""Tests for the task routing engine."""

from __future__ import annotations

import pytest

from kortex.core.exceptions import RoutingFailedError
from kortex.core.router import (
    HeuristicRoutingStrategy,
    ProviderModel,
    Router,
    RoutingStrategy,
)
from kortex.core.types import RoutingDecision, TaskSpec


# --- Fixtures ---

def _fast_cheap() -> ProviderModel:
    return ProviderModel(
        provider="openai",
        model="gpt-4o-mini",
        cost_per_1k_input_tokens=0.00015,
        cost_per_1k_output_tokens=0.0006,
        avg_latency_ms=200,
        capabilities=["reasoning", "code_generation"],
        max_context_tokens=128_000,
        tier="fast",
    )


def _balanced_mid() -> ProviderModel:
    return ProviderModel(
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        cost_per_1k_input_tokens=0.003,
        cost_per_1k_output_tokens=0.015,
        avg_latency_ms=800,
        capabilities=["reasoning", "code_generation", "vision"],
        max_context_tokens=200_000,
        tier="balanced",
    )


def _powerful_expensive() -> ProviderModel:
    return ProviderModel(
        provider="anthropic",
        model="claude-opus-4-20250514",
        cost_per_1k_input_tokens=0.015,
        cost_per_1k_output_tokens=0.075,
        avg_latency_ms=2000,
        capabilities=["reasoning", "code_generation", "vision", "data_processing"],
        max_context_tokens=200_000,
        tier="powerful",
    )


def _all_models() -> list[ProviderModel]:
    return [_fast_cheap(), _balanced_mid(), _powerful_expensive()]


def _make_router(models: list[ProviderModel] | None = None) -> Router:
    router = Router()
    for m in (models or _all_models()):
        router.register_model(m)
    return router


# --- Tests ---


class TestSimpleRouting:
    @pytest.mark.asyncio
    async def test_simple_task_routes_to_cheapest_fast(self) -> None:
        router = _make_router()
        task = TaskSpec(content="Say hello", complexity_hint="simple")
        decision = await router.route(task)
        assert decision.chosen_model == "gpt-4o-mini"
        assert decision.chosen_provider == "openai"
        assert "fast" in decision.reasoning.lower()


class TestComplexRouting:
    @pytest.mark.asyncio
    async def test_complex_task_routes_to_powerful_tier(self) -> None:
        router = _make_router()
        task = TaskSpec(content="Analyze this codebase", complexity_hint="complex")
        decision = await router.route(task)
        assert decision.chosen_model == "claude-opus-4-20250514"
        assert "powerful" in decision.reasoning.lower()


class TestCostFilter:
    @pytest.mark.asyncio
    async def test_cost_ceiling_filters_expensive(self) -> None:
        router = _make_router()
        # Set a ceiling that only the fast model can meet
        task = TaskSpec(
            content="Quick question",
            complexity_hint="complex",
            cost_ceiling_usd=0.001,
        )
        decision = await router.route(task)
        # The powerful and balanced models are too expensive; falls back
        assert decision.chosen_model == "gpt-4o-mini"


class TestLatencyFilter:
    @pytest.mark.asyncio
    async def test_latency_sla_filters_slow(self) -> None:
        router = _make_router()
        task = TaskSpec(
            content="Fast task",
            complexity_hint="complex",
            latency_sla_ms=500,
        )
        decision = await router.route(task)
        # Only the fast model survives the 500ms SLA
        assert decision.chosen_model == "gpt-4o-mini"


class TestCapabilityFilter:
    @pytest.mark.asyncio
    async def test_missing_capabilities_filtered(self) -> None:
        router = _make_router()
        task = TaskSpec(
            content="Analyze deeply",
            complexity_hint="simple",
            required_capabilities=["data_processing"],
        )
        decision = await router.route(task)
        # Only the powerful model has "data_processing"
        assert decision.chosen_model == "claude-opus-4-20250514"


class TestNoValidCandidates:
    @pytest.mark.asyncio
    async def test_raises_routing_failed_error(self) -> None:
        router = _make_router()
        task = TaskSpec(
            content="Impossible task",
            cost_ceiling_usd=0.0001,
            latency_sla_ms=10,
            required_capabilities=["audio"],
        )
        with pytest.raises(RoutingFailedError, match="No models available"):
            await router.route(task)

    @pytest.mark.asyncio
    async def test_error_message_includes_filter_details(self) -> None:
        router = _make_router()
        task = TaskSpec(
            content="Impossible",
            cost_ceiling_usd=0.0001,
        )
        with pytest.raises(RoutingFailedError, match="cost ceiling"):
            await router.route(task)


class TestFallbackModel:
    @pytest.mark.asyncio
    async def test_fallback_is_populated_and_different(self) -> None:
        router = _make_router()
        task = TaskSpec(content="Moderate task", complexity_hint="moderate")
        decision = await router.route(task)
        assert decision.fallback_model is not None
        assert decision.fallback_model != decision.chosen_model

    @pytest.mark.asyncio
    async def test_fallback_is_none_with_single_candidate(self) -> None:
        router = _make_router([_fast_cheap()])
        task = TaskSpec(content="Only option", complexity_hint="simple")
        decision = await router.route(task)
        assert decision.fallback_model is None


class TestStrategyPattern:
    @pytest.mark.asyncio
    async def test_custom_strategy_is_used(self) -> None:
        class AlwaysPickFirst:
            async def select(
                self, task: TaskSpec, candidates: list[ProviderModel]
            ) -> RoutingDecision:
                m = candidates[0]
                return RoutingDecision(
                    task_id=task.task_id,
                    chosen_provider=m.provider,
                    chosen_model=m.model,
                    reasoning="Always pick first",
                    estimated_cost_usd=m.estimated_cost(),
                    estimated_latency_ms=m.avg_latency_ms,
                )

        router = Router(strategy=AlwaysPickFirst())
        model = _powerful_expensive()
        router.register_model(model)
        task = TaskSpec(content="test", complexity_hint="simple")
        decision = await router.route(task)
        assert decision.chosen_model == model.model
        assert decision.reasoning == "Always pick first"


class TestRouterRegistry:
    def test_register_and_remove(self) -> None:
        router = Router()
        model = _fast_cheap()
        router.register_model(model)
        assert len(router.models) == 1
        router.remove_model(model.model)
        assert len(router.models) == 0

    def test_remove_nonexistent_is_noop(self) -> None:
        router = Router()
        router.remove_model("nonexistent")  # should not raise
