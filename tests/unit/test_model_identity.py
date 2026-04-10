"""Unit tests for model identity and composite-key routing.

Verifies that:
- ModelIdentity produces correct composite keys
- ProviderModel.identity property works
- Router uses composite keys (no silent collisions)
- RoutingDecision includes chosen_model_identity
- ProviderRegistry.get_all_models() retains same-name models from different providers
- Fallback selection uses identity, not just model name
"""

from __future__ import annotations

import pytest

from kortex.core.types import ModelIdentity, RoutingDecision, TaskSpec
from kortex.core.router import ProviderModel, Router, HeuristicRoutingStrategy
from kortex.providers.registry import ProviderRegistry
from kortex.providers.base import GenericOpenAIConnector


# ---------------------------------------------------------------------------
# 1. ModelIdentity composite key
# ---------------------------------------------------------------------------


class TestModelIdentityKey:
    def test_basic_key(self):
        mid = ModelIdentity(provider="anthropic", model_name="claude-sonnet-4-20250514")
        assert mid.key == "anthropic::claude-sonnet-4-20250514"

    def test_key_with_version(self):
        mid = ModelIdentity(
            provider="openai", model_name="gpt-4", model_version="0613"
        )
        assert mid.key == "openai::gpt-4::0613"

    def test_key_without_version(self):
        mid = ModelIdentity(provider="openrouter", model_name="llama-3-70b")
        assert mid.key == "openrouter::llama-3-70b"

    def test_equality_same_key(self):
        a = ModelIdentity(provider="x", model_name="m")
        b = ModelIdentity(provider="x", model_name="m")
        assert a == b
        assert hash(a) == hash(b)

    def test_inequality_different_provider(self):
        a = ModelIdentity(provider="provider-a", model_name="llama-3-70b")
        b = ModelIdentity(provider="provider-b", model_name="llama-3-70b")
        assert a != b

    def test_str_returns_key(self):
        mid = ModelIdentity(provider="x", model_name="y")
        assert str(mid) == "x::y"

    def test_usable_as_dict_key(self):
        a = ModelIdentity(provider="x", model_name="m")
        b = ModelIdentity(provider="x", model_name="m")
        d = {a: "value"}
        assert d[b] == "value"


# ---------------------------------------------------------------------------
# 2. ProviderModel.identity property
# ---------------------------------------------------------------------------


class TestProviderModelIdentity:
    def test_identity_from_provider_model(self):
        pm = ProviderModel(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            cost_per_1k_input_tokens=0.003,
            cost_per_1k_output_tokens=0.015,
            avg_latency_ms=500,
        )
        assert pm.identity.key == "anthropic::claude-sonnet-4-20250514"

    def test_identity_with_version(self):
        pm = ProviderModel(
            provider="openai",
            model="gpt-4",
            cost_per_1k_input_tokens=0.03,
            cost_per_1k_output_tokens=0.06,
            avg_latency_ms=800,
            model_version="0613",
        )
        assert pm.identity.key == "openai::gpt-4::0613"

    def test_same_model_different_providers_different_identity(self):
        a = ProviderModel(
            provider="openrouter",
            model="llama-3-70b",
            cost_per_1k_input_tokens=0.001,
            cost_per_1k_output_tokens=0.001,
            avg_latency_ms=300,
        )
        b = ProviderModel(
            provider="together",
            model="llama-3-70b",
            cost_per_1k_input_tokens=0.0008,
            cost_per_1k_output_tokens=0.0008,
            avg_latency_ms=250,
        )
        assert a.identity != b.identity
        assert a.identity.key != b.identity.key


# ---------------------------------------------------------------------------
# 3. Router uses composite keys — no silent collisions
# ---------------------------------------------------------------------------


class TestRouterCompositeKeys:
    def test_register_same_model_different_providers_both_survive(self):
        router = Router()
        model_a = ProviderModel(
            provider="provider-a",
            model="llama-3-70b",
            cost_per_1k_input_tokens=0.001,
            cost_per_1k_output_tokens=0.002,
            avg_latency_ms=300,
            capabilities=["reasoning"],
        )
        model_b = ProviderModel(
            provider="provider-b",
            model="llama-3-70b",
            cost_per_1k_input_tokens=0.0005,
            cost_per_1k_output_tokens=0.001,
            avg_latency_ms=250,
            capabilities=["reasoning"],
        )

        router.register_model(model_a)
        router.register_model(model_b)

        assert len(router.models) == 2

    def test_register_same_provider_same_model_overwrites(self):
        router = Router()
        model_v1 = ProviderModel(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            cost_per_1k_input_tokens=0.003,
            cost_per_1k_output_tokens=0.015,
            avg_latency_ms=500,
        )
        model_v2 = ProviderModel(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            cost_per_1k_input_tokens=0.002,
            cost_per_1k_output_tokens=0.010,
            avg_latency_ms=400,
        )

        router.register_model(model_v1)
        router.register_model(model_v2)

        # Same identity → overwrite, still 1 model
        assert len(router.models) == 1
        assert router.models[0].cost_per_1k_input_tokens == 0.002

    def test_remove_model_by_provider(self):
        router = Router()
        model_a = ProviderModel(
            provider="provider-a",
            model="llama-3-70b",
            cost_per_1k_input_tokens=0.001,
            cost_per_1k_output_tokens=0.002,
            avg_latency_ms=300,
        )
        model_b = ProviderModel(
            provider="provider-b",
            model="llama-3-70b",
            cost_per_1k_input_tokens=0.0005,
            cost_per_1k_output_tokens=0.001,
            avg_latency_ms=250,
        )
        router.register_model(model_a)
        router.register_model(model_b)

        router.remove_model("llama-3-70b", provider="provider-a")

        assert len(router.models) == 1
        assert router.models[0].provider == "provider-b"

    def test_remove_model_backward_compat_no_provider(self):
        router = Router()
        model = ProviderModel(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            cost_per_1k_input_tokens=0.003,
            cost_per_1k_output_tokens=0.015,
            avg_latency_ms=500,
        )
        router.register_model(model)
        router.remove_model("claude-sonnet-4-20250514")
        assert len(router.models) == 0


# ---------------------------------------------------------------------------
# 4. Routing produces chosen_model_identity
# ---------------------------------------------------------------------------


class TestRoutingDecisionIdentity:
    @pytest.mark.asyncio
    async def test_routing_decision_has_identity(self):
        router = Router()
        router.register_model(ProviderModel(
            provider="test-cloud",
            model="fast-model",
            cost_per_1k_input_tokens=0.001,
            cost_per_1k_output_tokens=0.002,
            avg_latency_ms=200,
            capabilities=["reasoning"],
            tier="fast",
        ))

        task = TaskSpec(content="test", complexity_hint="simple")
        decision = await router.route(task)

        assert decision.chosen_model_identity == "test-cloud::fast-model"

    @pytest.mark.asyncio
    async def test_routing_picks_cheaper_same_model_different_provider(self):
        router = Router()
        expensive = ProviderModel(
            provider="expensive-cloud",
            model="llama-3-70b",
            cost_per_1k_input_tokens=0.01,
            cost_per_1k_output_tokens=0.02,
            avg_latency_ms=300,
            capabilities=["reasoning"],
            tier="fast",
        )
        cheap = ProviderModel(
            provider="cheap-cloud",
            model="llama-3-70b",
            cost_per_1k_input_tokens=0.001,
            cost_per_1k_output_tokens=0.002,
            avg_latency_ms=250,
            capabilities=["reasoning"],
            tier="fast",
        )
        router.register_model(expensive)
        router.register_model(cheap)

        task = TaskSpec(content="test", complexity_hint="simple")
        decision = await router.route(task)

        # Should pick the cheaper one
        assert decision.chosen_provider == "cheap-cloud"
        assert decision.chosen_model_identity == "cheap-cloud::llama-3-70b"

    @pytest.mark.asyncio
    async def test_fallback_can_be_different_provider(self):
        router = Router()
        primary = ProviderModel(
            provider="provider-a",
            model="llama-3-70b",
            cost_per_1k_input_tokens=0.0005,
            cost_per_1k_output_tokens=0.001,
            avg_latency_ms=200,
            capabilities=["reasoning"],
            tier="fast",
        )
        fallback = ProviderModel(
            provider="provider-b",
            model="llama-3-70b",
            cost_per_1k_input_tokens=0.001,
            cost_per_1k_output_tokens=0.002,
            avg_latency_ms=300,
            capabilities=["reasoning"],
            tier="fast",
        )
        router.register_model(primary)
        router.register_model(fallback)

        task = TaskSpec(content="test", complexity_hint="simple")
        decision = await router.route(task)

        # Primary should be cheapest, fallback should be the other
        assert decision.chosen_provider == "provider-a"
        assert decision.fallback_model == "llama-3-70b"
        assert decision.fallback_provider == "provider-b"


# ---------------------------------------------------------------------------
# 5. ProviderRegistry retains same-name models from different providers
# ---------------------------------------------------------------------------


class TestRegistryNoDedup:
    def test_get_all_models_retains_same_name_different_providers(self):
        registry = ProviderRegistry()

        registry.register_openai_compatible(
            name="provider-a",
            base_url="https://a.test/v1",
            api_key="key-a",
            models=[
                ProviderModel(
                    provider="provider-a",
                    model="llama-3-70b",
                    cost_per_1k_input_tokens=0.001,
                    cost_per_1k_output_tokens=0.002,
                    avg_latency_ms=300,
                ),
            ],
        )
        registry.register_openai_compatible(
            name="provider-b",
            base_url="https://b.test/v1",
            api_key="key-b",
            models=[
                ProviderModel(
                    provider="provider-b",
                    model="llama-3-70b",
                    cost_per_1k_input_tokens=0.0005,
                    cost_per_1k_output_tokens=0.001,
                    avg_latency_ms=250,
                ),
            ],
        )

        all_models = registry.get_all_models()
        assert len(all_models) == 2

        providers = {m.provider for m in all_models}
        assert providers == {"provider-a", "provider-b"}

    def test_get_all_models_deduplicates_same_provider_same_model(self):
        """Same provider registering same model twice should deduplicate."""
        registry = ProviderRegistry()
        registry.register_openai_compatible(
            name="cloud",
            base_url="https://cloud.test/v1",
            api_key="key",
            models=[
                ProviderModel(
                    provider="cloud",
                    model="model-x",
                    cost_per_1k_input_tokens=0.001,
                    cost_per_1k_output_tokens=0.002,
                    avg_latency_ms=300,
                ),
                ProviderModel(
                    provider="cloud",
                    model="model-x",
                    cost_per_1k_input_tokens=0.001,
                    cost_per_1k_output_tokens=0.002,
                    avg_latency_ms=300,
                ),
            ],
        )

        all_models = registry.get_all_models()
        assert len(all_models) == 1


# ---------------------------------------------------------------------------
# 6. Backward compatibility — RoutingDecision default
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    def test_routing_decision_identity_defaults_empty(self):
        """Old code that creates RoutingDecision without identity still works."""
        rd = RoutingDecision(
            task_id="t1",
            chosen_provider="p",
            chosen_model="m",
            reasoning="test",
            estimated_cost_usd=0.01,
            estimated_latency_ms=100.0,
        )
        assert rd.chosen_model_identity == ""
        assert rd.fallback_provider is None
