"""Tests for the composable routing policy engine.

Covers RoutingPolicy, PolicyRouter, constraint filtering, objective scoring,
fallback selection, serialization, and Router integration.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from kortex.core.exceptions import RoutingFailedError
from kortex.core.policy import (
    EliminatedCandidate,
    FallbackRule,
    PolicyEvaluation,
    PolicyRouter,
    RoutingConstraint,
    RoutingObjective,
    RoutingPolicy,
    ScoredCandidate,
)
from kortex.core.router import ProviderModel, Router
from kortex.core.types import TaskSpec


# ---------------------------------------------------------------------------
# Shared model fixtures
# ---------------------------------------------------------------------------


def _cheap_fast() -> ProviderModel:
    return ProviderModel(
        provider="local",
        model="tiny-7b",
        cost_per_1k_input_tokens=0.0001,
        cost_per_1k_output_tokens=0.0002,
        avg_latency_ms=50,
        capabilities=["reasoning"],
        tier="fast",
    )


def _balanced() -> ProviderModel:
    return ProviderModel(
        provider="anthropic",
        model="claude-sonnet",
        cost_per_1k_input_tokens=0.003,
        cost_per_1k_output_tokens=0.015,
        avg_latency_ms=800,
        capabilities=["reasoning", "code_generation", "content_generation"],
        tier="balanced",
    )


def _expensive_powerful() -> ProviderModel:
    return ProviderModel(
        provider="anthropic",
        model="claude-opus",
        cost_per_1k_input_tokens=0.015,
        cost_per_1k_output_tokens=0.075,
        avg_latency_ms=2000,
        capabilities=["reasoning", "code_generation", "content_generation", "vision", "analysis"],
        tier="powerful",
    )


def _openai_balanced() -> ProviderModel:
    return ProviderModel(
        provider="openai",
        model="gpt-4o",
        cost_per_1k_input_tokens=0.005,
        cost_per_1k_output_tokens=0.015,
        avg_latency_ms=600,
        capabilities=["reasoning", "code_generation", "content_generation"],
        tier="balanced",
    )


def _all_models() -> list[ProviderModel]:
    return [_cheap_fast(), _balanced(), _expensive_powerful(), _openai_balanced()]


def _task(content: str = "Test task", **kwargs) -> TaskSpec:
    return TaskSpec(content=content, **kwargs)


# ---------------------------------------------------------------------------
# 1. Default policy with no constraints selects cheapest model
# ---------------------------------------------------------------------------


class TestDefaultPolicy:
    @pytest.mark.asyncio
    async def test_default_selects_cheapest(self) -> None:
        policy = RoutingPolicy()
        router = PolicyRouter(policy, _all_models())
        result = await router.evaluate(_task())

        # Default objective is minimize="cost", so cheapest wins
        assert result.chosen.model == "tiny-7b"
        assert result.policy_name == "default"


# ---------------------------------------------------------------------------
# 2. Cost-optimized factory returns lowest cost candidate
# ---------------------------------------------------------------------------


class TestCostOptimized:
    @pytest.mark.asyncio
    async def test_cost_optimized_factory(self) -> None:
        policy = RoutingPolicy.cost_optimized()
        router = PolicyRouter(policy, _all_models())
        result = await router.evaluate(_task())

        assert result.chosen.model == "tiny-7b"
        assert result.chosen.provider == "local"


# ---------------------------------------------------------------------------
# 3. Latency-optimized factory returns lowest latency candidate
# ---------------------------------------------------------------------------


class TestLatencyOptimized:
    @pytest.mark.asyncio
    async def test_latency_optimized_factory(self) -> None:
        policy = RoutingPolicy.latency_optimized()
        router = PolicyRouter(policy, _all_models())
        result = await router.evaluate(_task())

        assert result.chosen.model == "tiny-7b"
        assert result.chosen.avg_latency_ms == 50


# ---------------------------------------------------------------------------
# 4. Quality-optimized factory prefers powerful tier
# ---------------------------------------------------------------------------


class TestQualityOptimized:
    @pytest.mark.asyncio
    async def test_quality_optimized_factory(self) -> None:
        policy = RoutingPolicy.quality_optimized()
        router = PolicyRouter(policy, _all_models())
        result = await router.evaluate(_task())

        assert result.chosen.tier == "powerful"
        assert result.chosen.model == "claude-opus"


# ---------------------------------------------------------------------------
# 5. Cost ceiling constraint eliminates expensive models
# ---------------------------------------------------------------------------


class TestCostCeiling:
    @pytest.mark.asyncio
    async def test_cost_ceiling_eliminates_expensive(self) -> None:
        policy = RoutingPolicy(
            constraints=RoutingConstraint(max_cost_usd=0.005),
        )
        router = PolicyRouter(policy, _all_models())
        result = await router.evaluate(_task())

        # Only cheap models should survive
        for candidate in result.all_candidates:
            assert candidate.model.estimated_cost() <= 0.005

        # Expensive models should be eliminated
        eliminated_models = {e.model.model for e in result.eliminated}
        assert "claude-opus" in eliminated_models


# ---------------------------------------------------------------------------
# 6. Latency constraint eliminates slow models
# ---------------------------------------------------------------------------


class TestLatencyConstraint:
    @pytest.mark.asyncio
    async def test_latency_constraint_eliminates_slow(self) -> None:
        policy = RoutingPolicy(
            constraints=RoutingConstraint(max_latency_ms=700),
        )
        router = PolicyRouter(policy, _all_models())
        result = await router.evaluate(_task())

        for candidate in result.all_candidates:
            assert candidate.model.avg_latency_ms <= 700

        eliminated_models = {e.model.model for e in result.eliminated}
        assert "claude-sonnet" in eliminated_models
        assert "claude-opus" in eliminated_models


# ---------------------------------------------------------------------------
# 7. Capability constraint eliminates models missing required capabilities
# ---------------------------------------------------------------------------


class TestCapabilityConstraint:
    @pytest.mark.asyncio
    async def test_capability_constraint_filters(self) -> None:
        policy = RoutingPolicy(
            constraints=RoutingConstraint(required_capabilities=["vision"]),
        )
        router = PolicyRouter(policy, _all_models())
        result = await router.evaluate(_task())

        # Only claude-opus has vision
        assert result.chosen.model == "claude-opus"
        assert len(result.all_candidates) == 1
        assert len(result.eliminated) == 3


# ---------------------------------------------------------------------------
# 8. Provider allow list only permits listed providers
# ---------------------------------------------------------------------------


class TestProviderAllowList:
    @pytest.mark.asyncio
    async def test_allowed_providers(self) -> None:
        policy = RoutingPolicy(
            constraints=RoutingConstraint(allowed_providers=["anthropic"]),
        )
        router = PolicyRouter(policy, _all_models())
        result = await router.evaluate(_task())

        for candidate in result.all_candidates:
            assert candidate.model.provider == "anthropic"

        eliminated_providers = {e.model.provider for e in result.eliminated}
        assert "local" in eliminated_providers
        assert "openai" in eliminated_providers


# ---------------------------------------------------------------------------
# 9. Provider deny list excludes specific providers
# ---------------------------------------------------------------------------


class TestProviderDenyList:
    @pytest.mark.asyncio
    async def test_denied_providers(self) -> None:
        policy = RoutingPolicy(
            constraints=RoutingConstraint(denied_providers=["openai"]),
        )
        router = PolicyRouter(policy, _all_models())
        result = await router.evaluate(_task())

        for candidate in result.all_candidates:
            assert candidate.model.provider != "openai"

        eliminated_models = {e.model.model for e in result.eliminated}
        assert "gpt-4o" in eliminated_models


# ---------------------------------------------------------------------------
# 10. Model deny list excludes specific models
# ---------------------------------------------------------------------------


class TestModelDenyList:
    @pytest.mark.asyncio
    async def test_denied_models(self) -> None:
        policy = RoutingPolicy(
            constraints=RoutingConstraint(denied_models=["claude-opus", "gpt-4o"]),
        )
        router = PolicyRouter(policy, _all_models())
        result = await router.evaluate(_task())

        candidate_models = {c.model.model for c in result.all_candidates}
        assert "claude-opus" not in candidate_models
        assert "gpt-4o" not in candidate_models
        assert len(result.eliminated) == 2


# ---------------------------------------------------------------------------
# 11. All candidates eliminated raises RoutingFailedError with clear explanation
# ---------------------------------------------------------------------------


class TestAllEliminated:
    @pytest.mark.asyncio
    async def test_all_eliminated_raises_with_details(self) -> None:
        policy = RoutingPolicy(
            constraints=RoutingConstraint(
                required_capabilities=["telekinesis"],
            ),
        )
        router = PolicyRouter(policy, _all_models())

        with pytest.raises(RoutingFailedError) as exc_info:
            await router.evaluate(_task())

        msg = str(exc_info.value)
        assert "eliminated all" in msg
        assert "4 candidate(s)" in msg
        # Each model's elimination reason should be listed
        assert "tiny-7b" in msg
        assert "claude-sonnet" in msg
        assert "claude-opus" in msg
        assert "gpt-4o" in msg
        assert "missing capabilities" in msg


# ---------------------------------------------------------------------------
# 12. Explanation string is human-readable
# ---------------------------------------------------------------------------


class TestExplanation:
    @pytest.mark.asyncio
    async def test_explanation_contains_chosen_and_eliminated(self) -> None:
        policy = RoutingPolicy(
            constraints=RoutingConstraint(max_latency_ms=700),
        )
        router = PolicyRouter(policy, _all_models())
        result = await router.evaluate(_task())

        # Should mention what was selected
        assert "Selected" in result.explanation
        assert result.chosen.provider in result.explanation
        assert result.chosen.model in result.explanation

        # Should mention what was eliminated
        assert "Eliminated" in result.explanation
        assert "model(s)" in result.explanation


# ---------------------------------------------------------------------------
# 13. ScoredCandidate list is sorted by score descending
# ---------------------------------------------------------------------------


class TestScoredCandidateOrder:
    @pytest.mark.asyncio
    async def test_candidates_sorted_descending(self) -> None:
        policy = RoutingPolicy()
        router = PolicyRouter(policy, _all_models())
        result = await router.evaluate(_task())

        scores = [c.score for c in result.all_candidates]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# 14. EliminatedCandidate includes specific reason for each model
# ---------------------------------------------------------------------------


class TestEliminatedReasons:
    @pytest.mark.asyncio
    async def test_elimination_reasons_are_specific(self) -> None:
        policy = RoutingPolicy(
            constraints=RoutingConstraint(
                max_cost_usd=0.001,
                denied_providers=["local"],
            ),
        )
        router = PolicyRouter(policy, _all_models())

        with pytest.raises(RoutingFailedError):
            await router.evaluate(_task())

        # Use a less restrictive policy to verify individual reasons
        policy2 = RoutingPolicy(
            constraints=RoutingConstraint(
                max_cost_usd=0.02,
                denied_providers=["local"],
            ),
        )
        router2 = PolicyRouter(policy2, _all_models())
        result = await router2.evaluate(_task())

        reasons = {e.model.model: e.reason for e in result.eliminated}

        # local/tiny-7b: denied provider
        assert "denied" in reasons.get("tiny-7b", "")

        # claude-opus exceeds max_cost_usd of 0.02
        assert "max_cost_usd" in reasons.get("claude-opus", "")


# ---------------------------------------------------------------------------
# 15. Fallback uses next_cheapest by default
# ---------------------------------------------------------------------------


class TestFallbackNextCheapest:
    @pytest.mark.asyncio
    async def test_default_fallback_is_next_cheapest(self) -> None:
        policy = RoutingPolicy()
        router = PolicyRouter(policy, _all_models())
        result = await router.evaluate(_task())

        assert result.fallback is not None
        # The chosen is tiny-7b (cheapest). Fallback should be next cheapest.
        # All remaining models are candidates; fallback = cheapest among non-chosen
        assert result.fallback.model != result.chosen.model


# ---------------------------------------------------------------------------
# 16. Fallback with explicit strategy uses specified model
# ---------------------------------------------------------------------------


class TestFallbackExplicit:
    @pytest.mark.asyncio
    async def test_explicit_fallback(self) -> None:
        policy = RoutingPolicy(
            fallback=FallbackRule(
                strategy="explicit",
                explicit_model_identity="openai::gpt-4o",
            ),
        )
        router = PolicyRouter(policy, _all_models())
        result = await router.evaluate(_task())

        assert result.fallback is not None
        assert result.fallback.model == "gpt-4o"
        assert result.fallback.provider == "openai"

    @pytest.mark.asyncio
    async def test_explicit_fallback_not_found_returns_none(self) -> None:
        policy = RoutingPolicy(
            fallback=FallbackRule(
                strategy="explicit",
                explicit_model_identity="nonexistent::model",
            ),
        )
        router = PolicyRouter(policy, _all_models())
        result = await router.evaluate(_task())

        assert result.fallback is None


# ---------------------------------------------------------------------------
# 17. Policy serializes to dict and back without data loss
# ---------------------------------------------------------------------------


class TestDictSerialization:
    def test_roundtrip(self) -> None:
        policy = RoutingPolicy(
            name="test_policy",
            description="A test policy",
            budget_ceiling_usd=1.50,
            constraints=RoutingConstraint(
                max_cost_usd=0.01,
                max_latency_ms=500,
                required_capabilities=["reasoning", "vision"],
                allowed_providers=["anthropic", "openai"],
                denied_providers=["local"],
                allowed_models=["claude-opus"],
                denied_models=["gpt-3"],
            ),
            objective=RoutingObjective(
                minimize="latency",
                prefer_tier="powerful",
                prefer_provider="anthropic",
            ),
            fallback=FallbackRule(
                strategy="explicit",
                explicit_model_identity="openai::gpt-4o",
            ),
        )

        data = policy.to_dict()
        restored = RoutingPolicy.from_dict(data)

        assert restored.name == policy.name
        assert restored.description == policy.description
        assert restored.budget_ceiling_usd == policy.budget_ceiling_usd
        assert restored.constraints.max_cost_usd == policy.constraints.max_cost_usd
        assert restored.constraints.max_latency_ms == policy.constraints.max_latency_ms
        assert restored.constraints.required_capabilities == policy.constraints.required_capabilities
        assert restored.constraints.allowed_providers == policy.constraints.allowed_providers
        assert restored.constraints.denied_providers == policy.constraints.denied_providers
        assert restored.constraints.allowed_models == policy.constraints.allowed_models
        assert restored.constraints.denied_models == policy.constraints.denied_models
        assert restored.objective.minimize == policy.objective.minimize
        assert restored.objective.prefer_tier == policy.objective.prefer_tier
        assert restored.objective.prefer_provider == policy.objective.prefer_provider
        assert restored.fallback.strategy == policy.fallback.strategy
        assert restored.fallback.explicit_model_identity == policy.fallback.explicit_model_identity

    def test_roundtrip_default_policy(self) -> None:
        """Default policy should survive roundtrip with all None/empty defaults."""
        policy = RoutingPolicy()
        data = policy.to_dict()
        restored = RoutingPolicy.from_dict(data)

        assert restored.name == "default"
        assert restored.constraints.max_cost_usd is None
        assert restored.constraints.allowed_providers is None
        assert restored.objective.minimize == "cost"
        assert restored.fallback.strategy == "next_cheapest"


# ---------------------------------------------------------------------------
# 18. Policy serializes to TOML and back without data loss
# ---------------------------------------------------------------------------


class TestTomlSerialization:
    def test_roundtrip(self) -> None:
        policy = RoutingPolicy(
            name="toml_test",
            description="TOML roundtrip test",
            budget_ceiling_usd=2.0,
            constraints=RoutingConstraint(
                max_cost_usd=0.05,
                max_latency_ms=1000,
                required_capabilities=["reasoning"],
                denied_providers=["local"],
                denied_models=["gpt-3"],
            ),
            objective=RoutingObjective(
                minimize="cost",
                prefer_tier="balanced",
                prefer_provider="anthropic",
            ),
            fallback=FallbackRule(
                strategy="same_tier",
            ),
        )

        toml_str = policy.to_toml()
        assert "toml_test" in toml_str
        assert "[constraints]" in toml_str
        assert "[objective]" in toml_str
        assert "[fallback]" in toml_str

        # Write to temp file and read back
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".toml", delete=False
        ) as f:
            f.write(toml_str)
            temp_path = f.name

        try:
            restored = RoutingPolicy.from_toml(temp_path)

            assert restored.name == policy.name
            assert restored.description == policy.description
            assert restored.budget_ceiling_usd == policy.budget_ceiling_usd
            assert restored.constraints.max_cost_usd == policy.constraints.max_cost_usd
            assert restored.constraints.max_latency_ms == policy.constraints.max_latency_ms
            assert restored.constraints.required_capabilities == policy.constraints.required_capabilities
            assert restored.constraints.denied_providers == policy.constraints.denied_providers
            assert restored.constraints.denied_models == policy.constraints.denied_models
            assert restored.objective.minimize == policy.objective.minimize
            assert restored.objective.prefer_tier == policy.objective.prefer_tier
            assert restored.objective.prefer_provider == policy.objective.prefer_provider
            assert restored.fallback.strategy == policy.fallback.strategy
        finally:
            os.unlink(temp_path)


# ---------------------------------------------------------------------------
# 19. Router with policy set uses PolicyRouter instead of heuristic
# ---------------------------------------------------------------------------


class TestRouterWithPolicy:
    @pytest.mark.asyncio
    async def test_router_uses_policy_when_set(self) -> None:
        router = Router()
        for m in _all_models():
            router.register_model(m)

        # Set a policy that prefers powerful tier
        policy = RoutingPolicy.quality_optimized()
        router.set_policy(policy)

        task = _task()
        decision = await router.route(task)

        # Quality-optimized should pick powerful-tier model
        assert decision.chosen_model == "claude-opus"
        assert decision.metadata["policy_name"] == "quality_optimized"
        assert "candidates_count" in decision.metadata
        assert "eliminated_count" in decision.metadata


# ---------------------------------------------------------------------------
# 20. Router without policy uses existing heuristic (backward compat)
# ---------------------------------------------------------------------------


class TestRouterWithoutPolicy:
    @pytest.mark.asyncio
    async def test_router_heuristic_backward_compat(self) -> None:
        router = Router()
        for m in _all_models():
            router.register_model(m)

        assert router.get_policy() is None

        task = _task(complexity_hint="simple")
        decision = await router.route(task)

        # Heuristic: simple task -> fast tier -> cheapest
        assert decision.chosen_model == "tiny-7b"
        # No policy metadata
        assert decision.metadata == {}


# ---------------------------------------------------------------------------
# 21. Budget ceiling tracked across steps (test with mock coordination)
# ---------------------------------------------------------------------------


class TestBudgetCeiling:
    @pytest.mark.asyncio
    async def test_budget_ceiling_value_stored(self) -> None:
        """Verify budget_ceiling_usd is preserved through serialization."""
        policy = RoutingPolicy(
            name="budget_test",
            budget_ceiling_usd=0.10,
        )

        # Budget ceiling is available on the policy object
        assert policy.budget_ceiling_usd == 0.10

        # Survives roundtrip
        restored = RoutingPolicy.from_dict(policy.to_dict())
        assert restored.budget_ceiling_usd == 0.10

        # Router can access it via get_policy
        router = Router()
        for m in _all_models():
            router.register_model(m)
        router.set_policy(policy)

        active = router.get_policy()
        assert active is not None
        assert active.budget_ceiling_usd == 0.10

        # Individual evaluations still work under budget
        task = _task()
        decision = await router.route(task)
        assert decision.estimated_cost_usd < policy.budget_ceiling_usd


# ---------------------------------------------------------------------------
# 22. Same task + same models + same policy = same result (deterministic)
# ---------------------------------------------------------------------------


class TestDeterminism:
    @pytest.mark.asyncio
    async def test_deterministic_results(self) -> None:
        policy = RoutingPolicy(
            name="determinism_test",
            objective=RoutingObjective(minimize="cost", prefer_tier="balanced"),
        )
        models = _all_models()
        task = _task(content="Determinism test task")

        results: list[PolicyEvaluation] = []
        for _ in range(10):
            router = PolicyRouter(policy, models)
            result = await router.evaluate(task)
            results.append(result)

        # All 10 runs must pick the same model
        chosen_models = {r.chosen.identity.key for r in results}
        assert len(chosen_models) == 1

        # All 10 runs must have the same scores
        first_scores = [c.score for c in results[0].all_candidates]
        for r in results[1:]:
            assert [c.score for c in r.all_candidates] == first_scores

        # All 10 runs must have the same fallback
        fallbacks = {r.fallback.identity.key if r.fallback else None for r in results}
        assert len(fallbacks) == 1


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------


class TestFallbackSameTier:
    @pytest.mark.asyncio
    async def test_same_tier_fallback(self) -> None:
        policy = RoutingPolicy(
            objective=RoutingObjective(minimize="cost"),
            fallback=FallbackRule(strategy="same_tier"),
        )
        # Two balanced models
        models = [_balanced(), _openai_balanced(), _cheap_fast()]
        router = PolicyRouter(policy, models)
        result = await router.evaluate(_task())

        # Cheapest overall is tiny-7b (fast). Fallback should be same tier (fast).
        # But there's only one fast model, so it falls through to cheapest among others.
        assert result.chosen.model == "tiny-7b"
        assert result.fallback is not None


class TestFallbackNextFastest:
    @pytest.mark.asyncio
    async def test_next_fastest_fallback(self) -> None:
        policy = RoutingPolicy(
            fallback=FallbackRule(strategy="next_fastest"),
        )
        router = PolicyRouter(policy, _all_models())
        result = await router.evaluate(_task())

        assert result.fallback is not None
        # Fallback should be the fastest among non-chosen models
        non_chosen = [m for m in _all_models() if m.identity != result.chosen.identity]
        fastest_other = min(non_chosen, key=lambda m: m.avg_latency_ms)
        assert result.fallback.identity == fastest_other.identity


class TestTaskConstraintsMerge:
    @pytest.mark.asyncio
    async def test_task_and_policy_constraints_merge(self) -> None:
        """Task-level cost_ceiling and policy max_cost should both apply (min wins)."""
        policy = RoutingPolicy(
            constraints=RoutingConstraint(max_cost_usd=0.05),
        )
        # Task has a tighter ceiling
        task = _task(cost_ceiling_usd=0.005)

        router = PolicyRouter(policy, _all_models())
        result = await router.evaluate(task)

        # Only the cheapest model should survive
        for c in result.all_candidates:
            assert c.model.estimated_cost() <= 0.005


class TestSingleModel:
    @pytest.mark.asyncio
    async def test_single_model_no_fallback(self) -> None:
        policy = RoutingPolicy()
        router = PolicyRouter(policy, [_cheap_fast()])
        result = await router.evaluate(_task())

        assert result.chosen.model == "tiny-7b"
        assert result.fallback is None
        assert len(result.all_candidates) == 1
        assert len(result.eliminated) == 0
