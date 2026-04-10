"""Unit tests for the canonical capability vocabulary.

Verifies that:
- The Capability enum defines all expected values
- validate_capabilities accepts valid and rejects invalid capabilities
- normalize_capabilities resolves aliases and validates
- Registration boundaries enforce the vocabulary
- Agent capabilities influence routing
- Built-in providers use only canonical values
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from kortex.core.capabilities import (
    Capability,
    CAPABILITY_ALIASES,
    normalize_capabilities,
    validate_capabilities,
)
from kortex.core.router import ProviderModel, Router
from kortex.core.runtime import AgentDescriptor, KortexRuntime
from kortex.core.state import StateManager
from kortex.core.types import TaskSpec
from kortex.store.memory import InMemoryStateStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model(**overrides) -> ProviderModel:
    defaults = dict(
        provider="test",
        model="test-model",
        cost_per_1k_input_tokens=0.001,
        cost_per_1k_output_tokens=0.002,
        avg_latency_ms=200,
        capabilities=["reasoning"],
        tier="fast",
    )
    defaults.update(overrides)
    return ProviderModel(**defaults)


def _make_runtime(router: Router | None = None) -> KortexRuntime:
    r = router or Router()
    state = StateManager(store=InMemoryStateStore())
    return KortexRuntime(router=r, state_manager=state)


# ---------------------------------------------------------------------------
# 1. Capability enum values
# ---------------------------------------------------------------------------


class TestCapabilityEnum:
    def test_all_expected_values_exist(self):
        expected = {
            "reasoning", "analysis", "code_generation", "content_generation",
            "vision", "audio", "quality_assurance", "data_processing",
            "planning", "research", "testing",
        }
        actual = {c.value for c in Capability}
        assert actual == expected

    def test_values_are_strings(self):
        for c in Capability:
            assert isinstance(c.value, str)
            assert c.value == c.value.lower()


# ---------------------------------------------------------------------------
# 2. validate_capabilities
# ---------------------------------------------------------------------------


class TestValidateCapabilities:
    def test_accepts_valid_capabilities(self):
        caps = ["reasoning", "analysis", "code_generation"]
        result = validate_capabilities(caps)
        assert result == caps

    def test_accepts_empty_list(self):
        assert validate_capabilities([]) == []

    def test_rejects_invalid_capability(self):
        with pytest.raises(ValueError, match="Invalid capabilities"):
            validate_capabilities(["reasoning", "flying"])

    def test_error_message_lists_invalid(self):
        with pytest.raises(ValueError, match="'flying'"):
            validate_capabilities(["flying"])

    def test_suggests_close_match(self):
        with pytest.raises(ValueError, match="Did you mean"):
            validate_capabilities(["reasonin"])

    def test_suggests_alias_resolution_for_aliases(self):
        with pytest.raises(ValueError, match="normalize_capabilities"):
            validate_capabilities(["coding"])

    def test_all_canonical_values_pass(self):
        all_caps = [c.value for c in Capability]
        assert validate_capabilities(all_caps) == all_caps


# ---------------------------------------------------------------------------
# 3. normalize_capabilities
# ---------------------------------------------------------------------------


class TestNormalizeCapabilities:
    def test_resolves_coding_alias(self):
        result = normalize_capabilities(["coding"])
        assert result == ["code_generation"]

    def test_resolves_writing_alias(self):
        result = normalize_capabilities(["writing"])
        assert result == ["content_generation"]

    def test_resolves_review_alias(self):
        result = normalize_capabilities(["review"])
        assert result == ["quality_assurance"]

    def test_resolves_qa_alias(self):
        result = normalize_capabilities(["qa"])
        assert result == ["quality_assurance"]

    def test_resolves_design_alias(self):
        result = normalize_capabilities(["design"])
        assert result == ["planning"]

    def test_resolves_coordination_alias(self):
        result = normalize_capabilities(["coordination"])
        assert result == ["planning"]

    def test_resolves_manage_alias(self):
        result = normalize_capabilities(["manage"])
        assert result == ["planning"]

    def test_passes_through_canonical_unchanged(self):
        caps = ["reasoning", "analysis"]
        result = normalize_capabilities(caps)
        assert result == caps

    def test_deduplicates(self):
        # "coding" and "code" both map to "code_generation"
        result = normalize_capabilities(["coding", "code"])
        assert result == ["code_generation"]

    def test_mixed_aliases_and_canonical(self):
        result = normalize_capabilities(["coding", "reasoning", "qa"])
        assert "code_generation" in result
        assert "reasoning" in result
        assert "quality_assurance" in result
        assert len(result) == 3

    def test_rejects_unknown_after_alias_resolution(self):
        with pytest.raises(ValueError):
            normalize_capabilities(["reasoning", "teleportation"])

    def test_empty_list(self):
        assert normalize_capabilities([]) == []


# ---------------------------------------------------------------------------
# 4. All known aliases map to valid capabilities
# ---------------------------------------------------------------------------


class TestAliasIntegrity:
    def test_all_alias_targets_are_canonical(self):
        valid = {c.value for c in Capability}
        for alias, target in CAPABILITY_ALIASES.items():
            assert target in valid, (
                f"Alias '{alias}' -> '{target}' is not a valid Capability"
            )


# ---------------------------------------------------------------------------
# 5. Router.register_model rejects invalid capabilities
# ---------------------------------------------------------------------------


class TestRouterRegistrationValidation:
    def test_register_model_with_valid_capabilities(self):
        router = Router()
        model = _make_model(capabilities=["reasoning", "analysis"])
        router.register_model(model)
        assert len(router.models) == 1

    def test_register_model_with_invalid_capabilities_raises(self):
        router = Router()
        model = _make_model(capabilities=["reasoning", "flying"])
        with pytest.raises(ValueError, match="Invalid capabilities"):
            router.register_model(model)

    def test_register_model_with_empty_capabilities_ok(self):
        router = Router()
        model = _make_model(capabilities=[])
        router.register_model(model)
        assert len(router.models) == 1


# ---------------------------------------------------------------------------
# 6. TaskSpec with invalid required_capabilities raises
# ---------------------------------------------------------------------------


class TestTaskSpecValidation:
    def test_valid_required_capabilities(self):
        task = TaskSpec(
            content="test",
            required_capabilities=["reasoning", "analysis"],
        )
        assert task.required_capabilities == ["reasoning", "analysis"]

    def test_invalid_required_capabilities_raises(self):
        with pytest.raises(ValueError, match="Invalid capabilities"):
            TaskSpec(
                content="test",
                required_capabilities=["reasoning", "flying"],
            )

    def test_empty_required_capabilities_ok(self):
        task = TaskSpec(content="test")
        assert task.required_capabilities == []

    def test_alias_in_required_capabilities_rejected(self):
        """Aliases are not canonical — must use normalize first."""
        with pytest.raises(ValueError):
            TaskSpec(
                content="test",
                required_capabilities=["coding"],
            )


# ---------------------------------------------------------------------------
# 7. AgentDescriptor capabilities are normalized on registration
# ---------------------------------------------------------------------------


class TestAgentRegistrationNormalization:
    def test_capabilities_normalized_on_register(self):
        runtime = _make_runtime()
        desc = AgentDescriptor(
            agent_id="a1",
            name="Agent A",
            description="test",
            capabilities=["coding", "review"],
        )
        runtime.register_agent(desc)

        registered = runtime._agents["a1"]
        assert "code_generation" in registered.capabilities
        assert "quality_assurance" in registered.capabilities
        assert "coding" not in registered.capabilities
        assert "review" not in registered.capabilities

    def test_canonical_capabilities_unchanged(self):
        runtime = _make_runtime()
        desc = AgentDescriptor(
            agent_id="a1",
            name="Agent A",
            description="test",
            capabilities=["reasoning", "analysis"],
        )
        runtime.register_agent(desc)

        registered = runtime._agents["a1"]
        assert registered.capabilities == ["reasoning", "analysis"]

    def test_invalid_capabilities_rejected(self):
        runtime = _make_runtime()
        desc = AgentDescriptor(
            agent_id="a1",
            name="Agent A",
            description="test",
            capabilities=["flying"],
        )
        with pytest.raises(ValueError):
            runtime.register_agent(desc)


# ---------------------------------------------------------------------------
# 8. CrewAI adapter infers canonical capabilities
# ---------------------------------------------------------------------------


class TestCrewAICanonicalCapabilities:
    def test_infer_research_canonical(self):
        from kortex.adapters.crewai import _infer_capabilities

        caps = _infer_capabilities("Research Analyst")
        assert "research" in caps
        assert "analysis" in caps
        # No non-canonical values
        valid = {c.value for c in Capability}
        for cap in caps:
            assert cap in valid, f"'{cap}' is not a canonical capability"

    def test_infer_writer_canonical(self):
        from kortex.adapters.crewai import _infer_capabilities

        caps = _infer_capabilities("Content Writer")
        assert "content_generation" in caps

    def test_infer_designer_maps_to_planning(self):
        from kortex.adapters.crewai import _infer_capabilities

        caps = _infer_capabilities("UI Designer")
        assert "planning" in caps
        assert "design" not in caps

    def test_infer_manager_maps_to_planning(self):
        from kortex.adapters.crewai import _infer_capabilities

        caps = _infer_capabilities("Project Manager")
        assert "planning" in caps
        assert "coordination" not in caps

    def test_no_non_canonical_values(self):
        from kortex.adapters.crewai import _infer_capabilities

        valid = {c.value for c in Capability}
        roles = [
            "Research Analyst", "Content Writer", "Code Reviewer",
            "Code Developer", "UI Designer", "QA Tester",
            "Project Manager", "Data Analyst",
        ]
        for role in roles:
            caps = _infer_capabilities(role)
            for cap in caps:
                assert cap in valid, (
                    f"Role '{role}' inferred non-canonical capability '{cap}'"
                )


# ---------------------------------------------------------------------------
# 9. Agent capabilities influence routing when task has no required_capabilities
# ---------------------------------------------------------------------------


class TestAgentCapabilitiesInfluenceRouting:
    @pytest.mark.asyncio
    async def test_agent_caps_used_when_task_has_none(self):
        router = Router()
        # Register two models: one with analysis, one without
        router.register_model(ProviderModel(
            provider="test",
            model="with-analysis",
            cost_per_1k_input_tokens=0.001,
            cost_per_1k_output_tokens=0.002,
            avg_latency_ms=200,
            capabilities=["reasoning", "analysis"],
            tier="fast",
        ))
        router.register_model(ProviderModel(
            provider="test",
            model="no-analysis",
            cost_per_1k_input_tokens=0.0005,
            cost_per_1k_output_tokens=0.001,
            avg_latency_ms=150,
            capabilities=["reasoning"],
            tier="fast",
        ))

        runtime = _make_runtime(router)
        runtime.register_agent(AgentDescriptor(
            agent_id="analyst",
            name="Analyst",
            description="Data analyst",
            capabilities=["analysis"],
        ))

        # Task has no required_capabilities — agent's should be used
        task = TaskSpec(content="Analyze data", complexity_hint="simple")
        result = await runtime.coordinate(task, ["analyst"])

        # The router should have filtered to only "with-analysis"
        assert result.routing_decisions[0].chosen_model == "with-analysis"

    @pytest.mark.asyncio
    async def test_task_caps_override_agent_caps(self):
        router = Router()
        router.register_model(ProviderModel(
            provider="test",
            model="basic",
            cost_per_1k_input_tokens=0.001,
            cost_per_1k_output_tokens=0.002,
            avg_latency_ms=200,
            capabilities=["reasoning"],
            tier="fast",
        ))

        runtime = _make_runtime(router)
        runtime.register_agent(AgentDescriptor(
            agent_id="agent-a",
            name="Agent A",
            description="test",
            capabilities=["analysis"],  # Agent wants analysis
        ))

        # Task explicitly requires reasoning only — should NOT use agent caps
        task = TaskSpec(
            content="Think about this",
            complexity_hint="simple",
            required_capabilities=["reasoning"],
        )
        result = await runtime.coordinate(task, ["agent-a"])

        # Should route successfully since task requires reasoning (not analysis)
        assert result.routing_decisions[0].chosen_model == "basic"

    @pytest.mark.asyncio
    async def test_unregistered_agent_no_caps_applied(self):
        router = Router()
        router.register_model(_make_model())

        runtime = _make_runtime(router)
        # Don't register "unknown-agent" — it should still route fine
        task = TaskSpec(content="test", complexity_hint="simple")
        result = await runtime.coordinate(task, ["unknown-agent"])
        assert len(result.routing_decisions) == 1


# ---------------------------------------------------------------------------
# 10. All built-in provider capabilities are valid canonical values
# ---------------------------------------------------------------------------


class TestBuiltInProviderCapabilities:
    def test_anthropic_capabilities_canonical(self):
        from kortex.providers.anthropic import AnthropicConnector

        connector = AnthropicConnector(api_key="test")
        valid = {c.value for c in Capability}
        for model in connector.get_available_models():
            for cap in model.capabilities:
                assert cap in valid, (
                    f"Anthropic model '{model.model}' has non-canonical "
                    f"capability '{cap}'"
                )

    def test_openai_capabilities_canonical(self):
        from kortex.providers.openai import OpenAIConnector

        connector = OpenAIConnector(api_key="test")
        valid = {c.value for c in Capability}
        for model in connector.get_available_models():
            for cap in model.capabilities:
                assert cap in valid, (
                    f"OpenAI model '{model.model}' has non-canonical "
                    f"capability '{cap}'"
                )

    def test_openrouter_capabilities_canonical(self):
        from kortex.providers.openrouter import OpenRouterConnector

        connector = OpenRouterConnector(api_key="test")
        valid = {c.value for c in Capability}
        for model in connector.get_available_models():
            for cap in model.capabilities:
                assert cap in valid, (
                    f"OpenRouter model '{model.model}' has non-canonical "
                    f"capability '{cap}'"
                )

    def test_all_provider_models_pass_validation(self):
        """Every built-in model can be registered without ValueError."""
        from kortex.providers.anthropic import AnthropicConnector
        from kortex.providers.openai import OpenAIConnector
        from kortex.providers.openrouter import OpenRouterConnector

        router = Router()
        connectors = [
            AnthropicConnector(api_key="test"),
            OpenAIConnector(api_key="test"),
            OpenRouterConnector(api_key="test"),
        ]
        for conn in connectors:
            for model in conn.get_available_models():
                router.register_model(model)  # Should not raise

        assert len(router.models) > 0
