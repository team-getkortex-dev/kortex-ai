"""Tests for NLPolicyCompiler."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from kortex.core.nl_policy import NLPolicyCompiler
from kortex.core.policy import RoutingPolicy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_connector(response_json: dict) -> MagicMock:
    """Create a mock connector returning the given JSON as content."""
    response = MagicMock()
    response.content = json.dumps(response_json)
    connector = MagicMock()
    connector.complete = AsyncMock(return_value=response)
    return connector


def _cost_policy_json() -> dict:
    return {
        "name": "cheapest",
        "description": "Minimize cost.",
        "constraints": {
            "max_cost_usd": None, "max_latency_ms": None,
            "required_capabilities": [], "denied_providers": [],
        },
        "objective": {"minimize": "cost", "prefer_tier": "fast", "prefer_provider": None},
        "fallback": {"strategy": "next_cheapest", "explicit_model_identity": None},
        "budget_ceiling_usd": None,
    }


# ---------------------------------------------------------------------------
# compile_from_dict (synchronous, no LLM)
# ---------------------------------------------------------------------------


def test_compile_from_dict_basic() -> None:
    connector = MagicMock()
    compiler = NLPolicyCompiler(connector)
    policy = compiler.compile_from_dict(_cost_policy_json())

    assert isinstance(policy, RoutingPolicy)
    assert policy.name == "cheapest"
    assert policy.objective.minimize == "cost"


def test_compile_from_dict_denied_providers() -> None:
    data = _cost_policy_json()
    data["constraints"]["denied_providers"] = ["openai"]
    compiler = NLPolicyCompiler(MagicMock())
    policy = compiler.compile_from_dict(data)
    assert "openai" in policy.constraints.denied_providers


def test_compile_from_dict_max_cost_constraint() -> None:
    data = _cost_policy_json()
    data["constraints"]["max_cost_usd"] = 0.005
    compiler = NLPolicyCompiler(MagicMock())
    policy = compiler.compile_from_dict(data)
    assert policy.constraints.max_cost_usd == pytest.approx(0.005)


def test_compile_from_dict_latency_minimize() -> None:
    data = _cost_policy_json()
    data["objective"]["minimize"] = "latency"
    data["objective"]["prefer_tier"] = "fast"
    compiler = NLPolicyCompiler(MagicMock())
    policy = compiler.compile_from_dict(data)
    assert policy.objective.minimize == "latency"
    assert policy.objective.prefer_tier == "fast"


def test_compile_from_dict_quality_tier() -> None:
    data = _cost_policy_json()
    data["objective"]["minimize"] = "none"
    data["objective"]["prefer_tier"] = "powerful"
    compiler = NLPolicyCompiler(MagicMock())
    policy = compiler.compile_from_dict(data)
    assert policy.objective.prefer_tier == "powerful"


def test_compile_from_dict_allowed_providers() -> None:
    data = _cost_policy_json()
    data["constraints"]["allowed_providers"] = ["anthropic"]
    compiler = NLPolicyCompiler(MagicMock())
    policy = compiler.compile_from_dict(data)
    assert policy.constraints.allowed_providers == ["anthropic"]


def test_compile_from_dict_fallback_strategy() -> None:
    data = _cost_policy_json()
    data["fallback"]["strategy"] = "same_tier"
    compiler = NLPolicyCompiler(MagicMock())
    policy = compiler.compile_from_dict(data)
    assert policy.fallback.strategy == "same_tier"


def test_compile_from_dict_invalid_minimize_falls_back_to_cost() -> None:
    data = _cost_policy_json()
    data["objective"]["minimize"] = "invalid_value"
    compiler = NLPolicyCompiler(MagicMock())
    policy = compiler.compile_from_dict(data)
    assert policy.objective.minimize == "cost"


def test_compile_from_dict_invalid_prefer_tier_falls_back_to_any() -> None:
    data = _cost_policy_json()
    data["objective"]["prefer_tier"] = "super_fast"
    compiler = NLPolicyCompiler(MagicMock())
    policy = compiler.compile_from_dict(data)
    assert policy.objective.prefer_tier == "any"


# ---------------------------------------------------------------------------
# compile (async, with LLM mock)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compile_calls_connector() -> None:
    connector = _make_connector(_cost_policy_json())
    compiler = NLPolicyCompiler(connector)

    policy = await compiler.compile("Use the cheapest model")

    connector.complete.assert_called_once()
    assert isinstance(policy, RoutingPolicy)
    assert policy.name == "cheapest"


@pytest.mark.asyncio
async def test_compile_handles_markdown_fences() -> None:
    """LLM wrapping JSON in ```json ... ``` should still parse."""
    json_str = "```json\n" + json.dumps(_cost_policy_json()) + "\n```"
    response = MagicMock()
    response.content = json_str
    connector = MagicMock()
    connector.complete = AsyncMock(return_value=response)
    compiler = NLPolicyCompiler(connector)

    policy = await compiler.compile("cheapest option")
    assert policy.name == "cheapest"


@pytest.mark.asyncio
async def test_compile_retries_on_json_error() -> None:
    """LLM returns garbage first, valid JSON on second attempt."""
    good_response = MagicMock()
    good_response.content = json.dumps(_cost_policy_json())
    bad_response = MagicMock()
    bad_response.content = "This is not JSON at all!"

    connector = MagicMock()
    connector.complete = AsyncMock(side_effect=[bad_response, good_response])
    compiler = NLPolicyCompiler(connector, max_retries=2)

    policy = await compiler.compile("cheapest")
    assert policy.name == "cheapest"
    assert connector.complete.call_count == 2


@pytest.mark.asyncio
async def test_compile_raises_after_max_retries() -> None:
    response = MagicMock()
    response.content = "not json"
    connector = MagicMock()
    connector.complete = AsyncMock(return_value=response)
    compiler = NLPolicyCompiler(connector, max_retries=2)

    with pytest.raises(ValueError, match="Failed to compile policy after 2 attempts"):
        await compiler.compile("some description")
    assert connector.complete.call_count == 2


@pytest.mark.asyncio
async def test_compile_uses_custom_model() -> None:
    connector = _make_connector(_cost_policy_json())
    compiler = NLPolicyCompiler(connector, model="gpt-4o")

    await compiler.compile("use gpt-4o")
    call_args = connector.complete.call_args
    assert call_args[0][1] == "gpt-4o"  # second positional arg is model


@pytest.mark.asyncio
async def test_compile_extracts_json_from_prose() -> None:
    """LLM adds prose before/after JSON — should still extract the object."""
    prose = "Here is the compiled policy:\n" + json.dumps(_cost_policy_json()) + "\nDone."
    response = MagicMock()
    response.content = prose
    connector = MagicMock()
    connector.complete = AsyncMock(return_value=response)
    compiler = NLPolicyCompiler(connector)

    policy = await compiler.compile("cheap")
    assert policy.name == "cheapest"
