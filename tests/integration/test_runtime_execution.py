"""Integration tests for runtime execution mode (execute=True).

All HTTP calls are mocked -- no real API keys needed.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from kortex.core.exceptions import ProviderError, RoutingFailedError
from kortex.core.router import ProviderModel, Router
from kortex.core.runtime import AgentDescriptor, KortexRuntime
from kortex.core.state import StateManager
from kortex.core.types import TaskSpec
from kortex.providers.registry import ProviderRegistry
from kortex.store.memory import InMemoryStateStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_httpx_response(data: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        status_code=200,
        json=data,
        request=httpx.Request("POST", "https://mock"),
    )


def _ok_response(content: str = "mock output", input_tok: int = 50, output_tok: int = 20) -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": input_tok, "completion_tokens": output_tok},
    }


def _make_models() -> list[ProviderModel]:
    return [
        ProviderModel(
            provider="test-cloud",
            model="cloud-fast",
            cost_per_1k_input_tokens=0.001,
            cost_per_1k_output_tokens=0.002,
            avg_latency_ms=200,
            capabilities=["reasoning"],
            tier="fast",
        ),
        ProviderModel(
            provider="test-cloud",
            model="cloud-balanced",
            cost_per_1k_input_tokens=0.003,
            cost_per_1k_output_tokens=0.01,
            avg_latency_ms=600,
            capabilities=["reasoning", "analysis"],
            tier="balanced",
        ),
        ProviderModel(
            provider="test-cloud",
            model="cloud-powerful",
            cost_per_1k_input_tokens=0.01,
            cost_per_1k_output_tokens=0.03,
            avg_latency_ms=1500,
            capabilities=["reasoning", "analysis", "research"],
            tier="powerful",
        ),
        ProviderModel(
            provider="test-local",
            model="local-llama",
            cost_per_1k_input_tokens=0.0,
            cost_per_1k_output_tokens=0.0,
            avg_latency_ms=100,
            capabilities=["reasoning"],
            tier="fast",
        ),
    ]


def _build_runtime(
    mock_post: AsyncMock | None = None,
    include_local: bool = False,
) -> KortexRuntime:
    """Build a runtime with registry, router, and mocked HTTP."""
    models = _make_models()
    if not include_local:
        models = [m for m in models if m.provider != "test-local"]

    router = Router()
    for m in models:
        router.register_model(m)

    registry = ProviderRegistry()
    cloud_models = [m for m in models if m.provider == "test-cloud"]
    registry.register_openai_compatible(
        name="test-cloud",
        base_url="https://mock-cloud.test/v1",
        api_key="mock-key",
        models=cloud_models,
    )

    if include_local:
        local_models = [m for m in models if m.provider == "test-local"]
        registry.register_openai_compatible(
            name="test-local",
            base_url="http://localhost:11434/v1",
            api_key=None,
            models=local_models,
        )

    # Mock the HTTP client for each registered connector
    if mock_post is not None:
        for pname in registry.list_providers():
            connector = registry.get_provider(pname)
            if hasattr(connector, "_get_client"):
                client = connector._get_client()  # type: ignore[union-attr]
                client.post = mock_post  # type: ignore[method-assign]

    state = StateManager(store=InMemoryStateStore())
    runtime = KortexRuntime(router=router, state_manager=state, registry=registry)

    runtime.register_agent(AgentDescriptor("agent-a", "Agent A", "First agent", ["reasoning"]))
    runtime.register_agent(AgentDescriptor("agent-b", "Agent B", "Second agent", ["reasoning"]))
    runtime.register_agent(AgentDescriptor("agent-c", "Agent C", "Third agent", ["reasoning"]))

    return runtime


# ---------------------------------------------------------------------------
# 1. execute=False still works exactly as before (backward compat)
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    @pytest.mark.asyncio
    async def test_execute_false_no_responses(self) -> None:
        runtime = _build_runtime()
        task = TaskSpec(content="Test backward compat", complexity_hint="simple")
        result = await runtime.coordinate(task, ["agent-a", "agent-b"])

        assert result.success is True
        assert len(result.routing_decisions) == 2
        assert result.responses == []
        assert result.actual_cost_usd == 0.0

    @pytest.mark.asyncio
    async def test_execute_false_is_default(self) -> None:
        runtime = _build_runtime()
        task = TaskSpec(content="Default param test")
        result = await runtime.coordinate(task, ["agent-a"])

        assert result.responses == []
        assert result.actual_cost_usd == 0.0


# ---------------------------------------------------------------------------
# 2. execute=True calls the correct provider with correct model
# ---------------------------------------------------------------------------


class TestExecuteTrue:
    @pytest.mark.asyncio
    async def test_calls_provider_and_returns_responses(self) -> None:
        mock_post = AsyncMock(return_value=_mock_httpx_response(
            _ok_response("Hello from LLM", 30, 15)
        ))
        runtime = _build_runtime(mock_post=mock_post)
        task = TaskSpec(content="Say hello", complexity_hint="simple")

        result = await runtime.coordinate(task, ["agent-a"], execute=True)

        assert result.success is True
        assert len(result.responses) == 1
        assert result.responses[0]["content"] == "Hello from LLM"
        assert result.responses[0]["provider"] == "test-cloud"
        mock_post.assert_called()

    @pytest.mark.asyncio
    async def test_correct_model_selected(self) -> None:
        mock_post = AsyncMock(return_value=_mock_httpx_response(
            _ok_response("complex answer", 100, 50)
        ))
        runtime = _build_runtime(mock_post=mock_post)
        task = TaskSpec(content="Deep analysis", complexity_hint="complex", required_capabilities=["reasoning"])

        result = await runtime.coordinate(task, ["agent-a"], execute=True)

        # Complex task should route to powerful tier
        assert result.routing_decisions[0].chosen_model == "cloud-powerful"
        assert result.responses[0]["model"] == "cloud-powerful"


# ---------------------------------------------------------------------------
# 3. Fallback model is used when primary provider call fails
# ---------------------------------------------------------------------------


class TestFallback:
    @pytest.mark.asyncio
    async def test_fallback_on_primary_failure(self) -> None:
        call_count = {"n": 0}

        async def mock_post(*args: Any, **kwargs: Any) -> httpx.Response:
            call_count["n"] += 1
            body = kwargs.get("json", {})
            model = body.get("model", "")
            if model == "cloud-fast":
                raise httpx.HTTPStatusError(
                    "Server error",
                    request=httpx.Request("POST", "https://mock"),
                    response=httpx.Response(500),
                )
            return _mock_httpx_response(_ok_response(f"Fallback response for {model}"))

        runtime = _build_runtime(mock_post=AsyncMock(side_effect=mock_post))
        task = TaskSpec(content="Test fallback", complexity_hint="simple")

        result = await runtime.coordinate(task, ["agent-a"], execute=True)

        assert result.success is True
        # Should have gotten a response (from the fallback model)
        assert len(result.responses) == 1
        assert "Fallback response" in result.responses[0]["content"]


# ---------------------------------------------------------------------------
# 4. Provider response is in StepExecutionRecord, NOT in handoff state_snapshot
# ---------------------------------------------------------------------------


class TestResponseInSteps:
    @pytest.mark.asyncio
    async def test_provider_response_in_steps_not_handoff(self) -> None:
        responses = [
            _ok_response("Agent A output", 30, 10),
            _ok_response("Agent B got previous output", 60, 20),
        ]
        call_count = {"n": 0}

        async def mock_post(*args: Any, **kwargs: Any) -> httpx.Response:
            idx = min(call_count["n"], len(responses) - 1)
            call_count["n"] += 1
            return _mock_httpx_response(responses[idx])

        runtime = _build_runtime(mock_post=AsyncMock(side_effect=mock_post))
        task = TaskSpec(content="Pipeline test", complexity_hint="simple")

        result = await runtime.coordinate(task, ["agent-a", "agent-b"], execute=True)

        # Handoff state_snapshot must NOT contain provider_response
        for h in result.handoffs:
            assert "provider_response" not in h.state_snapshot
            assert "routing_decision" not in h.state_snapshot

        # Provider responses must be in result.steps
        assert len(result.steps) == 2
        for step in result.steps:
            assert "provider_response" in step
            assert step["provider_response"] is not None
            pr = step["provider_response"]
            assert "content" in pr
            assert "cost_usd" in pr
            assert pr["input_tokens"] > 0


# ---------------------------------------------------------------------------
# 5. actual_cost_usd sums correctly across all responses
# ---------------------------------------------------------------------------


class TestActualCost:
    @pytest.mark.asyncio
    async def test_actual_cost_sums_correctly(self) -> None:
        responses = [
            _ok_response("out1", 100, 50),
            _ok_response("out2", 200, 100),
            _ok_response("out3", 150, 75),
        ]
        call_count = {"n": 0}

        async def mock_post(*args: Any, **kwargs: Any) -> httpx.Response:
            idx = min(call_count["n"], len(responses) - 1)
            call_count["n"] += 1
            return _mock_httpx_response(responses[idx])

        runtime = _build_runtime(mock_post=AsyncMock(side_effect=mock_post))
        task = TaskSpec(content="Cost test", complexity_hint="simple")

        result = await runtime.coordinate(
            task, ["agent-a", "agent-b", "agent-c"], execute=True,
        )

        assert len(result.responses) == 3
        manual_sum = sum(r["cost_usd"] for r in result.responses)
        assert result.actual_cost_usd == pytest.approx(manual_sum)
        assert result.actual_cost_usd > 0.0

    @pytest.mark.asyncio
    async def test_actual_cost_less_than_estimated(self) -> None:
        """Actual cost from real token counts is typically less than the
        estimate which uses default token assumptions."""
        mock_post = AsyncMock(return_value=_mock_httpx_response(
            _ok_response("short", 20, 10)
        ))
        runtime = _build_runtime(mock_post=mock_post)
        task = TaskSpec(content="Short task", complexity_hint="simple")

        result = await runtime.coordinate(task, ["agent-a"], execute=True)

        assert result.actual_cost_usd < result.total_estimated_cost_usd


# ---------------------------------------------------------------------------
# 6. Summary shows cost savings percentage when actual < estimated
# ---------------------------------------------------------------------------


class TestSummaryWithCosts:
    @pytest.mark.asyncio
    async def test_summary_shows_actual_cost_and_savings(self) -> None:
        mock_post = AsyncMock(return_value=_mock_httpx_response(
            _ok_response("result", 20, 10)
        ))
        runtime = _build_runtime(mock_post=mock_post)
        task = TaskSpec(content="Summary test", complexity_hint="simple")

        result = await runtime.coordinate(task, ["agent-a"], execute=True)
        summary = runtime.get_coordination_summary(result)

        assert "Actual cost:" in summary
        assert "Saved:" in summary
        assert "%" in summary

    @pytest.mark.asyncio
    async def test_summary_without_execution_has_no_actual(self) -> None:
        runtime = _build_runtime()
        task = TaskSpec(content="Dry run", complexity_hint="simple")
        result = await runtime.coordinate(task, ["agent-a"])

        summary = runtime.get_coordination_summary(result)
        assert "Actual cost:" not in summary
        assert "Saved:" not in summary


# ---------------------------------------------------------------------------
# 7. coordinate with execute=True works with zero-cost local models
# ---------------------------------------------------------------------------


class TestZeroCostModels:
    @pytest.mark.asyncio
    async def test_local_model_zero_cost(self) -> None:
        """When routing picks a free local model, actual_cost_usd should be 0."""
        mock_post = AsyncMock(return_value=_mock_httpx_response(
            _ok_response("local output", 50, 20)
        ))
        runtime = _build_runtime(mock_post=mock_post, include_local=True)

        # Simple task should route to cheapest fast model (local-llama at $0)
        task = TaskSpec(content="Quick question", complexity_hint="simple")
        result = await runtime.coordinate(task, ["agent-a"], execute=True)

        assert result.success is True
        assert len(result.responses) == 1
        assert result.responses[0]["cost_usd"] == 0.0
        assert result.actual_cost_usd == 0.0

    @pytest.mark.asyncio
    async def test_mixed_local_and_cloud(self) -> None:
        """Pipeline with both local (free) and cloud (paid) models."""
        responses = [
            _ok_response("local step output", 30, 15),  # simple -> local
            _ok_response("cloud step output", 100, 50),  # moderate -> cloud
        ]
        call_count = {"n": 0}

        async def mock_post(*args: Any, **kwargs: Any) -> httpx.Response:
            idx = min(call_count["n"], len(responses) - 1)
            call_count["n"] += 1
            return _mock_httpx_response(responses[idx])

        runtime = _build_runtime(
            mock_post=AsyncMock(side_effect=mock_post),
            include_local=True,
        )

        # Complex task routes step 1 to powerful, step 2 to moderate (balanced)
        task = TaskSpec(content="Mixed pipeline", complexity_hint="complex")
        result = await runtime.coordinate(
            task, ["agent-a", "agent-b"], execute=True,
        )

        assert result.success is True
        assert len(result.responses) == 2
        # At least one should have non-zero cost (the cloud model)
        costs = [r["cost_usd"] for r in result.responses]
        assert any(c > 0 for c in costs)
