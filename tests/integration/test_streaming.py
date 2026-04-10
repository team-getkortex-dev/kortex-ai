"""Integration tests for stream_coordinate() on KortexRuntime."""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

import pytest

from kortex.core.router import ProviderModel, Router
from kortex.core.runtime import AgentDescriptor, KortexRuntime
from kortex.core.state import StateManager
from kortex.core.types import TaskSpec
from kortex.providers.base import ProviderResponse
from kortex.providers.registry import ProviderRegistry


# ---------------------------------------------------------------------------
# Mock streaming provider
# ---------------------------------------------------------------------------

_CHUNKS = ["Hello", ", ", "streaming", " ", "world", "!"]


class _StreamingProvider:
    """Minimal provider that yields fixed chunks from stream()."""

    @property
    def provider_name(self) -> str:
        return "mock-stream"

    def get_available_models(self) -> list[ProviderModel]:
        return [
            ProviderModel(
                provider="mock-stream",
                model="stream-model",
                cost_per_1k_input_tokens=0.001,
                cost_per_1k_output_tokens=0.001,
                avg_latency_ms=10,
                capabilities=["reasoning", "content_generation"],
                tier="fast",
            )
        ]

    async def complete(self, prompt: str, model: str, **kwargs: Any) -> ProviderResponse:
        return ProviderResponse(
            content="".join(_CHUNKS),
            model=model,
            provider=self.provider_name,
            input_tokens=10,
            output_tokens=6,
            cost_usd=0.0001,
            latency_ms=5.0,
        )

    async def stream(
        self, prompt: str, model: str, **kwargs: Any
    ) -> AsyncIterator[str]:
        for chunk in _CHUNKS:
            await asyncio.sleep(0)
            yield chunk

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        pass


class _NoStreamProvider:
    """Provider that has NO stream() method — only complete()."""

    @property
    def provider_name(self) -> str:
        return "mock-nostream"

    def get_available_models(self) -> list[ProviderModel]:
        return [
            ProviderModel(
                provider="mock-nostream",
                model="nostream-model",
                cost_per_1k_input_tokens=0.001,
                cost_per_1k_output_tokens=0.001,
                avg_latency_ms=10,
                capabilities=["reasoning"],
                tier="fast",
            )
        ]

    async def complete(self, prompt: str, model: str, **kwargs: Any) -> ProviderResponse:
        return ProviderResponse(
            content="no stream",
            model=model,
            provider=self.provider_name,
            input_tokens=5,
            output_tokens=2,
            cost_usd=0.0001,
            latency_ms=5.0,
        )

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_runtime(provider: Any = None) -> tuple[KortexRuntime, ProviderRegistry]:
    router = Router()

    registry = ProviderRegistry()
    if provider is not None:
        registry.register_provider(provider)
        for m in provider.get_available_models():
            router.register_model(m)
    else:
        router.register_model(ProviderModel(
            provider="dummy",
            model="dummy-model",
            cost_per_1k_input_tokens=0.001,
            cost_per_1k_output_tokens=0.001,
            avg_latency_ms=10,
            capabilities=["reasoning"],
            tier="fast",
        ))

    runtime = KortexRuntime(
        router=router,
        state_manager=StateManager.create("memory"),
        registry=registry if provider is not None else None,
    )
    runtime.register_agent(AgentDescriptor(
        agent_id="agent-a",
        name="Agent A",
        description="test",
        capabilities=["reasoning"],
    ))
    runtime.register_agent(AgentDescriptor(
        agent_id="agent-b",
        name="Agent B",
        description="test",
        capabilities=["reasoning"],
    ))
    return runtime, registry


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStreamCoordinateEventSequence:
    @pytest.mark.asyncio
    async def test_routing_decision_comes_first(self) -> None:
        runtime, _ = _make_runtime(_StreamingProvider())
        task = TaskSpec(content="hello", complexity_hint="simple")

        events: list[tuple[str, dict]] = []
        async with runtime:
            async for ev_type, data in runtime.stream_coordinate(task, ["agent-a"]):
                events.append((ev_type, data))

        types = [e[0] for e in events]
        assert "routing_decision" in types
        rd_idx = types.index("routing_decision")
        # If tokens exist, they come after routing
        if "token" in types:
            token_idx = types.index("token")
            assert rd_idx < token_idx

    @pytest.mark.asyncio
    async def test_completion_is_last_event(self) -> None:
        runtime, _ = _make_runtime(_StreamingProvider())
        task = TaskSpec(content="test", complexity_hint="simple")

        events: list[tuple[str, dict]] = []
        async with runtime:
            async for ev_type, data in runtime.stream_coordinate(task, ["agent-a"]):
                events.append((ev_type, data))

        assert events[-1][0] == "completion"

    @pytest.mark.asyncio
    async def test_token_events_contain_text(self) -> None:
        runtime, _ = _make_runtime(_StreamingProvider())
        task = TaskSpec(content="say hello", complexity_hint="simple")

        tokens: list[str] = []
        async with runtime:
            async for ev_type, data in runtime.stream_coordinate(task, ["agent-a"]):
                if ev_type == "token":
                    tokens.append(data["token"])

        assert len(tokens) == len(_CHUNKS)
        assert "".join(tokens) == "".join(_CHUNKS)

    @pytest.mark.asyncio
    async def test_token_events_include_agent_id(self) -> None:
        runtime, _ = _make_runtime(_StreamingProvider())
        task = TaskSpec(content="test", complexity_hint="simple")

        async with runtime:
            async for ev_type, data in runtime.stream_coordinate(task, ["agent-a"]):
                if ev_type == "token":
                    assert data["agent_id"] == "agent-a"
                    assert "step" in data
                    break

    @pytest.mark.asyncio
    async def test_completion_payload_fields(self) -> None:
        runtime, _ = _make_runtime(_StreamingProvider())
        task = TaskSpec(content="test", complexity_hint="simple")

        completion: dict | None = None
        async with runtime:
            async for ev_type, data in runtime.stream_coordinate(task, ["agent-a"]):
                if ev_type == "completion":
                    completion = data

        assert completion is not None
        assert completion["task_id"] == task.task_id
        assert completion["agents_routed"] == 1
        assert completion["total_agents"] == 1
        assert completion["success"] is True
        assert "duration_ms" in completion


class TestStreamCoordinateMultiAgent:
    @pytest.mark.asyncio
    async def test_two_agent_pipeline_produces_two_routing_events(self) -> None:
        runtime, _ = _make_runtime(_StreamingProvider())
        task = TaskSpec(content="two agents", complexity_hint="simple")

        routing_events: list[dict] = []
        async with runtime:
            async for ev_type, data in runtime.stream_coordinate(task, ["agent-a", "agent-b"]):
                if ev_type == "routing_decision":
                    routing_events.append(data)

        assert len(routing_events) == 2
        assert routing_events[0]["step"] == 0
        assert routing_events[1]["step"] == 1

    @pytest.mark.asyncio
    async def test_two_agent_pipeline_completion_counts(self) -> None:
        runtime, _ = _make_runtime(_StreamingProvider())
        task = TaskSpec(content="two agents", complexity_hint="simple")

        completion: dict | None = None
        async with runtime:
            async for ev_type, data in runtime.stream_coordinate(task, ["agent-a", "agent-b"]):
                if ev_type == "completion":
                    completion = data

        assert completion is not None
        assert completion["total_agents"] == 2
        assert completion["agents_routed"] == 2


class TestStreamCoordinateNoProvider:
    @pytest.mark.asyncio
    async def test_no_registry_still_yields_routing_and_completion(self) -> None:
        """Without a registry, stream_coordinate should yield routing + completion."""
        runtime, _ = _make_runtime(provider=None)
        task = TaskSpec(content="dry run", complexity_hint="simple")

        event_types: list[str] = []
        async with runtime:
            async for ev_type, _ in runtime.stream_coordinate(task, ["agent-a"]):
                event_types.append(ev_type)

        assert "routing_decision" in event_types
        assert event_types[-1] == "completion"
        assert "token" not in event_types

    @pytest.mark.asyncio
    async def test_provider_without_stream_yields_no_tokens(self) -> None:
        """Provider with no stream() method produces no token events."""
        runtime, _ = _make_runtime(_NoStreamProvider())
        task = TaskSpec(content="no stream", complexity_hint="simple")

        event_types: list[str] = []
        async with runtime:
            async for ev_type, _ in runtime.stream_coordinate(task, ["agent-a"]):
                event_types.append(ev_type)

        assert "token" not in event_types
        assert "routing_decision" in event_types
        assert event_types[-1] == "completion"
