"""Integration tests for the trace + replay architecture.

Tests the full flow: coordinate → trace → persist → replay.
"""

from __future__ import annotations

import pytest

from kortex.core.policy import RoutingPolicy
from kortex.core.replay import ReplayEngine
from kortex.core.router import ProviderModel, Router
from kortex.core.runtime import AgentDescriptor, KortexRuntime
from kortex.core.state import StateManager
from kortex.core.trace import TaskTrace
from kortex.core.trace_store import InMemoryTraceStore
from kortex.core.types import TaskSpec


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _models() -> list[ProviderModel]:
    return [
        ProviderModel(
            provider="local", model="tiny-7b",
            cost_per_1k_input_tokens=0.0001, cost_per_1k_output_tokens=0.0002,
            avg_latency_ms=50, capabilities=["reasoning", "content_generation"],
            tier="fast",
        ),
        ProviderModel(
            provider="anthropic", model="claude-sonnet",
            cost_per_1k_input_tokens=0.003, cost_per_1k_output_tokens=0.015,
            avg_latency_ms=800,
            capabilities=["reasoning", "code_generation", "content_generation"],
            tier="balanced",
        ),
        ProviderModel(
            provider="anthropic", model="claude-opus",
            cost_per_1k_input_tokens=0.015, cost_per_1k_output_tokens=0.075,
            avg_latency_ms=2000,
            capabilities=["reasoning", "code_generation", "content_generation", "vision"],
            tier="powerful",
        ),
    ]


def _make_runtime(
    enable_tracing: bool = True,
    trace_store: InMemoryTraceStore | None = None,
    policy: RoutingPolicy | None = None,
) -> KortexRuntime:
    router = Router()
    for m in _models():
        router.register_model(m)
    if policy is not None:
        router.set_policy(policy)

    runtime = KortexRuntime(
        router=router,
        state_manager=StateManager(),
        enable_tracing=enable_tracing,
        trace_store=trace_store,
    )
    runtime.register_agent(AgentDescriptor(
        "researcher", "Researcher", "Research agent",
        capabilities=["reasoning"],
    ))
    runtime.register_agent(AgentDescriptor(
        "writer", "Writer", "Writing agent",
        capabilities=["content_generation"],
    ))
    runtime.register_agent(AgentDescriptor(
        "reviewer", "Reviewer", "Review agent",
        capabilities=["reasoning"],
    ))
    return runtime


# ---------------------------------------------------------------------------
# 17. Full coordination with tracing enabled produces trace
# ---------------------------------------------------------------------------


class TestTracingEnabled:
    @pytest.mark.asyncio
    async def test_trace_in_result(self) -> None:
        runtime = _make_runtime()
        task = TaskSpec(content="Test tracing", complexity_hint="moderate")
        result = await runtime.coordinate(task, ["researcher", "writer"], export_trace=True)

        assert result.trace is not None
        assert result.trace["task_id"] == task.task_id
        assert result.trace["pipeline"] == ["researcher", "writer"]
        assert result.trace["success"] is True


# ---------------------------------------------------------------------------
# 18. Trace contains correct number of steps matching pipeline length
# ---------------------------------------------------------------------------


class TestTraceStepCount:
    @pytest.mark.asyncio
    async def test_step_count_matches_pipeline(self) -> None:
        runtime = _make_runtime()
        task = TaskSpec(content="3-step trace")

        result = await runtime.coordinate(
            task, ["researcher", "writer", "reviewer"], export_trace=True
        )

        trace_data = result.trace
        assert trace_data is not None
        assert len(trace_data["steps"]) == 3
        assert trace_data["steps"][0]["agent_id"] == "researcher"
        assert trace_data["steps"][1]["agent_id"] == "writer"
        assert trace_data["steps"][2]["agent_id"] == "reviewer"


# ---------------------------------------------------------------------------
# 19. Each trace step has routing decision and policy snapshot
# ---------------------------------------------------------------------------


class TestTraceStepContent:
    @pytest.mark.asyncio
    async def test_steps_have_routing_and_policy(self) -> None:
        policy = RoutingPolicy.cost_optimized()
        runtime = _make_runtime(policy=policy)
        task = TaskSpec(content="Policy trace test")

        result = await runtime.coordinate(task, ["researcher", "writer"], export_trace=True)
        trace_data = result.trace
        assert trace_data is not None

        for step in trace_data["steps"]:
            rd = step["routing_decision"]
            assert "chosen_model" in rd
            assert "chosen_provider" in rd
            assert "estimated_cost_usd" in rd

            ps = step["policy_snapshot"]
            assert ps["name"] == "cost_optimized"


# ---------------------------------------------------------------------------
# 20. Trace persisted to store is retrievable by trace_id
# ---------------------------------------------------------------------------


class TestTracePersistence:
    @pytest.mark.asyncio
    async def test_persisted_trace_retrievable(self) -> None:
        store = InMemoryTraceStore()
        runtime = _make_runtime(trace_store=store)
        task = TaskSpec(content="Persist trace test")

        result = await runtime.coordinate(task, ["researcher", "writer"], export_trace=True)
        assert result.trace is not None
        trace_id = result.trace["trace_id"]

        # Retrieve from store
        retrieved = await store.get_trace(trace_id)
        assert retrieved.task_id == task.task_id
        assert len(retrieved.steps) == 2

    @pytest.mark.asyncio
    async def test_list_traces_from_runtime(self) -> None:
        store = InMemoryTraceStore()
        runtime = _make_runtime(trace_store=store)

        await runtime.coordinate(
            TaskSpec(content="Task A"), ["researcher"]
        )
        await runtime.coordinate(
            TaskSpec(content="Task B"), ["writer"]
        )

        traces = await runtime.list_traces()
        assert len(traces) == 2


# ---------------------------------------------------------------------------
# 21. Replay of a persisted trace produces valid ReplayResult
# ---------------------------------------------------------------------------


class TestReplayPersistedTrace:
    @pytest.mark.asyncio
    async def test_replay_from_store(self) -> None:
        store = InMemoryTraceStore()
        policy = RoutingPolicy.cost_optimized()
        runtime = _make_runtime(trace_store=store, policy=policy)
        task = TaskSpec(content="Replay from store test")

        result = await runtime.coordinate(task, ["researcher", "writer"], export_trace=True)
        trace_id = result.trace["trace_id"]

        # Retrieve and replay
        trace = await store.get_trace(trace_id)
        engine = ReplayEngine(runtime._router)

        replay_result = await engine.replay(trace, RoutingPolicy.quality_optimized())

        assert len(replay_result.replayed_steps) == 2
        assert replay_result.summary  # non-empty summary
        assert replay_result.policy_used["name"] == "quality_optimized"

        # Quality should pick a different (more powerful) model
        for step in replay_result.replayed_steps:
            assert step.replayed_model == "claude-opus"


# ---------------------------------------------------------------------------
# 22. Tracing disabled = no trace in result (backward compat)
# ---------------------------------------------------------------------------


class TestTracingDisabled:
    @pytest.mark.asyncio
    async def test_no_trace_when_disabled(self) -> None:
        runtime = _make_runtime(enable_tracing=False)
        task = TaskSpec(content="No tracing")

        result = await runtime.coordinate(task, ["researcher", "writer"], export_trace=True)

        assert result.trace is None
