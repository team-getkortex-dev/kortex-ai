"""Integration tests for the Kortex runtime orchestrator."""

from __future__ import annotations

import pytest

from kortex.core.exceptions import RoutingFailedError
from kortex.core.router import HeuristicRoutingStrategy, ProviderModel, Router
from kortex.core.runtime import AgentDescriptor, KortexRuntime
from kortex.core.state import StateManager
from kortex.core.types import RoutingDecision, TaskSpec
from kortex.store.memory import InMemoryStateStore


# --- Helpers ---


def _fast_model() -> ProviderModel:
    return ProviderModel(
        provider="openai",
        model="gpt-4o-mini",
        cost_per_1k_input_tokens=0.00015,
        cost_per_1k_output_tokens=0.0006,
        avg_latency_ms=200,
        capabilities=["reasoning", "code_generation"],
        tier="fast",
    )


def _balanced_model() -> ProviderModel:
    return ProviderModel(
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        cost_per_1k_input_tokens=0.003,
        cost_per_1k_output_tokens=0.015,
        avg_latency_ms=800,
        capabilities=["reasoning", "code_generation", "vision"],
        tier="balanced",
    )


def _powerful_model() -> ProviderModel:
    return ProviderModel(
        provider="anthropic",
        model="claude-opus-4-20250514",
        cost_per_1k_input_tokens=0.015,
        cost_per_1k_output_tokens=0.075,
        avg_latency_ms=2000,
        capabilities=["reasoning", "code_generation", "vision", "data_processing"],
        tier="powerful",
    )


def _build_runtime(
    models: list[ProviderModel] | None = None,
    strategy: HeuristicRoutingStrategy | None = None,
) -> KortexRuntime:
    router = Router(strategy=strategy)
    for m in (models or [_fast_model(), _balanced_model(), _powerful_model()]):
        router.register_model(m)
    state = StateManager(store=InMemoryStateStore())
    runtime = KortexRuntime(router=router, state_manager=state)
    runtime.register_agent(AgentDescriptor("planner", "Planner", "Plans tasks"))
    runtime.register_agent(AgentDescriptor("coder", "Coder", "Writes code"))
    runtime.register_agent(AgentDescriptor("reviewer", "Reviewer", "Reviews code"))
    return runtime


# --- 1. Single-agent pipeline ---


class TestSingleAgent:
    @pytest.mark.asyncio
    async def test_single_agent_routes_and_checkpoints(self) -> None:
        runtime = _build_runtime()
        task = TaskSpec(content="Write a hello world", complexity_hint="simple")
        result = await runtime.coordinate(task, ["planner"])

        assert result.success is True
        assert len(result.routing_decisions) == 1
        # Initial checkpoint + no inter-agent handoff for single agent
        assert len(result.handoffs) >= 1
        assert result.task_id == task.task_id


# --- 2. Three-agent pipeline ---


class TestThreeAgentPipeline:
    @pytest.mark.asyncio
    async def test_three_agents_route_and_handoff(self) -> None:
        runtime = _build_runtime()
        task = TaskSpec(content="Build a REST API", complexity_hint="complex")
        result = await runtime.coordinate(task, ["planner", "coder", "reviewer"])

        assert result.success is True
        assert len(result.routing_decisions) == 3
        # initial + 2 inter-agent handoffs
        assert len(result.handoffs) == 3
        # Verify handoff chain links
        assert result.handoffs[1].parent_checkpoint_id == result.handoffs[0].checkpoint_id
        assert result.handoffs[2].parent_checkpoint_id == result.handoffs[1].checkpoint_id


# --- 3. Middle agent fails, still succeeds ---


class TestPartialFailure:
    @pytest.mark.asyncio
    async def test_middle_agent_failure_skipped(self) -> None:
        # Only register a model that won't match strict constraints
        # Use a strategy that fails for specific agents via cost ceiling
        runtime = _build_runtime(models=[_fast_model()])
        task = TaskSpec(
            content="Analyze code",
            complexity_hint="simple",
        )
        # All three agents will route to the fast model since that's all that's available
        # Instead, let's use a custom strategy that fails for the middle agent
        call_count = 0

        class FailMiddleStrategy:
            async def select(
                self, task: TaskSpec, candidates: list[ProviderModel]
            ) -> RoutingDecision:
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise RoutingFailedError("No model for middle agent")
                m = candidates[0]
                return RoutingDecision(
                    task_id=task.task_id,
                    chosen_provider=m.provider,
                    chosen_model=m.model,
                    reasoning="Test routing",
                    estimated_cost_usd=m.estimated_cost(),
                    estimated_latency_ms=m.avg_latency_ms,
                )

        runtime = _build_runtime(strategy=FailMiddleStrategy())
        result = await runtime.coordinate(task, ["planner", "coder", "reviewer"])

        assert result.success is True
        # Only 2 routing decisions (middle one failed)
        assert len(result.routing_decisions) == 2
        # Check that a failure event was emitted
        failure_events = [e for e in result.events if e.event_type == "failure"]
        assert len(failure_events) == 1


# --- 4. ALL agents fail ---


class TestAllAgentsFail:
    @pytest.mark.asyncio
    async def test_all_fail_raises_routing_failed(self) -> None:
        class AlwaysFail:
            async def select(
                self, task: TaskSpec, candidates: list[ProviderModel]
            ) -> RoutingDecision:
                raise RoutingFailedError("Cannot route")

        runtime = _build_runtime(strategy=AlwaysFail())
        task = TaskSpec(content="Impossible task")

        with pytest.raises(RoutingFailedError, match="All agents in pipeline failed"):
            await runtime.coordinate(task, ["planner", "coder", "reviewer"])


# --- 5. Rollback from completed coordination ---


class TestRollback:
    @pytest.mark.asyncio
    async def test_rollback_retrieves_checkpoint(self) -> None:
        runtime = _build_runtime()
        task = TaskSpec(content="Multi-step task", complexity_hint="moderate")
        result = await runtime.coordinate(task, ["planner", "coder"])

        first_checkpoint_id = result.handoffs[0].checkpoint_id
        restored = await runtime.rollback_to(first_checkpoint_id)
        assert restored.checkpoint_id == first_checkpoint_id


# --- 6. Total estimated cost ---


class TestCostAccumulation:
    @pytest.mark.asyncio
    async def test_total_cost_is_sum_of_decisions(self) -> None:
        runtime = _build_runtime()
        task = TaskSpec(content="Cost check", complexity_hint="moderate")
        result = await runtime.coordinate(task, ["planner", "coder", "reviewer"])

        expected_cost = sum(d.estimated_cost_usd for d in result.routing_decisions)
        assert result.total_estimated_cost_usd == pytest.approx(expected_cost)
        assert result.total_estimated_cost_usd > 0


# --- 7. Coordination summary ---


class TestSummary:
    @pytest.mark.asyncio
    async def test_summary_is_readable(self) -> None:
        runtime = _build_runtime()
        task = TaskSpec(content="Summarize this", complexity_hint="simple")
        result = await runtime.coordinate(task, ["planner", "coder"])

        summary = runtime.get_coordination_summary(result)
        assert result.task_id in summary
        assert "2 agent(s)" in summary
        assert "$" in summary
        assert "ms" in summary
        assert "->" in summary  # model chain arrow
        assert "handoffs successful" in summary.lower()


# --- 8. Event types in order ---


class TestEventOrder:
    @pytest.mark.asyncio
    async def test_events_contain_correct_types(self) -> None:
        runtime = _build_runtime()
        task = TaskSpec(content="Event check", complexity_hint="moderate")
        result = await runtime.coordinate(task, ["planner", "coder", "reviewer"])

        event_types = [e.event_type for e in result.events]

        # Should start with handoff (initial), then alternating route/handoff, end with completion
        assert event_types[0] == "handoff"  # initial checkpoint
        assert event_types[-1] == "completion"

        # Should have route events
        route_count = event_types.count("route")
        assert route_count == 3

        # Should have handoff events (initial + inter-agent)
        handoff_count = event_types.count("handoff")
        assert handoff_count == 3  # initial + 2 inter-agent

        # Completion at the end
        assert event_types.count("completion") == 1
