"""Integration tests for batch coordination and routing."""

from __future__ import annotations

import pytest

from kortex.core.router import ProviderModel, Router
from kortex.core.runtime import AgentDescriptor, KortexRuntime
from kortex.core.state import StateManager
from kortex.core.types import TaskSpec
from kortex.store.memory import InMemoryStateStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_runtime() -> KortexRuntime:
    router = Router()
    router.register_model(ProviderModel(
        provider="openai",
        model="gpt-4o-mini",
        cost_per_1k_input_tokens=0.00015,
        cost_per_1k_output_tokens=0.0006,
        avg_latency_ms=200,
        tier="fast",
    ))
    router.register_model(ProviderModel(
        provider="anthropic",
        model="claude-haiku",
        cost_per_1k_input_tokens=0.001,
        cost_per_1k_output_tokens=0.005,
        avg_latency_ms=300,
        tier="balanced",
    ))
    state = StateManager(InMemoryStateStore())
    runtime = KortexRuntime(router=router, state_manager=state)
    return runtime


def _make_tasks(n: int) -> list[TaskSpec]:
    return [TaskSpec(content=f"Task {i}", complexity_hint="simple") for i in range(n)]


# ---------------------------------------------------------------------------
# Router.route_batch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_route_batch_returns_all_decisions() -> None:
    router = Router()
    router.register_model(ProviderModel(
        provider="openai", model="gpt-4o-mini",
        cost_per_1k_input_tokens=0.001, cost_per_1k_output_tokens=0.002,
        avg_latency_ms=200, tier="fast",
    ))
    tasks = _make_tasks(5)
    decisions = await router.route_batch(tasks)
    assert len(decisions) == 5
    for d in decisions:
        assert d.chosen_model == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_route_batch_empty() -> None:
    router = Router()
    router.register_model(ProviderModel(
        provider="openai", model="gpt-4o-mini",
        cost_per_1k_input_tokens=0.001, cost_per_1k_output_tokens=0.002,
        avg_latency_ms=200, tier="fast",
    ))
    decisions = await router.route_batch([])
    assert decisions == []


@pytest.mark.asyncio
async def test_route_batch_preserves_order() -> None:
    router = Router()
    router.register_model(ProviderModel(
        provider="openai", model="gpt-4o-mini",
        cost_per_1k_input_tokens=0.001, cost_per_1k_output_tokens=0.002,
        avg_latency_ms=200, tier="fast",
    ))
    tasks = [TaskSpec(content=f"task-{i}") for i in range(10)]
    decisions = await router.route_batch(tasks)
    assert [d.task_id for d in decisions] == [t.task_id for t in tasks]


# ---------------------------------------------------------------------------
# KortexRuntime.coordinate_batch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_coordinate_batch_basic() -> None:
    runtime = _make_runtime()
    tasks = _make_tasks(3)
    pipelines = [["agent1"]] * 3
    async with runtime:
        results = await runtime.coordinate_batch(tasks, pipelines)
    assert len(results) == 3
    for r in results:
        assert r.success


@pytest.mark.asyncio
async def test_coordinate_batch_empty() -> None:
    runtime = _make_runtime()
    async with runtime:
        results = await runtime.coordinate_batch([], [])
    assert results == []


@pytest.mark.asyncio
async def test_coordinate_batch_length_mismatch_raises() -> None:
    runtime = _make_runtime()
    tasks = _make_tasks(3)
    pipelines = [["agent1"]] * 2  # wrong length
    with pytest.raises(ValueError, match="same length"):
        async with runtime:
            await runtime.coordinate_batch(tasks, pipelines)


@pytest.mark.asyncio
async def test_coordinate_batch_different_pipelines() -> None:
    runtime = _make_runtime()
    tasks = [
        TaskSpec(content="Task A"),
        TaskSpec(content="Task B"),
    ]
    pipelines = [["agent-a"], ["agent-b", "agent-c"]]
    async with runtime:
        results = await runtime.coordinate_batch(tasks, pipelines)
    assert len(results) == 2
    # Task B has a 2-agent pipeline → 2 routing decisions
    assert len(results[1].routing_decisions) == 2


@pytest.mark.asyncio
async def test_coordinate_batch_results_in_order() -> None:
    runtime = _make_runtime()
    tasks = [TaskSpec(content=f"task-{i}") for i in range(5)]
    pipelines = [["agent1"]] * 5
    async with runtime:
        results = await runtime.coordinate_batch(tasks, pipelines)
    task_ids = [t.task_id for t in tasks]
    result_ids = [r.task_id for r in results]
    assert task_ids == result_ids


@pytest.mark.asyncio
async def test_coordinate_batch_large_batch() -> None:
    """Verify batch handles 20 tasks without raising."""
    runtime = _make_runtime()
    tasks = _make_tasks(20)
    pipelines = [["agent1"]] * 20
    async with runtime:
        results = await runtime.coordinate_batch(tasks, pipelines)
    assert len(results) == 20
