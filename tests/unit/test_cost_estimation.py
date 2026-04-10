"""Tests for cost estimation (CostEstimate + KortexRuntime.estimate_cost)."""

from __future__ import annotations

import pytest

from kortex.core.router import ProviderModel, Router
from kortex.core.runtime import KortexRuntime
from kortex.core.state import StateManager
from kortex.core.types import TaskSpec
from kortex.router.cost_estimate import CostEstimate
from kortex.store.memory import InMemoryStateStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_runtime(
    cost_in: float = 0.001,
    cost_out: float = 0.002,
    latency: float = 200.0,
) -> KortexRuntime:
    router = Router()
    router.register_model(ProviderModel(
        provider="openai",
        model="gpt-4o-mini",
        cost_per_1k_input_tokens=cost_in,
        cost_per_1k_output_tokens=cost_out,
        avg_latency_ms=latency,
        tier="fast",
    ))
    state = StateManager(InMemoryStateStore())
    return KortexRuntime(router=router, state_manager=state)


def _task(content: str = "Do something", complexity: str = "simple") -> TaskSpec:
    return TaskSpec(content=content, complexity_hint=complexity)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# CostEstimate data class
# ---------------------------------------------------------------------------


def test_cost_estimate_to_dict() -> None:
    est = CostEstimate(
        total_usd=0.01,
        per_model={"openai::gpt-4o-mini": 0.01},
        per_task=[0.01],
        task_count=1,
        routing_failures=0,
    )
    d = est.to_dict()
    assert d["total_usd"] == 0.01
    assert d["task_count"] == 1
    assert d["routing_failures"] == 0


def test_cost_estimate_summary_zero_tasks() -> None:
    est = CostEstimate(total_usd=0.0, task_count=0)
    assert "No tasks" in est.summary()


def test_cost_estimate_summary_with_tasks() -> None:
    est = CostEstimate(
        total_usd=0.05,
        per_model={"openai::gpt": 0.05},
        per_task=[0.025, 0.025],
        task_count=2,
    )
    summary = est.summary()
    assert "0.0500" in summary
    assert "2 task" in summary


def test_cost_estimate_summary_with_failures() -> None:
    est = CostEstimate(
        total_usd=0.0,
        per_model={},
        per_task=[0.0, 0.0],
        task_count=2,
        routing_failures=2,
    )
    summary = est.summary()
    assert "could not be routed" in summary


def test_cost_estimate_summary_top_models() -> None:
    est = CostEstimate(
        total_usd=0.1,
        per_model={
            "openai::gpt-4o": 0.07,
            "anthropic::claude": 0.03,
        },
        per_task=[0.1],
        task_count=1,
    )
    summary = est.summary()
    assert "openai::gpt-4o" in summary


# ---------------------------------------------------------------------------
# KortexRuntime.estimate_cost
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_estimate_cost_single_task() -> None:
    runtime = _make_runtime(cost_in=1.0, cost_out=2.0)
    tasks = [_task()]
    pipelines = [["agent1"]]
    async with runtime:
        est = await runtime.estimate_cost(tasks, pipelines)
    assert est.task_count == 1
    assert est.total_usd > 0
    assert len(est.per_task) == 1
    assert "openai::gpt-4o-mini" in est.per_model


@pytest.mark.asyncio
async def test_estimate_cost_multi_agent_pipeline() -> None:
    runtime = _make_runtime(cost_in=1.0, cost_out=2.0)
    tasks = [_task()]
    pipelines = [["agent1", "agent2"]]
    async with runtime:
        est = await runtime.estimate_cost(tasks, pipelines)
    # 2 agents → cost should be doubled
    single_agent_runtime = _make_runtime(cost_in=1.0, cost_out=2.0)
    async with single_agent_runtime:
        single = await single_agent_runtime.estimate_cost(tasks, [["agent1"]])
    assert abs(est.total_usd - 2 * single.total_usd) < 0.0001


@pytest.mark.asyncio
async def test_estimate_cost_batch_of_tasks() -> None:
    runtime = _make_runtime()
    tasks = [_task(f"task {i}") for i in range(5)]
    pipelines = [["agent1"]] * 5
    async with runtime:
        est = await runtime.estimate_cost(tasks, pipelines)
    assert est.task_count == 5
    assert len(est.per_task) == 5
    assert abs(est.total_usd - sum(est.per_task)) < 0.0001


@pytest.mark.asyncio
async def test_estimate_cost_routing_failure_increments_counter() -> None:
    """A task with an impossible capability should count as a routing failure."""
    from kortex.core.capabilities import Capability

    runtime = _make_runtime()
    tasks = [TaskSpec(
        content="impossible",
        required_capabilities=[Capability.VISION.value],
    )]
    pipelines = [["agent1"]]
    async with runtime:
        est = await runtime.estimate_cost(tasks, pipelines)
    assert est.routing_failures == 1
    assert est.total_usd == 0.0


@pytest.mark.asyncio
async def test_estimate_cost_length_mismatch_raises() -> None:
    runtime = _make_runtime()
    with pytest.raises(ValueError, match="same length"):
        async with runtime:
            await runtime.estimate_cost([_task()], [["a"], ["b"]])


@pytest.mark.asyncio
async def test_estimate_cost_empty() -> None:
    runtime = _make_runtime()
    async with runtime:
        est = await runtime.estimate_cost([], [])
    assert est.task_count == 0
    assert est.total_usd == 0.0
    assert est.per_task == []
