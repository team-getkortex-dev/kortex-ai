"""Tests for the OptimizationPlayground."""

from __future__ import annotations

import pytest

from kortex.core.optimization import (
    OptimizationPlayground,
    OptimizationResult,
    PolicyEvaluation,
    _pareto_frontier,
    _knee_point,
)
from kortex.core.trace import TaskTrace, TraceStep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_evaluation(name: str, cost: float, latency: float) -> PolicyEvaluation:
    return PolicyEvaluation(
        policy_name=name,
        policy_dict={"name": name},
        avg_cost_usd=cost,
        avg_latency_ms=latency,
        total_cost_usd=cost * 10,
        total_latency_ms=latency * 10,
        num_tasks=10,
    )


def _make_trace(provider: str = "openai", model: str = "gpt-4o-mini") -> TaskTrace:
    step = TraceStep(
        step_index=0,
        agent_id="agent_a",
        input_payload={"content": "test"},
        routing_decision={
            "chosen_provider": provider,
            "chosen_model": model,
            "estimated_cost_usd": 0.001,
            "estimated_latency_ms": 200.0,
        },
        policy_snapshot={"name": "cost_optimized"},
    )
    return TaskTrace(
        task_id="t1",
        task_content="test",
        task_complexity="simple",
        pipeline=["agent_a"],
        steps=[step],
        success=True,
    )


# ---------------------------------------------------------------------------
# _pareto_frontier
# ---------------------------------------------------------------------------


def test_pareto_frontier_single() -> None:
    evals = [_make_evaluation("only", 0.001, 100.0)]
    frontier = _pareto_frontier(evals)
    assert len(frontier) == 1
    assert frontier[0].policy_name == "only"


def test_pareto_frontier_dominated_removed() -> None:
    # A dominates B: A has lower cost AND lower latency
    a = _make_evaluation("A", 0.001, 100.0)
    b = _make_evaluation("B", 0.002, 200.0)
    frontier = _pareto_frontier([a, b])
    names = [e.policy_name for e in frontier]
    assert "A" in names
    assert "B" not in names


def test_pareto_frontier_both_on_frontier() -> None:
    # A beats B on cost, B beats A on latency — both on frontier
    a = _make_evaluation("A", 0.001, 500.0)
    b = _make_evaluation("B", 0.010, 100.0)
    frontier = _pareto_frontier([a, b])
    assert len(frontier) == 2


def test_pareto_frontier_multiple() -> None:
    evals = [
        _make_evaluation("cheap", 0.0005, 800.0),
        _make_evaluation("balanced", 0.002, 300.0),
        _make_evaluation("fast", 0.005, 100.0),
        _make_evaluation("dominated", 0.003, 500.0),  # dominated by balanced
    ]
    frontier = _pareto_frontier(evals)
    names = {e.policy_name for e in frontier}
    assert "cheap" in names
    assert "balanced" in names
    assert "fast" in names
    assert "dominated" not in names


def test_pareto_frontier_sorted_by_cost() -> None:
    evals = [
        _make_evaluation("fast", 0.010, 100.0),
        _make_evaluation("cheap", 0.001, 900.0),
        _make_evaluation("mid", 0.005, 400.0),
    ]
    frontier = _pareto_frontier(evals)
    # Should be sorted by cost ascending
    costs = [e.avg_cost_usd for e in frontier]
    assert costs == sorted(costs)


# ---------------------------------------------------------------------------
# _knee_point
# ---------------------------------------------------------------------------


def test_knee_point_empty() -> None:
    assert _knee_point([]) is None


def test_knee_point_single() -> None:
    evals = [_make_evaluation("only", 0.001, 100.0)]
    result = _knee_point(evals)
    assert result is not None
    assert result.policy_name == "only"


def test_knee_point_picks_balanced() -> None:
    # Three points on a Pareto front: cheapest, fastest, balanced
    cheapest = _make_evaluation("cheap", 0.001, 1000.0)
    fastest = _make_evaluation("fast", 0.010, 100.0)
    balanced = _make_evaluation("balanced", 0.004, 400.0)
    knee = _knee_point([cheapest, fastest, balanced])
    # Balanced should be closest to ideal (0,0) after normalisation
    assert knee is not None
    assert knee.policy_name == "balanced"


# ---------------------------------------------------------------------------
# PolicyEvaluation
# ---------------------------------------------------------------------------


def test_policy_evaluation_to_dict() -> None:
    e = _make_evaluation("test", 0.001, 200.0)
    d = e.to_dict()
    assert d["policy_name"] == "test"
    assert "avg_cost_usd" in d
    assert "avg_latency_ms" in d
    assert "pareto_optimal" in d


def test_policy_evaluation_str() -> None:
    e = _make_evaluation("test", 0.001, 200.0)
    e.pareto_optimal = True
    s = str(e)
    assert "PARETO" in s
    assert "test" in s


# ---------------------------------------------------------------------------
# OptimizationPlayground
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_optimize_empty_traces() -> None:
    from unittest.mock import MagicMock

    router = MagicMock()
    playground = OptimizationPlayground(router)

    result = await playground.optimize([])
    assert result.num_traces == 0
    assert result.evaluations == []
    assert result.pareto_frontier == []


@pytest.mark.asyncio
async def test_optimize_returns_result_structure() -> None:
    from kortex.core.router import ProviderModel, Router

    router = Router()
    for m in [
        ProviderModel(
            provider="openai", model="gpt-4o-mini",
            cost_per_1k_input_tokens=0.00015, cost_per_1k_output_tokens=0.0006,
            avg_latency_ms=200, capabilities=["reasoning", "content_generation"],
            tier="fast",
        ),
        ProviderModel(
            provider="anthropic", model="claude-opus",
            cost_per_1k_input_tokens=0.015, cost_per_1k_output_tokens=0.075,
            avg_latency_ms=2000, capabilities=["reasoning", "code_generation"],
            tier="powerful",
        ),
    ]:
        router.register_model(m)

    traces = [_make_trace() for _ in range(3)]
    playground = OptimizationPlayground(router)

    result = await playground.optimize(traces, cost_weights=[0.3, 0.7])

    assert isinstance(result, OptimizationResult)
    assert result.num_traces == 3
    assert result.num_policies_evaluated > 0
    assert len(result.evaluations) > 0
    assert result.pareto_frontier is not None
    assert result.best_cost is not None
    assert result.best_latency is not None


@pytest.mark.asyncio
async def test_optimize_pareto_frontier_subset() -> None:
    from kortex.core.router import ProviderModel, Router

    router = Router()
    router.register_model(ProviderModel(
        provider="openai", model="gpt-4o-mini",
        cost_per_1k_input_tokens=0.00015, cost_per_1k_output_tokens=0.0006,
        avg_latency_ms=200, tier="fast",
    ))

    traces = [_make_trace() for _ in range(2)]
    playground = OptimizationPlayground(router, max_concurrent=2)

    result = await playground.optimize(traces, cost_weights=[0.3, 0.7])

    # All Pareto-optimal should have pareto_optimal=True
    for e in result.pareto_frontier:
        assert e.pareto_optimal is True


@pytest.mark.asyncio
async def test_optimize_summary() -> None:
    from kortex.core.router import ProviderModel, Router

    router = Router()
    router.register_model(ProviderModel(
        provider="openai", model="gpt-4o-mini",
        cost_per_1k_input_tokens=0.0, cost_per_1k_output_tokens=0.0,
        avg_latency_ms=100, tier="fast",
    ))

    traces = [_make_trace()]
    playground = OptimizationPlayground(router)
    result = await playground.optimize(traces, cost_weights=[0.5])

    summary = result.summary()
    assert "trace" in summary.lower()
    assert "pareto" in summary.lower()
