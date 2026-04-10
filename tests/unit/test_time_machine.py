"""Tests for TimeMachine and extended ReplayEngine."""

from __future__ import annotations

import pytest

from kortex.core.trace import TaskTrace, TraceStep
from kortex.core.time_machine import ExecutionSnapshot, TimeMachine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trace(n_steps: int = 3) -> TaskTrace:
    steps = []
    agents = ["researcher", "writer", "reviewer"]
    providers = ["openai", "anthropic", "openai"]
    models = ["gpt-4o-mini", "claude-sonnet", "gpt-4o-mini"]
    costs = [0.0004, 0.0105, 0.0004]
    latencies = [200.0, 800.0, 200.0]

    for i in range(n_steps):
        steps.append(TraceStep(
            step_index=i,
            agent_id=agents[i % len(agents)],
            input_payload={
                "content": f"step {i} content",
                "required_capabilities": ["reasoning"],
            },
            routing_decision={
                "chosen_provider": providers[i % len(providers)],
                "chosen_model": models[i % len(models)],
                "estimated_cost_usd": costs[i % len(costs)],
                "estimated_latency_ms": latencies[i % len(latencies)],
            },
            policy_snapshot={"name": "cost_optimized"},
            duration_ms=latencies[i % len(latencies)] + 10,
        ))

    total_cost = sum(costs[:n_steps])
    return TaskTrace(
        task_id="t1",
        task_content="test task",
        task_complexity="moderate",
        pipeline=[agents[i % len(agents)] for i in range(n_steps)],
        steps=steps,
        total_estimated_cost_usd=total_cost,
        total_duration_ms=sum(latencies[:n_steps]),
        success=True,
    )


# ---------------------------------------------------------------------------
# TimeMachine basic API
# ---------------------------------------------------------------------------


def test_time_machine_num_steps() -> None:
    trace = _make_trace(3)
    tm = TimeMachine(trace)
    assert tm.num_steps == 3


def test_time_machine_snapshot_returns_execution_snapshot() -> None:
    trace = _make_trace(3)
    tm = TimeMachine(trace)
    snap = tm.snapshot(0)
    assert isinstance(snap, ExecutionSnapshot)


def test_time_machine_snapshot_step_index() -> None:
    trace = _make_trace(3)
    tm = TimeMachine(trace)
    snap = tm.snapshot(1)
    assert snap.step_index == 1
    assert snap.agent_id == "writer"


def test_time_machine_snapshot_out_of_range() -> None:
    trace = _make_trace(2)
    tm = TimeMachine(trace)
    with pytest.raises(IndexError):
        tm.snapshot(5)


def test_time_machine_snapshot_negative_index() -> None:
    trace = _make_trace(2)
    tm = TimeMachine(trace)
    with pytest.raises(IndexError):
        tm.snapshot(-1)


def test_time_machine_cumulative_cost() -> None:
    trace = _make_trace(3)
    tm = TimeMachine(trace)

    # Step 0: no prior steps
    snap0 = tm.snapshot(0)
    assert snap0.cumulative_cost_usd == 0.0

    # Step 1: cost from step 0
    snap1 = tm.snapshot(1)
    assert snap1.cumulative_cost_usd == pytest.approx(0.0004, abs=1e-6)


def test_time_machine_snapshots_returns_all() -> None:
    trace = _make_trace(3)
    tm = TimeMachine(trace)
    snaps = tm.snapshots()
    assert len(snaps) == 3
    for i, snap in enumerate(snaps):
        assert snap.step_index == i


def test_time_machine_restore() -> None:
    trace = _make_trace(3)
    tm = TimeMachine(trace)
    snap = tm.snapshot(1)
    ctx = tm.restore(snap)

    assert "remaining_agents" in ctx
    assert ctx["from_step"] == 1
    assert len(ctx["remaining_agents"]) == 2  # steps 1 and 2


def test_time_machine_full_summary() -> None:
    trace = _make_trace(3)
    tm = TimeMachine(trace)
    summary = tm.full_summary()
    assert "Trace:" in summary
    assert "Steps: 3" in summary
    assert "researcher" in summary


def test_time_machine_step_summary() -> None:
    trace = _make_trace(2)
    tm = TimeMachine(trace)
    line = tm.step_summary(0)
    assert "[0]" in line
    assert "researcher" in line
    assert "openai" in line


# ---------------------------------------------------------------------------
# ExecutionSnapshot serialisation
# ---------------------------------------------------------------------------


def test_execution_snapshot_to_dict_and_back() -> None:
    trace = _make_trace(2)
    tm = TimeMachine(trace)
    snap = tm.snapshot(0)
    d = snap.to_dict()

    assert d["step_index"] == 0
    assert d["agent_id"] == "researcher"

    restored = ExecutionSnapshot.from_dict(d)
    assert restored.step_index == snap.step_index
    assert restored.agent_id == snap.agent_id
    assert restored.routing_decision == snap.routing_decision


# ---------------------------------------------------------------------------
# ReplayEngine.replay_from_step
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_replay_from_step_returns_suffix() -> None:
    from kortex.core.policy import RoutingPolicy
    from kortex.core.replay import ReplayEngine
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
            provider="anthropic", model="claude-sonnet",
            cost_per_1k_input_tokens=0.003, cost_per_1k_output_tokens=0.015,
            avg_latency_ms=800, capabilities=["reasoning", "code_generation"],
            tier="balanced",
        ),
    ]:
        router.register_model(m)

    trace = _make_trace(3)
    engine = ReplayEngine(router)

    # Replay from step 1 → should return 2 replayed steps (steps 1 and 2)
    result = await engine.replay_from_step(trace, from_step=1)
    assert len(result.replayed_steps) == 2
    assert result.from_step == 1
    assert "from step 1" in result.summary


@pytest.mark.asyncio
async def test_replay_from_step_out_of_range() -> None:
    from kortex.core.replay import ReplayEngine
    from kortex.core.router import ProviderModel, Router

    router = Router()
    router.register_model(ProviderModel(
        provider="openai", model="gpt-4o-mini",
        cost_per_1k_input_tokens=0.0, cost_per_1k_output_tokens=0.0,
        avg_latency_ms=100, tier="fast",
    ))

    trace = _make_trace(2)
    engine = ReplayEngine(router)

    with pytest.raises(IndexError):
        await engine.replay_from_step(trace, from_step=10)


@pytest.mark.asyncio
async def test_replay_from_step_zero_equals_full_replay() -> None:
    from kortex.core.replay import ReplayEngine
    from kortex.core.router import ProviderModel, Router

    router = Router()
    for m in [
        ProviderModel(
            provider="openai", model="gpt-4o-mini",
            cost_per_1k_input_tokens=0.0, cost_per_1k_output_tokens=0.0,
            avg_latency_ms=100, tier="fast",
        ),
    ]:
        router.register_model(m)

    trace = _make_trace(3)
    engine = ReplayEngine(router)

    result_full = await engine.replay(trace)
    result_from0 = await engine.replay_from_step(trace, from_step=0)

    assert len(result_full.replayed_steps) == len(result_from0.replayed_steps)


# ---------------------------------------------------------------------------
# ReplayResult.diff
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_replay_result_diff_same_policy() -> None:
    from kortex.core.replay import ReplayEngine
    from kortex.core.router import ProviderModel, Router

    router = Router()
    router.register_model(ProviderModel(
        provider="openai", model="gpt-4o-mini",
        cost_per_1k_input_tokens=0.0, cost_per_1k_output_tokens=0.0,
        avg_latency_ms=100, tier="fast",
    ))

    trace = _make_trace(2)
    engine = ReplayEngine(router)

    result_a = await engine.replay(trace)
    result_b = await engine.replay(trace)  # same policy

    diff = result_a.diff(result_b)
    assert diff["changed_steps"] == 0
    assert diff["cost_delta"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.asyncio
async def test_replay_result_diff_different_policies() -> None:
    from kortex.core.policy import RoutingPolicy
    from kortex.core.replay import ReplayEngine
    from kortex.core.router import ProviderModel, Router

    router = Router()
    for m in [
        ProviderModel(
            provider="openai", model="gpt-4o-mini",
            cost_per_1k_input_tokens=0.00015, cost_per_1k_output_tokens=0.0006,
            avg_latency_ms=200, capabilities=["reasoning"],
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

    trace = _make_trace(2)
    engine = ReplayEngine(router)

    result_cost = await engine.replay(trace, policy=RoutingPolicy.cost_optimized())
    result_quality = await engine.replay(trace, policy=RoutingPolicy.quality_optimized())

    diff = result_cost.diff(result_quality)
    assert "step_diffs" in diff
    assert "total_cost_self" in diff
    assert "total_cost_other" in diff
    assert diff["policy_self"] == "cost_optimized"
    assert diff["policy_other"] == "quality_optimized"


def test_replay_result_diff_keys() -> None:
    """Smoke test diff structure without async."""
    from kortex.core.replay import ReplayResult, ReplayedStep
    from kortex.core.trace import TaskTrace

    def _make_result(model: str, cost: float, policy_name: str) -> ReplayResult:
        steps = [ReplayedStep(
            step_index=0, agent_id="a",
            original_model=model, original_provider="p",
            replayed_model=model, replayed_provider="p",
            model_changed=False,
            original_estimated_cost=cost, replayed_estimated_cost=cost,
            cost_delta=0.0, explanation="",
        )]
        return ReplayResult(
            original_trace=TaskTrace(task_id="t"),
            replayed_steps=steps,
            policy_used={"name": policy_name},
            summary="",
        )

    ra = _make_result("gpt-4o-mini", 0.001, "cost_optimized")
    rb = _make_result("claude-opus", 0.02, "quality_optimized")
    diff = ra.diff(rb)

    assert diff["policy_self"] == "cost_optimized"
    assert diff["policy_other"] == "quality_optimized"
    assert diff["cost_delta"] == pytest.approx(0.019, abs=1e-6)
