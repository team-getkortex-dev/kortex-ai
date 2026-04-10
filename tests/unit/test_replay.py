"""Tests for the replay engine."""

from __future__ import annotations

import pytest

from kortex.core.policy import (
    FallbackRule,
    RoutingConstraint,
    RoutingObjective,
    RoutingPolicy,
)
from kortex.core.replay import ReplayEngine, ReplayResult
from kortex.core.router import ProviderModel, Router
from kortex.core.trace import TaskTrace, TraceStep


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _cheap_fast() -> ProviderModel:
    return ProviderModel(
        provider="local", model="tiny-7b",
        cost_per_1k_input_tokens=0.0001, cost_per_1k_output_tokens=0.0002,
        avg_latency_ms=50, capabilities=["reasoning"], tier="fast",
    )


def _balanced() -> ProviderModel:
    return ProviderModel(
        provider="anthropic", model="claude-sonnet",
        cost_per_1k_input_tokens=0.003, cost_per_1k_output_tokens=0.015,
        avg_latency_ms=800,
        capabilities=["reasoning", "code_generation", "content_generation"],
        tier="balanced",
    )


def _powerful() -> ProviderModel:
    return ProviderModel(
        provider="anthropic", model="claude-opus",
        cost_per_1k_input_tokens=0.015, cost_per_1k_output_tokens=0.075,
        avg_latency_ms=2000,
        capabilities=["reasoning", "code_generation", "content_generation", "vision"],
        tier="powerful",
    )


def _openai_balanced() -> ProviderModel:
    return ProviderModel(
        provider="openai", model="gpt-4o",
        cost_per_1k_input_tokens=0.005, cost_per_1k_output_tokens=0.015,
        avg_latency_ms=600,
        capabilities=["reasoning", "code_generation", "content_generation"],
        tier="balanced",
    )


def _make_router() -> Router:
    router = Router()
    for m in [_cheap_fast(), _balanced(), _powerful(), _openai_balanced()]:
        router.register_model(m)
    return router


def _make_trace(policy_name: str = "cost_optimized") -> TaskTrace:
    """Build a sample trace that was produced under cost_optimized policy.

    The trace shows two steps that chose the cheapest models.
    """
    cost_policy = RoutingPolicy.cost_optimized()
    policy_dict = cost_policy.to_dict()

    return TaskTrace(
        trace_id="trace-replay-001",
        task_id="task-001",
        task_content="Test replay task",
        task_complexity="moderate",
        pipeline=["researcher", "writer"],
        steps=[
            TraceStep(
                step_index=0,
                agent_id="researcher",
                input_payload={"content": "Test replay task", "task_id": "task-001"},
                routing_decision={
                    "task_id": "task-001",
                    "chosen_provider": "local",
                    "chosen_model": "tiny-7b",
                    "estimated_cost_usd": _cheap_fast().estimated_cost(),
                    "estimated_latency_ms": 50,
                },
                policy_snapshot=policy_dict,
                started_at="2026-03-29T10:00:00+00:00",
                completed_at="2026-03-29T10:00:01+00:00",
                duration_ms=50.0,
            ),
            TraceStep(
                step_index=1,
                agent_id="writer",
                input_payload={"content": "Test replay task", "task_id": "task-001"},
                routing_decision={
                    "task_id": "task-001",
                    "chosen_provider": "local",
                    "chosen_model": "tiny-7b",
                    "estimated_cost_usd": _cheap_fast().estimated_cost(),
                    "estimated_latency_ms": 50,
                },
                policy_snapshot=policy_dict,
                started_at="2026-03-29T10:00:01+00:00",
                completed_at="2026-03-29T10:00:02+00:00",
                duration_ms=50.0,
            ),
        ],
        policy_snapshot=policy_dict,
        total_estimated_cost_usd=_cheap_fast().estimated_cost() * 2,
        total_actual_cost_usd=0.0,
        total_duration_ms=100.0,
        success=True,
    )


# ---------------------------------------------------------------------------
# 5. Replay under same policy produces identical model selections
# ---------------------------------------------------------------------------


class TestReplaySamePolicy:
    @pytest.mark.asyncio
    async def test_same_policy_no_changes(self) -> None:
        router = _make_router()
        engine = ReplayEngine(router)
        trace = _make_trace()

        # Replay under the same cost_optimized policy
        result = await engine.replay(trace, RoutingPolicy.cost_optimized())

        for step in result.replayed_steps:
            assert not step.model_changed
            assert step.replayed_model == step.original_model
            assert step.cost_delta == 0.0


# ---------------------------------------------------------------------------
# 6. Replay under cost_optimized policy changes selections toward cheaper
# ---------------------------------------------------------------------------


class TestReplayCostOptimized:
    @pytest.mark.asyncio
    async def test_cost_optimized_picks_cheap(self) -> None:
        """Trace was under quality policy (expensive), replay under cost."""
        router = _make_router()
        engine = ReplayEngine(router)

        # Build a trace that was originally under quality_optimized
        quality_policy = RoutingPolicy.quality_optimized()
        trace = TaskTrace(
            trace_id="trace-quality",
            task_id="task-002",
            task_content="Quality task",
            task_complexity="moderate",
            pipeline=["agent_a"],
            steps=[
                TraceStep(
                    step_index=0,
                    agent_id="agent_a",
                    input_payload={"content": "Quality task"},
                    routing_decision={
                        "task_id": "task-002",
                        "chosen_provider": "anthropic",
                        "chosen_model": "claude-opus",
                        "estimated_cost_usd": _powerful().estimated_cost(),
                        "estimated_latency_ms": 2000,
                    },
                    policy_snapshot=quality_policy.to_dict(),
                ),
            ],
            policy_snapshot=quality_policy.to_dict(),
            total_estimated_cost_usd=_powerful().estimated_cost(),
        )

        result = await engine.replay(trace, RoutingPolicy.cost_optimized())

        step = result.replayed_steps[0]
        assert step.model_changed
        assert step.replayed_model == "tiny-7b"
        assert step.cost_delta < 0  # cheaper


# ---------------------------------------------------------------------------
# 7. Replay under quality_optimized policy changes toward powerful models
# ---------------------------------------------------------------------------


class TestReplayQualityOptimized:
    @pytest.mark.asyncio
    async def test_quality_picks_powerful(self) -> None:
        router = _make_router()
        engine = ReplayEngine(router)
        trace = _make_trace()  # originally cost_optimized (tiny-7b)

        result = await engine.replay(trace, RoutingPolicy.quality_optimized())

        for step in result.replayed_steps:
            assert step.model_changed
            assert step.replayed_model == "claude-opus"
            assert step.cost_delta > 0  # more expensive


# ---------------------------------------------------------------------------
# 8. policy_diff summary shows correct changed steps and cost delta
# ---------------------------------------------------------------------------


class TestPolicyDiff:
    @pytest.mark.asyncio
    async def test_diff_summary(self) -> None:
        router = _make_router()
        engine = ReplayEngine(router)
        trace = _make_trace()

        result = await engine.policy_diff(trace, RoutingPolicy.quality_optimized())

        # All 2 steps should change
        assert "2 of 2" in result.summary
        assert "would change model selection" in result.summary
        # Should show increase (quality is more expensive)
        assert "increase" in result.summary


# ---------------------------------------------------------------------------
# 9. what_if with denied_providers excludes that provider
# ---------------------------------------------------------------------------


class TestWhatIf:
    @pytest.mark.asyncio
    async def test_what_if_deny_provider(self) -> None:
        router = _make_router()
        engine = ReplayEngine(router)
        trace = _make_trace()  # originally chose local/tiny-7b

        result = await engine.what_if(trace, {"denied_providers": ["local"]})

        for step in result.replayed_steps:
            assert step.replayed_provider != "local"
            assert step.model_changed  # can't use local anymore


# ---------------------------------------------------------------------------
# 10. Replay with no policy change shows zero diffs
# ---------------------------------------------------------------------------


class TestReplayZeroDiff:
    @pytest.mark.asyncio
    async def test_no_changes_zero_diff(self) -> None:
        router = _make_router()
        engine = ReplayEngine(router)
        trace = _make_trace()

        result = await engine.replay(trace, RoutingPolicy.cost_optimized())

        changed = sum(1 for s in result.replayed_steps if s.model_changed)
        assert changed == 0
        assert "no model changes" in result.summary


# ---------------------------------------------------------------------------
# 11. cost_delta positive when more expensive, negative when cheaper
# ---------------------------------------------------------------------------


class TestCostDeltaSign:
    @pytest.mark.asyncio
    async def test_cost_delta_signs(self) -> None:
        router = _make_router()
        engine = ReplayEngine(router)

        # Trace was cheap (cost_optimized), replay under quality
        trace = _make_trace()
        result = await engine.replay(trace, RoutingPolicy.quality_optimized())
        for step in result.replayed_steps:
            assert step.cost_delta > 0  # more expensive

        # Trace was expensive (quality), replay under cost
        quality_trace = TaskTrace(
            trace_id="trace-q",
            task_id="task-q",
            task_content="Q task",
            task_complexity="moderate",
            pipeline=["a"],
            steps=[TraceStep(
                step_index=0, agent_id="a",
                input_payload={"content": "Q task"},
                routing_decision={
                    "task_id": "task-q",
                    "chosen_provider": "anthropic",
                    "chosen_model": "claude-opus",
                    "estimated_cost_usd": _powerful().estimated_cost(),
                    "estimated_latency_ms": 2000,
                },
                policy_snapshot=RoutingPolicy.quality_optimized().to_dict(),
            )],
            policy_snapshot=RoutingPolicy.quality_optimized().to_dict(),
            total_estimated_cost_usd=_powerful().estimated_cost(),
        )
        result2 = await engine.replay(quality_trace, RoutingPolicy.cost_optimized())
        for step in result2.replayed_steps:
            assert step.cost_delta < 0  # cheaper
