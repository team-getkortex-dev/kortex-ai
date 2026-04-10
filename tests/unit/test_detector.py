"""Unit tests for the failure detection system."""

from __future__ import annotations

import pytest

from kortex.core.detector import (
    AnomalyReport,
    AnomalyType,
    DetectionPolicy,
    FailureDetector,
)
from kortex.core.types import (
    CoordinationResult,
    HandoffContext,
    RoutingDecision,
    TaskSpec,
)


def _make_decision(**overrides: object) -> RoutingDecision:
    defaults = {
        "task_id": "t1",
        "chosen_provider": "openai",
        "chosen_model": "gpt-4o-mini",
        "reasoning": "test",
        "estimated_cost_usd": 0.01,
        "estimated_latency_ms": 200.0,
    }
    defaults.update(overrides)
    return RoutingDecision(**defaults)  # type: ignore[arg-type]


def _make_task(**overrides: object) -> TaskSpec:
    defaults = {"content": "test task"}
    defaults.update(overrides)
    return TaskSpec(**defaults)  # type: ignore[arg-type]


def _make_handoff(**overrides: object) -> HandoffContext:
    defaults = {
        "source_agent": "a",
        "target_agent": "b",
        "state_snapshot": {"key": "value"},
    }
    defaults.update(overrides)
    return HandoffContext(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 1. No anomaly when within thresholds
# ---------------------------------------------------------------------------


class TestNoAnomaly:
    @pytest.mark.asyncio
    async def test_no_anomaly_routing(self) -> None:
        detector = FailureDetector()
        decision = _make_decision()
        task = _make_task()
        result = await detector.check_routing(decision, task)
        assert result is None

    @pytest.mark.asyncio
    async def test_no_anomaly_execution(self) -> None:
        detector = FailureDetector()
        decision = _make_decision(estimated_cost_usd=0.01, estimated_latency_ms=200.0)
        response = {
            "content": "A perfectly fine response with enough length.",
            "cost_usd": 0.01,
            "latency_ms": 200.0,
        }
        result = await detector.check_execution(response, decision)
        assert result is None

    @pytest.mark.asyncio
    async def test_no_anomaly_handoff(self) -> None:
        detector = FailureDetector()
        prev = _make_handoff(state_snapshot={"data": "x" * 100})
        curr = _make_handoff(state_snapshot={"data": "x" * 80})
        result = await detector.check_handoff(curr, prev)
        assert result is None

    @pytest.mark.asyncio
    async def test_no_anomaly_coordination(self) -> None:
        detector = FailureDetector()
        result = CoordinationResult(task_id="t1", handoffs=[], success=True)
        anomalies = await detector.check_coordination(result)
        assert anomalies == []


# ---------------------------------------------------------------------------
# 2. COST_OVERRUN detected when actual > multiplier * estimated
# ---------------------------------------------------------------------------


class TestCostOverrun:
    @pytest.mark.asyncio
    async def test_cost_overrun_execution(self) -> None:
        detector = FailureDetector(DetectionPolicy(max_cost_multiplier=2.0))
        decision = _make_decision(estimated_cost_usd=0.01)
        response = {
            "content": "Normal response that is long enough.",
            "cost_usd": 0.025,  # 2.5x the estimate
            "latency_ms": 200.0,
        }
        result = await detector.check_execution(response, decision)
        assert result is not None
        assert result.anomaly_type == AnomalyType.COST_OVERRUN
        assert result.recommended_action == "fallback"

    @pytest.mark.asyncio
    async def test_cost_overrun_within_limit(self) -> None:
        detector = FailureDetector(DetectionPolicy(max_cost_multiplier=2.0))
        decision = _make_decision(estimated_cost_usd=0.01)
        response = {
            "content": "Normal response that is long enough.",
            "cost_usd": 0.015,  # 1.5x — within limit
            "latency_ms": 200.0,
        }
        result = await detector.check_execution(response, decision)
        assert result is None


# ---------------------------------------------------------------------------
# 3. LATENCY_SPIKE detected
# ---------------------------------------------------------------------------


class TestLatencySpike:
    @pytest.mark.asyncio
    async def test_latency_spike(self) -> None:
        detector = FailureDetector(DetectionPolicy(max_latency_multiplier=3.0))
        decision = _make_decision(estimated_latency_ms=200.0)
        response = {
            "content": "Normal response that is long enough.",
            "cost_usd": 0.01,
            "latency_ms": 700.0,  # 3.5x the estimate
        }
        result = await detector.check_execution(response, decision)
        assert result is not None
        assert result.anomaly_type == AnomalyType.LATENCY_SPIKE
        assert result.recommended_action == "retry"


# ---------------------------------------------------------------------------
# 4. OUTPUT_QUALITY_DROP for short response
# ---------------------------------------------------------------------------


class TestOutputQualityDrop:
    @pytest.mark.asyncio
    async def test_short_output(self) -> None:
        detector = FailureDetector(DetectionPolicy(min_output_length=10))
        decision = _make_decision()
        response = {
            "content": "Short",
            "cost_usd": 0.001,
            "latency_ms": 100.0,
        }
        result = await detector.check_execution(response, decision)
        assert result is not None
        assert result.anomaly_type == AnomalyType.OUTPUT_QUALITY_DROP

    @pytest.mark.asyncio
    async def test_empty_output(self) -> None:
        detector = FailureDetector(DetectionPolicy(min_output_length=10))
        decision = _make_decision()
        response = {"content": "", "cost_usd": 0.001, "latency_ms": 100.0}
        result = await detector.check_execution(response, decision)
        assert result is not None
        assert result.anomaly_type == AnomalyType.OUTPUT_QUALITY_DROP


# ---------------------------------------------------------------------------
# 5. CONTEXT_DEGRADATION when state shrinks >80%
# ---------------------------------------------------------------------------


class TestContextDegradation:
    @pytest.mark.asyncio
    async def test_context_shrunk(self) -> None:
        detector = FailureDetector()
        prev = _make_handoff(state_snapshot={"data": "x" * 1000})
        curr = _make_handoff(state_snapshot={"d": "y"})
        result = await detector.check_handoff(curr, prev)
        assert result is not None
        assert result.anomaly_type == AnomalyType.CONTEXT_DEGRADATION
        assert result.recommended_action == "rollback"


# ---------------------------------------------------------------------------
# 6. CHECKPOINT_CHAIN_BREAK at depth limit
# ---------------------------------------------------------------------------


class TestCheckpointChainBreak:
    @pytest.mark.asyncio
    async def test_chain_too_deep(self) -> None:
        detector = FailureDetector(DetectionPolicy(max_chain_depth=5))
        handoffs = [_make_handoff() for _ in range(10)]
        result = CoordinationResult(task_id="t1", handoffs=handoffs, success=True)
        anomalies = await detector.check_coordination(result)
        assert len(anomalies) == 1
        assert anomalies[0].anomaly_type == AnomalyType.CHECKPOINT_CHAIN_BREAK
        assert anomalies[0].recommended_action == "escalate"


# ---------------------------------------------------------------------------
# 7. Custom rule fires correctly
# ---------------------------------------------------------------------------


class TestCustomRule:
    @pytest.mark.asyncio
    async def test_custom_rule_fires(self) -> None:
        async def always_flag(result: CoordinationResult) -> AnomalyReport | None:
            return AnomalyReport(
                anomaly_type=AnomalyType.UNEXPECTED_TOOL_CHOICE,
                severity="low",
                task_id=result.task_id,
                description="Custom rule triggered",
                recommended_action="continue",
            )

        detector = FailureDetector()
        detector.register_custom_rule("always_flag", always_flag)

        result = CoordinationResult(task_id="t1", success=True)
        anomalies = await detector.check_coordination(result)
        assert len(anomalies) == 1
        assert anomalies[0].anomaly_type == AnomalyType.UNEXPECTED_TOOL_CHOICE

    @pytest.mark.asyncio
    async def test_custom_rule_returns_none(self) -> None:
        async def never_flag(result: CoordinationResult) -> AnomalyReport | None:
            return None

        detector = FailureDetector()
        detector.register_custom_rule("never_flag", never_flag)

        result = CoordinationResult(task_id="t1", success=True)
        anomalies = await detector.check_coordination(result)
        assert anomalies == []


# ---------------------------------------------------------------------------
# 8. Custom thresholds override defaults
# ---------------------------------------------------------------------------


class TestCustomThresholds:
    @pytest.mark.asyncio
    async def test_stricter_cost_threshold(self) -> None:
        detector = FailureDetector(DetectionPolicy(max_cost_multiplier=1.1))
        decision = _make_decision(estimated_cost_usd=0.01)
        response = {
            "content": "Normal response that is long enough.",
            "cost_usd": 0.012,  # 1.2x — exceeds 1.1x threshold
            "latency_ms": 100.0,
        }
        result = await detector.check_execution(response, decision)
        assert result is not None
        assert result.anomaly_type == AnomalyType.COST_OVERRUN

    @pytest.mark.asyncio
    async def test_relaxed_latency_threshold(self) -> None:
        detector = FailureDetector(DetectionPolicy(max_latency_multiplier=10.0))
        decision = _make_decision(estimated_latency_ms=200.0)
        response = {
            "content": "Normal response that is long enough.",
            "cost_usd": 0.01,
            "latency_ms": 1500.0,  # 7.5x — within 10x threshold
        }
        result = await detector.check_execution(response, decision)
        assert result is None

    def test_policy_defaults(self) -> None:
        policy = DetectionPolicy()
        assert policy.max_retries == 3
        assert policy.max_cost_multiplier == 2.0
        assert policy.max_latency_multiplier == 3.0
        assert policy.min_output_length == 10
        assert policy.max_chain_depth == 50
