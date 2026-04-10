"""Integration tests for detector + runtime interaction."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from kortex.core.detector import (
    AnomalyReport,
    AnomalyType,
    DetectionPolicy,
    FailureDetector,
)
from kortex.core.router import ProviderModel, Router
from kortex.core.runtime import AgentDescriptor, KortexRuntime
from kortex.core.state import StateManager
from kortex.core.types import CoordinationResult, TaskSpec
from kortex.providers.base import GenericOpenAIConnector, ProviderResponse
from kortex.providers.registry import ProviderRegistry
from kortex.store.memory import InMemoryStateStore


def _models() -> list[ProviderModel]:
    return [
        ProviderModel(
            provider="test",
            model="fast-model",
            cost_per_1k_input_tokens=0.0001,
            cost_per_1k_output_tokens=0.0002,
            avg_latency_ms=100,
            capabilities=["reasoning"],
            tier="fast",
        ),
        ProviderModel(
            provider="test",
            model="balanced-model",
            cost_per_1k_input_tokens=0.003,
            cost_per_1k_output_tokens=0.006,
            avg_latency_ms=500,
            capabilities=["reasoning", "content_generation"],
            tier="balanced",
        ),
    ]


def _build_runtime(
    detector: FailureDetector | None = None,
    registry: ProviderRegistry | None = None,
) -> KortexRuntime:
    router = Router()
    for m in _models():
        router.register_model(m)
    state = StateManager(store=InMemoryStateStore())
    runtime = KortexRuntime(
        router=router,
        state_manager=state,
        registry=registry,
        detector=detector,
    )
    runtime.register_agent(AgentDescriptor("researcher", "Researcher", "Researches"))
    runtime.register_agent(AgentDescriptor("writer", "Writer", "Writes"))
    return runtime


# ---------------------------------------------------------------------------
# 1. No anomalies produces empty list
# ---------------------------------------------------------------------------


class TestNoAnomalies:
    @pytest.mark.asyncio
    async def test_clean_coordination_has_no_anomalies(self) -> None:
        detector = FailureDetector()
        runtime = _build_runtime(detector=detector)
        task = TaskSpec(content="Test task", complexity_hint="simple")
        result = await runtime.coordinate(task, ["researcher", "writer"])
        assert result.success is True
        assert result.anomalies == []


# ---------------------------------------------------------------------------
# 2. Cost overrun triggers fallback
# ---------------------------------------------------------------------------


class TestCostOverrunFallback:
    @pytest.mark.asyncio
    async def test_cost_overrun_in_execution(self) -> None:
        """Execution cost anomaly is recorded in result.anomalies."""
        detector = FailureDetector(DetectionPolicy(max_cost_multiplier=1.5))

        # Set up a registry with a mock provider
        registry = ProviderRegistry()
        connector = GenericOpenAIConnector(
            base_url="http://test/v1",
            api_key="test",
            name="test",
            models=_models(),
        )
        registry.register_provider(connector)  # type: ignore[arg-type]

        runtime = _build_runtime(detector=detector, registry=registry)

        # Mock the complete method to return high cost
        async def mock_complete(prompt: str, model: str, **kwargs: Any) -> ProviderResponse:
            return ProviderResponse(
                content="A full response with enough content for quality check.",
                model=model,
                provider="test",
                input_tokens=100,
                output_tokens=50,
                cost_usd=0.05,  # Way higher than estimate
                latency_ms=100.0,
            )

        connector.complete = mock_complete  # type: ignore[assignment]

        task = TaskSpec(content="Test", complexity_hint="simple")
        result = await runtime.coordinate(task, ["researcher"], execute=True)

        # Should have detected cost overrun
        cost_anomalies = [a for a in result.anomalies if a["anomaly_type"] == "cost_overrun"]
        assert len(cost_anomalies) > 0


# ---------------------------------------------------------------------------
# 3. Latency spike triggers retry
# ---------------------------------------------------------------------------


class TestLatencySpike:
    @pytest.mark.asyncio
    async def test_latency_spike_recorded(self) -> None:
        detector = FailureDetector(DetectionPolicy(max_latency_multiplier=2.0))

        registry = ProviderRegistry()
        connector = GenericOpenAIConnector(
            base_url="http://test/v1",
            api_key="test",
            name="test",
            models=_models(),
        )
        registry.register_provider(connector)  # type: ignore[arg-type]

        runtime = _build_runtime(detector=detector, registry=registry)

        async def mock_complete(prompt: str, model: str, **kwargs: Any) -> ProviderResponse:
            return ProviderResponse(
                content="A full response with enough content for quality check.",
                model=model,
                provider="test",
                input_tokens=100,
                output_tokens=50,
                cost_usd=0.0001,
                latency_ms=5000.0,  # Way higher than estimate
            )

        connector.complete = mock_complete  # type: ignore[assignment]

        task = TaskSpec(content="Test", complexity_hint="simple")
        result = await runtime.coordinate(task, ["researcher"], execute=True)

        latency_anomalies = [a for a in result.anomalies if a["anomaly_type"] == "latency_spike"]
        assert len(latency_anomalies) > 0


# ---------------------------------------------------------------------------
# 4. Critical anomaly marks success=False
# ---------------------------------------------------------------------------


class TestCriticalAnomaly:
    @pytest.mark.asyncio
    async def test_escalation_marks_failure(self) -> None:
        """A custom rule that escalates should set success=False."""

        async def escalate_rule(result: CoordinationResult) -> AnomalyReport | None:
            return AnomalyReport(
                anomaly_type=AnomalyType.COST_OVERRUN,
                severity="critical",
                task_id=result.task_id,
                description="Critical cost overrun",
                recommended_action="escalate",
            )

        policy = DetectionPolicy()
        policy.custom_rules["escalate"] = escalate_rule
        detector = FailureDetector(policy)

        runtime = _build_runtime(detector=detector)
        task = TaskSpec(content="Test", complexity_hint="simple")
        result = await runtime.coordinate(task, ["researcher"])

        assert result.success is False
        assert len(result.anomalies) > 0


# ---------------------------------------------------------------------------
# 5. Anomalies in CoordinationResult and summary
# ---------------------------------------------------------------------------


class TestAnomaliesInResult:
    @pytest.mark.asyncio
    async def test_anomalies_appear_in_summary(self) -> None:
        async def always_flag(result: CoordinationResult) -> AnomalyReport | None:
            return AnomalyReport(
                anomaly_type=AnomalyType.UNEXPECTED_TOOL_CHOICE,
                severity="low",
                task_id=result.task_id,
                description="Test anomaly",
                recommended_action="continue",
            )

        policy = DetectionPolicy()
        policy.custom_rules["flag"] = always_flag
        detector = FailureDetector(policy)

        runtime = _build_runtime(detector=detector)
        task = TaskSpec(content="Test", complexity_hint="simple")
        result = await runtime.coordinate(task, ["researcher"])

        assert len(result.anomalies) >= 1
        summary = runtime.get_coordination_summary(result)
        assert "Anomalies:" in summary


# ---------------------------------------------------------------------------
# 6. Detector is optional — backward compatible
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    @pytest.mark.asyncio
    async def test_no_detector_works(self) -> None:
        runtime = _build_runtime(detector=None)
        task = TaskSpec(content="Test", complexity_hint="simple")
        result = await runtime.coordinate(task, ["researcher", "writer"])
        assert result.success is True
        assert result.anomalies == []

    @pytest.mark.asyncio
    async def test_runtime_init_without_detector(self) -> None:
        router = Router()
        for m in _models():
            router.register_model(m)
        state = StateManager()
        runtime = KortexRuntime(router=router, state_manager=state)
        assert runtime._detector is None
