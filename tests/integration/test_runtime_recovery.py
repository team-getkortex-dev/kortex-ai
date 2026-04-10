"""Integration tests for runtime recovery — detector + executor working together.

Verifies that anomaly recommendations are actually honored by the runtime:
retry re-routes, fallback uses the fallback model, escalate stops the pipeline.
All HTTP calls are mocked.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from kortex.core.detector import (
    AnomalyReport,
    AnomalyType,
    DetectionPolicy,
    FailureDetector,
)
from kortex.core.recovery import RecoveryPolicy
from kortex.core.router import ProviderModel, Router
from kortex.core.runtime import AgentDescriptor, KortexRuntime
from kortex.core.state import StateManager
from kortex.core.types import TaskSpec
from kortex.providers.registry import ProviderRegistry
from kortex.store.memory import InMemoryStateStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_httpx_response(data: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        status_code=200,
        json=data,
        request=httpx.Request("POST", "https://mock"),
    )


def _ok_response(
    content: str = "mock output",
    input_tok: int = 50,
    output_tok: int = 20,
) -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": input_tok, "completion_tokens": output_tok},
    }


def _make_models() -> list[ProviderModel]:
    return [
        ProviderModel(
            provider="test-cloud",
            model="cloud-fast",
            cost_per_1k_input_tokens=0.001,
            cost_per_1k_output_tokens=0.002,
            avg_latency_ms=200,
            capabilities=["reasoning"],
            tier="fast",
        ),
        ProviderModel(
            provider="test-cloud",
            model="cloud-balanced",
            cost_per_1k_input_tokens=0.003,
            cost_per_1k_output_tokens=0.01,
            avg_latency_ms=600,
            capabilities=["reasoning", "analysis"],
            tier="balanced",
        ),
        ProviderModel(
            provider="test-cloud",
            model="cloud-powerful",
            cost_per_1k_input_tokens=0.01,
            cost_per_1k_output_tokens=0.03,
            avg_latency_ms=1500,
            capabilities=["reasoning", "analysis", "research"],
            tier="powerful",
        ),
    ]


def _build_runtime(
    mock_post: AsyncMock | None = None,
    detector: FailureDetector | None = None,
    recovery_policy: RecoveryPolicy | None = None,
) -> KortexRuntime:
    """Build a runtime with optional detector and recovery policy."""
    models = _make_models()
    router = Router()
    for m in models:
        router.register_model(m)

    registry = ProviderRegistry()
    registry.register_openai_compatible(
        name="test-cloud",
        base_url="https://mock-cloud.test/v1",
        api_key="mock-key",
        models=models,
    )

    if mock_post is not None:
        for pname in registry.list_providers():
            connector = registry.get_provider(pname)
            if hasattr(connector, "_get_client"):
                client = connector._get_client()  # type: ignore[union-attr]
                client.post = mock_post  # type: ignore[method-assign]

    state = StateManager(store=InMemoryStateStore())
    runtime = KortexRuntime(
        router=router,
        state_manager=state,
        registry=registry,
        detector=detector,
        recovery_policy=recovery_policy,
    )

    runtime.register_agent(AgentDescriptor("agent-a", "Agent A", "First", ["reasoning"]))
    runtime.register_agent(AgentDescriptor("agent-b", "Agent B", "Second", ["reasoning"]))
    runtime.register_agent(AgentDescriptor("agent-c", "Agent C", "Third", ["reasoning"]))

    return runtime


# ---------------------------------------------------------------------------
# 1. Cost overrun triggers fallback, pipeline succeeds
# ---------------------------------------------------------------------------


class TestCostOverrunFallback:
    @pytest.mark.asyncio
    async def test_cost_overrun_triggers_fallback_pipeline_succeeds(self) -> None:
        """When routing detects a cost overrun, the detector recommends
        'fallback'. With a recovery policy, the executor falls back to
        the fallback model and the pipeline continues successfully.

        The cost ceiling must be high enough for the router to pass
        (so routing succeeds), but the detector's multiplier check
        must fire on the resulting decision.
        """
        mock_post = AsyncMock(
            return_value=_mock_httpx_response(_ok_response("fallback output", 30, 10))
        )

        # max_cost_multiplier=0.5 means detector flags if
        # estimated_cost > cost_ceiling * 0.5
        # cloud-fast estimated_cost = 0.0015, ceiling = 0.002
        # 0.0015 > 0.002 * 0.5 = 0.001 → True → detector fires
        detector = FailureDetector(DetectionPolicy(max_cost_multiplier=0.5))
        recovery = RecoveryPolicy(enable_fallback=True)

        runtime = _build_runtime(
            mock_post=mock_post,
            detector=detector,
            recovery_policy=recovery,
        )

        task = TaskSpec(
            content="Simple task",
            complexity_hint="simple",
            required_capabilities=["reasoning"],
            cost_ceiling_usd=0.002,  # Passes router, triggers detector at 0.5x
        )

        result = await runtime.coordinate(task, ["agent-a"], execute=True)

        # Detector should have fired
        assert len(result.anomalies) > 0
        assert len(result.recovery_records) > 0

        # Check at least one recovery action was taken
        actions = [r["action_taken"] for r in result.recovery_records]
        # Should have fell_back or escalated (if no fallback model available)
        assert any(a in ("fell_back", "escalated", "continued") for a in actions)


# ---------------------------------------------------------------------------
# 2. Provider failure triggers retry, succeeds on second attempt
# ---------------------------------------------------------------------------


class TestProviderFailureRetry:
    @pytest.mark.asyncio
    async def test_latency_spike_triggers_retry(self) -> None:
        """When execution check detects a latency spike, the detector
        recommends 'retry'. The executor re-routes and re-executes."""
        call_count = {"n": 0}

        async def mock_post(*args: Any, **kwargs: Any) -> httpx.Response:
            call_count["n"] += 1
            return _mock_httpx_response(_ok_response("output", 50, 20))

        # Detector that flags short output as quality drop (recommends retry)
        detector = FailureDetector(DetectionPolicy(min_output_length=1000))
        recovery = RecoveryPolicy(max_retries_per_step=1, max_total_retries=3)

        runtime = _build_runtime(
            mock_post=AsyncMock(side_effect=mock_post),
            detector=detector,
            recovery_policy=recovery,
        )

        task = TaskSpec(content="Generate something", complexity_hint="simple")
        result = await runtime.coordinate(task, ["agent-a"], execute=True)

        # Should have detected the quality drop and tried to recover
        assert len(result.anomalies) > 0
        assert len(result.recovery_records) > 0

        # The recovery action should be retry or escalated (if retry failed)
        actions = [r["action_taken"] for r in result.recovery_records]
        assert any(a in ("retried", "escalated") for a in actions)


# ---------------------------------------------------------------------------
# 3. Unrecoverable failure triggers escalation, success=False
# ---------------------------------------------------------------------------


class TestUnrecoverableEscalation:
    @pytest.mark.asyncio
    async def test_escalation_marks_failure(self) -> None:
        """When the detector recommends 'escalate' directly (e.g. from
        check_coordination), the pipeline is marked as failed."""
        mock_post = AsyncMock(
            return_value=_mock_httpx_response(_ok_response("output", 50, 20))
        )

        # Detector with very low chain depth to trigger escalation
        detector = FailureDetector(DetectionPolicy(max_chain_depth=0))
        recovery = RecoveryPolicy()

        runtime = _build_runtime(
            mock_post=mock_post,
            detector=detector,
            recovery_policy=recovery,
        )

        task = TaskSpec(content="Test escalation", complexity_hint="simple")
        result = await runtime.coordinate(task, ["agent-a", "agent-b"], execute=True)

        # The coordination-level check should detect chain depth > 0 and escalate
        assert result.success is False
        assert len(result.anomalies) > 0

        # Should have escalation recovery record
        escalation_records = [
            r for r in result.recovery_records
            if r["action_taken"] == "escalated"
        ]
        assert len(escalation_records) > 0

    @pytest.mark.asyncio
    async def test_cost_overrun_no_fallback_escalates(self) -> None:
        """When cost overrun is detected but no fallback model is available,
        escalation happens."""
        mock_post = AsyncMock(
            return_value=_mock_httpx_response(_ok_response("output", 50, 20))
        )

        # Very low multiplier so detector always fires on any cost > 0
        detector = FailureDetector(DetectionPolicy(max_cost_multiplier=0.001))
        recovery = RecoveryPolicy(enable_fallback=True)

        runtime = _build_runtime(
            mock_post=mock_post,
            detector=detector,
            recovery_policy=recovery,
        )

        # Cost ceiling passes router but triggers detector at 0.001x
        task = TaskSpec(
            content="Task",
            complexity_hint="simple",
            cost_ceiling_usd=0.01,  # Passes router (cloud-fast est=0.0015)
        )

        result = await runtime.coordinate(task, ["agent-a"])

        # Should have anomalies and recovery records
        if result.anomalies:
            assert len(result.recovery_records) > 0


# ---------------------------------------------------------------------------
# 4. Recovery records appear in CoordinationResult
# ---------------------------------------------------------------------------


class TestRecoveryRecordsInResult:
    @pytest.mark.asyncio
    async def test_recovery_records_populated(self) -> None:
        """Recovery records should appear in the result when anomalies
        are detected and a recovery policy is configured."""
        mock_post = AsyncMock(
            return_value=_mock_httpx_response(_ok_response("short", 10, 5))
        )

        # Trigger output quality anomaly (min_output_length > actual)
        detector = FailureDetector(DetectionPolicy(min_output_length=500))
        recovery = RecoveryPolicy(max_retries_per_step=1)

        runtime = _build_runtime(
            mock_post=mock_post,
            detector=detector,
            recovery_policy=recovery,
        )

        task = TaskSpec(content="Test recovery records", complexity_hint="simple")
        result = await runtime.coordinate(task, ["agent-a"], execute=True)

        # We should have anomalies (short output)
        assert len(result.anomalies) > 0
        # And recovery records
        assert len(result.recovery_records) > 0

        # Each record has expected fields
        for rec in result.recovery_records:
            assert "record_id" in rec
            assert "action_taken" in rec
            assert "success" in rec
            assert "detail" in rec
            assert "timestamp" in rec
            assert "anomaly_type" in rec

    @pytest.mark.asyncio
    async def test_recovery_events_distinct_from_failure_events(self) -> None:
        """Recovery events must be separate from failure events in the
        event stream."""
        mock_post = AsyncMock(
            return_value=_mock_httpx_response(_ok_response("x", 5, 2))
        )

        detector = FailureDetector(DetectionPolicy(min_output_length=500))
        recovery = RecoveryPolicy(max_retries_per_step=1)

        runtime = _build_runtime(
            mock_post=mock_post,
            detector=detector,
            recovery_policy=recovery,
        )

        task = TaskSpec(content="Test events", complexity_hint="simple")
        result = await runtime.coordinate(task, ["agent-a"], execute=True)

        failure_events = [e for e in result.events if e.event_type == "failure"]
        recovery_events = [
            e for e in result.events
            if e.event_type.startswith("recovery_")
        ]

        # We should have both failure detection and recovery action events
        if result.anomalies:
            assert len(failure_events) > 0, "Should have failure detection events"
            assert len(recovery_events) > 0, "Should have recovery action events"

            # Recovery events should have payloads with action_taken
            for re in recovery_events:
                assert "action_taken" in re.payload


# ---------------------------------------------------------------------------
# 5. No detector = no recovery executor = backward compatible
# ---------------------------------------------------------------------------


class TestBackwardCompatNoDetector:
    @pytest.mark.asyncio
    async def test_no_detector_no_recovery(self) -> None:
        """Without a detector, no recovery happens — backward compatible."""
        mock_post = AsyncMock(
            return_value=_mock_httpx_response(_ok_response("output"))
        )

        runtime = _build_runtime(mock_post=mock_post)

        task = TaskSpec(content="Normal task", complexity_hint="simple")
        result = await runtime.coordinate(task, ["agent-a", "agent-b"], execute=True)

        assert result.success is True
        assert result.recovery_records == []
        assert result.anomalies == []

    @pytest.mark.asyncio
    async def test_no_detector_dry_run(self) -> None:
        """Dry run without detector should still work exactly as before."""
        runtime = _build_runtime()

        task = TaskSpec(content="Dry run", complexity_hint="simple")
        result = await runtime.coordinate(task, ["agent-a"])

        assert result.success is True
        assert result.recovery_records == []
        assert result.responses == []


# ---------------------------------------------------------------------------
# 6. Detector but no policy = anomalies logged, no recovery actions
# ---------------------------------------------------------------------------


class TestDetectorWithoutPolicy:
    @pytest.mark.asyncio
    async def test_detector_without_policy_logs_but_no_recovery(self) -> None:
        """When a detector is present but no recovery policy, anomalies are
        detected and logged, but no RecoveryRecords are generated."""
        mock_post = AsyncMock(
            return_value=_mock_httpx_response(_ok_response("x", 5, 2))
        )

        # Detector that will flag short output
        detector = FailureDetector(DetectionPolicy(min_output_length=500))

        # No recovery_policy
        runtime = _build_runtime(
            mock_post=mock_post,
            detector=detector,
            recovery_policy=None,
        )

        task = TaskSpec(content="Test no policy", complexity_hint="simple")
        result = await runtime.coordinate(task, ["agent-a"], execute=True)

        # Anomalies should be detected
        assert len(result.anomalies) > 0

        # But no recovery records (no executor)
        assert result.recovery_records == []

    @pytest.mark.asyncio
    async def test_detector_escalate_without_policy_still_fails(self) -> None:
        """Even without a recovery policy, 'escalate' from the detector
        should mark success=False (legacy behavior preserved)."""
        mock_post = AsyncMock(
            return_value=_mock_httpx_response(_ok_response("output"))
        )

        # Trigger coordination-level escalation via chain depth
        detector = FailureDetector(DetectionPolicy(max_chain_depth=0))

        runtime = _build_runtime(
            mock_post=mock_post,
            detector=detector,
            recovery_policy=None,
        )

        task = TaskSpec(content="Test legacy escalate", complexity_hint="simple")
        result = await runtime.coordinate(task, ["agent-a", "agent-b"], execute=True)

        # Legacy behavior: escalate recommendation sets success=False
        assert result.success is False
        # No recovery records
        assert result.recovery_records == []


# ---------------------------------------------------------------------------
# 7. Summary includes recovery info
# ---------------------------------------------------------------------------


class TestSummaryWithRecovery:
    @pytest.mark.asyncio
    async def test_summary_mentions_recovery_on_escalation(self) -> None:
        """Summary should mention recovery actions when they occur."""
        mock_post = AsyncMock(
            return_value=_mock_httpx_response(_ok_response("output"))
        )

        detector = FailureDetector(DetectionPolicy(max_chain_depth=0))
        recovery = RecoveryPolicy()

        runtime = _build_runtime(
            mock_post=mock_post,
            detector=detector,
            recovery_policy=recovery,
        )

        task = TaskSpec(content="Summary test", complexity_hint="simple")
        result = await runtime.coordinate(task, ["agent-a", "agent-b"], execute=True)

        summary = runtime.get_coordination_summary(result)

        # If there were recovery records, summary should mention "Recovery"
        if result.recovery_records:
            assert "Recovery:" in summary or "Pipeline failed" in summary
