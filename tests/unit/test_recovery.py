"""Unit tests for the recovery executor.

These tests verify that every recommended_action from the detector
maps to real, verifiable runtime behavior. No silent swallowing.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from kortex.core.detector import AnomalyReport, AnomalyType
from kortex.core.exceptions import RoutingFailedError, StateError
from kortex.core.types import TaskSpec, RoutingDecision, ExecutionEvent
from kortex.core.recovery import (
    RecoveryAction,
    RecoveryPolicy,
    RecoveryRecord,
    RecoveryContext,
    RecoveryExecutor,
    recovery_event,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_anomaly(
    anomaly_type: AnomalyType = AnomalyType.COST_OVERRUN,
    severity: str = "medium",
    recommended_action: str = "continue",
    task_id: str = "task-001",
    agent_id: str = "agent-A",
) -> AnomalyReport:
    return AnomalyReport(
        anomaly_type=anomaly_type,
        severity=severity,
        task_id=task_id,
        agent_id=agent_id,
        description=f"Test anomaly: {anomaly_type.value}",
        evidence={"test": True},
        recommended_action=recommended_action,
    )


def make_routing_decision(
    model: str = "test-model",
    provider: str = "test-provider",
    fallback: str | None = "fallback-model",
) -> RoutingDecision:
    return RoutingDecision(
        task_id="task-001",
        chosen_provider=provider,
        chosen_model=model,
        reasoning="test routing",
        estimated_cost_usd=0.01,
        estimated_latency_ms=100.0,
        fallback_model=fallback,
    )


def make_context(
    execute_mode: bool = False,
    retry_count: int = 0,
    total_retries: int = 0,
    has_registry: bool = True,
    has_checkpoint: bool = True,
    routing_decision: RoutingDecision | None = None,
) -> RecoveryContext:
    router = AsyncMock()
    router.route = AsyncMock(return_value=make_routing_decision(
        model="retry-model", provider="retry-provider"
    ))

    state_manager = AsyncMock()
    state_manager.rollback = AsyncMock(return_value=MagicMock())

    registry = None
    if has_registry:
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(return_value=MagicMock(
            content="recovery response",
            model="test-model",
            provider="test-provider",
            input_tokens=10,
            output_tokens=20,
            cost_usd=0.005,
            latency_ms=50.0,
            raw_response={},
        ))
        registry = MagicMock()
        registry.get_provider = MagicMock(return_value=mock_provider)

    return RecoveryContext(
        task=TaskSpec(content="test task", complexity_hint="moderate"),
        current_step_index=1,
        agent_id="agent-A",
        router=router,
        state_manager=state_manager,
        registry=registry,
        last_checkpoint_id="ckpt-001" if has_checkpoint else None,
        execute_mode=execute_mode,
        current_routing_decision=routing_decision or make_routing_decision(),
        retry_count_this_step=retry_count,
        total_retry_count=total_retries,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestContinueAction:
    """'continue' should log and proceed — no retry, no fallback."""

    @pytest.mark.asyncio
    async def test_continue_returns_continued(self):
        executor = RecoveryExecutor()
        anomaly = make_anomaly(recommended_action="continue")
        context = make_context()

        record = await executor.execute(anomaly, context)

        assert record.action_taken == RecoveryAction.CONTINUED
        assert record.success is True
        assert record.anomaly is anomaly

    @pytest.mark.asyncio
    async def test_continue_does_not_call_router_or_provider(self):
        executor = RecoveryExecutor()
        anomaly = make_anomaly(recommended_action="continue")
        context = make_context()

        await executor.execute(anomaly, context)

        context.router.route.assert_not_called()
        if context.registry:
            context.registry.get_provider.assert_not_called()


class TestRetryAction:
    """'retry' should re-route and re-execute, respecting budgets."""

    @pytest.mark.asyncio
    async def test_retry_reroutes_task(self):
        executor = RecoveryExecutor()
        anomaly = make_anomaly(recommended_action="retry")
        context = make_context()

        record = await executor.execute(anomaly, context)

        assert record.action_taken == RecoveryAction.RETRIED
        assert record.success is True
        context.router.route.assert_called_once_with(context.task)

    @pytest.mark.asyncio
    async def test_retry_executes_in_execute_mode(self):
        executor = RecoveryExecutor()
        anomaly = make_anomaly(recommended_action="retry")
        context = make_context(execute_mode=True)

        record = await executor.execute(anomaly, context)

        assert record.action_taken == RecoveryAction.RETRIED
        assert record.success is True
        context.registry.get_provider.assert_called_once()

    @pytest.mark.asyncio
    async def test_retry_exhausted_step_budget_escalates(self):
        policy = RecoveryPolicy(max_retries_per_step=1)
        executor = RecoveryExecutor(policy)
        anomaly = make_anomaly(recommended_action="retry")
        context = make_context(retry_count=1)  # Already used 1, max is 1

        record = await executor.execute(anomaly, context)

        assert record.action_taken == RecoveryAction.ESCALATED
        assert record.success is False
        assert "budget" in record.detail.lower() or "exhausted" in record.detail.lower()

    @pytest.mark.asyncio
    async def test_retry_exhausted_total_budget_escalates(self):
        policy = RecoveryPolicy(max_total_retries=2)
        executor = RecoveryExecutor(policy)
        # Use up the budget
        executor._total_retries_used = 2

        anomaly = make_anomaly(recommended_action="retry")
        context = make_context(retry_count=0)

        record = await executor.execute(anomaly, context)

        assert record.action_taken == RecoveryAction.ESCALATED
        assert record.success is False

    @pytest.mark.asyncio
    async def test_retry_increments_total_counter(self):
        executor = RecoveryExecutor()
        anomaly = make_anomaly(recommended_action="retry")
        context = make_context()

        assert executor.total_retries_used == 0
        await executor.execute(anomaly, context)
        assert executor.total_retries_used == 1

    @pytest.mark.asyncio
    async def test_retry_reroute_failure_escalates(self):
        executor = RecoveryExecutor()
        anomaly = make_anomaly(recommended_action="retry")
        context = make_context()
        context.router.route = AsyncMock(
            side_effect=RoutingFailedError("no models available")
        )

        record = await executor.execute(anomaly, context)

        assert record.action_taken == RecoveryAction.ESCALATED
        assert record.success is False
        assert "re-route failed" in record.detail.lower() or "reroute" in record.detail.lower() or "route" in record.detail.lower()


class TestFallbackAction:
    """'fallback' should use fallback_model from routing decision."""

    @pytest.mark.asyncio
    async def test_fallback_uses_fallback_model(self):
        executor = RecoveryExecutor()
        anomaly = make_anomaly(recommended_action="fallback")
        decision = make_routing_decision(fallback="gpt-4o-mini")
        context = make_context(routing_decision=decision, execute_mode=True)

        record = await executor.execute(anomaly, context)

        assert record.action_taken == RecoveryAction.FELL_BACK
        assert record.success is True
        assert "gpt-4o-mini" in record.detail

    @pytest.mark.asyncio
    async def test_fallback_no_fallback_model_escalates(self):
        executor = RecoveryExecutor()
        anomaly = make_anomaly(recommended_action="fallback")
        decision = make_routing_decision(fallback=None)
        context = make_context(routing_decision=decision)

        record = await executor.execute(anomaly, context)

        assert record.action_taken == RecoveryAction.ESCALATED
        assert record.success is False
        assert "no fallback" in record.detail.lower()

    @pytest.mark.asyncio
    async def test_fallback_disabled_by_policy_escalates(self):
        policy = RecoveryPolicy(enable_fallback=False)
        executor = RecoveryExecutor(policy)
        anomaly = make_anomaly(recommended_action="fallback")
        context = make_context()

        record = await executor.execute(anomaly, context)

        assert record.action_taken == RecoveryAction.ESCALATED
        assert record.success is False
        assert "disabled" in record.detail.lower()

    @pytest.mark.asyncio
    async def test_fallback_execution_failure_escalates(self):
        executor = RecoveryExecutor()
        anomaly = make_anomaly(recommended_action="fallback")
        context = make_context(execute_mode=True)
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(
            side_effect=ConnectionError("provider down")
        )
        context.registry.get_provider = MagicMock(return_value=mock_provider)

        record = await executor.execute(anomaly, context)

        assert record.action_taken == RecoveryAction.ESCALATED
        assert record.success is False


class TestRollbackAction:
    """'rollback' should roll back to previous checkpoint and skip agent."""

    @pytest.mark.asyncio
    async def test_rollback_calls_state_manager(self):
        executor = RecoveryExecutor()
        anomaly = make_anomaly(recommended_action="rollback")
        context = make_context(has_checkpoint=True)

        record = await executor.execute(anomaly, context)

        assert record.action_taken == RecoveryAction.ROLLED_BACK
        assert record.success is True
        context.state_manager.rollback.assert_called_once_with("ckpt-001")

    @pytest.mark.asyncio
    async def test_rollback_no_checkpoint_escalates(self):
        executor = RecoveryExecutor()
        anomaly = make_anomaly(recommended_action="rollback")
        context = make_context(has_checkpoint=False)

        record = await executor.execute(anomaly, context)

        assert record.action_taken == RecoveryAction.ESCALATED
        assert record.success is False
        assert "no checkpoint" in record.detail.lower()

    @pytest.mark.asyncio
    async def test_rollback_disabled_by_policy_escalates(self):
        policy = RecoveryPolicy(enable_rollback=False)
        executor = RecoveryExecutor(policy)
        anomaly = make_anomaly(recommended_action="rollback")
        context = make_context(has_checkpoint=True)

        record = await executor.execute(anomaly, context)

        assert record.action_taken == RecoveryAction.ESCALATED
        assert record.success is False

    @pytest.mark.asyncio
    async def test_rollback_failure_escalates(self):
        executor = RecoveryExecutor()
        anomaly = make_anomaly(recommended_action="rollback")
        context = make_context(has_checkpoint=True)
        context.state_manager.rollback = AsyncMock(
            side_effect=StateError("store corrupted")
        )

        record = await executor.execute(anomaly, context)

        assert record.action_taken == RecoveryAction.ESCALATED
        assert record.success is False


class TestEscalateAction:
    """'escalate' should mark coordination as failed."""

    @pytest.mark.asyncio
    async def test_escalate_marks_failure(self):
        executor = RecoveryExecutor()
        anomaly = make_anomaly(recommended_action="escalate")
        context = make_context()

        record = await executor.execute(anomaly, context)

        assert record.action_taken == RecoveryAction.ESCALATED
        assert record.success is False

    @pytest.mark.asyncio
    async def test_escalate_disabled_continues_instead(self):
        policy = RecoveryPolicy(enable_escalation=False)
        executor = RecoveryExecutor(policy)
        anomaly = make_anomaly(recommended_action="escalate")
        context = make_context()

        record = await executor.execute(anomaly, context)

        assert record.action_taken == RecoveryAction.CONTINUED
        assert record.success is True


class TestPolicyControl:
    """RecoveryPolicy controls what the executor is allowed to do."""

    @pytest.mark.asyncio
    async def test_zero_retries_disables_retry(self):
        policy = RecoveryPolicy(max_retries_per_step=0)
        executor = RecoveryExecutor(policy)
        anomaly = make_anomaly(recommended_action="retry")
        context = make_context(retry_count=0)

        record = await executor.execute(anomaly, context)

        # With max_retries_per_step=0, even the first retry should escalate
        assert record.action_taken == RecoveryAction.ESCALATED

    @pytest.mark.asyncio
    async def test_default_policy_allows_everything(self):
        executor = RecoveryExecutor()  # default policy
        policy = executor.policy

        assert policy.max_retries_per_step == 1
        assert policy.max_total_retries == 3
        assert policy.enable_fallback is True
        assert policy.enable_rollback is True
        assert policy.enable_escalation is True

    @pytest.mark.asyncio
    async def test_reset_clears_retry_counter(self):
        executor = RecoveryExecutor()
        anomaly = make_anomaly(recommended_action="retry")
        context = make_context()

        await executor.execute(anomaly, context)
        assert executor.total_retries_used == 1

        executor.reset()
        assert executor.total_retries_used == 0


class TestRecoveryEvents:
    """Recovery events must be distinct from failure events."""

    @pytest.mark.asyncio
    async def test_recovery_event_types_are_distinct(self):
        records = []
        executor = RecoveryExecutor()

        for action_str in ["continue", "retry", "fallback", "rollback", "escalate"]:
            anomaly = make_anomaly(recommended_action=action_str)
            ctx = make_context(has_checkpoint=True)
            record = await executor.execute(anomaly, ctx)
            records.append(record)
            executor.reset()

        events = [
            recovery_event(r, task_id="task-001", agent_id="agent-A")
            for r in records
        ]

        event_types = [e.event_type for e in events]

        # All recovery events should start with "recovery_"
        for et in event_types:
            assert et.startswith("recovery_"), f"Event type {et} doesn't start with recovery_"

        # None should be "failure"
        for et in event_types:
            assert et != "failure", "Recovery events must not be typed as 'failure'"

    @pytest.mark.asyncio
    async def test_recovery_event_payload_contains_record_data(self):
        executor = RecoveryExecutor()
        anomaly = make_anomaly(recommended_action="continue")
        context = make_context()

        record = await executor.execute(anomaly, context)
        event = recovery_event(record, task_id="task-001")

        assert event.payload["action_taken"] == "continued"
        assert event.payload["success"] is True
        assert event.payload["anomaly_type"] == "COST_OVERRUN"


class TestMultipleAnomalies:
    """Each anomaly gets its own RecoveryRecord."""

    @pytest.mark.asyncio
    async def test_multiple_anomalies_each_get_record(self):
        executor = RecoveryExecutor(RecoveryPolicy(max_total_retries=10))
        anomalies = [
            make_anomaly(
                anomaly_type=AnomalyType.COST_OVERRUN,
                recommended_action="continue",
            ),
            make_anomaly(
                anomaly_type=AnomalyType.LATENCY_SPIKE,
                recommended_action="retry",
            ),
            make_anomaly(
                anomaly_type=AnomalyType.OUTPUT_QUALITY_DROP,
                recommended_action="fallback",
            ),
        ]

        records = []
        for anomaly in anomalies:
            ctx = make_context()
            record = await executor.execute(anomaly, ctx)
            records.append(record)

        assert len(records) == 3
        assert records[0].action_taken == RecoveryAction.CONTINUED
        assert records[1].action_taken == RecoveryAction.RETRIED
        assert records[2].action_taken == RecoveryAction.FELL_BACK

        # Each record references its own anomaly
        assert records[0].anomaly.anomaly_type == AnomalyType.COST_OVERRUN
        assert records[1].anomaly.anomaly_type == AnomalyType.LATENCY_SPIKE
        assert records[2].anomaly.anomaly_type == AnomalyType.OUTPUT_QUALITY_DROP


class TestRecoveryRecordSerialization:
    """RecoveryRecord.to_dict() must be clean and complete."""

    @pytest.mark.asyncio
    async def test_to_dict_structure(self):
        executor = RecoveryExecutor()
        anomaly = make_anomaly(recommended_action="continue")
        context = make_context()

        record = await executor.execute(anomaly, context)
        d = record.to_dict()

        assert "record_id" in d
        assert "anomaly_type" in d
        assert "anomaly_severity" in d
        assert "recommended_action" in d
        assert "action_taken" in d
        assert "success" in d
        assert "detail" in d
        assert "timestamp" in d
        assert d["action_taken"] == "continued"
        assert d["success"] is True
