"""Recovery executor for Kortex runtime.

When the failure detector identifies an anomaly, the recovery executor
determines and carries out the appropriate response: retry, fallback,
rollback, or escalation.

This module exists because detection without action is just logging.
Every recommended_action from the detector maps to real runtime behavior.
"""

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

import structlog

from kortex.core.types import (
    TaskSpec,
    RoutingDecision,
    ExecutionEvent,
)
from kortex.core.detector import AnomalyReport, AnomalyType
from kortex.core.exceptions import (
    KortexError,
    RouterError,
    RoutingFailedError,
    ProviderError,
    StateError,
)

logger = structlog.get_logger(component="recovery_executor")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class RecoveryAction(enum.Enum):
    """What actually happened in response to an anomaly."""

    NO_ACTION = "no_action"
    CONTINUED = "continued"
    RETRIED = "retried"
    FELL_BACK = "fell_back"
    ROLLED_BACK = "rolled_back"
    ESCALATED = "escalated"


@dataclass(frozen=True)
class RecoveryPolicy:
    """Controls what the recovery executor is allowed to do.

    Every field has a safe default. Disabling a recovery path
    causes the executor to escalate instead.
    """

    max_retries_per_step: int = 1
    max_total_retries: int = 3
    enable_fallback: bool = True
    enable_rollback: bool = True
    enable_escalation: bool = True


@dataclass
class RecoveryRecord:
    """What was detected, what was done about it, and whether it worked."""

    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    anomaly: AnomalyReport | None = None
    action_taken: RecoveryAction = RecoveryAction.NO_ACTION
    success: bool = False
    detail: str = ""
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for storage in CoordinationResult."""
        return {
            "record_id": self.record_id,
            "anomaly_type": (
                self.anomaly.anomaly_type.name if self.anomaly else None
            ),
            "anomaly_severity": (
                self.anomaly.severity if self.anomaly else None
            ),
            "recommended_action": (
                self.anomaly.recommended_action if self.anomaly else None
            ),
            "action_taken": self.action_taken.value,
            "success": self.success,
            "detail": self.detail,
            "timestamp": self.timestamp.isoformat(),
        }


# ---------------------------------------------------------------------------
# Recovery context — everything the executor needs to act
# ---------------------------------------------------------------------------

class RouterProtocol(Protocol):
    """Minimal interface the recovery executor needs from the router."""

    async def route(self, task: TaskSpec) -> RoutingDecision: ...


class StateManagerProtocol(Protocol):
    """Minimal interface the recovery executor needs from the state manager."""

    async def rollback(self, checkpoint_id: str) -> Any: ...


class ProviderRegistryProtocol(Protocol):
    """Minimal interface the recovery executor needs from providers."""

    def get_provider(self, name: str) -> Any: ...


@dataclass
class RecoveryContext:
    """Everything the executor needs to carry out a recovery action.

    This is built by the runtime and passed to the executor.
    The executor never reaches into runtime internals.
    """

    task: TaskSpec
    current_step_index: int
    agent_id: str
    router: RouterProtocol
    state_manager: StateManagerProtocol
    registry: ProviderRegistryProtocol | None = None
    last_checkpoint_id: str | None = None
    execute_mode: bool = False
    current_routing_decision: RoutingDecision | None = None
    retry_count_this_step: int = 0
    total_retry_count: int = 0


# ---------------------------------------------------------------------------
# The executor
# ---------------------------------------------------------------------------

class RecoveryExecutor:
    """Translates anomaly recommendations into real runtime actions.

    Each recommended_action maps to exactly one code path.
    If a recovery path is disabled by policy, the executor escalates.
    If a recovery attempt fails, the executor escalates.
    There is no silent swallowing of failures.
    """

    def __init__(self, policy: RecoveryPolicy | None = None) -> None:
        self._policy = policy or RecoveryPolicy()
        self._total_retries_used = 0

    @property
    def policy(self) -> RecoveryPolicy:
        return self._policy

    @property
    def total_retries_used(self) -> int:
        return self._total_retries_used

    def reset(self) -> None:
        """Reset retry counters. Call at the start of each coordination."""
        self._total_retries_used = 0

    async def execute(
        self,
        anomaly: AnomalyReport,
        context: RecoveryContext,
    ) -> RecoveryRecord:
        """Execute the recovery action recommended by the detector.

        Returns a RecoveryRecord describing what was done and whether
        it succeeded. Never raises — all failures are captured in the record.
        """
        action = anomaly.recommended_action

        logger.info(
            "recovery_start",
            anomaly_type=anomaly.anomaly_type.value,
            severity=anomaly.severity,
            recommended_action=action,
            agent_id=context.agent_id,
            step=context.current_step_index,
        )

        if action == "continue":
            return self._handle_continue(anomaly, context)

        if action == "retry":
            return await self._handle_retry(anomaly, context)

        if action == "fallback":
            return await self._handle_fallback(anomaly, context)

        if action == "rollback":
            return await self._handle_rollback(anomaly, context)

        if action == "escalate":
            return self._handle_escalate(anomaly, context)

        # Unknown action — treat as escalation
        logger.warning(
            "recovery_unknown_action",
            action=action,
            anomaly_type=anomaly.anomaly_type.value,
        )
        return self._handle_escalate(anomaly, context)

    # -------------------------------------------------------------------
    # Action handlers
    # -------------------------------------------------------------------

    def _handle_continue(
        self,
        anomaly: AnomalyReport,
        context: RecoveryContext,
    ) -> RecoveryRecord:
        """Log the anomaly and proceed. No corrective action."""
        logger.info(
            "recovery_continue",
            anomaly_type=anomaly.anomaly_type.value,
            agent_id=context.agent_id,
        )
        return RecoveryRecord(
            anomaly=anomaly,
            action_taken=RecoveryAction.CONTINUED,
            success=True,
            detail=(
                f"Anomaly {anomaly.anomaly_type.value} acknowledged "
                f"on step {context.current_step_index}, proceeding."
            ),
        )

    async def _handle_retry(
        self,
        anomaly: AnomalyReport,
        context: RecoveryContext,
    ) -> RecoveryRecord:
        """Re-route and optionally re-execute the current step."""
        # Check retry budget
        if context.retry_count_this_step >= self._policy.max_retries_per_step:
            logger.warning(
                "recovery_retry_budget_step_exhausted",
                step=context.current_step_index,
                retries_this_step=context.retry_count_this_step,
                max_per_step=self._policy.max_retries_per_step,
            )
            return self._handle_escalate(
                anomaly,
                context,
                reason="step retry budget exhausted",
            )

        if self._total_retries_used >= self._policy.max_total_retries:
            logger.warning(
                "recovery_retry_budget_total_exhausted",
                total_used=self._total_retries_used,
                max_total=self._policy.max_total_retries,
            )
            return self._handle_escalate(
                anomaly,
                context,
                reason="total retry budget exhausted",
            )

        # Attempt re-route
        try:
            new_decision = await context.router.route(context.task)
        except (RouterError, RoutingFailedError) as e:
            logger.error(
                "recovery_retry_reroute_failed",
                error=str(e),
            )
            return self._handle_escalate(
                anomaly,
                context,
                reason=f"re-route failed: {e}",
            )

        self._total_retries_used += 1

        # If in execute mode and we have a registry, re-execute
        new_response = None
        if context.execute_mode and context.registry is not None:
            try:
                provider = context.registry.get_provider(
                    new_decision.chosen_provider
                )
                new_response = await provider.complete(
                    prompt=context.task.content,
                    model=new_decision.chosen_model,
                )
            except Exception as e:
                logger.error(
                    "recovery_retry_execute_failed",
                    error=str(e),
                )
                return self._handle_escalate(
                    anomaly,
                    context,
                    reason=f"retry execution failed: {e}",
                )

        logger.info(
            "recovery_retried",
            new_model=new_decision.chosen_model,
            new_provider=new_decision.chosen_provider,
            step=context.current_step_index,
        )

        return RecoveryRecord(
            anomaly=anomaly,
            action_taken=RecoveryAction.RETRIED,
            success=True,
            detail=(
                f"Retried step {context.current_step_index} with "
                f"{new_decision.chosen_provider}/{new_decision.chosen_model}"
                f"{', execution succeeded' if new_response else ', dry-run'}"
            ),
        )

    async def _handle_fallback(
        self,
        anomaly: AnomalyReport,
        context: RecoveryContext,
    ) -> RecoveryRecord:
        """Execute with the fallback model from the routing decision."""
        if not self._policy.enable_fallback:
            return self._handle_escalate(
                anomaly,
                context,
                reason="fallback disabled by policy",
            )

        decision = context.current_routing_decision
        if decision is None or not decision.fallback_model:
            return self._handle_escalate(
                anomaly,
                context,
                reason="no fallback model available",
            )

        fallback_model = decision.fallback_model
        fallback_provider = decision.chosen_provider  # same provider by default

        # If in execute mode, actually call the fallback
        new_response = None
        if context.execute_mode and context.registry is not None:
            try:
                provider = context.registry.get_provider(fallback_provider)
                new_response = await provider.complete(
                    prompt=context.task.content,
                    model=fallback_model,
                )
            except Exception as e:
                logger.error(
                    "recovery_fallback_execute_failed",
                    error=str(e),
                    fallback_model=fallback_model,
                )
                return self._handle_escalate(
                    anomaly,
                    context,
                    reason=f"fallback execution failed: {e}",
                )

        logger.info(
            "recovery_fell_back",
            fallback_model=fallback_model,
            step=context.current_step_index,
        )

        return RecoveryRecord(
            anomaly=anomaly,
            action_taken=RecoveryAction.FELL_BACK,
            success=True,
            detail=(
                f"Fell back to {fallback_provider}/{fallback_model} "
                f"on step {context.current_step_index}"
                f"{', execution succeeded' if new_response else ', dry-run'}"
            ),
        )

    async def _handle_rollback(
        self,
        anomaly: AnomalyReport,
        context: RecoveryContext,
    ) -> RecoveryRecord:
        """Roll back to the previous checkpoint and skip current agent."""
        if not self._policy.enable_rollback:
            return self._handle_escalate(
                anomaly,
                context,
                reason="rollback disabled by policy",
            )

        if context.last_checkpoint_id is None:
            return self._handle_escalate(
                anomaly,
                context,
                reason="no checkpoint available for rollback",
            )

        try:
            await context.state_manager.rollback(context.last_checkpoint_id)
        except (StateError, Exception) as e:
            logger.error(
                "recovery_rollback_failed",
                error=str(e),
                checkpoint_id=context.last_checkpoint_id,
            )
            return self._handle_escalate(
                anomaly,
                context,
                reason=f"rollback failed: {e}",
            )

        logger.info(
            "recovery_rolled_back",
            checkpoint_id=context.last_checkpoint_id,
            skipped_agent=context.agent_id,
            step=context.current_step_index,
        )

        return RecoveryRecord(
            anomaly=anomaly,
            action_taken=RecoveryAction.ROLLED_BACK,
            success=True,
            detail=(
                f"Rolled back to checkpoint {context.last_checkpoint_id[:12]}... "
                f"and skipped agent {context.agent_id}"
            ),
        )

    def _handle_escalate(
        self,
        anomaly: AnomalyReport,
        context: RecoveryContext,
        reason: str = "",
    ) -> RecoveryRecord:
        """Mark the coordination as failed. Terminal action."""
        if not self._policy.enable_escalation:
            # Even escalation is disabled — just log and continue
            logger.warning(
                "recovery_escalation_disabled",
                anomaly_type=anomaly.anomaly_type.value,
                reason=reason,
            )
            return RecoveryRecord(
                anomaly=anomaly,
                action_taken=RecoveryAction.CONTINUED,
                success=True,
                detail=(
                    f"Escalation disabled by policy. "
                    f"Anomaly {anomaly.anomaly_type.value} on step "
                    f"{context.current_step_index}: {reason}"
                ),
            )

        detail = (
            f"Escalated: {anomaly.anomaly_type.value} "
            f"({anomaly.severity}) on agent {context.agent_id}"
        )
        if reason:
            detail += f". Reason: {reason}"

        logger.error(
            "recovery_escalated",
            anomaly_type=anomaly.anomaly_type.value,
            severity=anomaly.severity,
            agent_id=context.agent_id,
            step=context.current_step_index,
            reason=reason,
        )

        return RecoveryRecord(
            anomaly=anomaly,
            action_taken=RecoveryAction.ESCALATED,
            success=False,
            detail=detail,
        )


# ---------------------------------------------------------------------------
# Event helpers
# ---------------------------------------------------------------------------

def recovery_event(
    record: RecoveryRecord,
    task_id: str,
    agent_id: str | None = None,
) -> ExecutionEvent:
    """Create an ExecutionEvent for a recovery action.

    Recovery events are DISTINCT from failure events.
    A failure event says "something went wrong."
    A recovery event says "here is what was done about it."
    """
    event_type_map = {
        RecoveryAction.CONTINUED: "recovery_continue",
        RecoveryAction.RETRIED: "recovery_retry",
        RecoveryAction.FELL_BACK: "recovery_fallback",
        RecoveryAction.ROLLED_BACK: "recovery_rollback",
        RecoveryAction.ESCALATED: "recovery_escalate",
        RecoveryAction.NO_ACTION: "recovery_none",
    }

    return ExecutionEvent(
        event_type=event_type_map.get(
            record.action_taken, "recovery_unknown"
        ),
        task_id=task_id,
        agent_id=agent_id,
        payload=record.to_dict(),
    )
