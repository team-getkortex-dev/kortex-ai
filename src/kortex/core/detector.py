"""Failure and anomaly detection for Kortex.

Monitors routing decisions, execution responses, handoffs, and full
coordination results for anomalies such as cost overruns, latency
spikes, output quality drops, and checkpoint chain breaks.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Literal
from uuid import uuid4

import structlog

from kortex.core.types import (
    CoordinationResult,
    ExecutionEvent,
    HandoffContext,
    RoutingDecision,
    TaskSpec,
)

logger = structlog.get_logger(component="detector")


class AnomalyType(enum.Enum):
    """Types of anomalies the detector can identify."""

    EXCESSIVE_RETRIES = "excessive_retries"
    CONTEXT_DEGRADATION = "context_degradation"
    COST_OVERRUN = "cost_overrun"
    LATENCY_SPIKE = "latency_spike"
    UNEXPECTED_TOOL_CHOICE = "unexpected_tool_choice"
    OUTPUT_QUALITY_DROP = "output_quality_drop"
    CHECKPOINT_CHAIN_BREAK = "checkpoint_chain_break"


@dataclass
class AnomalyReport:
    """A detected anomaly with evidence and recommended action.

    Args:
        anomaly_id: Unique identifier for this report.
        anomaly_type: The category of anomaly.
        severity: How serious this anomaly is.
        task_id: The task that triggered the anomaly.
        agent_id: The agent involved, if applicable.
        description: Human-readable description.
        evidence: Supporting data for the anomaly.
        recommended_action: What the runtime should do about it.
        detected_at: When the anomaly was detected.
    """

    anomaly_id: str = field(default_factory=lambda: str(uuid4()))
    anomaly_type: AnomalyType = AnomalyType.COST_OVERRUN
    severity: Literal["low", "medium", "high", "critical"] = "medium"
    task_id: str = ""
    agent_id: str | None = None
    description: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    recommended_action: Literal[
        "continue", "retry", "fallback", "rollback", "escalate"
    ] = "continue"
    detected_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for CoordinationResult.anomalies."""
        return {
            "anomaly_id": self.anomaly_id,
            "anomaly_type": self.anomaly_type.value,
            "severity": self.severity,
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "description": self.description,
            "evidence": self.evidence,
            "recommended_action": self.recommended_action,
            "detected_at": self.detected_at.isoformat(),
        }


# Type alias for custom rules
CustomRule = Callable[..., Coroutine[Any, Any, AnomalyReport | None]]


@dataclass
class DetectionPolicy:
    """Configurable thresholds for anomaly detection.

    Args:
        max_retries: Maximum retries before EXCESSIVE_RETRIES.
        max_cost_multiplier: Actual/estimated cost ratio triggering COST_OVERRUN.
        max_latency_multiplier: Actual/estimated latency ratio triggering LATENCY_SPIKE.
        min_output_length: Minimum response length before OUTPUT_QUALITY_DROP.
        max_chain_depth: Maximum checkpoint chain depth before CHECKPOINT_CHAIN_BREAK.
        custom_rules: Named custom detection rules.
    """

    max_retries: int = 3
    max_cost_multiplier: float = 2.0
    max_latency_multiplier: float = 3.0
    min_output_length: int = 10
    max_chain_depth: int = 50
    custom_rules: dict[str, CustomRule] = field(default_factory=dict)


class FailureDetector:
    """Monitors Kortex operations for anomalies.

    Args:
        policy: Detection thresholds. Uses defaults if not provided.
    """

    def __init__(self, policy: DetectionPolicy | None = None) -> None:
        self._policy = policy or DetectionPolicy()
        self._retry_counts: dict[str, int] = {}
        self._log = structlog.get_logger(component="detector")

    @property
    def policy(self) -> DetectionPolicy:
        """Return the current detection policy."""
        return self._policy

    def register_custom_rule(self, name: str, rule: CustomRule) -> None:
        """Register a custom detection rule.

        Args:
            name: Rule name for identification.
            rule: Async callable that returns AnomalyReport or None.
        """
        self._policy.custom_rules[name] = rule

    async def check_routing(
        self,
        decision: RoutingDecision,
        task: TaskSpec,
    ) -> AnomalyReport | None:
        """Check a routing decision for anomalies.

        Args:
            decision: The routing decision to check.
            task: The original task specification.

        Returns:
            An AnomalyReport if an anomaly is detected, None otherwise.
        """
        # Cost overrun check: if task had a cost ceiling and estimate exceeds it
        if (
            task.cost_ceiling_usd is not None
            and decision.estimated_cost_usd > task.cost_ceiling_usd * self._policy.max_cost_multiplier
        ):
            report = AnomalyReport(
                anomaly_type=AnomalyType.COST_OVERRUN,
                severity="high",
                task_id=task.task_id,
                description=(
                    f"Estimated cost ${decision.estimated_cost_usd:.4f} exceeds "
                    f"{self._policy.max_cost_multiplier}x cost ceiling "
                    f"${task.cost_ceiling_usd:.4f}"
                ),
                evidence={
                    "estimated_cost": decision.estimated_cost_usd,
                    "cost_ceiling": task.cost_ceiling_usd,
                    "multiplier": self._policy.max_cost_multiplier,
                },
                recommended_action="fallback",
            )
            self._log.warning("anomaly_detected", **report.to_dict())
            return report

        return None

    async def check_execution(
        self,
        response: dict[str, Any],
        decision: RoutingDecision,
    ) -> AnomalyReport | None:
        """Check an execution response for anomalies.

        Args:
            response: The provider response dict.
            decision: The routing decision that led to this execution.

        Returns:
            An AnomalyReport if an anomaly is detected, None otherwise.
        """
        # Cost overrun: actual cost vs estimated
        actual_cost = response.get("cost_usd", 0.0)
        if (
            decision.estimated_cost_usd > 0
            and actual_cost > decision.estimated_cost_usd * self._policy.max_cost_multiplier
        ):
            report = AnomalyReport(
                anomaly_type=AnomalyType.COST_OVERRUN,
                severity="high",
                task_id=decision.task_id,
                description=(
                    f"Actual cost ${actual_cost:.4f} exceeds "
                    f"{self._policy.max_cost_multiplier}x estimate "
                    f"${decision.estimated_cost_usd:.4f}"
                ),
                evidence={
                    "actual_cost": actual_cost,
                    "estimated_cost": decision.estimated_cost_usd,
                    "multiplier": actual_cost / decision.estimated_cost_usd,
                },
                recommended_action="fallback",
            )
            self._log.warning("anomaly_detected", **report.to_dict())
            return report

        # Latency spike: actual vs estimated
        actual_latency = response.get("latency_ms", 0.0)
        if (
            decision.estimated_latency_ms > 0
            and actual_latency > decision.estimated_latency_ms * self._policy.max_latency_multiplier
        ):
            report = AnomalyReport(
                anomaly_type=AnomalyType.LATENCY_SPIKE,
                severity="medium",
                task_id=decision.task_id,
                description=(
                    f"Actual latency {actual_latency:.0f}ms exceeds "
                    f"{self._policy.max_latency_multiplier}x estimate "
                    f"{decision.estimated_latency_ms:.0f}ms"
                ),
                evidence={
                    "actual_latency_ms": actual_latency,
                    "estimated_latency_ms": decision.estimated_latency_ms,
                    "multiplier": actual_latency / decision.estimated_latency_ms,
                },
                recommended_action="retry",
            )
            self._log.warning("anomaly_detected", **report.to_dict())
            return report

        # Output quality: content too short
        content = response.get("content", "")
        if len(content) < self._policy.min_output_length:
            report = AnomalyReport(
                anomaly_type=AnomalyType.OUTPUT_QUALITY_DROP,
                severity="medium",
                task_id=decision.task_id,
                description=(
                    f"Response content length {len(content)} is below "
                    f"minimum {self._policy.min_output_length}"
                ),
                evidence={
                    "content_length": len(content),
                    "min_length": self._policy.min_output_length,
                    "content_preview": content[:100],
                },
                recommended_action="retry",
            )
            self._log.warning("anomaly_detected", **report.to_dict())
            return report

        return None

    async def check_handoff(
        self,
        context: HandoffContext,
        previous: HandoffContext | None = None,
    ) -> AnomalyReport | None:
        """Check a handoff for anomalies.

        Args:
            context: The current handoff context.
            previous: The previous handoff for comparison, if any.

        Returns:
            An AnomalyReport if an anomaly is detected, None otherwise.
        """
        # Context degradation: state snapshot shrank significantly
        if previous is not None:
            prev_size = len(str(previous.state_snapshot))
            curr_size = len(str(context.state_snapshot))
            if prev_size > 0 and curr_size < prev_size * 0.2:
                report = AnomalyReport(
                    anomaly_type=AnomalyType.CONTEXT_DEGRADATION,
                    severity="high",
                    task_id=context.state_snapshot.get("task_id", ""),
                    agent_id=context.target_agent,
                    description=(
                        f"State snapshot shrank from {prev_size} to {curr_size} chars "
                        f"({curr_size/prev_size*100:.0f}% of previous)"
                    ),
                    evidence={
                        "previous_size": prev_size,
                        "current_size": curr_size,
                        "ratio": curr_size / prev_size if prev_size > 0 else 0,
                    },
                    recommended_action="rollback",
                )
                self._log.warning("anomaly_detected", **report.to_dict())
                return report

        return None

    async def check_coordination(
        self,
        result: CoordinationResult,
    ) -> list[AnomalyReport]:
        """Check a full coordination result for aggregate anomalies.

        Args:
            result: The completed coordination result.

        Returns:
            List of AnomalyReport instances (empty if no anomalies).
        """
        anomalies: list[AnomalyReport] = []

        # Check handoff chain depth
        if len(result.handoffs) > self._policy.max_chain_depth:
            anomalies.append(AnomalyReport(
                anomaly_type=AnomalyType.CHECKPOINT_CHAIN_BREAK,
                severity="high",
                task_id=result.task_id,
                description=(
                    f"Handoff chain depth {len(result.handoffs)} exceeds "
                    f"maximum {self._policy.max_chain_depth}"
                ),
                evidence={
                    "chain_depth": len(result.handoffs),
                    "max_depth": self._policy.max_chain_depth,
                },
                recommended_action="escalate",
            ))

        # Aggregate cost overrun
        if (
            result.responses
            and result.total_estimated_cost_usd > 0
            and result.actual_cost_usd > result.total_estimated_cost_usd * self._policy.max_cost_multiplier
        ):
            anomalies.append(AnomalyReport(
                anomaly_type=AnomalyType.COST_OVERRUN,
                severity="critical",
                task_id=result.task_id,
                description=(
                    f"Total actual cost ${result.actual_cost_usd:.4f} exceeds "
                    f"{self._policy.max_cost_multiplier}x estimate "
                    f"${result.total_estimated_cost_usd:.4f}"
                ),
                evidence={
                    "actual_cost": result.actual_cost_usd,
                    "estimated_cost": result.total_estimated_cost_usd,
                },
                recommended_action="escalate",
            ))

        # Run custom rules
        for name, rule in self._policy.custom_rules.items():
            try:
                report = await rule(result)
                if report is not None:
                    anomalies.append(report)
            except Exception as exc:
                self._log.warning(
                    "custom_rule_error", rule=name, error=str(exc)
                )

        if anomalies:
            for a in anomalies:
                self._log.warning("anomaly_detected", **a.to_dict())

        return anomalies
