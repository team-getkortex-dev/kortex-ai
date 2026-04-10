"""Trace data model for Kortex replay architecture.

A ``TaskTrace`` captures everything that happened during a coordination run
— routing decisions, policy snapshots, provider responses, anomalies, recovery
actions, and timing — in a fully serializable format suitable for replay,
diffing, and auditing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from kortex.security.redaction import scan_and_redact


@dataclass
class TraceStep:
    """One step in a coordination trace.

    Captures the full context of a single agent's execution: what went in,
    what routing decision was made, what policy was active, what came out,
    and what anomalies or recoveries occurred.

    Args:
        step_index: Position in the pipeline (0-based).
        agent_id: The agent that executed this step.
        input_payload: The boundary-crossing payload received by this agent.
        routing_decision: Serialized RoutingDecision for this step.
        policy_snapshot: Serialized RoutingPolicy active at this step.
        provider_response: Serialized provider response, or None if dry-run.
        handoff_checkpoint_id: Checkpoint ID created at this step's handoff.
        anomalies: Anomalies detected during this step.
        recovery_records: Recovery actions taken during this step.
        started_at: ISO-format timestamp when the step started.
        completed_at: ISO-format timestamp when the step completed.
        duration_ms: Wall-clock time for this step.
    """

    step_index: int
    agent_id: str
    input_payload: dict[str, Any]
    routing_decision: dict[str, Any]
    policy_snapshot: dict[str, Any]
    provider_response: dict[str, Any] | None = None
    handoff_checkpoint_id: str | None = None
    anomalies: list[dict[str, Any]] = field(default_factory=list)
    recovery_records: list[dict[str, Any]] = field(default_factory=list)
    started_at: str = ""
    completed_at: str = ""
    duration_ms: float = 0.0


@dataclass
class TaskTrace:
    """Complete trace of a coordination run.

    Fully serializable — every field is a primitive, string, or dict.
    Round-trips through JSON without data loss.

    Args:
        trace_id: Unique identifier for this trace.
        task_id: The task that was coordinated.
        task_content: The original task content string.
        task_complexity: The complexity hint used.
        pipeline: Ordered list of agent IDs.
        steps: Per-step trace data.
        policy_snapshot: The routing policy active for this trace.
        total_estimated_cost_usd: Sum of estimated costs across all steps.
        total_actual_cost_usd: Sum of actual costs (0 for dry-runs).
        total_duration_ms: Wall-clock time for the full coordination.
        success: Whether the coordination completed without escalation.
        created_at: ISO-format timestamp.
    """

    trace_id: str = field(default_factory=lambda: str(uuid4()))
    task_id: str = ""
    task_content: str = ""
    task_complexity: str = "moderate"
    pipeline: list[str] = field(default_factory=list)
    steps: list[TraceStep] = field(default_factory=list)
    policy_snapshot: dict[str, Any] = field(default_factory=dict)
    total_estimated_cost_usd: float = 0.0
    total_actual_cost_usd: float = 0.0
    total_duration_ms: float = 0.0
    success: bool = True
    cache_hit: bool = False
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # Private: lazy serialization cache (excluded from __init__, repr, compare)
    _serialized_cache: dict[str, Any] | None = field(
        default=None, init=False, compare=False, repr=False
    )

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict with API keys redacted.

        The result is computed once and cached on the instance; subsequent
        calls return the same dict object without re-serializing.
        """
        if self._serialized_cache is None:
            self._serialized_cache = {
                "trace_id": self.trace_id,
                "task_id": self.task_id,
                "task_content": scan_and_redact(self.task_content),
                "task_complexity": self.task_complexity,
                "pipeline": list(self.pipeline),
                "steps": [self._step_to_dict(s) for s in self.steps],
                "policy_snapshot": self.policy_snapshot,
                "total_estimated_cost_usd": self.total_estimated_cost_usd,
                "total_actual_cost_usd": self.total_actual_cost_usd,
                "total_duration_ms": self.total_duration_ms,
                "success": self.success,
                "cache_hit": self.cache_hit,
                "created_at": self.created_at,
            }
        return self._serialized_cache

    def to_json(self) -> str:
        """Serialize to pretty-printed JSON."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskTrace:
        """Deserialize from a dict (inverse of ``to_dict``)."""
        steps = [cls._step_from_dict(s) for s in data.get("steps", [])]
        return cls(
            trace_id=data.get("trace_id", str(uuid4())),
            task_id=data.get("task_id", ""),
            task_content=data.get("task_content", ""),
            task_complexity=data.get("task_complexity", "moderate"),
            pipeline=data.get("pipeline", []),
            steps=steps,
            policy_snapshot=data.get("policy_snapshot", {}),
            total_estimated_cost_usd=data.get("total_estimated_cost_usd", 0.0),
            total_actual_cost_usd=data.get("total_actual_cost_usd", 0.0),
            total_duration_ms=data.get("total_duration_ms", 0.0),
            success=data.get("success", True),
            cache_hit=data.get("cache_hit", False),
            created_at=data.get("created_at", ""),
        )

    @classmethod
    def from_json(cls, json_str: str) -> TaskTrace:
        """Deserialize from a JSON string."""
        return cls.from_dict(json.loads(json_str))

    # -- internal helpers --------------------------------------------------

    @staticmethod
    def _step_to_dict(step: TraceStep) -> dict[str, Any]:
        return {
            "step_index": step.step_index,
            "agent_id": step.agent_id,
            "input_payload": step.input_payload,
            "routing_decision": step.routing_decision,
            "policy_snapshot": step.policy_snapshot,
            "provider_response": step.provider_response,
            "handoff_checkpoint_id": step.handoff_checkpoint_id,
            "anomalies": list(step.anomalies),
            "recovery_records": list(step.recovery_records),
            "started_at": step.started_at,
            "completed_at": step.completed_at,
            "duration_ms": step.duration_ms,
        }

    @staticmethod
    def _step_from_dict(data: dict[str, Any]) -> TraceStep:
        return TraceStep(
            step_index=data.get("step_index", 0),
            agent_id=data.get("agent_id", ""),
            input_payload=data.get("input_payload", {}),
            routing_decision=data.get("routing_decision", {}),
            policy_snapshot=data.get("policy_snapshot", {}),
            provider_response=data.get("provider_response"),
            handoff_checkpoint_id=data.get("handoff_checkpoint_id"),
            anomalies=data.get("anomalies", []),
            recovery_records=data.get("recovery_records", []),
            started_at=data.get("started_at", ""),
            completed_at=data.get("completed_at", ""),
            duration_ms=data.get("duration_ms", 0.0),
        )
