"""Core data models for Kortex.

Defines the fundamental types used throughout the system: task specifications,
routing decisions, handoff contexts, execution events, and step records.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


class ModelIdentity(BaseModel):
    """Composite identity for a model served by a specific provider.

    Prevents silent collisions when multiple providers serve the same
    model name (e.g. "llama-3-70b" on both OpenRouter and a local endpoint).

    Args:
        provider: Provider name (e.g. "anthropic", "openrouter").
        model_name: Model identifier (e.g. "claude-sonnet-4-20250514").
        model_version: Optional pinned version string.
        endpoint_id: Optional discriminator for same model on different endpoints.
    """

    provider: str
    model_name: str
    model_version: str = ""
    endpoint_id: str = ""

    @property
    def key(self) -> str:
        """Return a unique composite key for this model identity.

        Format: ``provider::model_name`` or ``provider::model_name::version``
        if a version is set.
        """
        parts = [self.provider, self.model_name]
        if self.model_version:
            parts.append(self.model_version)
        return "::".join(parts)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ModelIdentity):
            return NotImplemented
        return self.key == other.key

    def __hash__(self) -> int:
        return hash(self.key)

    def __str__(self) -> str:
        return self.key


class TaskSpec(BaseModel):
    """A task to be routed to an appropriate model/provider.

    Carries the task content along with constraints (cost ceiling, latency SLA)
    and hints (complexity, required capabilities) that the router uses to make
    an informed model selection.
    """

    task_id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    complexity_hint: Literal["simple", "moderate", "complex"] = "moderate"
    complexity_class: Literal["simple", "medium", "complex"] = "medium"
    cost_ceiling_usd: float | None = None
    latency_sla_ms: float | None = None
    required_capabilities: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="before")
    @classmethod
    def _infer_complexity_class(cls, data: Any) -> Any:
        """Auto-infer complexity_class from content length when not explicitly set."""
        if isinstance(data, dict) and "complexity_class" not in data and "content" in data:
            length = len(str(data["content"]))
            if length < 500:
                data["complexity_class"] = "simple"
            elif length < 2000:
                data["complexity_class"] = "medium"
            else:
                data["complexity_class"] = "complex"
        return data

    @field_validator("required_capabilities")
    @classmethod
    def _validate_capabilities(cls, v: list[str]) -> list[str]:
        if v:
            from kortex.core.capabilities import validate_capabilities

            validate_capabilities(v)
        return v


class RoutingDecision(BaseModel):
    """The router's output after evaluating a TaskSpec against available providers.

    Contains the chosen provider/model, the reasoning behind the decision,
    and cost/latency estimates for observability and ML training.
    """

    task_id: str
    chosen_provider: str
    chosen_model: str
    chosen_model_identity: str = ""
    reasoning: str
    estimated_cost_usd: float
    estimated_latency_ms: float
    fallback_model: str | None = None
    fallback_provider: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    decided_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HandoffContext(BaseModel):
    """State passed between agents at handoff boundaries.

    The state_snapshot contains ONLY the payload crossing the agent boundary
    -- the actual data arriving from the previous step. Execution metadata
    (routing decisions, provider responses) is stored separately in
    StepExecutionRecord within CoordinationResult.steps.
    """

    handoff_id: str = Field(default_factory=lambda: str(uuid4()))
    source_agent: str
    target_agent: str
    state_snapshot: dict[str, Any]
    compressed_summary: str | None = None
    checkpoint_id: str = Field(default_factory=lambda: str(uuid4()))
    parent_checkpoint_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ExecutionEvent(BaseModel):
    """A structured event emitted during workflow execution.

    Every routing decision, handoff, failure, recovery, and completion is
    captured as an event for the dashboard and ML training pipeline.
    """

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: Literal[
        "route", "handoff", "failure", "recovery", "completion",
        "recovery_continue", "recovery_retry", "recovery_fallback",
        "recovery_rollback", "recovery_escalate", "recovery_none",
    ]
    task_id: str
    agent_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Step-level models — separate concerns at agent boundaries
# ---------------------------------------------------------------------------


class StepInput(BaseModel):
    """What an agent receives at the start of its step.

    Args:
        step_index: Position in the pipeline (0-based).
        agent_id: The agent receiving this input.
        input_payload: The actual data arriving from the previous step.
        task: The original task specification.
    """

    step_index: int
    agent_id: str
    input_payload: dict[str, Any]
    task: TaskSpec


class StepExecutionRecord(BaseModel):
    """What happened during a single agent's execution.

    Captures routing, execution, anomalies, and timing separately from
    the handoff payload.

    Args:
        step_index: Position in the pipeline (0-based).
        agent_id: The agent that executed this step.
        routing_decision: The routing decision for this step.
        provider_response: Provider response dict, or None if dry-run.
        anomalies: Any anomalies detected during this step.
        started_at: When this step started.
        completed_at: When this step completed.
        duration_ms: Wall-clock time for this step.
    """

    step_index: int
    agent_id: str
    routing_decision: RoutingDecision
    provider_response: dict[str, Any] | None = None
    anomalies: list[dict[str, Any]] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    duration_ms: float = 0.0


class StepOutput(BaseModel):
    """What an agent produces to hand off to the next step.

    Args:
        step_index: Position in the pipeline (0-based).
        agent_id: The agent that produced this output.
        output_payload: The actual result to pass forward.
    """

    step_index: int
    agent_id: str
    output_payload: dict[str, Any]


class CoordinationResult(BaseModel):
    """Result of coordinating a task across a pipeline of agents.

    Contains routing decisions, handoff chain, all emitted events, and
    aggregate metrics for cost and duration. When ``execute=True`` is used,
    ``responses`` and ``actual_cost_usd`` are populated from real API calls.

    The ``steps`` field contains serialized StepExecutionRecords with
    execution metadata (routing, provider responses, anomalies, timing)
    kept separate from handoff state_snapshots.
    """

    task_id: str
    routing_decisions: list[RoutingDecision] = Field(default_factory=list)
    handoffs: list[HandoffContext] = Field(default_factory=list)
    events: list[ExecutionEvent] = Field(default_factory=list)
    total_estimated_cost_usd: float = 0.0
    responses: list[dict[str, Any]] = Field(default_factory=list)
    actual_cost_usd: float = 0.0
    anomalies: list[dict[str, Any]] = Field(default_factory=list)
    recovery_records: list[dict[str, Any]] = Field(default_factory=list)
    steps: list[dict[str, Any]] = Field(default_factory=list)
    duration_ms: float = 0.0
    success: bool = True
    trace: dict[str, Any] | None = None
