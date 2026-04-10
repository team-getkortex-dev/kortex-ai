"""Tests for core data models and exception hierarchy."""

from datetime import datetime, timezone
from uuid import UUID

import pytest
from pydantic import ValidationError

from kortex.core.exceptions import (
    KortexError,
    CheckpointNotFoundError,
    HandoffError,
    ProviderError,
    RouterError,
    RoutingFailedError,
    StateError,
)
from kortex.core.types import (
    ExecutionEvent,
    HandoffContext,
    RoutingDecision,
    TaskSpec,
)


# --- Default creation ---


class TestDefaults:
    def test_task_spec_defaults(self) -> None:
        task = TaskSpec(content="Summarize this document")
        UUID(task.task_id)  # valid uuid
        assert task.content == "Summarize this document"
        assert task.complexity_hint == "moderate"
        assert task.cost_ceiling_usd is None
        assert task.latency_sla_ms is None
        assert task.required_capabilities == []
        assert task.metadata == {}
        assert isinstance(task.created_at, datetime)

    def test_routing_decision_defaults(self) -> None:
        decision = RoutingDecision(
            task_id="t1",
            chosen_provider="anthropic",
            chosen_model="claude-sonnet-4-20250514",
            reasoning="Best fit for moderate complexity",
            estimated_cost_usd=0.003,
            estimated_latency_ms=800.0,
        )
        assert decision.fallback_model is None
        assert isinstance(decision.decided_at, datetime)

    def test_handoff_context_defaults(self) -> None:
        ctx = HandoffContext(
            source_agent="planner",
            target_agent="executor",
            state_snapshot={"plan": "step1"},
        )
        UUID(ctx.handoff_id)
        UUID(ctx.checkpoint_id)
        assert ctx.compressed_summary is None
        assert ctx.parent_checkpoint_id is None
        assert isinstance(ctx.created_at, datetime)

    def test_execution_event_defaults(self) -> None:
        event = ExecutionEvent(event_type="route", task_id="t1")
        UUID(event.event_id)
        assert event.agent_id is None
        assert event.payload == {}
        assert isinstance(event.timestamp, datetime)


# --- Validation ---


class TestValidation:
    def test_task_spec_valid_complexity_hints(self) -> None:
        for hint in ("simple", "moderate", "complex"):
            task = TaskSpec(content="test", complexity_hint=hint)
            assert task.complexity_hint == hint

    def test_task_spec_rejects_invalid_complexity(self) -> None:
        with pytest.raises(ValidationError, match="complexity_hint"):
            TaskSpec(content="test", complexity_hint="impossible")  # type: ignore[arg-type]

    def test_execution_event_rejects_invalid_type(self) -> None:
        with pytest.raises(ValidationError, match="event_type"):
            ExecutionEvent(event_type="unknown", task_id="t1")  # type: ignore[arg-type]


# --- Checkpoint chain ---


class TestCheckpointChain:
    def test_parent_checkpoint_chain(self) -> None:
        first = HandoffContext(
            source_agent="a", target_agent="b", state_snapshot={"step": 1}
        )
        second = HandoffContext(
            source_agent="b",
            target_agent="c",
            state_snapshot={"step": 2},
            parent_checkpoint_id=first.checkpoint_id,
        )
        third = HandoffContext(
            source_agent="c",
            target_agent="d",
            state_snapshot={"step": 3},
            parent_checkpoint_id=second.checkpoint_id,
        )
        assert first.parent_checkpoint_id is None
        assert second.parent_checkpoint_id == first.checkpoint_id
        assert third.parent_checkpoint_id == second.checkpoint_id


# --- JSON serialization round-trip ---


class TestSerialization:
    def test_task_spec_json_roundtrip(self) -> None:
        task = TaskSpec(
            content="Generate code",
            complexity_hint="complex",
            cost_ceiling_usd=0.05,
            required_capabilities=["code_generation"],
            metadata={"lang": "python"},
        )
        json_str = task.model_dump_json()
        restored = TaskSpec.model_validate_json(json_str)
        assert restored.task_id == task.task_id
        assert restored.content == task.content
        assert restored.metadata == task.metadata

    def test_routing_decision_json_roundtrip(self) -> None:
        decision = RoutingDecision(
            task_id="t1",
            chosen_provider="openai",
            chosen_model="gpt-4o",
            reasoning="Lowest latency option",
            estimated_cost_usd=0.01,
            estimated_latency_ms=500.0,
            fallback_model="gpt-4o-mini",
        )
        restored = RoutingDecision.model_validate_json(decision.model_dump_json())
        assert restored.chosen_model == "gpt-4o"
        assert restored.fallback_model == "gpt-4o-mini"

    def test_handoff_context_json_roundtrip(self) -> None:
        ctx = HandoffContext(
            source_agent="a",
            target_agent="b",
            state_snapshot={"key": [1, 2, 3]},
            compressed_summary="short summary",
        )
        restored = HandoffContext.model_validate_json(ctx.model_dump_json())
        assert restored.state_snapshot == {"key": [1, 2, 3]}
        assert restored.compressed_summary == "short summary"

    def test_execution_event_json_roundtrip(self) -> None:
        event = ExecutionEvent(
            event_type="failure",
            task_id="t1",
            agent_id="agent-1",
            payload={"error": "timeout"},
        )
        restored = ExecutionEvent.model_validate_json(event.model_dump_json())
        assert restored.event_type == "failure"
        assert restored.payload == {"error": "timeout"}


# --- Exception hierarchy ---


class TestExceptionHierarchy:
    def test_all_inherit_from_kortex_error(self) -> None:
        for exc_class in (
            RouterError,
            StateError,
            HandoffError,
            ProviderError,
            CheckpointNotFoundError,
            RoutingFailedError,
        ):
            assert issubclass(exc_class, KortexError)

    def test_checkpoint_not_found_is_state_error(self) -> None:
        err = CheckpointNotFoundError("cp-123")
        assert isinstance(err, StateError)
        assert isinstance(err, KortexError)

    def test_routing_failed_is_router_error(self) -> None:
        err = RoutingFailedError("no model available")
        assert isinstance(err, RouterError)
        assert isinstance(err, KortexError)

    def test_exceptions_carry_message(self) -> None:
        err = ProviderError("API key invalid")
        assert str(err) == "API key invalid"

    def test_catch_base_catches_all(self) -> None:
        for exc_class in (
            RouterError,
            StateError,
            HandoffError,
            ProviderError,
            CheckpointNotFoundError,
            RoutingFailedError,
        ):
            with pytest.raises(KortexError):
                raise exc_class("test")
