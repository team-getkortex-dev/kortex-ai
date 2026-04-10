"""Tests for the trace data model."""

from __future__ import annotations

import json

from kortex.core.trace import TaskTrace, TraceStep


def _sample_step(index: int = 0, agent_id: str = "agent_a") -> TraceStep:
    return TraceStep(
        step_index=index,
        agent_id=agent_id,
        input_payload={"content": "test input", "task_id": "t1"},
        routing_decision={
            "task_id": "t1",
            "chosen_provider": "anthropic",
            "chosen_model": "claude-sonnet",
            "chosen_model_identity": "anthropic::claude-sonnet",
            "reasoning": "Cheapest balanced-tier model.",
            "estimated_cost_usd": 0.0105,
            "estimated_latency_ms": 800,
            "fallback_model": "gpt-4o-mini",
            "fallback_provider": "openai",
        },
        policy_snapshot={
            "name": "default",
            "constraints": {"max_cost_usd": None},
            "objective": {"minimize": "cost"},
            "fallback": {"strategy": "next_cheapest"},
        },
        provider_response={"content": "response text", "cost_usd": 0.008},
        handoff_checkpoint_id="ckpt-001",
        anomalies=[],
        recovery_records=[],
        started_at="2026-03-29T10:00:00+00:00",
        completed_at="2026-03-29T10:00:01+00:00",
        duration_ms=800.5,
    )


def _sample_trace() -> TaskTrace:
    return TaskTrace(
        trace_id="trace-001",
        task_id="task-001",
        task_content="Write about AI coordination",
        task_complexity="moderate",
        pipeline=["researcher", "writer"],
        steps=[
            _sample_step(0, "researcher"),
            _sample_step(1, "writer"),
        ],
        policy_snapshot={
            "name": "default",
            "constraints": {"max_cost_usd": None},
            "objective": {"minimize": "cost"},
            "fallback": {"strategy": "next_cheapest"},
        },
        total_estimated_cost_usd=0.021,
        total_actual_cost_usd=0.016,
        total_duration_ms=1600.0,
        success=True,
        created_at="2026-03-29T10:00:00+00:00",
    )


# ---------------------------------------------------------------------------
# 1. TaskTrace serializes to JSON and back without data loss
# ---------------------------------------------------------------------------


class TestJsonRoundtrip:
    def test_to_json_and_back(self) -> None:
        trace = _sample_trace()
        json_str = trace.to_json()

        # Must be valid JSON
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

        # Round-trip
        restored = TaskTrace.from_json(json_str)
        assert restored.trace_id == trace.trace_id
        assert restored.task_id == trace.task_id
        assert restored.task_content == trace.task_content
        assert restored.task_complexity == trace.task_complexity
        assert restored.pipeline == trace.pipeline
        assert restored.total_estimated_cost_usd == trace.total_estimated_cost_usd
        assert restored.total_actual_cost_usd == trace.total_actual_cost_usd
        assert restored.total_duration_ms == trace.total_duration_ms
        assert restored.success == trace.success
        assert restored.created_at == trace.created_at
        assert len(restored.steps) == 2
        assert restored.steps[0].agent_id == "researcher"
        assert restored.steps[1].agent_id == "writer"


# ---------------------------------------------------------------------------
# 2. TaskTrace.from_dict(trace.to_dict()) round-trips perfectly
# ---------------------------------------------------------------------------


class TestDictRoundtrip:
    def test_from_dict_to_dict(self) -> None:
        trace = _sample_trace()
        data = trace.to_dict()
        restored = TaskTrace.from_dict(data)

        # Compare dicts instead of objects for deep equality
        assert restored.to_dict() == data


# ---------------------------------------------------------------------------
# 3. TraceStep captures all routing decision fields
# ---------------------------------------------------------------------------


class TestTraceStepFields:
    def test_routing_decision_fields_preserved(self) -> None:
        step = _sample_step()
        rd = step.routing_decision

        assert rd["task_id"] == "t1"
        assert rd["chosen_provider"] == "anthropic"
        assert rd["chosen_model"] == "claude-sonnet"
        assert rd["chosen_model_identity"] == "anthropic::claude-sonnet"
        assert rd["reasoning"] == "Cheapest balanced-tier model."
        assert rd["estimated_cost_usd"] == 0.0105
        assert rd["estimated_latency_ms"] == 800
        assert rd["fallback_model"] == "gpt-4o-mini"
        assert rd["fallback_provider"] == "openai"

    def test_step_metadata_preserved(self) -> None:
        step = _sample_step()
        assert step.provider_response is not None
        assert step.provider_response["content"] == "response text"
        assert step.handoff_checkpoint_id == "ckpt-001"
        assert step.started_at == "2026-03-29T10:00:00+00:00"
        assert step.duration_ms == 800.5


# ---------------------------------------------------------------------------
# 4. Policy snapshot is captured at trace creation time
# ---------------------------------------------------------------------------


class TestPolicySnapshot:
    def test_policy_snapshot_in_trace(self) -> None:
        trace = _sample_trace()
        assert trace.policy_snapshot["name"] == "default"
        assert trace.policy_snapshot["objective"]["minimize"] == "cost"

    def test_policy_snapshot_in_steps(self) -> None:
        trace = _sample_trace()
        for step in trace.steps:
            assert step.policy_snapshot["name"] == "default"

    def test_policy_snapshot_survives_serialization(self) -> None:
        trace = _sample_trace()
        restored = TaskTrace.from_dict(trace.to_dict())
        assert restored.policy_snapshot == trace.policy_snapshot
        assert restored.steps[0].policy_snapshot == trace.steps[0].policy_snapshot
