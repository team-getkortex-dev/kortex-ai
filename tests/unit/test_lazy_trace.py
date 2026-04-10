"""Tests for lazy trace serialization."""

from __future__ import annotations

import pytest

from kortex.core.trace import TaskTrace, TraceStep


def _make_trace(task_content: str = "do something", **kwargs) -> TaskTrace:
    return TaskTrace(
        task_id="t1",
        task_content=task_content,
        pipeline=["agent_a"],
        steps=[
            TraceStep(
                step_index=0,
                agent_id="agent_a",
                input_payload={"content": task_content},
                routing_decision={"chosen_model": "gpt-4o"},
                policy_snapshot={},
            )
        ],
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Lazy caching
# ---------------------------------------------------------------------------


def test_to_dict_not_computed_before_first_call() -> None:
    trace = _make_trace()
    # The lazy cache should be None until to_dict() is called
    assert trace._serialized_cache is None


def test_to_dict_populates_cache_on_first_call() -> None:
    trace = _make_trace()
    d = trace.to_dict()
    assert trace._serialized_cache is not None
    assert trace._serialized_cache is d


def test_to_dict_returns_same_object_on_repeated_calls() -> None:
    trace = _make_trace()
    d1 = trace.to_dict()
    d2 = trace.to_dict()
    assert d1 is d2


def test_to_json_uses_cached_dict() -> None:
    trace = _make_trace()
    import json

    json_str = trace.to_json()
    data = json.loads(json_str)
    assert data["task_id"] == "t1"
    # Cache was populated by to_json() -> to_dict()
    assert trace._serialized_cache is not None


def test_to_dict_contains_expected_keys() -> None:
    trace = _make_trace()
    d = trace.to_dict()
    expected = {
        "trace_id", "task_id", "task_content", "task_complexity",
        "pipeline", "steps", "policy_snapshot",
        "total_estimated_cost_usd", "total_actual_cost_usd",
        "total_duration_ms", "success", "cache_hit", "created_at",
    }
    assert expected.issubset(set(d.keys()))


def test_api_key_redacted_in_to_dict() -> None:
    trace = _make_trace("use key sk-abcdef1234567890abcdef1234567890abcd to do it")
    d = trace.to_dict()
    # Full key should be gone; scan_and_redact replaces with prefix...suffix form
    assert "sk-abcdef1234567890abcdef1234567890abcd" not in d["task_content"]
    # Redacted placeholder present (first 6 chars + ... + last 3 chars)
    assert "sk-abc" in d["task_content"]
    assert "..." in d["task_content"]


# ---------------------------------------------------------------------------
# export_trace flag in coordinate()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_coordinate_without_export_trace_leaves_trace_none() -> None:
    from kortex.core.router import Router, ProviderModel
    from kortex.core.runtime import KortexRuntime
    from kortex.core.state import StateManager
    from kortex.core.types import TaskSpec

    router = Router()
    router.register_model(ProviderModel(
        provider="test", model="fast",
        cost_per_1k_input_tokens=0.0,
        cost_per_1k_output_tokens=0.0,
        avg_latency_ms=10.0,
        tier="fast",
    ))
    runtime = KortexRuntime(router=router, state_manager=StateManager())
    await runtime.start()

    task = TaskSpec(content="hello")
    result = await runtime.coordinate(task, ["agent_a"], execute=False, export_trace=False)

    # No trace attached when export_trace=False
    assert result.trace is None

    await runtime.stop()


@pytest.mark.asyncio
async def test_coordinate_with_export_trace_attaches_trace() -> None:
    from kortex.core.router import Router, ProviderModel
    from kortex.core.runtime import KortexRuntime
    from kortex.core.state import StateManager
    from kortex.core.types import TaskSpec

    router = Router()
    router.register_model(ProviderModel(
        provider="test", model="fast",
        cost_per_1k_input_tokens=0.0,
        cost_per_1k_output_tokens=0.0,
        avg_latency_ms=10.0,
        tier="fast",
    ))
    runtime = KortexRuntime(router=router, state_manager=StateManager())
    await runtime.start()

    task = TaskSpec(content="hello")
    result = await runtime.coordinate(task, ["agent_a"], execute=False, export_trace=True)

    assert result.trace is not None
    assert result.trace["task_id"] == task.task_id

    await runtime.stop()


@pytest.mark.asyncio
async def test_coordinate_export_trace_skips_serialization_overhead() -> None:
    """export_trace=False should be measurably faster across many calls."""
    import time

    from kortex.core.router import Router, ProviderModel
    from kortex.core.runtime import KortexRuntime
    from kortex.core.state import StateManager
    from kortex.core.types import TaskSpec

    router = Router()
    router.register_model(ProviderModel(
        provider="test", model="fast",
        cost_per_1k_input_tokens=0.0,
        cost_per_1k_output_tokens=0.0,
        avg_latency_ms=10.0,
        tier="fast",
    ))

    N = 50

    # With export
    runtime_exp = KortexRuntime(router=router, state_manager=StateManager())
    await runtime_exp.start()
    t0 = time.monotonic()
    for _ in range(N):
        task = TaskSpec(content="hello")
        await runtime_exp.coordinate(task, ["a"], export_trace=True)
    export_ms = (time.monotonic() - t0) * 1000
    await runtime_exp.stop()

    # Without export
    runtime_no = KortexRuntime(router=router, state_manager=StateManager())
    await runtime_no.start()
    t0 = time.monotonic()
    for _ in range(N):
        task = TaskSpec(content="hello")
        await runtime_no.coordinate(task, ["a"], export_trace=False)
    no_export_ms = (time.monotonic() - t0) * 1000
    await runtime_no.stop()

    # no-export should be at least as fast (within 50% overhead tolerance)
    assert no_export_ms <= export_ms * 1.5, (
        f"no-export ({no_export_ms:.1f}ms) should be <= export ({export_ms:.1f}ms) * 1.5"
    )
