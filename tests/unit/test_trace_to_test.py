"""Tests for TraceToTestConverter."""

from __future__ import annotations

import pytest

from kortex.core.trace import TaskTrace, TraceStep
from kortex.testing.trace_to_test import ConversionConfig, TraceToTestConverter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trace(
    task_id: str = "task-001",
    task_content: str = "Summarise this document",
    complexity: str = "moderate",
    pipeline: list[str] | None = None,
    steps: list[TraceStep] | None = None,
    success: bool = True,
    cost: float = 0.001,
    policy: dict | None = None,
) -> TaskTrace:
    if pipeline is None:
        pipeline = ["researcher", "writer"]
    if steps is None:
        steps = [
            TraceStep(
                step_index=0,
                agent_id="researcher",
                input_payload={},
                routing_decision={
                    "chosen_model": "gpt-4o-mini",
                    "chosen_provider": "openai",
                    "estimated_cost_usd": 0.0005,
                    "estimated_latency_ms": 200.0,
                },
                policy_snapshot={},
            ),
            TraceStep(
                step_index=1,
                agent_id="writer",
                input_payload={},
                routing_decision={
                    "chosen_model": "claude-haiku",
                    "chosen_provider": "anthropic",
                    "estimated_cost_usd": 0.0005,
                    "estimated_latency_ms": 300.0,
                },
                policy_snapshot={},
            ),
        ]
    return TaskTrace(
        task_id=task_id,
        task_content=task_content,
        task_complexity=complexity,
        pipeline=pipeline,
        steps=steps,
        success=success,
        total_estimated_cost_usd=cost,
        policy_snapshot=policy or {},
    )


# ---------------------------------------------------------------------------
# ConversionConfig defaults
# ---------------------------------------------------------------------------


def test_conversion_config_defaults() -> None:
    cfg = ConversionConfig()
    assert cfg.cost_tolerance_pct == pytest.approx(0.20)
    assert cfg.latency_tolerance_pct == pytest.approx(0.50)
    assert cfg.include_execute_assertion is False
    assert cfg.min_cost_threshold == 0.0
    assert cfg.sample_rate == 1.0
    assert "regression" in cfg.test_module_docstring.lower()
    assert cfg.fixture_name == "routing_runtime"


# ---------------------------------------------------------------------------
# _filter_traces
# ---------------------------------------------------------------------------


def test_filter_skips_failed_traces() -> None:
    converter = TraceToTestConverter()
    traces = [_make_trace(success=False), _make_trace(success=True)]
    result = converter._filter_traces(traces)
    assert len(result) == 1
    assert result[0].success is True


def test_filter_skips_below_cost_threshold() -> None:
    cfg = ConversionConfig(min_cost_threshold=0.01)
    converter = TraceToTestConverter(cfg)
    traces = [_make_trace(cost=0.001), _make_trace(cost=0.05)]
    result = converter._filter_traces(traces)
    assert len(result) == 1
    assert result[0].total_estimated_cost_usd == pytest.approx(0.05)


def test_filter_sample_rate_half() -> None:
    cfg = ConversionConfig(sample_rate=0.5)
    converter = TraceToTestConverter(cfg)
    traces = [_make_trace(task_id=f"task-{i}") for i in range(10)]
    result = converter._filter_traces(traces)
    # Sample rate 0.5 → every other trace (indices 0, 2, 4, ...)
    assert len(result) == 5


def test_filter_sample_rate_full() -> None:
    converter = TraceToTestConverter()
    traces = [_make_trace(task_id=f"task-{i}") for i in range(8)]
    result = converter._filter_traces(traces)
    assert len(result) == 8


def test_filter_sample_rate_deterministic() -> None:
    cfg = ConversionConfig(sample_rate=0.5)
    converter = TraceToTestConverter(cfg)
    traces = [_make_trace(task_id=f"t{i}") for i in range(10)]
    r1 = converter._filter_traces(traces)
    r2 = converter._filter_traces(traces)
    assert [t.task_id for t in r1] == [t.task_id for t in r2]


# ---------------------------------------------------------------------------
# generate_tests — output structure
# ---------------------------------------------------------------------------


def test_generate_tests_returns_string() -> None:
    converter = TraceToTestConverter()
    traces = [_make_trace()]
    code = converter.generate_tests(traces)
    assert isinstance(code, str)
    assert len(code) > 0


def test_generate_tests_has_header() -> None:
    converter = TraceToTestConverter()
    code = converter.generate_tests([_make_trace()])
    assert "AUTO-GENERATED" in code
    assert "import pytest" in code
    assert "from kortex.core.runtime import KortexRuntime" in code


def test_generate_tests_has_fixture() -> None:
    converter = TraceToTestConverter()
    code = converter.generate_tests([_make_trace()])
    assert "@pytest.fixture" in code
    assert "routing_runtime" in code
    assert "KortexRuntime" in code


def test_generate_tests_has_test_function() -> None:
    converter = TraceToTestConverter()
    code = converter.generate_tests([_make_trace(task_id="my-task")])
    assert "async def test_routing_regression_" in code
    assert "my_task" in code  # sanitised task_id


def test_generate_tests_step_assertions() -> None:
    converter = TraceToTestConverter()
    code = converter.generate_tests([_make_trace()])
    assert "chosen_model" in code
    assert "gpt-4o-mini" in code
    assert "openai" in code
    assert "estimated_cost_usd" in code


def test_generate_tests_zero_traces() -> None:
    converter = TraceToTestConverter()
    code = converter.generate_tests([])
    # Should still produce valid header + fixtures, just no test functions
    assert "import pytest" in code
    assert "async def test_" not in code


def test_generate_tests_multiple_traces() -> None:
    converter = TraceToTestConverter()
    traces = [_make_trace(task_id=f"task-{i}") for i in range(3)]
    code = converter.generate_tests(traces)
    count = code.count("async def test_routing_regression_")
    assert count == 3


def test_generate_tests_filters_failures() -> None:
    converter = TraceToTestConverter()
    traces = [_make_trace(success=False), _make_trace(task_id="good-task")]
    code = converter.generate_tests(traces)
    assert "good_task" in code
    assert code.count("async def test_") == 1


# ---------------------------------------------------------------------------
# generate_tests — file output
# ---------------------------------------------------------------------------


def test_generate_tests_writes_file(tmp_path) -> None:
    converter = TraceToTestConverter()
    out = tmp_path / "tests" / "generated" / "test_out.py"
    code = converter.generate_tests([_make_trace()], output_path=str(out))
    assert out.exists()
    assert out.read_text(encoding="utf-8") == code


def test_generate_tests_creates_parent_dirs(tmp_path) -> None:
    converter = TraceToTestConverter()
    out = tmp_path / "deep" / "nested" / "test_out.py"
    converter.generate_tests([_make_trace()], output_path=str(out))
    assert out.exists()


# ---------------------------------------------------------------------------
# _render_policy
# ---------------------------------------------------------------------------


def test_render_policy_empty() -> None:
    converter = TraceToTestConverter()
    code = converter._render_policy({})
    assert "No policy snapshot" in code


def test_render_policy_with_data() -> None:
    converter = TraceToTestConverter()
    policy = {"name": "cost_optimized", "minimize": "cost"}
    code = converter._render_policy(policy)
    assert "RoutingPolicy" in code
    assert "from_dict" in code


# ---------------------------------------------------------------------------
# _render_step_assertion tolerances
# ---------------------------------------------------------------------------


def test_step_assertion_cost_tolerance() -> None:
    cfg = ConversionConfig(cost_tolerance_pct=0.10)
    converter = TraceToTestConverter(cfg)
    step = TraceStep(
        step_index=0,
        agent_id="agent",
        input_payload={},
        routing_decision={
            "chosen_model": "m",
            "chosen_provider": "p",
            "estimated_cost_usd": 0.001,
            "estimated_latency_ms": 100.0,
        },
        policy_snapshot={},
    )
    code = converter._render_step_assertion(step, 0)
    # tolerance = 0.001 * 1.10 = 0.001100
    assert "0.001100" in code


# ---------------------------------------------------------------------------
# Custom fixture name
# ---------------------------------------------------------------------------


def test_custom_fixture_name() -> None:
    cfg = ConversionConfig(fixture_name="my_runtime")
    converter = TraceToTestConverter(cfg)
    code = converter.generate_tests([_make_trace()])
    assert "my_runtime" in code
