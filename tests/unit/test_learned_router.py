"""Tests for LearnedRouter."""

from __future__ import annotations

import pytest

from kortex.core.router import ProviderModel
from kortex.core.trace import TaskTrace, TraceStep
from kortex.core.types import TaskSpec
from kortex.router.learned_router import (
    LearnedRouter,
    TrainingReport,
    _task_to_features,
    _extract_training_pairs,
    _FEATURE_NAMES,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MODELS = [
    ProviderModel(
        provider="openai", model="gpt-4o-mini",
        cost_per_1k_input_tokens=0.00015, cost_per_1k_output_tokens=0.0006,
        avg_latency_ms=200, capabilities=["reasoning", "content_generation"],
        tier="fast",
    ),
    ProviderModel(
        provider="anthropic", model="claude-sonnet",
        cost_per_1k_input_tokens=0.003, cost_per_1k_output_tokens=0.015,
        avg_latency_ms=800, capabilities=["reasoning", "code_generation", "content_generation"],
        tier="balanced",
    ),
    ProviderModel(
        provider="anthropic", model="claude-opus",
        cost_per_1k_input_tokens=0.015, cost_per_1k_output_tokens=0.075,
        avg_latency_ms=2000, capabilities=["reasoning", "code_generation", "vision"],
        tier="powerful",
    ),
]


def _make_trace(
    chosen_provider: str = "openai",
    chosen_model: str = "gpt-4o-mini",
    complexity: str = "simple",
    caps: list[str] | None = None,
    success: bool = True,
) -> TaskTrace:
    rd = {
        "chosen_provider": chosen_provider,
        "chosen_model": chosen_model,
        "estimated_cost_usd": 0.0003,
        "estimated_latency_ms": 200.0,
    }
    step = TraceStep(
        step_index=0,
        agent_id="agent_a",
        input_payload={"required_capabilities": caps or []},
        routing_decision=rd,
        policy_snapshot={},
        duration_ms=210.0,
    )
    return TaskTrace(
        task_id="t1",
        task_content="test task",
        task_complexity=complexity,
        pipeline=["agent_a"],
        steps=[step],
        success=success,
    )


def _make_many_traces(n: int = 20) -> list[TaskTrace]:
    """Make a mix of traces across models."""
    traces = []
    models_cycle = [
        ("openai", "gpt-4o-mini", "simple"),
        ("anthropic", "claude-sonnet", "moderate"),
        ("anthropic", "claude-opus", "complex"),
    ]
    for i in range(n):
        prov, model, complexity = models_cycle[i % len(models_cycle)]
        traces.append(_make_trace(prov, model, complexity))
    return traces


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def test_task_to_features_length() -> None:
    task = TaskSpec(content="test", complexity_hint="moderate")
    features = _task_to_features(task)
    assert len(features) == len(_FEATURE_NAMES)


def test_task_to_features_complexity_encoding() -> None:
    simple_f = _task_to_features(TaskSpec(content="t", complexity_hint="simple"))
    moderate_f = _task_to_features(TaskSpec(content="t", complexity_hint="moderate"))
    complex_f = _task_to_features(TaskSpec(content="t", complexity_hint="complex"))
    assert simple_f[0] < moderate_f[0] < complex_f[0]


def test_task_to_features_capability_flags() -> None:
    task = TaskSpec(content="t", required_capabilities=["reasoning", "vision"])
    features = _task_to_features(task)
    # num_required_caps = 2
    assert features[1] == 2.0
    # reasoning at index 2 (first cap), vision at index 5
    assert features[2] == 1.0  # reasoning
    assert features[5] == 1.0  # vision


def test_extract_training_pairs_skips_failed_traces() -> None:
    failed = _make_trace(success=False)
    X, y = _extract_training_pairs([failed])
    assert len(X) == 0
    assert len(y) == 0


def test_extract_training_pairs_happy_path() -> None:
    traces = _make_many_traces(9)
    X, y = _extract_training_pairs(traces)
    assert len(X) == 9
    assert len(y) == 9
    assert all("::" in label for label in y)


# ---------------------------------------------------------------------------
# LearnedRouter — cold start / no model
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_learned_router_falls_back_before_training(tmp_path) -> None:
    router = LearnedRouter(model_dir=str(tmp_path / "model"))
    for m in _MODELS:
        router.register_model(m)

    task = TaskSpec(content="hello", complexity_hint="simple")
    decision = await router.route(task)

    # Should still get a valid decision from heuristic fallback
    assert decision.chosen_model in [m.model for m in _MODELS]
    stats = router.comparison_stats()
    assert stats["heuristic_routes"] == 1
    assert stats["ml_routes"] == 0


# ---------------------------------------------------------------------------
# LearnedRouter — training
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_learned_router_train_requires_min_samples(tmp_path) -> None:
    pytest.importorskip("sklearn")
    router = LearnedRouter(model_dir=str(tmp_path / "model"), min_training_samples=5)
    # Only 2 traces — below minimum
    with pytest.raises(ValueError, match="Need at least"):
        router.train(_make_many_traces(2))


@pytest.mark.asyncio
async def test_learned_router_train_returns_report(tmp_path) -> None:
    pytest.importorskip("sklearn")
    router = LearnedRouter(model_dir=str(tmp_path / "model"), min_training_samples=5)
    for m in _MODELS:
        router.register_model(m)

    report = router.train(_make_many_traces(15))

    assert isinstance(report, TrainingReport)
    assert report.num_samples >= 5
    assert 0.0 <= report.accuracy <= 1.0
    assert len(report.feature_importances) == len(_FEATURE_NAMES)
    assert report.trained_at


@pytest.mark.asyncio
async def test_learned_router_is_trained_flag(tmp_path) -> None:
    pytest.importorskip("sklearn")
    router = LearnedRouter(model_dir=str(tmp_path / "model"), min_training_samples=5)
    assert router.is_trained is False

    for m in _MODELS:
        router.register_model(m)
    router.train(_make_many_traces(15))
    assert router.is_trained is True


# ---------------------------------------------------------------------------
# LearnedRouter — prediction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_learned_router_predicts_registered_model(tmp_path) -> None:
    pytest.importorskip("sklearn")
    router = LearnedRouter(
        model_dir=str(tmp_path / "model"),
        min_training_samples=5,
        min_confidence=0.0,  # accept any confidence
    )
    for m in _MODELS:
        router.register_model(m)

    router.train(_make_many_traces(30))

    task = TaskSpec(content="simple task", complexity_hint="simple")
    decision = await router.route(task)

    model_names = [m.model for m in _MODELS]
    assert decision.chosen_model in model_names


@pytest.mark.asyncio
async def test_learned_router_falls_back_on_low_confidence(tmp_path) -> None:
    pytest.importorskip("sklearn")
    router = LearnedRouter(
        model_dir=str(tmp_path / "model"),
        min_training_samples=5,
        min_confidence=1.1,  # impossible to satisfy
    )
    for m in _MODELS:
        router.register_model(m)
    router.train(_make_many_traces(15))

    task = TaskSpec(content="test", complexity_hint="moderate")
    decision = await router.route(task)

    # Must fall through to heuristic (still valid)
    assert decision.chosen_model in [m.model for m in _MODELS]
    assert router.comparison_stats()["heuristic_routes"] == 1


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_learned_router_save_and_load(tmp_path) -> None:
    pytest.importorskip("sklearn")
    model_dir = str(tmp_path / "model")
    router = LearnedRouter(model_dir=model_dir, min_training_samples=5)
    for m in _MODELS:
        router.register_model(m)
    router.train(_make_many_traces(15))
    router.save()

    # Load into a new instance
    router2 = LearnedRouter(model_dir=model_dir)
    assert router2.is_trained is True
    assert router2.training_report is not None


@pytest.mark.asyncio
async def test_learned_router_load_returns_false_when_no_file(tmp_path) -> None:
    router = LearnedRouter(model_dir=str(tmp_path / "nonexistent"))
    loaded = router.load()
    assert loaded is False


# ---------------------------------------------------------------------------
# Auto-retraining
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_learned_router_auto_retrain_triggers(tmp_path) -> None:
    pytest.importorskip("sklearn")
    router = LearnedRouter(
        model_dir=str(tmp_path / "model"),
        min_training_samples=5,
        retrain_every_n=10,
    )
    for m in _MODELS:
        router.register_model(m)

    assert router.is_trained is False

    for trace in _make_many_traces(10):
        router.add_trace(trace)

    assert router.is_trained is True


# ---------------------------------------------------------------------------
# Comparison stats
# ---------------------------------------------------------------------------


def test_comparison_stats_initial_state(tmp_path) -> None:
    router = LearnedRouter(model_dir=str(tmp_path / "model"))
    stats = router.comparison_stats()
    assert stats["total_routes"] == 0
    assert stats["ml_routes"] == 0
    assert stats["heuristic_routes"] == 0
    assert stats["estimated_savings_usd"] == 0.0
