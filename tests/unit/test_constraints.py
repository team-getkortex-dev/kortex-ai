"""Tests for constraint-based model filtering."""

from __future__ import annotations

import pytest

from kortex.core.router import ProviderModel
from kortex.router.constraints import (
    CapabilityConstraint,
    ConstraintSet,
    CostConstraint,
    LatencyConstraint,
    ProviderConstraint,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _model(
    provider: str = "test",
    model: str = "model-a",
    cost_in: float = 0.001,
    cost_out: float = 0.002,
    latency: float = 300.0,
    capabilities: list[str] | None = None,
    tier: str = "balanced",
) -> ProviderModel:
    return ProviderModel(
        provider=provider,
        model=model,
        cost_per_1k_input_tokens=cost_in,
        cost_per_1k_output_tokens=cost_out,
        avg_latency_ms=latency,
        capabilities=capabilities or [],
        tier=tier,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# LatencyConstraint
# ---------------------------------------------------------------------------


def test_latency_constraint_passes() -> None:
    c = LatencyConstraint(max_ms=500.0)
    m = _model(latency=300.0)
    assert c.evaluate(m) is True


def test_latency_constraint_fails() -> None:
    c = LatencyConstraint(max_ms=200.0)
    m = _model(latency=300.0)
    assert c.evaluate(m) is False


def test_latency_constraint_exact_boundary() -> None:
    c = LatencyConstraint(max_ms=300.0)
    m = _model(latency=300.0)
    assert c.evaluate(m) is True


def test_latency_constraint_describe() -> None:
    c = LatencyConstraint(max_ms=400.0)
    assert "400" in c.describe()


def test_latency_constraint_failure_reason() -> None:
    c = LatencyConstraint(max_ms=100.0)
    m = _model(latency=300.0)
    reason = c.failure_reason(m)
    assert "300" in reason
    assert "100" in reason


# ---------------------------------------------------------------------------
# CostConstraint
# ---------------------------------------------------------------------------


def test_cost_constraint_passes() -> None:
    c = CostConstraint(max_usd=1.0)
    m = _model(cost_in=0.001, cost_out=0.002)
    assert c.evaluate(m) is True


def test_cost_constraint_fails() -> None:
    c = CostConstraint(max_usd=0.000001)
    m = _model(cost_in=1.0, cost_out=2.0)
    assert c.evaluate(m) is False


def test_cost_constraint_failure_reason() -> None:
    c = CostConstraint(max_usd=0.0001)
    m = _model(cost_in=1.0, cost_out=2.0)
    reason = c.failure_reason(m)
    assert "max" in reason.lower() or "0.0001" in reason


# ---------------------------------------------------------------------------
# CapabilityConstraint
# ---------------------------------------------------------------------------


def test_capability_constraint_passes() -> None:
    c = CapabilityConstraint(["reasoning", "code_generation"])
    m = _model(capabilities=["reasoning", "code_generation", "vision"])
    assert c.evaluate(m) is True


def test_capability_constraint_fails_missing_cap() -> None:
    c = CapabilityConstraint(["reasoning", "vision"])
    m = _model(capabilities=["reasoning"])
    assert c.evaluate(m) is False


def test_capability_constraint_failure_reason() -> None:
    c = CapabilityConstraint(["vision"])
    m = _model(capabilities=[])
    reason = c.failure_reason(m)
    assert "vision" in reason


def test_capability_constraint_empty_required() -> None:
    c = CapabilityConstraint([])
    m = _model(capabilities=[])
    assert c.evaluate(m) is True


# ---------------------------------------------------------------------------
# ProviderConstraint
# ---------------------------------------------------------------------------


def test_provider_constraint_passes() -> None:
    c = ProviderConstraint(["anthropic", "openai"])
    m = _model(provider="anthropic")
    assert c.evaluate(m) is True


def test_provider_constraint_fails() -> None:
    c = ProviderConstraint(["anthropic"])
    m = _model(provider="openai")
    assert c.evaluate(m) is False


def test_provider_constraint_failure_reason() -> None:
    c = ProviderConstraint(["anthropic"])
    m = _model(provider="openai")
    reason = c.failure_reason(m)
    assert "openai" in reason or "anthropic" in reason


# ---------------------------------------------------------------------------
# ConstraintSet
# ---------------------------------------------------------------------------


def test_constraint_set_empty_passes_all() -> None:
    cs = ConstraintSet()
    m = _model()
    passed, reasons = cs.evaluate_all(m)
    assert passed is True
    assert reasons == []


def test_constraint_set_single_constraint_pass() -> None:
    cs = ConstraintSet()
    cs.add(LatencyConstraint(500.0))
    m = _model(latency=300.0)
    passed, reasons = cs.evaluate_all(m)
    assert passed is True


def test_constraint_set_single_constraint_fail() -> None:
    cs = ConstraintSet()
    cs.add(LatencyConstraint(100.0))
    m = _model(latency=300.0)
    passed, reasons = cs.evaluate_all(m)
    assert passed is False
    assert len(reasons) == 1


def test_constraint_set_multiple_failures() -> None:
    cs = ConstraintSet()
    cs.add(LatencyConstraint(50.0))
    cs.add(CostConstraint(0.000001))
    m = _model(latency=300.0, cost_in=1.0, cost_out=2.0)
    passed, reasons = cs.evaluate_all(m)
    assert passed is False
    assert len(reasons) == 2


def test_constraint_set_chaining() -> None:
    cs = ConstraintSet()
    result = cs.add(LatencyConstraint(500.0)).add(CostConstraint(1.0))
    assert result is cs
    assert len(cs) == 2


def test_constraint_set_filter_partitions() -> None:
    cs = ConstraintSet()
    cs.add(LatencyConstraint(250.0))

    fast = _model(model="fast", latency=200.0)
    slow = _model(model="slow", latency=300.0)

    passed, failures = cs.filter([fast, slow])
    assert fast in passed
    assert slow not in passed
    assert slow.identity.key in failures


def test_constraint_set_filter_empty_models() -> None:
    cs = ConstraintSet()
    cs.add(LatencyConstraint(500.0))
    passed, failures = cs.filter([])
    assert passed == []
    assert failures == {}


def test_constraint_set_all_pass() -> None:
    cs = ConstraintSet()
    cs.add(LatencyConstraint(500.0))
    cs.add(CostConstraint(1.0))

    models = [_model(model=f"m{i}", latency=100.0) for i in range(3)]
    passed, failures = cs.filter(models)
    assert len(passed) == 3
    assert failures == {}


def test_constraint_set_constraints_property() -> None:
    cs = ConstraintSet()
    c = LatencyConstraint(500.0)
    cs.add(c)
    assert c in cs.constraints
