"""Tests for RoutingDiagnostics and enhanced RoutingFailedError."""

from __future__ import annotations

import pytest

from kortex.core.exceptions import RoutingFailedError
from kortex.core.router import ProviderModel, Router
from kortex.core.types import TaskSpec
from kortex.router.constraints import ConstraintSet, LatencyConstraint
from kortex.router.diagnostics import RoutingDiagnostics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _model(
    provider: str = "openai",
    model: str = "gpt-4o",
    latency: float = 300.0,
    cost_in: float = 0.001,
    cost_out: float = 0.002,
    capabilities: list[str] | None = None,
) -> ProviderModel:
    return ProviderModel(
        provider=provider,
        model=model,
        cost_per_1k_input_tokens=cost_in,
        cost_per_1k_output_tokens=cost_out,
        avg_latency_ms=latency,
        capabilities=capabilities or [],
        tier="balanced",
    )


def _task(**kwargs: object) -> TaskSpec:
    return TaskSpec(content="Test task", **kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# RoutingFailedError enhanced fields
# ---------------------------------------------------------------------------


def test_routing_failed_error_default_fields() -> None:
    err = RoutingFailedError("no model found")
    assert err.failed_models == []
    assert err.closest_model is None
    assert err.suggestion is None


def test_routing_failed_error_with_fields() -> None:
    err = RoutingFailedError(
        "no model found",
        failed_models=[("openai::gpt-4o", "too slow")],
        closest_model="openai::gpt-4o",
        suggestion="Raise the latency SLA.",
    )
    assert len(err.failed_models) == 1
    assert err.closest_model == "openai::gpt-4o"
    assert "latency" in err.suggestion  # type: ignore[operator]


def test_routing_failed_error_is_router_error() -> None:
    from kortex.core.exceptions import RouterError

    err = RoutingFailedError("test")
    assert isinstance(err, RouterError)


def test_routing_failed_error_message_preserved() -> None:
    err = RoutingFailedError("custom message")
    assert "custom message" in str(err)


# ---------------------------------------------------------------------------
# RoutingDiagnostics.explain_failure
# ---------------------------------------------------------------------------


def test_explain_failure_no_candidates() -> None:
    diag = RoutingDiagnostics()
    task = _task()
    msg = diag.explain_failure(task, candidates=[])
    assert task.task_id in msg
    assert "Register models" in msg


def test_explain_failure_includes_task_id() -> None:
    diag = RoutingDiagnostics()
    task = _task()
    msg = diag.explain_failure(task, [_model()])
    assert task.task_id in msg


def test_explain_failure_includes_constraint_failures() -> None:
    diag = RoutingDiagnostics()
    task = _task()
    m = _model()
    failures = {m.identity.key: ["latency 300ms > max 100ms"]}
    msg = diag.explain_failure(task, [m], constraint_failures=failures)
    assert m.identity.key in msg
    assert "latency" in msg


def test_explain_failure_suggests_relax_constraints() -> None:
    diag = RoutingDiagnostics()
    task = _task()
    m = _model()
    failures = {m.identity.key: ["too slow"]}
    msg = diag.explain_failure(task, [m], constraint_failures=failures)
    assert "Relax" in msg or "relax" in msg.lower() or "constraint" in msg


def test_explain_failure_cost_ceiling_suggestion() -> None:
    diag = RoutingDiagnostics()
    # Task has a very tight cost ceiling that no model can satisfy
    task = _task(cost_ceiling_usd=0.0000001)
    m = _model(cost_in=1.0, cost_out=2.0)
    msg = diag.explain_failure(task, [m])
    assert "cost" in msg.lower() or "ceiling" in msg.lower()


def test_explain_failure_latency_sla_suggestion() -> None:
    diag = RoutingDiagnostics()
    task = _task(latency_sla_ms=10.0)
    m = _model(latency=300.0)
    msg = diag.explain_failure(task, [m])
    assert "latency" in msg.lower() or "sla" in msg.lower()


def test_explain_failure_capability_suggestion() -> None:
    diag = RoutingDiagnostics()
    from kortex.core.capabilities import Capability

    task = _task(required_capabilities=[Capability.VISION.value])
    m = _model(capabilities=[])
    msg = diag.explain_failure(task, [m])
    assert "vision" in msg.lower() or "capabilities" in msg.lower()


def test_explain_failure_closest_model_shown() -> None:
    diag = RoutingDiagnostics()
    task = _task()
    m = _model(model="near-miss")
    failures = {m.identity.key: ["one failure"]}
    msg = diag.explain_failure(task, [m], constraint_failures=failures)
    assert "near-miss" in msg or "Closest" in msg


def test_explain_failure_heuristic_failures_shown() -> None:
    diag = RoutingDiagnostics()
    task = _task()
    m = _model()
    msg = diag.explain_failure(
        task, [m], heuristic_failures=["1 model exceeded cost ceiling"]
    )
    assert "cost ceiling" in msg


# ---------------------------------------------------------------------------
# Router raises RoutingFailedError with diagnostics when constraints block all
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_router_raises_with_diagnostics_when_constraints_block_all() -> None:
    router = Router()
    router.register_model(_model(latency=500.0))

    cs = ConstraintSet()
    cs.add(LatencyConstraint(max_ms=100.0))
    router.set_constraints(cs)

    task = _task()
    with pytest.raises(RoutingFailedError) as exc_info:
        await router.route(task)

    err = exc_info.value
    assert len(err.failed_models) > 0
    assert err.closest_model is not None


@pytest.mark.asyncio
async def test_router_passes_when_some_models_satisfy_constraints() -> None:
    router = Router()
    router.register_model(_model(model="slow", latency=500.0))
    router.register_model(_model(model="fast", latency=50.0))

    cs = ConstraintSet()
    cs.add(LatencyConstraint(max_ms=100.0))
    router.set_constraints(cs)

    task = _task()
    decision = await router.route(task)
    assert decision.chosen_model == "fast"
