"""Tests for the A/B testing framework."""

from __future__ import annotations

import math

import pytest

from kortex.core.ab_testing import (
    ABTest,
    ExperimentConfig,
    ExperimentResult,
    PolicyMetrics,
    _welch_t_test,
)
from kortex.core.policy import RoutingPolicy


# ---------------------------------------------------------------------------
# _welch_t_test
# ---------------------------------------------------------------------------


def test_welch_t_test_empty() -> None:
    assert _welch_t_test([], []) == 1.0


def test_welch_t_test_single_element() -> None:
    assert _welch_t_test([1.0], [2.0]) == 1.0


def test_welch_t_test_identical_samples() -> None:
    a = [1.0] * 20
    b = [1.0] * 20
    p = _welch_t_test(a, b)
    assert p == pytest.approx(1.0, abs=0.05)


def test_welch_t_test_clearly_different() -> None:
    # Group A: ~1.0, Group B: ~10.0 — very different means
    a = [1.0 + i * 0.01 for i in range(30)]
    b = [10.0 + i * 0.01 for i in range(30)]
    p = _welch_t_test(a, b)
    assert p < 0.001


def test_welch_t_test_p_value_in_range() -> None:
    a = [1.0, 1.2, 1.1, 0.9, 1.05]
    b = [1.0, 1.2, 1.1, 0.9, 1.05]
    p = _welch_t_test(a, b)
    assert 0.0 <= p <= 1.0


def test_welch_t_test_slightly_different() -> None:
    import random
    rng = random.Random(42)
    a = [rng.gauss(1.0, 0.1) for _ in range(30)]
    b = [rng.gauss(1.0, 0.1) for _ in range(30)]
    p = _welch_t_test(a, b)
    # Same distribution — should NOT be significant
    assert p > 0.05


# ---------------------------------------------------------------------------
# PolicyMetrics
# ---------------------------------------------------------------------------


def test_policy_metrics_avg_cost_empty() -> None:
    m = PolicyMetrics(policy_name="test")
    assert m.avg_cost == 0.0


def test_policy_metrics_avg_cost() -> None:
    m = PolicyMetrics(policy_name="test", costs=[0.001, 0.002, 0.003])
    assert m.avg_cost == pytest.approx(0.002)


def test_policy_metrics_sample_count() -> None:
    m = PolicyMetrics(policy_name="test", costs=[0.001, 0.002])
    assert m.sample_count == 2


def test_policy_metrics_to_dict() -> None:
    m = PolicyMetrics(policy_name="test", costs=[0.001], latencies=[200.0])
    d = m.to_dict()
    assert d["policy_name"] == "test"
    assert "avg_cost" in d
    assert "avg_latency" in d


# ---------------------------------------------------------------------------
# ABTest — traffic splitting
# ---------------------------------------------------------------------------


def _make_experiment(traffic_split: float = 0.5) -> ABTest:
    config = ExperimentConfig(
        name="test_exp",
        control_policy=RoutingPolicy.cost_optimized(),
        treatment_policy=RoutingPolicy.quality_optimized(),
        traffic_split=traffic_split,
        min_samples=5,
        auto_promote=False,
    )
    return ABTest(config, rng_seed=42)


def test_split_traffic_returns_policy() -> None:
    exp = _make_experiment()
    policy = exp.split_traffic()
    assert isinstance(policy, RoutingPolicy)


def test_split_traffic_honors_100pct_treatment() -> None:
    exp = _make_experiment(traffic_split=1.0)
    policies = [exp.split_traffic() for _ in range(20)]
    names = {p.name for p in policies}
    assert "quality_optimized" in names
    assert "cost_optimized" not in names


def test_split_traffic_honors_0pct_treatment() -> None:
    exp = _make_experiment(traffic_split=0.0)
    policies = [exp.split_traffic() for _ in range(20)]
    names = {p.name for p in policies}
    assert "cost_optimized" in names
    assert "quality_optimized" not in names


def test_split_traffic_approximately_50_50() -> None:
    exp = _make_experiment(traffic_split=0.5)
    control_count = 0
    treatment_count = 0
    for _ in range(1000):
        p = exp.split_traffic()
        if p.name == "cost_optimized":
            control_count += 1
        else:
            treatment_count += 1
    # Should be roughly 50/50 (within 10%)
    assert abs(control_count - treatment_count) < 150


# ---------------------------------------------------------------------------
# ABTest — recording results
# ---------------------------------------------------------------------------


def test_record_result_control() -> None:
    exp = _make_experiment()
    exp.record_result("cost_optimized", cost=0.001, latency_ms=200.0)
    result = exp.get_result()
    assert result.control.sample_count == 1
    assert result.treatment.sample_count == 0


def test_record_result_treatment() -> None:
    exp = _make_experiment()
    exp.record_result("quality_optimized", cost=0.005, latency_ms=800.0)
    result = exp.get_result()
    assert result.treatment.sample_count == 1


def test_record_result_unknown_policy_ignored() -> None:
    exp = _make_experiment()
    # Should not raise
    exp.record_result("unknown_policy", cost=0.001)
    result = exp.get_result()
    assert result.control.sample_count == 0
    assert result.treatment.sample_count == 0


# ---------------------------------------------------------------------------
# ABTest — should_promote
# ---------------------------------------------------------------------------


def test_should_promote_not_enough_samples() -> None:
    exp = _make_experiment()
    exp.record_result("cost_optimized", cost=0.01)
    exp.record_result("quality_optimized", cost=0.001)
    assert exp.should_promote() is False  # only 1 sample each


def test_should_promote_no_improvement() -> None:
    config = ExperimentConfig(
        name="test",
        control_policy=RoutingPolicy.cost_optimized(),
        treatment_policy=RoutingPolicy.quality_optimized(),
        min_samples=5,
        improvement_threshold=0.10,
        auto_promote=False,
    )
    exp = ABTest(config, rng_seed=42)
    # Treatment is WORSE (higher cost)
    for _ in range(10):
        exp.record_result("cost_optimized", cost=0.001)
    for _ in range(10):
        exp.record_result("quality_optimized", cost=0.01)
    assert exp.should_promote() is False


def test_should_promote_significant_improvement() -> None:
    config = ExperimentConfig(
        name="test",
        control_policy=RoutingPolicy.cost_optimized(),
        treatment_policy=RoutingPolicy.quality_optimized(),
        min_samples=20,
        significance_level=0.05,
        improvement_threshold=0.10,
        auto_promote=False,
    )
    exp = ABTest(config, rng_seed=42)
    # Treatment is MUCH cheaper (50% reduction — clearly significant)
    import random
    rng = random.Random(99)
    for _ in range(30):
        exp.record_result("cost_optimized", cost=rng.gauss(0.010, 0.0005))
    for _ in range(30):
        exp.record_result("quality_optimized", cost=rng.gauss(0.005, 0.0005))
    assert exp.should_promote() is True


# ---------------------------------------------------------------------------
# ABTest — promote
# ---------------------------------------------------------------------------


def test_promote_sets_winner() -> None:
    exp = _make_experiment()
    promoted_policy = exp.promote()
    assert promoted_policy.name == "quality_optimized"
    assert exp.is_complete is True
    result = exp.get_result()
    assert result.winner == "treatment"
    assert result.promoted is True


def test_promote_locks_traffic_to_winner() -> None:
    exp = _make_experiment()
    exp.promote()
    # After promotion, all traffic should go to winner
    policies = [exp.split_traffic() for _ in range(10)]
    assert all(p.name == "quality_optimized" for p in policies)


# ---------------------------------------------------------------------------
# ABTest — auto-promote
# ---------------------------------------------------------------------------


def test_auto_promote_fires_when_conditions_met() -> None:
    config = ExperimentConfig(
        name="auto_test",
        control_policy=RoutingPolicy.cost_optimized(),
        treatment_policy=RoutingPolicy.quality_optimized(),
        min_samples=10,
        significance_level=0.05,
        improvement_threshold=0.10,
        auto_promote=True,
    )
    exp = ABTest(config, rng_seed=42)
    import random
    rng = random.Random(99)
    for _ in range(20):
        exp.record_result("cost_optimized", cost=rng.gauss(0.010, 0.0002))
    for _ in range(20):
        exp.record_result("quality_optimized", cost=rng.gauss(0.005, 0.0002))
    # Should have auto-promoted by now
    assert exp.is_complete is True


# ---------------------------------------------------------------------------
# ABTest — reset
# ---------------------------------------------------------------------------


def test_reset_clears_metrics() -> None:
    exp = _make_experiment()
    exp.record_result("cost_optimized", cost=0.001)
    exp.record_result("quality_optimized", cost=0.005)
    exp.reset()

    result = exp.get_result()
    assert result.control.sample_count == 0
    assert result.treatment.sample_count == 0
    assert exp.is_complete is False


# ---------------------------------------------------------------------------
# ExperimentResult
# ---------------------------------------------------------------------------


def test_experiment_result_to_dict() -> None:
    exp = _make_experiment()
    result = exp.get_result()
    d = result.to_dict()
    assert "name" in d
    assert "control" in d
    assert "treatment" in d
    assert "significant" in d
    assert "p_value" in d


def test_experiment_result_summary() -> None:
    exp = _make_experiment()
    result = exp.get_result()
    summary = result.summary()
    assert "Experiment" in summary
    assert "Control" in summary
    assert "Treatment" in summary
