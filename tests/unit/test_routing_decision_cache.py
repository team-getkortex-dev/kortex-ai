"""Tests for RoutingDecisionCache."""

from __future__ import annotations

import pytest

from kortex.core.types import RoutingDecision, TaskSpec
from kortex.router.decision_cache import RoutingDecisionCache


def _task(content: str = "test", complexity: str = "simple") -> TaskSpec:
    return TaskSpec(content=content, complexity_hint=complexity)  # type: ignore[call-arg]


def _decision(model: str = "gpt-4o", provider: str = "openai") -> RoutingDecision:
    return RoutingDecision(
        task_id="t1",
        chosen_provider=provider,
        chosen_model=model,
        chosen_model_identity=f"{provider}::{model}",
        reasoning="test",
        estimated_cost_usd=0.001,
        estimated_latency_ms=100.0,
    )


# ---------------------------------------------------------------------------
# Basic get / set
# ---------------------------------------------------------------------------


def test_cache_miss_returns_none() -> None:
    cache = RoutingDecisionCache()
    assert cache.get(_task(), None) is None


def test_cache_hit_returns_decision() -> None:
    cache = RoutingDecisionCache()
    task = _task()
    dec = _decision()
    cache.set(task, None, dec)
    result = cache.get(task, None)
    assert result is dec


def test_cache_hit_increments_counter() -> None:
    cache = RoutingDecisionCache()
    task = _task()
    cache.set(task, None, _decision())
    cache.get(task, None)
    assert cache.hits == 1
    assert cache.misses == 0


def test_cache_miss_increments_counter() -> None:
    cache = RoutingDecisionCache()
    cache.get(_task(), None)
    assert cache.hits == 0
    assert cache.misses == 1


def test_hit_rate_calculation() -> None:
    cache = RoutingDecisionCache()
    task = _task()
    cache.set(task, None, _decision())
    cache.get(task, None)   # hit
    cache.get(_task("x"), None)  # miss
    assert cache.hit_rate == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------


def test_lru_eviction_removes_oldest_entry() -> None:
    cache = RoutingDecisionCache(max_size=3)
    tasks = [_task(f"task{i}") for i in range(4)]
    dec = _decision()
    for t in tasks[:3]:
        cache.set(t, None, dec)

    # Access tasks[0] to make it MRU
    cache.get(tasks[0], None)

    # Add 4th — tasks[1] should be evicted (LRU)
    cache.set(tasks[3], None, dec)

    assert cache.get(tasks[1], None) is None
    assert cache.get(tasks[0], None) is not None
    assert cache.get(tasks[2], None) is not None
    assert cache.get(tasks[3], None) is not None


def test_lru_max_size_respected() -> None:
    cache = RoutingDecisionCache(max_size=5)
    dec = _decision()
    for i in range(10):
        cache.set(_task(f"t{i}"), None, dec)
    assert cache.size <= 5


# ---------------------------------------------------------------------------
# Policy differentiation
# ---------------------------------------------------------------------------


def test_different_policies_produce_different_keys() -> None:
    """Tasks with the same content but different policy names cache separately."""
    from kortex.core.policy import RoutingPolicy

    cache = RoutingDecisionCache()
    task = _task()
    dec1 = _decision("model-a")
    dec2 = _decision("model-b")

    policy1 = RoutingPolicy(name="p1")
    policy2 = RoutingPolicy(name="p2")

    cache.set(task, policy1, dec1)
    cache.set(task, policy2, dec2)

    assert cache.get(task, policy1) is dec1
    assert cache.get(task, policy2) is dec2


def test_none_policy_vs_policy_object_differ() -> None:
    from kortex.core.policy import RoutingPolicy

    cache = RoutingDecisionCache()
    task = _task()
    dec_no_policy = _decision("no-policy-model")
    dec_with_policy = _decision("policy-model")
    policy = RoutingPolicy(name="p1")

    cache.set(task, None, dec_no_policy)
    cache.set(task, policy, dec_with_policy)

    assert cache.get(task, None) is dec_no_policy
    assert cache.get(task, policy) is dec_with_policy


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------


def test_clear_resets_cache_and_counters() -> None:
    cache = RoutingDecisionCache()
    task = _task()
    cache.set(task, None, _decision())
    cache.get(task, None)

    cache.clear()

    assert cache.size == 0
    assert cache.hits == 0
    assert cache.misses == 0
    assert cache.get(task, None) is None


# ---------------------------------------------------------------------------
# Integration with Router
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_router_uses_decision_cache() -> None:
    """Second identical route() call returns cached decision."""
    from kortex.core.router import Router
    from kortex.core.router import ProviderModel

    router = Router(enable_decision_cache=True)
    router.register_model(ProviderModel(
        provider="openai", model="gpt-4o-mini",
        cost_per_1k_input_tokens=0.0, cost_per_1k_output_tokens=0.0,
        avg_latency_ms=100.0, tier="fast",
    ))

    task = _task("hello")
    d1 = await router.route(task)
    d2 = await router.route(task)

    assert d1 is d2  # same object from cache
    assert router._decision_cache is not None
    assert router._decision_cache.hits == 1


@pytest.mark.asyncio
async def test_router_with_cache_disabled() -> None:
    from kortex.core.router import Router, ProviderModel

    router = Router(enable_decision_cache=False)
    assert router._decision_cache is None

    router.register_model(ProviderModel(
        provider="openai", model="gpt-4o-mini",
        cost_per_1k_input_tokens=0.0, cost_per_1k_output_tokens=0.0,
        avg_latency_ms=100.0, tier="fast",
    ))
    task = _task("hello")
    d1 = await router.route(task)
    d2 = await router.route(task)
    # Different objects since no cache
    assert d1 is not d2
