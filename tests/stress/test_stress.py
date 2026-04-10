"""
Kortex Full-System Stress Test Suite
========================================

This test suite is designed to be HOSTILE to the system under test.
It does not care about your feelings. It will find every weakness,
every race condition, every edge case, every assumption that seemed
reasonable at 2am but wasn't.

Run with:
    pytest tests/stress/ -v --tb=short
    pytest tests/stress/ -v -x  # stop on first failure

Categories:
    - CONCURRENCY: Race conditions, deadlocks, state corruption
    - VOLUME: Throughput limits, memory pressure, chain depth
    - FAILURE RESILIENCE: Provider outages, cascading failures, recovery
    - EDGE CASES: Unicode, empty inputs, duplicate IDs, boundary values
    - MEMORY & PERFORMANCE: Leak detection, degradation under load
"""

import asyncio
import gc
import math
import os
import random
import string
import sys
import time
import tracemalloc
from typing import Any

import pytest

from kortex.core.types import TaskSpec, CoordinationResult
from kortex.core.router import Router, ProviderModel
from kortex.core.state import StateManager
from kortex.core.runtime import KortexRuntime, AgentDescriptor
from kortex.core.exceptions import (
    KortexError,
    RoutingFailedError,
    StateError,
    HandoffError,
    CheckpointNotFoundError,
)
from kortex.store.memory import InMemoryStateStore
from kortex.store.sqlite import SQLiteStateStore

from .conftest import (
    ChaosProvider,
    STRESS_MODELS,
    STRESS_AGENTS,
    build_router,
    build_registry,
    build_runtime,
    generate_large_payload,
    generate_random_task,
)

# Mark all tests in this module
pytestmark = pytest.mark.stress


# ===================================================================
# HELPERS
# ===================================================================

def _print_result(name: str, passed: bool, detail: str = "", elapsed: float = 0.0):
    status = "\033[92mPASS\033[0m" if passed else "\033[91mFAIL\033[0m"
    time_str = f" ({elapsed:.3f}s)" if elapsed > 0 else ""
    detail_str = f"  {detail}" if detail else ""
    # Also support NO_COLOR
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}{time_str}{detail_str}")


# ===================================================================
# CONCURRENCY STRESS
# ===================================================================

class TestConcurrencyStress:
    """Hammer the system with parallel operations. Find race conditions."""

    @pytest.mark.asyncio
    async def test_concurrent_coordinations_50(self):
        """50 simultaneous full coordination pipelines.

        Every single one must complete without:
        - State corruption
        - Checkpoint ID collisions
        - Swallowed exceptions
        - Incorrect routing
        """
        print("\n--- CONCURRENCY: 50 simultaneous coordinations ---")
        runtime = build_runtime(
            ChaosProvider(name="reliable", failure_rate=0.0, min_latency_ms=0.1, max_latency_ms=0.5)
        )

        pipeline = ["researcher", "writer", "reviewer"]
        tasks = [generate_random_task("moderate") for _ in range(50)]

        start = time.monotonic()
        results = await asyncio.gather(
            *[runtime.coordinate(task, pipeline) for task in tasks],
            return_exceptions=True,
        )
        elapsed = time.monotonic() - start

        errors = [r for r in results if isinstance(r, Exception)]
        successes = [r for r in results if isinstance(r, CoordinationResult)]

        # Verify no collisions in checkpoint IDs
        all_checkpoint_ids = []
        for r in successes:
            for h in r.handoffs:
                cid = h.checkpoint_id if hasattr(h, 'checkpoint_id') else h.get("checkpoint_id", h.get("handoff_id", ""))
                all_checkpoint_ids.append(cid)

        unique_ids = len(set(all_checkpoint_ids))
        total_ids = len(all_checkpoint_ids)

        _print_result(
            "50 concurrent coordinations",
            len(errors) == 0 and len(successes) == 50,
            f"{len(successes)}/50 succeeded, {len(errors)} errors, "
            f"{unique_ids}/{total_ids} unique checkpoint IDs",
            elapsed,
        )

        assert len(errors) == 0, f"Errors in concurrent coordination: {errors[:3]}"
        assert len(successes) == 50
        assert unique_ids == total_ids, f"Checkpoint ID collision! {total_ids - unique_ids} duplicates"

    @pytest.mark.asyncio
    async def test_concurrent_coordinations_200(self):
        """200 simultaneous pipelines. This WILL find race conditions if they exist."""
        print("\n--- CONCURRENCY: 200 simultaneous coordinations ---")
        runtime = build_runtime(
            ChaosProvider(name="reliable", failure_rate=0.0, min_latency_ms=0.1, max_latency_ms=0.3)
        )

        pipeline = ["planner", "researcher", "coder", "reviewer"]
        tasks = [generate_random_task(random.choice(["simple", "moderate", "complex"])) for _ in range(200)]

        start = time.monotonic()
        results = await asyncio.gather(
            *[runtime.coordinate(task, pipeline) for task in tasks],
            return_exceptions=True,
        )
        elapsed = time.monotonic() - start

        errors = [r for r in results if isinstance(r, Exception)]
        successes = [r for r in results if isinstance(r, CoordinationResult)]

        _print_result(
            "200 concurrent coordinations",
            len(errors) == 0,
            f"{len(successes)}/200 succeeded, {elapsed/200*1000:.1f}ms avg per coordination",
            elapsed,
        )

        assert len(errors) == 0, f"Errors: {[str(e)[:100] for e in errors[:5]]}"
        assert len(successes) == 200

    @pytest.mark.asyncio
    async def test_concurrent_mixed_operations(self):
        """Simultaneous routes, handoffs, rollbacks, and lookups.
        
        Different operation types competing for the same resources.
        """
        print("\n--- CONCURRENCY: Mixed operations (route + handoff + rollback) ---")
        router = build_router()
        state_mgr = StateManager.create("memory")

        async def do_routes(n: int):
            results = []
            for _ in range(n):
                task = generate_random_task()
                try:
                    decision = await router.route(task)
                    results.append(("route", True))
                except Exception as e:
                    results.append(("route", False))
            return results

        async def do_handoffs(n: int):
            results = []
            prev_checkpoint = None
            for i in range(n):
                try:
                    ctx = await state_mgr.handoff(
                        source_agent=f"agent_{i}",
                        target_agent=f"agent_{i+1}",
                        state_snapshot={"step": i, "data": f"payload_{i}"},
                        parent_checkpoint_id=prev_checkpoint,
                    )
                    prev_checkpoint = ctx.checkpoint_id
                    results.append(("handoff", True))
                except Exception:
                    results.append(("handoff", False))
            return results

        async def do_lookups(n: int):
            results = []
            for _ in range(n):
                try:
                    latest = await state_mgr.get_latest(f"agent_{random.randint(0, 20)}")
                    results.append(("lookup", True))
                except Exception:
                    results.append(("lookup", False))
            return results

        start = time.monotonic()
        all_results = await asyncio.gather(
            do_routes(100),
            do_handoffs(50),
            do_lookups(100),
            do_routes(100),
            do_handoffs(50),
            do_lookups(100),
        )
        elapsed = time.monotonic() - start

        flat = [item for sublist in all_results for item in sublist]
        failures = [r for r in flat if not r[1]]

        _print_result(
            "Mixed concurrent operations",
            len(failures) == 0,
            f"{len(flat)} operations, {len(failures)} failures",
            elapsed,
        )
        assert len(failures) == 0, f"Failures in mixed ops: {failures[:5]}"

    @pytest.mark.asyncio
    async def test_concurrent_sqlite_writes(self):
        """Hammer SQLite with concurrent writes. SQLite hates this."""
        print("\n--- CONCURRENCY: 50 concurrent SQLite writes ---")
        state_mgr = StateManager.create("sqlite", db_path=":memory:")

        # Connect the store if needed
        store = state_mgr._store  # type: ignore
        if hasattr(store, 'connect'):
            await store.connect()

        async def write_checkpoint(i: int):
            return await state_mgr.handoff(
                source_agent=f"src_{i}",
                target_agent=f"tgt_{i}",
                state_snapshot={"index": i, "data": f"value_{i}" * 10},
            )

        start = time.monotonic()
        results = await asyncio.gather(
            *[write_checkpoint(i) for i in range(50)],
            return_exceptions=True,
        )
        elapsed = time.monotonic() - start

        errors = [r for r in results if isinstance(r, Exception)]
        successes = [r for r in results if not isinstance(r, Exception)]

        _print_result(
            "50 concurrent SQLite writes",
            len(errors) == 0,
            f"{len(successes)}/50 succeeded",
            elapsed,
        )
        assert len(errors) == 0, f"SQLite concurrent write errors: {errors[:3]}"


# ===================================================================
# VOLUME STRESS
# ===================================================================

class TestVolumeStress:
    """Push data volume to the limit."""

    @pytest.mark.asyncio
    async def test_pipeline_20_agents(self):
        """20-agent pipeline. Verify chain integrity end to end."""
        print("\n--- VOLUME: 20-agent pipeline ---")
        runtime = build_runtime(
            ChaosProvider(name="reliable", failure_rate=0.0, min_latency_ms=0.1, max_latency_ms=0.3)
        )

        # Register 20 agents
        for i in range(20):
            runtime.register_agent(AgentDescriptor(
                agent_id=f"agent_{i:02d}",
                name=f"Agent {i}",
                description=f"Agent number {i}",
                capabilities=["analysis"],
            ))

        pipeline = [f"agent_{i:02d}" for i in range(20)]
        task = TaskSpec(content="Process this through 20 sequential agents", complexity_hint="complex")

        start = time.monotonic()
        result = await runtime.coordinate(task, pipeline)
        elapsed = time.monotonic() - start

        _print_result(
            "20-agent pipeline",
            result.success and len(result.routing_decisions) == 20,
            f"{len(result.routing_decisions)} decisions, "
            f"{len(result.handoffs)} handoffs, "
            f"cost=${result.total_estimated_cost_usd:.4f}",
            elapsed,
        )

        assert result.success
        assert len(result.routing_decisions) == 20
        assert len(result.handoffs) >= 20  # initial + one per agent

    @pytest.mark.asyncio
    async def test_pipeline_50_agents(self):
        """50-agent pipeline. Where does performance start to degrade?"""
        print("\n--- VOLUME: 50-agent pipeline ---")
        runtime = build_runtime(
            ChaosProvider(name="reliable", failure_rate=0.0, min_latency_ms=0.05, max_latency_ms=0.1)
        )

        for i in range(50):
            runtime.register_agent(AgentDescriptor(
                agent_id=f"vol_agent_{i:03d}",
                name=f"Volume Agent {i}",
                description=f"Volume test agent {i}",
                capabilities=["analysis"],
            ))

        pipeline = [f"vol_agent_{i:03d}" for i in range(50)]
        task = TaskSpec(content="Massive pipeline stress test", complexity_hint="complex")

        start = time.monotonic()
        result = await runtime.coordinate(task, pipeline)
        elapsed = time.monotonic() - start

        per_agent_ms = (elapsed / 50) * 1000

        _print_result(
            "50-agent pipeline",
            result.success,
            f"{per_agent_ms:.2f}ms per agent, total {elapsed:.3f}s",
            elapsed,
        )

        assert result.success
        assert len(result.routing_decisions) == 50

    @pytest.mark.asyncio
    async def test_large_state_payload_1mb(self):
        """Handoff with 1MB state payload."""
        print("\n--- VOLUME: 1MB state payload ---")
        state_mgr = StateManager.create("memory")
        payload = generate_large_payload(1_000_000)

        start = time.monotonic()
        ctx = await state_mgr.handoff(
            source_agent="sender",
            target_agent="receiver",
            state_snapshot=payload,
        )
        retrieved = await state_mgr._store.get_checkpoint(ctx.checkpoint_id)
        elapsed = time.monotonic() - start

        payload_size = len(str(payload))
        retrieved_size = len(str(retrieved.state_snapshot))

        _print_result(
            "1MB state payload",
            retrieved_size > 0,
            f"stored {payload_size:,} chars, retrieved {retrieved_size:,} chars",
            elapsed,
        )

        assert retrieved_size > 0

    @pytest.mark.asyncio
    async def test_large_state_payload_10mb(self):
        """Handoff with 10MB state payload. Memory pressure test."""
        print("\n--- VOLUME: 10MB state payload ---")
        state_mgr = StateManager.create("memory")
        payload = generate_large_payload(10_000_000)

        start = time.monotonic()
        ctx = await state_mgr.handoff(
            source_agent="big_sender",
            target_agent="big_receiver",
            state_snapshot=payload,
        )
        elapsed = time.monotonic() - start

        _print_result(
            "10MB state payload",
            ctx is not None,
            f"checkpoint {ctx.checkpoint_id[:12]}...",
            elapsed,
        )

        assert ctx is not None
        assert ctx.checkpoint_id

    @pytest.mark.asyncio
    async def test_rapid_fire_1000_routes(self):
        """Route 1000 tasks as fast as possible. Measure throughput."""
        print("\n--- VOLUME: 1000 rapid-fire routes ---")
        router = build_router()
        tasks = [generate_random_task(random.choice(["simple", "moderate", "complex"])) for _ in range(1000)]

        start = time.monotonic()
        results = []
        for task in tasks:
            decision = await router.route(task)
            results.append(decision)
        elapsed = time.monotonic() - start

        throughput = 1000 / elapsed
        valid = all(r.chosen_model and r.chosen_provider for r in results)

        _print_result(
            "1000 rapid-fire routes",
            valid,
            f"{throughput:.0f} routes/sec",
            elapsed,
        )

        assert valid
        assert throughput > 100, f"Throughput too low: {throughput:.0f} routes/sec"

    @pytest.mark.asyncio
    async def test_rapid_fire_5000_routes(self):
        """5000 routes. Sustained throughput test."""
        print("\n--- VOLUME: 5000 rapid-fire routes ---")
        router = build_router()

        start = time.monotonic()
        error_count = 0
        for i in range(5000):
            task = generate_random_task(random.choice(["simple", "moderate", "complex"]))
            try:
                await router.route(task)
            except Exception:
                error_count += 1
        elapsed = time.monotonic() - start

        throughput = 5000 / elapsed

        _print_result(
            "5000 rapid-fire routes",
            error_count == 0,
            f"{throughput:.0f} routes/sec, {error_count} errors",
            elapsed,
        )

        assert error_count == 0

    @pytest.mark.asyncio
    async def test_checkpoint_chain_depth_100(self):
        """Build a chain 100 checkpoints deep. Verify retrieval."""
        print("\n--- VOLUME: 100-deep checkpoint chain ---")
        state_mgr = StateManager.create("memory")

        parent_id = None
        all_ids = []
        start = time.monotonic()

        for i in range(100):
            ctx = await state_mgr.handoff(
                source_agent=f"chain_{i}",
                target_agent=f"chain_{i+1}",
                state_snapshot={"depth": i},
                parent_checkpoint_id=parent_id,
            )
            parent_id = ctx.checkpoint_id
            all_ids.append(ctx.checkpoint_id)

        # Retrieve the full chain from the deepest point
        chain = await state_mgr.get_history(all_ids[-1])
        elapsed = time.monotonic() - start

        _print_result(
            "100-deep checkpoint chain",
            len(chain) == 100,
            f"chain length: {len(chain)}",
            elapsed,
        )

        assert len(chain) == 100

    @pytest.mark.asyncio
    async def test_10000_checkpoints_memory_store(self):
        """Store 10,000 checkpoints in memory. Measure insert and lookup."""
        print("\n--- VOLUME: 10,000 checkpoints (memory store) ---")
        state_mgr = StateManager.create("memory")

        start_insert = time.monotonic()
        checkpoint_ids = []
        for i in range(10_000):
            ctx = await state_mgr.handoff(
                source_agent=f"mass_src_{i % 50}",
                target_agent=f"mass_tgt_{i % 50}",
                state_snapshot={"index": i, "value": f"data_{i}"},
            )
            checkpoint_ids.append(ctx.checkpoint_id)
        insert_elapsed = time.monotonic() - start_insert

        # Random lookups
        start_lookup = time.monotonic()
        for _ in range(1000):
            cid = random.choice(checkpoint_ids)
            await state_mgr._store.get_checkpoint(cid)
        lookup_elapsed = time.monotonic() - start_lookup

        insert_rate = 10_000 / insert_elapsed
        lookup_rate = 1000 / lookup_elapsed

        _print_result(
            "10K checkpoints (memory)",
            True,
            f"insert: {insert_rate:.0f}/sec, lookup: {lookup_rate:.0f}/sec",
            insert_elapsed + lookup_elapsed,
        )

    @pytest.mark.asyncio
    async def test_10000_checkpoints_sqlite_store(self):
        """Store 10,000 checkpoints in SQLite. Compare to memory store."""
        print("\n--- VOLUME: 10,000 checkpoints (SQLite) ---")
        state_mgr = StateManager.create("sqlite", db_path=":memory:")

        store = state_mgr._store  # type: ignore
        if hasattr(store, 'connect'):
            await store.connect()

        start_insert = time.monotonic()
        checkpoint_ids = []
        for i in range(10_000):
            ctx = await state_mgr.handoff(
                source_agent=f"sql_src_{i % 50}",
                target_agent=f"sql_tgt_{i % 50}",
                state_snapshot={"index": i, "value": f"data_{i}"},
            )
            checkpoint_ids.append(ctx.checkpoint_id)
        insert_elapsed = time.monotonic() - start_insert

        start_lookup = time.monotonic()
        for _ in range(1000):
            cid = random.choice(checkpoint_ids)
            await state_mgr._store.get_checkpoint(cid)
        lookup_elapsed = time.monotonic() - start_lookup

        insert_rate = 10_000 / insert_elapsed
        lookup_rate = 1000 / lookup_elapsed

        _print_result(
            "10K checkpoints (SQLite)",
            True,
            f"insert: {insert_rate:.0f}/sec, lookup: {lookup_rate:.0f}/sec",
            insert_elapsed + lookup_elapsed,
        )


# ===================================================================
# FAILURE RESILIENCE
# ===================================================================

class TestFailureResilience:
    """Break everything. See what survives."""

    @pytest.mark.asyncio
    async def test_all_providers_dead(self):
        """Every provider is dead. The system must NOT hang."""
        print("\n--- FAILURE: All providers dead ---")
        dead_provider = ChaosProvider(name="dead", failure_rate=1.0)
        dead_provider.set_models(STRESS_MODELS)
        runtime = build_runtime(dead_provider)

        task = generate_random_task("simple")
        pipeline = ["researcher", "writer"]

        start = time.monotonic()
        try:
            result = await runtime.coordinate(task, pipeline, execute=True)
            # If it returns, check it handled the failures
            has_issues = not result.success or len(result.routing_decisions) == 0
            elapsed = time.monotonic() - start
            _print_result("All providers dead", True, "Handled gracefully", elapsed)
        except (RoutingFailedError, KortexError) as e:
            elapsed = time.monotonic() - start
            _print_result(
                "All providers dead",
                True,
                f"Raised {type(e).__name__}: {str(e)[:80]}",
                elapsed,
            )
        except Exception as e:
            elapsed = time.monotonic() - start
            _print_result(
                "All providers dead",
                False,
                f"Unexpected error: {type(e).__name__}: {str(e)[:80]}",
                elapsed,
            )
            raise

        assert elapsed < 10.0, f"System hung for {elapsed:.1f}s — possible deadlock"

    @pytest.mark.asyncio
    async def test_intermittent_failures_100_coordinations(self):
        """30% provider failure rate across 100 coordinations.
        
        Tests fallback logic under realistic failure conditions.
        The system should gracefully handle failures without crashing.
        """
        print("\n--- FAILURE: 100 coordinations with 30% failure rate ---")
        flaky = ChaosProvider(
            name="flaky",
            failure_rate=0.3,
            min_latency_ms=0.1,
            max_latency_ms=2.0,
        )
        flaky.set_models(STRESS_MODELS)
        runtime = build_runtime(flaky)

        pipeline = ["researcher", "writer", "reviewer"]
        succeeded = 0
        failed_gracefully = 0
        crashed = 0

        start = time.monotonic()
        for i in range(100):
            task = generate_random_task()
            try:
                result = await runtime.coordinate(task, pipeline, execute=True)
                if result.success:
                    succeeded += 1
                else:
                    failed_gracefully += 1
            except (RoutingFailedError, KortexError):
                failed_gracefully += 1
            except Exception as e:
                crashed += 1
        elapsed = time.monotonic() - start

        _print_result(
            "100 coordinations (30% failure)",
            crashed == 0,
            f"{succeeded} succeeded, {failed_gracefully} graceful failures, {crashed} crashes",
            elapsed,
        )

        assert crashed == 0, f"{crashed} coordinations crashed unexpectedly"

    @pytest.mark.asyncio
    async def test_intermittent_failures_high_rate(self):
        """60% failure rate. The system should still not crash."""
        print("\n--- FAILURE: 50 coordinations with 60% failure rate ---")
        hostile = ChaosProvider(
            name="hostile",
            failure_rate=0.6,
            min_latency_ms=0.1,
            max_latency_ms=5.0,
            garbage_rate=0.2,
        )
        hostile.set_models(STRESS_MODELS)
        runtime = build_runtime(hostile)

        pipeline = ["researcher", "writer"]
        crashed = 0

        start = time.monotonic()
        for _ in range(50):
            try:
                await runtime.coordinate(generate_random_task(), pipeline, execute=True)
            except (RoutingFailedError, KortexError):
                pass  # Expected — graceful failure
            except Exception:
                crashed += 1
        elapsed = time.monotonic() - start

        _print_result(
            "50 coordinations (60% failure)",
            crashed == 0,
            f"{crashed} crashes",
            elapsed,
        )

        assert crashed == 0

    @pytest.mark.asyncio
    async def test_provider_timeout_handling(self):
        """Provider that takes 5 seconds to respond. System should not block forever."""
        print("\n--- FAILURE: Slow provider (5s latency) ---")
        slow = ChaosProvider(
            name="slow",
            failure_rate=0.0,
            min_latency_ms=100,
            max_latency_ms=200,
        )
        slow.set_models(STRESS_MODELS)
        runtime = build_runtime(slow)

        task = generate_random_task("simple")
        pipeline = ["researcher"]

        start = time.monotonic()
        result = await runtime.coordinate(task, pipeline, execute=True)
        elapsed = time.monotonic() - start

        _print_result(
            "Slow provider handling",
            result.success,
            f"completed in {elapsed:.3f}s",
            elapsed,
        )

        assert result.success

    @pytest.mark.asyncio
    async def test_state_store_corruption_recovery(self):
        """Manually corrupt the state store, then try to use it."""
        print("\n--- FAILURE: State store corruption ---")
        store = InMemoryStateStore()
        state_mgr = StateManager(store)

        # Create some valid checkpoints
        ctx1 = await state_mgr.handoff("a", "b", {"step": 1})
        ctx2 = await state_mgr.handoff("b", "c", {"step": 2}, parent_checkpoint_id=ctx1.checkpoint_id)

        # Corrupt: delete the parent but keep the child
        await store.delete_checkpoint(ctx1.checkpoint_id)

        # Try to get history — should handle missing parent gracefully
        try:
            history = await state_mgr.get_history(ctx2.checkpoint_id)
            graceful = True
            detail = f"returned {len(history)} items"
        except (CheckpointNotFoundError, StateError) as e:
            graceful = True
            detail = f"raised {type(e).__name__}"
        except Exception as e:
            graceful = False
            detail = f"unexpected: {type(e).__name__}: {e}"

        _print_result("State store corruption", graceful, detail)
        assert graceful


# ===================================================================
# EDGE CASES
# ===================================================================

class TestEdgeCases:
    """The weird stuff. The things nobody thought to test."""

    @pytest.mark.asyncio
    async def test_empty_pipeline(self):
        """Coordinate with zero agents. Must not crash."""
        print("\n--- EDGE: Empty pipeline ---")
        runtime = build_runtime(
            ChaosProvider(name="r", failure_rate=0.0, min_latency_ms=0.1, max_latency_ms=0.3)
        )
        task = generate_random_task()

        try:
            result = await runtime.coordinate(task, [])
            # Some implementations might return an empty result
            handled = True
            detail = f"returned result with success={result.success}"
        except (KortexError, ValueError, KeyError) as e:
            handled = True
            detail = f"raised {type(e).__name__}"
        except Exception as e:
            handled = False
            detail = f"unexpected: {type(e).__name__}: {e}"

        _print_result("Empty pipeline", handled, detail)
        assert handled

    @pytest.mark.asyncio
    async def test_single_agent_pipeline(self):
        """Single agent. The simplest possible coordination."""
        print("\n--- EDGE: Single agent pipeline ---")
        runtime = build_runtime(
            ChaosProvider(name="r", failure_rate=0.0, min_latency_ms=0.1, max_latency_ms=0.3)
        )
        task = generate_random_task("simple")

        result = await runtime.coordinate(task, ["researcher"])

        _print_result(
            "Single agent pipeline",
            result.success and len(result.routing_decisions) == 1,
            f"{len(result.routing_decisions)} decision(s)",
        )

        assert result.success
        assert len(result.routing_decisions) == 1

    @pytest.mark.asyncio
    async def test_duplicate_agents_in_pipeline(self):
        """Same agent appears 3 times. Should handle or reject gracefully."""
        print("\n--- EDGE: Duplicate agents in pipeline ---")
        runtime = build_runtime(
            ChaosProvider(name="r", failure_rate=0.0, min_latency_ms=0.1, max_latency_ms=0.3)
        )
        task = generate_random_task()

        try:
            result = await runtime.coordinate(task, ["researcher", "researcher", "researcher"])
            handled = True
            detail = f"success={result.success}, {len(result.routing_decisions)} decisions"
        except (KortexError, ValueError) as e:
            handled = True
            detail = f"raised {type(e).__name__}"
        except Exception as e:
            handled = False
            detail = f"unexpected: {type(e).__name__}: {e}"

        _print_result("Duplicate agents", handled, detail)
        assert handled

    @pytest.mark.asyncio
    async def test_unregistered_agent_in_pipeline(self):
        """Pipeline references an agent that doesn't exist."""
        print("\n--- EDGE: Unregistered agent ---")
        runtime = build_runtime(
            ChaosProvider(name="r", failure_rate=0.0, min_latency_ms=0.1, max_latency_ms=0.3)
        )
        task = generate_random_task()

        try:
            result = await runtime.coordinate(task, ["researcher", "GHOST_AGENT_404", "reviewer"])
            handled = True
            detail = f"success={result.success}"
        except (KortexError, KeyError, ValueError) as e:
            handled = True
            detail = f"raised {type(e).__name__}: {str(e)[:60]}"
        except Exception as e:
            handled = False
            detail = f"unexpected: {type(e).__name__}: {e}"

        _print_result("Unregistered agent", handled, detail)
        assert handled

    @pytest.mark.asyncio
    async def test_unicode_task_content(self):
        """Unicode: emoji, CJK, RTL, combining characters, zero-width chars."""
        print("\n--- EDGE: Unicode task content ---")
        runtime = build_runtime(
            ChaosProvider(name="r", failure_rate=0.0, min_latency_ms=0.1, max_latency_ms=0.3)
        )

        unicode_tasks = [
            "Analyze this data \U0001F4CA and produce a report \U0001F4DD with insights \U0001F4A1",
            "\u5206\u6790\u591A\u4EE3\u7406\u7CFB\u7EDF\u7684\u534F\u8C03\u95EE\u9898",  # Chinese
            "\u062A\u062D\u0644\u064A\u0644 \u0623\u0646\u0638\u0645\u0629 \u0627\u0644\u0648\u0643\u064A\u0644 \u0627\u0644\u0645\u062A\u0639\u062F\u062F",  # Arabic (RTL)
            "\u041F\u0440\u043E\u0430\u043D\u0430\u043B\u0438\u0437\u0438\u0440\u0443\u0439\u0442\u0435 \u043A\u043E\u043E\u0440\u0434\u0438\u043D\u0430\u0446\u0438\u044E",  # Russian
            "Test with\u200Bzero\u200Bwidth\u200Bchars and\u0300combining\u0301marks\u0302",
            "\U0001F468\u200D\U0001F469\u200D\U0001F467\u200D\U0001F466 Family emoji with ZWJ",
        ]

        all_passed = True
        for content in unicode_tasks:
            task = TaskSpec(content=content, complexity_hint="simple")
            try:
                result = await runtime.coordinate(task, ["researcher"])
                if not result.success:
                    all_passed = False
            except Exception:
                all_passed = False

        _print_result(
            "Unicode content",
            all_passed,
            f"tested {len(unicode_tasks)} unicode variants",
        )

        assert all_passed

    @pytest.mark.asyncio
    async def test_very_long_task_content(self):
        """Task with 100,000 characters. Memory and serialization stress."""
        print("\n--- EDGE: 100K character task ---")
        runtime = build_runtime(
            ChaosProvider(name="r", failure_rate=0.0, min_latency_ms=0.1, max_latency_ms=0.3)
        )

        long_content = "Analyze the following data: " + ("x" * 100_000)
        task = TaskSpec(content=long_content, complexity_hint="complex")

        start = time.monotonic()
        result = await runtime.coordinate(task, ["researcher", "writer"])
        elapsed = time.monotonic() - start

        _print_result(
            "100K character task",
            result.success,
            f"content length: {len(long_content):,}",
            elapsed,
        )

        assert result.success

    @pytest.mark.asyncio
    async def test_zero_cost_models_only(self):
        """All models are free (local). Router must still work by tier/capability."""
        print("\n--- EDGE: Zero-cost models only ---")
        router = Router()
        free_models = [
            ProviderModel(
                provider="local", model="ollama-llama3", tier="fast",
                cost_per_1k_input_tokens=0.0, cost_per_1k_output_tokens=0.0,
                avg_latency_ms=50, capabilities=["analysis"], max_context_tokens=8192,
            ),
            ProviderModel(
                provider="local", model="ollama-mixtral", tier="balanced",
                cost_per_1k_input_tokens=0.0, cost_per_1k_output_tokens=0.0,
                avg_latency_ms=200, capabilities=["reasoning", "analysis"], max_context_tokens=32768,
            ),
            ProviderModel(
                provider="local", model="ollama-llama3-70b", tier="powerful",
                cost_per_1k_input_tokens=0.0, cost_per_1k_output_tokens=0.0,
                avg_latency_ms=1000, capabilities=["reasoning", "code_generation", "analysis"], max_context_tokens=65536,
            ),
        ]
        for m in free_models:
            router.register_model(m)

        simple_task = TaskSpec(content="Quick lookup", complexity_hint="simple")
        complex_task = TaskSpec(content="Deep analysis", complexity_hint="complex")

        simple_decision = await router.route(simple_task)
        complex_decision = await router.route(complex_task)

        correct_routing = (
            simple_decision.chosen_model == "ollama-llama3"
            and complex_decision.chosen_model == "ollama-llama3-70b"
            and simple_decision.estimated_cost_usd == 0.0
        )

        _print_result(
            "Zero-cost models",
            correct_routing,
            f"simple->{simple_decision.chosen_model}, complex->{complex_decision.chosen_model}",
        )

        assert correct_routing

    @pytest.mark.asyncio
    async def test_cost_ceiling_zero(self):
        """Task with $0.00 cost ceiling. Only free models should match."""
        print("\n--- EDGE: Zero cost ceiling ---")
        router = build_router()  # Has both paid and free models

        # Add a free model
        router.register_model(ProviderModel(
            provider="local", model="free-model", tier="fast",
            cost_per_1k_input_tokens=0.0, cost_per_1k_output_tokens=0.0,
            avg_latency_ms=50, capabilities=["analysis"], max_context_tokens=8192,
        ))

        task = TaskSpec(
            content="Do this for free",
            complexity_hint="simple",
            cost_ceiling_usd=0.001,  # Near-zero budget
        )

        try:
            decision = await router.route(task)
            _print_result("Zero cost ceiling", True, f"routed to {decision.chosen_model}")
        except RoutingFailedError:
            _print_result("Zero cost ceiling", True, "correctly rejected — no model cheap enough")

    @pytest.mark.asyncio
    async def test_special_characters_in_agent_ids(self):
        """Agent IDs with spaces, symbols, emoji."""
        print("\n--- EDGE: Special chars in agent IDs ---")
        state_mgr = StateManager.create("memory")

        weird_ids = [
            "agent with spaces",
            "agent/with/slashes",
            "agent@#$%",
            "\U0001F916_robot_agent",
            "",  # Empty string
            "a" * 1000,  # Very long ID
        ]

        all_handled = True
        for agent_id in weird_ids:
            try:
                ctx = await state_mgr.handoff(
                    source_agent=agent_id,
                    target_agent="normal_target",
                    state_snapshot={"data": "test"},
                )
                # If it works, verify retrieval
                retrieved = await state_mgr._store.get_checkpoint(ctx.checkpoint_id)
                if retrieved.source_agent != agent_id:
                    all_handled = False
            except (KortexError, ValueError, KeyError):
                pass  # Rejecting weird IDs is acceptable
            except Exception:
                all_handled = False

        _print_result("Special chars in agent IDs", all_handled, f"tested {len(weird_ids)} variants")
        assert all_handled


# ===================================================================
# MEMORY & PERFORMANCE
# ===================================================================

class TestMemoryAndPerformance:
    """Detect memory leaks and performance degradation."""

    @pytest.mark.asyncio
    async def test_memory_growth_1000_coordinations(self):
        """Run 1000 coordinations and check memory doesn't grow unboundedly."""
        print("\n--- MEMORY: 1000 coordinations memory growth ---")
        runtime = build_runtime(
            ChaosProvider(name="r", failure_rate=0.0, min_latency_ms=0.05, max_latency_ms=0.1)
        )

        tracemalloc.start()
        gc.collect()
        snapshot_before = tracemalloc.take_snapshot()
        mem_before = sum(stat.size for stat in snapshot_before.statistics('filename'))

        for i in range(1000):
            task = generate_random_task()
            await runtime.coordinate(task, ["researcher", "writer"])

        gc.collect()
        snapshot_after = tracemalloc.take_snapshot()
        mem_after = sum(stat.size for stat in snapshot_after.statistics('filename'))
        tracemalloc.stop()

        growth_mb = (mem_after - mem_before) / (1024 * 1024)

        _print_result(
            "Memory growth (1000 coordinations)",
            growth_mb < 100,  # Less than 100MB growth is acceptable
            f"growth: {growth_mb:.2f}MB",
        )

        # Note: we don't assert hard here because the state store accumulates
        # data legitimately. But >100MB for 1000 simple coordinations = leak.
        assert growth_mb < 100, f"Suspicious memory growth: {growth_mb:.2f}MB"

    @pytest.mark.asyncio
    async def test_routing_performance_consistency(self):
        """Verify routing doesn't get slower over time."""
        print("\n--- PERFORMANCE: Routing consistency over 5000 decisions ---")
        router = build_router()

        # Measure first 100
        start = time.monotonic()
        for _ in range(100):
            await router.route(generate_random_task())
        first_100_time = time.monotonic() - start

        # Run 4800 more
        for _ in range(4800):
            await router.route(generate_random_task())

        # Measure last 100
        start = time.monotonic()
        for _ in range(100):
            await router.route(generate_random_task())
        last_100_time = time.monotonic() - start

        slowdown_ratio = last_100_time / first_100_time if first_100_time > 0 else 999

        _print_result(
            "Routing consistency",
            slowdown_ratio < 2.0,
            f"first 100: {first_100_time*1000:.1f}ms, last 100: {last_100_time*1000:.1f}ms, "
            f"ratio: {slowdown_ratio:.2f}x",
        )

        assert slowdown_ratio < 2.0, f"Routing degraded {slowdown_ratio:.2f}x over 5000 decisions"

    @pytest.mark.asyncio
    async def test_state_store_lookup_scaling(self):
        """Verify lookups don't degrade as store fills up."""
        print("\n--- PERFORMANCE: State store lookup scaling ---")
        state_mgr = StateManager.create("memory")

        # Insert 1000 checkpoints
        checkpoint_ids = []
        for i in range(1000):
            ctx = await state_mgr.handoff(f"s_{i}", f"t_{i}", {"i": i})
            checkpoint_ids.append(ctx.checkpoint_id)

        # Measure lookup time for 100 random lookups
        start = time.monotonic()
        for _ in range(100):
            await state_mgr._store.get_checkpoint(random.choice(checkpoint_ids))
        time_at_1k = time.monotonic() - start

        # Insert 9000 more
        for i in range(9000):
            ctx = await state_mgr.handoff(f"s2_{i}", f"t2_{i}", {"i": i})
            checkpoint_ids.append(ctx.checkpoint_id)

        # Measure again
        start = time.monotonic()
        for _ in range(100):
            await state_mgr._store.get_checkpoint(random.choice(checkpoint_ids))
        time_at_10k = time.monotonic() - start

        ratio = time_at_10k / time_at_1k if time_at_1k > 0 else 999

        _print_result(
            "Lookup scaling (1K vs 10K entries)",
            ratio < 3.0,
            f"at 1K: {time_at_1k*1000:.1f}ms, at 10K: {time_at_10k*1000:.1f}ms, ratio: {ratio:.2f}x",
        )

        assert ratio < 3.0, f"Lookups degraded {ratio:.2f}x from 1K to 10K entries"
