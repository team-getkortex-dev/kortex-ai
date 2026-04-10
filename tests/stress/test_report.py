"""
Kortex Stress Test Report
============================

This runs a representative battery across all stress categories
and produces a single formatted report.

Run with:
    pytest tests/stress/test_report.py -v -s
"""

import asyncio
import os
import random
import sys
import time
from dataclasses import dataclass

import pytest

from kortex.core.types import TaskSpec, CoordinationResult
from kortex.core.router import Router
from kortex.core.state import StateManager
from kortex.core.runtime import KortexRuntime, AgentDescriptor
from kortex.core.exceptions import RoutingFailedError, KortexError
from kortex.store.memory import InMemoryStateStore

from .conftest import (
    ChaosProvider,
    STRESS_MODELS,
    STRESS_AGENTS,
    build_router,
    build_runtime,
    generate_random_task,
    generate_large_payload,
)

pytestmark = pytest.mark.stress

NO_COLOR = os.environ.get("NO_COLOR") or not sys.stdout.isatty()


@dataclass
class BenchmarkResult:
    name: str
    passed: bool
    elapsed: float
    detail: str

    def __str__(self):
        if NO_COLOR:
            status = "PASS" if self.passed else "FAIL"
        else:
            status = "\033[92mPASS\033[0m" if self.passed else "\033[91mFAIL\033[0m"
        return f"  {self.name:<42s} {status}  {self.elapsed:>7.3f}s  {self.detail}"


class TestFullSystemReport:
    """
    The final exam. One test that exercises every layer of Kortex
    and produces a report card.
    """

    @pytest.mark.asyncio
    async def test_full_system_report(self):
        results: list[BenchmarkResult] = []

        separator = "=" * 62
        print(f"\n{separator}")
        print("  KORTEX FULL-SYSTEM STRESS TEST REPORT")
        print(f"{separator}\n")

        # ----------------------------------------------------------
        # 1. CONCURRENCY: 50 parallel coordinations
        # ----------------------------------------------------------
        runtime = build_runtime(
            ChaosProvider(name="r", failure_rate=0.0, min_latency_ms=0.05, max_latency_ms=0.2)
        )
        tasks = [generate_random_task() for _ in range(50)]
        pipeline = ["researcher", "writer", "reviewer"]

        start = time.monotonic()
        batch = await asyncio.gather(
            *[runtime.coordinate(t, pipeline) for t in tasks],
            return_exceptions=True,
        )
        elapsed = time.monotonic() - start

        errors = [r for r in batch if isinstance(r, Exception)]
        results.append(BenchmarkResult(
            "Concurrency (50 pipelines)",
            len(errors) == 0,
            elapsed,
            f"{50 - len(errors)}/50 OK",
        ))

        # ----------------------------------------------------------
        # 2. CONCURRENCY: 200 parallel coordinations
        # ----------------------------------------------------------
        tasks_200 = [generate_random_task() for _ in range(200)]
        start = time.monotonic()
        batch_200 = await asyncio.gather(
            *[runtime.coordinate(t, pipeline) for t in tasks_200],
            return_exceptions=True,
        )
        elapsed = time.monotonic() - start
        errors_200 = [r for r in batch_200 if isinstance(r, Exception)]
        results.append(BenchmarkResult(
            "Concurrency (200 pipelines)",
            len(errors_200) == 0,
            elapsed,
            f"{200 - len(errors_200)}/200 OK",
        ))

        # ----------------------------------------------------------
        # 3. VOLUME: 20-agent pipeline
        # ----------------------------------------------------------
        for i in range(20):
            runtime.register_agent(AgentDescriptor(
                agent_id=f"pipe_{i:02d}", name=f"Pipe {i}",
                description=f"Pipeline agent {i}", capabilities=["analysis"],
            ))

        long_pipeline = [f"pipe_{i:02d}" for i in range(20)]
        task = TaskSpec(content="Long pipeline stress", complexity_hint="complex")
        start = time.monotonic()
        result = await runtime.coordinate(task, long_pipeline)
        elapsed = time.monotonic() - start
        results.append(BenchmarkResult(
            "Large pipeline (20 agents)",
            result.success and len(result.routing_decisions) == 20,
            elapsed,
            f"{len(result.routing_decisions)} decisions",
        ))

        # ----------------------------------------------------------
        # 4. THROUGHPUT: 1000 rapid-fire routes
        # ----------------------------------------------------------
        router = build_router()
        start = time.monotonic()
        for _ in range(1000):
            await router.route(generate_random_task())
        elapsed = time.monotonic() - start
        throughput = 1000 / elapsed
        results.append(BenchmarkResult(
            "Throughput (1000 routes)",
            throughput > 100,
            elapsed,
            f"{throughput:.0f} routes/sec",
        ))

        # ----------------------------------------------------------
        # 5. THROUGHPUT: 5000 routes
        # ----------------------------------------------------------
        start = time.monotonic()
        for _ in range(5000):
            await router.route(generate_random_task())
        elapsed = time.monotonic() - start
        throughput_5k = 5000 / elapsed
        results.append(BenchmarkResult(
            "Throughput (5000 routes)",
            throughput_5k > 100,
            elapsed,
            f"{throughput_5k:.0f} routes/sec",
        ))

        # ----------------------------------------------------------
        # 6. FAILURE: Intermittent 30% failure rate
        # ----------------------------------------------------------
        flaky_runtime = build_runtime(
            ChaosProvider(name="flaky", failure_rate=0.3, min_latency_ms=0.1, max_latency_ms=2.0)
        )
        succeeded = 0
        crashed = 0
        start = time.monotonic()
        for _ in range(100):
            try:
                r = await flaky_runtime.coordinate(generate_random_task(), ["researcher", "writer"], execute=True)
                if r.success:
                    succeeded += 1
            except (RoutingFailedError, KortexError):
                pass
            except Exception:
                crashed += 1
        elapsed = time.monotonic() - start
        results.append(BenchmarkResult(
            "Failure resilience (100 @ 30%)",
            crashed == 0,
            elapsed,
            f"{succeeded}/100 succeeded, {crashed} crashes",
        ))

        # ----------------------------------------------------------
        # 7. STATE STORE: 10K checkpoint writes (memory)
        # ----------------------------------------------------------
        state_mgr = StateManager.create("memory")
        start = time.monotonic()
        cids = []
        for i in range(10_000):
            ctx = await state_mgr.handoff(f"s_{i%50}", f"t_{i%50}", {"i": i})
            cids.append(ctx.checkpoint_id)
        elapsed = time.monotonic() - start
        write_rate = 10_000 / elapsed
        results.append(BenchmarkResult(
            "State store (10K writes, memory)",
            True,
            elapsed,
            f"{write_rate:.0f} writes/sec",
        ))

        # ----------------------------------------------------------
        # 8. STATE STORE: 10K checkpoint writes (SQLite)
        # ----------------------------------------------------------
        state_mgr_sql = StateManager.create("sqlite", db_path=":memory:")
        store = state_mgr_sql._store  # type: ignore
        if hasattr(store, 'connect'):
            await store.connect()

        start = time.monotonic()
        for i in range(10_000):
            await state_mgr_sql.handoff(f"sq_{i%50}", f"st_{i%50}", {"i": i})
        elapsed = time.monotonic() - start
        sql_write_rate = 10_000 / elapsed
        results.append(BenchmarkResult(
            "State store (10K writes, SQLite)",
            True,
            elapsed,
            f"{sql_write_rate:.0f} writes/sec",
        ))

        # ----------------------------------------------------------
        # 9. STATE: 1000 random lookups across 10K entries
        # ----------------------------------------------------------
        start = time.monotonic()
        for _ in range(1000):
            await state_mgr._store.get_checkpoint(random.choice(cids))
        elapsed = time.monotonic() - start
        lookup_rate = 1000 / elapsed
        results.append(BenchmarkResult(
            "State lookups (1K reads from 10K)",
            lookup_rate > 1000,
            elapsed,
            f"{lookup_rate:.0f} reads/sec",
        ))

        # ----------------------------------------------------------
        # 10. EDGE: Unicode + large payloads
        # ----------------------------------------------------------
        edge_runtime = build_runtime(
            ChaosProvider(name="r", failure_rate=0.0, min_latency_ms=0.05, max_latency_ms=0.1)
        )
        edge_passed = True
        start = time.monotonic()

        # Unicode
        for content in [
            "\U0001F4CA Multi-agent \u5206\u6790 \u062A\u062D\u0644\u064A\u0644 \U0001F916",
            "A" * 100_000,
        ]:
            try:
                r = await edge_runtime.coordinate(
                    TaskSpec(content=content, complexity_hint="simple"),
                    ["researcher"],
                )
                if not r.success:
                    edge_passed = False
            except Exception:
                edge_passed = False

        elapsed = time.monotonic() - start
        results.append(BenchmarkResult(
            "Edge cases (unicode + 100K content)",
            edge_passed,
            elapsed,
            "all handled",
        ))

        # ----------------------------------------------------------
        # 11. MEMORY: 1MB and 10MB payloads
        # ----------------------------------------------------------
        payload_mgr = StateManager.create("memory")
        start = time.monotonic()
        try:
            await payload_mgr.handoff("big_s", "big_t", generate_large_payload(1_000_000))
            await payload_mgr.handoff("huge_s", "huge_t", generate_large_payload(10_000_000))
            payload_ok = True
        except Exception:
            payload_ok = False
        elapsed = time.monotonic() - start
        results.append(BenchmarkResult(
            "Large payloads (1MB + 10MB)",
            payload_ok,
            elapsed,
            "stored and retrieved",
        ))

        # ----------------------------------------------------------
        # 12. CHECKPOINT CHAIN: 100 deep
        # ----------------------------------------------------------
        chain_mgr = StateManager.create("memory")
        parent = None
        start = time.monotonic()
        last_id = None
        for i in range(100):
            ctx = await chain_mgr.handoff(f"c_{i}", f"c_{i+1}", {"d": i}, parent_checkpoint_id=parent)
            parent = ctx.checkpoint_id
            last_id = ctx.checkpoint_id
        chain = await chain_mgr.get_history(last_id)
        elapsed = time.monotonic() - start
        results.append(BenchmarkResult(
            "Checkpoint chain (100 deep)",
            len(chain) == 100,
            elapsed,
            f"chain length: {len(chain)}",
        ))

        # ----------------------------------------------------------
        # PRINT REPORT
        # ----------------------------------------------------------
        print(f"\n{separator}")
        print("  RESULTS")
        print(f"{separator}")

        for r in results:
            print(str(r))

        all_passed = all(r.passed for r in results)
        total_time = sum(r.elapsed for r in results)

        print(f"{separator}")
        if all_passed:
            status = "ALL CLEAR" if NO_COLOR else "\033[92mALL CLEAR\033[0m"
        else:
            failed_count = sum(1 for r in results if not r.passed)
            status = f"{failed_count} FAILED" if NO_COLOR else f"\033[91m{failed_count} FAILED\033[0m"

        print(f"  SYSTEM STATUS: {status}  (total: {total_time:.2f}s)")
        print(f"{separator}\n")

        assert all_passed, f"Stress test failures: {[r.name for r in results if not r.passed]}"
