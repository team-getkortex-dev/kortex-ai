#!/usr/bin/env python3
"""
Kortex system load / stress harness.

Purpose
-------
Exercise Kortex under simulated user load with randomized spikes and dips in
concurrency. This is designed to test the orchestration/runtime layer itself,
not upstream live model APIs.

Key features
------------
- Simulates a configurable min/max number of concurrent "users"
- Randomized spikes, dips, and baseline drift over time
- Supports both:
  - dry-run mode (execute=False): stresses routing/state/event paths only
  - mock mode   (execute=True): stresses runtime + mocked provider execution
- Optional memory/sqlite/redis state backend selection
- Tracks throughput, latency, failures, anomaly counts, estimated/actual cost,
  and concurrency peaks
- Emits a summary to stdout and optionally writes JSON

Notes / assumptions
-------------------
- This harness targets the current public/imported Kortex API and a few private
  implementation details that are already used in the project's own examples
  (notably provider client monkeypatching, and best-effort backend connect).
- For non-memory backends, current Kortex store lifecycle may require explicit
  connect()/disconnect() calls. This harness performs those best-effort calls.
- Mock execution mode intentionally avoids live provider traffic so that results
  measure Kortex more than external APIs.

Example usage
-------------
Dry-run (pure orchestration stress):
    python scripts/kortex_system_load_test.py \
        --mode dry-run \
        --run-seconds 120 \
        --min-users 5 \
        --max-users 150 \
        --initial-users 20

Mocked execution (runtime + mocked provider calls):
    python scripts/kortex_system_load_test.py \
        --mode mock \
        --run-seconds 180 \
        --min-users 10 \
        --max-users 200 \
        --initial-users 25 \
        --backend memory \
        --json-out load_test_report.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import statistics
import string
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

from kortex.core.detector import DetectionPolicy, FailureDetector
from kortex.core.router import ProviderModel, Router
from kortex.core.runtime import AgentDescriptor, KortexRuntime
from kortex.core.state import StateManager
from kortex.core.types import TaskSpec
from kortex.providers.registry import ProviderRegistry

# ---------------------------------------------------------------------------
# Static catalogs — uses canonical Kortex capability vocabulary
# ---------------------------------------------------------------------------

MODEL_CATALOG: list[ProviderModel] = [
    ProviderModel(
        provider="openai",
        model="gpt-4o-mini",
        cost_per_1k_input_tokens=0.00015,
        cost_per_1k_output_tokens=0.0006,
        avg_latency_ms=220,
        capabilities=["reasoning", "analysis", "code_generation", "content_generation"],
        max_context_tokens=128_000,
        tier="fast",
    ),
    ProviderModel(
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        cost_per_1k_input_tokens=0.003,
        cost_per_1k_output_tokens=0.015,
        avg_latency_ms=800,
        capabilities=["reasoning", "analysis", "code_generation", "content_generation", "vision"],
        max_context_tokens=200_000,
        tier="balanced",
    ),
    ProviderModel(
        provider="anthropic",
        model="claude-opus-4-20250514",
        cost_per_1k_input_tokens=0.015,
        cost_per_1k_output_tokens=0.075,
        avg_latency_ms=2_000,
        capabilities=["reasoning", "analysis", "code_generation", "content_generation", "vision"],
        max_context_tokens=200_000,
        tier="powerful",
    ),
]

AGENTS: list[AgentDescriptor] = [
    AgentDescriptor(
        agent_id="researcher",
        name="Researcher",
        description="Gathers and synthesizes information",
        capabilities=["reasoning", "analysis"],
    ),
    AgentDescriptor(
        agent_id="writer",
        name="Writer",
        description="Drafts clear output",
        capabilities=["content_generation"],
    ),
    AgentDescriptor(
        agent_id="reviewer",
        name="Reviewer",
        description="Performs review and quality checks",
        capabilities=["reasoning", "analysis"],
    ),
]

PIPELINES: list[list[str]] = [
    ["researcher"],
    ["writer"],
    ["researcher", "writer"],
    ["writer", "reviewer"],
    ["researcher", "writer", "reviewer"],
]

TASK_TOPICS: list[str] = [
    "multi-agent coordination",
    "routing policy analysis",
    "failure recovery patterns",
    "code review automation",
    "API integration testing",
    "distributed orchestration",
    "cost-aware model selection",
    "trace replay diagnostics",
    "state checkpoint semantics",
    "agent workflow optimization",
]

CAPABILITY_POOL: list[str] = [
    "reasoning",
    "analysis",
    "code_generation",
    "content_generation",
]


# ---------------------------------------------------------------------------
# Metrics / report models
# ---------------------------------------------------------------------------

@dataclass
class RequestMetric:
    request_id: str
    user_id: int
    started_at: float
    ended_at: float
    latency_ms: float
    success: bool
    execute: bool
    pipeline: list[str]
    complexity: str
    anomaly_count: int = 0
    error_type: str | None = None
    estimated_cost_usd: float = 0.0
    actual_cost_usd: float = 0.0


@dataclass
class ControllerTick:
    t: float
    active_users: int
    reason: str


@dataclass
class MetricsCollector:
    started_at: float = field(default_factory=time.perf_counter)
    total_requests: int = 0
    successes: int = 0
    failures: int = 0
    timeouts: int = 0
    total_anomalies: int = 0
    estimated_cost_usd: float = 0.0
    actual_cost_usd: float = 0.0
    current_inflight: int = 0
    peak_inflight: int = 0
    peak_users: int = 0
    request_latencies_ms: list[float] = field(default_factory=list)
    failure_types: dict[str, int] = field(default_factory=dict)
    complexity_counts: dict[str, int] = field(default_factory=dict)
    pipeline_counts: dict[str, int] = field(default_factory=dict)
    controller_history: list[ControllerTick] = field(default_factory=list)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    async def set_active_users(self, active_users: int, reason: str) -> None:
        async with self._lock:
            self.peak_users = max(self.peak_users, active_users)
            self.controller_history.append(
                ControllerTick(
                    t=time.perf_counter() - self.started_at,
                    active_users=active_users,
                    reason=reason,
                )
            )

    async def inflight_started(self) -> None:
        async with self._lock:
            self.current_inflight += 1
            self.peak_inflight = max(self.peak_inflight, self.current_inflight)

    async def inflight_ended(self) -> None:
        async with self._lock:
            self.current_inflight = max(0, self.current_inflight - 1)

    async def record(self, metric: RequestMetric) -> None:
        async with self._lock:
            self.total_requests += 1
            self.request_latencies_ms.append(metric.latency_ms)
            self.estimated_cost_usd += metric.estimated_cost_usd
            self.actual_cost_usd += metric.actual_cost_usd
            self.anomaly_count_key = metric.anomaly_count
            self.total_anomalies += metric.anomaly_count

            ckey = metric.complexity
            self.complexity_counts[ckey] = self.complexity_counts.get(ckey, 0) + 1

            pkey = " -> ".join(metric.pipeline)
            self.pipeline_counts[pkey] = self.pipeline_counts.get(pkey, 0) + 1

            if metric.success:
                self.successes += 1
            else:
                self.failures += 1
                etype = metric.error_type or "unknown"
                self.failure_types[etype] = self.failure_types.get(etype, 0) + 1
                if etype == "timeout":
                    self.timeouts += 1

    def snapshot(
        self,
        run_seconds: float,
        mode: str,
        backend: str,
        seed: int,
    ) -> dict[str, Any]:
        elapsed = time.perf_counter() - self.started_at
        latencies = sorted(self.request_latencies_ms) if self.request_latencies_ms else [0.0]

        def _pct(data: list[float], p: float) -> float:
            idx = max(0, min(len(data) - 1, int(math.ceil(p / 100.0 * len(data)) - 1)))
            return round(data[idx], 2)

        return {
            "mode": mode,
            "backend": backend,
            "seed": seed,
            "target_run_seconds": run_seconds,
            "actual_elapsed_seconds": round(elapsed, 2),
            "total_requests": self.total_requests,
            "successes": self.successes,
            "failures": self.failures,
            "timeouts": self.timeouts,
            "success_rate": round(self.successes / max(1, self.total_requests), 4),
            "throughput_rps": round(self.total_requests / max(0.01, elapsed), 2),
            "peak_active_users": self.peak_users,
            "peak_inflight_requests": self.peak_inflight,
            "total_anomalies": self.total_anomalies,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "actual_cost_usd": round(self.actual_cost_usd, 6),
            "latency_ms": {
                "min": round(latencies[0], 2),
                "avg": round(statistics.mean(latencies), 2),
                "p50": _pct(latencies, 50),
                "p95": _pct(latencies, 95),
                "p99": _pct(latencies, 99),
                "max": round(latencies[-1], 2),
            },
            "failure_types": dict(self.failure_types),
            "complexity_counts": dict(self.complexity_counts),
            "pipeline_counts": dict(self.pipeline_counts),
            "controller_history": [
                {"t": round(c.t, 2), "active_users": c.active_users, "reason": c.reason}
                for c in self.controller_history
            ],
        }


# ---------------------------------------------------------------------------
# Task generation
# ---------------------------------------------------------------------------


def random_task(random_source: random.Random) -> tuple[TaskSpec, list[str]]:
    topic = random_source.choice(TASK_TOPICS)
    complexity = random_source.choices(
        ["simple", "moderate", "complex"],
        weights=[40, 35, 25],
        k=1,
    )[0]

    num_caps = random_source.randint(0, 2)
    caps = random_source.sample(CAPABILITY_POOL, k=min(num_caps, len(CAPABILITY_POOL)))

    task = TaskSpec(
        content=f"Analyze and report on {topic}",
        complexity_hint=complexity,
        required_capabilities=caps,
    )
    pipeline = random_source.choice(PIPELINES)
    return task, pipeline


# ---------------------------------------------------------------------------
# Mock provider helpers (for --mode mock)
# ---------------------------------------------------------------------------


class MockProviderResponse:
    def __init__(self, content: str, model: str, provider: str) -> None:
        self.content = content
        self.model = model
        self.provider = provider
        self.input_tokens = random.randint(50, 500)
        self.output_tokens = random.randint(20, 300)
        self.cost_usd = round(
            self.input_tokens * 0.00003 + self.output_tokens * 0.00006, 6
        )
        self.latency_ms = round(random.uniform(10, 200), 2)
        self.raw_response: dict[str, Any] = {"mock": True}


class MockProvider:
    """Minimal mock that satisfies the provider protocol."""

    def __init__(self, name: str, models: list[ProviderModel]) -> None:
        self._name = name
        self._models = models

    @property
    def provider_name(self) -> str:
        return self._name

    def get_available_models(self) -> list[ProviderModel]:
        return self._models

    async def health_check(self) -> bool:
        return True

    async def complete(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> MockProviderResponse:
        await asyncio.sleep(random.uniform(0.001, 0.01))
        return MockProviderResponse(
            content=f"Mock response for: {prompt[:60]}",
            model=model,
            provider=self._name,
        )

    async def stream(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("stream not needed for load test")

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Runtime builder
# ---------------------------------------------------------------------------


async def build_runtime(
    mode: str,
    backend: str,
    sqlite_path: str = "kortex_load_test.db",
    redis_url: str = "redis://localhost:6379",
    redis_key_prefix: str = "kortex-load-test:",
    detector_enabled: bool = True,
    random_source: random.Random | None = None,
) -> tuple[KortexRuntime, StateManager, ProviderRegistry | None]:
    router = Router()
    for m in MODEL_CATALOG:
        router.register_model(m)

    store_kwargs: dict[str, Any] = {}
    if backend == "sqlite":
        store_kwargs["db_path"] = sqlite_path
    elif backend == "redis":
        store_kwargs["redis_url"] = redis_url
        store_kwargs["key_prefix"] = redis_key_prefix

    state_manager = StateManager.create(backend, **store_kwargs)

    # Best-effort connect for non-memory backends
    if hasattr(state_manager, "start"):
        try:
            await state_manager.start()
        except Exception:
            pass

    detector = FailureDetector(DetectionPolicy()) if detector_enabled else None

    registry: ProviderRegistry | None = None
    if mode == "mock":
        registry = ProviderRegistry()
        # Group models by provider and create mock providers
        providers_map: dict[str, list[ProviderModel]] = {}
        for m in MODEL_CATALOG:
            providers_map.setdefault(m.provider, []).append(m)
        for pname, pmodels in providers_map.items():
            registry.register_provider(MockProvider(pname, pmodels))

    runtime = KortexRuntime(
        router=router,
        state_manager=state_manager,
        registry=registry,
        detector=detector,
    )

    for agent in AGENTS:
        runtime.register_agent(agent)

    return runtime, state_manager, registry


async def maybe_disconnect_state_backend(state_manager: StateManager) -> None:
    if hasattr(state_manager, "stop"):
        try:
            await state_manager.stop()
        except Exception:
            pass


async def maybe_close_registry_clients(registry: ProviderRegistry | None) -> None:
    if registry is not None and hasattr(registry, "close_all"):
        try:
            await registry.close_all()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Simulated user
# ---------------------------------------------------------------------------


class SimulatedUser:
    def __init__(
        self,
        user_id: int,
        runtime: KortexRuntime,
        mode: str,
        request_timeout_seconds: float,
        min_think_seconds: float,
        max_think_seconds: float,
        random_source: random.Random,
        metrics: MetricsCollector,
    ) -> None:
        self.user_id = user_id
        self.runtime = runtime
        self.mode = mode
        self.request_timeout_seconds = request_timeout_seconds
        self.min_think_seconds = min_think_seconds
        self.max_think_seconds = max_think_seconds
        self.random_source = random_source
        self.metrics = metrics
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    def start(self) -> None:
        self._stop.clear()
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=self.request_timeout_seconds + 2)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass

    async def _loop(self) -> None:
        while not self._stop.is_set():
            think = self.random_source.uniform(
                self.min_think_seconds, self.max_think_seconds
            )
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=think)
                return  # stop was set during think time
            except asyncio.TimeoutError:
                pass  # think time elapsed normally

            await self._make_request()

    async def _make_request(self) -> None:
        task, pipeline = random_task(self.random_source)
        execute = self.mode == "mock"
        request_id = str(uuid4())

        await self.metrics.inflight_started()
        t0 = time.perf_counter()

        try:
            result = await asyncio.wait_for(
                self.runtime.coordinate(task, pipeline, execute=execute),
                timeout=self.request_timeout_seconds,
            )
            t1 = time.perf_counter()

            anomaly_count = len(result.anomalies) if hasattr(result, "anomalies") else 0
            actual_cost = result.actual_cost_usd if hasattr(result, "actual_cost_usd") else 0.0

            metric = RequestMetric(
                request_id=request_id,
                user_id=self.user_id,
                started_at=t0,
                ended_at=t1,
                latency_ms=round((t1 - t0) * 1000, 2),
                success=result.success,
                execute=execute,
                pipeline=pipeline,
                complexity=task.complexity_hint,
                anomaly_count=anomaly_count,
                estimated_cost_usd=result.total_estimated_cost_usd,
                actual_cost_usd=actual_cost,
            )

        except asyncio.TimeoutError:
            t1 = time.perf_counter()
            metric = RequestMetric(
                request_id=request_id,
                user_id=self.user_id,
                started_at=t0,
                ended_at=t1,
                latency_ms=round((t1 - t0) * 1000, 2),
                success=False,
                execute=execute,
                pipeline=pipeline,
                complexity=task.complexity_hint,
                error_type="timeout",
            )

        except Exception as exc:
            t1 = time.perf_counter()
            metric = RequestMetric(
                request_id=request_id,
                user_id=self.user_id,
                started_at=t0,
                ended_at=t1,
                latency_ms=round((t1 - t0) * 1000, 2),
                success=False,
                execute=execute,
                pipeline=pipeline,
                complexity=task.complexity_hint,
                error_type=type(exc).__name__,
            )

        finally:
            await self.metrics.inflight_ended()

        await self.metrics.record(metric)


# ---------------------------------------------------------------------------
# User population controller
# ---------------------------------------------------------------------------


async def adjust_user_population(
    users: dict[int, SimulatedUser],
    target_users: int,
    runtime: KortexRuntime,
    mode: str,
    request_timeout_seconds: float,
    min_think_seconds: float,
    max_think_seconds: float,
    random_source: random.Random,
    metrics: MetricsCollector,
) -> None:
    current = len(users)

    if target_users > current:
        for _ in range(target_users - current):
            uid = max(users.keys(), default=-1) + 1
            user = SimulatedUser(
                user_id=uid,
                runtime=runtime,
                mode=mode,
                request_timeout_seconds=request_timeout_seconds,
                min_think_seconds=min_think_seconds,
                max_think_seconds=max_think_seconds,
                random_source=random_source,
                metrics=metrics,
            )
            users[uid] = user
            user.start()

    elif target_users < current:
        to_remove = current - target_users
        keys = list(users.keys())[-to_remove:]
        for k in keys:
            await users[k].stop()
            del users[k]


def next_target_user_count(
    current: int,
    min_users: int,
    max_users: int,
    max_normal_step: int = 5,
    max_spike_size: int = 35,
    max_dip_size: int = 25,
    spike_probability: float = 0.20,
    dip_probability: float = 0.18,
    random_source: random.Random | None = None,
) -> tuple[int, str]:
    if random_source is None:
        random_source = random.Random()

    roll = random_source.random()

    if roll < spike_probability:
        new_value = current + random_source.randint(1, max_spike_size)
        return min(max_users, new_value), "spike"
    if roll < spike_probability + dip_probability:
        new_value = current - random_source.randint(1, max_dip_size)
        return max(min_users, new_value), "dip"

    drift = random_source.randint(-max_normal_step, max_normal_step)
    return max(min_users, min(max_users, current + drift)), "drift"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


async def run_load_test(args: argparse.Namespace) -> dict[str, Any]:
    random_source = random.Random(args.seed)
    metrics = MetricsCollector()
    runtime, state_manager, registry = await build_runtime(
        mode=args.mode,
        backend=args.backend,
        sqlite_path=args.sqlite_path,
        redis_url=args.redis_url,
        redis_key_prefix=args.redis_key_prefix,
        detector_enabled=(not args.disable_detector),
        random_source=random_source,
    )

    users: dict[int, SimulatedUser] = {}
    current_target = max(args.min_users, min(args.max_users, args.initial_users))
    await adjust_user_population(
        users=users,
        target_users=current_target,
        runtime=runtime,
        mode=args.mode,
        request_timeout_seconds=args.request_timeout_seconds,
        min_think_seconds=args.min_think_seconds,
        max_think_seconds=args.max_think_seconds,
        random_source=random_source,
        metrics=metrics,
    )
    await metrics.set_active_users(current_target, reason="initial")

    started = time.perf_counter()
    try:
        while (time.perf_counter() - started) < args.run_seconds:
            await asyncio.sleep(args.adjust_every_seconds)
            current_target, reason = next_target_user_count(
                current=current_target,
                min_users=args.min_users,
                max_users=args.max_users,
                max_normal_step=args.max_normal_step,
                max_spike_size=args.max_spike_size,
                max_dip_size=args.max_dip_size,
                spike_probability=args.spike_probability,
                dip_probability=args.dip_probability,
                random_source=random_source,
            )
            await adjust_user_population(
                users=users,
                target_users=current_target,
                runtime=runtime,
                mode=args.mode,
                request_timeout_seconds=args.request_timeout_seconds,
                min_think_seconds=args.min_think_seconds,
                max_think_seconds=args.max_think_seconds,
                random_source=random_source,
                metrics=metrics,
            )
            await metrics.set_active_users(current_target, reason=reason)
    finally:
        await adjust_user_population(
            users=users,
            target_users=0,
            runtime=runtime,
            mode=args.mode,
            request_timeout_seconds=args.request_timeout_seconds,
            min_think_seconds=args.min_think_seconds,
            max_think_seconds=args.max_think_seconds,
            random_source=random_source,
            metrics=metrics,
        )
        await maybe_disconnect_state_backend(state_manager)
        await maybe_close_registry_clients(registry)

    return metrics.snapshot(
        run_seconds=args.run_seconds,
        mode=args.mode,
        backend=args.backend,
        seed=args.seed,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kortex system load / stress harness")

    parser.add_argument("--mode", choices=["dry-run", "mock"], default="dry-run")
    parser.add_argument("--backend", choices=["memory", "sqlite", "redis"], default="memory")
    parser.add_argument("--run-seconds", type=float, default=60.0)
    parser.add_argument("--initial-users", type=int, default=10)
    parser.add_argument("--min-users", type=int, default=5)
    parser.add_argument("--max-users", type=int, default=100)
    parser.add_argument("--adjust-every-seconds", type=float, default=5.0)
    parser.add_argument("--max-normal-step", type=int, default=5)
    parser.add_argument("--max-spike-size", type=int, default=35)
    parser.add_argument("--max-dip-size", type=int, default=25)
    parser.add_argument("--spike-probability", type=float, default=0.20)
    parser.add_argument("--dip-probability", type=float, default=0.18)
    parser.add_argument("--min-think-seconds", type=float, default=0.05)
    parser.add_argument("--max-think-seconds", type=float, default=0.35)
    parser.add_argument("--request-timeout-seconds", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-detector", action="store_true")
    parser.add_argument("--sqlite-path", default="kortex_load_test.db")
    parser.add_argument("--redis-url", default="redis://localhost:6379")
    parser.add_argument("--redis-key-prefix", default="kortex-load-test:")
    parser.add_argument("--json-out", default=None)

    args = parser.parse_args()

    if args.min_users < 0:
        parser.error("--min-users must be >= 0")
    if args.max_users < 1:
        parser.error("--max-users must be >= 1")
    if args.min_users > args.max_users:
        parser.error("--min-users cannot exceed --max-users")
    if args.initial_users < args.min_users or args.initial_users > args.max_users:
        parser.error("--initial-users must be within [min-users, max-users]")
    if not (0.0 <= args.spike_probability <= 1.0):
        parser.error("--spike-probability must be between 0 and 1")
    if not (0.0 <= args.dip_probability <= 1.0):
        parser.error("--dip-probability must be between 0 and 1")
    if args.spike_probability + args.dip_probability > 1.0:
        parser.error("spike_probability + dip_probability cannot exceed 1")
    if args.min_think_seconds > args.max_think_seconds:
        parser.error("--min-think-seconds cannot exceed --max-think-seconds")
    if args.run_seconds <= 0:
        parser.error("--run-seconds must be > 0")
    if args.adjust_every_seconds <= 0:
        parser.error("--adjust-every-seconds must be > 0")
    if args.request_timeout_seconds <= 0:
        parser.error("--request-timeout-seconds must be > 0")

    return args


def print_summary(summary: dict[str, Any]) -> None:
    print("=" * 88)
    print("KORTEX LOAD TEST SUMMARY")
    print("=" * 88)
    print(f"Mode:                 {summary['mode']}")
    print(f"Backend:              {summary['backend']}")
    print(f"Seed:                 {summary['seed']}")
    print(f"Target run seconds:   {summary['target_run_seconds']}")
    print(f"Actual elapsed:       {summary['actual_elapsed_seconds']}")
    print(f"Total requests:       {summary['total_requests']}")
    print(f"Successes:            {summary['successes']}")
    print(f"Failures:             {summary['failures']}")
    print(f"Timeouts:             {summary['timeouts']}")
    print(f"Success rate:         {summary['success_rate']:.2%}")
    print(f"Throughput (RPS):     {summary['throughput_rps']}")
    print(f"Peak active users:    {summary['peak_active_users']}")
    print(f"Peak inflight reqs:   {summary['peak_inflight_requests']}")
    print(f"Total anomalies:      {summary['total_anomalies']}")
    print(f"Estimated cost USD:   {summary['estimated_cost_usd']}")
    print(f"Actual cost USD:      {summary['actual_cost_usd']}")

    latency = summary["latency_ms"]
    print("-" * 88)
    print(
        "Latency ms:"
        f" min={latency['min']}"
        f" avg={latency['avg']}"
        f" p50={latency['p50']}"
        f" p95={latency['p95']}"
        f" p99={latency['p99']}"
        f" max={latency['max']}"
    )

    print("-" * 88)
    print("Failure types:")
    if summary["failure_types"]:
        for key, value in summary["failure_types"].items():
            print(f"  {key:<32} {value}")
    else:
        print("  none")

    print("-" * 88)
    print("Complexity counts:")
    for key, value in summary["complexity_counts"].items():
        print(f"  {key:<32} {value}")

    print("-" * 88)
    print("Pipeline counts:")
    for key, value in summary["pipeline_counts"].items():
        print(f"  {key:<32} {value}")

    print("-" * 88)
    print("Controller history:")
    for item in summary["controller_history"]:
        print(f"  t={item['t']:>7}s users={item['active_users']:<5} reason={item['reason']}")
    print("=" * 88)


async def async_main() -> int:
    args = parse_args()
    summary = await run_load_test(args)
    print_summary(summary)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote JSON report to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(async_main()))
