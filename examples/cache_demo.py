"""Kortex cache demo — semantic cache speedup measurement.

Issues the same task 10 times.  The first call is a cache miss (full
routing path).  Calls 2-10 are cache hits and skip routing entirely.
Prints hit/miss stats and the observed speedup ratio.
No API keys required.

Run with:
    uv run python examples/cache_demo.py
"""

from __future__ import annotations

import asyncio
import time

from kortex.cache.backends import MemoryCache
from kortex.cache.semantic_cache import SemanticCache
from kortex.core.router import ProviderModel, Router
from kortex.core.runtime import AgentDescriptor, KortexRuntime
from kortex.core.state import StateManager
from kortex.core.types import TaskSpec
from kortex.store.memory import InMemoryStateStore

_REPEATS = 10


async def main() -> None:
    router = Router()
    for model in [
        ProviderModel(
            provider="openai",
            model="gpt-4o-mini",
            cost_per_1k_input_tokens=0.00015,
            cost_per_1k_output_tokens=0.0006,
            avg_latency_ms=250,
            capabilities=["reasoning", "summarization"],
            tier="fast",
        ),
        ProviderModel(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            cost_per_1k_input_tokens=0.003,
            cost_per_1k_output_tokens=0.015,
            avg_latency_ms=800,
            capabilities=["reasoning", "content_generation"],
            tier="balanced",
        ),
    ]:
        router.register_model(model)

    cache = SemanticCache(backend=MemoryCache(max_size=256), ttl_seconds=300)
    state = StateManager(store=InMemoryStateStore())

    async with KortexRuntime(
        router=router, state_manager=state, cache=cache
    ) as runtime:
        runtime.register_agent(
            AgentDescriptor("summariser", "Summariser", "Summarises text", ["summarization"])
        )

        task = TaskSpec(
            content="Summarise the architecture of transformer-based language models.",
            complexity_hint="moderate",
            required_capabilities=["reasoning"],
        )

        print("=" * 60)
        print("Kortex Semantic Cache Demo")
        print("=" * 60)
        print(f"Task: {task.content}")
        print(f"Repeats: {_REPEATS}\n")

        latencies: list[float] = []
        for i in range(1, _REPEATS + 1):
            t0 = time.perf_counter()
            result = await runtime.coordinate(task, agent_pipeline=["summariser"])
            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed_ms)
            hit = getattr(result, "cache_hit", False)
            label = "HIT " if hit else "MISS"
            print(f"  Call {i:>2}: [{label}]  {elapsed_ms:6.2f} ms")

    miss_ms = latencies[0]
    hit_ms  = sum(latencies[1:]) / (len(latencies) - 1) if len(latencies) > 1 else 0.0
    speedup = miss_ms / hit_ms if hit_ms > 0 else float("inf")

    print(f"\nResults:")
    print(f"  Cache miss (call 1):   {miss_ms:.2f} ms")
    print(f"  Cache hit avg (2-{_REPEATS}):  {hit_ms:.2f} ms")
    print(f"  Speedup ratio:         {speedup:.0f}x")
    print(f"\n  Hits:   {cache.hits}")
    print(f"  Misses: {cache.misses}")
    print(f"  Hit rate: {cache.hit_rate:.1%}")
    print("\nDone — no API key required.")


if __name__ == "__main__":
    asyncio.run(main())
