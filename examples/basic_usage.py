"""Kortex basic usage — minimal runtime setup.

Registers two models (fast + balanced), routes a single task, and
prints the routing decision. No API keys required.

Run with:
    uv run python examples/basic_usage.py
"""

from __future__ import annotations

import asyncio

from kortex.core.router import ProviderModel, Router
from kortex.core.runtime import AgentDescriptor, KortexRuntime
from kortex.core.state import StateManager
from kortex.core.types import TaskSpec
from kortex.store.memory import InMemoryStateStore


async def main() -> None:
    # 1. Register models -------------------------------------------------------
    router = Router()
    router.register_model(ProviderModel(
        provider="openai",
        model="gpt-4o-mini",
        cost_per_1k_input_tokens=0.00015,
        cost_per_1k_output_tokens=0.0006,
        avg_latency_ms=250,
        capabilities=["reasoning", "summarization"],
        tier="fast",
    ))
    router.register_model(ProviderModel(
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        cost_per_1k_input_tokens=0.003,
        cost_per_1k_output_tokens=0.015,
        avg_latency_ms=800,
        capabilities=["reasoning", "code_generation", "content_generation"],
        tier="balanced",
    ))

    # 2. Build runtime with one agent -----------------------------------------
    state = StateManager(store=InMemoryStateStore())
    async with KortexRuntime(router=router, state_manager=state) as runtime:
        runtime.register_agent(
            AgentDescriptor("analyst", "Analyst", "Analyses data", ["reasoning"])
        )

        # 3. Define a simple task ----------------------------------------------
        task = TaskSpec(
            content="Summarise the key points of this quarterly earnings report.",
            complexity_hint="moderate",
            required_capabilities=["reasoning"],
        )

        # 4. Route (dry-run — no LLM call) -------------------------------------
        result = await runtime.coordinate(task, agent_pipeline=["analyst"])

    # 5. Print the outcome ------------------------------------------------------
    decision = result.routing_decisions[0]
    print("=" * 60)
    print("Kortex Basic Usage")
    print("=" * 60)
    print(f"Task:     {task.content[:60]}...")
    print(f"Model:    {decision.chosen_provider}/{decision.chosen_model}")
    print(f"Tier:     {decision.chosen_model}")
    print(f"Reason:   {decision.reasoning}")
    print(f"Est cost: ${decision.estimated_cost_usd:.5f}")
    print(f"Est lat:  {decision.estimated_latency_ms:.0f} ms")
    print(f"Handoffs: {len(result.handoffs)}")
    print(f"Events:   {len(result.events)}")
    print("=" * 60)
    print("Done — no API key required.")


if __name__ == "__main__":
    asyncio.run(main())
