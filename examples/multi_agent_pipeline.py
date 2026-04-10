"""Kortex multi-agent pipeline — writer → reviewer → editor.

Demonstrates a 3-step pipeline with checkpoint persistence and rollback.
Each handoff creates a checkpoint; the example shows how to retrieve and
inspect the checkpoint chain.  No API keys required.

Run with:
    uv run python examples/multi_agent_pipeline.py
"""

from __future__ import annotations

import asyncio

from kortex.core.router import ProviderModel, Router
from kortex.core.runtime import AgentDescriptor, KortexRuntime
from kortex.core.state import StateManager
from kortex.core.types import TaskSpec
from kortex.store.memory import InMemoryStateStore


def _build_runtime() -> KortexRuntime:
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
            capabilities=["reasoning", "content_generation", "writing"],
            tier="balanced",
        ),
        ProviderModel(
            provider="anthropic",
            model="claude-opus-4-20250514",
            cost_per_1k_input_tokens=0.015,
            cost_per_1k_output_tokens=0.075,
            avg_latency_ms=2000,
            capabilities=["reasoning", "content_generation", "writing", "deep_analysis"],
            tier="powerful",
        ),
    ]:
        router.register_model(model)

    state = StateManager(store=InMemoryStateStore())
    runtime = KortexRuntime(router=router, state_manager=state)

    for agent in [
        AgentDescriptor("writer",   "Writer",   "Drafts content",   ["writing"]),
        AgentDescriptor("reviewer", "Reviewer", "Reviews for quality", ["reasoning", "deep_analysis"]),
        AgentDescriptor("editor",   "Editor",   "Finalises copy",   ["writing"]),
    ]:
        runtime.register_agent(agent)

    return runtime


async def main() -> None:
    task = TaskSpec(
        content="Write a blog post explaining how Kortex routes tasks across models.",
        complexity_hint="complex",
        required_capabilities=["writing"],
        metadata={"audience": "developers"},
    )

    print("=" * 65)
    print("Kortex Multi-Agent Pipeline: writer → reviewer → editor")
    print("=" * 65)
    print(f"Task: {task.content}\n")

    async with _build_runtime() as runtime:
        result = await runtime.coordinate(
            task,
            agent_pipeline=["writer", "reviewer", "editor"],
        )

    # Routing decisions --------------------------------------------------------
    print("Routing decisions:")
    for i, d in enumerate(result.routing_decisions, 1):
        print(f"  Step {i} ({d.agent_id}): {d.chosen_provider}/{d.chosen_model}"
              f"  est. ${d.estimated_cost_usd:.4f}  {d.estimated_latency_ms:.0f} ms")
        print(f"    Reason: {d.reasoning}")

    # Checkpoint chain ---------------------------------------------------------
    print(f"\nCheckpoint chain ({len(result.handoffs)} handoffs):")
    for h in result.handoffs:
        print(f"  {h.source_agent:>12} → {h.target_agent:<12} "
              f"checkpoint={h.checkpoint_id[:12]}…")

    # Cost summary -------------------------------------------------------------
    print(f"\nEstimated total cost: ${result.total_estimated_cost_usd:.4f}")
    print(f"Events emitted:       {len(result.events)}")

    # Show how to roll back to checkpoint 0 ------------------------------------
    first_cp = result.handoffs[0].checkpoint_id
    print(f"\nTo roll back to the initial handoff:\n"
          f"  await state_manager.rollback_to(checkpoint_id='{first_cp[:12]}…')")
    print("\nDone — no API key required.")


if __name__ == "__main__":
    asyncio.run(main())
