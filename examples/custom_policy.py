"""Kortex custom routing policy example.

Demonstrates RoutingPolicy, RoutingConstraint, RoutingObjective, and
FallbackRule.  Shows the difference in model selection between three
policies: cost-optimised, latency-optimised, and quality-first.
No API keys required.

Run with:
    uv run python examples/custom_policy.py
"""

from __future__ import annotations

import asyncio

from kortex.core.policy import (
    FallbackRule,
    RoutingConstraint,
    RoutingObjective,
    RoutingPolicy,
)
from kortex.core.router import ProviderModel, Router
from kortex.core.runtime import AgentDescriptor, KortexRuntime
from kortex.core.state import StateManager
from kortex.core.types import TaskSpec
from kortex.store.memory import InMemoryStateStore


def _models() -> list[ProviderModel]:
    return [
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
            capabilities=["reasoning", "code_generation", "content_generation"],
            tier="balanced",
        ),
        ProviderModel(
            provider="anthropic",
            model="claude-opus-4-20250514",
            cost_per_1k_input_tokens=0.015,
            cost_per_1k_output_tokens=0.075,
            avg_latency_ms=2000,
            capabilities=["reasoning", "code_generation", "content_generation", "deep_analysis"],
            tier="powerful",
        ),
    ]


# Three example policies -------------------------------------------------------

COST_OPTIMISED = RoutingPolicy(
    name="cost_optimised",
    description="Keep spend below $0.005 per step; fall back to next cheapest.",
    constraints=RoutingConstraint(max_cost_usd=0.005),
    objective=RoutingObjective(minimize="cost", prefer_tier="fast"),
    fallback=FallbackRule(strategy="next_cheapest"),
    budget_ceiling_usd=0.02,
)

LATENCY_OPTIMISED = RoutingPolicy(
    name="latency_optimised",
    description="P95 latency < 400 ms; prefer fast tier.",
    constraints=RoutingConstraint(max_latency_ms=400.0),
    objective=RoutingObjective(minimize="latency", prefer_tier="fast"),
    fallback=FallbackRule(strategy="next_fastest"),
)

QUALITY_FIRST = RoutingPolicy(
    name="quality_first",
    description="Best model available; cost secondary.",
    constraints=RoutingConstraint(
        required_capabilities=["deep_analysis"],
    ),
    objective=RoutingObjective(minimize="none", prefer_tier="powerful"),
    fallback=FallbackRule(strategy="same_tier"),
    budget_ceiling_usd=1.00,
)


async def run_with_policy(
    runtime: KortexRuntime,
    task: TaskSpec,
    policy: RoutingPolicy,
) -> None:
    """Route a task with a specific policy and print the decision."""
    # Inject policy via task metadata — the Router reads task.policy_override
    task_with_policy = task.model_copy(
        update={"metadata": {**task.metadata, "_policy": policy.name}}
    )
    # Register the policy on the router for this run
    runtime._router.set_active_policy(policy)  # type: ignore[attr-defined]
    result = await runtime.coordinate(task_with_policy, agent_pipeline=["analyst"])
    d = result.routing_decisions[0]
    print(f"  [{policy.name:<22}] {d.chosen_provider}/{d.chosen_model:<32} "
          f"est=${d.estimated_cost_usd:.5f}  {d.estimated_latency_ms:.0f}ms")
    print(f"    Reason: {d.reasoning}")


async def main() -> None:
    router = Router()
    for m in _models():
        router.register_model(m)

    state = StateManager(store=InMemoryStateStore())
    async with KortexRuntime(router=router, state_manager=state) as runtime:
        runtime.register_agent(
            AgentDescriptor("analyst", "Analyst", "Deep analysis", ["reasoning", "deep_analysis"])
        )

        task = TaskSpec(
            content="Perform a deep competitive analysis of open-source LLM routing frameworks.",
            complexity_hint="complex",
            required_capabilities=["reasoning"],
        )

        print("=" * 70)
        print("Kortex Custom Policy Example")
        print("=" * 70)
        print(f"Task: {task.content[:65]}...\n")
        print("Policy comparison:")

        for policy in [COST_OPTIMISED, LATENCY_OPTIMISED, QUALITY_FIRST]:
            await run_with_policy(runtime, task, policy)

    print("\nSame task — different policies — different model selections.")
    print("Done — no API key required.")


if __name__ == "__main__":
    asyncio.run(main())
