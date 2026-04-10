"""Kortex streaming example — stream_coordinate() event loop.

Demonstrates the streaming API: routing decisions, handoff checkpoints,
and completion events are yielded as they occur rather than returned in
a single CoordinationResult.  No API keys required.

Run with:
    uv run python examples/streaming_example.py
"""

from __future__ import annotations

import asyncio

from kortex.core.router import ProviderModel, Router
from kortex.core.runtime import AgentDescriptor, KortexRuntime
from kortex.core.state import StateManager
from kortex.core.types import TaskSpec
from kortex.store.memory import InMemoryStateStore

_EVENT_ICONS: dict[str, str] = {
    "routing_decision": "->",
    "handoff":          "<>",
    "token":            "~~",
    "completion":       "OK",
    "error":            "!!",
}


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

    state = StateManager(store=InMemoryStateStore())

    async with KortexRuntime(router=router, state_manager=state) as runtime:
        for agent in [
            AgentDescriptor("researcher", "Researcher", "Gathers info", ["reasoning"]),
            AgentDescriptor("writer",     "Writer",     "Drafts copy",  ["content_generation"]),
        ]:
            runtime.register_agent(agent)

        task = TaskSpec(
            content="Explain the benefits of streaming APIs for real-time agent pipelines.",
            complexity_hint="moderate",
        )

        print("=" * 65)
        print("Kortex Streaming Example — stream_coordinate()")
        print("=" * 65)
        print(f"Task: {task.content}\n")
        print("Event stream:")

        event_count = 0
        async for event_type, payload in runtime.stream_coordinate(
            task, agent_pipeline=["researcher", "writer"]
        ):
            icon = _EVENT_ICONS.get(event_type, "?")
            event_count += 1

            if event_type == "routing_decision":
                print(f"  [{icon}] routing_decision  step={payload.get('step')} "
                      f"model={payload.get('chosen_provider')}/{payload.get('chosen_model')}  "
                      f"est=${payload.get('estimated_cost_usd', 0):.4f}")

            elif event_type == "handoff":
                print(f"  [{icon}] handoff           "
                      f"{payload.get('source')} → {payload.get('target')}  "
                      f"checkpoint={str(payload.get('checkpoint_id', ''))[:12]}…")

            elif event_type == "token":
                # Tokens are yielded when a provider supports streaming
                print(f"  [{icon}] token             agent={payload.get('agent_id')} "
                      f"'{payload.get('token', '')}'")

            elif event_type == "completion":
                print(f"  [{icon}] completion        agents={payload.get('agents_routed')} "
                      f"duration={payload.get('duration_ms', 0):.0f} ms  "
                      f"success={payload.get('success')}")

            elif event_type == "error":
                print(f"  [{icon}] error             agent={payload.get('agent_id')} "
                      f"{payload.get('error')}")

    print(f"\nTotal events received: {event_count}")
    print("Done — no API key required.")


if __name__ == "__main__":
    asyncio.run(main())
