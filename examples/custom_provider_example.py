"""Custom provider example -- mixing local and cloud models.

Demonstrates how to:
1. Register a local Ollama instance with zero-cost models
2. Register a custom corporate API endpoint with auth
3. Mix local + cloud models in the same Router
4. Route simple tasks to local Ollama and complex tasks to cloud Anthropic

Runnable without any API keys -- HTTP calls are mocked.

Run with:
    uv run python examples/custom_provider_example.py
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import httpx

from kortex.core.router import ProviderModel, Router
from kortex.core.runtime import AgentDescriptor, KortexRuntime
from kortex.core.state import StateManager
from kortex.core.types import TaskSpec
from kortex.providers.anthropic import AnthropicConnector
from kortex.providers.registry import ProviderRegistry
from kortex.store.memory import InMemoryStateStore


def _mock_response(data: dict) -> httpx.Response:  # type: ignore[type-arg]
    return httpx.Response(
        status_code=200,
        json=data,
        request=httpx.Request("POST", "https://mock"),
    )


async def main() -> None:
    registry = ProviderRegistry()

    # ---------------------------------------------------------------
    # 1. Register a local Ollama instance (no API key, zero cost)
    # ---------------------------------------------------------------
    registry.register_openai_compatible(
        name="ollama",
        base_url="http://localhost:11434/v1",
        api_key=None,  # no auth needed for local models
        models=[
            ProviderModel(
                provider="ollama",
                model="llama3",
                cost_per_1k_input_tokens=0.0,  # free!
                cost_per_1k_output_tokens=0.0,
                avg_latency_ms=150,
                capabilities=["reasoning", "code_generation"],
                max_context_tokens=8_192,
                tier="fast",
            ),
            ProviderModel(
                provider="ollama",
                model="codellama",
                cost_per_1k_input_tokens=0.0,
                cost_per_1k_output_tokens=0.0,
                avg_latency_ms=200,
                capabilities=["code_generation"],
                max_context_tokens=16_384,
                tier="fast",
            ),
        ],
    )

    # ---------------------------------------------------------------
    # 2. Register a custom corporate API endpoint with auth
    # ---------------------------------------------------------------
    registry.register_openai_compatible(
        name="corp-llm",
        base_url="https://llm.internal.corp.com/v1",
        api_key="corp-secret-token-xyz",
        extra_headers={"X-Team": "platform", "X-Project": "kortex"},
        models=[
            ProviderModel(
                provider="corp-llm",
                model="corp-gpt-internal",
                cost_per_1k_input_tokens=0.001,
                cost_per_1k_output_tokens=0.003,
                avg_latency_ms=500,
                capabilities=["reasoning", "code_generation", "analysis"],
                max_context_tokens=32_000,
                tier="balanced",
            ),
        ],
    )

    # ---------------------------------------------------------------
    # 3. Register cloud Anthropic (mock the key)
    # ---------------------------------------------------------------
    anthropic = AnthropicConnector(api_key="sk-ant-mock-key")
    registry.register_provider(anthropic)  # type: ignore[arg-type]

    # ---------------------------------------------------------------
    # Show what we have
    # ---------------------------------------------------------------
    print("=" * 70)
    print("Custom Provider Example -- Local + Cloud Model Mixing")
    print("=" * 70)

    print(f"\nRegistered providers: {registry.list_providers()}")
    all_models = registry.get_all_models()
    print(f"Total models available: {len(all_models)}\n")

    for m in all_models:
        cost = f"${m.cost_per_1k_input_tokens:.4f}" if m.cost_per_1k_input_tokens > 0 else "FREE"
        print(f"  [{m.tier:>8}] {m.provider}/{m.model}  cost={cost}  latency={m.avg_latency_ms}ms")

    # ---------------------------------------------------------------
    # 4. Build a Router with ALL models and coordinate tasks
    # ---------------------------------------------------------------
    router = Router()
    for model in all_models:
        router.register_model(model)

    state = StateManager(store=InMemoryStateStore())
    runtime = KortexRuntime(router=router, state_manager=state)

    runtime.register_agent(AgentDescriptor("fast-worker", "Fast Worker", "Handles simple tasks"))
    runtime.register_agent(AgentDescriptor("deep-thinker", "Deep Thinker", "Handles complex tasks"))

    async with runtime:
        # --- Simple task: should route to local Ollama (cheapest fast tier) ---
        simple_task = TaskSpec(
            content="Translate 'hello' to Spanish",
            complexity_hint="simple",
            required_capabilities=["reasoning"],
        )

        print(f"\n{'=' * 70}")
        print("SIMPLE TASK (should route to local model)")
        print("=" * 70)

        simple_result = await runtime.coordinate(simple_task, ["fast-worker"])
        decision = simple_result.routing_decisions[0]
        print(f"  Task: {simple_task.content}")
        print(f"  Routed to: {decision.chosen_provider}/{decision.chosen_model}")
        print(f"  Cost: ${decision.estimated_cost_usd:.4f}")
        print(f"  Reasoning: {decision.reasoning}")

        # --- Complex task: should route to powerful tier (Anthropic Opus) ---
        complex_task = TaskSpec(
            content="Analyze the architectural trade-offs of event sourcing vs CQRS",
            complexity_hint="complex",
            required_capabilities=["reasoning", "analysis"],
        )

        print(f"\n{'=' * 70}")
        print("COMPLEX TASK (should route to cloud model)")
        print("=" * 70)

        complex_result = await runtime.coordinate(complex_task, ["deep-thinker"])
        decision = complex_result.routing_decisions[0]
        print(f"  Task: {complex_task.content}")
        print(f"  Routed to: {decision.chosen_provider}/{decision.chosen_model}")
        print(f"  Cost: ${decision.estimated_cost_usd:.4f}")
        print(f"  Reasoning: {decision.reasoning}")

        # --- Mixed pipeline: simple->complex across both providers ---
        print(f"\n{'=' * 70}")
        print("MIXED PIPELINE (local fast + cloud powerful)")
        print("=" * 70)

        mixed_task = TaskSpec(
            content="Research and write a technical design doc",
            complexity_hint="complex",
            required_capabilities=["reasoning"],
        )

        mixed_result = await runtime.coordinate(mixed_task, ["fast-worker", "deep-thinker"])
        print(f"  Task: {mixed_task.content}")
        print(f"  Pipeline: fast-worker -> deep-thinker")
        for i, d in enumerate(mixed_result.routing_decisions):
            print(f"  Step {i + 1}: {d.chosen_provider}/{d.chosen_model} (${d.estimated_cost_usd:.4f})")
        print(f"  Total cost: ${mixed_result.total_estimated_cost_usd:.4f}")
        print(f"  Summary: {runtime.get_coordination_summary(mixed_result)}")

    # ---------------------------------------------------------------
    # Show that Ollama connector has no auth header
    # ---------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("PROVIDER DETAILS")
    print("=" * 70)
    ollama_connector = registry.get_provider("ollama")
    ollama_client = ollama_connector._get_client()  # type: ignore[union-attr]
    has_auth = "authorization" in {k.lower() for k in ollama_client.headers.keys()}
    print(f"  Ollama has auth header: {has_auth} (expected: False)")

    corp_connector = registry.get_provider("corp-llm")
    corp_client = corp_connector._get_client()  # type: ignore[union-attr]
    print(f"  Corp API auth: {corp_client.headers.get('authorization', 'N/A')}")
    print(f"  Corp API X-Team: {corp_client.headers.get('x-team', 'N/A')}")

    print(f"\n{'=' * 70}")
    print("Done. All routing used real Kortex logic, no HTTP calls needed.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
