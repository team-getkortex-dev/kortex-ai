"""Kortex coordination example -- simulated 3-agent workflow.

Demonstrates both dry-run (execute=False) and live execution (execute=True)
modes using a researcher -> writer -> reviewer pipeline.
No LLM API key required -- HTTP calls are mocked for the execution demo.

Run with:
    uv run python examples/langgraph_example.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx

from kortex.core.router import ProviderModel, Router
from kortex.core.runtime import AgentDescriptor, KortexRuntime
from kortex.core.state import StateManager
from kortex.core.types import TaskSpec
from kortex.providers.base import GenericOpenAIConnector
from kortex.providers.registry import ProviderRegistry
from kortex.store.memory import InMemoryStateStore


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def _models() -> list[ProviderModel]:
    return [
        ProviderModel(
            provider="openai",
            model="gpt-4o-mini",
            cost_per_1k_input_tokens=0.00015,
            cost_per_1k_output_tokens=0.0006,
            avg_latency_ms=200,
            capabilities=["reasoning", "summarization"],
            tier="fast",
        ),
        ProviderModel(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            cost_per_1k_input_tokens=0.003,
            cost_per_1k_output_tokens=0.015,
            avg_latency_ms=800,
            capabilities=["reasoning", "code_generation", "writing"],
            tier="balanced",
        ),
        ProviderModel(
            provider="anthropic",
            model="claude-opus-4-20250514",
            cost_per_1k_input_tokens=0.015,
            cost_per_1k_output_tokens=0.075,
            avg_latency_ms=2000,
            capabilities=["reasoning", "code_generation", "writing", "deep_analysis"],
            tier="powerful",
        ),
    ]


def _agents() -> list[AgentDescriptor]:
    return [
        AgentDescriptor("researcher", "Researcher", "Gathers info", ["reasoning", "summarization"]),
        AgentDescriptor("writer", "Writer", "Drafts articles", ["writing"]),
        AgentDescriptor("reviewer", "Reviewer", "Reviews drafts", ["reasoning", "deep_analysis"]),
    ]


def build_runtime(registry: ProviderRegistry | None = None) -> KortexRuntime:
    """Build a fully configured Kortex runtime."""
    router = Router()
    for m in _models():
        router.register_model(m)

    state = StateManager(store=InMemoryStateStore())
    runtime = KortexRuntime(router=router, state_manager=state, registry=registry)
    for a in _agents():
        runtime.register_agent(a)
    return runtime


def print_result(runtime: KortexRuntime, result: Any, label: str) -> None:
    """Pretty-print a coordination result."""
    print(f"\n{'=' * 70}")
    print(label)
    print("=" * 70)
    print(runtime.get_coordination_summary(result))

    print(f"\n  Routing decisions:")
    for i, d in enumerate(result.routing_decisions, 1):
        print(f"    Step {i}: {d.chosen_provider}/{d.chosen_model} "
              f"(est. ${d.estimated_cost_usd:.4f}) -- {d.reasoning}")

    print(f"\n  Handoff chain:")
    for h in result.handoffs:
        print(f"    {h.source_agent} -> {h.target_agent} "
              f"[{h.checkpoint_id[:8]}...]")

    # Show step execution records with LLM outputs
    if result.steps:
        print(f"\n  Step Execution Records ({len(result.steps)}):")
        for step in result.steps:
            pr = step.get("provider_response")
            if pr:
                content_preview = pr["content"][:60] + "..." if len(pr["content"]) > 60 else pr["content"]
                print(f"    Step {step['step_index']}: {step['agent_id']}")
                print(f"      LLM output: {content_preview}")

    print(f"\n  Cost:")
    print(f"    Estimated: ${result.total_estimated_cost_usd:.4f}")
    if result.responses:
        print(f"    Actual:    ${result.actual_cost_usd:.4f}")
        if result.total_estimated_cost_usd > 0:
            pct = (result.total_estimated_cost_usd - result.actual_cost_usd) / result.total_estimated_cost_usd * 100
            print(f"    Savings:   {pct:.0f}%")

    print(f"\n  Events ({len(result.events)}):")
    for e in result.events:
        agent = e.agent_id or "system"
        print(f"    [{e.event_type:>10}] {agent}")

    if result.responses:
        print(f"\n  LLM Responses ({len(result.responses)}):")
        for i, r in enumerate(result.responses, 1):
            content = r["content"]
            preview = content[:70] + "..." if len(content) > 70 else content
            print(f"    {i}. [{r['provider']}/{r['model']}] "
                  f"tokens={r['input_tokens']}+{r['output_tokens']} "
                  f"cost=${r['cost_usd']:.4f}")
            print(f"       {preview}")


# ---------------------------------------------------------------------------
# Mock HTTP responses for execute=True demo
# ---------------------------------------------------------------------------

_MOCK_RESPONSES = [
    {
        "choices": [{"message": {"content":
            "Research findings:\n- Multi-agent systems are evolving rapidly\n"
            "- Key challenge: coordination overhead\n- Promising: shared memory architectures"
        }}],
        "usage": {"prompt_tokens": 45, "completion_tokens": 30},
    },
    {
        "choices": [{"message": {"content":
            "DRAFT: The Future of Multi-Agent AI\n\n"
            "Multi-agent AI systems are transforming how we build intelligent software. "
            "This article explores coordination patterns, shared memory, and cost optimization."
        }}],
        "usage": {"prompt_tokens": 80, "completion_tokens": 40},
    },
    {
        "choices": [{"message": {"content":
            "REVIEW: Approved with minor edits. The draft covers key points well. "
            "Suggest adding a section on failure recovery patterns."
        }}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 25},
    },
]


def _mock_httpx_response(data: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        status_code=200,
        json=data,
        request=httpx.Request("POST", "https://mock"),
    )


async def main() -> None:
    """Run the 3-agent coordination pipeline in both modes."""
    task = TaskSpec(
        content="Write an article about the future of multi-agent AI systems",
        complexity_hint="complex",
        required_capabilities=["reasoning"],
        metadata={"author": "demo"},
    )

    print("=" * 70)
    print("Kortex Coordination Example")
    print("=" * 70)
    print(f"Task: {task.content}")
    print(f"Task ID: {task.task_id}")

    # -----------------------------------------------------------------
    # Mode 1: Dry run (execute=False) -- no API calls
    # -----------------------------------------------------------------
    async with build_runtime() as runtime_dry:
        dry_result = await runtime_dry.coordinate(
            task, agent_pipeline=["researcher", "writer", "reviewer"],
        )
        print_result(runtime_dry, dry_result, "MODE 1: DRY RUN (execute=False)")
        assert len(dry_result.responses) == 0
        assert dry_result.actual_cost_usd == 0.0

    # -----------------------------------------------------------------
    # Mode 2: Live execution (execute=True) with mocked HTTP
    # -----------------------------------------------------------------
    # Build a registry with connectors that use GenericOpenAIConnector
    registry = ProviderRegistry()

    # Register connectors matching the models in the router
    registry.register_openai_compatible(
        name="anthropic",
        base_url="https://mock-anthropic.test/v1",
        api_key="mock-key",
        models=[m for m in _models() if m.provider == "anthropic"],
    )
    registry.register_openai_compatible(
        name="openai",
        base_url="https://mock-openai.test/v1",
        api_key="mock-key",
        models=[m for m in _models() if m.provider == "openai"],
    )

    # Mock the HTTP calls -- return pre-canned responses in order
    call_count = {"n": 0}

    async def mock_post(*args: Any, **kwargs: Any) -> httpx.Response:
        idx = min(call_count["n"], len(_MOCK_RESPONSES) - 1)
        call_count["n"] += 1
        return _mock_httpx_response(_MOCK_RESPONSES[idx])

    # Patch all GenericOpenAIConnector clients
    for pname in registry.list_providers():
        connector = registry.get_provider(pname)
        if hasattr(connector, "_get_client"):
            client = connector._get_client()  # type: ignore[union-attr]
            client.post = AsyncMock(side_effect=mock_post)  # type: ignore[method-assign]

    async with build_runtime(registry=registry) as runtime_exec:
        exec_result = await runtime_exec.coordinate(
            task, agent_pipeline=["researcher", "writer", "reviewer"],
            execute=True,
        )
        print_result(runtime_exec, exec_result, "MODE 2: LIVE EXECUTION (execute=True, mocked HTTP)")

        assert len(exec_result.responses) == 3
        assert exec_result.actual_cost_usd > 0
        assert exec_result.actual_cost_usd < exec_result.total_estimated_cost_usd

    print(f"\n{'=' * 70}")
    print("Both modes completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
