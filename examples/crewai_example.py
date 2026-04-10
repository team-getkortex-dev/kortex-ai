"""Kortex CrewAI integration example -- simulated 3-agent crew.

Demonstrates how Kortex wraps a CrewAI crew to add intelligent routing,
cost tracking, and stateful handoffs. No CrewAI install or API keys needed --
the crew structure is mocked.

Run with:
    python examples/crewai_example.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx

from kortex.adapters.crewai import KortexCrewAIAdapter
from kortex.core.router import ProviderModel, Router
from kortex.core.runtime import AgentDescriptor, KortexRuntime
from kortex.core.state import StateManager
from kortex.core.types import TaskSpec
from kortex.providers.base import GenericOpenAIConnector
from kortex.providers.registry import ProviderRegistry
from kortex.store.memory import InMemoryStateStore


# ---------------------------------------------------------------------------
# Mock CrewAI structures (no crewai install needed)
# ---------------------------------------------------------------------------


@dataclass
class MockCrewAgent:
    """Simulates a CrewAI Agent."""

    role: str
    goal: str = ""
    backstory: str = ""


@dataclass
class MockCrewTask:
    """Simulates a CrewAI Task."""

    description: str
    agent: MockCrewAgent | None = None


@dataclass
class MockCrew:
    """Simulates a CrewAI Crew."""

    agents: list[MockCrewAgent] = field(default_factory=list)
    tasks: list[MockCrewTask] = field(default_factory=list)

    def kickoff(self, **kwargs: Any) -> dict[str, str]:
        return {"status": "completed", "output": "Crew execution finished"}


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


def build_runtime(registry: ProviderRegistry | None = None) -> KortexRuntime:
    """Build a fully configured Kortex runtime."""
    router = Router()
    for m in _models():
        router.register_model(m)

    state = StateManager(store=InMemoryStateStore())
    return KortexRuntime(router=router, state_manager=state, registry=registry)


def build_crew() -> MockCrew:
    """Build a mock CrewAI crew with 3 agents."""
    analyst = MockCrewAgent(
        role="Research Analyst",
        goal="Find and synthesize relevant information",
        backstory="Senior research analyst with expertise in data analysis",
    )
    writer = MockCrewAgent(
        role="Content Writer",
        goal="Produce clear, engaging content",
        backstory="Experienced technical writer",
    )
    editor = MockCrewAgent(
        role="Editor",
        goal="Review and polish final content",
        backstory="Meticulous editor with an eye for detail",
    )

    return MockCrew(
        agents=[analyst, writer, editor],
        tasks=[
            MockCrewTask("Research the topic thoroughly", agent=analyst),
            MockCrewTask("Write a comprehensive article", agent=writer),
            MockCrewTask("Review and edit the article", agent=editor),
        ],
    )


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

    print(f"\n  Cost:")
    print(f"    Estimated: ${result.total_estimated_cost_usd:.4f}")
    if result.responses:
        print(f"    Actual:    ${result.actual_cost_usd:.4f}")


# ---------------------------------------------------------------------------
# Mock HTTP responses for execute=True demo
# ---------------------------------------------------------------------------

_MOCK_RESPONSES = [
    {
        "choices": [{"message": {"content":
            "Research findings: Multi-agent systems show 40% efficiency gains "
            "when using intelligent task routing over static assignment."
        }}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 25},
    },
    {
        "choices": [{"message": {"content":
            "ARTICLE: The Rise of Intelligent Agent Coordination\n\n"
            "New approaches to multi-agent coordination are transforming AI workflows..."
        }}],
        "usage": {"prompt_tokens": 85, "completion_tokens": 45},
    },
    {
        "choices": [{"message": {"content":
            "REVIEW: Article approved. Strong opening, clear structure. "
            "Minor suggestion: add metrics on coordination overhead."
        }}],
        "usage": {"prompt_tokens": 110, "completion_tokens": 20},
    },
]


def _mock_httpx_response(data: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        status_code=200,
        json=data,
        request=httpx.Request("POST", "https://mock"),
    )


async def main() -> None:
    """Run the CrewAI integration demo."""
    crew = build_crew()

    print("=" * 70)
    print("Kortex CrewAI Integration Example")
    print("=" * 70)

    # -----------------------------------------------------------------
    # Step 1: Auto-generate AgentDescriptors from crew
    # -----------------------------------------------------------------
    async with build_runtime() as runtime:
        adapter = KortexCrewAIAdapter(runtime)

        descriptors = adapter.create_agents_from_crew(crew)

        print("\nAuto-generated AgentDescriptors from CrewAI crew:")
        for desc in descriptors:
            print(f"  {desc.agent_id}: {desc.name}")
            print(f"    Capabilities: {desc.capabilities}")
            print(f"    Description: {desc.description}")

        # Register auto-generated agents
        for desc in descriptors:
            runtime.register_agent(desc)

        # -----------------------------------------------------------------
        # Step 2: Dry-run mode (no API calls)
        # -----------------------------------------------------------------
        agent_mapping = {
            "Research Analyst": "research-analyst",
            "Content Writer": "content-writer",
            "Editor": "editor",
        }

        wrapped = adapter.wrap_crew(crew, agent_mapping)
        crew_output, dry_result = await wrapped(
            "Write an article about multi-agent coordination patterns",
        )
        print_result(runtime, dry_result, "MODE 1: DRY RUN (execute=False)")
        print(f"\n  Crew output: {crew_output}")

    # -----------------------------------------------------------------
    # Step 3: Execute mode with mocked HTTP
    # -----------------------------------------------------------------
    registry = ProviderRegistry()
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

    # Mock HTTP calls
    call_count = {"n": 0}

    async def mock_post(*args: Any, **kwargs: Any) -> httpx.Response:
        idx = min(call_count["n"], len(_MOCK_RESPONSES) - 1)
        call_count["n"] += 1
        return _mock_httpx_response(_MOCK_RESPONSES[idx])

    for pname in registry.list_providers():
        connector = registry.get_provider(pname)
        if hasattr(connector, "_get_client"):
            client = connector._get_client()  # type: ignore[union-attr]
            client.post = AsyncMock(side_effect=mock_post)  # type: ignore[method-assign]

    async with build_runtime(registry=registry) as exec_runtime:
        for desc in descriptors:
            exec_runtime.register_agent(desc)
        exec_adapter = KortexCrewAIAdapter(exec_runtime)

        exec_wrapped = exec_adapter.wrap_crew(crew, agent_mapping)
        crew_output, exec_result = await exec_wrapped(
            "Write an article about multi-agent coordination patterns",
            execute=True,
        )
        print_result(exec_runtime, exec_result, "MODE 2: LIVE EXECUTION (mocked HTTP)")

        if exec_result.responses:
            print(f"\n  LLM Responses ({len(exec_result.responses)}):")
            for i, r in enumerate(exec_result.responses, 1):
                content = r["content"]
                preview = content[:70] + "..." if len(content) > 70 else content
                print(f"    {i}. [{r['provider']}/{r['model']}] "
                      f"tokens={r['input_tokens']}+{r['output_tokens']} "
                      f"cost=${r['cost_usd']:.4f}")
                print(f"       {preview}")

    print(f"\n{'=' * 70}")
    print("CrewAI integration demo completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
