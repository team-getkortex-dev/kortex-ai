"""Integration tests for the CrewAI adapter.

Does NOT import crewai -- mocks what is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

from kortex.adapters.crewai import (
    KortexCrewAIAdapter,
    _infer_capabilities,
)
from kortex.core.router import ProviderModel, Router
from kortex.core.runtime import AgentDescriptor, KortexRuntime
from kortex.core.state import StateManager
from kortex.core.types import TaskSpec
from kortex.store.memory import InMemoryStateStore


# --- Mock CrewAI structures ---


@dataclass
class FakeAgent:
    role: str
    goal: str = ""
    backstory: str = ""


@dataclass
class FakeTask:
    description: str
    agent: FakeAgent | None = None


@dataclass
class FakeCrew:
    agents: list[FakeAgent] = field(default_factory=list)
    tasks: list[FakeTask] = field(default_factory=list)

    def kickoff(self, **kwargs: Any) -> dict[str, str]:
        return {"status": "done"}


# --- Helpers ---


def _fast_model() -> ProviderModel:
    return ProviderModel(
        provider="openai",
        model="gpt-4o-mini",
        cost_per_1k_input_tokens=0.00015,
        cost_per_1k_output_tokens=0.0006,
        avg_latency_ms=200,
        capabilities=["reasoning"],
        tier="fast",
    )


def _balanced_model() -> ProviderModel:
    return ProviderModel(
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        cost_per_1k_input_tokens=0.003,
        cost_per_1k_output_tokens=0.015,
        avg_latency_ms=800,
        capabilities=["reasoning", "content_generation"],
        tier="balanced",
    )


def _build_runtime(register_models: bool = True) -> KortexRuntime:
    router = Router()
    if register_models:
        router.register_model(_fast_model())
        router.register_model(_balanced_model())
    state = StateManager(store=InMemoryStateStore())
    runtime = KortexRuntime(router=router, state_manager=state)
    runtime.register_agent(AgentDescriptor("researcher", "Researcher", "Researches"))
    runtime.register_agent(AgentDescriptor("writer", "Writer", "Writes"))
    runtime.register_agent(AgentDescriptor("reviewer", "Reviewer", "Reviews"))
    return runtime


def _build_crew() -> FakeCrew:
    analyst = FakeAgent("Research Analyst", goal="Find info")
    writer = FakeAgent("Content Writer", goal="Write content")
    editor = FakeAgent("Editor", goal="Review content")
    return FakeCrew(
        agents=[analyst, writer, editor],
        tasks=[
            FakeTask("Research the topic", agent=analyst),
            FakeTask("Write the article", agent=writer),
            FakeTask("Edit the article", agent=editor),
        ],
    )


# --- 1. wrap_task routes and creates checkpoint ---


class TestWrapTask:
    @pytest.mark.asyncio
    async def test_wrap_task_routes_and_checkpoints(self) -> None:
        runtime = _build_runtime()
        adapter = KortexCrewAIAdapter(runtime)

        @adapter.wrap_task("Research Analyst", "researcher")
        async def research(topic: str) -> dict[str, str]:
            return {"findings": f"Research on {topic}"}

        result = await research("AI coordination")

        assert result == {"findings": "Research on AI coordination"}
        # Verify checkpoint was created
        latest = await runtime._state.get_latest("researcher")
        assert latest is not None
        assert latest.state_snapshot["task_role"] == "Research Analyst"


# --- 2. wrap_task falls back gracefully on KortexError ---


class TestWrapTaskFallback:
    @pytest.mark.asyncio
    async def test_fallback_when_no_models(self) -> None:
        runtime = _build_runtime(register_models=False)
        adapter = KortexCrewAIAdapter(runtime)

        @adapter.wrap_task("Writer", "writer")
        async def write(data: str) -> str:
            return f"Article about {data}"

        # Should not crash -- falls back to direct execution
        result = await write("testing")
        assert result == "Article about testing"


# --- 3. create_agents_from_crew generates correct AgentDescriptors ---


class TestCreateAgentsFromCrew:
    def test_generates_descriptors(self) -> None:
        runtime = _build_runtime()
        adapter = KortexCrewAIAdapter(runtime)
        crew = _build_crew()

        descriptors = adapter.create_agents_from_crew(crew)

        assert len(descriptors) == 3
        ids = {d.agent_id for d in descriptors}
        assert "research-analyst" in ids
        assert "content-writer" in ids
        assert "editor" in ids

        # Check that goals are used as descriptions
        analyst_desc = next(d for d in descriptors if d.agent_id == "research-analyst")
        assert analyst_desc.description == "Find info"
        assert analyst_desc.name == "Research Analyst"


# --- 4. Capabilities inference maps keywords correctly ---


class TestCapabilitiesInference:
    def test_research_keyword(self) -> None:
        caps = _infer_capabilities("Research Analyst")
        assert "research" in caps
        assert "analysis" in caps

    def test_writer_keyword(self) -> None:
        caps = _infer_capabilities("Content Writer")
        assert "content_generation" in caps

    def test_review_keyword(self) -> None:
        caps = _infer_capabilities("Code Reviewer")
        assert "analysis" in caps
        assert "quality_assurance" in caps
        assert "code_generation" in caps

    def test_code_keyword(self) -> None:
        caps = _infer_capabilities("Code Developer")
        assert "code_generation" in caps

    def test_design_keyword(self) -> None:
        caps = _infer_capabilities("UI Designer")
        assert "planning" in caps

    def test_test_keyword(self) -> None:
        caps = _infer_capabilities("QA Tester")
        assert "testing" in caps
        assert "quality_assurance" in caps

    def test_manage_keyword(self) -> None:
        caps = _infer_capabilities("Project Manager")
        assert "planning" in caps

    def test_analysis_keyword(self) -> None:
        caps = _infer_capabilities("Data Analyst")
        assert "analysis" in caps
        assert "data_processing" in caps

    def test_no_match_returns_empty(self) -> None:
        caps = _infer_capabilities("Chef")
        assert caps == []

    def test_deduplication(self) -> None:
        # "Research Analyst" triggers both "research" and "analy"
        # Both map to "analysis" -- should be deduplicated
        caps = _infer_capabilities("Research Analyst")
        assert caps.count("analysis") == 1


# --- 5. Full pipeline through wrap_crew produces correct CoordinationResult ---


class TestFullPipeline:
    @pytest.mark.asyncio
    async def test_wrap_crew_produces_coordination_result(self) -> None:
        runtime = _build_runtime()
        adapter = KortexCrewAIAdapter(runtime)
        crew = _build_crew()

        agent_mapping = {
            "Research Analyst": "researcher",
            "Content Writer": "writer",
            "Editor": "reviewer",
        }

        wrapped = adapter.wrap_crew(crew, agent_mapping)
        crew_output, coordination = await wrapped("Write about AI agents")

        assert crew_output == {"status": "done"}
        assert coordination.task_id is not None
        assert coordination.success is True
        assert len(coordination.routing_decisions) == 3
        assert len(coordination.handoffs) >= 1


# --- 6. Agent mapping links CrewAI roles to Kortex agent_ids ---


class TestAgentMapping:
    @pytest.mark.asyncio
    async def test_mapping_derives_correct_pipeline(self) -> None:
        runtime = _build_runtime()
        adapter = KortexCrewAIAdapter(runtime)
        crew = _build_crew()

        # Map only 2 of 3 agents
        agent_mapping = {
            "Research Analyst": "researcher",
            "Content Writer": "writer",
        }

        wrapped = adapter.wrap_crew(crew, agent_mapping)
        _output, coordination = await wrapped("Selective mapping test")

        # Only mapped agents should be in the pipeline
        assert coordination.success is True
        assert len(coordination.routing_decisions) == 2

    @pytest.mark.asyncio
    async def test_mapping_preserves_task_order(self) -> None:
        runtime = _build_runtime()
        adapter = KortexCrewAIAdapter(runtime)
        crew = _build_crew()

        # Mapping dict in reverse order, but crew tasks define the order
        agent_mapping = {
            "Editor": "reviewer",
            "Content Writer": "writer",
            "Research Analyst": "researcher",
        }

        wrapped = adapter.wrap_crew(crew, agent_mapping)
        _output, coordination = await wrapped("Order test")

        # First handoff target should be researcher (first in crew task order)
        assert coordination.handoffs[0].target_agent == "researcher"
