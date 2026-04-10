"""Integration tests for the LangGraph adapter.

Does NOT import langgraph — mocks what is needed.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from kortex.adapters.langgraph import KortexLangGraphAdapter
from kortex.core.router import ProviderModel, Router
from kortex.core.runtime import AgentDescriptor, KortexRuntime
from kortex.core.state import StateManager
from kortex.core.types import TaskSpec
from kortex.store.memory import InMemoryStateStore


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


def _mock_graph(node_names: list[str]) -> MagicMock:
    """Create a mock LangGraph graph with ordered nodes."""
    graph = MagicMock()
    # Use a regular dict to preserve insertion order
    graph.nodes = {name: MagicMock() for name in node_names}
    graph.invoke = MagicMock(return_value={"result": "done"})
    return graph


# --- 1. wrap_node routes and creates checkpoint ---


class TestWrapNode:
    @pytest.mark.asyncio
    async def test_wrap_node_routes_and_checkpoints(self) -> None:
        runtime = _build_runtime()
        adapter = KortexLangGraphAdapter(runtime)

        @adapter.wrap_node("research_node", "researcher")
        async def research(topic: str) -> dict[str, str]:
            return {"findings": f"Research on {topic}"}

        result = await research("AI agents")

        assert result == {"findings": "Research on AI agents"}
        # Verify checkpoint was created
        latest = await runtime._state.get_latest("researcher")
        assert latest is not None
        assert latest.state_snapshot["node_name"] == "research_node"


# --- 2. wrap_node falls back gracefully ---


class TestWrapNodeFallback:
    @pytest.mark.asyncio
    async def test_fallback_when_no_models(self) -> None:
        runtime = _build_runtime(register_models=False)
        adapter = KortexLangGraphAdapter(runtime)

        @adapter.wrap_node("write_node", "writer")
        async def write(data: str) -> str:
            return f"Article about {data}"

        # Should not crash — falls back to direct execution
        result = await write("testing")
        assert result == "Article about testing"


# --- 3. Full pipeline through adapter ---


class TestFullPipeline:
    @pytest.mark.asyncio
    async def test_wrap_graph_produces_coordination_result(self) -> None:
        runtime = _build_runtime()
        adapter = KortexLangGraphAdapter(runtime)

        graph = _mock_graph(["research_node", "write_node", "review_node"])
        agent_mapping = {
            "research_node": "researcher",
            "write_node": "writer",
            "review_node": "reviewer",
        }

        wrapped = adapter.wrap_graph(graph, agent_mapping)
        graph_output, coordination = await wrapped("Write about AI")

        assert graph_output == {"result": "done"}
        assert coordination.task_id is not None
        assert coordination.success is True
        assert len(coordination.routing_decisions) == 3
        assert len(coordination.handoffs) >= 1
        graph.invoke.assert_called_once_with("Write about AI")


# --- 4. Agent mapping links node names to agent_ids ---


class TestAgentMapping:
    @pytest.mark.asyncio
    async def test_mapping_derives_correct_pipeline(self) -> None:
        runtime = _build_runtime()
        adapter = KortexLangGraphAdapter(runtime)

        # Graph with 4 nodes, but only 2 are mapped
        graph = _mock_graph(["preprocess", "research_node", "postprocess", "write_node"])
        agent_mapping = {
            "research_node": "researcher",
            "write_node": "writer",
        }

        wrapped = adapter.wrap_graph(graph, agent_mapping)
        _output, coordination = await wrapped("Selective mapping test")

        # Only mapped nodes should be in the pipeline
        assert coordination.success is True
        assert len(coordination.routing_decisions) == 2

        # Verify the pipeline used the right agent_ids by checking handoffs
        agent_ids_in_handoffs = {h.target_agent for h in coordination.handoffs}
        assert "researcher" in agent_ids_in_handoffs

    @pytest.mark.asyncio
    async def test_mapping_preserves_graph_node_order(self) -> None:
        runtime = _build_runtime()
        adapter = KortexLangGraphAdapter(runtime)

        # Mapping dict is in reverse order, but graph nodes define the order
        graph = _mock_graph(["research_node", "write_node"])
        agent_mapping = {
            "write_node": "writer",
            "research_node": "researcher",
        }

        wrapped = adapter.wrap_graph(graph, agent_mapping)
        _output, coordination = await wrapped("Order test")

        # The first handoff target should be researcher (first in graph node order)
        assert coordination.handoffs[0].target_agent == "researcher"
