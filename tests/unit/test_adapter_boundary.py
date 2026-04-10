"""Tests verifying that adapters use only the public KortexRuntime API.

No adapter should access private attributes (._<name>) on the runtime.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from kortex.adapters.crewai import KortexCrewAIAdapter
from kortex.adapters.langgraph import KortexLangGraphAdapter, kortex_middleware
from kortex.core.exceptions import KortexError
from kortex.core.runtime import KortexRuntime
from kortex.core.types import HandoffContext, RoutingDecision, TaskSpec


# ---------------------------------------------------------------------------
# Helper: collect all private-attribute accesses in an adapter source file
# ---------------------------------------------------------------------------


def _find_runtime_private_accesses(source_path: Path) -> list[str]:
    """Return a list of lines containing runtime._<attr> accesses."""
    source = source_path.read_text(encoding="utf-8")
    hits = []
    for lineno, line in enumerate(source.splitlines(), start=1):
        # Match patterns like runtime._ or self._runtime._ or adapter._runtime._
        stripped = line.strip()
        if "_runtime._" in stripped or "runtime._" in stripped:
            hits.append(f"  line {lineno}: {stripped}")
    return hits


ADAPTERS_DIR = Path(__file__).parent.parent.parent / "src" / "kortex" / "adapters"


# ---------------------------------------------------------------------------
# 1. langgraph.py has zero private runtime attribute accesses
# ---------------------------------------------------------------------------


class TestLangGraphBoundary:
    def test_no_private_access_in_langgraph(self) -> None:
        hits = _find_runtime_private_accesses(ADAPTERS_DIR / "langgraph.py")
        assert hits == [], (
            "langgraph.py accesses private runtime attributes:\n" + "\n".join(hits)
        )


# ---------------------------------------------------------------------------
# 2. crewai.py has zero private runtime attribute accesses
# ---------------------------------------------------------------------------


class TestCrewAIBoundary:
    def test_no_private_access_in_crewai(self) -> None:
        hits = _find_runtime_private_accesses(ADAPTERS_DIR / "crewai.py")
        assert hits == [], (
            "crewai.py accesses private runtime attributes:\n" + "\n".join(hits)
        )


# ---------------------------------------------------------------------------
# Helper: build a mock runtime with public methods
# ---------------------------------------------------------------------------


def _mock_runtime() -> MagicMock:
    runtime = MagicMock(spec=KortexRuntime)

    decision = RoutingDecision(
        task_id="t1",
        chosen_provider="test",
        chosen_model="fast-model",
        reasoning="unit test",
        estimated_cost_usd=0.001,
        estimated_latency_ms=100,
    )
    handoff_ctx = HandoffContext(
        checkpoint_id="ckpt-1",
        source_agent="src",
        target_agent="tgt",
        state_snapshot={},
    )

    runtime.route_task = AsyncMock(return_value=decision)
    runtime.persist_handoff = AsyncMock(return_value=handoff_ctx)
    runtime.coordinate = AsyncMock()
    return runtime


# ---------------------------------------------------------------------------
# 3. LangGraph adapter uses only the public API
# ---------------------------------------------------------------------------


class TestLangGraphPublicAPI:
    @pytest.mark.asyncio
    async def test_wrap_node_uses_public_api(self) -> None:
        runtime = _mock_runtime()
        adapter = KortexLangGraphAdapter(runtime)  # type: ignore[arg-type]

        @adapter.wrap_node(node_name="node_a", agent_id="agent-1")
        async def dummy_node(state: dict) -> dict:
            return {"done": True}

        result = await dummy_node({"input": "hello"})

        # Public methods should have been called, not private attrs
        runtime.route_task.assert_awaited_once()
        runtime.persist_handoff.assert_awaited_once()
        assert result == {"done": True}


# ---------------------------------------------------------------------------
# 4. CrewAI adapter uses only the public API
# ---------------------------------------------------------------------------


class TestCrewAIPublicAPI:
    @pytest.mark.asyncio
    async def test_wrap_task_uses_public_api(self) -> None:
        runtime = _mock_runtime()
        adapter = KortexCrewAIAdapter(runtime)  # type: ignore[arg-type]

        @adapter.wrap_task(task_role="researcher", agent_id="agent-2")
        async def dummy_task() -> dict:
            return {"findings": "42"}

        result = await dummy_task()

        runtime.route_task.assert_awaited_once()
        runtime.persist_handoff.assert_awaited_once()
        assert result == {"findings": "42"}


# ---------------------------------------------------------------------------
# 5. Adapters fall back gracefully when runtime methods raise KortexError
# ---------------------------------------------------------------------------


class TestAdapterFallback:
    @pytest.mark.asyncio
    async def test_langgraph_fallback_on_route_error(self) -> None:
        runtime = _mock_runtime()
        runtime.route_task = AsyncMock(side_effect=KortexError("router down"))
        runtime.persist_handoff = AsyncMock(side_effect=KortexError("state unavailable"))

        adapter = KortexLangGraphAdapter(runtime)  # type: ignore[arg-type]

        @adapter.wrap_node(node_name="node_b", agent_id="agent-3")
        async def dummy_node(state: dict) -> dict:
            return {"ok": True}

        # Should not raise — adapters are fail-open
        result = await dummy_node({})
        assert result == {"ok": True}

    @pytest.mark.asyncio
    async def test_crewai_fallback_on_route_error(self) -> None:
        runtime = _mock_runtime()
        runtime.route_task = AsyncMock(side_effect=KortexError("router down"))
        runtime.persist_handoff = AsyncMock(side_effect=KortexError("state unavailable"))

        adapter = KortexCrewAIAdapter(runtime)  # type: ignore[arg-type]

        @adapter.wrap_task(task_role="writer", agent_id="agent-4")
        async def dummy_task() -> str:
            return "output"

        # Should not raise — adapters are fail-open
        result = await dummy_task()
        assert result == "output"
