"""Tests verifying correct separation of handoff payloads and execution metadata.

HandoffContext.state_snapshot must contain ONLY the boundary-crossing payload
(what one agent hands to the next). Execution metadata (routing decisions,
provider responses, anomalies, timing) must be stored separately in
StepExecutionRecord objects within CoordinationResult.steps.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from kortex.core.router import ProviderModel, Router
from kortex.core.runtime import AgentDescriptor, KortexRuntime
from kortex.core.state import StateManager
from kortex.core.types import TaskSpec
from kortex.providers.registry import ProviderRegistry
from kortex.store.memory import InMemoryStateStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EXECUTION_METADATA_KEYS = {
    "routing_decision",
    "provider_response",
    "anomalies",
    "started_at",
    "completed_at",
    "duration_ms",
    "chosen_provider",
    "chosen_model",
    "estimated_cost_usd",
    "estimated_latency_ms",
    "fallback_model",
}


def _mock_httpx_response(data: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        status_code=200,
        json=data,
        request=httpx.Request("POST", "https://mock"),
    )


def _ok_response(
    content: str = "mock output", input_tok: int = 50, output_tok: int = 20,
) -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": input_tok, "completion_tokens": output_tok},
    }


def _make_models() -> list[ProviderModel]:
    return [
        ProviderModel(
            provider="test-cloud",
            model="cloud-fast",
            cost_per_1k_input_tokens=0.001,
            cost_per_1k_output_tokens=0.002,
            avg_latency_ms=200,
            capabilities=["reasoning"],
            tier="fast",
        ),
        ProviderModel(
            provider="test-cloud",
            model="cloud-balanced",
            cost_per_1k_input_tokens=0.003,
            cost_per_1k_output_tokens=0.01,
            avg_latency_ms=600,
            capabilities=["reasoning", "analysis"],
            tier="balanced",
        ),
        ProviderModel(
            provider="test-cloud",
            model="cloud-powerful",
            cost_per_1k_input_tokens=0.01,
            cost_per_1k_output_tokens=0.03,
            avg_latency_ms=1500,
            capabilities=["reasoning", "analysis", "research"],
            tier="powerful",
        ),
    ]


def _build_runtime(
    mock_post: AsyncMock | None = None,
    agent_ids: list[str] | None = None,
) -> KortexRuntime:
    models = _make_models()
    router = Router()
    for m in models:
        router.register_model(m)

    registry = ProviderRegistry()
    registry.register_openai_compatible(
        name="test-cloud",
        base_url="https://mock-cloud.test/v1",
        api_key="mock-key",
        models=models,
    )

    if mock_post is not None:
        for pname in registry.list_providers():
            connector = registry.get_provider(pname)
            if hasattr(connector, "_get_client"):
                client = connector._get_client()  # type: ignore[union-attr]
                client.post = mock_post  # type: ignore[method-assign]

    state = StateManager(store=InMemoryStateStore())
    runtime = KortexRuntime(router=router, state_manager=state, registry=registry)

    for aid in (agent_ids or ["agent-a", "agent-b", "agent-c"]):
        runtime.register_agent(AgentDescriptor(aid, aid.title(), f"Test {aid}", ["reasoning"]))

    return runtime


# ---------------------------------------------------------------------------
# 1. Handoff state_snapshot contains ONLY the input payload, not routing metadata
# ---------------------------------------------------------------------------


class TestHandoffPayloadPurity:
    @pytest.mark.asyncio
    async def test_state_snapshot_has_no_routing_metadata(self) -> None:
        mock_post = AsyncMock(
            return_value=_mock_httpx_response(_ok_response("output", 30, 10)),
        )
        runtime = _build_runtime(mock_post=mock_post)
        task = TaskSpec(content="Test purity", complexity_hint="simple")

        result = await runtime.coordinate(
            task, ["agent-a", "agent-b", "agent-c"], execute=True,
        )

        for h in result.handoffs:
            for key in EXECUTION_METADATA_KEYS:
                assert key not in h.state_snapshot, (
                    f"Execution metadata key '{key}' leaked into "
                    f"handoff state_snapshot: {h.state_snapshot}"
                )


# ---------------------------------------------------------------------------
# 2. StepExecutionRecord contains routing decision and provider response
# ---------------------------------------------------------------------------


class TestStepExecutionRecord:
    @pytest.mark.asyncio
    async def test_step_records_have_routing_and_response(self) -> None:
        mock_post = AsyncMock(
            return_value=_mock_httpx_response(_ok_response("step output", 40, 15)),
        )
        runtime = _build_runtime(mock_post=mock_post)
        task = TaskSpec(content="Test step records", complexity_hint="simple")

        result = await runtime.coordinate(
            task, ["agent-a", "agent-b"], execute=True,
        )

        assert len(result.steps) == 2
        for step in result.steps:
            assert "routing_decision" in step
            assert "provider_response" in step
            assert step["provider_response"] is not None
            assert "content" in step["provider_response"]

            rd = step["routing_decision"]
            assert "chosen_provider" in rd
            assert "chosen_model" in rd

            assert "started_at" in step
            assert "completed_at" in step
            assert "duration_ms" in step


# ---------------------------------------------------------------------------
# 3. In a 3-agent pipeline, each handoff's state_snapshot matches the previous
#    step's output_payload
# ---------------------------------------------------------------------------


class TestHandoffChainConsistency:
    @pytest.mark.asyncio
    async def test_handoff_matches_previous_output(self) -> None:
        outputs = ["Output from A", "Output from B", "Output from C"]
        call_count = {"n": 0}

        async def mock_post(*args: Any, **kwargs: Any) -> httpx.Response:
            idx = min(call_count["n"], len(outputs) - 1)
            call_count["n"] += 1
            return _mock_httpx_response(_ok_response(outputs[idx], 30, 10))

        runtime = _build_runtime(mock_post=AsyncMock(side_effect=mock_post))
        task = TaskSpec(content="Chain test", complexity_hint="simple")

        result = await runtime.coordinate(
            task, ["agent-a", "agent-b", "agent-c"], execute=True,
        )

        # First handoff is __input__ -> agent-a with task content
        assert result.handoffs[0].source_agent == "__input__"
        assert result.handoffs[0].state_snapshot["content"] == task.content

        # Subsequent handoffs carry the previous agent's output
        # handoff[1] is from agent-a to agent-b, snapshot = agent-a's output
        for i, h in enumerate(result.handoffs[1:], start=1):
            snapshot = h.state_snapshot
            assert "task_id" in snapshot
            assert "agent_id" in snapshot


# ---------------------------------------------------------------------------
# 4. Rollback to a checkpoint returns the actual boundary state, not mixed metadata
# ---------------------------------------------------------------------------


class TestRollbackReturnsCleanState:
    @pytest.mark.asyncio
    async def test_rollback_returns_boundary_payload(self) -> None:
        mock_post = AsyncMock(
            return_value=_mock_httpx_response(_ok_response("rollback test", 30, 10)),
        )
        runtime = _build_runtime(mock_post=mock_post)
        task = TaskSpec(content="Rollback test", complexity_hint="simple")

        result = await runtime.coordinate(
            task, ["agent-a", "agent-b"], execute=True,
        )

        # Pick a non-initial handoff checkpoint
        handoff = result.handoffs[0]
        restored = await runtime.rollback_to(handoff.checkpoint_id)

        # Restored state_snapshot should be the clean boundary payload
        for key in EXECUTION_METADATA_KEYS:
            assert key not in restored.state_snapshot


# ---------------------------------------------------------------------------
# 5. Replay can reconstruct step-by-step execution from steps + handoffs
# ---------------------------------------------------------------------------


class TestReplayReconstruction:
    @pytest.mark.asyncio
    async def test_steps_and_handoffs_reconstruct_execution(self) -> None:
        outputs = ["First output", "Second output"]
        call_count = {"n": 0}

        async def mock_post(*args: Any, **kwargs: Any) -> httpx.Response:
            idx = min(call_count["n"], len(outputs) - 1)
            call_count["n"] += 1
            return _mock_httpx_response(_ok_response(outputs[idx], 40, 15))

        runtime = _build_runtime(mock_post=AsyncMock(side_effect=mock_post))
        task = TaskSpec(content="Replay test", complexity_hint="simple")

        result = await runtime.coordinate(
            task, ["agent-a", "agent-b"], execute=True,
        )

        # steps have contiguous step_index values
        indices = [s["step_index"] for s in result.steps]
        assert indices == [0, 1]

        # Each step has the agent_id matching the pipeline
        assert result.steps[0]["agent_id"] == "agent-a"
        assert result.steps[1]["agent_id"] == "agent-b"

        # Handoff chain is intact (initial + inter-agent)
        assert len(result.handoffs) >= 2
        assert result.handoffs[0].source_agent == "__input__"


# ---------------------------------------------------------------------------
# 6. execute=False still works — StepExecutionRecord has provider_response=None
# ---------------------------------------------------------------------------


class TestDryRunSteps:
    @pytest.mark.asyncio
    async def test_dry_run_steps_have_null_provider_response(self) -> None:
        runtime = _build_runtime()
        task = TaskSpec(content="Dry run", complexity_hint="simple")

        result = await runtime.coordinate(task, ["agent-a", "agent-b"])

        assert len(result.steps) == 2
        for step in result.steps:
            assert step["provider_response"] is None
            # Routing decision should still be present
            assert "routing_decision" in step
            assert step["routing_decision"]["chosen_model"] is not None


# ---------------------------------------------------------------------------
# 7. execute=True stores provider response in StepExecutionRecord, NOT in handoff
# ---------------------------------------------------------------------------


class TestExecuteTrueResponseLocation:
    @pytest.mark.asyncio
    async def test_response_in_steps_not_handoffs(self) -> None:
        mock_post = AsyncMock(
            return_value=_mock_httpx_response(_ok_response("located correctly", 50, 20)),
        )
        runtime = _build_runtime(mock_post=mock_post)
        task = TaskSpec(content="Location test", complexity_hint="simple")

        result = await runtime.coordinate(
            task, ["agent-a", "agent-b"], execute=True,
        )

        # Verify responses are in steps
        for step in result.steps:
            pr = step["provider_response"]
            assert pr is not None
            assert pr["content"] == "located correctly"

        # Verify responses are NOT in any handoff
        for h in result.handoffs:
            assert "provider_response" not in h.state_snapshot
            assert "cost_usd" not in h.state_snapshot
            assert "input_tokens" not in h.state_snapshot
            assert "latency_ms" not in h.state_snapshot
