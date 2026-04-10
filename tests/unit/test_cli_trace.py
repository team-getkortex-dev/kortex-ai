"""Unit tests for trace, replay, and policy CLI commands."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from kortex.core.policy import RoutingPolicy
from kortex.core.router import ProviderModel, Router
from kortex.core.runtime import AgentDescriptor, KortexRuntime
from kortex.core.state import StateManager
from kortex.core.trace import TaskTrace, TraceStep
from kortex.core.trace_store import InMemoryTraceStore
from kortex.core.types import TaskSpec
from kortex.dashboard.cli import KortexCLI, _build_demo_trace, main
from kortex.providers.registry import ProviderRegistry
from kortex.store.memory import InMemoryStateStore
from unittest.mock import AsyncMock, MagicMock


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _models() -> list[ProviderModel]:
    return [
        ProviderModel(
            provider="local", model="tiny-7b",
            cost_per_1k_input_tokens=0.0001, cost_per_1k_output_tokens=0.0002,
            avg_latency_ms=50, capabilities=["reasoning", "content_generation"],
            tier="fast",
        ),
        ProviderModel(
            provider="anthropic", model="claude-sonnet",
            cost_per_1k_input_tokens=0.003, cost_per_1k_output_tokens=0.015,
            avg_latency_ms=800,
            capabilities=["reasoning", "code_generation", "content_generation"],
            tier="balanced",
        ),
        ProviderModel(
            provider="anthropic", model="claude-opus",
            cost_per_1k_input_tokens=0.015, cost_per_1k_output_tokens=0.075,
            avg_latency_ms=2000,
            capabilities=["reasoning", "code_generation", "content_generation"],
            tier="powerful",
        ),
    ]


def _make_registry() -> ProviderRegistry:
    registry = ProviderRegistry()
    mock_provider = MagicMock()
    mock_provider.provider_name = "local"
    mock_provider.health_check = AsyncMock(return_value=True)
    mock_provider.get_available_models.return_value = _models()
    registry.register_provider(mock_provider)
    return registry


def _make_cli() -> tuple[KortexCLI, InMemoryTraceStore]:
    registry = _make_registry()
    router = Router()
    for m in _models():
        router.register_model(m)

    state_manager = StateManager(InMemoryStateStore())
    store = InMemoryTraceStore()

    runtime = KortexRuntime(
        router=router,
        state_manager=state_manager,
        registry=registry,
        trace_store=store,
    )
    runtime.register_agent(AgentDescriptor(
        "researcher", "Researcher", "Research agent",
        capabilities=["reasoning"],
    ))
    runtime.register_agent(AgentDescriptor(
        "writer", "Writer", "Writing agent",
        capabilities=["content_generation"],
    ))
    runtime.register_agent(AgentDescriptor(
        "reviewer", "Reviewer", "Review agent",
        capabilities=["reasoning"],
    ))

    cli = KortexCLI(runtime, registry, trace_store=store)
    return cli, store


def _sample_trace(
    trace_id: str = "trace-001",
    task_id: str = "task-001",
) -> TaskTrace:
    policy = RoutingPolicy.cost_optimized()
    return TaskTrace(
        trace_id=trace_id,
        task_id=task_id,
        task_content="Write about AI coordination",
        task_complexity="moderate",
        pipeline=["researcher", "writer"],
        steps=[
            TraceStep(
                step_index=0,
                agent_id="researcher",
                input_payload={"content": "Write about AI coordination"},
                routing_decision={
                    "task_id": task_id,
                    "chosen_provider": "local",
                    "chosen_model": "tiny-7b",
                    "estimated_cost_usd": 0.0002,
                    "estimated_latency_ms": 50,
                    "reasoning": "Cost-optimized selection.",
                },
                policy_snapshot=policy.to_dict(),
                started_at="2026-03-29T10:00:00+00:00",
                completed_at="2026-03-29T10:00:01+00:00",
                duration_ms=50.0,
            ),
            TraceStep(
                step_index=1,
                agent_id="writer",
                input_payload={"content": "Write about AI coordination"},
                routing_decision={
                    "task_id": task_id,
                    "chosen_provider": "local",
                    "chosen_model": "tiny-7b",
                    "estimated_cost_usd": 0.0002,
                    "estimated_latency_ms": 50,
                    "reasoning": "Cost-optimized selection.",
                },
                policy_snapshot=policy.to_dict(),
                started_at="2026-03-29T10:00:01+00:00",
                completed_at="2026-03-29T10:00:02+00:00",
                duration_ms=50.0,
            ),
        ],
        policy_snapshot=policy.to_dict(),
        total_estimated_cost_usd=0.0004,
        total_actual_cost_usd=0.0,
        total_duration_ms=100.0,
        success=True,
        created_at="2026-03-29T10:00:00+00:00",
    )


# ---------------------------------------------------------------------------
# 1. trace list shows table header and row per trace
# ---------------------------------------------------------------------------


class TestTraceList:
    @pytest.mark.asyncio
    async def test_trace_list_shows_traces(self) -> None:
        cli, store = _make_cli()
        await store.save_trace(_sample_trace("t1", "task-A"))
        await store.save_trace(_sample_trace("t2", "task-B"))

        output = await cli.cmd_trace_list()

        assert "TRACE_ID" in output
        assert "PIPELINE" in output
        assert "t1" in output
        assert "t2" in output

    @pytest.mark.asyncio
    async def test_trace_list_empty(self) -> None:
        cli, _store = _make_cli()
        output = await cli.cmd_trace_list()
        assert "No traces found" in output


# ---------------------------------------------------------------------------
# 2. trace list with no store returns helpful message
# ---------------------------------------------------------------------------


class TestTraceListNoStore:
    @pytest.mark.asyncio
    async def test_no_store_message(self) -> None:
        registry = _make_registry()
        router = Router()
        for m in _models():
            router.register_model(m)
        runtime = KortexRuntime(
            router=router,
            state_manager=StateManager(InMemoryStateStore()),
            registry=registry,
        )
        cli = KortexCLI(runtime, registry, trace_store=None)

        output = await cli.cmd_trace_list()
        assert "No trace store configured" in output


# ---------------------------------------------------------------------------
# 3. trace show for a known trace_id includes step detail
# ---------------------------------------------------------------------------


class TestTraceShow:
    @pytest.mark.asyncio
    async def test_trace_show_detail(self) -> None:
        cli, store = _make_cli()
        await store.save_trace(_sample_trace("t1"))

        output = await cli.cmd_trace_show("t1")

        assert "t1" in output
        assert "researcher" in output
        assert "writer" in output
        assert "tiny-7b" in output
        assert "Step 0" in output
        assert "Step 1" in output


# ---------------------------------------------------------------------------
# 4. trace show for unknown trace_id returns error
# ---------------------------------------------------------------------------


class TestTraceShowMissing:
    @pytest.mark.asyncio
    async def test_missing_trace_error(self) -> None:
        cli, _store = _make_cli()
        output = await cli.cmd_trace_show("nonexistent")
        assert "not found" in output.lower()


# ---------------------------------------------------------------------------
# 5. trace export produces valid JSON
# ---------------------------------------------------------------------------


class TestTraceExport:
    @pytest.mark.asyncio
    async def test_export_valid_json(self) -> None:
        cli, store = _make_cli()
        await store.save_trace(_sample_trace("t1"))

        output = await cli.cmd_trace_export("t1")

        data = json.loads(output)
        assert data["trace_id"] == "t1"
        assert len(data["steps"]) == 2

    @pytest.mark.asyncio
    async def test_export_to_file(self) -> None:
        cli, store = _make_cli()
        await store.save_trace(_sample_trace("t1"))

        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as tmp:
            tmp_path = tmp.name

        try:
            output = await cli.cmd_trace_export("t1", output_file=tmp_path)
            assert "exported" in output.lower()

            with open(tmp_path) as f:
                data = json.loads(f.read())
            assert data["trace_id"] == "t1"
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# 6. replay under same policy shows zero diffs
# ---------------------------------------------------------------------------


class TestReplaySamePolicy:
    @pytest.mark.asyncio
    async def test_replay_same_policy_zero_diffs(self) -> None:
        cli, store = _make_cli()
        await store.save_trace(_sample_trace("t1"))

        # Replay with no --policy uses the original policy from the trace
        output = await cli.cmd_replay("t1")

        assert "Replay Results" in output
        assert "no model changes" in output.lower() or "No cost change" in output


# ---------------------------------------------------------------------------
# 7. replay under quality_optimized changes model selections
# ---------------------------------------------------------------------------


class TestReplayCommand:
    @pytest.mark.asyncio
    async def test_replay_shows_comparison(self) -> None:
        cli, store = _make_cli()
        await store.save_trace(_sample_trace("t1"))

        # Write a quality policy to a temp file
        policy = RoutingPolicy.quality_optimized()
        with tempfile.NamedTemporaryFile(
            suffix=".toml", delete=False, mode="w"
        ) as tmp:
            tmp.write(policy.to_toml())
            tmp_path = tmp.name

        try:
            output = await cli.cmd_replay("t1", policy_file=tmp_path)
            assert "Replay Results" in output
            assert "ORIGINAL" in output
            assert "REPLAYED" in output
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# 7. replay with missing trace_id returns error
# ---------------------------------------------------------------------------


class TestReplayMissing:
    @pytest.mark.asyncio
    async def test_replay_missing_trace(self) -> None:
        cli, _store = _make_cli()
        output = await cli.cmd_replay("nonexistent")
        assert "not found" in output.lower()


# ---------------------------------------------------------------------------
# 8. policy diff shows changed steps
# ---------------------------------------------------------------------------


class TestPolicyDiff:
    @pytest.mark.asyncio
    async def test_policy_diff_shows_changes(self) -> None:
        cli, store = _make_cli()
        await store.save_trace(_sample_trace("t1"))

        # Quality policy should pick different models than cost_optimized
        policy = RoutingPolicy.quality_optimized()
        with tempfile.NamedTemporaryFile(
            suffix=".toml", delete=False, mode="w"
        ) as tmp:
            tmp.write(policy.to_toml())
            tmp_path = tmp.name

        try:
            output = await cli.cmd_policy_diff("t1", policy_file=tmp_path)
            assert "Policy Diff" in output
            assert "quality_optimized" in output
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# 9. policy show displays active policy details
# ---------------------------------------------------------------------------


class TestPolicyShow:
    def test_policy_show_active(self) -> None:
        cli, _store = _make_cli()
        # Set a policy on the router
        cli._runtime._router.set_policy(RoutingPolicy.cost_optimized())

        output = cli.cmd_policy_show()

        assert "cost_optimized" in output
        assert "Constraints" in output
        assert "Objective" in output
        assert "Fallback" in output

    def test_policy_show_from_file(self) -> None:
        cli, _store = _make_cli()
        policy = RoutingPolicy.quality_optimized()
        with tempfile.NamedTemporaryFile(
            suffix=".toml", delete=False, mode="w"
        ) as tmp:
            tmp.write(policy.to_toml())
            tmp_path = tmp.name

        try:
            output = cli.cmd_policy_show(policy_file=tmp_path)
            assert "quality_optimized" in output
            assert "Constraints" in output
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# 10. _build_demo_trace produces a valid 3-step trace
# ---------------------------------------------------------------------------


class TestBuildDemoTrace:
    def test_demo_trace_structure(self) -> None:
        router = Router()
        for m in _models():
            router.register_model(m)

        trace = _build_demo_trace(router)

        assert trace.trace_id == "demo-trace-001"
        assert len(trace.steps) == 3
        assert trace.pipeline == ["researcher", "writer", "reviewer"]
        assert trace.success is True
        assert trace.total_estimated_cost_usd > 0


# ---------------------------------------------------------------------------
# 11. invalid policy file gives clear error
# ---------------------------------------------------------------------------


class TestInvalidPolicyFile:
    @pytest.mark.asyncio
    async def test_replay_invalid_policy_file(self) -> None:
        cli, store = _make_cli()
        await store.save_trace(_sample_trace("t1"))

        output = await cli.cmd_replay("t1", policy_file="/nonexistent/policy.toml")
        assert "error" in output.lower()

    @pytest.mark.asyncio
    async def test_policy_diff_invalid_file(self) -> None:
        cli, store = _make_cli()
        await store.save_trace(_sample_trace("t1"))

        output = await cli.cmd_policy_diff("t1", policy_file="/nonexistent/policy.toml")
        assert "error" in output.lower()

    def test_policy_show_invalid_file(self) -> None:
        cli, _store = _make_cli()
        output = cli.cmd_policy_show(policy_file="/nonexistent/policy.toml")
        assert "error" in output.lower()


# ---------------------------------------------------------------------------
# 12. main() trace list subcommand runs without error
# ---------------------------------------------------------------------------


class TestMainTraceSubcommand:
    def test_trace_list_via_main(self) -> None:
        exit_code = main(["trace", "list"])
        assert exit_code == 0
