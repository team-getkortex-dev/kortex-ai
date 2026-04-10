"""Unit tests for the Kortex CLI dashboard."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kortex.core.router import ProviderModel, Router
from kortex.core.runtime import AgentDescriptor, KortexRuntime
from kortex.core.state import StateManager
from kortex.core.types import (
    CoordinationResult,
    HandoffContext,
    RoutingDecision,
)
from kortex.dashboard.cli import KortexCLI, main
from kortex.dashboard.formatter import (
    colorize,
    format_cost,
    format_duration,
    format_table,
)
from kortex.providers.registry import ProviderRegistry
from kortex.store.memory import InMemoryStateStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_registry_with_models() -> ProviderRegistry:
    """Create a registry with a mock provider that has models."""
    registry = ProviderRegistry()

    mock_provider = MagicMock()
    mock_provider.provider_name = "test-provider"
    mock_provider.health_check = AsyncMock(return_value=True)
    mock_provider.get_available_models.return_value = [
        ProviderModel(
            provider="test-provider",
            model="fast-model",
            cost_per_1k_input_tokens=0.0001,
            cost_per_1k_output_tokens=0.0002,
            avg_latency_ms=50,
            capabilities=["content_generation"],
            tier="fast",
        ),
        ProviderModel(
            provider="test-provider",
            model="powerful-model",
            cost_per_1k_input_tokens=0.01,
            cost_per_1k_output_tokens=0.03,
            avg_latency_ms=2000,
            capabilities=["content_generation", "code_generation", "reasoning"],
            tier="powerful",
        ),
    ]
    registry.register_provider(mock_provider)
    return registry


def _make_runtime_and_registry() -> tuple[KortexRuntime, ProviderRegistry]:
    """Build a runtime with mock provider and agents."""
    registry = _make_registry_with_models()

    router = Router()
    for model in registry.get_all_models():
        router.register_model(model)

    state_manager = StateManager(InMemoryStateStore())

    runtime = KortexRuntime(router=router, state_manager=state_manager, registry=registry)
    runtime.register_agent(AgentDescriptor(
        agent_id="planner",
        name="Planner",
        description="Plans tasks",
    ))
    runtime.register_agent(AgentDescriptor(
        agent_id="coder",
        name="Coder",
        description="Writes code",
    ))
    return runtime, registry


# ---------------------------------------------------------------------------
# 1. format_table produces aligned columns
# ---------------------------------------------------------------------------


class TestFormatTable:
    def test_aligned_columns(self) -> None:
        headers = ["NAME", "AGE", "CITY"]
        rows = [
            ["Alice", "30", "New York"],
            ["Bob", "25", "LA"],
        ]
        result = format_table(headers, rows)
        lines = result.split("\n")

        assert len(lines) == 4  # header + separator + 2 rows
        # All lines should have the same column alignment
        assert "NAME" in lines[0]
        assert "AGE" in lines[0]
        assert "---" in lines[1]
        assert "Alice" in lines[2]
        assert "Bob" in lines[3]

    def test_empty_headers(self) -> None:
        assert format_table([], []) == ""


# ---------------------------------------------------------------------------
# 2. format_cost handles zero, tiny, and normal costs
# ---------------------------------------------------------------------------


class TestFormatCost:
    def test_zero_cost(self) -> None:
        assert format_cost(0.0) == "FREE"

    def test_tiny_cost(self) -> None:
        result = format_cost(0.00001)
        assert result.startswith("$")
        assert "0.000010" in result

    def test_small_cost(self) -> None:
        result = format_cost(0.0025)
        assert result == "$0.0025"

    def test_normal_cost(self) -> None:
        result = format_cost(1.50)
        assert result == "$1.50"


# ---------------------------------------------------------------------------
# 3. format_duration switches units correctly
# ---------------------------------------------------------------------------


class TestFormatDuration:
    def test_milliseconds(self) -> None:
        assert format_duration(45) == "45ms"

    def test_seconds(self) -> None:
        assert format_duration(1200) == "1.2s"

    def test_exact_boundary(self) -> None:
        assert format_duration(999) == "999ms"

    def test_one_second(self) -> None:
        assert format_duration(1000) == "1.0s"


# ---------------------------------------------------------------------------
# 4. colorize respects NO_COLOR env var
# ---------------------------------------------------------------------------


class TestColorize:
    def test_no_color_env_var(self) -> None:
        with patch.dict(os.environ, {"NO_COLOR": "1"}):
            result = colorize("hello", "red")
            assert result == "hello"
            assert "\033" not in result

    def test_unknown_color_returns_plain(self) -> None:
        # Unknown color code should return text as-is
        result = colorize("hello", "nonexistent")
        assert result == "hello"


# ---------------------------------------------------------------------------
# 5. dry-run subcommand produces correct output
# ---------------------------------------------------------------------------


class TestDryRun:
    @pytest.mark.asyncio
    async def test_dry_run_output(self) -> None:
        runtime, registry = _make_runtime_and_registry()
        cli = KortexCLI(runtime, registry)

        output = await cli.cmd_dry_run(
            task_content="Write a hello world program",
            complexity="simple",
            pipeline=["planner", "coder"],
        )

        assert "Dry Run" in output
        assert "Write a hello world program" in output
        assert "simple" in output
        assert "planner" in output
        assert "coder" in output
        assert "Routing Decisions" in output
        assert "Cost Summary" in output
        assert "Handoff Chain" in output

    @pytest.mark.asyncio
    async def test_dry_run_empty_pipeline(self) -> None:
        runtime, registry = _make_runtime_and_registry()
        cli = KortexCLI(runtime, registry)

        output = await cli.cmd_dry_run(
            task_content="test",
            pipeline=[],
        )
        assert "Error" in output


# ---------------------------------------------------------------------------
# 6. models subcommand lists all providers' models
# ---------------------------------------------------------------------------


class TestModelsCommand:
    def test_models_output(self) -> None:
        runtime, registry = _make_runtime_and_registry()
        cli = KortexCLI(runtime, registry)

        output = cli.cmd_models()

        assert "PROVIDER" in output
        assert "MODEL" in output
        assert "TIER" in output
        assert "INPUT/1K" in output
        assert "OUTPUT/1K" in output
        assert "LATENCY" in output
        assert "fast-model" in output
        assert "powerful-model" in output
        assert "test-provider" in output


# ---------------------------------------------------------------------------
# 7. status subcommand shows provider health
# ---------------------------------------------------------------------------


class TestStatusCommand:
    @pytest.mark.asyncio
    async def test_status_shows_providers(self) -> None:
        runtime, registry = _make_runtime_and_registry()
        cli = KortexCLI(runtime, registry)

        output = await cli.cmd_status()

        assert "Status" in output
        assert "Providers" in output
        assert "test-provider" in output
        assert "Models" in output
        assert "Agents" in output
        assert "planner" in output
        assert "coder" in output

    @pytest.mark.asyncio
    async def test_status_unhealthy_provider(self) -> None:
        runtime, registry = _make_runtime_and_registry()

        # Make health check fail
        provider = registry.get_provider("test-provider")
        provider.health_check = AsyncMock(side_effect=Exception("down"))

        cli = KortexCLI(runtime, registry)
        output = await cli.cmd_status()

        assert "test-provider" in output


# ---------------------------------------------------------------------------
# 8. history subcommand with --last flag
# ---------------------------------------------------------------------------


class TestHistoryCommand:
    @pytest.mark.asyncio
    async def test_history_with_last(self) -> None:
        runtime, registry = _make_runtime_and_registry()
        store = runtime._state._store

        # Save some checkpoints
        for i in range(5):
            ctx = HandoffContext(
                source_agent=f"agent-{i}",
                target_agent=f"agent-{i+1}",
                state_snapshot={"step": i},
                compressed_summary=f"Step {i} summary",
            )
            await store.save_checkpoint(ctx)

        cli = KortexCLI(runtime, registry)
        output = await cli.cmd_history(last=2)

        assert "Checkpoints" in output

    @pytest.mark.asyncio
    async def test_history_empty(self) -> None:
        runtime, registry = _make_runtime_and_registry()
        cli = KortexCLI(runtime, registry)

        output = await cli.cmd_history()
        assert "No checkpoints found" in output


# ---------------------------------------------------------------------------
# 9. CLI main() handles unknown subcommands gracefully
# ---------------------------------------------------------------------------


class TestMainEntryPoint:
    def test_no_subcommand_shows_help(self) -> None:
        """main() with no args should print help and return 1."""
        exit_code = main([])
        assert exit_code == 1

    def test_unknown_subcommand(self) -> None:
        """main() with an unknown subcommand should return non-zero."""
        # argparse will raise SystemExit for unrecognized args
        with pytest.raises(SystemExit):
            main(["nonexistent-command"])

    def test_config_subcommand(self) -> None:
        """config subcommand should run without error."""
        # This will try to build default runtime which may fail without
        # providers, but the parser should at least accept the command
        exit_code = main(["config"])
        assert exit_code == 0
