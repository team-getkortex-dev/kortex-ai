"""Unit tests for CLI standalone mode (demo models, default agents)."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kortex.core.router import ProviderModel, Router
from kortex.core.runtime import AgentDescriptor, KortexRuntime
from kortex.core.state import StateManager
from kortex.dashboard.cli import (
    KortexCLI,
    _DEFAULT_AGENTS,
    _DEMO_MODELS,
    _build_default_runtime,
    main,
)
from kortex.providers.registry import ProviderRegistry
from kortex.store.memory import InMemoryStateStore


# ---------------------------------------------------------------------------
# 1. models command shows demo models when no API keys set
# ---------------------------------------------------------------------------


class TestModelsDemo:
    def test_models_shows_demo_output(self) -> None:
        """With no API keys, models command should show demo models."""
        # Clear any API keys that might be set
        env = {k: v for k, v in os.environ.items()
               if k not in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY")}

        with patch.dict(os.environ, env, clear=True):
            runtime, registry, demo_mode = _build_default_runtime()
            assert demo_mode is True

            cli = KortexCLI(runtime, registry, demo_mode=True)
            output = cli.cmd_models()

            assert "demo mode" in output
            assert "gpt-4o-mini" in output
            assert "claude-sonnet-4-20250514" in output
            assert "claude-opus-4-20250514" in output

    def test_demo_models_have_all_tiers(self) -> None:
        """Demo models should cover fast, balanced, and powerful tiers."""
        tiers = {m.tier for m in _DEMO_MODELS}
        assert "fast" in tiers
        assert "balanced" in tiers
        assert "powerful" in tiers


# ---------------------------------------------------------------------------
# 2. dry-run produces output with default agents
# ---------------------------------------------------------------------------


class TestDryRunDefault:
    @pytest.mark.asyncio
    async def test_dry_run_with_default_agents(self) -> None:
        """dry-run should work with default agents when none are registered."""
        env = {k: v for k, v in os.environ.items()
               if k not in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY")}

        with patch.dict(os.environ, env, clear=True):
            runtime, registry, demo_mode = _build_default_runtime()

            # Default agents should be registered
            assert len(runtime._agents) == 3
            assert "researcher" in runtime._agents
            assert "writer" in runtime._agents
            assert "reviewer" in runtime._agents

            cli = KortexCLI(runtime, registry, demo_mode=demo_mode)
            output = await cli.cmd_dry_run(
                task_content="Analyze some data",
                complexity="complex",
                pipeline=["researcher", "writer", "reviewer"],
            )

            assert "Dry Run" in output
            assert "Routing Decisions" in output
            assert "Cost Summary" in output
            assert "Handoff Chain" in output


# ---------------------------------------------------------------------------
# 3. status shows demo mode indicator
# ---------------------------------------------------------------------------


class TestStatusDemo:
    @pytest.mark.asyncio
    async def test_status_shows_demo_indicator(self) -> None:
        """Status output should indicate demo mode when no API keys are set."""
        env = {k: v for k, v in os.environ.items()
               if k not in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY")}

        with patch.dict(os.environ, env, clear=True):
            runtime, registry, demo_mode = _build_default_runtime()
            cli = KortexCLI(runtime, registry, demo_mode=demo_mode)

            output = await cli.cmd_status()

            assert "demo mode" in output
            assert "Agents" in output
            assert "researcher" in output


# ---------------------------------------------------------------------------
# 4. auto_discover registers providers when keys are present
# ---------------------------------------------------------------------------


class TestAutoDiscover:
    def test_auto_discover_with_mock_key(self) -> None:
        """When an API key env var is set, auto_discover should register it."""
        env = {
            "OPENAI_API_KEY": "sk-test-key-123",
        }
        # Remove other keys to isolate
        clean_env = {k: v for k, v in os.environ.items()
                     if k not in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY")}
        clean_env.update(env)

        with patch.dict(os.environ, clean_env, clear=True):
            runtime, registry, demo_mode = _build_default_runtime()

            assert demo_mode is False
            assert "openai" in registry.list_providers()

    def test_no_keys_means_demo_mode(self) -> None:
        """With no API keys at all, demo_mode should be True."""
        env = {k: v for k, v in os.environ.items()
               if k not in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY")}

        with patch.dict(os.environ, env, clear=True):
            _runtime, _registry, demo_mode = _build_default_runtime()
            assert demo_mode is True
