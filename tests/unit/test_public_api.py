"""Tests verifying that top-level imports from the kortex package work correctly."""

from __future__ import annotations

import importlib

import pytest


class TestTopLevelImports:
    """All primary public-API names must be importable from kortex directly."""

    def test_kortex_runtime(self) -> None:
        from kortex import KortexRuntime
        assert KortexRuntime is not None

    def test_agent_descriptor(self) -> None:
        from kortex import AgentDescriptor
        assert AgentDescriptor is not None

    def test_task_spec(self) -> None:
        from kortex import TaskSpec
        assert TaskSpec is not None

    def test_routing_decision(self) -> None:
        from kortex import RoutingDecision
        assert RoutingDecision is not None

    def test_handoff_context(self) -> None:
        from kortex import HandoffContext
        assert HandoffContext is not None

    def test_execution_event(self) -> None:
        from kortex import ExecutionEvent
        assert ExecutionEvent is not None

    def test_coordination_result(self) -> None:
        from kortex import CoordinationResult
        assert CoordinationResult is not None

    def test_router(self) -> None:
        from kortex import Router
        assert Router is not None

    def test_provider_model(self) -> None:
        from kortex import ProviderModel
        assert ProviderModel is not None

    def test_routing_policy(self) -> None:
        from kortex import RoutingPolicy
        assert RoutingPolicy is not None

    def test_state_manager(self) -> None:
        from kortex import StateManager
        assert StateManager is not None

    def test_failure_detector(self) -> None:
        from kortex import FailureDetector
        assert FailureDetector is not None

    def test_detection_policy(self) -> None:
        from kortex import DetectionPolicy
        assert DetectionPolicy is not None

    def test_recovery_executor(self) -> None:
        from kortex import RecoveryExecutor
        assert RecoveryExecutor is not None

    def test_recovery_policy(self) -> None:
        from kortex import RecoveryPolicy
        assert RecoveryPolicy is not None

    def test_replay_engine(self) -> None:
        from kortex import ReplayEngine
        assert ReplayEngine is not None

    def test_capability(self) -> None:
        from kortex import Capability
        assert Capability is not None

    def test_provider_registry(self) -> None:
        from kortex import ProviderRegistry
        assert ProviderRegistry is not None

    def test_provider_response(self) -> None:
        from kortex import ProviderResponse
        assert ProviderResponse is not None

    def test_kortex_error(self) -> None:
        from kortex import KortexError
        assert KortexError is not None

    def test_routing_failed_error(self) -> None:
        from kortex import RoutingFailedError
        assert RoutingFailedError is not None

    def test_kortex_config(self) -> None:
        from kortex import KortexConfig
        assert KortexConfig is not None

    def test_get_config(self) -> None:
        from kortex import get_config
        assert get_config is not None

    def test_version(self) -> None:
        import kortex
        assert kortex.__version__ == "0.1.0"


class TestImportIdentity:
    """Names imported from kortex must be the same objects as their source modules."""

    def test_runtime_same_object(self) -> None:
        from kortex import KortexRuntime as top_level
        from kortex.core.runtime import KortexRuntime as direct
        assert top_level is direct

    def test_router_same_object(self) -> None:
        from kortex import Router as top_level
        from kortex.core.router import Router as direct
        assert top_level is direct

    def test_task_spec_same_object(self) -> None:
        from kortex import TaskSpec as top_level
        from kortex.core.types import TaskSpec as direct
        assert top_level is direct

    def test_state_manager_same_object(self) -> None:
        from kortex import StateManager as top_level
        from kortex.core.state import StateManager as direct
        assert top_level is direct
