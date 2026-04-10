"""Kortex — Agent coordination runtime for multi-agent AI systems.

Provides intelligent task routing, stateful handoff management, and
real-time failure detection for multi-agent workflows.

Quick start::

    from kortex import KortexRuntime, AgentDescriptor, Router, ProviderModel, StateManager

    router = Router()
    router.register_model(ProviderModel(
        provider="openai", model="gpt-4o-mini",
        cost_per_1k_input_tokens=0.00015, cost_per_1k_output_tokens=0.0006,
        avg_latency_ms=250, capabilities=["reasoning"], tier="fast",
    ))

    async with KortexRuntime(
        router=router, state_manager=StateManager.create("memory"),
    ) as runtime:
        runtime.register_agent(AgentDescriptor(
            agent_id="agent-1", name="Agent", description="...",
        ))
        result = await runtime.coordinate(task, ["agent-1"])
"""

from __future__ import annotations

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Core runtime
# ---------------------------------------------------------------------------
from kortex.core.runtime import AgentDescriptor, KortexRuntime

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
from kortex.core.types import (
    CoordinationResult,
    ExecutionEvent,
    HandoffContext,
    ModelIdentity,
    RoutingDecision,
    TaskSpec,
)

# ---------------------------------------------------------------------------
# Router + policy
# ---------------------------------------------------------------------------
from kortex.core.router import ProviderModel, Router
from kortex.core.policy import RoutingPolicy
from kortex.core.metrics import ObservedMetrics

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
from kortex.core.state import StateManager

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
from kortex.core.detector import DetectionPolicy, FailureDetector

# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------
from kortex.core.recovery import (
    RecoveryAction,
    RecoveryExecutor,
    RecoveryPolicy,
    RecoveryRecord,
)

# ---------------------------------------------------------------------------
# Tracing / replay
# ---------------------------------------------------------------------------
from kortex.core.trace import TaskTrace
from kortex.core.replay import ReplayEngine

# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------
from kortex.core.capabilities import Capability, normalize_capabilities

# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------
from kortex.providers.base import ProviderResponse
from kortex.providers.registry import ProviderRegistry

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
from kortex.core.exceptions import (
    CheckpointNotFoundError,
    HandoffError,
    KortexError,
    ProviderError,
    RouterError,
    RoutingFailedError,
    StateError,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
from kortex.config import KortexConfig, get_config, reset_config

__all__ = [
    "__version__",
    # Runtime
    "KortexRuntime",
    "AgentDescriptor",
    # Types
    "TaskSpec",
    "RoutingDecision",
    "HandoffContext",
    "ExecutionEvent",
    "CoordinationResult",
    "ModelIdentity",
    # Router + policy + metrics
    "Router",
    "ProviderModel",
    "RoutingPolicy",
    "ObservedMetrics",
    # State
    "StateManager",
    # Detection
    "FailureDetector",
    "DetectionPolicy",
    # Recovery
    "RecoveryExecutor",
    "RecoveryPolicy",
    "RecoveryAction",
    "RecoveryRecord",
    # Tracing / replay
    "TaskTrace",
    "ReplayEngine",
    # Capabilities
    "Capability",
    "normalize_capabilities",
    # Providers
    "ProviderRegistry",
    "ProviderResponse",
    # Exceptions
    "KortexError",
    "CheckpointNotFoundError",
    "HandoffError",
    "ProviderError",
    "RouterError",
    "RoutingFailedError",
    "StateError",
    # Config
    "KortexConfig",
    "get_config",
    "reset_config",
]
