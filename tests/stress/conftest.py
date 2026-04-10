"""Shared fixtures and utilities for stress testing Kortex.

These fixtures create realistic, pre-configured environments that
mirror production deployments. Nothing is simplified or softened.
"""

import asyncio
import random
import string
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock

import pytest

from kortex.core.types import TaskSpec, HandoffContext, RoutingDecision
from kortex.core.router import Router, ProviderModel, HeuristicRoutingStrategy
from kortex.core.state import StateManager
from kortex.core.runtime import KortexRuntime, AgentDescriptor
from kortex.providers.base import ProviderConnector, ProviderResponse
from kortex.providers.registry import ProviderRegistry
from kortex.store.memory import InMemoryStateStore


# ---------------------------------------------------------------------------
# Chaos Provider — simulates real-world unreliability
# ---------------------------------------------------------------------------

class ChaosProvider:
    """A provider that misbehaves on purpose.

    - Configurable failure rate (0.0 to 1.0)
    - Configurable latency range (simulates slow APIs)
    - Configurable cost jitter (simulates unpredictable billing)
    - Can produce garbage output on demand
    - Can hang indefinitely (timeout testing)
    """

    def __init__(
        self,
        name: str = "chaos",
        failure_rate: float = 0.0,
        min_latency_ms: float = 1.0,
        max_latency_ms: float = 5.0,
        cost_jitter: float = 0.0,
        garbage_rate: float = 0.0,
        hang_rate: float = 0.0,
        hang_timeout: float = 30.0,
    ):
        self._name = name
        self.failure_rate = failure_rate
        self.min_latency_ms = min_latency_ms
        self.max_latency_ms = max_latency_ms
        self.cost_jitter = cost_jitter
        self.garbage_rate = garbage_rate
        self.hang_rate = hang_rate
        self.hang_timeout = hang_timeout
        self.call_count = 0
        self.failure_count = 0
        self.success_count = 0
        self._models: list[ProviderModel] = []

    @property
    def provider_name(self) -> str:
        return self._name

    def set_models(self, models: list[ProviderModel]) -> None:
        self._models = models

    def get_available_models(self) -> list[ProviderModel]:
        return self._models

    async def health_check(self) -> bool:
        return True

    async def complete(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> ProviderResponse:
        self.call_count += 1
        start = time.monotonic()

        # Simulate hang
        if random.random() < self.hang_rate:
            await asyncio.sleep(self.hang_timeout)

        # Simulate latency
        latency_s = random.uniform(
            self.min_latency_ms / 1000, self.max_latency_ms / 1000
        )
        await asyncio.sleep(latency_s)

        # Simulate failure
        if random.random() < self.failure_rate:
            self.failure_count += 1
            raise ConnectionError(
                f"ChaosProvider '{self._name}' intentional failure "
                f"(call #{self.call_count})"
            )

        # Simulate garbage output
        if random.random() < self.garbage_rate:
            content = ""  # Empty response — should trigger quality detection
        else:
            content = (
                f"Response from {self._name}/{model} for: "
                f"{prompt[:80]}... "
                f"(generated {random.randint(50, 500)} words of analysis)"
            )

        elapsed_ms = (time.monotonic() - start) * 1000
        input_tokens = len(prompt.split()) * 2
        output_tokens = len(content.split()) * 2

        # Find cost for this model
        base_cost = 0.001
        for m in self._models:
            if m.model == model:
                base_cost = (
                    m.cost_per_1k_input_tokens * input_tokens / 1000
                    + m.cost_per_1k_output_tokens * output_tokens / 1000
                )
                break

        # Apply cost jitter
        jitter_multiplier = 1.0 + random.uniform(-self.cost_jitter, self.cost_jitter)
        cost = base_cost * jitter_multiplier

        self.success_count += 1
        return ProviderResponse(
            content=content,
            model=model,
            provider=self._name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_ms=elapsed_ms,
            raw_response={"chaos": True, "call_number": self.call_count},
        )

    async def stream(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        response = await self.complete(prompt, model, max_tokens, temperature, **kwargs)
        for word in response.content.split():
            yield word + " "


# ---------------------------------------------------------------------------
# Model catalog — realistic pricing from real providers
# ---------------------------------------------------------------------------

STRESS_MODELS = [
    # Fast tier — cheap, quick, limited
    ProviderModel(
        provider="chaos_fast",
        model="speed-7b",
        cost_per_1k_input_tokens=0.0001,
        cost_per_1k_output_tokens=0.0002,
        avg_latency_ms=50,
        capabilities=["analysis"],
        max_context_tokens=8192,
        tier="fast",
    ),
    ProviderModel(
        provider="chaos_fast",
        model="turbo-mini",
        cost_per_1k_input_tokens=0.00015,
        cost_per_1k_output_tokens=0.0003,
        avg_latency_ms=80,
        capabilities=["analysis", "code_generation"],
        max_context_tokens=16384,
        tier="fast",
    ),
    # Balanced tier — moderate cost and capability
    ProviderModel(
        provider="chaos_balanced",
        model="standard-70b",
        cost_per_1k_input_tokens=0.003,
        cost_per_1k_output_tokens=0.006,
        avg_latency_ms=400,
        capabilities=["reasoning", "code_generation", "analysis", "content_generation", "research"],
        max_context_tokens=32768,
        tier="balanced",
    ),
    ProviderModel(
        provider="chaos_balanced",
        model="mid-range-v2",
        cost_per_1k_input_tokens=0.005,
        cost_per_1k_output_tokens=0.010,
        avg_latency_ms=600,
        capabilities=["reasoning", "code_generation", "analysis", "vision", "content_generation", "quality_assurance"],
        max_context_tokens=65536,
        tier="balanced",
    ),
    # Powerful tier — expensive, capable
    ProviderModel(
        provider="chaos_powerful",
        model="apex-405b",
        cost_per_1k_input_tokens=0.015,
        cost_per_1k_output_tokens=0.030,
        avg_latency_ms=1500,
        capabilities=["reasoning", "code_generation", "analysis", "vision", "research", "content_generation", "quality_assurance"],
        max_context_tokens=131072,
        tier="powerful",
    ),
    ProviderModel(
        provider="chaos_powerful",
        model="titan-ultra",
        cost_per_1k_input_tokens=0.020,
        cost_per_1k_output_tokens=0.040,
        avg_latency_ms=2000,
        capabilities=["reasoning", "code_generation", "analysis", "vision", "audio", "research", "content_generation", "quality_assurance"],
        max_context_tokens=200000,
        tier="powerful",
    ),
]

STRESS_AGENTS = [
    AgentDescriptor(
        agent_id="planner",
        name="Strategic Planner",
        description="Decomposes complex tasks into subtasks",
        capabilities=["reasoning", "analysis"],
    ),
    AgentDescriptor(
        agent_id="researcher",
        name="Deep Researcher",
        description="Finds and synthesizes information",
        capabilities=["research", "analysis"],
    ),
    AgentDescriptor(
        agent_id="coder",
        name="Code Generator",
        description="Writes and refactors code",
        capabilities=["code_generation"],
    ),
    AgentDescriptor(
        agent_id="writer",
        name="Content Writer",
        description="Produces polished written content",
        capabilities=["content_generation"],
    ),
    AgentDescriptor(
        agent_id="reviewer",
        name="Quality Reviewer",
        description="Reviews and validates outputs",
        capabilities=["analysis", "quality_assurance"],
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_large_payload(size_bytes: int) -> dict[str, Any]:
    """Generate a dict that serializes to approximately `size_bytes`."""
    entries = {}
    entry_size = 100  # rough size per entry
    count = max(1, size_bytes // entry_size)
    for i in range(count):
        entries[f"key_{i:06d}"] = "".join(
            random.choices(string.ascii_letters + string.digits, k=80)
        )
    return entries


def generate_random_task(complexity: str = "moderate") -> TaskSpec:
    """Generate a random but realistic task."""
    tasks = [
        "Analyze the quarterly revenue data and identify growth trends",
        "Refactor the authentication middleware to support OAuth 2.1",
        "Write a comprehensive market analysis for the AI agent space",
        "Review the pull request for security vulnerabilities",
        "Design a database schema for multi-tenant SaaS architecture",
        "Debug the intermittent timeout in the payment processing pipeline",
        "Generate test cases for the new recommendation engine",
        "Optimize the vector search query for sub-100ms latency",
        "Draft a technical RFC for migrating to event-driven architecture",
        "Evaluate three competing ML frameworks for production deployment",
    ]
    return TaskSpec(
        content=random.choice(tasks) + f" [stress-{random.randint(1000,9999)}]",
        complexity_hint=complexity,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def chaos_provider_reliable():
    """A chaos provider that never fails — baseline for comparison."""
    provider = ChaosProvider(
        name="reliable",
        failure_rate=0.0,
        min_latency_ms=0.1,
        max_latency_ms=1.0,
    )
    provider.set_models(STRESS_MODELS)
    return provider


@pytest.fixture
def chaos_provider_flaky():
    """A chaos provider that fails 30% of the time."""
    provider = ChaosProvider(
        name="flaky",
        failure_rate=0.3,
        min_latency_ms=1.0,
        max_latency_ms=10.0,
        cost_jitter=0.5,
    )
    provider.set_models(STRESS_MODELS)
    return provider


@pytest.fixture
def chaos_provider_hostile():
    """A chaos provider that fails 60% of the time and produces garbage."""
    provider = ChaosProvider(
        name="hostile",
        failure_rate=0.6,
        min_latency_ms=5.0,
        max_latency_ms=50.0,
        cost_jitter=1.0,
        garbage_rate=0.3,
    )
    provider.set_models(STRESS_MODELS)
    return provider


@pytest.fixture
def chaos_provider_dead():
    """A chaos provider that fails 100% of the time."""
    provider = ChaosProvider(
        name="dead",
        failure_rate=1.0,
    )
    provider.set_models(STRESS_MODELS)
    return provider


def build_router() -> Router:
    """Build a router with all stress test models registered."""
    router = Router()
    for model in STRESS_MODELS:
        router.register_model(model)
    return router


def build_registry(provider: ChaosProvider) -> ProviderRegistry:
    """Build a registry with chaos providers for each tier."""
    registry = ProviderRegistry()

    # Create separate providers per tier so routing can find them
    for tier_name in ["chaos_fast", "chaos_balanced", "chaos_powerful"]:
        tier_provider = ChaosProvider(
            name=tier_name,
            failure_rate=provider.failure_rate,
            min_latency_ms=provider.min_latency_ms,
            max_latency_ms=provider.max_latency_ms,
            cost_jitter=provider.cost_jitter,
            garbage_rate=provider.garbage_rate,
        )
        tier_models = [m for m in STRESS_MODELS if m.provider == tier_name]
        tier_provider.set_models(tier_models)
        registry.register_provider(tier_provider)

    return registry


def build_runtime(
    provider: ChaosProvider | None = None,
    store_backend: str = "memory",
    with_detector: bool = False,
    **store_kwargs: Any,
) -> KortexRuntime:
    """Build a fully configured runtime for stress testing."""
    router = build_router()
    state_manager = StateManager.create(store_backend, **store_kwargs)

    registry = None
    if provider is not None:
        registry = build_registry(provider)

    detector = None
    if with_detector:
        try:
            from kortex.core.detector import FailureDetector, DetectionPolicy
            detector = FailureDetector(DetectionPolicy())
        except ImportError:
            pass  # detector not built yet — skip

    runtime = KortexRuntime(
        router=router,
        state_manager=state_manager,
        registry=registry,
        detector=detector,
    )

    for agent in STRESS_AGENTS:
        runtime.register_agent(agent)

    return runtime


@pytest.fixture
def reliable_runtime(chaos_provider_reliable):
    return build_runtime(chaos_provider_reliable)


@pytest.fixture
def flaky_runtime(chaos_provider_flaky):
    return build_runtime(chaos_provider_flaky)


@pytest.fixture
def hostile_runtime(chaos_provider_hostile):
    return build_runtime(chaos_provider_hostile)


@pytest.fixture
def detection_runtime(chaos_provider_flaky):
    return build_runtime(chaos_provider_flaky, with_detector=True)
