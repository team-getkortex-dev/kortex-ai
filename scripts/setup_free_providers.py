"""Set up free-tier LLM providers for real API testing.

This script registers all available free or near-free providers into a
``KortexRuntime`` using ``GenericOpenAIConnector``.  It checks for API
keys via environment variables before registering each provider, so you
can run it with only the keys you have — models for providers with no
key are silently skipped.

Available providers
-------------------
- **Groq** — ultra-fast inference, truly free tier
  - Get key: https://console.groq.com
  - Env var: ``GROQ_API_KEY``

- **Cerebras** — world's fastest inference engine, free tier
  - Get key: https://cloud.cerebras.ai
  - Env var: ``CEREBRAS_API_KEY``

- **Together AI** — broad model catalogue, cheap/free tier models
  - Get key: https://api.together.xyz
  - Env var: ``TOGETHER_API_KEY``

- **OpenRouter** — aggregator with free ":free" model variants
  - Get key: https://openrouter.ai/keys
  - Env var: ``OPENROUTER_API_KEY``

Usage
-----
.. code-block:: bash

    # Set at least one key
    export GROQ_API_KEY="gsk_..."
    export CEREBRAS_API_KEY="csk_..."
    export TOGETHER_API_KEY="..."
    export OPENROUTER_API_KEY="sk-or-..."

    # Run the setup (prints a summary of registered models)
    python scripts/setup_free_providers.py

The script can also be imported and used programmatically::

    from scripts.setup_free_providers import setup_free_providers
    runtime = await setup_free_providers()
"""

from __future__ import annotations

import asyncio
import os
import sys

# Allow running from repo root or scripts/ directory
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from dotenv import load_dotenv  # type: ignore[import-untyped]

load_dotenv(os.path.join(_REPO_ROOT, ".env"))

from kortex.core.router import ProviderModel, Router
from kortex.core.runtime import KortexRuntime
from kortex.core.state import StateManager
from kortex.providers.registry import ProviderRegistry
from kortex.store.memory import InMemoryStateStore


# ---------------------------------------------------------------------------
# Provider model catalogues
# ---------------------------------------------------------------------------

_GROQ_MODELS: list[ProviderModel] = [
    ProviderModel(
        provider="groq",
        model="llama-3.3-70b-versatile",
        cost_per_1k_input_tokens=0.0,
        cost_per_1k_output_tokens=0.0,
        avg_latency_ms=50,
        capabilities=["reasoning", "analysis", "content_generation"],
        tier="powerful",
    ),
    ProviderModel(
        provider="groq",
        model="llama-3.1-8b-instant",
        cost_per_1k_input_tokens=0.0,
        cost_per_1k_output_tokens=0.0,
        avg_latency_ms=40,
        capabilities=["reasoning", "content_generation", "code_generation"],
        tier="fast",
    ),
]

_CEREBRAS_MODELS: list[ProviderModel] = [
    ProviderModel(
        provider="cerebras",
        model="llama3.1-8b",
        cost_per_1k_input_tokens=0.0,
        cost_per_1k_output_tokens=0.0,
        avg_latency_ms=25,
        capabilities=["reasoning", "content_generation"],
        tier="fast",
    ),
]

_TOGETHER_MODELS: list[ProviderModel] = [
    ProviderModel(
        provider="together",
        model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        cost_per_1k_input_tokens=0.00088,
        cost_per_1k_output_tokens=0.00088,
        avg_latency_ms=200,
        capabilities=["reasoning", "analysis", "code_generation"],
        tier="powerful",
    ),
    ProviderModel(
        provider="together",
        model="Qwen/Qwen2.5-72B-Instruct-Turbo",
        cost_per_1k_input_tokens=0.00088,
        cost_per_1k_output_tokens=0.00088,
        avg_latency_ms=220,
        capabilities=["reasoning", "analysis", "content_generation"],
        tier="powerful",
    ),
]

_OPENROUTER_FREE_MODELS: list[ProviderModel] = [
    ProviderModel(
        provider="openrouter",
        model="meta-llama/llama-3.2-3b-instruct:free",
        cost_per_1k_input_tokens=0.0,
        cost_per_1k_output_tokens=0.0,
        avg_latency_ms=300,
        capabilities=["reasoning", "content_generation"],
        tier="balanced",
    ),
]


# ---------------------------------------------------------------------------
# Setup function
# ---------------------------------------------------------------------------


async def setup_free_providers() -> KortexRuntime:
    """Create and return a KortexRuntime configured with free providers.

    Each provider is registered only if its API key is present in the
    environment.  At least one key must be set or the returned runtime
    will have no models registered.

    Returns:
        A fully configured ``KortexRuntime`` with all available free
        providers registered.
    """
    registry = ProviderRegistry()
    router = Router()
    registered_providers: list[str] = []

    # -- Cerebras (registered first — fastest free-tier, wins cost tie-breaks) --
    cerebras_key = os.getenv("CEREBRAS_API_KEY")
    if cerebras_key:
        registry.register_openai_compatible(
            name="cerebras",
            base_url="https://api.cerebras.ai/v1",
            api_key=cerebras_key,
            models=_CEREBRAS_MODELS,
        )
        for m in _CEREBRAS_MODELS:
            router.register_model(m)
        registered_providers.append("cerebras")

    # -- Groq -----------------------------------------------------------------
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        registry.register_openai_compatible(
            name="groq",
            base_url="https://api.groq.com/openai/v1",
            api_key=groq_key,
            models=_GROQ_MODELS,
        )
        for m in _GROQ_MODELS:
            router.register_model(m)
        registered_providers.append("groq")

    # -- Together AI ----------------------------------------------------------
    together_key = os.getenv("TOGETHER_API_KEY")
    if together_key:
        registry.register_openai_compatible(
            name="together",
            base_url="https://api.together.xyz/v1",
            api_key=together_key,
            models=_TOGETHER_MODELS,
        )
        for m in _TOGETHER_MODELS:
            router.register_model(m)
        registered_providers.append("together")

    # -- OpenRouter -----------------------------------------------------------
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        registry.register_openai_compatible(
            name="openrouter",
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_key,
            models=_OPENROUTER_FREE_MODELS,
            extra_headers={"HTTP-Referer": "https://github.com/kortex-ai/kortex"},
        )
        for m in _OPENROUTER_FREE_MODELS:
            router.register_model(m)
        registered_providers.append("openrouter")

    state = StateManager(InMemoryStateStore())
    runtime = KortexRuntime(router=router, state_manager=state, registry=registry)

    # Print summary
    all_models = router.models
    print(f"Registered {len(registered_providers)} provider(s): "
          f"{', '.join(registered_providers) if registered_providers else 'none'}")
    print(f"Total models available: {len(all_models)}")
    if all_models:
        print("\nModel catalogue:")
        for m in sorted(all_models, key=lambda x: x.avg_latency_ms):
            cost_str = f"${m.cost_per_1k_input_tokens:.5f}/1k" if m.cost_per_1k_input_tokens > 0 else "free"
            print(f"  {m.provider}/{m.model}  "
                  f"latency={m.avg_latency_ms}ms  "
                  f"cost={cost_str}  "
                  f"tier={m.tier}")

    return runtime


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    runtime = asyncio.run(setup_free_providers())
    all_models = runtime.get_router().models
    if not all_models:
        print("\nNo API keys found.")
        print("Set one or more of: GROQ_API_KEY, CEREBRAS_API_KEY, "
              "TOGETHER_API_KEY, OPENROUTER_API_KEY")
        sys.exit(1)
    print(f"\nRuntime ready with {len(all_models)} model(s).")
