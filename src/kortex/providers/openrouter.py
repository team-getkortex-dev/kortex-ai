"""OpenRouter provider connector for third-party model access.

Thin wrapper around GenericOpenAIConnector with pre-configured model catalog
and OpenRouter-specific headers. Reads OPENROUTER_API_KEY from environment.
"""

from __future__ import annotations

from kortex.core.router import ProviderModel
from kortex.providers.base import GenericOpenAIConnector

_API_BASE = "https://openrouter.ai/api/v1"


class OpenRouterConnector(GenericOpenAIConnector):
    """Connector for the OpenRouter Chat Completions API.

    Args:
        api_key: The OpenRouter API key.
        base_url: Override the API base URL (useful for testing).
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = _API_BASE,
        timeout: float = 60.0,
    ) -> None:
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            name="openrouter",
            models=_default_models(),
            extra_headers={
                "HTTP-Referer": "https://kortex.dev",
                "X-Title": "Kortex",
            },
            timeout=timeout,
        )


def _default_models() -> list[ProviderModel]:
    return [
        ProviderModel(
            provider="openrouter",
            model="deepseek/deepseek-r1",
            cost_per_1k_input_tokens=0.0008,
            cost_per_1k_output_tokens=0.002,
            avg_latency_ms=1500,
            capabilities=["reasoning", "code_generation", "analysis"],
            max_context_tokens=128_000,
            tier="powerful",
        ),
        ProviderModel(
            provider="openrouter",
            model="meta-llama/llama-4-maverick",
            cost_per_1k_input_tokens=0.0005,
            cost_per_1k_output_tokens=0.0015,
            avg_latency_ms=400,
            capabilities=["reasoning", "code_generation"],
            max_context_tokens=128_000,
            tier="fast",
        ),
        ProviderModel(
            provider="openrouter",
            model="mistralai/mistral-large",
            cost_per_1k_input_tokens=0.002,
            cost_per_1k_output_tokens=0.006,
            avg_latency_ms=700,
            capabilities=["reasoning", "code_generation", "analysis"],
            max_context_tokens=128_000,
            tier="balanced",
        ),
    ]
