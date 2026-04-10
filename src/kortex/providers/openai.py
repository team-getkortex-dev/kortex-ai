"""OpenAI provider connector for GPT and o-series models.

Thin wrapper around GenericOpenAIConnector with pre-configured model catalog.
Reads OPENAI_API_KEY from environment variables.
"""

from __future__ import annotations

from kortex.core.router import ProviderModel
from kortex.providers.base import GenericOpenAIConnector

_API_BASE = "https://api.openai.com/v1"


class OpenAIConnector(GenericOpenAIConnector):
    """Connector for the OpenAI Chat Completions API.

    Args:
        api_key: The OpenAI API key.
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
            name="openai",
            models=_default_models(),
            timeout=timeout,
        )


def _default_models() -> list[ProviderModel]:
    return [
        ProviderModel(
            provider="openai",
            model="gpt-4o",
            cost_per_1k_input_tokens=0.0025,
            cost_per_1k_output_tokens=0.01,
            avg_latency_ms=600,
            capabilities=["reasoning", "code_generation", "analysis", "vision"],
            max_context_tokens=128_000,
            tier="balanced",
        ),
        ProviderModel(
            provider="openai",
            model="gpt-4o-mini",
            cost_per_1k_input_tokens=0.00015,
            cost_per_1k_output_tokens=0.0006,
            avg_latency_ms=250,
            capabilities=["code_generation", "analysis"],
            max_context_tokens=128_000,
            tier="fast",
        ),
        ProviderModel(
            provider="openai",
            model="o3",
            cost_per_1k_input_tokens=0.01,
            cost_per_1k_output_tokens=0.04,
            avg_latency_ms=3000,
            capabilities=["reasoning", "code_generation", "analysis"],
            max_context_tokens=200_000,
            tier="powerful",
        ),
    ]
