"""Provider registry for managing and auto-discovering LLM provider connectors.

Aggregates models from all registered providers and supports automatic
discovery based on which API keys are present in the environment.
"""

from __future__ import annotations

import os

import structlog

from kortex.core.router import ProviderModel
from kortex.providers.base import GenericOpenAIConnector, ProviderConnector

logger = structlog.get_logger(component="provider_registry")


class ProviderRegistry:
    """Central registry for LLM provider connectors.

    Manages provider lifecycle and aggregates available models across
    all registered providers.
    """

    def __init__(self) -> None:
        self._providers: dict[str, ProviderConnector] = {}

    def register_provider(self, connector: ProviderConnector) -> None:
        """Register a provider connector.

        Args:
            connector: A ProviderConnector instance to register.
        """
        self._providers[connector.provider_name] = connector
        logger.info("provider_registered", provider=connector.provider_name)

    def register_openai_compatible(
        self,
        name: str,
        base_url: str,
        api_key: str | None = None,
        models: list[ProviderModel] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> GenericOpenAIConnector:
        """Register any OpenAI-compatible endpoint in a few lines.

        This is the easy path for users to add local models (Ollama, vLLM,
        llama.cpp, LM Studio) or cloud providers (Together AI, Fireworks,
        Groq, Mistral, Azure OpenAI) without writing a custom connector.

        Args:
            name: Provider name (e.g. "ollama", "corporate-api").
            base_url: The API base URL (e.g. "http://localhost:11434/v1").
            api_key: Optional API key. None for local models.
            models: ProviderModel definitions. Empty list if not specified.
            extra_headers: Additional headers for every request.

        Returns:
            The created GenericOpenAIConnector instance.
        """
        connector = GenericOpenAIConnector(
            base_url=base_url,
            api_key=api_key,
            name=name,
            models=models,
            extra_headers=extra_headers,
        )
        self.register_provider(connector)  # type: ignore[arg-type]
        return connector

    def get_provider(self, name: str) -> ProviderConnector:
        """Retrieve a registered provider by name.

        Args:
            name: The provider name (e.g. "anthropic").

        Returns:
            The registered ProviderConnector.

        Raises:
            KeyError: If the provider is not registered.
        """
        if name not in self._providers:
            raise KeyError(f"Provider '{name}' is not registered")
        return self._providers[name]

    def list_providers(self) -> list[str]:
        """Return names of all registered providers.

        Returns:
            List of registered provider names.
        """
        return list(self._providers.keys())

    def auto_discover(self) -> None:
        """Check which API keys are present in env and auto-register providers.

        Looks for ANTHROPIC_API_KEY, OPENAI_API_KEY, and OPENROUTER_API_KEY.
        Only registers providers whose keys are set and non-empty.
        """
        key_map: dict[str, tuple[str, type]] = {
            "ANTHROPIC_API_KEY": ("anthropic", _lazy_anthropic),
            "OPENAI_API_KEY": ("openai", _lazy_openai),
            "OPENROUTER_API_KEY": ("openrouter", _lazy_openrouter),
        }

        for env_var, (provider_name, factory) in key_map.items():
            api_key = os.environ.get(env_var, "")
            if api_key:
                if provider_name not in self._providers:
                    connector = factory(api_key)
                    self.register_provider(connector)
                    logger.info(
                        "provider_auto_discovered",
                        provider=provider_name,
                        env_var=env_var,
                    )

    async def close_all(self) -> None:
        """Close all registered provider clients.

        Calls ``close()`` on each connector that implements it.
        Errors during close are logged but not raised.
        """
        for name, connector in self._providers.items():
            close_fn = getattr(connector, "close", None)
            if close_fn is not None and callable(close_fn):
                try:
                    await close_fn()
                except Exception:
                    logger.warning("provider_close_error", provider=name)

    def get_all_models(self) -> list[ProviderModel]:
        """Aggregate available models from all registered providers.

        Deduplicates by composite identity key (provider::model_name) so
        that two providers serving the same model name are both retained.

        Returns:
            List of unique ProviderModel instances across all providers.
        """
        seen: set[str] = set()
        models: list[ProviderModel] = []

        for connector in self._providers.values():
            for model in connector.get_available_models():
                key = model.identity.key
                if key not in seen:
                    seen.add(key)
                    models.append(model)

        return models


def _lazy_anthropic(api_key: str) -> ProviderConnector:
    """Lazily import and create an AnthropicConnector."""
    from kortex.providers.anthropic import AnthropicConnector

    return AnthropicConnector(api_key=api_key)  # type: ignore[return-value]


def _lazy_openai(api_key: str) -> ProviderConnector:
    """Lazily import and create an OpenAIConnector."""
    from kortex.providers.openai import OpenAIConnector

    return OpenAIConnector(api_key=api_key)  # type: ignore[return-value]


def _lazy_openrouter(api_key: str) -> ProviderConnector:
    """Lazily import and create an OpenRouterConnector."""
    from kortex.providers.openrouter import OpenRouterConnector

    return OpenRouterConnector(api_key=api_key)  # type: ignore[return-value]
