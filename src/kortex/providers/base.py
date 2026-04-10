"""Abstract base for LLM provider connectors.

Defines the ProviderConnector protocol, the ProviderResponse dataclass,
and GenericOpenAIConnector — a universal connector for any OpenAI-compatible API.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from types import TracebackType
from typing import TYPE_CHECKING, Any, AsyncIterator, Protocol, runtime_checkable

import httpx
import structlog

from kortex.core.router import ProviderModel

if TYPE_CHECKING:
    from kortex.providers.resilient_client import ResilientClient

logger = structlog.get_logger(component="provider.generic")


@dataclass
class ProviderResponse:
    """Response from an LLM provider API call.

    Args:
        content: The generated text content.
        model: Model identifier used for the request.
        provider: Provider name (e.g. "anthropic", "openai").
        input_tokens: Number of input tokens consumed.
        output_tokens: Number of output tokens generated.
        cost_usd: Calculated cost in USD from token counts and model pricing.
            Always 0.0 for local/free models.
        latency_ms: Measured wall-clock latency in milliseconds.
        raw_response: The full API response for debugging.
    """

    content: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    raw_response: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ProviderConnector(Protocol):
    """Protocol that all LLM provider connectors must implement."""

    @property
    def provider_name(self) -> str:
        """Return the canonical provider name (e.g. 'anthropic')."""
        ...

    async def complete(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Send a completion request and return the full response.

        Args:
            prompt: The input prompt text.
            model: Model identifier to use.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            **kwargs: Additional provider-specific parameters.

        Returns:
            A ProviderResponse with content, token counts, cost, and latency.
        """
        ...

    async def stream(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream completion tokens as they are generated.

        Args:
            prompt: The input prompt text.
            model: Model identifier to use.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            **kwargs: Additional provider-specific parameters.

        Yields:
            Text chunks as they arrive from the provider.
        """
        ...

    async def health_check(self) -> bool:
        """Ping the provider API to verify connectivity.

        Returns:
            True if the API is reachable, False otherwise.
        """
        ...

    def get_available_models(self) -> list[ProviderModel]:
        """Return pre-configured ProviderModel instances with pricing data.

        Returns:
            List of ProviderModel instances this connector supports.
        """
        ...


class GenericOpenAIConnector:
    """Universal connector for any OpenAI-compatible API.

    Works with: OpenAI, Ollama, vLLM, llama.cpp, LM Studio, Together AI,
    Fireworks AI, Groq, Mistral, Azure OpenAI, or any endpoint that
    implements the OpenAI chat completions format.

    Args:
        base_url: The API base URL (e.g. "http://localhost:11434/v1" for Ollama).
        api_key: Optional API key. If None or empty, requests are sent without
            an Authorization header (suitable for local models).
        name: The provider name returned by ``provider_name``.
        models: Pre-configured ProviderModel definitions for this provider.
        extra_headers: Additional headers to include in every request.
        timeout: Request timeout in seconds.
        resilient_client: Optional ResilientClient for retry/circuit-breaker
            support. If not provided, a plain httpx.AsyncClient is used.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        name: str = "generic",
        models: list[ProviderModel] | None = None,
        extra_headers: dict[str, str] | None = None,
        timeout: float = 60.0,
        resilient_client: ResilientClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._name = name
        self._models = list(models) if models else []
        self._extra_headers = extra_headers or {}
        self._timeout = timeout
        self._resilient_client: ResilientClient | None = resilient_client
        self._log = structlog.get_logger(component=f"provider.{name}")

        # Build pricing lookup from model definitions
        self._pricing: dict[str, dict[str, float]] = {}
        for m in self._models:
            self._pricing[m.model] = {
                "input": m.cost_per_1k_input_tokens,
                "output": m.cost_per_1k_output_tokens,
            }

    @property
    def provider_name(self) -> str:
        """Return the canonical provider name."""
        return self._name

    def _get_headers(self) -> dict[str, str]:
        """Build per-request auth and content-type headers."""
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        headers.update(self._extra_headers)
        return headers

    def _get_client(self) -> httpx.AsyncClient:
        """Return the pooled httpx.AsyncClient for this connector's base URL.

        Uses the process-wide ``ConnectionPool`` singleton so that all
        connectors pointing at the same endpoint share the same underlying
        TCP/HTTP2 connections.  Auth headers are **not** embedded here —
        they are passed per-request via ``_get_headers()``.
        """
        from kortex.providers.http_pool import ConnectionPool

        return ConnectionPool.get_instance().get_client(self._base_url, self._timeout)

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in USD. Returns 0.0 for zero-cost models."""
        pricing = self._pricing.get(model, {"input": 0.0, "output": 0.0})
        if pricing["input"] == 0.0 and pricing["output"] == 0.0:
            return 0.0
        return (
            pricing["input"] * input_tokens / 1000
            + pricing["output"] * output_tokens / 1000
        )

    async def complete(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Send a completion request to the OpenAI-compatible endpoint.

        Args:
            prompt: The input prompt text.
            model: Model identifier.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            **kwargs: Additional parameters forwarded to the API.

        Returns:
            A ProviderResponse with content, token counts, cost, and latency.
        """
        body: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
            **kwargs,
        }

        start = time.monotonic()

        if self._resilient_client is not None:
            url = f"{self._base_url}/chat/completions"
            response = await self._resilient_client.request(
                "POST", url, headers=self._get_headers(), json=body,
            )
        else:
            client = self._get_client()
            response = await client.post(
                "/chat/completions", json=body, headers=self._get_headers()
            )

        latency_ms = (time.monotonic() - start) * 1000

        response.raise_for_status()
        data = response.json()

        choices = data.get("choices", [])
        content = choices[0]["message"]["content"] if choices else ""

        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        cost_usd = self._calculate_cost(model, input_tokens, output_tokens)

        return ProviderResponse(
            content=content,
            model=model,
            provider=self.provider_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            raw_response=data,
        )

    async def stream(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream completion tokens from the OpenAI-compatible endpoint.

        Args:
            prompt: The input prompt text.
            model: Model identifier.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            **kwargs: Additional parameters forwarded to the API.

        Yields:
            Text chunks as they arrive.
        """
        client = self._get_client()

        body: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "messages": [{"role": "user", "content": prompt}],
            **kwargs,
        }

        async with client.stream(
            "POST", "/chat/completions", json=body, headers=self._get_headers()
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: ") and line.strip() != "data: [DONE]":
                    chunk = json.loads(line[6:])
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        text = delta.get("content", "")
                        if text:
                            yield text

    async def health_check(self) -> bool:
        """Ping the API to verify connectivity.

        Returns:
            True if the API is reachable, False otherwise.
        """
        try:
            client = self._get_client()
            response = await client.get("/models", headers=self._get_headers())
            return response.status_code < 500
        except (httpx.ConnectError, httpx.TimeoutException, OSError):
            return False

    async def close(self) -> None:
        """Release this connector's resources.

        The pooled ``httpx.AsyncClient`` is managed by ``ConnectionPool`` and
        is **not** closed here — call ``ConnectionPool.get_instance().close_all()``
        (done automatically by ``KortexRuntime.stop()``) to release TCP connections.
        """
        if self._resilient_client is not None:
            await self._resilient_client.close()

    async def __aenter__(self) -> GenericOpenAIConnector:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()

    def get_available_models(self) -> list[ProviderModel]:
        """Return pre-configured models for this provider.

        Returns:
            List of ProviderModel instances.
        """
        return list(self._models)
