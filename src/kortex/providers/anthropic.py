"""Anthropic provider connector for the Claude model family.

Uses httpx.AsyncClient to call the Anthropic Messages API.
Reads ANTHROPIC_API_KEY from environment variables.
"""

from __future__ import annotations

import time
from types import TracebackType
from typing import TYPE_CHECKING, Any, AsyncIterator

import httpx
import structlog

from kortex.core.router import ProviderModel
from kortex.providers.base import ProviderResponse

if TYPE_CHECKING:
    from kortex.providers.resilient_client import ResilientClient

logger = structlog.get_logger(component="provider.anthropic")

_API_BASE = "https://api.anthropic.com/v1"
_API_VERSION = "2023-06-01"

# Model pricing per 1k tokens (USD)
_MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4-20250514": {"input": 0.015, "output": 0.075},
    "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
    "claude-haiku-3-5-20241022": {"input": 0.0008, "output": 0.004},
}


class AnthropicConnector:
    """Connector for the Anthropic Messages API.

    Args:
        api_key: The Anthropic API key. If not provided, reads from
            the ANTHROPIC_API_KEY environment variable at call time.
        base_url: Override the API base URL (useful for testing).
        timeout: Request timeout in seconds.
        resilient_client: Optional ResilientClient for retry/circuit-breaker
            support. If not provided, a plain httpx.AsyncClient is used.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = _API_BASE,
        timeout: float = 60.0,
        resilient_client: ResilientClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._resilient_client: ResilientClient | None = resilient_client

    @property
    def provider_name(self) -> str:
        """Return the canonical provider name."""
        return "anthropic"

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": _API_VERSION,
                    "content-type": "application/json",
                },
            )
        return self._client

    async def complete(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Send a completion request to the Anthropic Messages API.

        Args:
            prompt: The input prompt text.
            model: Model identifier (e.g. "claude-sonnet-4-20250514").
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
            url = f"{self._base_url}/v1/messages"
            headers = {
                "x-api-key": self._api_key,
                "anthropic-version": _API_VERSION,
                "content-type": "application/json",
            }
            response = await self._resilient_client.request(
                "POST", url, headers=headers, json=body,
            )
        else:
            client = self._get_client()
            response = await client.post("/v1/messages", json=body)

        latency_ms = (time.monotonic() - start) * 1000

        response.raise_for_status()
        data = response.json()

        content = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                content += block.get("text", "")

        usage = data.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        pricing = _MODEL_PRICING.get(model, {"input": 0.003, "output": 0.015})
        cost_usd = (
            pricing["input"] * input_tokens / 1000
            + pricing["output"] * output_tokens / 1000
        )

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
        """Stream completion tokens from the Anthropic Messages API.

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

        async with client.stream("POST", "/v1/messages", json=body) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    import json

                    event = json.loads(line[6:])
                    if event.get("type") == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            yield delta.get("text", "")

    async def health_check(self) -> bool:
        """Ping the Anthropic API to verify connectivity.

        Returns:
            True if the API is reachable, False otherwise.
        """
        try:
            client = self._get_client()
            # A minimal request — will 401 without valid key but proves connectivity
            response = await client.get("/v1/models")
            # Any non-connection-error response means the API is reachable
            return response.status_code < 500
        except (httpx.ConnectError, httpx.TimeoutException, OSError):
            return False

    async def close(self) -> None:
        """Close all underlying HTTP clients."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        if self._resilient_client is not None:
            await self._resilient_client.close()

    async def __aenter__(self) -> AnthropicConnector:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()

    def get_available_models(self) -> list[ProviderModel]:
        """Return pre-configured Anthropic models with current pricing.

        Returns:
            List of ProviderModel instances for Claude models.
        """
        return [
            ProviderModel(
                provider="anthropic",
                model="claude-opus-4-20250514",
                cost_per_1k_input_tokens=0.015,
                cost_per_1k_output_tokens=0.075,
                avg_latency_ms=2000,
                capabilities=["reasoning", "code_generation", "analysis", "vision"],
                max_context_tokens=200_000,
                tier="powerful",
            ),
            ProviderModel(
                provider="anthropic",
                model="claude-sonnet-4-20250514",
                cost_per_1k_input_tokens=0.003,
                cost_per_1k_output_tokens=0.015,
                avg_latency_ms=800,
                capabilities=["reasoning", "code_generation", "analysis", "vision"],
                max_context_tokens=200_000,
                tier="balanced",
            ),
            ProviderModel(
                provider="anthropic",
                model="claude-haiku-3-5-20241022",
                cost_per_1k_input_tokens=0.0008,
                cost_per_1k_output_tokens=0.004,
                avg_latency_ms=300,
                capabilities=["code_generation", "analysis"],
                max_context_tokens=200_000,
                tier="fast",
            ),
        ]
