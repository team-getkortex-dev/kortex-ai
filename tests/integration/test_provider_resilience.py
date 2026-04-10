"""Integration tests for provider resilience layer."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from kortex.core.exceptions import CircuitOpenError, ProviderOverloadError
from kortex.core.router import ProviderModel, Router
from kortex.core.runtime import AgentDescriptor, KortexRuntime
from kortex.core.state import StateManager
from kortex.providers.base import GenericOpenAIConnector
from kortex.providers.registry import ProviderRegistry
from kortex.providers.resilience import CircuitBreaker, RetryPolicy
from kortex.providers.resilient_client import ResilientClient
from kortex.store.memory import InMemoryStateStore


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _test_models() -> list[ProviderModel]:
    return [
        ProviderModel(
            provider="test", model="fast-model",
            cost_per_1k_input_tokens=0.001, cost_per_1k_output_tokens=0.002,
            avg_latency_ms=100, capabilities=["reasoning"], tier="fast",
        ),
    ]


def _success_response() -> httpx.Response:
    data = {
        "choices": [{"message": {"content": "Hello!"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    return httpx.Response(
        200, json=data,
        request=httpx.Request("POST", "https://test.example.com/chat/completions"),
    )


def _error_response(status: int = 503) -> httpx.Response:
    return httpx.Response(
        status, json={"error": "overloaded"},
        request=httpx.Request("POST", "https://test.example.com/chat/completions"),
    )


# ---------------------------------------------------------------------------
# 19. GenericOpenAIConnector with ResilientClient retries transient failure
# ---------------------------------------------------------------------------


class TestConnectorWithResilientClient:
    @pytest.mark.asyncio
    async def test_retries_transient_failure(self) -> None:
        retry_policy = RetryPolicy(max_retries=1, backoff_base_ms=1)
        resilient = ResilientClient(retry_policy=retry_policy)

        call_count = 0

        async def mock_request(method, url, *, headers=None, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _error_response(503)
            return _success_response()

        with patch.object(resilient._get_client(), "request", side_effect=mock_request):
            connector = GenericOpenAIConnector(
                base_url="https://test.example.com",
                api_key="test-key",
                name="test",
                models=_test_models(),
                resilient_client=resilient,
            )
            response = await connector.complete("Hello", "fast-model")

        assert response.content == "Hello!"
        assert call_count == 2
        await resilient.close()


# ---------------------------------------------------------------------------
# 20. Circuit breaker opens after repeated failures, rejects subsequent calls
# ---------------------------------------------------------------------------


class TestCircuitBreakerIntegration:
    @pytest.mark.asyncio
    async def test_circuit_opens_and_rejects(self) -> None:
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout_s=9999)
        retry_policy = RetryPolicy(max_retries=0, backoff_base_ms=1)
        resilient = ResilientClient(
            retry_policy=retry_policy, circuit_breaker=breaker,
        )

        async def mock_request(*args, **kwargs):
            return _error_response(500)

        with patch.object(resilient._get_client(), "request", side_effect=mock_request):
            connector = GenericOpenAIConnector(
                base_url="https://test.example.com",
                name="test",
                models=_test_models(),
                resilient_client=resilient,
            )

            # First two calls fail and open the circuit
            with pytest.raises(ProviderOverloadError):
                await connector.complete("Hello", "fast-model")
            with pytest.raises(ProviderOverloadError):
                await connector.complete("Hello", "fast-model")

            # Third call should be rejected by circuit breaker
            with pytest.raises(CircuitOpenError):
                await connector.complete("Hello", "fast-model")

        await resilient.close()


# ---------------------------------------------------------------------------
# 21. Provider registry close_all shuts down cleanly
# ---------------------------------------------------------------------------


class TestRegistryCloseAll:
    @pytest.mark.asyncio
    async def test_close_all(self) -> None:
        registry = ProviderRegistry()

        connector = GenericOpenAIConnector(
            base_url="https://test.example.com",
            name="test",
            models=_test_models(),
        )
        # Force client creation via the connection pool
        from kortex.providers.http_pool import ConnectionPool
        ConnectionPool.reset()
        client = connector._get_client()
        assert not client.is_closed
        registry.register_provider(connector)  # type: ignore[arg-type]

        # registry.close_all() closes resilient clients; pool closed by runtime.stop()
        await registry.close_all()

        # After runtime-level pool teardown the pooled client is closed
        await ConnectionPool.get_instance().close_all()
        assert client.is_closed
        ConnectionPool.reset()


# ---------------------------------------------------------------------------
# 22. Runtime.stop() closes provider registry
# ---------------------------------------------------------------------------


class TestRuntimeClosesRegistry:
    @pytest.mark.asyncio
    async def test_stop_closes_registry(self) -> None:
        registry = ProviderRegistry()
        close_called = False

        original_close = registry.close_all

        async def tracking_close():
            nonlocal close_called
            close_called = True
            await original_close()

        registry.close_all = tracking_close  # type: ignore[assignment]

        router = Router()
        state_manager = StateManager(InMemoryStateStore())
        runtime = KortexRuntime(
            router=router, state_manager=state_manager, registry=registry,
        )

        async with runtime:
            pass  # start + stop

        assert close_called is True


# ---------------------------------------------------------------------------
# 23. Backward compatible — connector without explicit ResilientClient works
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    @pytest.mark.asyncio
    async def test_connector_without_resilient_client(self) -> None:
        connector = GenericOpenAIConnector(
            base_url="https://test.example.com",
            api_key="test-key",
            name="test",
            models=_test_models(),
        )

        # Should have no resilient client
        assert connector._resilient_client is None

        # Should still be able to create the plain httpx client
        client = connector._get_client()
        assert client is not None
        assert not client.is_closed

        await connector.close()
