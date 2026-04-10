"""Unit tests for LLM provider connectors and registry."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from kortex.core.router import ProviderModel, Router
from kortex.core.types import TaskSpec
from kortex.providers.anthropic import AnthropicConnector
from kortex.providers.base import GenericOpenAIConnector, ProviderResponse
from kortex.providers.openai import OpenAIConnector
from kortex.providers.openrouter import OpenRouterConnector
from kortex.providers.registry import ProviderRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_httpx_response(data: dict[str, Any], status_code: int = 200) -> httpx.Response:
    """Build a fake httpx.Response."""
    return httpx.Response(
        status_code=status_code,
        json=data,
        request=httpx.Request("POST", "https://test.example.com"),
    )


_OPENAI_RESPONSE = {
    "choices": [{"message": {"content": "mock reply"}}],
    "usage": {"prompt_tokens": 20, "completion_tokens": 10},
}


# ---------------------------------------------------------------------------
# 1. GenericOpenAIConnector builds correct request for any base_url
# ---------------------------------------------------------------------------


class TestGenericOpenAIConnectorRequest:
    @pytest.mark.asyncio
    async def test_sends_to_correct_url_with_auth(self) -> None:
        connector = GenericOpenAIConnector(
            base_url="https://custom-llm.example.com/v1",
            api_key="my-secret",
            name="custom",
            models=[
                ProviderModel(
                    provider="custom",
                    model="custom-model",
                    cost_per_1k_input_tokens=0.001,
                    cost_per_1k_output_tokens=0.002,
                    avg_latency_ms=100,
                ),
            ],
        )
        # Auth headers are passed per-request (not baked into the pooled client)
        assert connector._get_headers()["Authorization"] == "Bearer my-secret"

        client = connector._get_client()
        mock_resp = _mock_httpx_response(_OPENAI_RESPONSE)
        with patch.object(client, "post", new_callable=AsyncMock, return_value=mock_resp) as mock_post:
            result = await connector.complete("Hello", "custom-model")

            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[0][0] == "/chat/completions"
            body = call_args[1]["json"]
            assert body["model"] == "custom-model"
            assert body["messages"] == [{"role": "user", "content": "Hello"}]
            # Auth must be present in per-request headers
            assert call_args[1]["headers"]["Authorization"] == "Bearer my-secret"

        assert result.content == "mock reply"
        assert result.provider == "custom"

    @pytest.mark.asyncio
    async def test_ollama_url(self) -> None:
        connector = GenericOpenAIConnector(
            base_url="http://localhost:11434/v1",
            name="ollama",
        )
        client = connector._get_client()
        assert str(client.base_url).rstrip("/") == "http://localhost:11434/v1"


# ---------------------------------------------------------------------------
# 2. GenericOpenAIConnector works without api_key (local model scenario)
# ---------------------------------------------------------------------------


class TestGenericNoApiKey:
    @pytest.mark.asyncio
    async def test_no_auth_header_when_key_is_none(self) -> None:
        connector = GenericOpenAIConnector(
            base_url="http://localhost:11434/v1",
            api_key=None,
            name="ollama-local",
        )
        client = connector._get_client()
        assert "authorization" not in {k.lower() for k in client.headers.keys()}

    @pytest.mark.asyncio
    async def test_no_auth_header_when_key_is_empty(self) -> None:
        connector = GenericOpenAIConnector(
            base_url="http://localhost:11434/v1",
            api_key="",
            name="ollama-local",
        )
        client = connector._get_client()
        assert "authorization" not in {k.lower() for k in client.headers.keys()}

    @pytest.mark.asyncio
    async def test_complete_works_without_key(self) -> None:
        connector = GenericOpenAIConnector(
            base_url="http://localhost:11434/v1",
            api_key=None,
            name="ollama",
            models=[
                ProviderModel(
                    provider="ollama",
                    model="llama3",
                    cost_per_1k_input_tokens=0.0,
                    cost_per_1k_output_tokens=0.0,
                    avg_latency_ms=100,
                ),
            ],
        )
        client = connector._get_client()
        mock_resp = _mock_httpx_response(_OPENAI_RESPONSE)
        with patch.object(client, "post", new_callable=AsyncMock, return_value=mock_resp):
            result = await connector.complete("test", "llama3")

        assert result.content == "mock reply"
        assert result.provider == "ollama"


# ---------------------------------------------------------------------------
# 3. GenericOpenAIConnector calculates cost_usd=0 for zero-cost models
# ---------------------------------------------------------------------------


class TestGenericZeroCost:
    @pytest.mark.asyncio
    async def test_zero_cost_for_free_models(self) -> None:
        connector = GenericOpenAIConnector(
            base_url="http://localhost:11434/v1",
            api_key=None,
            name="ollama",
            models=[
                ProviderModel(
                    provider="ollama",
                    model="llama3",
                    cost_per_1k_input_tokens=0.0,
                    cost_per_1k_output_tokens=0.0,
                    avg_latency_ms=100,
                ),
            ],
        )
        client = connector._get_client()

        # Response with real token counts -- cost should still be 0
        mock_resp = _mock_httpx_response({
            "choices": [{"message": {"content": "result"}}],
            "usage": {"prompt_tokens": 500, "completion_tokens": 200},
        })
        with patch.object(client, "post", new_callable=AsyncMock, return_value=mock_resp):
            result = await connector.complete("test", "llama3")

        assert result.cost_usd == 0.0
        assert result.input_tokens == 500
        assert result.output_tokens == 200

    def test_calculate_cost_zero(self) -> None:
        connector = GenericOpenAIConnector(
            base_url="http://localhost:11434/v1",
            name="ollama",
            models=[
                ProviderModel(
                    provider="ollama",
                    model="llama3",
                    cost_per_1k_input_tokens=0.0,
                    cost_per_1k_output_tokens=0.0,
                    avg_latency_ms=100,
                ),
            ],
        )
        assert connector._calculate_cost("llama3", 1000, 500) == 0.0

    def test_calculate_cost_nonzero(self) -> None:
        connector = GenericOpenAIConnector(
            base_url="https://api.example.com",
            name="paid",
            models=[
                ProviderModel(
                    provider="paid",
                    model="paid-model",
                    cost_per_1k_input_tokens=0.01,
                    cost_per_1k_output_tokens=0.02,
                    avg_latency_ms=100,
                ),
            ],
        )
        # 0.01 * 1000/1000 + 0.02 * 500/1000 = 0.01 + 0.01 = 0.02
        assert connector._calculate_cost("paid-model", 1000, 500) == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# 4. AnthropicConnector builds correct Anthropic-specific headers and body
# ---------------------------------------------------------------------------


class TestAnthropicConnectorRequest:
    @pytest.mark.asyncio
    async def test_headers_and_body(self) -> None:
        connector = AnthropicConnector(api_key="sk-ant-test-123")
        client = connector._get_client()

        assert client.headers["x-api-key"] == "sk-ant-test-123"
        assert client.headers["anthropic-version"] == "2023-06-01"
        assert client.headers["content-type"] == "application/json"

        mock_response = _mock_httpx_response({
            "content": [{"type": "text", "text": "Hello"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        })

        with patch.object(client, "post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
            result = await connector.complete("Say hello", "claude-sonnet-4-20250514")

            assert result.content == "Hello"
            assert result.provider == "anthropic"
            assert result.model == "claude-sonnet-4-20250514"
            assert result.input_tokens == 10
            assert result.output_tokens == 5

            call_args = mock_post.call_args
            assert call_args[0][0] == "/v1/messages"
            body = call_args[1]["json"]
            assert body["model"] == "claude-sonnet-4-20250514"
            assert body["messages"] == [{"role": "user", "content": "Say hello"}]
            assert body["max_tokens"] == 1024
            assert body["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_anthropic_uses_messages_format_not_chat_completions(self) -> None:
        """Anthropic API uses /v1/messages, not /chat/completions."""
        connector = AnthropicConnector(api_key="test")
        client = connector._get_client()

        mock_resp = _mock_httpx_response({
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 5, "output_tokens": 2},
        })
        with patch.object(client, "post", new_callable=AsyncMock, return_value=mock_resp) as mock_post:
            await connector.complete("test", "claude-haiku-3-5-20241022")
            assert mock_post.call_args[0][0] == "/v1/messages"


# ---------------------------------------------------------------------------
# 5. ProviderResponse calculates cost_usd correctly from token counts
# ---------------------------------------------------------------------------


class TestProviderResponseCost:
    def test_cost_calculation_anthropic_sonnet(self) -> None:
        response = ProviderResponse(
            content="test",
            model="claude-sonnet-4-20250514",
            provider="anthropic",
            input_tokens=1000,
            output_tokens=500,
            cost_usd=0.003 * 1000 / 1000 + 0.015 * 500 / 1000,
            latency_ms=100.0,
        )
        assert response.cost_usd == pytest.approx(0.0105)

    def test_cost_calculation_openai_mini(self) -> None:
        response = ProviderResponse(
            content="test",
            model="gpt-4o-mini",
            provider="openai",
            input_tokens=2000,
            output_tokens=1000,
            cost_usd=0.00015 * 2000 / 1000 + 0.0006 * 1000 / 1000,
            latency_ms=50.0,
        )
        assert response.cost_usd == pytest.approx(0.0009)

    @pytest.mark.asyncio
    async def test_anthropic_connector_computes_cost_from_tokens(self) -> None:
        connector = AnthropicConnector(api_key="test")
        client = connector._get_client()

        mock_response = _mock_httpx_response({
            "content": [{"type": "text", "text": "result"}],
            "usage": {"input_tokens": 500, "output_tokens": 200},
        })

        with patch.object(client, "post", new_callable=AsyncMock, return_value=mock_response):
            result = await connector.complete("test", "claude-sonnet-4-20250514")

        expected = 0.003 * 500 / 1000 + 0.015 * 200 / 1000
        assert result.cost_usd == pytest.approx(expected)

    def test_zero_cost_response(self) -> None:
        response = ProviderResponse(
            content="local result",
            model="llama3",
            provider="ollama",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.0,
            latency_ms=30.0,
        )
        assert response.cost_usd == 0.0


# ---------------------------------------------------------------------------
# 6. Health check returns False on connection error
# ---------------------------------------------------------------------------


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_anthropic_health_check_fails_on_connection_error(self) -> None:
        connector = AnthropicConnector(api_key="test")
        client = connector._get_client()
        with patch.object(
            client, "get", new_callable=AsyncMock, side_effect=httpx.ConnectError("refused")
        ):
            assert await connector.health_check() is False

    @pytest.mark.asyncio
    async def test_generic_health_check_fails_on_timeout(self) -> None:
        connector = GenericOpenAIConnector(
            base_url="http://localhost:11434/v1",
            name="ollama",
        )
        client = connector._get_client()
        with patch.object(
            client, "get", new_callable=AsyncMock, side_effect=httpx.TimeoutException("timed out")
        ):
            assert await connector.health_check() is False

    @pytest.mark.asyncio
    async def test_generic_health_check_fails_on_os_error(self) -> None:
        connector = GenericOpenAIConnector(
            base_url="http://localhost:11434/v1",
            name="ollama",
        )
        client = connector._get_client()
        with patch.object(
            client, "get", new_callable=AsyncMock, side_effect=OSError("network down")
        ):
            assert await connector.health_check() is False

    @pytest.mark.asyncio
    async def test_health_check_succeeds_on_200(self) -> None:
        connector = AnthropicConnector(api_key="test")
        client = connector._get_client()
        mock_response = _mock_httpx_response({"models": []}, status_code=200)
        with patch.object(client, "get", new_callable=AsyncMock, return_value=mock_response):
            assert await connector.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_succeeds_on_401(self) -> None:
        """A 401 means the API is reachable (just bad key), not a health failure."""
        connector = OpenAIConnector(api_key="bad-key")
        client = connector._get_client()
        mock_response = _mock_httpx_response({"error": "unauthorized"}, status_code=401)
        with patch.object(client, "get", new_callable=AsyncMock, return_value=mock_response):
            assert await connector.health_check() is True


# ---------------------------------------------------------------------------
# 7. ProviderRegistry.register_openai_compatible creates working connector
# ---------------------------------------------------------------------------


class TestRegisterOpenAICompatible:
    def test_creates_and_registers_connector(self) -> None:
        registry = ProviderRegistry()
        connector = registry.register_openai_compatible(
            name="ollama",
            base_url="http://localhost:11434/v1",
            models=[
                ProviderModel(
                    provider="ollama",
                    model="llama3",
                    cost_per_1k_input_tokens=0.0,
                    cost_per_1k_output_tokens=0.0,
                    avg_latency_ms=100,
                ),
            ],
        )
        assert "ollama" in registry.list_providers()
        assert connector.provider_name == "ollama"
        assert len(connector.get_available_models()) == 1

    def test_no_auth_for_local(self) -> None:
        registry = ProviderRegistry()
        connector = registry.register_openai_compatible(
            name="local",
            base_url="http://localhost:8080/v1",
            api_key=None,
        )
        client = connector._get_client()
        assert "authorization" not in {k.lower() for k in client.headers.keys()}

    def test_auth_for_cloud(self) -> None:
        registry = ProviderRegistry()
        connector = registry.register_openai_compatible(
            name="together",
            base_url="https://api.together.xyz/v1",
            api_key="tog-key-123",
        )
        # Auth passed per-request via _get_headers(), not baked into pooled client
        assert connector._get_headers()["Authorization"] == "Bearer tog-key-123"

    def test_extra_headers_applied(self) -> None:
        registry = ProviderRegistry()
        connector = registry.register_openai_compatible(
            name="corp",
            base_url="https://llm.corp.com/v1",
            api_key="corp-key",
            extra_headers={"X-Team": "platform"},
        )
        # Extra headers included in per-request headers
        assert connector._get_headers()["X-Team"] == "platform"

    @pytest.mark.asyncio
    async def test_registered_connector_can_complete(self) -> None:
        registry = ProviderRegistry()
        connector = registry.register_openai_compatible(
            name="test-api",
            base_url="https://test.example.com/v1",
            api_key="key",
            models=[
                ProviderModel(
                    provider="test-api",
                    model="test-model",
                    cost_per_1k_input_tokens=0.001,
                    cost_per_1k_output_tokens=0.002,
                    avg_latency_ms=100,
                ),
            ],
        )
        client = connector._get_client()
        mock_resp = _mock_httpx_response(_OPENAI_RESPONSE)
        with patch.object(client, "post", new_callable=AsyncMock, return_value=mock_resp):
            result = await connector.complete("hello", "test-model")
        assert result.content == "mock reply"
        assert result.provider == "test-api"


# ---------------------------------------------------------------------------
# 8. ProviderRegistry auto-discovers based on env vars
# ---------------------------------------------------------------------------


class TestProviderRegistryAutoDiscover:
    def test_auto_discover_with_anthropic_key(self) -> None:
        registry = ProviderRegistry()
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=False):
            registry.auto_discover()
        assert "anthropic" in registry.list_providers()
        assert "openai" not in registry.list_providers()

    def test_auto_discover_with_all_keys(self) -> None:
        registry = ProviderRegistry()
        with patch.dict(
            "os.environ",
            {
                "ANTHROPIC_API_KEY": "sk-ant-test",
                "OPENAI_API_KEY": "sk-openai-test",
                "OPENROUTER_API_KEY": "sk-or-test",
            },
            clear=False,
        ):
            registry.auto_discover()
        providers = registry.list_providers()
        assert "anthropic" in providers
        assert "openai" in providers
        assert "openrouter" in providers

    def test_auto_discover_skips_empty_keys(self) -> None:
        registry = ProviderRegistry()
        with patch.dict(
            "os.environ",
            {"ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": "sk-openai-test"},
            clear=False,
        ):
            registry.auto_discover()
        assert "anthropic" not in registry.list_providers()
        assert "openai" in registry.list_providers()

    def test_auto_discover_does_not_duplicate(self) -> None:
        registry = ProviderRegistry()
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=False):
            registry.auto_discover()
            registry.auto_discover()
        assert registry.list_providers().count("anthropic") == 1


# ---------------------------------------------------------------------------
# 9. get_all_models aggregates across mixed local + cloud providers
# ---------------------------------------------------------------------------


class TestGetAllModels:
    def test_aggregates_local_and_cloud(self) -> None:
        registry = ProviderRegistry()

        # Register local Ollama
        registry.register_openai_compatible(
            name="ollama",
            base_url="http://localhost:11434/v1",
            models=[
                ProviderModel(
                    provider="ollama",
                    model="llama3",
                    cost_per_1k_input_tokens=0.0,
                    cost_per_1k_output_tokens=0.0,
                    avg_latency_ms=100,
                    tier="fast",
                ),
            ],
        )

        # Register cloud Anthropic
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=False):
            registry.auto_discover()

        models = registry.get_all_models()
        model_names = [m.model for m in models]

        assert "llama3" in model_names
        assert "claude-sonnet-4-20250514" in model_names
        assert len(model_names) == len(set(model_names)), "Duplicates found"

    def test_no_duplicates_when_model_names_overlap(self) -> None:
        registry = ProviderRegistry()

        mock1 = MagicMock()
        mock1.provider_name = "provider_a"
        mock1.get_available_models.return_value = [
            ProviderModel(
                provider="provider_a",
                model="shared-model",
                cost_per_1k_input_tokens=0.001,
                cost_per_1k_output_tokens=0.002,
                avg_latency_ms=100,
            ),
        ]

        mock2 = MagicMock()
        mock2.provider_name = "provider_b"
        mock2.get_available_models.return_value = [
            ProviderModel(
                provider="provider_b",
                model="shared-model",
                cost_per_1k_input_tokens=0.005,
                cost_per_1k_output_tokens=0.01,
                avg_latency_ms=200,
            ),
        ]

        registry.register_provider(mock1)
        registry.register_provider(mock2)

        models = registry.get_all_models()
        shared = [m for m in models if m.model == "shared-model"]
        # Same model name from different providers should both be retained
        # (composite identity key prevents silent collisions)
        assert len(shared) == 2
        providers = {m.provider for m in shared}
        assert providers == {"provider_a", "provider_b"}

    def test_mixed_zero_cost_and_paid(self) -> None:
        registry = ProviderRegistry()
        registry.register_openai_compatible(
            name="local",
            base_url="http://localhost:8080/v1",
            models=[
                ProviderModel(
                    provider="local",
                    model="local-model",
                    cost_per_1k_input_tokens=0.0,
                    cost_per_1k_output_tokens=0.0,
                    avg_latency_ms=50,
                ),
            ],
        )
        registry.register_openai_compatible(
            name="cloud",
            base_url="https://api.example.com/v1",
            api_key="key",
            models=[
                ProviderModel(
                    provider="cloud",
                    model="cloud-model",
                    cost_per_1k_input_tokens=0.01,
                    cost_per_1k_output_tokens=0.02,
                    avg_latency_ms=500,
                ),
            ],
        )

        models = registry.get_all_models()
        local = [m for m in models if m.model == "local-model"][0]
        cloud = [m for m in models if m.model == "cloud-model"][0]
        assert local.cost_per_1k_input_tokens == 0.0
        assert cloud.cost_per_1k_input_tokens == 0.01


# ---------------------------------------------------------------------------
# 10. All built-in connectors' models work with the Router
# ---------------------------------------------------------------------------


class TestModelsWorkWithRouter:
    @pytest.mark.asyncio
    async def test_anthropic_models_route(self) -> None:
        connector = AnthropicConnector(api_key="test")
        router = Router()
        for model in connector.get_available_models():
            router.register_model(model)

        task = TaskSpec(content="Test", complexity_hint="simple")
        decision = await router.route(task)
        assert decision.chosen_provider == "anthropic"

    @pytest.mark.asyncio
    async def test_openai_models_route(self) -> None:
        connector = OpenAIConnector(api_key="test")
        router = Router()
        for model in connector.get_available_models():
            router.register_model(model)

        task = TaskSpec(content="Test", complexity_hint="complex")
        decision = await router.route(task)
        assert decision.chosen_provider == "openai"

    @pytest.mark.asyncio
    async def test_openrouter_models_route(self) -> None:
        connector = OpenRouterConnector(api_key="test")
        router = Router()
        for model in connector.get_available_models():
            router.register_model(model)

        task = TaskSpec(content="Test", complexity_hint="moderate")
        decision = await router.route(task)
        assert decision.chosen_provider == "openrouter"

    @pytest.mark.asyncio
    async def test_generic_local_models_route(self) -> None:
        connector = GenericOpenAIConnector(
            base_url="http://localhost:11434/v1",
            name="ollama",
            models=[
                ProviderModel(
                    provider="ollama",
                    model="llama3",
                    cost_per_1k_input_tokens=0.0,
                    cost_per_1k_output_tokens=0.0,
                    avg_latency_ms=100,
                    capabilities=["reasoning"],
                    tier="fast",
                ),
            ],
        )
        router = Router()
        for model in connector.get_available_models():
            router.register_model(model)

        task = TaskSpec(content="Test", complexity_hint="simple")
        decision = await router.route(task)
        assert decision.chosen_provider == "ollama"
        assert decision.estimated_cost_usd == 0.0

    @pytest.mark.asyncio
    async def test_all_providers_models_route_together(self) -> None:
        router = Router()

        for ConnectorClass in [AnthropicConnector, OpenAIConnector, OpenRouterConnector]:
            connector = ConnectorClass(api_key="test")
            for model in connector.get_available_models():
                router.register_model(model)

        assert len(router.models) == 9

        task = TaskSpec(content="Quick test", complexity_hint="simple")
        decision = await router.route(task)
        assert decision.chosen_model is not None

    def test_all_models_have_valid_fields(self) -> None:
        connectors: list[Any] = [
            AnthropicConnector(api_key="test"),
            OpenAIConnector(api_key="test"),
            OpenRouterConnector(api_key="test"),
            GenericOpenAIConnector(
                base_url="http://localhost:11434/v1",
                name="ollama",
                models=[
                    ProviderModel(
                        provider="ollama",
                        model="llama3",
                        cost_per_1k_input_tokens=0.0,
                        cost_per_1k_output_tokens=0.0,
                        avg_latency_ms=100,
                        capabilities=["reasoning"],
                        tier="fast",
                    ),
                ],
            ),
        ]
        for connector in connectors:
            for model in connector.get_available_models():
                assert model.provider, f"Missing provider on {model.model}"
                assert model.model, f"Missing model name"
                assert model.cost_per_1k_input_tokens >= 0
                assert model.cost_per_1k_output_tokens >= 0
                assert model.avg_latency_ms > 0
                assert model.tier in ("fast", "balanced", "powerful")
                assert len(model.capabilities) > 0
                assert model.max_context_tokens > 0


class TestProviderRegistryGetProvider:
    def test_get_registered_provider(self) -> None:
        registry = ProviderRegistry()
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=False):
            registry.auto_discover()
        provider = registry.get_provider("anthropic")
        assert provider.provider_name == "anthropic"

    def test_get_unregistered_provider_raises(self) -> None:
        registry = ProviderRegistry()
        with pytest.raises(KeyError, match="not registered"):
            registry.get_provider("nonexistent")
