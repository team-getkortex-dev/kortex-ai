# Provider Setup

Kortex connects to LLM providers through the `ProviderConnector` protocol. The built-in `GenericOpenAIConnector` works with any OpenAI-compatible API.

## GenericOpenAIConnector

The universal connector for any endpoint that implements the OpenAI chat completions format:

```python
from kortex.core.router import ProviderModel
from kortex.providers.base import GenericOpenAIConnector

connector = GenericOpenAIConnector(
    base_url="https://api.example.com/v1",
    api_key="sk-your-key",         # None for local models
    name="my-provider",
    models=[ProviderModel(...)],
    extra_headers={"X-Custom": "value"},  # optional
    timeout=60.0,                  # seconds
)
```

Key behaviors:

- When `api_key` is `None` or empty, no `Authorization` header is sent
- Cost is calculated from the `ProviderModel` pricing and actual token counts
- Models with zero pricing report `cost_usd=0.0`

## Built-in Connectors

### Anthropic

```python
from kortex.providers.anthropic import AnthropicConnector

connector = AnthropicConnector(api_key="sk-ant-...")
```

Uses the Anthropic `/v1/messages` format (not OpenAI-compatible). Pre-configured with Claude model definitions.

### OpenAI

```python
from kortex.providers.openai import OpenAIConnector

connector = OpenAIConnector(api_key="sk-...")
```

Extends `GenericOpenAIConnector` with OpenAI's base URL and model catalog.

### OpenRouter

```python
from kortex.providers.openrouter import OpenRouterConnector

connector = OpenRouterConnector(api_key="sk-or-...")
```

Extends `GenericOpenAIConnector` with OpenRouter headers (`HTTP-Referer`, `X-Title`).

## Local Models

### Ollama

```python
from kortex.providers.registry import ProviderRegistry

registry = ProviderRegistry()
registry.register_openai_compatible(
    name="ollama",
    base_url="http://localhost:11434/v1",
    api_key=None,  # no auth needed
    models=[ProviderModel(
        provider="ollama", model="llama3",
        cost_per_1k_input_tokens=0, cost_per_1k_output_tokens=0,
        avg_latency_ms=100, capabilities=["reasoning"], tier="fast",
    )],
)
```

### vLLM

```python
registry.register_openai_compatible(
    name="vllm",
    base_url="http://localhost:8000/v1",
    models=[ProviderModel(
        provider="vllm", model="meta-llama/Llama-3-8b",
        cost_per_1k_input_tokens=0, cost_per_1k_output_tokens=0,
        avg_latency_ms=50, capabilities=["reasoning"], tier="fast",
    )],
)
```

### llama.cpp / LM Studio

```python
registry.register_openai_compatible(
    name="lmstudio",
    base_url="http://localhost:1234/v1",
    models=[ProviderModel(
        provider="lmstudio", model="local-model",
        cost_per_1k_input_tokens=0, cost_per_1k_output_tokens=0,
        avg_latency_ms=80, capabilities=["reasoning"], tier="fast",
    )],
)
```

## Custom Endpoints with Authentication

```python
registry.register_openai_compatible(
    name="corporate-llm",
    base_url="https://llm.internal.corp/v1",
    api_key="corp-api-key",
    extra_headers={
        "X-Team": "platform",
        "X-Environment": "production",
    },
    models=[ProviderModel(
        provider="corporate-llm", model="internal-gpt4",
        cost_per_1k_input_tokens=0.002, cost_per_1k_output_tokens=0.008,
        avg_latency_ms=300, capabilities=["reasoning", "code_generation"],
        tier="balanced",
    )],
)
```

## ProviderRegistry

The registry manages all providers and aggregates their models:

```python
from kortex.providers.registry import ProviderRegistry

registry = ProviderRegistry()

# Register manually
registry.register_provider(connector)

# Or register OpenAI-compatible endpoints in one call
registry.register_openai_compatible(name="...", base_url="...", models=[...])

# List all providers
registry.list_providers()  # ["anthropic", "openai", "ollama"]

# Get all models across all providers
registry.get_all_models()  # [ProviderModel(...), ...]
```

## Auto-Discovery

`auto_discover()` checks environment variables and registers providers automatically:

| Env Var | Provider |
|---------|----------|
| `OPENAI_API_KEY` | OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic |
| `OPENROUTER_API_KEY` | OpenRouter |

```python
registry = ProviderRegistry()
registry.auto_discover()
```

Only registers providers whose keys are set and non-empty. Safe to call even if no keys are present.
