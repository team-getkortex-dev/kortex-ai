# Provider Resilience

Kortex ships a production-hardened HTTP client layer that sits between the runtime and every LLM provider. It adds retry logic, circuit breaking, and request-level timeouts with no changes required to provider connector code.

---

## Components

### RetryPolicy

Controls how the resilient client retries failed requests.

```python
from kortex.providers.resilience import RetryPolicy

policy = RetryPolicy(
    max_retries=3,
    backoff_base_ms=100,       # first retry waits 100 ms
    backoff_multiplier=2.0,    # 100 → 200 → 400 ms
    backoff_max_ms=5000,       # never wait more than 5 s
    retryable_status_codes=(429, 500, 502, 503, 504),
)

# Compute wait time for retry N (0-indexed)
wait_ms = policy.delay_ms(attempt=0)   # 100.0
wait_ms = policy.delay_ms(attempt=1)   # 200.0
wait_ms = policy.delay_ms(attempt=2)   # 400.0
```

Status codes **401** and **403** are never retried — they indicate auth failures that a retry cannot fix.

### CircuitBreaker

Prevents cascading failures by short-circuiting requests to a provider that is repeatedly failing.

```python
from kortex.providers.resilience import CircuitBreaker

cb = CircuitBreaker(
    failure_threshold=5,      # open after 5 consecutive failures
    recovery_timeout_s=30,    # try again after 30 s (HALF_OPEN)
    half_open_max_calls=1,    # allow 1 probe before re-closing
)
```

State machine:

```
CLOSED ──(5 failures)──► OPEN ──(30 s timeout)──► HALF_OPEN
   ▲                                                    │
   └──────────── probe succeeded ─────────────────────┘
                        │
                   probe failed → OPEN
```

When the circuit is **OPEN**, a `CircuitOpenError` is raised immediately without making a network call.

### ResilientClient

Wraps `httpx` with retry + circuit breaker + timeouts. Use it inside custom provider connectors:

```python
from kortex.providers.resilience import RetryPolicy, CircuitBreaker
from kortex.providers.resilient_client import ResilientClient, TimeoutPolicy

client = ResilientClient(
    retry_policy=RetryPolicy(max_retries=2),
    circuit_breaker=CircuitBreaker(failure_threshold=5),
    timeout_policy=TimeoutPolicy(connect_s=5.0, read_s=30.0),
)

response = await client.request(
    "POST",
    "https://api.example.com/v1/chat/completions",
    headers={"Authorization": "Bearer sk-..."},
    json={"model": "gpt-4o-mini", "messages": [...]},
)
```

---

## Using Resilience with GenericOpenAIConnector

Pass a `ResilientClient` to any connector that extends `GenericOpenAIConnector`:

```python
from kortex.providers.base import GenericOpenAIConnector
from kortex.providers.resilience import RetryPolicy, CircuitBreaker
from kortex.providers.resilient_client import ResilientClient

resilient = ResilientClient(
    retry_policy=RetryPolicy(max_retries=3, backoff_base_ms=200),
    circuit_breaker=CircuitBreaker(failure_threshold=10, recovery_timeout_s=60),
)

connector = GenericOpenAIConnector(
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
    name="openai",
    models=[...],
    resilient_client=resilient,
)
```

---

## Exception Taxonomy

| Exception | Trigger | Retried |
|-----------|---------|---------|
| `ProviderTimeoutError` | Request exceeded timeout | Yes |
| `ProviderRateLimitError` | HTTP 429 | Yes (with backoff) |
| `ProviderOverloadError` | HTTP 500/502/503/504 | Yes |
| `ProviderAuthError` | HTTP 401/403 | **No** |
| `CircuitOpenError` | Circuit breaker is OPEN | **No** (fast fail) |

All of these inherit from `ProviderError` → `KortexError`, so a single `except KortexError` catches them all.

---

## Monitoring

The `CircuitBreaker` exposes its current state for health checks:

```python
from kortex.providers.resilience import CircuitBreakerState

if cb.state == CircuitBreakerState.OPEN:
    print("Provider is unavailable — circuit is open")
elif cb.state == CircuitBreakerState.HALF_OPEN:
    print("Provider is recovering — probing with one request")
else:
    print("Provider is healthy")
```
