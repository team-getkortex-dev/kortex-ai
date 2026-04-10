# Kortex Scripts

Utility scripts for operating and benchmarking Kortex.

---

## kortex_system_load_test.py

A system-level load and stress harness that exercises the Kortex runtime under
simulated concurrent user traffic. Does **not** call live LLM APIs — all provider
traffic is either skipped (dry-run) or handled by in-process mock providers (mock).

### Modes

| Mode | Description |
|------|-------------|
| `dry-run` | Exercises routing, state, and event paths only (`execute=False`). No provider calls. |
| `mock` | Exercises the full runtime including provider execution via in-process mock providers. |

### Backends

| Backend | Description |
|---------|-------------|
| `memory` | In-memory store (default, no setup required). |
| `sqlite` | SQLite durable checkpoint store (local dev). |
| `redis` | Redis hot state store (requires a running Redis). |

### Quick start

```bash
# Dry-run, 60 seconds, 5–50 users
python scripts/kortex_system_load_test.py \
    --mode dry-run \
    --run-seconds 60 \
    --min-users 5 \
    --max-users 50 \
    --initial-users 10

# Mock execution, 2 minutes, higher concurrency, JSON output
python scripts/kortex_system_load_test.py \
    --mode mock \
    --run-seconds 120 \
    --min-users 10 \
    --max-users 150 \
    --initial-users 20 \
    --json-out load_test_report.json
```

### All options

```
--mode               dry-run | mock  (default: dry-run)
--backend            memory | sqlite | redis  (default: memory)
--run-seconds        Duration in seconds  (default: 60)
--initial-users      Starting user count  (default: 10)
--min-users          Minimum concurrent users  (default: 5)
--max-users          Maximum concurrent users  (default: 100)
--adjust-every-seconds  Interval between concurrency adjustments  (default: 5)
--max-normal-step    Max drift per tick  (default: 5)
--max-spike-size     Max spike magnitude  (default: 35)
--max-dip-size       Max dip magnitude  (default: 25)
--spike-probability  Probability of a spike tick  (default: 0.20)
--dip-probability    Probability of a dip tick  (default: 0.18)
--min-think-seconds  Min user think time between requests  (default: 0.05)
--max-think-seconds  Max user think time between requests  (default: 0.35)
--request-timeout-seconds  Per-request timeout  (default: 15)
--seed               Random seed for reproducibility  (default: 42)
--disable-detector   Disable the anomaly detector  (flag)
--sqlite-path        Path for SQLite database  (default: kortex_load_test.db)
--redis-url          Redis connection URL  (default: redis://localhost:6379)
--redis-key-prefix   Redis key namespace  (default: kortex-load-test:)
--json-out           Path to write JSON report  (optional)
```

### Output

The script prints a summary table to stdout:

```
========================================================================================
KORTEX LOAD TEST SUMMARY
========================================================================================
Mode:                 dry-run
Backend:              memory
Total requests:       <N>
Success rate:         <X>%
Throughput (RPS):     <Y>
Peak active users:    <Z>
...
```

When `--json-out` is specified, the full report (including per-pipeline breakdowns,
latency percentiles, and concurrency controller history) is written as JSON.

---

## setup_free_providers.py

Registers all available free or near-free LLM providers into a `KortexRuntime`.
Checks for API keys via environment variables; providers with no key are silently
skipped.

### Supported providers

| Provider | Key | Base URL | Notes |
|----------|-----|----------|-------|
| Groq | `GROQ_API_KEY` | `https://api.groq.com/openai/v1` | Truly free tier, ultra-fast |
| Cerebras | `CEREBRAS_API_KEY` | `https://api.cerebras.ai/v1` | World's fastest inference |
| Together AI | `TOGETHER_API_KEY` | `https://api.together.xyz/v1` | Broad model catalogue |
| OpenRouter | `OPENROUTER_API_KEY` | `https://openrouter.ai/api/v1` | Free `:free` model variants |

### Getting API keys

- **Groq** — https://console.groq.com (free, no credit card)
- **Cerebras** — https://cloud.cerebras.ai (free tier)
- **Together AI** — https://api.together.xyz (free credits on signup)
- **OpenRouter** — https://openrouter.ai/keys (free `:free` models, no charge)

### Usage

```bash
export GROQ_API_KEY="gsk_..."
export CEREBRAS_API_KEY="csk_..."
export TOGETHER_API_KEY="..."
export OPENROUTER_API_KEY="sk-or-..."

python scripts/setup_free_providers.py
```

Expected output:
```
Registered 4 provider(s): groq, cerebras, together, openrouter
Total models available: 6

Model catalogue:
  cerebras/llama3.1-8b  latency=25ms  cost=free  tier=fast
  groq/llama-3.1-70b-versatile  latency=50ms  cost=free  tier=powerful
  ...
```

### Programmatic usage

```python
from scripts.setup_free_providers import setup_free_providers

runtime = await setup_free_providers()
result = await runtime.coordinate(task, pipeline, execute=True)
```

---

## validate_with_real_apis.py

End-to-end validation suite that proves Kortex features work correctly with
live LLM providers.  Requires at least one API key (see above).

### Tests

| # | Name | What it measures | Pass criterion |
|---|------|-----------------|----------------|
| 1 | Cache Speedup | Latency of cold vs cached call | ≥10× speedup |
| 2 | EWMA Convergence | EWMA estimate vs observed avg latency | Drift ≤20% |
| 3 | Batch Throughput | Serial vs `coordinate_batch()` time | No regression (≥0.33×) |
| 4 | Constraint Enforcement | `LatencyConstraint(100ms)` respected | 0 violations |
| 5 | Cost Estimation | Predicted vs actual spend | Error ≤20% |

### Usage

```bash
export GROQ_API_KEY="gsk_..."
python scripts/validate_with_real_apis.py
```

Expected output (with keys set):
```
============================================================
=== TEST 1: Cache Speedup ===
============================================================
  [INFO] Cold call: 450.3ms
  [INFO] Cached call: 0.8ms
  [INFO] Speedup: 562.9x
  [PASS] Cache speedup 562.9x ≥ 10x
...
============================================================
RESULTS: 5/5 tests passed
============================================================
```

Expected output (no keys):
```
No API keys found. Set environment variables to run validation.
Required (at least one): GROQ_API_KEY, CEREBRAS_API_KEY, TOGETHER_API_KEY, OPENROUTER_API_KEY
```

### Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `No API keys found` | Env vars not set | Export the key variables before running |
| `TEST 4 FAIL: no fast models` | Groq/Cerebras keys missing | Set `GROQ_API_KEY` or `CEREBRAS_API_KEY` |
| `TEST 5 SKIP: free-tier models` | Only free-tier providers registered | Add a Together AI key for cost testing |
| `RoutingFailedError` | No models match task constraints | Check that registered models have the required capabilities |
| `httpx.ConnectError` | Network issue or wrong base_url | Verify the provider's status page |
