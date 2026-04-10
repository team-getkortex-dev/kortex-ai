# Kortex Stress Test Suite

A hostile, independent test suite designed to break Kortex and find every weakness.

## Running

```bash
# Full stress suite (all tests, verbose output)
pytest tests/stress/ -v -s

# Just the summary report
pytest tests/stress/test_report.py -v -s

# Stop on first failure
pytest tests/stress/ -v -s -x

# Run with timing info
pytest tests/stress/ -v -s --durations=0
```

## Test Categories

### Concurrency Stress
Finds race conditions, deadlocks, and state corruption under parallel load.

| Test | What it does | What it finds |
|------|-------------|---------------|
| 50 concurrent coordinations | 50 full pipelines via asyncio.gather | Checkpoint ID collisions, state corruption |
| 200 concurrent coordinations | 200 simultaneous pipelines | Race conditions that only appear at scale |
| Mixed concurrent operations | Routes + handoffs + lookups simultaneously | Resource contention between operation types |
| Concurrent SQLite writes | 50 parallel writes to SQLite | SQLite locking issues |

### Volume Stress
Pushes data volume and pipeline depth to find scaling limits.

| Test | What it does | What it finds |
|------|-------------|---------------|
| 20-agent pipeline | Single pipeline with 20 sequential agents | Handoff chain integrity at depth |
| 50-agent pipeline | Single pipeline with 50 agents | Performance degradation curve |
| 1MB state payload | Handoff with 1MB of data | Serialization bottlenecks |
| 10MB state payload | Handoff with 10MB of data | Memory pressure, OOM risk |
| 1000 rapid-fire routes | Route 1000 tasks sequentially | Throughput ceiling |
| 5000 rapid-fire routes | Sustained routing throughput | Performance over time |
| 100-deep checkpoint chain | 100 linked checkpoints | Chain traversal performance |
| 10K checkpoints (memory) | Store 10,000 checkpoints | Insert/lookup rate at volume |
| 10K checkpoints (SQLite) | Same with SQLite backend | Compare backends at scale |

### Failure Resilience
Breaks providers and verifies the system recovers.

| Test | What it does | What it finds |
|------|-------------|---------------|
| All providers dead | 100% failure rate | Deadlocks, hung tasks, missing error details |
| 30% failure rate (100 runs) | Intermittent failures | Fallback logic, graceful degradation |
| 60% failure rate (50 runs) | Hostile environment | Crash resistance under extreme conditions |
| Slow provider (5s latency) | Very slow responses | Timeout handling, blocking |
| State store corruption | Manually corrupt stored data | Recovery from inconsistent state |

### Edge Cases
Tests the things nobody thinks to test.

| Test | What it does | What it finds |
|------|-------------|---------------|
| Empty pipeline | Zero agents | Null/empty handling |
| Single agent | Minimal pipeline | Boundary behavior |
| Duplicate agents | Same agent 3x in pipeline | ID collision handling |
| Unregistered agent | Non-existent agent in pipeline | Missing reference handling |
| Unicode content | Emoji, CJK, RTL, ZWJ, combining marks | Encoding issues |
| 100K character task | Massive input text | Truncation, memory, serialization |
| Zero-cost models | All models free | Routing logic without cost signals |
| Zero cost ceiling | $0.00 budget | Filter behavior at boundaries |
| Special chars in agent IDs | Spaces, slashes, emoji, empty string | ID validation gaps |

### Memory & Performance
Detects leaks and degradation over time.

| Test | What it does | What it finds |
|------|-------------|---------------|
| 1000 coordinations memory growth | Track memory via tracemalloc | Memory leaks |
| Routing consistency | Compare first 100 vs last 100 of 5000 | Performance degradation |
| Lookup scaling | Compare lookups at 1K vs 10K entries | O(n) vs O(1) behavior |

## Interpreting Results

The report shows:
- **PASS/FAIL** for each test
- **Elapsed time** in seconds
- **Detail** with specific metrics

Key thresholds:
- Routing throughput should be >100 routes/sec
- Memory growth should be <100MB per 1000 coordinations  
- Lookup time should not degrade >3x from 1K to 10K entries
- Routing speed should not degrade >2x over 5000 decisions
- Zero crashes expected at any failure rate
