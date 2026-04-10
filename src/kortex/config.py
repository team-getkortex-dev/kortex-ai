"""Kortex configuration management.

All settings are readable from environment variables with the ``KORTEX_``
prefix. Environment variables take precedence over defaults.

Example::

    export KORTEX_LOG_LEVEL=DEBUG
    export KORTEX_DEFAULT_BACKEND=sqlite
    export KORTEX_TRACE_ENABLED=true
    export KORTEX_TRACE_STORE=sqlite
    export KORTEX_TRACE_DB=./traces.db

In code::

    from kortex import get_config

    cfg = get_config()
    print(cfg.log_level)        # "DEBUG"
    print(cfg.default_backend)  # "sqlite"
"""

from __future__ import annotations

import os
from typing import Literal

import structlog
from pydantic_settings import BaseSettings, SettingsConfigDict

_log = structlog.get_logger(component="config")


class KortexConfig(BaseSettings):
    """Central configuration for the Kortex runtime.

    Every field is overridable via ``KORTEX_<FIELD_NAME>`` environment
    variables (case-insensitive). Values are validated at load time.

    Args:
        log_level: Logging verbosity. One of DEBUG, INFO, WARNING, ERROR.
        default_backend: Default state store backend for new runtimes.
        trace_enabled: Whether to attach a TaskTrace to every coordination.
        trace_store: Trace persistence backend. Empty string disables persistence.
        trace_db: Path to the SQLite trace database (used when trace_store="sqlite").
        default_timeout_ms: Default provider request timeout in milliseconds.
        max_retries: Default number of provider retries on transient failure.
        circuit_breaker_threshold: Failures before opening the circuit breaker.
        circuit_breaker_recovery_s: Seconds before the circuit breaker tries again.
    """

    model_config = SettingsConfigDict(
        env_prefix="KORTEX_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    default_backend: Literal["memory", "sqlite", "redis"] = "memory"
    trace_enabled: bool = True
    trace_store: Literal["memory", "sqlite", ""] = ""
    trace_db: str = "kortex_traces.db"
    default_timeout_ms: int = 30_000
    max_retries: int = 2
    circuit_breaker_threshold: int = 5
    circuit_breaker_recovery_s: float = 30.0

    # -- semantic cache -------------------------------------------------------
    cache_enabled: bool = True
    cache_backend: Literal["memory", "redis"] = "memory"
    cache_max_size: int = 1000
    cache_ttl_seconds: int | None = None

    # -- routing decision cache -----------------------------------------------
    routing_decision_cache_enabled: bool = True
    routing_decision_cache_size: int = 10_000

    # -- parallel handoffs ----------------------------------------------------
    parallel_handoffs_enabled: bool = True

    # -- cost management ------------------------------------------------------
    cost_warning_threshold: float | None = None

    # -- adaptive EWMA & percentile tracking ----------------------------------
    track_percentiles: bool = True
    percentile_window_size: int = 100

    def model_post_init(self, __context: object) -> None:
        """Warn if API key environment variables are detected."""
        _API_KEY_ENV_VARS = [
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GROQ_API_KEY",
            "CEREBRAS_API_KEY",
            "TOGETHER_API_KEY",
            "OPENROUTER_API_KEY",
        ]
        found = [v for v in _API_KEY_ENV_VARS if os.getenv(v)]
        if found:
            _log.warning(
                "api_keys_detected_in_environment",
                vars=found,
                message=(
                    "API keys detected in config. Ensure .env is in .gitignore."
                ),
            )


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_config: KortexConfig | None = None


def get_config() -> KortexConfig:
    """Return the global KortexConfig singleton, creating it on first call.

    The singleton is constructed once from environment variables and cached.
    Call :func:`reset_config` to force re-creation (useful in tests).

    Returns:
        The active KortexConfig instance.
    """
    global _config
    if _config is None:
        _config = KortexConfig()
    return _config


def reset_config() -> None:
    """Clear the cached config singleton, forcing re-creation on next call.

    Primarily useful in tests that need to simulate different env-var
    configurations without process restarts.
    """
    global _config
    _config = None
