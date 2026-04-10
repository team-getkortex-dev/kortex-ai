"""Tests for KortexConfig and the get_config / reset_config singleton."""

from __future__ import annotations

import os

import pytest

from kortex.config import KortexConfig, get_config, reset_config


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Ensure config singleton is clean before and after each test."""
    reset_config()
    yield
    reset_config()


class TestDefaults:
    def test_log_level_default(self) -> None:
        cfg = KortexConfig()
        assert cfg.log_level == "INFO"

    def test_default_backend(self) -> None:
        cfg = KortexConfig()
        assert cfg.default_backend == "memory"

    def test_trace_enabled_default(self) -> None:
        cfg = KortexConfig()
        assert cfg.trace_enabled is True

    def test_trace_store_default_empty(self) -> None:
        cfg = KortexConfig()
        assert cfg.trace_store == ""

    def test_trace_db_default(self) -> None:
        cfg = KortexConfig()
        assert cfg.trace_db == "kortex_traces.db"

    def test_default_timeout_ms(self) -> None:
        cfg = KortexConfig()
        assert cfg.default_timeout_ms == 30_000

    def test_max_retries_default(self) -> None:
        cfg = KortexConfig()
        assert cfg.max_retries == 2

    def test_circuit_breaker_threshold_default(self) -> None:
        cfg = KortexConfig()
        assert cfg.circuit_breaker_threshold == 5

    def test_circuit_breaker_recovery_s_default(self) -> None:
        cfg = KortexConfig()
        assert cfg.circuit_breaker_recovery_s == 30.0


class TestEnvVarOverride:
    def test_log_level_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KORTEX_LOG_LEVEL", "DEBUG")
        cfg = KortexConfig()
        assert cfg.log_level == "DEBUG"

    def test_backend_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KORTEX_DEFAULT_BACKEND", "sqlite")
        cfg = KortexConfig()
        assert cfg.default_backend == "sqlite"

    def test_trace_enabled_false_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KORTEX_TRACE_ENABLED", "false")
        cfg = KortexConfig()
        assert cfg.trace_enabled is False

    def test_trace_store_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KORTEX_TRACE_STORE", "sqlite")
        cfg = KortexConfig()
        assert cfg.trace_store == "sqlite"

    def test_trace_db_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KORTEX_TRACE_DB", "/tmp/custom.db")
        cfg = KortexConfig()
        assert cfg.trace_db == "/tmp/custom.db"

    def test_max_retries_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KORTEX_MAX_RETRIES", "5")
        cfg = KortexConfig()
        assert cfg.max_retries == 5

    def test_invalid_log_level_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KORTEX_LOG_LEVEL", "TRACE")
        with pytest.raises(Exception):
            KortexConfig()

    def test_invalid_backend_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KORTEX_DEFAULT_BACKEND", "postgres")
        with pytest.raises(Exception):
            KortexConfig()


class TestSingleton:
    def test_get_config_returns_same_instance(self) -> None:
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_reset_clears_singleton(self) -> None:
        cfg1 = get_config()
        reset_config()
        cfg2 = get_config()
        assert cfg1 is not cfg2

    def test_get_config_returns_kortex_config(self) -> None:
        cfg = get_config()
        assert isinstance(cfg, KortexConfig)

    def test_reset_then_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # First call with default
        cfg1 = get_config()
        assert cfg1.log_level == "INFO"
        # Reset and change env
        reset_config()
        monkeypatch.setenv("KORTEX_LOG_LEVEL", "WARNING")
        cfg2 = get_config()
        assert cfg2.log_level == "WARNING"


class TestRuntimeIntegration:
    """KortexRuntime accepts a config parameter."""

    @pytest.mark.asyncio
    async def test_runtime_accepts_config(self) -> None:
        from kortex.core.router import Router
        from kortex.core.runtime import KortexRuntime
        from kortex.core.state import StateManager

        cfg = KortexConfig()
        runtime = KortexRuntime(
            router=Router(),
            state_manager=StateManager.create("memory"),
            config=cfg,
        )
        assert runtime._config is cfg

    @pytest.mark.asyncio
    async def test_runtime_works_without_config(self) -> None:
        from kortex.core.router import Router
        from kortex.core.runtime import KortexRuntime
        from kortex.core.state import StateManager

        runtime = KortexRuntime(
            router=Router(),
            state_manager=StateManager.create("memory"),
        )
        assert runtime._config is None
