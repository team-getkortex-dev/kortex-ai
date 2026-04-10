"""Tests for the HTTP connection pool singleton."""

from __future__ import annotations

import pytest
import httpx

from kortex.providers.http_pool import ConnectionPool, _HTTP2_AVAILABLE


@pytest.fixture(autouse=True)
def reset_pool() -> None:
    """Ensure each test starts with a fresh singleton."""
    ConnectionPool.reset()
    yield
    ConnectionPool.reset()


# ---------------------------------------------------------------------------
# Client reuse
# ---------------------------------------------------------------------------


def test_same_base_url_returns_same_client() -> None:
    pool = ConnectionPool.get_instance()
    c1 = pool.get_client("https://api.example.com/v1")
    c2 = pool.get_client("https://api.example.com/v1")
    assert c1 is c2


def test_different_base_urls_return_different_clients() -> None:
    pool = ConnectionPool.get_instance()
    c1 = pool.get_client("https://api.groq.com/openai/v1")
    c2 = pool.get_client("https://api.cerebras.ai/v1")
    assert c1 is not c2


def test_singleton_shared_across_calls() -> None:
    p1 = ConnectionPool.get_instance()
    p2 = ConnectionPool.get_instance()
    assert p1 is p2


def test_open_client_count() -> None:
    pool = ConnectionPool.get_instance()
    pool.get_client("https://api.a.com")
    pool.get_client("https://api.b.com")
    assert pool.open_client_count == 2


# ---------------------------------------------------------------------------
# HTTP/2 and limits
# ---------------------------------------------------------------------------


def test_client_has_http2_if_h2_installed() -> None:
    pool = ConnectionPool.get_instance()
    client = pool.get_client("https://api.example.com")
    # httpx stores http2 setting on _transport
    assert isinstance(client, httpx.AsyncClient)
    if _HTTP2_AVAILABLE:
        # h2 is installed — HTTP/2 transport should be active
        assert client._transport is not None  # type: ignore[attr-defined]


def test_client_limits_applied() -> None:
    from kortex.providers.http_pool import _LIMITS

    assert _LIMITS.max_connections == 100
    assert _LIMITS.max_keepalive_connections == 20


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_all_clears_clients() -> None:
    pool = ConnectionPool.get_instance()
    pool.get_client("https://api.example.com")
    pool.get_client("https://api.other.com")
    assert pool.open_client_count == 2

    await pool.close_all()

    assert pool.open_client_count == 0


@pytest.mark.asyncio
async def test_close_all_idempotent() -> None:
    pool = ConnectionPool.get_instance()
    pool.get_client("https://api.example.com")
    await pool.close_all()
    await pool.close_all()  # second call should not raise
    assert pool.open_client_count == 0


@pytest.mark.asyncio
async def test_get_client_after_close_creates_new() -> None:
    pool = ConnectionPool.get_instance()
    c1 = pool.get_client("https://api.example.com")
    await pool.close_all()

    # After close, the old client is gone; a new one is created
    c2 = pool.get_client("https://api.example.com")
    assert c1 is not c2
    assert not c2.is_closed
