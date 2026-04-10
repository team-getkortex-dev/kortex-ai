"""Shared HTTP connection pool for Kortex provider connectors.

Provides a singleton ``ConnectionPool`` that maintains one
``httpx.AsyncClient`` per base URL, enabling connection reuse and HTTP/2
multiplexing across all provider connector instances.

HTTP/2 is enabled automatically when the ``h2`` package is installed.
"""

from __future__ import annotations

import structlog

import httpx

try:
    import h2  # noqa: F401

    _HTTP2_AVAILABLE = True
except ImportError:
    _HTTP2_AVAILABLE = False

logger = structlog.get_logger(component="http_pool")

# Shared connection limits for every pooled client.
_LIMITS = httpx.Limits(
    max_connections=100,
    max_keepalive_connections=20,
    keepalive_expiry=30.0,
)


class ConnectionPool:
    """Singleton HTTP connection pool keyed by base URL.

    All ``GenericOpenAIConnector`` instances sharing the same base URL will
    receive the same underlying ``httpx.AsyncClient``, reusing TCP/TLS
    connections across requests.

    Auth headers are **not** baked into the pooled client; callers must pass
    them on each request so that different API keys can share the same pool.

    Usage::

        pool = ConnectionPool.get_instance()
        client = pool.get_client("https://api.groq.com")
        response = await client.post("/openai/v1/chat/completions", ...)

    """

    _instance: ConnectionPool | None = None

    def __init__(self) -> None:
        self._clients: dict[str, httpx.AsyncClient] = {}

    @classmethod
    def get_instance(cls) -> ConnectionPool:
        """Return the process-wide singleton pool."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Destroy the singleton (for testing only)."""
        cls._instance = None

    def get_client(self, base_url: str, timeout: float = 60.0) -> httpx.AsyncClient:
        """Return an existing or newly-created pooled client for *base_url*.

        The same ``httpx.AsyncClient`` is returned for every call with the
        same *base_url*, reusing the underlying connection pool.

        Args:
            base_url: The API root URL (e.g. ``"https://api.groq.com/openai/v1"``).
            timeout: Request timeout in seconds used when a new client is
                created.  Ignored for subsequent calls with the same URL.

        Returns:
            A shared, open ``httpx.AsyncClient``.
        """
        if base_url not in self._clients or self._clients[base_url].is_closed:
            self._clients[base_url] = httpx.AsyncClient(
                base_url=base_url,
                timeout=timeout,
                http2=_HTTP2_AVAILABLE,
                limits=_LIMITS,
            )
            logger.debug(
                "http_pool_client_created",
                base_url=base_url,
                http2=_HTTP2_AVAILABLE,
            )
        return self._clients[base_url]

    async def close_all(self) -> None:
        """Close every pooled client and clear the registry.

        Called by ``KortexRuntime.stop()`` to ensure all TCP connections are
        released on shutdown.
        """
        for base_url, client in list(self._clients.items()):
            if not client.is_closed:
                await client.aclose()
                logger.debug("http_pool_client_closed", base_url=base_url)
        self._clients.clear()

    @property
    def open_client_count(self) -> int:
        """Number of currently open pooled clients."""
        return sum(1 for c in self._clients.values() if not c.is_closed)
