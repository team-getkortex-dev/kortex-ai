"""Redis-backed state store for production deployments.

Uses redis.asyncio for non-blocking operations. Stores checkpoints as JSON
with sorted-set indexes for fast agent and task lookups.
"""

from __future__ import annotations

import json
from types import TracebackType
from typing import Any

import structlog

from kortex.core.exceptions import CheckpointNotFoundError
from kortex.core.types import HandoffContext

logger = structlog.get_logger(component="store.redis")

_MAX_CHAIN_DEPTH = 100


class RedisStateStore:
    """Production state store backed by Redis.

    Args:
        redis_url: Redis connection URL.
        key_prefix: Prefix for all Kortex keys.
        ttl_seconds: Optional TTL for checkpoint keys. None means no expiry.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        key_prefix: str = "kortex:",
        ttl_seconds: int | None = None,
    ) -> None:
        self._redis_url = redis_url
        self._prefix = key_prefix
        self._ttl = ttl_seconds
        self._redis: Any = None

    # -- Key helpers --------------------------------------------------------

    def _ck_key(self, checkpoint_id: str) -> str:
        return f"{self._prefix}checkpoint:{checkpoint_id}"

    def _agent_key(self, agent_id: str) -> str:
        return f"{self._prefix}agent:{agent_id}:checkpoints"

    def _task_key(self, task_id: str) -> str:
        return f"{self._prefix}task:{task_id}:checkpoints"

    # -- Lifecycle ----------------------------------------------------------

    async def connect(self) -> None:
        """Initialize the Redis connection and verify connectivity."""
        import redis.asyncio as aioredis

        self._redis = aioredis.from_url(self._redis_url, decode_responses=True)
        await self._redis.ping()
        logger.info("redis_connected", url=self._redis_url)

    async def disconnect(self) -> None:
        """Close the Redis connection cleanly."""
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
            logger.info("redis_disconnected")

    async def __aenter__(self) -> RedisStateStore:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.disconnect()

    def _ensure_connected(self) -> Any:
        if self._redis is None:
            raise RuntimeError("RedisStateStore is not connected. Call connect() first.")
        return self._redis

    # -- Serialization ------------------------------------------------------

    @staticmethod
    def _serialize(context: HandoffContext) -> str:
        return context.model_dump_json()

    @staticmethod
    def _deserialize(data: str) -> HandoffContext:
        return HandoffContext.model_validate_json(data)

    # -- StateStore protocol ------------------------------------------------

    async def save_checkpoint(self, context: HandoffContext) -> str:
        """Persist a handoff context and return its checkpoint_id."""
        r = self._ensure_connected()
        key = self._ck_key(context.checkpoint_id)
        payload = self._serialize(context)
        timestamp = context.created_at.timestamp()

        await r.set(key, payload)
        if self._ttl is not None:
            await r.expire(key, self._ttl)

        # Index by agent (both source and target)
        for agent in (context.source_agent, context.target_agent):
            await r.zadd(self._agent_key(agent), {context.checkpoint_id: timestamp})

        # Index by task_id if present
        task_id = context.state_snapshot.get("task_id")
        if task_id:
            await r.zadd(self._task_key(task_id), {context.checkpoint_id: timestamp})

        return context.checkpoint_id

    async def save_checkpoints_batch(
        self, contexts: list[HandoffContext]
    ) -> list[str]:
        """Persist multiple checkpoints using a Redis pipeline (single round-trip)."""
        r = self._ensure_connected()
        pipe = r.pipeline()

        for context in contexts:
            key = self._ck_key(context.checkpoint_id)
            payload = self._serialize(context)
            timestamp = context.created_at.timestamp()

            pipe.set(key, payload)
            if self._ttl is not None:
                pipe.expire(key, self._ttl)

            for agent in (context.source_agent, context.target_agent):
                pipe.zadd(self._agent_key(agent), {context.checkpoint_id: timestamp})

            task_id = context.state_snapshot.get("task_id")
            if task_id:
                pipe.zadd(self._task_key(task_id), {context.checkpoint_id: timestamp})

        await pipe.execute()
        return [ctx.checkpoint_id for ctx in contexts]

    async def get_checkpoint(self, checkpoint_id: str) -> HandoffContext:
        """Retrieve a checkpoint by ID.

        Raises:
            CheckpointNotFoundError: If the checkpoint does not exist.
        """
        r = self._ensure_connected()
        data = await r.get(self._ck_key(checkpoint_id))
        if data is None:
            raise CheckpointNotFoundError(f"Checkpoint '{checkpoint_id}' not found")
        return self._deserialize(data)

    async def list_checkpoints(
        self,
        task_id: str | None = None,
        agent_id: str | None = None,
    ) -> list[HandoffContext]:
        """List checkpoints, optionally filtered by task_id or agent_id."""
        r = self._ensure_connected()

        if agent_id is not None:
            ids = await r.zrangebyscore(self._agent_key(agent_id), "-inf", "+inf")
        elif task_id is not None:
            ids = await r.zrangebyscore(self._task_key(task_id), "-inf", "+inf")
        else:
            logger.warning(
                "list_checkpoints_scan",
                msg="No filter provided; falling back to SCAN which may be slow",
            )
            pattern = f"{self._prefix}checkpoint:*"
            ids = []
            async for key in r.scan_iter(match=pattern):
                cp_id = key.replace(f"{self._prefix}checkpoint:", "")
                ids.append(cp_id)

        results: list[HandoffContext] = []
        for cp_id in ids:
            try:
                ctx = await self.get_checkpoint(cp_id)
                results.append(ctx)
            except CheckpointNotFoundError:
                continue  # expired or deleted between index read and get

        # Apply both filters if both provided
        if task_id is not None and agent_id is not None:
            results = [
                c for c in results
                if c.state_snapshot.get("task_id") == task_id
            ]

        return results

    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint. Returns True if it existed, False otherwise."""
        r = self._ensure_connected()
        key = self._ck_key(checkpoint_id)

        # Try to read it first so we can clean up indexes
        data = await r.get(key)
        if data is None:
            return False

        ctx = self._deserialize(data)
        await r.delete(key)

        # Clean up index sets
        for agent in (ctx.source_agent, ctx.target_agent):
            await r.zrem(self._agent_key(agent), checkpoint_id)

        task_id = ctx.state_snapshot.get("task_id")
        if task_id:
            await r.zrem(self._task_key(task_id), checkpoint_id)

        return True

    async def get_checkpoint_chain(self, checkpoint_id: str) -> list[HandoffContext]:
        """Walk parent_checkpoint_id links to build the full history.

        Returns a list ordered from the root (oldest) to the given checkpoint.
        Uses a max_depth of 100 to prevent infinite loops.

        Raises:
            CheckpointNotFoundError: If any checkpoint in the chain is missing.
        """
        chain: list[HandoffContext] = []
        current_id: str | None = checkpoint_id
        seen: set[str] = set()

        while current_id is not None and len(chain) < _MAX_CHAIN_DEPTH:
            if current_id in seen:
                logger.warning("checkpoint_chain_cycle", checkpoint_id=current_id)
                break
            seen.add(current_id)
            ctx = await self.get_checkpoint(current_id)
            chain.append(ctx)
            current_id = ctx.parent_checkpoint_id

        chain.reverse()
        return chain

    # -- Utility ------------------------------------------------------------

    async def flush(self, prefix_only: bool = True) -> int:
        """Delete Kortex keys. Returns count of deleted keys.

        Args:
            prefix_only: If True, only delete keys matching the configured prefix.
                If False, flushes the entire database (use with caution).
        """
        r = self._ensure_connected()

        if not prefix_only:
            await r.flushdb()
            return -1  # unknown count

        count = 0
        async for key in r.scan_iter(match=f"{self._prefix}*"):
            await r.delete(key)
            count += 1

        return count
