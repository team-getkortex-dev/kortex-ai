"""SQLite-backed state store for local development and single-machine production.

Uses aiosqlite for non-blocking operations. Auto-creates the schema on connect.
Optimized with WAL journal mode and batch insert support.
"""

from __future__ import annotations

import json
from types import TracebackType

import aiosqlite
import structlog

from kortex.core.exceptions import CheckpointNotFoundError
from kortex.core.types import HandoffContext

logger = structlog.get_logger(component="store.sqlite")

_MAX_CHAIN_DEPTH = 100

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    handoff_id TEXT NOT NULL,
    source_agent TEXT NOT NULL,
    target_agent TEXT NOT NULL,
    state_snapshot JSON NOT NULL,
    compressed_summary TEXT,
    parent_checkpoint_id TEXT,
    task_id TEXT,
    created_at TEXT NOT NULL
)
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_task_id ON checkpoints (task_id)",
    "CREATE INDEX IF NOT EXISTS idx_source_agent ON checkpoints (source_agent)",
    "CREATE INDEX IF NOT EXISTS idx_target_agent ON checkpoints (target_agent)",
    "CREATE INDEX IF NOT EXISTS idx_created_at ON checkpoints (created_at)",
]

_INSERT_SQL = """INSERT OR REPLACE INTO checkpoints
    (checkpoint_id, handoff_id, source_agent, target_agent,
     state_snapshot, compressed_summary, parent_checkpoint_id,
     task_id, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"""


class SQLiteStateStore:
    """State store backed by SQLite via aiosqlite.

    Optimizations:
    - WAL journal mode for concurrent read/write
    - synchronous=NORMAL (safe with WAL, much faster than FULL)
    - executemany for batch inserts
    - Connection kept open for the store lifetime

    Args:
        db_path: Path to the SQLite database file. Use ":memory:" for testing.
    """

    def __init__(self, db_path: str = "kortex.db") -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        """Open the database, enable WAL mode, and create the schema."""
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row

        # Performance: WAL mode allows concurrent readers during writes
        await self._db.execute("PRAGMA journal_mode=WAL")
        # Performance: NORMAL sync is safe with WAL and much faster
        await self._db.execute("PRAGMA synchronous=NORMAL")

        await self._db.execute(_CREATE_TABLE)
        for idx_sql in _CREATE_INDEXES:
            await self._db.execute(idx_sql)
        await self._db.commit()
        logger.info("sqlite_connected", db_path=self._db_path, journal_mode="WAL")

    async def disconnect(self) -> None:
        """Close the database connection."""
        if self._db is not None:
            await self._db.close()
            self._db = None
            logger.info("sqlite_disconnected")

    async def __aenter__(self) -> SQLiteStateStore:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.disconnect()

    def _ensure_connected(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("SQLiteStateStore is not connected. Call connect() first.")
        return self._db

    # -- Serialization helpers ----------------------------------------------

    @staticmethod
    def _to_row(ctx: HandoffContext) -> tuple[str, ...]:
        return (
            ctx.checkpoint_id,
            ctx.handoff_id,
            ctx.source_agent,
            ctx.target_agent,
            json.dumps(ctx.state_snapshot),
            ctx.compressed_summary,
            ctx.parent_checkpoint_id,
            ctx.state_snapshot.get("task_id"),
            ctx.created_at.isoformat(),
        )

    @staticmethod
    def _from_row(row: aiosqlite.Row) -> HandoffContext:
        return HandoffContext(
            checkpoint_id=row["checkpoint_id"],
            handoff_id=row["handoff_id"],
            source_agent=row["source_agent"],
            target_agent=row["target_agent"],
            state_snapshot=json.loads(row["state_snapshot"]),
            compressed_summary=row["compressed_summary"],
            parent_checkpoint_id=row["parent_checkpoint_id"],
            created_at=row["created_at"],
        )

    # -- StateStore protocol ------------------------------------------------

    async def save_checkpoint(self, context: HandoffContext) -> str:
        """Persist a handoff context and return its checkpoint_id."""
        db = self._ensure_connected()
        await db.execute(_INSERT_SQL, self._to_row(context))
        await db.commit()
        return context.checkpoint_id

    async def save_checkpoints_batch(
        self, contexts: list[HandoffContext]
    ) -> list[str]:
        """Persist multiple checkpoints in a single transaction using executemany."""
        db = self._ensure_connected()
        rows = [self._to_row(ctx) for ctx in contexts]
        await db.executemany(_INSERT_SQL, rows)
        await db.commit()
        return [ctx.checkpoint_id for ctx in contexts]

    async def get_checkpoint(self, checkpoint_id: str) -> HandoffContext:
        """Retrieve a checkpoint by ID.

        Raises:
            CheckpointNotFoundError: If the checkpoint does not exist.
        """
        db = self._ensure_connected()
        async with db.execute(
            "SELECT * FROM checkpoints WHERE checkpoint_id = ?",
            (checkpoint_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise CheckpointNotFoundError(f"Checkpoint '{checkpoint_id}' not found")
        return self._from_row(row)

    async def list_checkpoints(
        self,
        task_id: str | None = None,
        agent_id: str | None = None,
    ) -> list[HandoffContext]:
        """List checkpoints, optionally filtered by task_id or agent_id."""
        db = self._ensure_connected()

        conditions: list[str] = []
        params: list[str] = []

        if task_id is not None:
            conditions.append("task_id = ?")
            params.append(task_id)
        if agent_id is not None:
            conditions.append("(source_agent = ? OR target_agent = ?)")
            params.extend([agent_id, agent_id])

        query = "SELECT * FROM checkpoints"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at"

        async with db.execute(query, params) as cursor:
            rows = await cursor.fetchall()

        return [self._from_row(row) for row in rows]

    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint. Returns True if it existed, False otherwise."""
        db = self._ensure_connected()
        cursor = await db.execute(
            "DELETE FROM checkpoints WHERE checkpoint_id = ?",
            (checkpoint_id,),
        )
        await db.commit()
        return cursor.rowcount > 0

    async def get_checkpoint_chain(self, checkpoint_id: str) -> list[HandoffContext]:
        """Walk parent_checkpoint_id links to build the full history.

        Returns a list ordered from the root (oldest) to the given checkpoint.

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
