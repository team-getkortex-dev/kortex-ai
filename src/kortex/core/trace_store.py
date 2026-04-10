"""Trace persistence for Kortex replay architecture.

Provides pluggable storage backends for ``TaskTrace`` objects.
In-memory for testing, SQLite for durable single-machine persistence.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Protocol

import structlog

from kortex.core.trace import TaskTrace

logger = structlog.get_logger(component="trace_store")


class TraceStore(Protocol):
    """Protocol for pluggable trace storage backends."""

    async def save_trace(self, trace: TaskTrace) -> str:
        """Persist a trace. Returns the trace_id."""
        ...

    async def get_trace(self, trace_id: str) -> TaskTrace:
        """Retrieve a trace by ID.

        Raises:
            KeyError: If the trace does not exist.
        """
        ...

    async def list_traces(
        self,
        limit: int = 50,
        task_id: str | None = None,
    ) -> list[TaskTrace]:
        """List traces, most recent first.

        Args:
            limit: Maximum number of traces to return.
            task_id: If set, filter to traces for this task.
        """
        ...

    async def delete_trace(self, trace_id: str) -> bool:
        """Delete a trace. Returns True if deleted, False if not found."""
        ...


class InMemoryTraceStore:
    """Dict-backed trace store for testing and development."""

    def __init__(self) -> None:
        self._traces: dict[str, TaskTrace] = {}

    async def save_trace(self, trace: TaskTrace) -> str:
        """Persist a trace in memory."""
        self._traces[trace.trace_id] = trace
        return trace.trace_id

    async def get_trace(self, trace_id: str) -> TaskTrace:
        """Retrieve a trace by ID."""
        if trace_id not in self._traces:
            raise KeyError(f"Trace '{trace_id}' not found")
        return self._traces[trace_id]

    async def list_traces(
        self,
        limit: int = 50,
        task_id: str | None = None,
    ) -> list[TaskTrace]:
        """List traces, most recent first."""
        traces = list(self._traces.values())
        if task_id is not None:
            traces = [t for t in traces if t.task_id == task_id]
        # Sort by created_at descending
        traces.sort(key=lambda t: t.created_at, reverse=True)
        return traces[:limit]

    async def delete_trace(self, trace_id: str) -> bool:
        """Delete a trace."""
        if trace_id in self._traces:
            del self._traces[trace_id]
            return True
        return False


class SQLiteTraceStore:
    """SQLite-backed trace store for durable single-machine persistence.

    Args:
        db_path: Path to the SQLite database file. Use ``:memory:`` for
            in-memory testing.
    """

    def __init__(self, db_path: str = "kortex_traces.db") -> None:
        self._db_path = db_path
        self._conn: Any = None

    async def connect(self) -> None:
        """Open the database and create tables if needed."""
        import aiosqlite

        self._conn = await aiosqlite.connect(self._db_path)
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS traces (
                trace_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                data JSON NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        await self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_traces_task_id ON traces(task_id)"
        )
        await self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_traces_created_at ON traces(created_at)"
        )
        await self._conn.commit()

    async def disconnect(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def __aenter__(self) -> SQLiteTraceStore:
        await self.connect()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.disconnect()

    async def save_trace(self, trace: TaskTrace) -> str:
        """Persist a trace to SQLite."""
        assert self._conn is not None, "Call connect() first"
        data_json = trace.to_json()
        await self._conn.execute(
            "INSERT OR REPLACE INTO traces (trace_id, task_id, data, created_at) "
            "VALUES (?, ?, ?, ?)",
            (trace.trace_id, trace.task_id, data_json, trace.created_at),
        )
        await self._conn.commit()
        return trace.trace_id

    async def get_trace(self, trace_id: str) -> TaskTrace:
        """Retrieve a trace by ID."""
        assert self._conn is not None, "Call connect() first"
        cursor = await self._conn.execute(
            "SELECT data FROM traces WHERE trace_id = ?", (trace_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            raise KeyError(f"Trace '{trace_id}' not found")
        return TaskTrace.from_json(row[0])

    async def list_traces(
        self,
        limit: int = 50,
        task_id: str | None = None,
    ) -> list[TaskTrace]:
        """List traces, most recent first."""
        assert self._conn is not None, "Call connect() first"
        if task_id is not None:
            cursor = await self._conn.execute(
                "SELECT data FROM traces WHERE task_id = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (task_id, limit),
            )
        else:
            cursor = await self._conn.execute(
                "SELECT data FROM traces ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
        rows = await cursor.fetchall()
        return [TaskTrace.from_json(row[0]) for row in rows]

    async def delete_trace(self, trace_id: str) -> bool:
        """Delete a trace."""
        assert self._conn is not None, "Call connect() first"
        cursor = await self._conn.execute(
            "DELETE FROM traces WHERE trace_id = ?", (trace_id,)
        )
        await self._conn.commit()
        return cursor.rowcount > 0
