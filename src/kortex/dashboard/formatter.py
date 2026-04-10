"""Formatting helpers for the Kortex CLI.

Provides ASCII table rendering, cost/duration formatting, and ANSI color
support with NO_COLOR/non-TTY fallback.
"""

from __future__ import annotations

import os
import sys


def _colors_enabled() -> bool:
    """Return True if ANSI colors should be used."""
    if os.environ.get("NO_COLOR", ""):
        return False
    if not hasattr(sys.stdout, "isatty"):
        return False
    return sys.stdout.isatty()


_ANSI_CODES: dict[str, str] = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
}


def colorize(text: str, color: str) -> str:
    """Wrap text in ANSI color codes. Respects NO_COLOR env var.

    Args:
        text: The text to colorize.
        color: Color name (red, green, yellow, blue, magenta, cyan, white,
            bold, dim).

    Returns:
        Colored text if colors are enabled, plain text otherwise.
    """
    if not _colors_enabled():
        return text
    code = _ANSI_CODES.get(color, "")
    reset = _ANSI_CODES["reset"]
    if not code:
        return text
    return f"{code}{text}{reset}"


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render an aligned ASCII table.

    Args:
        headers: Column header strings.
        rows: List of rows, each a list of cell strings.

    Returns:
        A formatted table string with aligned columns.
    """
    if not headers:
        return ""

    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))
            else:
                widths.append(len(cell))

    def _format_row(cells: list[str]) -> str:
        parts = []
        for i, cell in enumerate(cells):
            w = widths[i] if i < len(widths) else len(cell)
            parts.append(cell.ljust(w))
        return "  ".join(parts)

    lines = [_format_row(headers)]
    lines.append("  ".join("-" * w for w in widths))
    for row in rows:
        lines.append(_format_row(row))

    return "\n".join(lines)


def format_cost(usd: float) -> str:
    """Format a USD cost value with appropriate precision.

    Args:
        usd: Cost in USD.

    Returns:
        Formatted cost string like "$0.0025" or "FREE".
    """
    if usd == 0.0:
        return "FREE"
    if usd < 0.0001:
        return f"${usd:.6f}"
    if usd < 0.01:
        return f"${usd:.4f}"
    return f"${usd:.2f}"


def format_duration(ms: float) -> str:
    """Format a duration in milliseconds to a human-readable string.

    Args:
        ms: Duration in milliseconds.

    Returns:
        Formatted string like "45ms" or "1.2s".
    """
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms / 1000:.1f}s"
