"""CLI and web dashboard for monitoring Kortex routing and handoff activity."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kortex.dashboard.cli import KortexCLI, main

__all__ = [
    "KortexCLI",
    "main",
]


def __getattr__(name: str) -> object:
    """Lazy-import to avoid RuntimeWarning when using ``python -m``."""
    if name in ("KortexCLI", "main"):
        from kortex.dashboard.cli import KortexCLI, main

        _exports = {"KortexCLI": KortexCLI, "main": main}
        return _exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
