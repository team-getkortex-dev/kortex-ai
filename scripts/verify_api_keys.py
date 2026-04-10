"""Verify that all expected LLM provider API keys are configured.

Loads from a ``.env`` file in the repository root (if present) via
``python-dotenv``, then checks for each of the 4 expected keys and
prints a masked preview of each one that is set.

Usage::

    python scripts/verify_api_keys.py
"""

from __future__ import annotations

import os
import sys

# Allow running from scripts/ or repo root
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from dotenv import load_dotenv  # type: ignore[import-untyped]

# Load .env from repo root (silently a no-op if the file doesn't exist)
load_dotenv(os.path.join(_REPO_ROOT, ".env"))


# ---------------------------------------------------------------------------
# Key definitions
# ---------------------------------------------------------------------------

_PROVIDERS: list[dict[str, str]] = [
    {
        "env_var": "GROQ_API_KEY",
        "name": "Groq",
        "signup": "https://console.groq.com",
    },
    {
        "env_var": "CEREBRAS_API_KEY",
        "name": "Cerebras",
        "signup": "https://cloud.cerebras.ai",
    },
    {
        "env_var": "OPENROUTER_API_KEY",
        "name": "OpenRouter",
        "signup": "https://openrouter.ai/keys",
    },
    {
        "env_var": "HF_TOKEN",
        "name": "Hugging Face",
        "signup": "https://huggingface.co/settings/tokens",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mask_key(key: str) -> str:
    """Return first 8 + '...' + last 4 characters of an API key."""
    if len(key) <= 12:
        return key[:4] + "..." if len(key) >= 4 else "[too short]"
    return f"{key[:8]}...{key[-4:]}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print("Kortex — API Key Verification")
    print("=" * 40)

    configured = 0
    for p in _PROVIDERS:
        value = os.getenv(p["env_var"])
        if value:
            masked = _mask_key(value)
            print(f"  [OK]     {p['name']:<14} {p['env_var']} = {masked}")
            configured += 1
        else:
            print(f"  [MISSING] {p['name']:<13} {p['env_var']} not set  "
                  f"(get key: {p['signup']})")

    print()
    print(f"Result: {configured}/4 providers configured")

    if configured == 0:
        print()
        print("No keys found. Copy .env.example to .env and fill in your keys:")
        print("  cp .env.example .env")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
