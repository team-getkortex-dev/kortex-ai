"""Tests for API key redaction across all output paths."""

from __future__ import annotations

import pytest

from kortex.security.redaction import redact_api_key, scan_and_redact


# ---------------------------------------------------------------------------
# redact_api_key
# ---------------------------------------------------------------------------


def test_redact_api_key_sk_prefix() -> None:
    key = "sk-abcdefghijklmnopqrstuvwxyz123456"
    result = redact_api_key(key)
    assert result.startswith("sk-")
    assert "..." in result
    assert result.endswith("456")
    # Full key must not appear
    assert key not in result


def test_redact_api_key_gsk_prefix() -> None:
    key = "gsk_abcdefghijklmnopqrstuvwxyz1234567890"
    result = redact_api_key(key)
    assert "..." in result
    assert key not in result


def test_redact_api_key_csk_prefix() -> None:
    key = "csk_abcdefghijklmnopqrstuvwxyz1234567890"
    result = redact_api_key(key)
    assert "..." in result
    assert key not in result


def test_redact_api_key_short_key_fully_redacted() -> None:
    result = redact_api_key("short")
    assert result == "[REDACTED]"


def test_redact_api_key_preserves_first_and_last() -> None:
    key = "sk-AAABBBCCC111222333444555666777888"
    result = redact_api_key(key)
    assert result[:3] == key[:3]
    assert result[-3:] == key[-3:]


def test_redact_api_key_exactly_nine_chars() -> None:
    key = "123456789"
    result = redact_api_key(key)
    assert "..." in result
    assert key not in result


# ---------------------------------------------------------------------------
# scan_and_redact
# ---------------------------------------------------------------------------


def test_scan_and_redact_sk_key_in_text() -> None:
    key = "sk-abcdefghijklmnopqrstuvwxyz1234567890abc"
    text = f"API key is {key} and should be hidden"
    result = scan_and_redact(text)
    assert key not in result
    assert "..." in result


def test_scan_and_redact_gsk_key() -> None:
    key = "gsk_abcdefghijklmnopqrstuvwxyz1234567890"
    text = f"Authorization: {key}"
    result = scan_and_redact(text)
    assert key not in result


def test_scan_and_redact_csk_key() -> None:
    key = "csk_abcdefghijklmnopqrstuvwxyz1234567890"
    text = f"CEREBRAS_API_KEY={key}"
    result = scan_and_redact(text)
    assert key not in result


def test_scan_and_redact_bearer_token() -> None:
    token = "Bearer abcdefghijklmnopqrstuvwxyz1234567890"
    text = f"Authorization: {token}"
    result = scan_and_redact(text)
    # The full bearer value should be redacted
    assert "abcdefghijklmnopqrstuvwxyz1234567890" not in result


def test_scan_and_redact_multiple_keys_in_text() -> None:
    key1 = "sk-key1abcdefghijklmnopqrstuvwxyz1234"
    key2 = "gsk_key2abcdefghijklmnopqrstuvwxyz5678"
    text = f"primary={key1}, secondary={key2}"
    result = scan_and_redact(text)
    assert key1 not in result
    assert key2 not in result


def test_scan_and_redact_no_keys_unchanged() -> None:
    text = "This text has no API keys in it at all."
    result = scan_and_redact(text)
    assert result == text


def test_scan_and_redact_empty_string() -> None:
    assert scan_and_redact("") == ""


def test_scan_and_redact_preserves_surrounding_text() -> None:
    key = "sk-abcdefghijklmnopqrstuvwxyz1234567890"
    text = f"prefix {key} suffix"
    result = scan_and_redact(text)
    assert result.startswith("prefix ")
    assert result.endswith(" suffix")


def test_scan_and_redact_json_blob() -> None:
    key = "sk-secretkeyabcdefghijklmnopqrstuvwxyz"
    text = f'{{"api_key": "{key}", "model": "gpt-4"}}'
    result = scan_and_redact(text)
    assert key not in result
    assert '"model": "gpt-4"' in result


# ---------------------------------------------------------------------------
# TaskTrace redaction
# ---------------------------------------------------------------------------


def test_task_trace_to_dict_redacts_content() -> None:
    from kortex.core.trace import TaskTrace

    key = "sk-tracetestabcdefghijklmnopqrstuvwxyz"
    trace = TaskTrace(
        task_content=f"Process this key: {key}",
        task_id="test-trace",
    )
    d = trace.to_dict()
    assert key not in d["task_content"]
    assert "..." in d["task_content"]


def test_task_trace_to_json_redacts_content() -> None:
    from kortex.core.trace import TaskTrace

    key = "gsk_jsontest1234567890abcdefghijklmnopqrstuvwxyz"
    trace = TaskTrace(task_content=f"Key: {key}")
    json_str = trace.to_json()
    assert key not in json_str


def test_task_trace_roundtrip_preserves_redaction() -> None:
    from kortex.core.trace import TaskTrace

    key = "sk-roundtripabcdefghijklmnopqrstuvwxyz"
    trace = TaskTrace(task_content=f"Key: {key}")
    d = trace.to_dict()
    restored = TaskTrace.from_dict(d)
    # Restored content should not contain the original key
    assert key not in restored.task_content


# ---------------------------------------------------------------------------
# RoutingDiagnostics redaction
# ---------------------------------------------------------------------------


def test_routing_diagnostics_redacts_keys_in_output() -> None:
    from kortex.core.router import ProviderModel
    from kortex.core.types import TaskSpec
    from kortex.router.diagnostics import RoutingDiagnostics

    diag = RoutingDiagnostics()
    key = "sk-diagtestabcdefghijklmnopqrstuvwxyz"
    task = TaskSpec(content=f"Task with key {key}")
    m = ProviderModel(
        provider="openai",
        model="gpt-4o",
        cost_per_1k_input_tokens=0.001,
        cost_per_1k_output_tokens=0.002,
        avg_latency_ms=300,
        tier="balanced",
    )
    failures = {m.identity.key: [f"auth error with Bearer {key}aaaaaaaaaaaaaaaaaaaaa"]}
    msg = diag.explain_failure(task, [m], constraint_failures=failures)
    # The raw key should not appear in the output
    assert key not in msg


# ---------------------------------------------------------------------------
# CLI redaction (unit-level — no full runtime spin-up)
# ---------------------------------------------------------------------------


def test_cli_models_output_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI cmd_models output passes through scan_and_redact."""
    from unittest.mock import MagicMock, patch

    import kortex.dashboard.cli as cli_module

    key = "sk-clitestabcdefghijklmnopqrstuvwxyz1234"

    # Build a fake registry that returns one model with a key in the provider name
    fake_model = MagicMock()
    fake_model.provider = f"openai-{key}"
    fake_model.model = "gpt-4o"
    fake_model.tier = "balanced"
    fake_model.cost_per_1k_input_tokens = 0.001
    fake_model.cost_per_1k_output_tokens = 0.002
    fake_model.avg_latency_ms = 300
    fake_model.capabilities = []
    fake_model.estimated_cost.return_value = 0.0015

    fake_registry = MagicMock()
    fake_registry.get_all_models.return_value = [fake_model]

    cli = cli_module.KortexCLI(
        runtime=MagicMock(),
        registry=fake_registry,
        demo_mode=False,
    )

    output = cli.cmd_models()
    assert key not in output
