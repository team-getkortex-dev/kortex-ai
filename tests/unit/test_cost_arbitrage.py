"""Tests for CostArbitrage."""

from __future__ import annotations

import pytest

from kortex.router.cost_arbitrage import (
    ArbitrageDecision,
    CostArbitrage,
    ModelPrice,
    SavingsReport,
)


# ---------------------------------------------------------------------------
# ModelPrice
# ---------------------------------------------------------------------------


def test_model_price_estimated_cost() -> None:
    price = ModelPrice(
        provider="openai", model="gpt-4o-mini",
        input_per_1k=0.00015, output_per_1k=0.0006,
    )
    cost = price.estimated_cost(input_tokens=1000, output_tokens=500)
    expected = 0.00015 + 0.0003
    assert cost == pytest.approx(expected, abs=1e-7)


def test_model_price_to_dict() -> None:
    price = ModelPrice(
        provider="openai", model="gpt-4o",
        input_per_1k=0.005, output_per_1k=0.015,
    )
    d = price.to_dict()
    assert d["provider"] == "openai"
    assert d["model"] == "gpt-4o"
    assert "estimated_cost_default" in d


# ---------------------------------------------------------------------------
# CostArbitrage — model registration
# ---------------------------------------------------------------------------


def test_register_equivalent_models_basic() -> None:
    arb = CostArbitrage()
    arb.register_equivalent_models("gpt-4o-mini", "claude-haiku")
    equivs = arb.get_equivalent_models("gpt-4o-mini")
    assert "gpt-4o-mini" in equivs
    assert "claude-haiku" in equivs


def test_register_equivalent_models_symmetric() -> None:
    arb = CostArbitrage()
    arb.register_equivalent_models("model-a", "model-b")
    assert "model-a" in arb.get_equivalent_models("model-b")
    assert "model-b" in arb.get_equivalent_models("model-a")


def test_register_equivalent_models_requires_two_or_more() -> None:
    arb = CostArbitrage()
    with pytest.raises(ValueError):
        arb.register_equivalent_models("only-one")


def test_register_equivalent_models_merges_groups() -> None:
    arb = CostArbitrage()
    arb.register_equivalent_models("a", "b")
    arb.register_equivalent_models("b", "c")  # merges with existing
    equivs = arb.get_equivalent_models("a")
    assert "c" in equivs


def test_register_equivalent_models_three_way() -> None:
    arb = CostArbitrage()
    arb.register_equivalent_models("a", "b", "c")
    assert len(arb.get_equivalent_models("a")) == 3


def test_get_equivalent_models_unknown() -> None:
    arb = CostArbitrage()
    assert arb.get_equivalent_models("unknown-model") == []


# ---------------------------------------------------------------------------
# CostArbitrage — pricing
# ---------------------------------------------------------------------------


def test_update_price() -> None:
    arb = CostArbitrage()
    arb.update_price("openai", "gpt-4o-mini", input_per_1k=0.00015, output_per_1k=0.0006)
    price = arb.get_price("openai", "gpt-4o-mini")
    assert price is not None
    assert price.input_per_1k == 0.00015


def test_update_price_overwrite() -> None:
    arb = CostArbitrage()
    arb.update_price("openai", "gpt-4o-mini", input_per_1k=0.00015, output_per_1k=0.0006)
    arb.update_price("openai", "gpt-4o-mini", input_per_1k=0.0001, output_per_1k=0.0004)
    price = arb.get_price("openai", "gpt-4o-mini")
    assert price.input_per_1k == 0.0001


def test_get_price_none_for_unknown() -> None:
    arb = CostArbitrage()
    assert arb.get_price("openai", "nonexistent") is None


def test_list_prices() -> None:
    arb = CostArbitrage()
    arb.update_price("openai", "gpt-4o-mini", 0.00015, 0.0006)
    arb.update_price("anthropic", "claude-haiku", 0.00025, 0.00125)
    prices = arb.list_prices()
    assert len(prices) == 2


# ---------------------------------------------------------------------------
# CostArbitrage — find_cheapest
# ---------------------------------------------------------------------------


def test_find_cheapest_returns_none_no_equivalents() -> None:
    arb = CostArbitrage()
    arb.update_price("openai", "gpt-4o-mini", 0.00015, 0.0006)
    # No equivalents registered
    result = arb.find_cheapest("gpt-4o-mini")
    assert result is None


def test_find_cheapest_returns_cheapest() -> None:
    arb = CostArbitrage()
    arb.register_equivalent_models("gpt-4o-mini", "claude-haiku")
    arb.update_price("openai", "gpt-4o-mini", input_per_1k=0.00015, output_per_1k=0.0006)
    arb.update_price("anthropic", "claude-haiku", input_per_1k=0.00025, output_per_1k=0.00125)

    decision = arb.find_cheapest("gpt-4o-mini")
    assert decision is not None
    # gpt-4o-mini should be cheaper
    assert decision.chosen_model == "gpt-4o-mini"
    assert decision.chosen_provider == "openai"


def test_find_cheapest_switches_to_cheaper_provider() -> None:
    arb = CostArbitrage()
    arb.register_equivalent_models("expensive", "cheap")
    arb.update_price("provider-a", "expensive", input_per_1k=0.010, output_per_1k=0.030)
    arb.update_price("provider-b", "cheap", input_per_1k=0.0001, output_per_1k=0.0002)

    decision = arb.find_cheapest("expensive")
    assert decision is not None
    assert decision.chosen_model == "cheap"
    assert decision.chosen_provider == "provider-b"
    assert decision.savings_usd > 0


def test_find_cheapest_excludes_providers() -> None:
    arb = CostArbitrage()
    arb.register_equivalent_models("gpt-4o-mini", "claude-haiku")
    arb.update_price("openai", "gpt-4o-mini", input_per_1k=0.00015, output_per_1k=0.0006)
    arb.update_price("anthropic", "claude-haiku", input_per_1k=0.00005, output_per_1k=0.0002)

    # Exclude anthropic — should fall back to openai
    decision = arb.find_cheapest("gpt-4o-mini", excluded_providers=["anthropic"])
    assert decision is not None
    assert decision.chosen_provider == "openai"


def test_find_cheapest_records_decision() -> None:
    arb = CostArbitrage()
    arb.register_equivalent_models("a", "b")
    arb.update_price("p1", "a", 0.001, 0.002)
    arb.update_price("p2", "b", 0.0005, 0.001)

    arb.find_cheapest("a")
    report = arb.savings_report()
    assert report.total_decisions == 1


def test_find_cheapest_no_prices_for_equivalents() -> None:
    arb = CostArbitrage()
    arb.register_equivalent_models("model-x", "model-y")
    # No prices registered
    result = arb.find_cheapest("model-x")
    assert result is None


# ---------------------------------------------------------------------------
# CostArbitrage — savings report
# ---------------------------------------------------------------------------


def test_savings_report_initial() -> None:
    arb = CostArbitrage()
    report = arb.savings_report()
    assert report.total_decisions == 0
    assert report.total_savings_usd == 0.0
    assert report.savings_pct == 0.0


def test_savings_report_accumulates() -> None:
    arb = CostArbitrage()
    arb.register_equivalent_models("expensive", "cheap")
    arb.update_price("p1", "expensive", 0.010, 0.030)
    arb.update_price("p2", "cheap", 0.001, 0.003)

    arb.find_cheapest("expensive")
    arb.find_cheapest("expensive")
    report = arb.savings_report()

    assert report.total_decisions == 2
    assert report.total_savings_usd > 0
    assert report.savings_pct > 0


def test_savings_report_to_dict() -> None:
    arb = CostArbitrage()
    report = arb.savings_report()
    d = report.to_dict()
    assert "total_decisions" in d
    assert "total_savings_usd" in d
    assert "savings_pct" in d


def test_savings_report_summary() -> None:
    arb = CostArbitrage()
    report = arb.savings_report()
    summary = report.summary()
    assert "savings" in summary.lower()


def test_reset_savings() -> None:
    arb = CostArbitrage()
    arb.register_equivalent_models("a", "b")
    arb.update_price("p1", "a", 0.01, 0.02)
    arb.update_price("p2", "b", 0.001, 0.002)
    arb.find_cheapest("a")

    arb.reset_savings()
    report = arb.savings_report()
    assert report.total_decisions == 0
    assert report.total_savings_usd == 0.0

    # Model registry should still be intact
    assert len(arb.get_equivalent_models("a")) > 0
