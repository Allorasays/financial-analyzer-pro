"""Tests for lite screener ranking."""

from screener_service import build_lite_report_from_fmp, _rank_lists
from screener_universe import resolve_universe


def test_resolve_universe_core():
    tickers = resolve_universe("core", limit=10)
    assert len(tickers) == 10
    assert tickers[0] == "AAPL"


def test_lite_report_and_rank():
    strong = build_lite_report_from_fmp(
        "GOOD",
        {
            "current_price": 100,
            "change_percent": 2.5,
            "pe_ratio": 18,
            "profit_margin": 0.22,
            "gross_margin": 0.45,
            "return_on_equity": 0.25,
            "free_cash_flow": 1e9,
            "debt_to_equity": 0.4,
            "revenue_growth": 0.12,
            "revenue": 1e10,
            "week52_low": 70,
            "week52_high": 110,
        },
    )
    weak = build_lite_report_from_fmp(
        "BAD",
        {
            "current_price": 20,
            "change_percent": -3.0,
            "pe_ratio": 80,
            "profit_margin": -0.1,
            "gross_margin": 0.1,
            "return_on_equity": -0.2,
            "free_cash_flow": -1e8,
            "debt_to_equity": 3.0,
            "revenue_growth": -0.2,
            "revenue": 1e9,
            "week52_low": 18,
            "week52_high": 60,
        },
    )
    lists = _rank_lists([strong, weak], top_n=5)
    assert lists["long_term"][0]["ticker"] == "GOOD"
    assert any(x["ticker"] == "BAD" for x in lists["avoid_long"]) or weak["recommendation_bucket"] == "avoid_long"
