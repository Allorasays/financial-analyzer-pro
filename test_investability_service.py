"""Smoke tests for dual-horizon investability scoring."""

from investability_service import build_investability_report


def test_investability_positive_short_long():
    report = build_investability_report(
        "AAPL",
        ml={
            "status": "success",
            "current_price": 100.0,
            "next_day": 102.0,
            "next_week": 104.0,
            "next_month": 108.0,
            "next_quarter": 112.0,
            "confidence_score": 0.8,
            "predictions": {"confidence_scores": [0.8, 0.76, 0.72, 0.68]},
        },
        technical={
            "indicators": {
                "rsi": 45,
                "signals": {"trend": "Bullish", "macd_signal": "Bullish"},
            }
        },
        sentiment={"sentiment_score": 0.3, "overall_sentiment": "Bullish", "confidence": 0.7},
        risk={
            "risk_assessment": {"risk_rating": "Moderate", "risk_factors": ["Moderate volatility"]},
            "risk_metrics": {"volatility_1y": 22.0},
        },
        growth={
            "growth_analysis": {"growth_grade": "B"},
            "returns": {"returns_1y": 18.0},
            "fundamental_growth": {"revenue_growth": 12.0},
        },
        financials={
            "current_price": 100.0,
            "revenue": 1e11,
            "profit_margin": 0.25,
            "gross_margin": 0.45,
            "return_on_equity": 0.3,
            "free_cash_flow": 1e10,
            "debt_to_equity": 0.8,
            "revenue_growth": 0.12,
        },
    )
    assert report["ticker"] == "AAPL"
    assert "short_term" in report and "long_term" in report
    assert report["short_term"]["score"] >= 50
    assert report["long_term"]["score"] >= 50
    assert report["recommendation_bucket"] in {
        "short_buy",
        "long_buy",
        "short_and_long",
        "hold",
        "avoid_long",
    }
    assert report["disclaimer"]


def test_investability_avoid_long_weak_fundamentals():
    report = build_investability_report(
        "WEAK",
        ml={
            "status": "success",
            "current_price": 50.0,
            "next_day": 49.0,
            "next_week": 47.0,
            "next_month": 40.0,
            "next_quarter": 35.0,
            "confidence_score": 0.5,
            "predictions": {"confidence_scores": [0.5, 0.45, 0.4, 0.35]},
        },
        technical={
            "indicators": {
                "rsi": 75,
                "signals": {"trend": "Bearish", "macd_signal": "Bearish"},
            }
        },
        sentiment={"sentiment_score": -0.4, "overall_sentiment": "Bearish", "confidence": 0.6},
        risk={
            "risk_assessment": {
                "risk_rating": "Very High",
                "risk_factors": ["High volatility", "Large historical losses"],
            },
            "risk_metrics": {"volatility_1y": 55.0},
        },
        growth={
            "growth_analysis": {"growth_grade": "D"},
            "returns": {"returns_1y": -35.0},
        },
        financials={
            "current_price": 50.0,
            "revenue": 1e9,
            "profit_margin": -0.1,
            "gross_margin": 0.15,
            "return_on_equity": -0.2,
            "free_cash_flow": -1e8,
            "debt_to_equity": 3.5,
            "revenue_growth": -0.15,
        },
    )
    assert report["long_term"]["score"] < 45
    assert report["recommendation_bucket"] == "avoid_long"
    assert any("margin" in r.lower() or "leverage" in r.lower() or "cash" in r.lower()
               for r in report["long_term"]["risks"])
