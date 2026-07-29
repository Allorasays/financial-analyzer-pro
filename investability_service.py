"""
Dual-horizon investability scoring for personal research use.

Composes ML predictions, technicals, sentiment, risk, growth, and fundamentals
into short-term and long-term outlooks. Not financial advice.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(score: float, low: float = 0.0, high: float = 100.0) -> float:
    return round(max(low, min(high, score)), 1)


def _pct_change(future: Optional[float], current: Optional[float]) -> Optional[float]:
    if future is None or current is None or current == 0:
        return None
    return ((future - current) / current) * 100.0


def _outlook_from_score(score: float) -> str:
    if score >= 70:
        return "Positive"
    if score >= 55:
        return "Cautiously Positive"
    if score >= 45:
        return "Neutral"
    if score >= 30:
        return "Cautiously Negative"
    return "Negative"


def _ml_horizon_score(
    current: Optional[float],
    predicted: Optional[float],
    conf: Optional[float],
) -> Tuple[float, List[str]]:
    """Map predicted % move + confidence into 0–100 component score."""
    drivers: List[str] = []
    pct = _pct_change(predicted, current)
    if pct is None:
        return 50.0, ["ML horizon unavailable"]

    # Map roughly -8%..+8% → 20..80, then weight by confidence
    raw = 50.0 + max(-30.0, min(30.0, pct * 3.5))
    weight = 0.55 + 0.45 * max(0.0, min(1.0, conf if conf is not None else 0.5))
    score = 50.0 + (raw - 50.0) * weight
    direction = "up" if pct >= 0 else "down"
    drivers.append(f"ML predicts {direction} {abs(pct):.1f}% (conf {int((conf or 0.5) * 100)}%)")
    return score, drivers


def _technical_score(tech: Dict[str, Any]) -> Tuple[float, List[str], List[str]]:
    drivers: List[str] = []
    risks: List[str] = []
    indicators = tech.get("indicators") or {}
    signals = indicators.get("signals") or {}
    score = 50.0

    trend = str(signals.get("trend", "")).lower()
    if trend == "bullish":
        score += 12
        drivers.append("Price above SMA trend (bullish)")
    elif trend == "bearish":
        score -= 12
        risks.append("Price below SMA trend (bearish)")

    rsi = _safe_float(indicators.get("rsi"))
    if rsi is not None:
        if rsi < 30:
            score += 8
            drivers.append(f"RSI oversold ({rsi:.0f})")
        elif rsi > 70:
            score -= 8
            risks.append(f"RSI overbought ({rsi:.0f})")
        else:
            drivers.append(f"RSI neutral ({rsi:.0f})")

    macd_sig = str(signals.get("macd_signal", "")).lower()
    if macd_sig == "bullish":
        score += 8
        drivers.append("MACD bullish")
    elif macd_sig == "bearish":
        score -= 8
        risks.append("MACD bearish")

    return _clamp(score), drivers, risks


def _sentiment_score(sentiment: Dict[str, Any]) -> Tuple[float, List[str], List[str]]:
    drivers: List[str] = []
    risks: List[str] = []
    raw = _safe_float(sentiment.get("sentiment_score"), 0.0) or 0.0
    overall = sentiment.get("overall_sentiment") or "Neutral"
    conf = _safe_float(
        sentiment.get("confidence"),
        _safe_float(sentiment.get("confidence_score"), 0.5),
    ) or 0.5
    if conf > 1.0:
        conf = conf / 100.0
    # sentiment_score typically -1..1
    score = 50.0 + max(-25.0, min(25.0, raw * 40.0)) * (0.5 + 0.5 * conf)
    if raw > 0.15:
        drivers.append(f"Sentiment {overall} ({raw:+.2f})")
    elif raw < -0.15:
        risks.append(f"Sentiment {overall} ({raw:+.2f})")
    else:
        drivers.append(f"Sentiment {overall}")
    return _clamp(score), drivers, risks


def _risk_penalty(risk: Dict[str, Any]) -> Tuple[float, List[str]]:
    """Return 0–100 where higher is safer (less penalty when inverted into scores)."""
    risks: List[str] = []
    assessment = risk.get("risk_assessment") or {}
    rating = str(assessment.get("risk_rating", "")).lower()
    score_map = {"low": 80.0, "moderate": 60.0, "high": 35.0, "very high": 20.0}
    score = score_map.get(rating, 50.0)
    factors = assessment.get("risk_factors") or []
    if factors:
        risks.extend([str(f) for f in factors[:4]])
    elif rating:
        risks.append(f"Risk rating: {assessment.get('risk_rating')}")
    metrics = risk.get("risk_metrics") or {}
    vol = _safe_float(metrics.get("volatility_1y"))
    if vol is not None and vol > 40:
        risks.append(f"High 1y volatility ({vol:.0f}%)")
    return score, risks


def _fundamental_long_score(financials: Dict[str, Any], growth: Dict[str, Any]) -> Tuple[float, List[str], List[str]]:
    drivers: List[str] = []
    risks: List[str] = []
    score = 50.0

    profit_margin = _safe_float(financials.get("profit_margin"))
    if profit_margin is not None:
        # margins often 0–1 or already percent
        pm = profit_margin * 100 if abs(profit_margin) <= 1.5 else profit_margin
        if pm >= 15:
            score += 10
            drivers.append(f"Strong profit margin ({pm:.1f}%)")
        elif pm >= 5:
            score += 4
            drivers.append(f"Positive profit margin ({pm:.1f}%)")
        elif pm < 0:
            score -= 12
            risks.append(f"Negative profit margin ({pm:.1f}%)")

    gross = _safe_float(financials.get("gross_margin"))
    if gross is not None:
        gm = gross * 100 if abs(gross) <= 1.5 else gross
        if gm >= 40:
            score += 6
            drivers.append(f"Healthy gross margin ({gm:.1f}%)")
        elif gm < 20:
            score -= 4
            risks.append(f"Thin gross margin ({gm:.1f}%)")

    roe = _safe_float(financials.get("return_on_equity"))
    if roe is not None:
        roe_pct = roe * 100 if abs(roe) <= 1.5 else roe
        if roe_pct >= 15:
            score += 8
            drivers.append(f"Strong ROE ({roe_pct:.1f}%)")
        elif roe_pct < 0:
            score -= 10
            risks.append(f"Negative ROE ({roe_pct:.1f}%)")

    fcf = _safe_float(financials.get("free_cash_flow"))
    if fcf is not None:
        if fcf > 0:
            score += 6
            drivers.append("Positive free cash flow")
        else:
            score -= 8
            risks.append("Negative free cash flow")

    debt_equity = _safe_float(financials.get("debt_to_equity"))
    if debt_equity is not None:
        de = debt_equity / 100 if debt_equity > 10 else debt_equity
        if de > 2.0:
            score -= 10
            risks.append(f"High leverage (D/E {de:.2f})")
        elif de < 0.5:
            score += 4
            drivers.append(f"Conservative leverage (D/E {de:.2f})")

    rev_growth = _safe_float(financials.get("revenue_growth"))
    if rev_growth is None:
        fund = growth.get("fundamental_growth") or {}
        rev_growth = _safe_float(fund.get("revenue_growth"))
        if rev_growth is not None and abs(rev_growth) > 1.5:
            pass  # already percent
        elif rev_growth is not None and abs(rev_growth) <= 1.5:
            rev_growth = rev_growth * 100
    else:
        if abs(rev_growth) <= 1.5:
            rev_growth = rev_growth * 100

    if rev_growth is not None:
        if rev_growth >= 10:
            score += 8
            drivers.append(f"Revenue growth {rev_growth:.1f}%")
        elif rev_growth <= -5:
            score -= 10
            risks.append(f"Revenue declining ({rev_growth:.1f}%)")

    g_analysis = growth.get("growth_analysis") or {}
    grade = g_analysis.get("growth_grade")
    if grade == "A":
        score += 8
        drivers.append("Growth grade A")
    elif grade == "B":
        score += 4
        drivers.append("Growth grade B")
    elif grade == "D":
        score -= 8
        risks.append("Growth grade D")

    returns = growth.get("returns") or {}
    r1y = _safe_float(returns.get("returns_1y"))
    if r1y is not None:
        if r1y >= 15:
            score += 5
            drivers.append(f"1y price return {r1y:.1f}%")
        elif r1y <= -20:
            score -= 8
            risks.append(f"Weak 1y price return ({r1y:.1f}%)")

    return _clamp(score), drivers, risks


def _macro_adjustment(fred_data: Optional[Dict[str, Any]]) -> Tuple[float, List[str]]:
    if not fred_data:
        return 0.0, []
    notes: List[str] = []
    delta = 0.0
    vix = _safe_float(fred_data.get("vix"))
    if vix is not None:
        if vix >= 30:
            delta -= 6
            notes.append(f"Elevated VIX ({vix:.1f}) — risk-off")
        elif vix <= 15:
            delta += 3
            notes.append(f"Calm VIX ({vix:.1f}) — risk-on bias")
    unemp = _safe_float(fred_data.get("unemployment_rate"))
    if unemp is not None and unemp >= 5.5:
        delta -= 2
        notes.append(f"Unemployment elevated ({unemp:.1f}%)")
    return delta, notes


def _bucket(short: Dict[str, Any], long: Dict[str, Any]) -> str:
    s, l = short["score"], long["score"]
    if l < 30 or (l < 35 and str(long.get("outlook", "")).startswith("Negative")):
        return "avoid_long"
    if s >= 65 and l >= 55:
        return "short_and_long"
    if s >= 65:
        return "short_buy"
    if l >= 65:
        return "long_buy"
    if l < 40:
        return "avoid_long"
    return "hold"


def build_investability_report(
    ticker: str,
    *,
    ml: Optional[Dict[str, Any]] = None,
    technical: Optional[Dict[str, Any]] = None,
    sentiment: Optional[Dict[str, Any]] = None,
    risk: Optional[Dict[str, Any]] = None,
    growth: Optional[Dict[str, Any]] = None,
    financials: Optional[Dict[str, Any]] = None,
    macro: Optional[Dict[str, Any]] = None,
    peers: Optional[Dict[str, Any]] = None,
    ml_accuracy: Optional[Dict[str, Any]] = None,
    alt_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build dual-horizon investability report.

    Short-term (~days–weeks): ML day/week 40%, technical 30%, sentiment 20%, risk 10%
    Long-term (~months–years): fundamentals 35%, growth/ML longer 25%, risk 20%, ML month/quarter 15%, sentiment 5%
    """
    ml = ml or {}
    technical = technical or {}
    sentiment = sentiment or {}
    risk = risk or {}
    growth = growth or {}
    financials = financials or {}
    macro = macro or {}
    peers = peers or {}
    alt_data = alt_data or {}

    current = _safe_float(ml.get("current_price")) or _safe_float(financials.get("current_price"))
    conf = _safe_float(ml.get("confidence_score"), 0.5)
    conf_scores = (ml.get("predictions") or {}).get("confidence_scores") or []

    # --- Short term ---
    day_score, day_drivers = _ml_horizon_score(
        current, _safe_float(ml.get("next_day")), conf_scores[0] if conf_scores else conf
    )
    week_score, week_drivers = _ml_horizon_score(
        current,
        _safe_float(ml.get("next_week")),
        conf_scores[1] if len(conf_scores) > 1 else (conf * 0.95 if conf else 0.5),
    )
    ml_short = 0.55 * day_score + 0.45 * week_score
    tech_score, tech_drivers, tech_risks = _technical_score(technical)
    sent_score, sent_drivers, sent_risks = _sentiment_score(sentiment)
    risk_safe, risk_notes = _risk_penalty(risk)

    short_raw = (
        ml_short * 0.40
        + tech_score * 0.30
        + sent_score * 0.20
        + risk_safe * 0.10
    )
    short_score = _clamp(short_raw)
    short_drivers = (day_drivers + week_drivers + tech_drivers + sent_drivers)[:6]
    short_risks = (tech_risks + sent_risks + risk_notes)[:5]
    short_conf = _clamp(
        (
            (conf or 0.5) * 0.5
            + (_safe_float(sentiment.get("confidence"), _safe_float(sentiment.get("confidence_score"), 0.5)) or 0.5) * 0.3
            + (0.7 if technical.get("indicators") else 0.4) * 0.2
        )
        * 100,
        0,
        100,
    )

    short_block = {
        "horizon": "short_term",
        "horizon_label": "Days to weeks",
        "score": short_score,
        "outlook": _outlook_from_score(short_score),
        "confidence": short_conf,
        "drivers": short_drivers or ["Limited short-term signals"],
        "risks": short_risks or ["No major short-term risk flags"],
        "components": {
            "ml": round(ml_short, 1),
            "technical": tech_score,
            "sentiment": sent_score,
            "risk_safety": risk_safe,
        },
    }

    # --- Long term ---
    month_score, month_drivers = _ml_horizon_score(
        current,
        _safe_float(ml.get("next_month")),
        conf_scores[2] if len(conf_scores) > 2 else (conf * 0.9 if conf else 0.5),
    )
    quarter_score, quarter_drivers = _ml_horizon_score(
        current,
        _safe_float(ml.get("next_quarter")),
        conf_scores[3] if len(conf_scores) > 3 else (conf * 0.85 if conf else 0.5),
    )
    ml_long = 0.55 * month_score + 0.45 * quarter_score
    fund_score, fund_drivers, fund_risks = _fundamental_long_score(financials, growth)

    long_raw = (
        fund_score * 0.35
        + ml_long * 0.25
        + risk_safe * 0.20
        + tech_score * 0.10
        + sent_score * 0.10
    )
    if fund_score < 40 and risk_safe < 40:
        long_raw -= 8

    # Peer-relative fundamentals (Phase 4)
    peer_notes: List[str] = []
    peer_list = peers.get("peers") if isinstance(peers.get("peers"), list) else []
    my_pe = _safe_float(financials.get("pe_ratio"))
    peer_pes = [
        _safe_float(p.get("pe_ratio") or p.get("pe"))
        for p in peer_list
        if isinstance(p, dict)
    ]
    peer_pes = [p for p in peer_pes if p is not None and p > 0]
    if my_pe and peer_pes:
        peer_avg = sum(peer_pes) / len(peer_pes)
        if my_pe < peer_avg * 0.85:
            long_raw += 4
            peer_notes.append(f"P/E below peer avg ({my_pe:.1f} vs {peer_avg:.1f})")
        elif my_pe > peer_avg * 1.25:
            long_raw -= 4
            peer_notes.append(f"P/E above peer avg ({my_pe:.1f} vs {peer_avg:.1f})")

    # Macro overlay
    macro_delta, macro_notes = _macro_adjustment(macro)
    short_block["score"] = _clamp(short_block["score"] + macro_delta * 0.5)
    short_block["outlook"] = _outlook_from_score(short_block["score"])
    if macro_notes:
        short_block["drivers"] = (short_block["drivers"] + macro_notes)[:6]
    long_raw += macro_delta * 0.35

    # Alt-data avoid penalties (insider selling / weak institutional)
    alt_risks: List[str] = []
    insider = alt_data.get("insider_transactions") if isinstance(alt_data.get("insider_transactions"), dict) else {}
    if insider.get("net_sentiment") == "bearish" or insider.get("signal") == "selling":
        long_raw -= 5
        alt_risks.append("Insider selling pressure flagged")
    holdings = alt_data.get("institutional_holdings") if isinstance(alt_data.get("institutional_holdings"), dict) else {}
    if holdings.get("trend") == "decreasing":
        long_raw -= 3
        alt_risks.append("Institutional holdings trending down")

    long_score = _clamp(long_raw)
    long_drivers = (fund_drivers + month_drivers + quarter_drivers + peer_notes + macro_notes)[:6]
    long_risks = (fund_risks + risk_notes + alt_risks)[:5]
    long_conf = _clamp(
        (
            (0.75 if financials.get("revenue") is not None else 0.4) * 0.45
            + (conf or 0.5) * 0.35
            + (0.7 if growth else 0.4) * 0.20
        )
        * 100,
        0,
        100,
    )

    # ML historical accuracy dampens confidence when known weak
    if ml_accuracy and isinstance(ml_accuracy, dict):
        dir_acc = _safe_float(ml_accuracy.get("direction_accuracy_pct") or ml_accuracy.get("direction_accuracy"))
        if dir_acc is not None:
            if dir_acc <= 1.0:
                dir_acc = dir_acc * 100
            if dir_acc < 50:
                short_block["confidence"] = _clamp(short_block["confidence"] * 0.85)
                long_conf = _clamp(long_conf * 0.85)
                short_block["risks"] = (short_block["risks"] + [f"ML direction accuracy low ({dir_acc:.0f}%)"])[:5]

    long_block = {
        "horizon": "long_term",
        "horizon_label": "Months to years",
        "score": long_score,
        "outlook": _outlook_from_score(long_score),
        "confidence": long_conf,
        "drivers": long_drivers or ["Limited long-term signals"],
        "risks": long_risks or ["No major long-term risk flags"],
        "components": {
            "fundamentals": fund_score,
            "ml": round(ml_long, 1),
            "risk_safety": risk_safe,
            "technical": tech_score,
            "sentiment": sent_score,
        },
    }

    recommendation_bucket = _bucket(short_block, long_block)
    bucket_labels = {
        "short_buy": "More attractive short-term than long-term",
        "long_buy": "More attractive for longer-term holding",
        "short_and_long": "Constructive on both horizons",
        "avoid_long": "Weak long-term outlook — research caution",
        "hold": "Mixed / wait for clearer signals",
    }

    data_gaps: List[str] = []
    if not ml or ml.get("status") != "success":
        data_gaps.append("ML predictions incomplete")
    if not (technical.get("indicators")):
        data_gaps.append("Technical indicators unavailable")
    if financials.get("revenue") is None:
        data_gaps.append("Financial statements incomplete (set FMP_API_KEY)")
    if not sentiment:
        data_gaps.append("Sentiment unavailable")
    if not risk:
        data_gaps.append("Risk assessment unavailable")
    if not peer_list:
        data_gaps.append("Peer set incomplete")

    result = {
        "ticker": ticker.upper(),
        "timestamp": datetime.now().isoformat(),
        "current_price": current,
        "short_term": short_block,
        "long_term": long_block,
        "recommendation_bucket": recommendation_bucket,
        "recommendation_label": bucket_labels.get(recommendation_bucket, "Hold"),
        "peers": {
            "tickers": [p.get("ticker") for p in peer_list if isinstance(p, dict) and p.get("ticker")][:8],
            "count": len(peer_list),
            "notes": peer_notes,
        },
        "data_gaps": data_gaps,
        "disclaimer": (
            "Personal research score only — not investment advice. "
            "Scores combine model outputs and public data and can be wrong."
        ),
        "personal_use_only": True,
    }
    if ml_accuracy:
        result["ml_accuracy"] = ml_accuracy
    if macro_notes:
        result["macro_notes"] = macro_notes
    return result
