"""
Curated liquid-stock universes for personal-use screening.
Keep lists modest so Render free tier can finish within rate limits.
"""

from __future__ import annotations

from typing import Dict, List

# Highly liquid mega/large caps — default personal screener universe
CORE_LIQUID: List[str] = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "BRK-B",
    "JPM", "V", "UNH", "XOM", "JNJ", "WMT", "MA", "PG", "HD", "CVX",
    "MRK", "ABBV", "COST", "PEP", "KO", "AVGO", "CRM", "AMD", "NFLX",
    "ADBE", "ORCL", "CSCO", "INTC", "QCOM", "TXN", "AMAT", "NOW",
    "BAC", "WFC", "GS", "MS", "BX",
]

TECH_HEAVY: List[str] = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "ORCL", "CRM",
    "AMD", "ADBE", "CSCO", "INTC", "QCOM", "TXN", "AMAT", "NOW",
    "SNOW", "PLTR", "PANW", "CRWD",
]

DEFENSIVE: List[str] = [
    "JNJ", "PG", "KO", "PEP", "WMT", "COST", "MRK", "ABBV", "UNH",
    "XOM", "CVX", "VZ", "T", "NEE", "DUK", "SO", "MDT", "PFE",
]

GROWTH_TILT: List[str] = [
    "NVDA", "TSLA", "META", "AMZN", "GOOGL", "AVGO", "AMD", "NFLX",
    "CRM", "NOW", "SNOW", "PLTR", "SHOP", "UBER", "SQ", "COIN",
]

UNIVERSES: Dict[str, List[str]] = {
    "core": CORE_LIQUID,
    "liquid": CORE_LIQUID,
    "sp500_sample": CORE_LIQUID,
    "tech": TECH_HEAVY,
    "defensive": DEFENSIVE,
    "growth": GROWTH_TILT,
}


def resolve_universe(name: str, custom: List[str] | None = None, limit: int | None = None) -> List[str]:
    if custom:
        tickers = [t.strip().upper().replace(".", "-") for t in custom if t and str(t).strip()]
    else:
        key = (name or "core").lower().strip()
        tickers = list(UNIVERSES.get(key, CORE_LIQUID))
    # de-dupe preserve order
    seen = set()
    out: List[str] = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            out.append(t)
    if limit and limit > 0:
        out = out[:limit]
    return out
