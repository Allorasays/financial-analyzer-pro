"""
Stock screener: scan a universe, score short/long investability, rank, and cache.

Lite mode uses FMP quotes + fundamentals (rate-limit friendly).
Full mode calls build_investability_report with richer inputs when available.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from investability_service import (
    _bucket,
    _clamp,
    _fundamental_long_score,
    _macro_adjustment,
    _outlook_from_score,
    _safe_float,
    build_investability_report,
)
from screener_universe import resolve_universe

logger = logging.getLogger(__name__)

SCREENER_CACHE_KEY = "_SCREENER_"
SCREENER_CACHE_TYPE = "screener_results"


def _macro_adjustment(fred_data: Optional[Dict[str, Any]]) -> tuple[float, List[str]]:
    """Return score delta (-10..+10) and notes from FRED-like indicators."""
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


def build_lite_report_from_fmp(
    ticker: str,
    financials: Dict[str, Any],
    *,
    macro: Optional[Dict[str, Any]] = None,
    ml_accuracy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Fast dual-horizon score from FMP-style financial/quote fields."""
    financials = financials or {}
    current = _safe_float(financials.get("current_price") or financials.get("price"))
    chg = _safe_float(financials.get("change_percent") or financials.get("changesPercentage"))
    # FMP sometimes returns percent already
    if chg is not None and abs(chg) < 1.5 and financials.get("change") is not None:
        # could be fractional; leave as-is if already looks like percent
        pass

    # Short-term: momentum from day change + volume proxy
    short = 50.0
    short_drivers: List[str] = []
    short_risks: List[str] = []
    if chg is not None:
        short += max(-20.0, min(20.0, chg * 2.5))
        if chg >= 1:
            short_drivers.append(f"Day change {chg:+.1f}%")
        elif chg <= -1:
            short_risks.append(f"Day change {chg:+.1f}%")
        else:
            short_drivers.append(f"Flat session ({chg:+.1f}%)")
    else:
        short_drivers.append("Quote momentum unavailable")

    pe = _safe_float(financials.get("pe_ratio") or financials.get("pe"))
    if pe is not None and 0 < pe < 15:
        short += 3
        short_drivers.append(f"Attractive P/E ({pe:.1f})")
    elif pe is not None and pe > 45:
        short -= 3
        short_risks.append(f"Rich P/E ({pe:.1f})")

    fund_score, fund_drivers, fund_risks = _fundamental_long_score(financials, {})
    long = fund_score

    # Use year range position as crude long/short technical
    lo = _safe_float(financials.get("week52_low") or financials.get("year_low") or financials.get("52_week_low"))
    hi = _safe_float(financials.get("week52_high") or financials.get("year_high") or financials.get("52_week_high"))
    if current and lo and hi and hi > lo:
        pos = (current - lo) / (hi - lo)
        if pos > 0.85:
            short -= 4
            short_risks.append("Near 52-week high")
            long += 2
            fund_drivers.append("Trading near 52-week high (momentum)")
        elif pos < 0.25:
            short += 4
            short_drivers.append("Near 52-week low (mean-reversion watch)")
            long -= 3
            fund_risks.append("Near 52-week low")

    macro_delta, macro_notes = _macro_adjustment(macro)
    short = _clamp(short + macro_delta * 0.5)
    long = _clamp(long + macro_delta * 0.35)

    short_block = {
        "horizon": "short_term",
        "horizon_label": "Days to weeks",
        "score": short,
        "outlook": _outlook_from_score(short),
        "confidence": 55.0 if chg is not None else 40.0,
        "drivers": (short_drivers + macro_notes)[:6] or ["Limited short-term signals"],
        "risks": short_risks[:5] or ["No major short-term risk flags"],
        "components": {"momentum": short, "fundamentals": fund_score},
    }
    long_block = {
        "horizon": "long_term",
        "horizon_label": "Months to years",
        "score": long,
        "outlook": _outlook_from_score(long),
        "confidence": 60.0 if financials.get("revenue") is not None else 40.0,
        "drivers": (fund_drivers + macro_notes)[:6] or ["Limited long-term signals"],
        "risks": fund_risks[:5] or ["No major long-term risk flags"],
        "components": {"fundamentals": fund_score},
    }

    gaps = ["Lite screener mode (ML/tech/sentiment skipped for speed)"]
    if financials.get("revenue") is None:
        gaps.append("Incomplete financial statements")

    report = {
        "ticker": ticker.upper(),
        "timestamp": datetime.now().isoformat(),
        "current_price": current,
        "short_term": short_block,
        "long_term": long_block,
        "recommendation_bucket": _bucket(short_block, long_block),
        "mode": "lite",
        "data_gaps": gaps,
        "disclaimer": (
            "Personal research score only — not investment advice. "
            "Lite mode emphasizes fundamentals and price momentum."
        ),
        "personal_use_only": True,
    }
    if ml_accuracy:
        report["ml_accuracy"] = ml_accuracy
    # label
    labels = {
        "short_buy": "More attractive short-term than long-term",
        "long_buy": "More attractive for longer-term holding",
        "short_and_long": "Constructive on both horizons",
        "avoid_long": "Weak long-term outlook — research caution",
        "hold": "Mixed / wait for clearer signals",
    }
    report["recommendation_label"] = labels.get(report["recommendation_bucket"], "Hold")
    return report


def _rank_lists(rows: List[Dict[str, Any]], top_n: int = 10) -> Dict[str, List[Dict[str, Any]]]:
    def summary(r: Dict[str, Any]) -> Dict[str, Any]:
        st = r.get("short_term") or {}
        lt = r.get("long_term") or {}
        return {
            "ticker": r.get("ticker"),
            "current_price": r.get("current_price"),
            "short_score": st.get("score"),
            "short_outlook": st.get("outlook"),
            "long_score": lt.get("score"),
            "long_outlook": lt.get("outlook"),
            "recommendation_bucket": r.get("recommendation_bucket"),
            "recommendation_label": r.get("recommendation_label"),
            "drivers_short": (st.get("drivers") or [])[:3],
            "drivers_long": (lt.get("drivers") or [])[:3],
            "risks_long": (lt.get("risks") or [])[:3],
            "mode": r.get("mode", "lite"),
            "why": r.get("recommendation_label"),
        }

    scored = [r for r in rows if r.get("short_term") and r.get("long_term")]
    short_ranked = sorted(scored, key=lambda r: (r["short_term"].get("score") or 0), reverse=True)
    long_ranked = sorted(scored, key=lambda r: (r["long_term"].get("score") or 0), reverse=True)
    avoid = [
        r for r in scored
        if r.get("recommendation_bucket") == "avoid_long" or (r["long_term"].get("score") or 100) < 40
    ]
    avoid_ranked = sorted(avoid, key=lambda r: (r["long_term"].get("score") or 0))

    return {
        "short_term": [summary(r) for r in short_ranked[:top_n]],
        "long_term": [summary(r) for r in long_ranked[:top_n]],
        "avoid_long": [summary(r) for r in avoid_ranked[:top_n]],
    }


class ScreenerEngine:
    def __init__(
        self,
        *,
        fmp_service=None,
        fetch_financials: Optional[Callable[[str], Dict[str, Any]]] = None,
        build_full_report: Optional[Callable[[str], Dict[str, Any]]] = None,
        get_macro: Optional[Callable[[], Dict[str, Any]]] = None,
        get_ml_accuracy: Optional[Callable[[Optional[str]], Dict[str, Any]]] = None,
        persist: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        load_persisted: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
    ):
        self.fmp = fmp_service
        self.fetch_financials = fetch_financials
        self.build_full_report = build_full_report
        self.get_macro = get_macro
        self.get_ml_accuracy = get_ml_accuracy
        self.persist = persist
        self.load_persisted = load_persisted

    def _fetch_one_lite(self, ticker: str, macro: Optional[Dict[str, Any]], accuracy: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        financials: Dict[str, Any] = {}
        try:
            if self.fetch_financials:
                financials = self.fetch_financials(ticker) or {}
            elif self.fmp and getattr(self.fmp, "enabled", False):
                # Prefer quote + profile + ratios (lighter than full statements)
                quote = self.fmp.get_quote(ticker) or {}
                profile = self.fmp.get_company_profile(ticker) or {}
                ratios = self.fmp.get_financial_ratios(ticker) or {}
                metrics = self.fmp.get_key_metrics(ticker) or {}
                financials = {
                    "current_price": quote.get("price") or profile.get("price"),
                    "change_percent": quote.get("changesPercentage") or quote.get("changePercentage"),
                    "change": quote.get("change"),
                    "pe_ratio": quote.get("pe") or ratios.get("priceToEarningsRatio") or metrics.get("peRatio"),
                    "market_cap": quote.get("marketCap") or profile.get("mktCap"),
                    "week52_low": quote.get("yearLow"),
                    "week52_high": quote.get("yearHigh"),
                    "revenue": None,
                    "profit_margin": ratios.get("netProfitMargin"),
                    "gross_margin": ratios.get("grossProfitMargin"),
                    "return_on_equity": ratios.get("returnOnEquity"),
                    "return_on_assets": ratios.get("returnOnAssets"),
                    "debt_to_equity": ratios.get("debtEquityRatio"),
                    "free_cash_flow": metrics.get("freeCashFlowYield"),  # may be yield; ok as soft signal
                    "revenue_growth": ratios.get("revenuePerShareGrowth") or metrics.get("revenueGrowth"),
                    "company_name": profile.get("companyName"),
                    "sector": profile.get("sector"),
                    "industry": profile.get("industry"),
                }
                # freeCashFlowYield is not FCF dollars — drop if confusing
                if financials.get("free_cash_flow") is not None and abs(float(financials["free_cash_flow"] or 0)) < 5:
                    # likely a yield/ratio; ignore for FCF sign logic
                    financials["free_cash_flow"] = None
        except Exception as e:
            logger.warning("Lite fetch failed for %s: %s", ticker, e)
            return {
                "ticker": ticker,
                "error": str(e),
                "short_term": {"score": 50, "outlook": "Neutral", "drivers": [], "risks": [str(e)]},
                "long_term": {"score": 50, "outlook": "Neutral", "drivers": [], "risks": [str(e)]},
                "recommendation_bucket": "hold",
                "recommendation_label": "Mixed / wait for clearer signals",
                "mode": "lite",
                "data_gaps": ["Fetch failed"],
            }
        return build_lite_report_from_fmp(ticker, financials, macro=macro, ml_accuracy=accuracy)

    def run(
        self,
        *,
        universe: str = "core",
        tickers: Optional[List[str]] = None,
        limit: Optional[int] = None,
        top_n: int = 10,
        mode: str = "lite",
        max_workers: int = 4,
    ) -> Dict[str, Any]:
        symbols = resolve_universe(universe, tickers, limit=limit)
        macro = {}
        try:
            if self.get_macro:
                macro = self.get_macro() or {}
        except Exception as e:
            logger.debug("macro unavailable: %s", e)

        accuracy = None
        try:
            if self.get_ml_accuracy:
                accuracy = self.get_ml_accuracy(None)
        except Exception:
            accuracy = None

        rows: List[Dict[str, Any]] = []
        errors: List[Dict[str, str]] = []
        started = time.time()

        if mode == "full" and self.build_full_report:
            # Cap full mode for rate limits
            symbols = symbols[: min(len(symbols), limit or 12)]
            with ThreadPoolExecutor(max_workers=min(max_workers, 2)) as pool:
                futs = {pool.submit(self.build_full_report, t): t for t in symbols}
                for fut in as_completed(futs):
                    t = futs[fut]
                    try:
                        report = fut.result()
                        if report:
                            report["mode"] = "full"
                            rows.append(report)
                    except Exception as e:
                        errors.append({"ticker": t, "error": str(e)})
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futs = {pool.submit(self._fetch_one_lite, t, macro, accuracy): t for t in symbols}
                for fut in as_completed(futs):
                    t = futs[fut]
                    try:
                        rows.append(fut.result())
                    except Exception as e:
                        errors.append({"ticker": t, "error": str(e)})

        ranked = _rank_lists(rows, top_n=top_n)
        payload = {
            "status": "success",
            "universe": universe if not tickers else "custom",
            "mode": mode if mode == "full" else "lite",
            "scanned": len(symbols),
            "scored": len(rows),
            "errors": errors,
            "elapsed_seconds": round(time.time() - started, 2),
            "generated_at": datetime.now().isoformat(),
            "lists": ranked,
            "results": rows,
            "ml_accuracy": accuracy,
            "macro_notes": _macro_adjustment(macro)[1],
            "disclaimer": (
                "Personal research rankings only — not investment advice. "
                "Past performance and model scores do not guarantee future results."
            ),
            "personal_use_only": True,
        }

        if self.persist:
            try:
                cache_body = {
                    k: v for k, v in payload.items() if k != "results"
                }
                # Keep compact summaries for cache; full results optional
                cache_body["results_count"] = len(rows)
                self.persist(SCREENER_CACHE_KEY, cache_body)
            except Exception as e:
                logger.warning("Failed to persist screener results: %s", e)

        return payload

    def score_tickers(self, tickers: List[str], *, mode: str = "lite", top_n: int = 20) -> Dict[str, Any]:
        """Score an arbitrary list (watchlist / portfolio personalization)."""
        return self.run(universe="custom", tickers=tickers, mode=mode, top_n=top_n)

    def latest(self) -> Optional[Dict[str, Any]]:
        if not self.load_persisted:
            return None
        return self.load_persisted(SCREENER_CACHE_KEY)
