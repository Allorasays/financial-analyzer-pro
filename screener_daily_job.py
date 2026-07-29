"""
Nightly / on-demand screener refresh job.
Persists ranked short/long/avoid lists via the backend ScreenerEngine cache.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_screener_refresh(
    universe: str = "core",
    limit: int = 25,
    top_n: int = 10,
    mode: str = "lite",
) -> dict:
    from dotenv import load_dotenv

    load_dotenv(override=True)

    from fmp_service import fmp_service
    from prediction_tracker import prediction_tracker
    from proxy import db_manager
    from screener_service import SCREENER_CACHE_TYPE, ScreenerEngine
    import json

    def persist(key: str, payload: dict) -> None:
        ttl = max(30, int(os.getenv("SCREENER_CACHE_TTL_MINUTES", "720")))
        db_manager.cache_market_data(key, SCREENER_CACHE_TYPE, json.dumps(payload), cache_duration_minutes=ttl)

    def load(key: str):
        return db_manager.get_cached_market_data(key, SCREENER_CACHE_TYPE) or db_manager.get_cached_market_data_stale(
            key, SCREENER_CACHE_TYPE
        )

    def macro():
        try:
            from fred_indicators import get_fred_indicators

            return get_fred_indicators() or {}
        except Exception:
            return {}

    def accuracy(ticker=None):
        try:
            m = prediction_tracker.calculate_accuracy_metrics(ticker=ticker)
            if m and m.get("status") != "insufficient_data":
                return m
            return prediction_tracker.calculate_accuracy_metrics()
        except Exception:
            return {}

    engine = ScreenerEngine(
        fmp_service=fmp_service,
        get_macro=macro,
        get_ml_accuracy=accuracy,
        persist=persist,
        load_persisted=load,
    )
    result = engine.run(universe=universe, limit=limit, top_n=top_n, mode=mode, max_workers=3)
    logger.info(
        "Screener refresh complete: scanned=%s scored=%s short=%s long=%s avoid=%s",
        result.get("scanned"),
        result.get("scored"),
        len((result.get("lists") or {}).get("short_term") or []),
        len((result.get("lists") or {}).get("long_term") or []),
        len((result.get("lists") or {}).get("avoid_long") or []),
    )
    return result


def main():
    universe = os.getenv("SCREENER_UNIVERSE", "core")
    limit = int(os.getenv("SCREENER_LIMIT", "25"))
    top_n = int(os.getenv("SCREENER_TOP_N", "10"))
    mode = os.getenv("SCREENER_MODE", "lite")
    logger.info("Starting screener refresh universe=%s limit=%s mode=%s", universe, limit, mode)
    run_screener_refresh(universe=universe, limit=limit, top_n=top_n, mode=mode)


if __name__ == "__main__":
    main()
