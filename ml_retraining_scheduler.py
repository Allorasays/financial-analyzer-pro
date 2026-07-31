#!/usr/bin/env python3
"""
Trigger proactive ML retraining by invoking the Skill pipeline
(force_retrain) so joblib models are refreshed for a ticker basket.
"""

from __future__ import annotations

import argparse
from datetime import datetime

from proxy import get_ml_predictions
from ml_model_store import MODEL_DIR

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN", "SPY",
    "META", "AMD", "JPM", "QQQ", "AVGO", "COST", "XOM", "UNH",
]
DEFAULT_DAYS_AHEAD = 30


def retrain_ticker(ticker: str, days_ahead: int) -> dict:
    print(f"[RETRAIN] Updating skill model for {ticker} ({days_ahead}d)")
    result = get_ml_predictions(ticker, days_ahead, force_retrain=True)
    result["retrained_at"] = datetime.utcnow().isoformat()
    return result


def main():
    parser = argparse.ArgumentParser(description="Run proactive ML skill retraining.")
    parser.add_argument(
        "--tickers",
        default=",".join(DEFAULT_TICKERS),
        help="Comma separated list of tickers to retrain.",
    )
    parser.add_argument(
        "--days-ahead",
        type=int,
        default=DEFAULT_DAYS_AHEAD,
        help="Prediction horizon to refresh.",
    )
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    ok = 0
    for ticker in tickers:
        payload = retrain_ticker(ticker, args.days_ahead)
        if payload.get("status") == "success":
            ok += 1
            print(
                f"[RETRAIN] {ticker} ok dir={payload.get('direction_accuracy_pct')}% "
                f"cache={payload.get('model_metadata', {}).get('loaded_from_cache')}"
            )
        else:
            print(f"[RETRAIN] {ticker} failed: {payload.get('error')}")

    print(f"[RETRAIN] Completed {ok}/{len(tickers)}. Models in {MODEL_DIR}")


if __name__ == "__main__":
    main()
