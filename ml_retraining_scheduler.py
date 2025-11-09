#!/usr/bin/env python3
"""
Trigger proactive ML retraining by invoking the core prediction pipeline
for a basket of tickers and persisting the resulting models.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from proxy import get_ml_predictions

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN", "SPY"]
DEFAULT_DAYS_AHEAD = 30
OUTPUT_DIR = Path("data/model_cache")


def retrain_ticker(ticker: str, days_ahead: int) -> dict:
    print(f"[RETRAIN] Updating model for {ticker} ({days_ahead} days horizon)")
    result = get_ml_predictions(ticker, days_ahead)
    result["retrained_at"] = datetime.utcnow().isoformat()
    return result


def persist_model(ticker: str, payload: dict):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{ticker.upper()}_model.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"[RETRAIN] Cached {ticker} model to {path}")


def main():
    parser = argparse.ArgumentParser(description="Run proactive ML retraining.")
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

    for ticker in tickers:
        payload = retrain_ticker(ticker, args.days_ahead)
        persist_model(ticker, payload)

    print("[RETRAIN] Completed proactive retraining run.")


if __name__ == "__main__":
    main()

