#!/usr/bin/env python3
"""Seed production ML predictions so the nightly validator has pending rows."""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

BASE = "https://moneta-backend-api.onrender.com"
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "AMD", "JPM", "SPY",
    "QQQ", "AVGO", "COST", "XOM", "UNH", "NFLX", "BAC", "WMT", "CRM", "ORCL",
    "IWM", "XLF", "DIS", "KO", "V", "MA", "INTC", "PLTR", "SOFI", "DIA",
]


def hit(ticker: str) -> dict:
    url = f"{BASE}/api/ml/predictions/{ticker}?prediction_days=5"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    ok = 0
    fail = 0
    for i, t in enumerate(TICKERS):
        try:
            body = hit(t)
            status = body.get("status")
            if status == "success":
                ok += 1
                print(f"OK {t} ver={body.get('model_metadata', {}).get('model_version')} next={body.get('next_day')}")
            else:
                fail += 1
                print(f"FAIL {t} status={status} err={body.get('error')}")
        except Exception as exc:
            fail += 1
            print(f"FAIL {t} {exc}")
        time.sleep(1.5)
    # pending summary
    try:
        with urllib.request.urlopen(f"{BASE}/api/prediction-pending?max_days_past=7", timeout=60) as resp:
            pending = json.loads(resp.read().decode("utf-8"))
            print("pending_count", pending.get("pending_count"))
    except Exception as exc:
        print("pending_error", exc)
    print(json.dumps({"ok": ok, "fail": fail, "total": len(TICKERS)}))
    return 0 if ok >= 15 else 1


if __name__ == "__main__":
    raise SystemExit(main())
