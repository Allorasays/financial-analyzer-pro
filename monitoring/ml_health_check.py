#!/usr/bin/env python3
"""
Helper CLI to run the ML accuracy evaluation on demand or at a fixed interval.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ml_accuracy_evaluation import DEFAULT_BASE_URL, DEFAULT_TICKERS, main as run_evaluation


def run_once(base_url: str, tickers):
    print(f"[ML-HEALTH] Running evaluation against {base_url} for {len(tickers)} tickers")
    run_evaluation(base_url=base_url, tickers=tickers)


def main():
    parser = argparse.ArgumentParser(description="Run ML health checks on a cadence.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Backend base URL to evaluate.")
    parser.add_argument(
        "--tickers",
        default=",".join(DEFAULT_TICKERS),
        help="Comma separated list of tickers to test.",
    )
    parser.add_argument(
        "--interval-hours",
        type=float,
        help="If provided, reruns the evaluation every N hours.",
    )
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    interval_hours = args.interval_hours

    while True:
        start_time = datetime.now()
        run_once(args.base_url, tickers)

        if not interval_hours:
            break

        next_run = start_time + timedelta(hours=interval_hours)
        sleep_seconds = max(0, (next_run - datetime.now()).total_seconds())
        print(f"[ML-HEALTH] Next run scheduled at {next_run.isoformat()}")
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()

