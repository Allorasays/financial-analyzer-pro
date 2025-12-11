#!/usr/bin/env python3
"""
Aggregate ML prediction logs to surface confidence/accuracy trends.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

LOG_PATH_DEFAULT = Path("logs/ml_metrics.jsonl")


def load_entries(path: Path):
    if not path.exists():
        print(f"[INFO] No metrics log found at {path}")
        return []
    entries = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def summarize(entries):
    if not entries:
        return {}

    by_ticker = defaultdict(list)
    for entry in entries:
        by_ticker[entry.get("ticker", "UNKNOWN")].append(entry)

    summary = {
        "total_records": len(entries),
        "tickers": {},
    }

    for ticker, items in by_ticker.items():
        confidences = [item.get("confidence") for item in items if item.get("confidence") is not None]
        accuracies = [item.get("model_accuracy") for item in items if item.get("model_accuracy") is not None]
        r2_scores = [item.get("r2_score") for item in items if item.get("r2_score") is not None]

        summary["tickers"][ticker] = {
            "records": len(items),
            "avg_confidence": round(mean(confidences), 4) if confidences else None,
            "avg_model_accuracy": round(mean(accuracies), 4) if accuracies else None,
            "avg_r2_score": round(mean(r2_scores), 4) if r2_scores else None,
            "min_confidence": round(min(confidences), 4) if confidences else None,
            "max_confidence": round(max(confidences), 4) if confidences else None,
        }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Analyze ML metrics log.")
    parser.add_argument("--log-path", default=str(LOG_PATH_DEFAULT), help="Path to metrics log file.")
    parser.add_argument("--output", help="Optional path to write JSON summary.")
    args = parser.parse_args()

    path = Path(args.log_path)
    entries = load_entries(path)
    summary = summarize(entries)

    if not summary:
        print("[INFO] No metrics to summarize yet.")
        return

    print(json.dumps(summary, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2))
        print(f"[INFO] Summary written to {output_path}")


if __name__ == "__main__":
    main()


