"""
Utility helpers for recording ML prediction metrics to disk so we can
monitor confidence and accuracy distribution over time.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

LOG_PATH = Path(os.getenv("ML_METRICS_LOG_PATH", "logs/ml_metrics.jsonl"))
_LOCK = threading.Lock()


def log_prediction_metrics(ticker: str, metrics: Dict[str, Any]) -> None:
    """
    Persist key ML metrics for later drift analysis.
    """
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "ticker": ticker.upper(),
        **metrics,
    }

    try:
        with _LOCK:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with LOG_PATH.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(log_entry) + "\n")
    except Exception as exc:
        # We intentionally keep this soft-fail; prediction endpoints must not break due to logging.
        print(f"[ML-METRICS] Failed to write metrics log: {exc}")


