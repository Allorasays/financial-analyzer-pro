"""
ML model persistence for Skill (P1).
Stores sklearn VotingRegressor + scaler as joblib with a 24h freshness window.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

MODEL_DIR = Path("data/model_cache")
MODEL_VERSION = "2.4.0-skill"
FRESHNESS_HOURS = 24


def _artifact_paths(ticker: str) -> Tuple[Path, Path]:
    sym = ticker.upper()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return MODEL_DIR / f"{sym}_skill.joblib", MODEL_DIR / f"{sym}_skill_meta.json"


def save_model(
    ticker: str,
    model: Any,
    scaler: Any,
    features: List[str],
    metrics: Optional[Dict[str, Any]] = None,
) -> Path:
    """Persist model + scaler + feature list. Returns joblib path."""
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("joblib is required for model persistence") from exc

    joblib_path, meta_path = _artifact_paths(ticker)
    payload = {
        "model": model,
        "scaler": scaler,
        "features": list(features),
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker.upper(),
        "target": "next_day_return",
    }
    joblib.dump(payload, joblib_path)

    meta = {
        "ticker": ticker.upper(),
        "model_version": MODEL_VERSION,
        "trained_at": payload["trained_at"],
        "features_count": len(features),
        "features": features[:20],
        "target": "next_day_return",
        "metrics": metrics or {},
        "joblib_path": str(joblib_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Saved skill model for %s → %s", ticker.upper(), joblib_path)
    return joblib_path


def load_fresh_model(
    ticker: str,
    max_age_hours: float = FRESHNESS_HOURS,
) -> Optional[Dict[str, Any]]:
    """
    Load cached model if present and fresher than max_age_hours.
    Returns dict with model, scaler, features, trained_at, model_version — or None.
    """
    joblib_path, meta_path = _artifact_paths(ticker)
    if not joblib_path.exists():
        return None

    try:
        import joblib
        payload = joblib.load(joblib_path)
    except Exception as exc:
        logger.warning("Failed to load skill model for %s: %s", ticker, exc)
        return None

    trained_at_raw = payload.get("trained_at")
    if not trained_at_raw:
        return None
    try:
        trained_at = datetime.fromisoformat(trained_at_raw.replace("Z", "+00:00"))
        if trained_at.tzinfo is None:
            trained_at = trained_at.replace(tzinfo=timezone.utc)
    except Exception:
        return None

    age_hours = (datetime.now(timezone.utc) - trained_at).total_seconds() / 3600.0
    if age_hours > max_age_hours:
        logger.info(
            "Skill model for %s is stale (%.1fh > %sh) — will retrain",
            ticker.upper(),
            age_hours,
            max_age_hours,
        )
        return None

    if payload.get("model_version") != MODEL_VERSION:
        logger.info(
            "Skill model version mismatch for %s (%s != %s) — will retrain",
            ticker.upper(),
            payload.get("model_version"),
            MODEL_VERSION,
        )
        return None

    if not payload.get("model") or not payload.get("scaler") or not payload.get("features"):
        return None

    payload["age_hours"] = round(age_hours, 2)
    payload["loaded_from_cache"] = True
    if meta_path.exists():
        try:
            payload["meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            payload["meta"] = {}
    logger.info(
        "Loaded fresh skill model for %s (age %.1fh)",
        ticker.upper(),
        age_hours,
    )
    return payload


def append_skill_metrics(ticker: str, metrics: Dict[str, Any]) -> None:
    """Append walk-forward / holdout skill metrics to JSONL for offline review."""
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "ml_skill_metrics.jsonl"
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker.upper(),
        "model_version": MODEL_VERSION,
        **metrics,
    }
    try:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
    except Exception as exc:
        logger.warning("Failed to write skill metrics for %s: %s", ticker, exc)
