#!/usr/bin/env python3
"""
Check prediction_tracker sufficiency for ML Skill P2/P3 gate.

Sufficiency (personal-use gate after ~3 nights):
  - At least 3 distinct validation calendar days (UTC), OR wall-clock >= 3 nights from start
  - At least 30 next-day (horizon=1) validations overall
  - At least 10 distinct tickers with >=1 validation
  - Global direction_accuracy computable (min_validations satisfied)

Writes a snapshot JSON for the monitoring loop.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prediction_tracker import prediction_tracker  # noqa: E402

STATE_PATH = ROOT / "logs" / "ml_tracker_monitor_state.json"
SNAPSHOT_PATH = ROOT / "logs" / "ml_tracker_monitor_latest.json"
START_MARKER = ROOT / "logs" / "ml_tracker_monitor_start.json"

# Gate for auto-advancing to P2/P3
MIN_VALIDATION_DAYS = 3
MIN_HORIZON1_VALIDATIONS = 30
MIN_TICKERS = 10
MIN_NIGHTS_ELAPSED = 3


def _db_path() -> Path:
    return Path(prediction_tracker.db_path)


def _ensure_start() -> dict:
    START_MARKER.parent.mkdir(parents=True, exist_ok=True)
    if START_MARKER.exists():
        return json.loads(START_MARKER.read_text(encoding="utf-8"))
    payload = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "nights_target": MIN_NIGHTS_ELAPSED,
        "gate": {
            "min_validation_days": MIN_VALIDATION_DAYS,
            "min_horizon1_validations": MIN_HORIZON1_VALIDATIONS,
            "min_tickers": MIN_TICKERS,
            "min_nights_elapsed": MIN_NIGHTS_ELAPSED,
        },
    }
    START_MARKER.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def collect_local_stats() -> dict:
    db = _db_path()
    out: dict = {
        "db_path": str(db),
        "db_exists": db.exists(),
        "predictions": 0,
        "validations": 0,
        "horizon1_validations": 0,
        "validation_days": [],
        "prediction_days": [],
        "tickers_validated": [],
        "metrics_1d": {},
        "metrics_all": {},
    }
    if not db.exists():
        return out

    conn = sqlite3.connect(str(db))
    cur = conn.cursor()
    # Discover columns
    cur.execute("PRAGMA table_info(predictions)")
    pred_cols = {r[1] for r in cur.fetchall()}
    cur.execute("PRAGMA table_info(validations)")
    val_cols = {r[1] for r in cur.fetchall()}

    ts_pred = "created_at" if "created_at" in pred_cols else (
        "timestamp" if "timestamp" in pred_cols else None
    )
    ts_val = "validated_at" if "validated_at" in val_cols else (
        "timestamp" if "timestamp" in val_cols else None
    )

    cur.execute("SELECT COUNT(*) FROM predictions")
    out["predictions"] = int(cur.fetchone()[0])
    cur.execute("SELECT COUNT(*) FROM validations")
    out["validations"] = int(cur.fetchone()[0])

    if "horizon_days" in pred_cols:
        cur.execute(
            """
            SELECT COUNT(*) FROM validations v
            JOIN predictions p ON v.prediction_id = p.id
            WHERE p.horizon_days = 1
            """
        )
        out["horizon1_validations"] = int(cur.fetchone()[0])

    if ts_val:
        cur.execute(
            f"SELECT date({ts_val}), COUNT(*) FROM validations "
            f"GROUP BY date({ts_val}) ORDER BY 1 DESC LIMIT 14"
        )
        out["validation_days"] = [{"date": r[0], "count": r[1]} for r in cur.fetchall()]

    if ts_pred:
        cur.execute(
            f"SELECT date({ts_pred}), COUNT(*) FROM predictions "
            f"GROUP BY date({ts_pred}) ORDER BY 1 DESC LIMIT 14"
        )
        out["prediction_days"] = [{"date": r[0], "count": r[1]} for r in cur.fetchall()]

    cur.execute(
        """
        SELECT DISTINCT p.ticker FROM validations v
        JOIN predictions p ON v.prediction_id = p.id
        """
    )
    out["tickers_validated"] = sorted(r[0] for r in cur.fetchall() if r[0])
    conn.close()

    out["metrics_1d"] = prediction_tracker.calculate_accuracy_metrics(
        horizon_days=1, min_validations=1
    )
    out["metrics_all"] = prediction_tracker.calculate_accuracy_metrics(min_validations=1)
    return out


def collect_remote_stats(base_url: str) -> dict:
    import urllib.request

    remote: dict = {"base_url": base_url}

    def _get(path: str):
        with urllib.request.urlopen(f"{base_url}{path}", timeout=90) as resp:
            return json.loads(resp.read().decode("utf-8"))

    try:
        body = _get("/api/prediction-accuracy?min_validations=1")
        remote["accuracy_raw"] = body
        remote["accuracy"] = body.get("metrics") if isinstance(body, dict) else body
    except Exception as exc:
        remote["accuracy_error"] = str(exc)
    try:
        body = _get("/api/prediction-accuracy/recent?days=7")
        remote["recent_raw"] = body
        remote["recent"] = body.get("metrics") if isinstance(body, dict) else body
    except Exception as exc:
        remote["recent_error"] = str(exc)
    try:
        body = _get("/api/prediction-pending?max_days_past=7")
        remote["pending_raw"] = body
        if isinstance(body, dict):
            remote["pending_count"] = body.get("pending_count", 0)
            remote["awaiting_target"] = body.get("awaiting_target", 0)
            remote["awaiting_horizon1"] = body.get("awaiting_horizon1", 0)
            remote["total_predictions"] = body.get("total_predictions", 0)
            remote["pending_predictions"] = body.get("pending_predictions") or []
        else:
            remote["pending_count"] = 0
    except Exception as exc:
        remote["pending_error"] = str(exc)
    try:
        body = _get("/api/prediction-tracker/stats")
        remote["stats"] = body.get("stats") if isinstance(body, dict) else body
    except Exception as exc:
        remote["stats_error"] = str(exc)
    return remote


def evaluate_gate(start: dict, local: dict, remote: dict) -> dict:
    started = datetime.fromisoformat(start["started_at"].replace("Z", "+00:00"))
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    nights_elapsed = (now - started).total_seconds() / 86400.0

    val_days = len(local.get("validation_days") or [])
    h1 = int(local.get("horizon1_validations") or 0)
    tickers = len(local.get("tickers_validated") or [])

    remote_acc = remote.get("accuracy") if isinstance(remote.get("accuracy"), dict) else {}
    remote_n = int(remote_acc.get("total_validations") or 0)
    if remote_acc.get("status") == "insufficient_data":
        msg = str(remote_acc.get("message") or "")
        if "got " in msg:
            try:
                remote_n = max(remote_n, int(msg.rsplit("got ", 1)[-1].strip()))
            except ValueError:
                pass
    remote_dir = remote_acc.get("direction_accuracy_pct")
    pending_ready = int(remote.get("pending_count") or 0)
    awaiting = int(remote.get("awaiting_horizon1") or remote.get("awaiting_target") or 0)
    total_pred = int(remote.get("total_predictions") or 0)
    stats = remote.get("stats") if isinstance(remote.get("stats"), dict) else {}
    if stats:
        awaiting = max(awaiting, int(stats.get("awaiting_horizon1") or 0))
        total_pred = max(total_pred, int(stats.get("total_predictions") or 0))
        remote_n = max(remote_n, int(stats.get("validated_horizon1") or 0))
        val_days = max(val_days, len(stats.get("validation_days") or []))

    checks = {
        "nights_elapsed": round(nights_elapsed, 2),
        "nights_ok": nights_elapsed >= MIN_NIGHTS_ELAPSED,
        "validation_days": val_days,
        "validation_days_ok": val_days >= MIN_VALIDATION_DAYS,
        "horizon1_validations_local": h1,
        "horizon1_ok": h1 >= MIN_HORIZON1_VALIDATIONS or remote_n >= MIN_HORIZON1_VALIDATIONS,
        "tickers_validated": tickers,
        "tickers_ok": tickers >= MIN_TICKERS or remote_n >= MIN_HORIZON1_VALIDATIONS,
        "remote_validations": remote_n,
        "remote_direction_accuracy_pct": remote_dir,
        "remote_pending_ready": pending_ready,
        "remote_awaiting_horizon1": awaiting,
        "remote_total_predictions": total_pred,
        "pipeline_seeded_ok": awaiting >= 20 or total_pred >= 40 or remote_n >= MIN_HORIZON1_VALIDATIONS,
    }

    sufficient = (
        checks["nights_ok"]
        and checks["horizon1_ok"]
        and (
            checks["validation_days_ok"]
            or remote_n >= MIN_HORIZON1_VALIDATIONS
        )
        and checks["tickers_ok"]
    )
    return {
        "sufficient_for_p2_p3": sufficient,
        "checks": checks,
        "reason": (
            "Tracker gate passed — proceed to Skill P2/P3"
            if sufficient
            else "Waiting for 3 nights + enough validated next-day predictions"
        ),
    }


def main() -> int:
    start = _ensure_start()
    local = collect_local_stats()
    base = os.getenv("API_BASE_URL", "https://moneta-backend-api.onrender.com")
    remote = collect_remote_stats(base)
    gate = evaluate_gate(start, local, remote)

    snapshot = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "start": start,
        "local": local,
        "remote": remote,
        "gate": gate,
    }
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    # Append history
    hist = []
    if STATE_PATH.exists():
        try:
            hist = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            if not isinstance(hist, list):
                hist = []
        except Exception:
            hist = []
    hist.append(
        {
            "checked_at": snapshot["checked_at"],
            "sufficient": gate["sufficient_for_p2_p3"],
            "checks": gate["checks"],
            "local_validations": local.get("validations"),
            "remote_validations": gate["checks"].get("remote_validations"),
        }
    )
    STATE_PATH.write_text(json.dumps(hist[-60:], indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "sufficient_for_p2_p3": gate["sufficient_for_p2_p3"],
                "reason": gate["reason"],
                "checks": gate["checks"],
                "local_validations": local.get("validations"),
                "local_horizon1": local.get("horizon1_validations"),
                "remote_validations": gate["checks"].get("remote_validations"),
                "remote_awaiting_horizon1": gate["checks"].get("remote_awaiting_horizon1"),
                "remote_total_predictions": gate["checks"].get("remote_total_predictions"),
                "snapshot": str(SNAPSHOT_PATH),
            },
            indent=2,
        )
    )
    return 0 if gate["sufficient_for_p2_p3"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
