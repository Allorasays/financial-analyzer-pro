#!/usr/bin/env python3
"""
Pre-deploy smoke checks for personal-use backend.

Usage:
  python test_pre_deploy_smoke.py
  SMOKE_LIVE=1 python test_pre_deploy_smoke.py   # also hit Render /api/financials/AAPL
"""

import os
import sys


def test_config_personal_use():
    from config import PERSONAL_USE_CONFIG

    assert PERSONAL_USE_CONFIG["enabled"] is True
    assert PERSONAL_USE_CONFIG["financials_cache_ttl"] >= 60


def test_imports():
    import comprehensive_financial_aggregator  # noqa: F401
    import fmp_service  # noqa: F401


def test_fmp_no_bundled_default():
    from fmp_service import FMPService

    svc = FMPService()
    env_key = os.getenv("FMP_API_KEY", "")
    assert svc.api_key in ("", env_key)


def test_aggregator_interface():
    from comprehensive_financial_aggregator import comprehensive_financial_aggregator

    assert hasattr(comprehensive_financial_aggregator, "get_comprehensive_financial_data")


def test_proxy_financials_helpers():
    import proxy

    assert hasattr(proxy, "_count_financial_fields")
    assert hasattr(proxy, "format_sentiment_for_android")


def test_live_render_financials():
    if os.getenv("SMOKE_LIVE") != "1":
        return
    import requests

    url = os.getenv("SMOKE_BASE_URL", "https://moneta-backend-api.onrender.com")
    resp = requests.get(f"{url.rstrip('/')}/api/financials/AAPL", timeout=120)
    assert resp.status_code in (200, 503), resp.text[:500]
    if resp.status_code == 200:
        body = resp.json()
        assert body.get("personal_use_only") is True
        assert body.get("ticker") == "AAPL"


def main():
    tests = [
        test_config_personal_use,
        test_imports,
        test_fmp_no_bundled_default,
        test_aggregator_interface,
        test_proxy_financials_helpers,
        test_live_render_financials,
    ]
    failures = []
    for test_fn in tests:
        name = test_fn.__name__
        try:
            test_fn()
            print(f"OK  {name}")
        except Exception as exc:
            failures.append((name, exc))
            print(f"FAIL {name}: {exc}")

    if failures:
        print(f"\n{len(failures)} check(s) failed.")
        sys.exit(1)
    print("\nAll smoke checks passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
