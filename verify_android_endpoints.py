"""
Verify all Android app endpoints exist and work on Render backend
"""
import requests
import json
from typing import Dict, List, Tuple

BASE_URL = "https://moneta-backend-api.onrender.com"

# All endpoints the Android app uses (from ApiService.kt)
ANDROID_ENDPOINTS = [
    ("GET", "/api/ai/market-data/{ticker}", {"ticker": "AAPL"}),
    ("GET", "/api/ai/market-overview", {}),
    ("GET", "/api/ai/global-markets", {}),
    ("GET", "/api/ai/batch-market-data", {"tickers": "AAPL,TSLA,MSFT"}),
    ("GET", "/api/ai/portfolio", {}),
    ("GET", "/api/ai/technical-analysis/{ticker}", {"ticker": "AAPL"}),
    ("GET", "/api/ai/risk-analysis/{ticker}", {"ticker": "AAPL"}),
    ("GET", "/api/ml/predictions/{ticker}", {"ticker": "AAPL", "prediction_days": 30}),
    ("GET", "/api/ai/sentiment/{ticker}", {"ticker": "AAPL"}),
    ("GET", "/api/ai/comprehensive-analysis/{ticker}", {"ticker": "AAPL", "prediction_days": 30}),
    ("GET", "/api/ai/status", {}),
    ("GET", "/api/ai/health", {}),
]

def test_endpoint(method: str, path: str, params: Dict) -> Tuple[bool, int, str]:
    """Test a single endpoint"""
    try:
        # Replace path parameters
        url = path
        for key, value in params.items():
            if key in ["ticker"]:  # Path parameters
                url = url.replace(f"{{{key}}}", str(value))
                params_copy = params.copy()
                params_copy.pop(key)
                params = params_copy
        
        full_url = f"{BASE_URL}{url}"
        
        # Handle query parameters
        if method == "GET":
            response = requests.get(full_url, params=params, timeout=15)
        else:
            response = requests.post(full_url, json=params, timeout=15)
        
        status = response.status_code
        if status == 200:
            try:
                data = response.json()
                return (True, status, f"OK - {json.dumps(data)[:100]}")
            except:
                return (True, status, f"OK - {response.text[:100]}")
        elif status == 404:
            return (False, status, "NOT FOUND - Endpoint missing")
        elif status == 429:
            return (False, status, "RATE LIMITED")
        elif status == 500:
            return (False, status, f"ERROR - {response.text[:100]}")
        else:
            return (False, status, f"Status {status}")
            
    except requests.exceptions.Timeout:
        return (False, -1, "TIMEOUT - Service may be sleeping")
    except requests.exceptions.ConnectionError:
        return (False, -1, "CONNECTION ERROR - Service may be down")
    except Exception as e:
        return (False, -1, f"ERROR - {str(e)[:100]}")

def main():
    print("=" * 70)
    print("Android App Endpoint Verification")
    print("=" * 70)
    print(f"Testing against: {BASE_URL}\n")
    
    results = []
    for method, path, params in ANDROID_ENDPOINTS:
        print(f"[TEST] {method} {path}")
        success, status, message = test_endpoint(method, path, params)
        status_icon = "[OK]" if success else "[FAIL]"
        print(f"   {status_icon} Status {status}: {message}\n")
        results.append((path, success, status, message))
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    working = sum(1 for _, success, _, _ in results if success)
    total = len(results)
    
    print(f"Working Endpoints: {working}/{total}")
    print(f"\nDetails:")
    
    for path, success, status, message in results:
        status_str = "OK" if success else "FAILED"
        print(f"  {status_str:6} - {path}")
    
    if working == total:
        print("\n[SUCCESS] All endpoints are working!")
    else:
        print(f"\n[WARNING] {total - working} endpoint(s) need attention")
        print("\nMissing or broken endpoints:")
        for path, success, status, message in results:
            if not success:
                print(f"  - {path} (Status: {status}) - {message}")

if __name__ == "__main__":
    main()



