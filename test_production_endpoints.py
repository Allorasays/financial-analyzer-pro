#!/usr/bin/env python3
"""
Production Endpoint Testing Script
Tests the Render backend API endpoints to verify they're working correctly.
"""

import requests
import json
import sys
from datetime import datetime

# Production backend URL - Update this if your service has a different name
BACKEND_URL = "https://moneta-backend-api.onrender.com"

def test_endpoint(name, url, method="GET", expected_status=200, json_data=None):
    """Test a single endpoint"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"URL: {url}")
    print(f"{'='*60}")
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=30)
        elif method == "POST":
            response = requests.post(url, json=json_data, timeout=30)
        else:
            print(f"❌ Unknown method: {method}")
            return False
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == expected_status:
            print(f"✅ PASS - Status {response.status_code}")
            
            # Try to parse JSON response
            try:
                data = response.json()
                print(f"Response Preview: {json.dumps(data, indent=2)[:200]}...")
            except:
                print(f"Response: {response.text[:200]}...")
            
            return True
        else:
            print(f"❌ FAIL - Expected {expected_status}, got {response.status_code}")
            print(f"Response: {response.text[:500]}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"❌ FAIL - Request timed out")
        return False
    except requests.exceptions.ConnectionError:
        print(f"❌ FAIL - Could not connect to server")
        print(f"   Check if the backend service is running at {BACKEND_URL}")
        return False
    except Exception as e:
        print(f"❌ FAIL - Error: {str(e)}")
        return False

def main():
    """Run all endpoint tests"""
    print("="*60)
    print("MONETA Financial Analyzer - Production Endpoint Tests")
    print("="*60)
    print(f"Backend URL: {BACKEND_URL}")
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    
    # Test 1: Root endpoint
    results.append((
        "Root Endpoint",
        test_endpoint("Root Endpoint", f"{BACKEND_URL}/")
    ))
    
    # Test 2: Health check
    results.append((
        "Health Check",
        test_endpoint("Health Check", f"{BACKEND_URL}/health", expected_status=200)
    ))
    
    # Test 3: System status
    results.append((
        "System Status",
        test_endpoint("System Status", f"{BACKEND_URL}/api/system/status", expected_status=200)
    ))
    
    # Test 4: Market overview
    results.append((
        "Market Overview",
        test_endpoint("Market Overview", f"{BACKEND_URL}/api/market/overview", expected_status=200)
    ))
    
    # Test 5: Stock data (AAPL)
    results.append((
        "Stock Data (AAPL)",
        test_endpoint("Stock Data", f"{BACKEND_URL}/api/market/realtime/AAPL", expected_status=200)
    ))
    
    # Test 6: ML Predictions (AAPL)
    results.append((
        "ML Predictions (AAPL)",
        test_endpoint("ML Predictions", f"{BACKEND_URL}/api/ml/predictions/AAPL", expected_status=200)
    ))
    
    # Test 7: Technical Analysis (AAPL)
    results.append((
        "Technical Analysis (AAPL)",
        test_endpoint("Technical Analysis", f"{BACKEND_URL}/api/technical/AAPL", expected_status=200)
    ))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Production backend is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    # Allow custom backend URL via command line
    if len(sys.argv) > 1:
        BACKEND_URL = sys.argv[1].rstrip('/')
        print(f"Using custom backend URL: {BACKEND_URL}")
    
    exit_code = main()
    sys.exit(exit_code)



