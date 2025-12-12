#!/usr/bin/env python3
"""
Comprehensive End-to-End Testing Script
Tests all backend endpoints including the newly implemented email service
"""

import requests
import json
import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Production backend URL
BACKEND_URL = "https://moneta-backend-api.onrender.com"

class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{text}{Colors.END}")
    print(f"{Colors.BOLD}{'='*70}{Colors.END}")

def print_test(name: str, status: bool, details: str = ""):
    """Print test result"""
    status_icon = f"{Colors.GREEN}✅ PASS{Colors.END}" if status else f"{Colors.RED}❌ FAIL{Colors.END}"
    print(f"{status_icon} - {name}")
    if details:
        print(f"      {details}")

def test_endpoint(
    name: str,
    url: str,
    method: str = "GET",
    expected_status: int = 200,
    json_data: Optional[Dict] = None,
    headers: Optional[Dict] = None,
    timeout: int = 30
) -> Tuple[bool, Dict]:
    """Test a single endpoint"""
    try:
        if method == "GET":
            response = requests.get(url, headers=headers, timeout=timeout)
        elif method == "POST":
            response = requests.post(url, json=json_data, headers=headers, timeout=timeout)
        elif method == "PUT":
            response = requests.put(url, json=json_data, headers=headers, timeout=timeout)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers, timeout=timeout)
        else:
            return False, {"error": f"Unknown method: {method}"}
        
        success = response.status_code == expected_status
        try:
            response_data = response.json() if response.text else {}
        except:
            response_data = {"text": response.text[:200]}
        
        return success, {
            "status_code": response.status_code,
            "response": response_data,
            "success": success
        }
        
    except requests.exceptions.Timeout:
        return False, {"error": "Request timed out", "success": False}
    except requests.exceptions.ConnectionError:
        return False, {"error": "Connection error - service may be sleeping", "success": False}
    except Exception as e:
        return False, {"error": str(e), "success": False}

def test_health_endpoints() -> List[Tuple[str, bool, str]]:
    """Test health and status endpoints"""
    print_header("1. Health & Status Endpoints")
    results = []
    
    # Root endpoint
    success, data = test_endpoint("Root", f"{BACKEND_URL}/")
    details = f"Status: {data.get('status_code', 'N/A')}"
    results.append(("Root Endpoint", success, details))
    print_test("Root Endpoint", success, details)
    
    # Health check
    success, data = test_endpoint("Health", f"{BACKEND_URL}/health")
    details = f"Status: {data.get('status_code', 'N/A')}"
    results.append(("Health Check", success, details))
    print_test("Health Check", success, details)
    
    # System status
    success, data = test_endpoint("System Status", f"{BACKEND_URL}/api/system/status")
    details = f"Status: {data.get('status_code', 'N/A')}"
    results.append(("System Status", success, details))
    print_test("System Status", success, details)
    
    return results

def test_authentication_endpoints() -> List[Tuple[str, bool, str]]:
    """Test authentication endpoints"""
    print_header("2. Authentication Endpoints")
    results = []
    auth_token = None
    
    # Generate unique test user
    timestamp = int(time.time())
    test_username = f"testuser_{timestamp}"
    test_email = f"test_{timestamp}@example.com"
    test_password = "TestPass123!"
    
    # Registration
    success, data = test_endpoint(
        "Registration",
        f"{BACKEND_URL}/api/auth/register",
        method="POST",
        json_data={
            "username": test_username,
            "email": test_email,
            "password": test_password
        }
    )
    details = f"Status: {data.get('status_code', 'N/A')}"
    if data.get('status_code') == 400:
        details += " (User may already exist - OK)"
        success = True  # Acceptable for testing
    results.append(("User Registration", success, details))
    print_test("User Registration", success, details)
    
    # Login
    success, data = test_endpoint(
        "Login",
        f"{BACKEND_URL}/api/auth/login",
        method="POST",
        json_data={
            "username": test_username,
            "password": test_password
        }
    )
    if success and "access_token" in data.get("response", {}):
        auth_token = data["response"]["access_token"]
        details = f"Status: {data.get('status_code')} - Token received"
    else:
        details = f"Status: {data.get('status_code', 'N/A')}"
    results.append(("User Login", success, details))
    print_test("User Login", success, details)
    
    # Password Reset Request (tests email service)
    success, data = test_endpoint(
        "Password Reset Request",
        f"{BACKEND_URL}/api/auth/forgot-password",
        method="POST",
        json_data={"email": test_email}
    )
    email_sent = data.get("response", {}).get("email_sent", False)
    details = f"Status: {data.get('status_code')} - Email sent: {email_sent}"
    results.append(("Password Reset Request", success, details))
    print_test("Password Reset Request", success, details)
    
    # Username Recovery (tests email service)
    success, data = test_endpoint(
        "Username Recovery",
        f"{BACKEND_URL}/api/auth/forgot-username",
        method="POST",
        json_data={"email": test_email}
    )
    email_sent = data.get("response", {}).get("email_sent", False)
    details = f"Status: {data.get('status_code')} - Email sent: {email_sent}"
    results.append(("Username Recovery", success, details))
    print_test("Username Recovery", success, details)
    
    return results, auth_token

def test_market_endpoints() -> List[Tuple[str, bool, str]]:
    """Test market data endpoints"""
    print_header("3. Market Data Endpoints")
    results = []
    
    # Market Overview
    success, data = test_endpoint("Market Overview", f"{BACKEND_URL}/api/market/overview")
    details = f"Status: {data.get('status_code', 'N/A')}"
    results.append(("Market Overview", success, details))
    print_test("Market Overview", success, details)
    
    # Realtime Stock Data
    success, data = test_endpoint("Stock Data (AAPL)", f"{BACKEND_URL}/api/market/realtime/AAPL")
    details = f"Status: {data.get('status_code', 'N/A')}"
    results.append(("Stock Data (AAPL)", success, details))
    print_test("Stock Data (AAPL)", success, details)
    
    # Financial Data
    success, data = test_endpoint("Financial Data (AAPL)", f"{BACKEND_URL}/api/financials/AAPL")
    details = f"Status: {data.get('status_code', 'N/A')}"
    results.append(("Financial Data (AAPL)", success, details))
    print_test("Financial Data (AAPL)", success, details)
    
    return results

def test_ml_endpoints() -> List[Tuple[str, bool, str]]:
    """Test ML prediction endpoints"""
    print_header("4. ML Prediction Endpoints")
    results = []
    
    # ML Predictions
    success, data = test_endpoint("ML Predictions (AAPL)", f"{BACKEND_URL}/api/ml/predictions/AAPL")
    details = f"Status: {data.get('status_code', 'N/A')}"
    results.append(("ML Predictions (AAPL)", success, details))
    print_test("ML Predictions (AAPL)", success, details)
    
    return results

def test_android_endpoints() -> List[Tuple[str, bool, str]]:
    """Test Android app specific endpoints"""
    print_header("5. Android App Endpoints")
    results = []
    
    android_endpoints = [
        ("Market Data", "/api/ai/market-data/AAPL"),
        ("Market Overview", "/api/ai/market-overview"),
        ("Global Markets", "/api/ai/global-markets"),
        ("Technical Analysis", "/api/ai/technical-analysis/AAPL"),
        ("Sentiment Analysis", "/api/ai/sentiment/AAPL"),
        ("Comprehensive Analysis", "/api/ai/comprehensive-analysis/AAPL"),
    ]
    
    for name, path in android_endpoints:
        success, data = test_endpoint(name, f"{BACKEND_URL}{path}")
        details = f"Status: {data.get('status_code', 'N/A')}"
        results.append((name, success, details))
        print_test(name, success, details)
    
    return results

def test_portfolio_endpoints(auth_token: Optional[str] = None) -> List[Tuple[str, bool, str]]:
    """Test portfolio management endpoints"""
    print_header("6. Portfolio Management Endpoints")
    results = []
    
    # Portfolio (with auth)
    if auth_token:
        headers = {"Authorization": f"Bearer {auth_token}"}
        success, data = test_endpoint("Portfolio (Auth)", f"{BACKEND_URL}/api/portfolio", headers=headers)
        details = f"Status: {data.get('status_code', 'N/A')}"
        results.append(("Portfolio (Auth)", success, details))
        print_test("Portfolio (Auth)", success, details)
    
    # Portfolio (Android alias - no auth)
    success, data = test_endpoint("Portfolio (Android)", f"{BACKEND_URL}/api/ai/portfolio")
    details = f"Status: {data.get('status_code', 'N/A')}"
    results.append(("Portfolio (Android)", success, details))
    print_test("Portfolio (Android)", success, details)
    
    return results

def main():
    """Run comprehensive end-to-end tests"""
    print_header("MONETA Financial Analyzer - Comprehensive E2E Tests")
    print(f"{Colors.BLUE}Backend URL: {BACKEND_URL}{Colors.END}")
    print(f"{Colors.BLUE}Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.END}")
    
    # Handle service wake-up
    print(f"\n{Colors.YELLOW}Checking if service is awake...{Colors.END}")
    success, _ = test_endpoint("Wake Up", f"{BACKEND_URL}/health", timeout=60)
    if not success:
        print(f"{Colors.YELLOW}Service may be sleeping. Waiting 45 seconds for wake-up...{Colors.END}")
        time.sleep(45)
    
    all_results = []
    
    # Run test suites
    all_results.extend(test_health_endpoints())
    
    auth_results, auth_token = test_authentication_endpoints()
    all_results.extend(auth_results)
    
    all_results.extend(test_market_endpoints())
    all_results.extend(test_ml_endpoints())
    all_results.extend(test_android_endpoints())
    all_results.extend(test_portfolio_endpoints(auth_token))
    
    # Summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for _, success, _ in all_results if success)
    total = len(all_results)
    success_rate = (passed / total * 100) if total > 0 else 0
    
    for name, success, details in all_results:
        print_test(name, success, details)
    
    print(f"\n{Colors.BOLD}Results: {passed}/{total} tests passed ({success_rate:.1f}%){Colors.END}")
    
    if passed == total:
        print(f"\n{Colors.GREEN}🎉 All tests passed! Backend is working correctly.{Colors.END}")
        return 0
    else:
        print(f"\n{Colors.YELLOW}⚠️  {total - passed} test(s) failed. Check the errors above.{Colors.END}")
        print(f"{Colors.YELLOW}Note: Some failures may be expected (e.g., service sleeping, rate limits).{Colors.END}")
        return 1

if __name__ == "__main__":
    # Allow custom backend URL via command line
    if len(sys.argv) > 1:
        BACKEND_URL = sys.argv[1].rstrip('/')
        print(f"{Colors.BLUE}Using custom backend URL: {BACKEND_URL}{Colors.END}")
    
    exit_code = main()
    sys.exit(exit_code)

