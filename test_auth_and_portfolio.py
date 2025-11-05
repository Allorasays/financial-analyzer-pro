"""
Test authentication and portfolio endpoints on Render backend
"""
import requests
import json
import time

BASE_URL = "https://moneta-backend-api.onrender.com"

def test_endpoint(method, path, data=None, headers=None):
    """Test an endpoint"""
    try:
        url = f"{BASE_URL}{path}"
        
        if method == "GET":
            response = requests.get(url, headers=headers, timeout=30)
        elif method == "POST":
            response = requests.post(url, json=data, headers=headers, timeout=30)
        
        return response.status_code, response.json() if response.text else {}
    except requests.exceptions.Timeout:
        return -1, {"error": "Timeout"}
    except requests.exceptions.ConnectionError:
        return -1, {"error": "Connection Error"}
    except Exception as e:
        return -1, {"error": str(e)}

def main():
    print("=" * 70)
    print("Testing Authentication and Portfolio Endpoints")
    print("=" * 70)
    print(f"Backend URL: {BASE_URL}\n")
    
    # Test 1: Health check (wake up service)
    print("[1] Testing health endpoint (wake up service)...")
    status, response = test_endpoint("GET", "/health")
    print(f"    Status: {status}")
    if status == 200:
        print("    [OK] Service is awake")
    else:
        print("    [WARNING] Service may be sleeping, waiting 30 seconds...")
        time.sleep(30)
    
    # Test 2: Register endpoint
    print("\n[2] Testing registration endpoint...")
    register_data = {
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpass123"
    }
    status, response = test_endpoint("POST", "/api/auth/register", register_data)
    print(f"    Status: {status}")
    if status == 200:
        print("    [OK] Registration successful")
        token = response.get("access_token")
        print(f"    Token received: {token[:20] if token else 'None'}...")
    elif status == 400:
        print("    [INFO] User already exists (this is OK)")
    elif status == 404:
        print("    [ERROR] Endpoint not found!")
    else:
        print(f"    [ERROR] Registration failed: {response}")
    
    # Test 3: Login endpoint
    print("\n[3] Testing login endpoint...")
    login_data = {
        "username": "testuser",
        "password": "testpass123"
    }
    status, response = test_endpoint("POST", "/api/auth/login", login_data)
    print(f"    Status: {status}")
    if status == 200:
        print("    [OK] Login successful")
        token = response.get("access_token")
        print(f"    Token received: {token[:20] if token else 'None'}...")
    elif status == 401:
        print("    [ERROR] Invalid credentials")
        token = None
    elif status == 404:
        print("    [ERROR] Endpoint not found!")
        token = None
    else:
        print(f"    [ERROR] Login failed: {response}")
        token = None
    
    # Test 4: Portfolio endpoint (with auth)
    if token:
        print("\n[4] Testing portfolio endpoint (with authentication)...")
        headers = {"Authorization": f"Bearer {token}"}
        status, response = test_endpoint("GET", "/api/portfolio", headers=headers)
        print(f"    Status: {status}")
        if status == 200:
            print("    [OK] Portfolio endpoint working")
            print(f"    Portfolio: {json.dumps(response, indent=2)[:200]}")
        elif status == 401:
            print("    [ERROR] Authentication failed")
        elif status == 404:
            print("    [ERROR] Endpoint not found!")
        else:
            print(f"    [ERROR] Portfolio request failed: {response}")
    else:
        print("\n[4] Skipping portfolio test (no token)")
    
    # Test 5: Android portfolio alias (no auth required)
    print("\n[5] Testing Android portfolio alias (no auth)...")
    status, response = test_endpoint("GET", "/api/ai/portfolio")
    print(f"    Status: {status}")
    if status == 200:
        print("    [OK] Android portfolio alias working")
        print(f"    Response: {json.dumps(response, indent=2)[:200]}")
    elif status == 404:
        print("    [ERROR] Endpoint not found!")
    else:
        print(f"    [ERROR] Request failed: {response}")
    
    # Test 6: Add to portfolio (with auth)
    if token:
        print("\n[6] Testing add to portfolio endpoint (with authentication)...")
        headers = {"Authorization": f"Bearer {token}"}
        add_data = {
            "ticker": "AAPL",
            "shares": 10.0,
            "avg_price": 150.0
        }
        status, response = test_endpoint("POST", "/api/portfolio/add", add_data, headers)
        print(f"    Status: {status}")
        if status == 200:
            print("    [OK] Add to portfolio working")
        elif status == 401:
            print("    [ERROR] Authentication failed")
        elif status == 404:
            print("    [ERROR] Endpoint not found!")
        else:
            print(f"    [ERROR] Add to portfolio failed: {response}")
    else:
        print("\n[6] Skipping add to portfolio test (no token)")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nIssues Found:")
    print("1. Check if endpoints return 404 (service sleeping or not deployed)")
    print("2. Check if database is initialized (SQLite file on Render)")
    print("3. Check if SECRET_KEY is set for JWT tokens")
    print("4. Check service logs in Render dashboard")

if __name__ == "__main__":
    main()

