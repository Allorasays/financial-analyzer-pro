"""
Test portfolio management endpoints on Render backend
"""
import requests
import json

BASE_URL = "https://moneta-backend-api.onrender.com"

def test_health():
    """Test backend health"""
    print("[TEST] Testing backend health...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"   [ERROR] {e}")
        return False

def test_portfolio_no_auth():
    """Test portfolio endpoint without authentication"""
    print("\n[TEST] Testing /api/portfolio (no auth)...")
    try:
        response = requests.get(f"{BASE_URL}/api/portfolio", timeout=10)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.text[:200]}")
        return response.status_code
    except Exception as e:
        print(f"   [ERROR] {e}")
        return None

def test_ai_portfolio_alias():
    """Test Android compatibility alias"""
    print("\n[TEST] Testing /api/ai/portfolio (Android alias)...")
    try:
        response = requests.get(f"{BASE_URL}/api/ai/portfolio", timeout=10)
        print(f"   Status: {response.status_code}")
        data = response.json()
        print(f"   Response: {json.dumps(data, indent=2)[:300]}")
        return response.status_code, data
    except Exception as e:
        print(f"   [ERROR] {e}")
        return None, None

def test_system_status():
    """Test system status endpoint"""
    print("\n[TEST] Testing /api/system/status...")
    try:
        response = requests.get(f"{BASE_URL}/api/system/status", timeout=10)
        print(f"   Status: {response.status_code}")
        data = response.json()
        print(f"   System Status: {data.get('system_status', 'N/A')}")
        print(f"   API Status: {len(data.get('api_status', {}))} APIs")
        return response.status_code, data
    except Exception as e:
        print(f"   [ERROR] {e}")
        return None, None

def check_portfolio_endpoints():
    """Check what portfolio endpoints exist"""
    print("\n[TEST] Checking available portfolio endpoints...")
    try:
        response = requests.get(f"{BASE_URL}/", timeout=10)
        if response.status_code == 200:
            print("   [OK] Root endpoint accessible")
            # Try to find portfolio endpoints in response
            text = response.text.lower()
            if "portfolio" in text:
                print("   [OK] Portfolio endpoints mentioned in API docs")
    except Exception as e:
        print(f"   ⚠️  Could not check root endpoint: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("Portfolio Management Endpoint Test")
    print("=" * 60)
    
    # Test health
    health_ok = test_health()
    
    # Test system status
    status_code, status_data = test_system_status()
    
    # Test portfolio endpoints
    portfolio_status = test_portfolio_no_auth()
    ai_portfolio_status, ai_portfolio_data = test_ai_portfolio_alias()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"[OK] Backend Health: {'OK' if health_ok else 'FAILED'}")
    print(f"[OK] System Status: {'OK' if status_code == 200 else f'Status {status_code}'}")
    print(f"[INFO] Portfolio Endpoint (/api/portfolio): Status {portfolio_status}")
    print(f"[INFO] AI Portfolio Alias (/api/ai/portfolio): Status {ai_portfolio_status}")
    
    if ai_portfolio_data:
        print(f"\n[INFO] Portfolio Data:")
        print(f"   - Success: {ai_portfolio_data.get('success', 'N/A')}")
        print(f"   - Portfolio Items: {len(ai_portfolio_data.get('portfolio', []))}")
        print(f"   - Total Value: ${ai_portfolio_data.get('total_value', 0):.2f}")
        print(f"   - Message: {ai_portfolio_data.get('message', 'N/A')}")
    
    print("\n" + "=" * 60)

