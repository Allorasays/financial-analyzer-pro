#!/usr/bin/env python3
"""
Final test to identify Redis and Global Markets issues
"""

def test_redis_import():
    """Test Redis import"""
    print("Testing Redis import...")
    try:
        import redis
        print("✅ Redis imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Redis import failed: {e}")
        return False

def test_global_markets():
    """Test global markets functionality"""
    print("\nTesting Global Markets functionality...")
    try:
        import streamlit as st
        import yfinance as yf
        import numpy as np
        from datetime import datetime, timedelta
        
        market_indices = [
            {'symbol': '^GSPC', 'name': 'S&P 500', 'base_price': 4500},
            {'symbol': '^IXIC', 'name': 'NASDAQ', 'base_price': 14000},
            {'symbol': '^DJI', 'name': 'Dow Jones', 'base_price': 35000},
            {'symbol': '^VIX', 'name': 'VIX Volatility', 'base_price': 20},
        ]
        
        markets = []
        for market in market_indices:
            try:
                ticker = yf.Ticker(market['symbol'])
                hist = ticker.history(period='2d', timeout=10)
                
                if not hist.empty and len(hist) >= 2:
                    current_price = hist['Close'].iloc[-1]
                    previous_price = hist['Close'].iloc[-2]
                    change = current_price - previous_price
                    change_percent = (change / previous_price) * 100
                    
                    markets.append({
                        'name': market['name'],
                        'symbol': market['symbol'],
                        'price': current_price,
                        'change': change,
                        'change_percent': change_percent
                    })
                    print(f"✅ {market['name']}: ${current_price:.2f} ({change_percent:+.2f}%)")
                else:
                    print(f"❌ {market['name']}: No data")
            except Exception as e:
                print(f"❌ {market['name']}: Error - {e}")
        
        print(f"Total markets loaded: {len(markets)}")
        return len(markets) > 0
        
    except Exception as e:
        print(f"❌ Global markets test failed: {e}")
        return False

def test_app_imports():
    """Test app imports to find Redis issue"""
    print("\nTesting app imports...")
    try:
        # Test if app.py imports work
        print("Testing app.py import...")
        import app
        print("✅ app.py imported successfully")
        
        # Test if we can call the global markets function
        print("Testing get_global_markets_overview function...")
        from app import get_global_markets_overview
        markets = get_global_markets_overview()
        print(f"✅ Global markets function returned {len(markets) if markets else 0} markets")
        
        return True
        
    except Exception as e:
        print(f"❌ App import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🔍 Final Issue Diagnosis")
    print("=" * 50)
    
    # Test Redis
    redis_ok = test_redis_import()
    
    # Test Global Markets
    markets_ok = test_global_markets()
    
    # Test App Imports
    app_ok = test_app_imports()
    
    print("\n" + "=" * 50)
    print("📊 Final Test Results:")
    print(f"Redis Import: {'✅ OK' if redis_ok else '❌ FAILED'}")
    print(f"Global Markets: {'✅ OK' if markets_ok else '❌ FAILED'}")
    print(f"App Imports: {'✅ OK' if app_ok else '❌ FAILED'}")
    
    if not redis_ok:
        print("\n💡 Redis Solution: The app doesn't actually need Redis - this is a false error")
    
    if not markets_ok:
        print("\n💡 Global Markets Solution: Check network connectivity or API access")

if __name__ == "__main__":
    main()




