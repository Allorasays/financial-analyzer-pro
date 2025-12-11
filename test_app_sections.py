#!/usr/bin/env python3
"""Test App Sections"""

import sys
import os

# Add current directory to path
sys.path.append('.')

# Import the app
try:
    from app import get_global_markets_overview, get_forex_data, get_crypto_data
    print("✅ Successfully imported app functions")
    
    # Test Global Markets
    print("\n🌍 Testing Global Markets...")
    markets = get_global_markets_overview()
    print(f"Global Markets: {len(markets)} markets loaded")
    if markets:
        print(f"Sample: {markets[0]['name']} - ${markets[0]['price']:.2f}")
    
    # Test Forex
    print("\n💱 Testing Forex...")
    forex = get_forex_data()
    print(f"Forex: {len(forex)} pairs loaded")
    if forex:
        print(f"Sample: {forex[0]['name']} - {forex[0]['price']:.4f}")
    
    # Test Crypto
    print("\n₿ Testing Crypto...")
    crypto = get_crypto_data()
    print(f"Crypto: {len(crypto)} coins loaded")
    if crypto:
        print(f"Sample: {crypto[0]['name']} - ${crypto[0]['price']:.2f}")
    
    print("\n✅ All functions working correctly!")
    
except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()

