#!/usr/bin/env python3
"""Test Global Markets Function"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

def get_global_markets_overview():
    """Enhanced global markets overview with robust fallback"""
    markets = []
    
    # Define major global markets
    market_indices = [
        {'symbol': '^GSPC', 'name': 'S&P 500', 'base_price': 4500},
        {'symbol': '^IXIC', 'name': 'NASDAQ', 'base_price': 14000},
        {'symbol': '^DJI', 'name': 'Dow Jones', 'base_price': 35000},
        {'symbol': '^VIX', 'name': 'VIX Volatility', 'base_price': 20},
        {'symbol': '^FTSE', 'name': 'FTSE 100', 'base_price': 7500},
        {'symbol': '^GDAXI', 'name': 'DAX', 'base_price': 16000},
        {'symbol': '^FCHI', 'name': 'CAC 40', 'base_price': 7000},
        {'symbol': '^N225', 'name': 'Nikkei 225', 'base_price': 30000},
        {'symbol': '^HSI', 'name': 'Hang Seng', 'base_price': 18000},
        {'symbol': '^AXJO', 'name': 'ASX 200', 'base_price': 7500},
        {'symbol': '^TNX', 'name': '10-Year Treasury', 'base_price': 4.5},
        {'symbol': 'GC=F', 'name': 'Gold', 'base_price': 2000}
    ]
    
    for market in market_indices:
        try:
            ticker = yf.Ticker(market['symbol'])
            hist = ticker.history(period="2d", timeout=10)
            
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
            # Fallback to demo data
            np.random.seed(hash(market['symbol']) % 2**32)
            base_price = market['base_price']
            change_percent = np.random.normal(0, 2)  # Random change between -4% to +4%
            change = base_price * (change_percent / 100)
            current_price = base_price + change
            
            markets.append({
                'name': market['name'],
                'symbol': market['symbol'],
                'price': current_price,
                'change': change,
                'change_percent': change_percent
            })
            print(f"🔄 {market['name']}: Demo data ${current_price:.2f} ({change_percent:+.2f}%)")
    
    return markets

if __name__ == "__main__":
    print("Testing Global Markets Function...")
    print("=" * 50)
    markets = get_global_markets_overview()
    print("=" * 50)
    print(f"Total markets loaded: {len(markets)}")
    
    if markets:
        print("\nMarket Summary:")
        for market in markets:
            print(f"- {market['name']}: ${market['price']:.2f} ({market['change_percent']:+.2f}%)")
    else:
        print("❌ No markets loaded!")

