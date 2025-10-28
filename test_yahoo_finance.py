#!/usr/bin/env python3
"""
Yahoo Finance API Availability Test
"""

import yfinance as yf
import time
from datetime import datetime

def test_yahoo_finance():
    """Test Yahoo Finance API availability"""
    print("Testing Yahoo Finance API Availability...")
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 50)
    
    # Test tickers
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY', 'QQQ', 'NVDA']
    results = {}
    
    for ticker in tickers:
        try:
            print(f"Testing {ticker}...", end=" ")
            stock = yf.Ticker(ticker)
            
            # Test basic info
            info = stock.info
            if not info:
                print("FAILED - No info")
                results[ticker] = "FAILED - No info"
                continue
            
            # Test historical data
            hist = stock.history(period="1d")
            if len(hist) == 0:
                print("FAILED - No historical data")
                results[ticker] = "FAILED - No historical data"
                continue
            
            # Get latest price
            latest_price = hist['Close'].iloc[-1]
            print(f"SUCCESS - ${latest_price:.2f}")
            results[ticker] = f"SUCCESS - ${latest_price:.2f}"
            
            # Small delay to avoid rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            print(f"FAILED - {str(e)}")
            results[ticker] = f"FAILED - {str(e)}"
    
    print("-" * 50)
    print("SUMMARY:")
    success_count = sum(1 for result in results.values() if "SUCCESS" in result)
    total_count = len(results)
    
    print(f"Successful: {success_count}/{total_count}")
    print(f"Failed: {total_count - success_count}/{total_count}")
    print(f"Success Rate: {(success_count/total_count)*100:.1f}%")
    
    if success_count == total_count:
        print("Yahoo Finance API is FULLY OPERATIONAL!")
    elif success_count > total_count // 2:
        print("Yahoo Finance API is PARTIALLY OPERATIONAL")
    else:
        print("Yahoo Finance API has SIGNIFICANT ISSUES")
    
    print("\nDetailed Results:")
    for ticker, result in results.items():
        print(f"  {ticker}: {result}")
    
    return results

def test_yahoo_finance_rate_limits():
    """Test Yahoo Finance rate limits"""
    print("\nTesting Yahoo Finance Rate Limits...")
    
    ticker = 'AAPL'
    stock = yf.Ticker(ticker)
    
    # Test multiple rapid requests
    start_time = time.time()
    requests_made = 0
    
    for i in range(10):
        try:
            hist = stock.history(period="1d")
            requests_made += 1
            time.sleep(0.1)  # 100ms delay
        except Exception as e:
            print(f"Request {i+1} failed: {e}")
            break
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Rate Limit Test Results:")
    print(f"  Requests made: {requests_made}/10")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Rate: {requests_made/duration:.2f} requests/second")
    
    if requests_made == 10:
        print("Rate limits are GENEROUS")
    elif requests_made >= 5:
        print("Rate limits are MODERATE")
    else:
        print("Rate limits are STRICT")

if __name__ == "__main__":
    results = test_yahoo_finance()
    test_yahoo_finance_rate_limits()
    
    print("\nYahoo Finance API Test Complete!")
