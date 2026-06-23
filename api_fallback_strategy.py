"""
API Fallback Strategy
Handles API failures and provides alternative data sources for price/history flows.

For comprehensive financial metrics, use comprehensive_financial_aggregator.py instead.
"""

import os
import requests
import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import yfinance as yf
from tiingo_service import tiingo_service
from fred_service import fred_service

logger = logging.getLogger(__name__)

class APIFallbackStrategy:
    """Manages API fallbacks and rate limiting"""
    
    def __init__(self):
        self.rate_limits = {
            'yahoo_finance': {'last_request': 0, 'min_interval': 2},  # 2 seconds between requests
            'alpha_vantage': {'last_request': 0, 'min_interval': 12},  # 5 requests per minute = 12 seconds
            'tiingo': {'last_request': 0, 'min_interval': 1},  # 1 second between requests
            'fred': {'last_request': 0, 'min_interval': 1}  # 1 second between requests
        }
        
        # API status tracking
        self.api_status = {
            'yahoo_finance': True,
            'alpha_vantage': True,
            'tiingo': True,
            'fred': True,
            'fmp': bool(os.getenv('FMP_API_KEY')),
        }
    
    def _check_rate_limit(self, api_name: str) -> bool:
        """Check if enough time has passed since last request"""
        if api_name not in self.rate_limits:
            return True
        
        last_request = self.rate_limits[api_name]['last_request']
        min_interval = self.rate_limits[api_name]['min_interval']
        
        time_since_last = time.time() - last_request
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            logger.info(f"Rate limiting {api_name}: sleeping {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.rate_limits[api_name]['last_request'] = time.time()
        return True
    
    def get_stock_data(self, ticker: str, period: str = "1y") -> Optional[Dict]:
        """Get stock data with fallback strategy"""
        
        # Try Yahoo Finance first (primary)
        if self.api_status['yahoo_finance']:
            try:
                self._check_rate_limit('yahoo_finance')
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period)
                
                if not hist.empty and len(hist) >= 60:
                    logger.info(f"Yahoo Finance: Successfully retrieved {len(hist)} days for {ticker}")
                    return {
                        'source': 'yahoo_finance',
                        'data': hist,
                        'success': True
                    }
                else:
                    logger.warning(f"Yahoo Finance: Insufficient data for {ticker}")
                    
            except Exception as e:
                logger.error(f"Yahoo Finance failed for {ticker}: {e}")
                self.api_status['yahoo_finance'] = False
        
        # Fallback to Tiingo API
        if self.api_status['tiingo']:
            try:
                self._check_rate_limit('tiingo')
                
                # Get historical data from Tiingo
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
                
                tiingo_data = tiingo_service.get_historical_data(ticker, start_date, end_date)
                if tiingo_data and len(tiingo_data) >= 60:
                    # Convert Tiingo data to pandas DataFrame format
                    import pandas as pd
                    hist = pd.DataFrame(tiingo_data)
                    hist['Date'] = pd.to_datetime(hist['date'])
                    hist.set_index('Date', inplace=True)
                    hist = hist[['open', 'high', 'low', 'close', 'volume']]
                    hist.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    
                    logger.info(f"Tiingo API: Successfully retrieved {len(hist)} days for {ticker}")
                    return {
                        'source': 'tiingo',
                        'data': hist,
                        'success': True
                    }
                else:
                    logger.warning(f"Tiingo API: Insufficient data for {ticker}")
                    
            except Exception as e:
                logger.error(f"Tiingo API failed for {ticker}: {e}")
                self.api_status['tiingo'] = False
        
        # Fallback to Alpha Vantage
        if self.api_status['alpha_vantage']:
            try:
                self._check_rate_limit('alpha_vantage')
                
                api_key = os.getenv('ALPHAVANTAGE_API_KEY', '')
                if not api_key:
                    raise ValueError("ALPHAVANTAGE_API_KEY not configured")
                url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={api_key}&outputsize=full'
                
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'Time Series (Daily)' in data:
                        # Convert Alpha Vantage data to pandas DataFrame
                        import pandas as pd
                        time_series = data['Time Series (Daily)']
                        
                        df_data = []
                        for date, values in time_series.items():
                            df_data.append({
                                'Date': pd.to_datetime(date),
                                'Open': float(values['1. open']),
                                'High': float(values['2. high']),
                                'Low': float(values['3. low']),
                                'Close': float(values['4. close']),
                                'Volume': int(values['5. volume'])
                            })
                        
                        hist = pd.DataFrame(df_data)
                        hist.set_index('Date', inplace=True)
                        hist = hist.sort_index()
                        
                        if len(hist) >= 60:
                            logger.info(f"Alpha Vantage: Successfully retrieved {len(hist)} days for {ticker}")
                            return {
                                'source': 'alpha_vantage',
                                'data': hist,
                                'success': True
                            }
                        else:
                            logger.warning(f"Alpha Vantage: Insufficient data for {ticker}")
                    else:
                        logger.warning(f"Alpha Vantage: No time series data for {ticker}")
                        
            except Exception as e:
                logger.error(f"Alpha Vantage failed for {ticker}: {e}")
                self.api_status['alpha_vantage'] = False
        
        # If all APIs fail, return None
        logger.error(f"All APIs failed for {ticker}")
        return None
    
    def get_economic_data(self, series_id: str) -> Optional[Dict]:
        """Get economic data with fallback strategy"""
        
        if self.api_status['fred']:
            try:
                self._check_rate_limit('fred')
                
                # Use FRED service
                params = {
                    'series_id': series_id,
                    'limit': 1,
                    'sort_order': 'desc'
                }
                
                data = fred_service._make_request('/series/observations', params)
                if data and data.get('observations'):
                    logger.info(f"FRED API: Successfully retrieved data for {series_id}")
                    return {
                        'source': 'fred',
                        'data': data,
                        'success': True
                    }
                else:
                    logger.warning(f"FRED API: No data for {series_id}")
                    
            except Exception as e:
                logger.error(f"FRED API failed for {series_id}: {e}")
                self.api_status['fred'] = False
        
        # Return None if FRED fails (no fallback for economic data)
        logger.error(f"FRED API failed for {series_id}")
        return None
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get current API status"""
        return {
            'api_status': self.api_status.copy(),
            'rate_limits': {k: v['min_interval'] for k, v in self.rate_limits.items()},
            'timestamp': datetime.now().isoformat()
        }
    
    def reset_api_status(self, api_name: str = None):
        """Reset API status (useful for testing)"""
        if api_name:
            if api_name in self.api_status:
                self.api_status[api_name] = True
                logger.info(f"Reset API status for {api_name}")
        else:
            for api in self.api_status:
                self.api_status[api] = True
            logger.info("Reset all API statuses")

# Create global instance
api_fallback = APIFallbackStrategy()

def test_api_fallback():
    """Test the API fallback strategy"""
    print("Testing API Fallback Strategy...")
    
    # Test stock data
    print("\n=== Testing Stock Data ===")
    result = api_fallback.get_stock_data('AAPL')
    if result:
        print(f"Success: {result['source']} - {len(result['data'])} days")
    else:
        print("Failed: All APIs failed")
    
    # Test economic data
    print("\n=== Testing Economic Data ===")
    result = api_fallback.get_economic_data('FEDFUNDS')
    if result:
        print(f"Success: {result['source']} - {len(result['data'].get('observations', []))} observations")
    else:
        print("Failed: FRED API failed")
    
    # Show API status
    print("\n=== API Status ===")
    status = api_fallback.get_api_status()
    for api, is_active in status['api_status'].items():
        print(f"{api}: {'Active' if is_active else 'Inactive'}")

if __name__ == "__main__":
    test_api_fallback()

