"""
Tiingo API Service
Replaces IEX Cloud functionality with Tiingo API
"""

import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from config import TIINGO_CONFIG
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TiingoService:
    """Service for interacting with Tiingo API"""
    
    def __init__(self):
        self.api_key = TIINGO_CONFIG['api_key']
        self.base_url = TIINGO_CONFIG['base_url']
        self.timeout = TIINGO_CONFIG['timeout']
        self.rate_limit = TIINGO_CONFIG['rate_limit']
        self.request_count = 0
        self.last_reset = datetime.now()
        
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make API request with rate limiting"""
        # Check rate limit
        if self.request_count >= self.rate_limit:
            logger.warning(f"Rate limit reached: {self.request_count}/{self.rate_limit}")
            return None
            
        url = f"{self.base_url}{endpoint}"
        if params is None:
            params = {}
        params['token'] = self.api_key
        
        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            self.request_count += 1
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Tiingo API request failed: {e}")
            return None
    
    def get_company_info(self, ticker: str) -> Optional[Dict]:
        """Get company information"""
        endpoint = f"/daily/{ticker.upper()}"
        return self._make_request(endpoint)
    
    def get_stock_price(self, ticker: str) -> Optional[Dict]:
        """Get current stock price"""
        endpoint = f"/daily/{ticker.upper()}/prices"
        params = {'resampleFreq': 'daily'}
        return self._make_request(endpoint, params)
    
    def get_historical_data(self, ticker: str, start_date: str = None, end_date: str = None) -> Optional[List[Dict]]:
        """Get historical stock data"""
        endpoint = f"/daily/{ticker.upper()}/prices"
        params = {'resampleFreq': 'daily'}
        
        if start_date:
            params['startDate'] = start_date
        if end_date:
            params['endDate'] = end_date
            
        return self._make_request(endpoint, params)
    
    def get_multiple_stocks(self, tickers: List[str]) -> Dict[str, Any]:
        """Get data for multiple stocks"""
        results = {}
        for ticker in tickers:
            try:
                price_data = self.get_stock_price(ticker)
                if price_data:
                    results[ticker] = price_data
                time.sleep(0.1)  # Small delay to avoid rate limiting
            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {e}")
                results[ticker] = None
        return results
    
    def get_market_overview(self) -> Dict[str, Any]:
        """Get market overview data"""
        major_indices = ['^GSPC', '^IXIC', '^DJI', '^VIX', '^RUT']
        return self.get_multiple_stocks(major_indices)
    
    def get_crypto_prices(self) -> Dict[str, Any]:
        """Get cryptocurrency prices"""
        crypto_tickers = ['BTC-USD', 'ETH-USD', 'ADA-USD', 'SOL-USD']
        return self.get_multiple_stocks(crypto_tickers)
    
    def get_forex_rates(self) -> Dict[str, Any]:
        """Get forex exchange rates"""
        forex_pairs = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X']
        return self.get_multiple_stocks(forex_pairs)
    
    def get_commodity_prices(self) -> Dict[str, Any]:
        """Get commodity prices"""
        commodity_tickers = ['GC=F', 'CL=F', 'NG=F', 'SI=F']
        return self.get_multiple_stocks(commodity_tickers)
    
    def search_tickers(self, query: str) -> Optional[List[Dict]]:
        """Search for tickers by company name"""
        # Note: Tiingo doesn't have a direct search endpoint
        # This would need to be implemented with a separate service
        logger.warning("Ticker search not directly supported by Tiingo API")
        return None
    
    def get_earnings_calendar(self, start_date: str = None, end_date: str = None) -> Optional[List[Dict]]:
        """Get earnings calendar"""
        # Note: Tiingo doesn't have earnings calendar
        # This would need to be implemented with a separate service
        logger.warning("Earnings calendar not supported by Tiingo API")
        return None
    
    def get_news(self, ticker: str = None, limit: int = 10) -> Optional[List[Dict]]:
        """Get news articles"""
        # Note: Tiingo doesn't have news API
        # This would need to be implemented with NewsAPI
        logger.warning("News not supported by Tiingo API - use NewsAPI instead")
        return None
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit status"""
        return {
            'requests_made': self.request_count,
            'rate_limit': self.rate_limit,
            'remaining': self.rate_limit - self.request_count,
            'reset_time': self.last_reset + timedelta(days=1)
        }

# Global instance
tiingo_service = TiingoService()

# Convenience functions
def get_company_info(ticker: str) -> Optional[Dict]:
    """Get company information for a ticker"""
    return tiingo_service.get_company_info(ticker)

def get_stock_price(ticker: str) -> Optional[Dict]:
    """Get current stock price for a ticker"""
    return tiingo_service.get_stock_price(ticker)

def get_historical_data(ticker: str, start_date: str = None, end_date: str = None) -> Optional[List[Dict]]:
    """Get historical data for a ticker"""
    return tiingo_service.get_historical_data(ticker, start_date, end_date)

def get_market_overview() -> Dict[str, Any]:
    """Get market overview data"""
    return tiingo_service.get_market_overview()

def get_crypto_prices() -> Dict[str, Any]:
    """Get cryptocurrency prices"""
    return tiingo_service.get_crypto_prices()

def get_forex_rates() -> Dict[str, Any]:
    """Get forex exchange rates"""
    return tiingo_service.get_forex_rates()

def get_commodity_prices() -> Dict[str, Any]:
    """Get commodity prices"""
    return tiingo_service.get_commodity_prices()

def get_rate_limit_status() -> Dict[str, Any]:
    """Get current rate limit status"""
    return tiingo_service.get_rate_limit_status()

# Test function
def test_tiingo_api():
    """Test Tiingo API functionality"""
    print("Testing Tiingo API...")
    
    # Test company info
    print("Testing company info for AAPL...")
    company_info = get_company_info('AAPL')
    if company_info:
        print(f"[SUCCESS] Company info retrieved: {company_info.get('name', 'N/A')}")
    else:
        print("[ERROR] Failed to retrieve company info")
    
    # Test stock price
    print("Testing stock price for AAPL...")
    price_data = get_stock_price('AAPL')
    if price_data:
        print(f"[SUCCESS] Stock price retrieved: {len(price_data)} data points")
    else:
        print("[ERROR] Failed to retrieve stock price")
    
    # Test rate limit status
    print("Testing rate limit status...")
    rate_status = get_rate_limit_status()
    print(f"[SUCCESS] Rate limit status: {rate_status}")
    
    print("Tiingo API test completed!")

if __name__ == "__main__":
    test_tiingo_api()
