"""
FRED API Service
Federal Reserve Economic Data integration
"""

import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FREDService:
    """Service for interacting with FRED API"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or "9371fbb0a2b505b3262b5578f44016c5"  # Real FRED API key
        self.base_url = "https://api.stlouisfed.org/fred"
        self.timeout = 30
        self.rate_limit = 1200  # requests per day (free tier)
        self.request_count = 0
        self.last_reset = datetime.now()
        
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make API request with rate limiting"""
        # Check rate limit
        if self.request_count >= self.rate_limit:
            logger.warning(f"FRED rate limit reached: {self.request_count}/{self.rate_limit}")
            return None
            
        url = f"{self.base_url}{endpoint}"
        if params is None:
            params = {}
        params['api_key'] = self.api_key
        params['file_type'] = 'json'
        
        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            self.request_count += 1
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"FRED API request failed: {e}")
            return None
    
    def get_interest_rates(self) -> Optional[Dict]:
        """Get current interest rates"""
        params = {
            'series_id': 'FEDFUNDS',  # Federal Funds Rate
            'limit': 1,
            'sort_order': 'desc'
        }
        return self._make_request('/series/observations', params)
    
    def get_inflation_data(self) -> Optional[Dict]:
        """Get inflation data (CPI)"""
        params = {
            'series_id': 'CPIAUCSL',  # Consumer Price Index
            'limit': 1,
            'sort_order': 'desc'
        }
        return self._make_request('/series/observations', params)
    
    def get_gdp_data(self) -> Optional[Dict]:
        """Get GDP data"""
        params = {
            'series_id': 'GDP',  # Gross Domestic Product
            'limit': 1,
            'sort_order': 'desc'
        }
        return self._make_request('/series/observations', params)
    
    def get_unemployment_rate(self) -> Optional[Dict]:
        """Get unemployment rate"""
        params = {
            'series_id': 'UNRATE',  # Unemployment Rate
            'limit': 1,
            'sort_order': 'desc'
        }
        return self._make_request('/series/observations', params)
    
    def get_treasury_rates(self) -> Dict[str, Any]:
        """Get various Treasury rates"""
        treasury_series = {
            'DGS3MO': '3-Month Treasury Rate',
            'DGS6MO': '6-Month Treasury Rate',
            'DGS1': '1-Year Treasury Rate',
            'DGS2': '2-Year Treasury Rate',
            'DGS5': '5-Year Treasury Rate',
            'DGS10': '10-Year Treasury Rate',
            'DGS30': '30-Year Treasury Rate'
        }
        
        results = {}
        for series_id, description in treasury_series.items():
            params = {
                'series_id': series_id,
                'limit': 1,
                'sort_order': 'desc'
            }
            data = self._make_request('/series/observations', params)
            if data and 'observations' in data and len(data['observations']) > 0:
                latest = data['observations'][0]
                results[description] = {
                    'value': latest.get('value'),
                    'date': latest.get('date'),
                    'series_id': series_id
                }
            time.sleep(0.1)  # Small delay to avoid rate limiting
            
        return results
    
    def get_economic_indicators(self) -> Dict[str, Any]:
        """Get comprehensive economic indicators"""
        indicators = {
            'interest_rate': self.get_interest_rates(),
            'inflation': self.get_inflation_data(),
            'gdp': self.get_gdp_data(),
            'unemployment': self.get_unemployment_rate(),
            'treasury_rates': self.get_treasury_rates()
        }
        return indicators
    
    def get_series_info(self, series_id: str) -> Optional[Dict]:
        """Get information about a specific series"""
        params = {'series_id': series_id}
        return self._make_request('/series', params)
    
    def search_series(self, search_text: str) -> Optional[Dict]:
        """Search for economic series"""
        params = {
            'search_text': search_text,
            'limit': 20
        }
        return self._make_request('/series/search', params)
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit status"""
        return {
            'requests_made': self.request_count,
            'rate_limit': self.rate_limit,
            'remaining': self.rate_limit - self.request_count,
            'reset_time': self.last_reset + timedelta(days=1)
        }

# Global instance
fred_service = FREDService()

# Convenience functions
def get_interest_rates() -> Optional[Dict]:
    """Get current interest rates"""
    return fred_service.get_interest_rates()

def get_inflation_data() -> Optional[Dict]:
    """Get inflation data"""
    return fred_service.get_inflation_data()

def get_gdp_data() -> Optional[Dict]:
    """Get GDP data"""
    return fred_service.get_gdp_data()

def get_unemployment_rate() -> Optional[Dict]:
    """Get unemployment rate"""
    return fred_service.get_unemployment_rate()

def get_treasury_rates() -> Dict[str, Any]:
    """Get Treasury rates"""
    return fred_service.get_treasury_rates()

def get_economic_indicators() -> Dict[str, Any]:
    """Get all economic indicators"""
    return fred_service.get_economic_indicators()

def get_rate_limit_status() -> Dict[str, Any]:
    """Get FRED API rate limit status"""
    return fred_service.get_rate_limit_status()

# Test function
def test_fred_api():
    """Test FRED API functionality"""
    print("Testing FRED API...")
    
    # Test interest rates
    print("Testing interest rates...")
    interest_data = get_interest_rates()
    if interest_data:
        print(f"[SUCCESS] Interest rates retrieved: {interest_data}")
    else:
        print("[ERROR] Failed to retrieve interest rates")
    
    # Test Treasury rates
    print("Testing Treasury rates...")
    treasury_data = get_treasury_rates()
    if treasury_data:
        print(f"[SUCCESS] Treasury rates retrieved: {len(treasury_data)} rates")
        for name, data in treasury_data.items():
            print(f"  {name}: {data.get('value', 'N/A')}%")
    else:
        print("[ERROR] Failed to retrieve Treasury rates")
    
    # Test rate limit status
    print("Testing rate limit status...")
    rate_status = get_rate_limit_status()
    print(f"[SUCCESS] Rate limit status: {rate_status}")
    
    print("FRED API test completed!")

if __name__ == "__main__":
    test_fred_api()

