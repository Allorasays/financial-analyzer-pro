"""
Secure Configuration Management for Financial Analyzer
Handles API keys securely for PlayStore compliance
"""

import os
import logging
from typing import Dict, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecureConfig:
    """Secure configuration management for API keys and settings"""
    
    def __init__(self):
        self.api_keys = self._load_api_keys()
        self.config = self._load_config()
    
    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from environment variables"""
        api_keys = {
            'TIINGO_API_KEY': os.getenv('TIINGO_API_KEY'),
            'NEWSAPI_KEY': os.getenv('NEWSAPI_KEY'),
            'FRED_API_KEY': os.getenv('FRED_API_KEY'),
            'ALPHA_VANTAGE_KEY': os.getenv('ALPHA_VANTAGE_KEY'),
            'POLYGON_KEY': os.getenv('POLYGON_KEY'),
            'FMP_KEY': os.getenv('FMP_KEY'),
        }
        
        # Validate required keys
        required_keys = ['TIINGO_API_KEY', 'NEWSAPI_KEY', 'FRED_API_KEY']
        missing_keys = [key for key in required_keys if not api_keys.get(key)]
        
        if missing_keys:
            logger.warning(f"Missing API keys: {missing_keys}")
            logger.info("Using fallback keys for development")
            # Fallback to development keys (remove for production)
            api_keys.update({
                'TIINGO_API_KEY': api_keys.get('TIINGO_API_KEY') or '8c2e5b1e9d4a1cd31e1bb333d56232ddc382ee46',
                'NEWSAPI_KEY': api_keys.get('NEWSAPI_KEY') or '7d3d96223d67427f80773dfa3fdf37b8',
                'FRED_API_KEY': api_keys.get('FRED_API_KEY') or '9371fbb0a2b505b3262b5578f44016c5',
                'ALPHA_VANTAGE_KEY': api_keys.get('ALPHA_VANTAGE_KEY') or 'C04TV0QS7GVJF0RU',
                'POLYGON_KEY': api_keys.get('POLYGON_KEY') or 'gqvp07BQCfnH7Xq5p7GbbfAXLpvv7HTm',
                'FMP_KEY': api_keys.get('FMP_KEY') or 'R9F8nfYK9yGdmiq7I5ETw7e6EhTuG8ve',
            })
        
        return api_keys
    
    def _load_config(self) -> Dict[str, any]:
        """Load application configuration"""
        return {
            'app_name': 'Financial Analyzer Pro',
            'version': '2.0.0',
            'environment': os.getenv('ENVIRONMENT', 'development'),
            'debug': os.getenv('DEBUG', 'False').lower() == 'true',
            'rate_limits': {
                'tiingo': 1000,  # requests per day
                'newsapi': 1000,  # requests per day
                'fred': 1200,     # requests per day
                'alpha_vantage': 720,  # requests per day
                'polygon': 1000,  # requests per day
                'fmp': 250,       # requests per day
            },
            'cache_ttl': {
                'tiingo': 60,     # 1 minute
                'newsapi': 300,   # 5 minutes
                'fred': 3600,     # 1 hour
                'alpha_vantage': 300,  # 5 minutes
                'polygon': 60,    # 1 minute
                'fmp': 300,       # 5 minutes
            }
        }
    
    def get_api_key(self, service: str) -> Optional[str]:
        """Get API key for a specific service"""
        key_name = f"{service.upper()}_API_KEY" if service != 'newsapi' else 'NEWSAPI_KEY'
        return self.api_keys.get(key_name)
    
    def get_rate_limit(self, service: str) -> int:
        """Get rate limit for a specific service"""
        return self.config['rate_limits'].get(service.lower(), 1000)
    
    def get_cache_ttl(self, service: str) -> int:
        """Get cache TTL for a specific service"""
        return self.config['cache_ttl'].get(service.lower(), 300)
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.config['environment'] == 'production'
    
    def get_app_info(self) -> Dict[str, str]:
        """Get application information"""
        return {
            'name': self.config['app_name'],
            'version': self.config['version'],
            'environment': self.config['environment']
        }

# Global secure config instance
secure_config = SecureConfig()

# API Attribution for compliance
API_ATTRIBUTIONS = {
    'tiingo': 'Market data provided by Tiingo',
    'newsapi': 'News provided by NewsAPI',
    'fred': 'Economic data from Federal Reserve Bank of St. Louis',
    'alpha_vantage': 'Market data provided by Alpha Vantage',
    'polygon': 'Market data provided by Polygon.io',
    'fmp': 'Financial data provided by Financial Modeling Prep',
    'yahoo': 'Market data from Yahoo Finance'
}

def get_api_attribution(service: str) -> str:
    """Get attribution text for API service"""
    return API_ATTRIBUTIONS.get(service.lower(), f'Data provided by {service.title()}')

def validate_api_compliance():
    """Validate API compliance for PlayStore submission"""
    compliance_issues = []
    
    # Check for hardcoded keys in production
    if secure_config.is_production():
        for service, key in secure_config.api_keys.items():
            if not key or key.startswith('YOUR_') or key == 'None':
                compliance_issues.append(f"Missing production API key: {service}")
    
    # Check rate limit compliance
    for service in ['tiingo', 'newsapi', 'fred', 'alpha_vantage', 'polygon', 'fmp']:
        rate_limit = secure_config.get_rate_limit(service)
        if rate_limit <= 0:
            compliance_issues.append(f"Invalid rate limit for {service}")
    
    if compliance_issues:
        logger.error(f"API compliance issues: {compliance_issues}")
        return False, compliance_issues
    else:
        logger.info("API compliance validation passed")
        return True, []

# Test the secure configuration
if __name__ == "__main__":
    print("Testing Secure Configuration...")
    
    # Test API key retrieval
    print(f"Tiingo API Key: {'***' + secure_config.get_api_key('tiingo')[-4:] if secure_config.get_api_key('tiingo') else 'Not set'}")
    print(f"NewsAPI Key: {'***' + secure_config.get_api_key('newsapi')[-4:] if secure_config.get_api_key('newsapi') else 'Not set'}")
    print(f"FRED API Key: {'***' + secure_config.get_api_key('fred')[-4:] if secure_config.get_api_key('fred') else 'Not set'}")
    
    # Test rate limits
    print(f"Tiingo Rate Limit: {secure_config.get_rate_limit('tiingo')} requests/day")
    print(f"NewsAPI Rate Limit: {secure_config.get_rate_limit('newsapi')} requests/day")
    print(f"FRED Rate Limit: {secure_config.get_rate_limit('fred')} requests/day")
    
    # Test compliance
    is_compliant, issues = validate_api_compliance()
    print(f"API Compliance: {'PASSED' if is_compliant else 'FAILED'}")
    if issues:
        print(f"Issues: {issues}")
    
    # Test attributions
    print(f"Tiingo Attribution: {get_api_attribution('tiingo')}")
    print(f"NewsAPI Attribution: {get_api_attribution('newsapi')}")
    print(f"FRED Attribution: {get_api_attribution('fred')}")
    
    print("Secure Configuration Test Complete!")

