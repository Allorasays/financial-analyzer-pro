"""
Configuration for Real-Time Data Service
API keys and settings for multiple data providers
"""

import os
from typing import Dict, Any

# API Keys Configuration
# Set these in your environment variables or Streamlit secrets
API_KEYS = {
    'ALPHA_VANTAGE_API_KEY': os.getenv('ALPHA_VANTAGE_API_KEY', ''),
    'IEX_CLOUD_API_KEY': os.getenv('IEX_CLOUD_API_KEY', ''),
    'POLYGON_API_KEY': os.getenv('POLYGON_API_KEY', ''),
    'FINNHUB_API_KEY': os.getenv('FINNHUB_API_KEY', ''),
    'QUANDL_API_KEY': os.getenv('QUANDL_API_KEY', ''),
}

# Data Provider Settings
DATA_PROVIDER_CONFIG = {
    'alpha_vantage': {
        'name': 'Alpha Vantage',
        'base_url': 'https://www.alphavantage.co/query',
        'rate_limit': 5,  # Free tier: 5 requests per minute
        'premium_rate_limit': 1200,  # Premium tier: 1200 requests per minute
        'requires_api_key': True,
        'supports_realtime': False,
        'supports_historical': True,
        'max_data_points': 1000,
        'timeout': 10
    },
    'iex_cloud': {
        'name': 'IEX Cloud',
        'base_url': 'https://cloud.iexapis.com/stable',
        'rate_limit': 500000,  # Paid tier
        'free_rate_limit': 50000,  # Free tier
        'requires_api_key': True,
        'supports_realtime': True,
        'supports_historical': True,
        'max_data_points': 5000,
        'timeout': 10
    },
    'polygon': {
        'name': 'Polygon.io',
        'base_url': 'https://api.polygon.io',
        'rate_limit': 1000,  # Free tier
        'premium_rate_limit': 10000,  # Premium tier
        'requires_api_key': True,
        'supports_realtime': True,
        'supports_historical': True,
        'max_data_points': 10000,
        'timeout': 10
    },
    'yfinance': {
        'name': 'Yahoo Finance',
        'base_url': 'https://query1.finance.yahoo.com',
        'rate_limit': 2000,  # Estimated
        'requires_api_key': False,
        'supports_realtime': False,
        'supports_historical': True,
        'max_data_points': 10000,
        'timeout': 15
    }
}

# Caching Configuration
CACHE_CONFIG = {
    'default_ttl': 60,  # seconds
    'market_overview_ttl': 30,  # seconds
    'live_price_ttl': 5,  # seconds
    'historical_data_ttl': 300,  # 5 minutes
    'max_cache_size': 1000,  # maximum number of cached items
    'enable_redis': False,  # Set to True if Redis is available
    'redis_url': 'redis://localhost:6379/0'
}

# WebSocket Configuration
WEBSOCKET_CONFIG = {
    'host': 'localhost',
    'port': 8765,
    'ping_interval': 20,  # seconds
    'ping_timeout': 10,   # seconds
    'max_connections': 100,
    'message_queue_size': 1000
}

# Real-time Update Configuration
REALTIME_CONFIG = {
    'auto_refresh_interval': 5,  # seconds
    'max_symbols_per_client': 50,
    'update_batch_size': 10,
    'enable_auto_refresh': True,
    'enable_websocket': True,
    'enable_price_alerts': True
}

# Price Alert Configuration
ALERT_CONFIG = {
    'max_alerts_per_user': 100,
    'alert_check_interval': 30,  # seconds
    'alert_retention_days': 30,
    'enable_email_alerts': False,
    'enable_push_notifications': False,
    'email_smtp_server': '',
    'email_smtp_port': 587,
    'email_username': '',
    'email_password': ''
}

# Market Data Configuration
MARKET_DATA_CONFIG = {
    'default_symbols': ['^GSPC', '^IXIC', '^DJI', '^VIX', '^RUT'],
    'crypto_symbols': ['BTC-USD', 'ETH-USD', 'ADA-USD', 'SOL-USD'],
    'forex_symbols': ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X'],
    'commodity_symbols': ['GC=F', 'CL=F', 'NG=F', 'SI=F'],
    'sector_etfs': ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB', 'XLRE']
}

# Performance Monitoring Configuration
PERFORMANCE_CONFIG = {
    'enable_monitoring': True,
    'log_level': 'INFO',
    'enable_metrics': True,
    'metrics_retention_days': 7,
    'slow_query_threshold': 5.0,  # seconds
    'error_rate_threshold': 0.05  # 5%
}

# Error Handling Configuration
ERROR_CONFIG = {
    'max_retries': 3,
    'retry_delay': 1,  # seconds
    'circuit_breaker_threshold': 5,
    'circuit_breaker_timeout': 60,  # seconds
    'fallback_to_yfinance': True,
    'log_errors': True
}

# Data Quality Configuration
DATA_QUALITY_CONFIG = {
    'validate_price_data': True,
    'min_price_threshold': 0.01,
    'max_price_threshold': 1000000,
    'validate_volume_data': True,
    'min_volume_threshold': 0,
    'detect_outliers': True,
    'outlier_threshold': 3.0  # standard deviations
}

def get_config() -> Dict[str, Any]:
    """Get complete configuration"""
    return {
        'api_keys': API_KEYS,
        'data_providers': DATA_PROVIDER_CONFIG,
        'cache': CACHE_CONFIG,
        'websocket': WEBSOCKET_CONFIG,
        'realtime': REALTIME_CONFIG,
        'alerts': ALERT_CONFIG,
        'market_data': MARKET_DATA_CONFIG,
        'performance': PERFORMANCE_CONFIG,
        'error_handling': ERROR_CONFIG,
        'data_quality': DATA_QUALITY_CONFIG
    }

def get_api_key(provider: str) -> str:
    """Get API key for provider"""
    key_name = f"{provider.upper()}_API_KEY"
    return API_KEYS.get(key_name, '')

def is_provider_available(provider: str) -> bool:
    """Check if provider is available and configured"""
    if provider not in DATA_PROVIDER_CONFIG:
        return False
    
    config = DATA_PROVIDER_CONFIG[provider]
    if config['requires_api_key']:
        return bool(get_api_key(provider))
    
    return True

def get_provider_config(provider: str) -> Dict[str, Any]:
    """Get configuration for specific provider"""
    return DATA_PROVIDER_CONFIG.get(provider, {})

def get_rate_limit(provider: str, is_premium: bool = False) -> int:
    """Get rate limit for provider"""
    config = get_provider_config(provider)
    if is_premium and 'premium_rate_limit' in config:
        return config['premium_rate_limit']
    return config.get('rate_limit', 60)
