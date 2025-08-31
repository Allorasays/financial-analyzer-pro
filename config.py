"""
Configuration file for Financial Analyzer Pro
Contains constants, settings, and configuration options
"""

import os
from typing import Dict, List

# API Configuration
API_CONFIG = {
    'base_url': os.getenv('API_BASE_URL', 'http://localhost:8000'),
    'timeout': 30,
    'retry_attempts': 3,
    'cache_ttl': 300  # 5 minutes
}

# Financial Analysis Settings
FINANCIAL_CONFIG = {
    'default_discount_rate': 0.10,  # 10%
    'default_growth_rate': 0.03,    # 3%
    'forecast_period': 5,           # 5 years
    'risk_thresholds': {
        'low_volatility': 0.15,     # 15%
        'high_volatility': 0.30,    # 30%
        'debt_coverage_min': 3.0,   # 3x coverage
        'current_ratio_min': 1.5,   # 1.5x
        'quick_ratio_min': 1.0      # 1.0x
    }
}

# Chart Configuration
CHART_CONFIG = {
    'themes': ['plotly', 'plotly_white', 'plotly_dark', 'seaborn', 'simple_white'],
    'default_theme': 'plotly',
    'colors': {
        'primary': '#667eea',
        'secondary': '#764ba2',
        'success': '#28a745',
        'warning': '#ffc107',
        'danger': '#dc3545',
        'info': '#17a2b8'
    },
    'height': 500,
    'width': '100%'
}

# Portfolio Configuration
PORTFOLIO_CONFIG = {
    'max_positions': 50,
    'default_currency': 'USD',
    'rebalance_threshold': 0.05,  # 5%
    'risk_categories': ['Conservative', 'Moderate', 'Aggressive']
}

# Export Configuration
EXPORT_CONFIG = {
    'formats': ['csv', 'excel', 'json', 'pdf'],
    'default_format': 'csv',
    'include_timestamp': True,
    'max_file_size': 10 * 1024 * 1024  # 10MB
}

# Cache Configuration
CACHE_CONFIG = {
    'real_time_data': 300,      # 5 minutes
    'financial_data': 3600,     # 1 hour
    'market_data': 1800,        # 30 minutes
    'analysis_results': 7200    # 2 hours
}

# Risk Assessment Weights
RISK_WEIGHTS = {
    'liquidity': 0.25,
    'profitability': 0.25,
    'efficiency': 0.20,
    'leverage': 0.20,
    'growth': 0.10
}

# Industry Benchmarks (example data)
INDUSTRY_BENCHMARKS = {
    'Technology': {
        'avg_pe': 28.5,
        'avg_growth': 15.2,
        'avg_margin': 18.5
    },
    'Healthcare': {
        'avg_pe': 22.1,
        'avg_growth': 8.5,
        'avg_margin': 12.3
    },
    'Finance': {
        'avg_pe': 15.8,
        'avg_growth': 6.2,
        'avg_margin': 25.1
    },
    'Energy': {
        'avg_pe': 12.3,
        'avg_growth': 4.1,
        'avg_margin': 8.7
    },
    'Consumer_Goods': {
        'avg_pe': 18.9,
        'avg_growth': 9.8,
        'avg_margin': 15.2
    }
}

# Notification Settings
NOTIFICATION_CONFIG = {
    'price_alerts': True,
    'portfolio_alerts': True,
    'news_alerts': False,
    'email_notifications': False,
    'push_notifications': False
}

# Performance Monitoring
PERFORMANCE_CONFIG = {
    'enable_logging': True,
    'log_level': 'INFO',
    'performance_tracking': True,
    'error_reporting': True
}
