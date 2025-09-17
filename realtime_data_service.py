"""
Real-Time Data Service for Financial Analyzer Pro
Supports multiple data providers with intelligent fallback and caching
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import threading
from dataclasses import dataclass
from enum import Enum
import logging

# Optional imports with graceful fallbacks
try:
    import asyncio
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProvider(Enum):
    YFINANCE = "yfinance"
    ALPHA_VANTAGE = "alpha_vantage"
    IEX_CLOUD = "iex_cloud"
    POLYGON = "polygon"

@dataclass
class DataSource:
    name: str
    api_key: str
    base_url: str
    rate_limit: int  # requests per minute
    last_request: float = 0
    requests_made: int = 0
    is_active: bool = True

class RealTimeDataService:
    """Real-time data service with multiple providers and intelligent caching"""
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 60  # seconds
        self.data_sources = self._initialize_data_sources()
        self.websocket_connections = {}
        self.price_alerts = {}
        
    def _initialize_data_sources(self) -> Dict[DataProvider, DataSource]:
        """Initialize available data sources"""
        # Try to get API keys from secrets, fallback to empty string
        try:
            alpha_vantage_key = st.secrets.get("ALPHA_VANTAGE_API_KEY", "")
        except:
            alpha_vantage_key = ""
        
        try:
            iex_cloud_key = st.secrets.get("IEX_CLOUD_API_KEY", "")
        except:
            iex_cloud_key = ""
        
        try:
            polygon_key = st.secrets.get("POLYGON_API_KEY", "")
        except:
            polygon_key = ""
        
        return {
            DataProvider.YFINANCE: DataSource(
                name="Yahoo Finance",
                api_key="",  # No API key needed
                base_url="https://query1.finance.yahoo.com",
                rate_limit=2000
            ),
            DataProvider.ALPHA_VANTAGE: DataSource(
                name="Alpha Vantage",
                api_key=alpha_vantage_key,
                base_url="https://www.alphavantage.co/query",
                rate_limit=5  # Free tier limit
            ),
            DataProvider.IEX_CLOUD: DataSource(
                name="IEX Cloud",
                api_key=iex_cloud_key,
                base_url="https://cloud.iexapis.com/stable",
                rate_limit=500000  # Paid tier
            ),
            DataProvider.POLYGON: DataSource(
                name="Polygon.io",
                api_key=polygon_key,
                base_url="https://api.polygon.io",
                rate_limit=1000  # Free tier
            )
        }
    
    def _is_rate_limited(self, provider: DataProvider) -> bool:
        """Check if provider is rate limited"""
        source = self.data_sources[provider]
        current_time = time.time()
        
        # Reset counter if more than a minute has passed
        if current_time - source.last_request > 60:
            source.requests_made = 0
            source.last_request = current_time
        
        return source.requests_made >= source.rate_limit
    
    def _update_rate_limit(self, provider: DataProvider):
        """Update rate limit counter"""
        source = self.data_sources[provider]
        source.requests_made += 1
        source.last_request = time.time()
    
    def _get_cached_data(self, key: str) -> Optional[Any]:
        """Get data from cache if not expired"""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.cache_ttl:
                return data
            else:
                del self.cache[key]
        return None
    
    def _set_cached_data(self, key: str, data: Any):
        """Store data in cache with timestamp"""
        self.cache[key] = (data, time.time())
    
    def get_stock_data(self, symbol: str, period: str = "1d", provider: Optional[DataProvider] = None) -> Tuple[pd.DataFrame, str]:
        """Get real-time stock data with fallback between providers"""
        cache_key = f"stock_{symbol}_{period}"
        cached_data = self._get_cached_data(cache_key)
        
        if cached_data is not None:
            return cached_data, "cached"
        
        # Try providers in order of preference
        providers_to_try = [provider] if provider else [
            DataProvider.YFINANCE,
            DataProvider.IEX_CLOUD,
            DataProvider.ALPHA_VANTAGE,
            DataProvider.POLYGON
        ]
        
        for prov in providers_to_try:
            if not self.data_sources[prov].is_active or self._is_rate_limited(prov):
                continue
                
            try:
                data, source = self._fetch_from_provider(symbol, period, prov)
                if data is not None and not data.empty:
                    self._set_cached_data(cache_key, (data, source))
                    self._update_rate_limit(prov)
                    return data, source
            except Exception as e:
                logger.warning(f"Provider {prov.value} failed for {symbol}: {str(e)}")
                continue
        
        return pd.DataFrame(), "no_data"
    
    def _fetch_from_provider(self, symbol: str, period: str, provider: DataProvider) -> Tuple[pd.DataFrame, str]:
        """Fetch data from specific provider"""
        if provider == DataProvider.YFINANCE:
            return self._fetch_yfinance(symbol, period)
        elif provider == DataProvider.ALPHA_VANTAGE:
            return self._fetch_alpha_vantage(symbol, period)
        elif provider == DataProvider.IEX_CLOUD:
            return self._fetch_iex_cloud(symbol, period)
        elif provider == DataProvider.POLYGON:
            return self._fetch_polygon(symbol, period)
        else:
            return pd.DataFrame(), "unknown_provider"
    
    def _fetch_yfinance(self, symbol: str, period: str) -> Tuple[pd.DataFrame, str]:
        """Fetch data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval="1m" if period == "1d" else "1d")
            return data, "yfinance"
        except Exception as e:
            logger.error(f"Yahoo Finance error: {str(e)}")
            return pd.DataFrame(), "yfinance_error"
    
    def _fetch_alpha_vantage(self, symbol: str, period: str) -> Tuple[pd.DataFrame, str]:
        """Fetch data from Alpha Vantage"""
        try:
            api_key = self.data_sources[DataProvider.ALPHA_VANTAGE].api_key
            if not api_key:
                return pd.DataFrame(), "no_api_key"
            
            # Map period to Alpha Vantage function
            function_map = {
                "1d": "TIME_SERIES_INTRADAY",
                "5d": "TIME_SERIES_INTRADAY",
                "1mo": "TIME_SERIES_DAILY",
                "3mo": "TIME_SERIES_DAILY",
                "6mo": "TIME_SERIES_DAILY",
                "1y": "TIME_SERIES_DAILY",
                "2y": "TIME_SERIES_DAILY"
            }
            
            function = function_map.get(period, "TIME_SERIES_DAILY")
            interval = "1min" if period in ["1d", "5d"] else "daily"
            
            url = f"{self.data_sources[DataProvider.ALPHA_VANTAGE].base_url}"
            params = {
                "function": function,
                "symbol": symbol,
                "apikey": api_key,
                "interval": interval,
                "outputsize": "compact"
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if "Error Message" in data:
                return pd.DataFrame(), "alpha_vantage_error"
            
            # Parse Alpha Vantage data
            time_series_key = list(data.keys())[1]  # Get the time series key
            time_series = data[time_series_key]
            
            df_data = []
            for timestamp, values in time_series.items():
                df_data.append({
                    'Open': float(values['1. open']),
                    'High': float(values['2. high']),
                    'Low': float(values['3. low']),
                    'Close': float(values['4. close']),
                    'Volume': int(values['5. volume']),
                    'Date': pd.to_datetime(timestamp)
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            
            return df, "alpha_vantage"
            
        except Exception as e:
            logger.error(f"Alpha Vantage error: {str(e)}")
            return pd.DataFrame(), "alpha_vantage_error"
    
    def _fetch_iex_cloud(self, symbol: str, period: str) -> Tuple[pd.DataFrame, str]:
        """Fetch data from IEX Cloud"""
        try:
            api_key = self.data_sources[DataProvider.IEX_CLOUD].api_key
            if not api_key:
                return pd.DataFrame(), "no_api_key"
            
            # Map period to IEX Cloud range
            range_map = {
                "1d": "1d",
                "5d": "5d",
                "1mo": "1m",
                "3mo": "3m",
                "6mo": "6m",
                "1y": "1y",
                "2y": "2y"
            }
            
            range_param = range_map.get(period, "1m")
            
            url = f"{self.data_sources[DataProvider.IEX_CLOUD].base_url}/stock/{symbol}/chart/{range_param}"
            params = {"token": api_key}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                return pd.DataFrame(), "iex_cloud_empty"
            
            # Parse IEX Cloud data
            df_data = []
            for item in data:
                df_data.append({
                    'Open': item.get('open', 0),
                    'High': item.get('high', 0),
                    'Low': item.get('low', 0),
                    'Close': item.get('close', 0),
                    'Volume': item.get('volume', 0),
                    'Date': pd.to_datetime(item['date'])
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            
            return df, "iex_cloud"
            
        except Exception as e:
            logger.error(f"IEX Cloud error: {str(e)}")
            return pd.DataFrame(), "iex_cloud_error"
    
    def _fetch_polygon(self, symbol: str, period: str) -> Tuple[pd.DataFrame, str]:
        """Fetch data from Polygon.io"""
        try:
            api_key = self.data_sources[DataProvider.POLYGON].api_key
            if not api_key:
                return pd.DataFrame(), "no_api_key"
            
            # Calculate date range
            end_date = datetime.now()
            if period == "1d":
                start_date = end_date - timedelta(days=1)
                multiplier, timespan = 1, "minute"
            elif period == "5d":
                start_date = end_date - timedelta(days=5)
                multiplier, timespan = 1, "minute"
            else:
                start_date = end_date - timedelta(days=365 if period == "1y" else 730)
                multiplier, timespan = 1, "day"
            
            url = f"{self.data_sources[DataProvider.POLYGON].base_url}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            params = {"apikey": api_key}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') != 'OK':
                return pd.DataFrame(), "polygon_error"
            
            results = data.get('results', [])
            if not results:
                return pd.DataFrame(), "polygon_empty"
            
            # Parse Polygon data
            df_data = []
            for item in results:
                df_data.append({
                    'Open': item.get('o', 0),
                    'High': item.get('h', 0),
                    'Low': item.get('l', 0),
                    'Close': item.get('c', 0),
                    'Volume': item.get('v', 0),
                    'Date': pd.to_datetime(item['t'], unit='ms')
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            
            return df, "polygon"
            
        except Exception as e:
            logger.error(f"Polygon error: {str(e)}")
            return pd.DataFrame(), "polygon_error"
    
    def get_market_overview_realtime(self) -> Dict[str, Any]:
        """Get real-time market overview"""
        cache_key = "market_overview_realtime"
        cached_data = self._get_cached_data(cache_key)
        
        if cached_data is not None:
            return cached_data
        
        symbols = ['^GSPC', '^IXIC', '^DJI', '^VIX', '^RUT']
        overview = {}
        
        for symbol in symbols:
            try:
                data, source = self.get_stock_data(symbol, "1d")
                if not data.empty:
                    current_price = data['Close'].iloc[-1]
                    previous_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
                    change = current_price - previous_price
                    change_percent = (change / previous_price) * 100
                    
                    overview[symbol] = {
                        'price': current_price,
                        'change': change,
                        'change_percent': change_percent,
                        'volume': data['Volume'].iloc[-1] if 'Volume' in data.columns else 0,
                        'last_updated': datetime.now().isoformat(),
                        'data_source': source
                    }
            except Exception as e:
                logger.warning(f"Failed to get data for {symbol}: {str(e)}")
                continue
        
        self._set_cached_data(cache_key, overview)
        return overview
    
    def get_live_price(self, symbol: str) -> Dict[str, Any]:
        """Get live price for a single symbol"""
        cache_key = f"live_price_{symbol}"
        cached_data = self._get_cached_data(cache_key)
        
        if cached_data is not None:
            return cached_data
        
        try:
            data, source = self.get_stock_data(symbol, "1d")
            if not data.empty:
                current_price = data['Close'].iloc[-1]
                previous_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
                change = current_price - previous_price
                change_percent = (change / previous_price) * 100
                
                result = {
                    'symbol': symbol,
                    'price': current_price,
                    'change': change,
                    'change_percent': change_percent,
                    'volume': data['Volume'].iloc[-1] if 'Volume' in data.columns else 0,
                    'last_updated': datetime.now().isoformat(),
                    'data_source': source
                }
                
                self._set_cached_data(cache_key, result)
                return result
        except Exception as e:
            logger.error(f"Failed to get live price for {symbol}: {str(e)}")
        
        return {
            'symbol': symbol,
            'price': 0,
            'change': 0,
            'change_percent': 0,
            'volume': 0,
            'last_updated': datetime.now().isoformat(),
            'data_source': 'error'
        }
    
    def add_price_alert(self, symbol: str, target_price: float, condition: str = "above"):
        """Add price alert for a symbol"""
        alert_id = f"{symbol}_{target_price}_{condition}_{int(time.time())}"
        self.price_alerts[alert_id] = {
            'symbol': symbol,
            'target_price': target_price,
            'condition': condition,
            'created_at': datetime.now(),
            'triggered': False
        }
        return alert_id
    
    def check_price_alerts(self) -> List[Dict[str, Any]]:
        """Check all price alerts and return triggered ones"""
        triggered_alerts = []
        
        for alert_id, alert in self.price_alerts.items():
            if alert['triggered']:
                continue
                
            try:
                live_data = self.get_live_price(alert['symbol'])
                current_price = live_data['price']
                target_price = alert['target_price']
                condition = alert['condition']
                
                triggered = False
                if condition == "above" and current_price >= target_price:
                    triggered = True
                elif condition == "below" and current_price <= target_price:
                    triggered = True
                
                if triggered:
                    alert['triggered'] = True
                    alert['triggered_at'] = datetime.now()
                    alert['triggered_price'] = current_price
                    triggered_alerts.append(alert)
                    
            except Exception as e:
                logger.error(f"Error checking alert {alert_id}: {str(e)}")
                continue
        
        return triggered_alerts
    
    def get_data_source_status(self) -> Dict[str, Any]:
        """Get status of all data sources"""
        status = {}
        for provider, source in self.data_sources.items():
            status[provider.value] = {
                'name': source.name,
                'is_active': source.is_active,
                'rate_limit': source.rate_limit,
                'requests_made': source.requests_made,
                'has_api_key': bool(source.api_key),
                'last_request': source.last_request
            }
        return status

# Global instance
realtime_service = RealTimeDataService()

# Streamlit integration functions
@st.cache_data(ttl=30)  # Cache for 30 seconds
def get_cached_stock_data(symbol: str, period: str = "1d"):
    """Cached wrapper for stock data"""
    return realtime_service.get_stock_data(symbol, period)

@st.cache_data(ttl=10)  # Cache for 10 seconds
def get_cached_market_overview():
    """Cached wrapper for market overview"""
    return realtime_service.get_market_overview_realtime()

@st.cache_data(ttl=5)  # Cache for 5 seconds
def get_cached_live_price(symbol: str):
    """Cached wrapper for live price"""
    return realtime_service.get_live_price(symbol)
