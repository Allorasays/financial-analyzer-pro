#!/usr/bin/env python3
"""
Simple Real-Time Data Service for Financial Analyzer Pro
Uses only Yahoo Finance with intelligent caching and refresh
"""

import streamlit as st
import pandas as pd
import yfinance as yf
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import threading
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleRealtimeService:
    """Simple real-time data service using only Yahoo Finance"""
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 30  # seconds
        self.last_update = {}
        
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid"""
        if key not in self.cache or key not in self.last_update:
            return False
        return time.time() - self.last_update[key] < self.cache_ttl
    
    def get_live_price(self, symbol: str) -> Tuple[Optional[float], Optional[str]]:
        """Get live price for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", interval="1m")
            
            if data.empty:
                return None, f"No data available for {symbol}"
            
            current_price = data['Close'].iloc[-1]
            return current_price, None
            
        except Exception as e:
            return None, f"Error fetching live price for {symbol}: {str(e)}"
    
    def get_cached_live_price(self, symbol: str) -> Tuple[Optional[float], Optional[str]]:
        """Get cached live price or fetch new one"""
        cache_key = f"live_price_{symbol}"
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key], None
        
        price, error = self.get_live_price(symbol)
        if price is not None:
            self.cache[cache_key] = price
            self.last_update[cache_key] = time.time()
        
        return price, error
    
    def get_market_overview(self) -> Dict[str, Dict[str, Any]]:
        """Get market overview for major indices"""
        indices = {
            '^GSPC': 'S&P 500',
            '^IXIC': 'NASDAQ',
            '^DJI': 'DOW',
            '^VIX': 'VIX'
        }
        
        market_data = {}
        for symbol, name in indices.items():
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1d")
                
                if not data.empty:
                    current_price = data['Close'].iloc[-1]
                    previous_price = data['Open'].iloc[-1]
                    change = current_price - previous_price
                    change_percent = (change / previous_price) * 100
                    
                    market_data[symbol] = {
                        'name': name,
                        'price': current_price,
                        'change': change,
                        'change_percent': change_percent,
                        'volume': data['Volume'].iloc[-1] if 'Volume' in data.columns else 0
                    }
            except Exception as e:
                logger.warning(f"Could not fetch {name}: {str(e)}")
                continue
        
        return market_data
    
    def get_cached_market_overview(self) -> Dict[str, Dict[str, Any]]:
        """Get cached market overview or fetch new one"""
        cache_key = "market_overview"
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]
        
        market_data = self.get_market_overview()
        self.cache[cache_key] = market_data
        self.last_update[cache_key] = time.time()
        
        return market_data
    
    def get_stock_data(self, symbol: str, period: str = "1d", interval: str = "1m") -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Get stock data with specified period and interval"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                return None, f"No data available for {symbol}"
            
            return data, None
            
        except Exception as e:
            return None, f"Error fetching data for {symbol}: {str(e)}"
    
    def get_cached_stock_data(self, symbol: str, period: str = "1d", interval: str = "1m") -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Get cached stock data or fetch new one"""
        cache_key = f"stock_data_{symbol}_{period}_{interval}"
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key], None
        
        data, error = self.get_stock_data(symbol, period, interval)
        if data is not None:
            self.cache[cache_key] = data
            self.last_update[cache_key] = time.time()
        
        return data, error
    
    def get_trending_stocks(self) -> List[Dict[str, Any]]:
        """Get trending stocks (simplified version)"""
        # Popular stocks for trending
        trending_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']
        trending_data = []
        
        for symbol in trending_symbols:
            price, error = self.get_cached_live_price(symbol)
            if price is not None:
                # Get some basic info
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    
                    trending_data.append({
                        'symbol': symbol,
                        'price': price,
                        'name': info.get('longName', symbol),
                        'change': 0,  # Simplified - would need historical data for real change
                        'change_percent': 0
                    })
                except:
                    trending_data.append({
                        'symbol': symbol,
                        'price': price,
                        'name': symbol,
                        'change': 0,
                        'change_percent': 0
                    })
        
        return trending_data

# Global service instance
realtime_service = SimpleRealtimeService()

# Convenience functions
def get_live_price(symbol: str) -> Tuple[Optional[float], Optional[str]]:
    """Get live price for a symbol"""
    return realtime_service.get_cached_live_price(symbol)

def get_market_overview() -> Dict[str, Dict[str, Any]]:
    """Get market overview"""
    return realtime_service.get_cached_market_overview()

def get_stock_data(symbol: str, period: str = "1d", interval: str = "1m") -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Get stock data"""
    return realtime_service.get_cached_stock_data(symbol, period, interval)

def get_trending_stocks() -> List[Dict[str, Any]]:
    """Get trending stocks"""
    return realtime_service.get_trending_stocks()

