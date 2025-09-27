"""
Real-time Data Service for Financial Analyzer Pro
Provides caching and real-time data functionality
"""

import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import time
import threading
from collections import defaultdict
import json

class RealtimeService:
    def __init__(self):
        self.cache = {}
        self.cache_timestamps = {}
        self.cache_duration = 300  # 5 minutes
        self.running = False
        self.thread = None
        
    def start_service(self):
        """Start the real-time service"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._update_loop, daemon=True)
            self.thread.start()
            return True
        return False
    
    def stop_service(self):
        """Stop the real-time service"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
    
    def _update_loop(self):
        """Background loop to update cached data"""
        while self.running:
            try:
                self._update_market_overview()
                time.sleep(60)  # Update every minute
            except Exception as e:
                print(f"Error in real-time update loop: {e}")
                time.sleep(60)
    
    def _update_market_overview(self):
        """Update market overview cache"""
        symbols = ['^GSPC', '^IXIC', '^DJI', '^VIX']
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="2d")
                
                if not hist.empty and len(hist) >= 2:
                    current_price = hist['Close'].iloc[-1]
                    previous_price = hist['Close'].iloc[-2]
                    change = current_price - previous_price
                    change_percent = (change / previous_price) * 100
                    
                    self.cache[f"market_{symbol}"] = {
                        'price': current_price,
                        'change': change,
                        'change_percent': change_percent,
                        'timestamp': datetime.now()
                    }
            except Exception as e:
                print(f"Error updating {symbol}: {e}")
    
    def get_cached_data(self, key):
        """Get cached data if it's still valid"""
        if key in self.cache and key in self.cache_timestamps:
            if (datetime.now() - self.cache_timestamps[key]).seconds < self.cache_duration:
                return self.cache[key]
        return None
    
    def set_cached_data(self, key, data):
        """Set cached data with timestamp"""
        self.cache[key] = data
        self.cache_timestamps[key] = datetime.now()

# Global instance
realtime_service = RealtimeService()

def get_cached_market_overview():
    """Get cached market overview data"""
    symbols = ['^GSPC', '^IXIC', '^DJI', '^VIX']
    overview = {}
    
    for symbol in symbols:
        cached_data = realtime_service.get_cached_data(f"market_{symbol}")
        if cached_data:
            overview[symbol] = cached_data
        else:
            # Fallback to direct API call
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="2d")
                
                if not hist.empty and len(hist) >= 2:
                    current_price = hist['Close'].iloc[-1]
                    previous_price = hist['Close'].iloc[-2]
                    change = current_price - previous_price
                    change_percent = (change / previous_price) * 100
                    
                    data = {
                        'price': current_price,
                        'change': change,
                        'change_percent': change_percent,
                        'timestamp': datetime.now()
                    }
                    
                    overview[symbol] = data
                    realtime_service.set_cached_data(f"market_{symbol}", data)
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
    
    return overview

def get_cached_live_price(symbol):
    """Get cached live price for a symbol"""
    cached_data = realtime_service.get_cached_data(f"price_{symbol}")
    if cached_data:
        return cached_data
    
    # Fallback to direct API call
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="2d")
        
        if not hist.empty:
            current_price = hist['Close'].iloc[-1]
            previous_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            change = current_price - previous_price
            change_percent = (change / previous_price) * 100
            
            data = {
                'price': current_price,
                'change': change,
                'change_percent': change_percent,
                'timestamp': datetime.now()
            }
            
            realtime_service.set_cached_data(f"price_{symbol}", data)
            return data
    except Exception as e:
        print(f"Error fetching price for {symbol}: {e}")
    
    return None

def get_cached_stock_data(symbol, period="1d"):
    """Get cached stock data for a symbol"""
    cache_key = f"stock_{symbol}_{period}"
    cached_data = realtime_service.get_cached_data(cache_key)
    if cached_data:
        return cached_data
    
    # Fallback to direct API call
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        
        if not data.empty:
            realtime_service.set_cached_data(cache_key, data)
            return data
    except Exception as e:
        print(f"Error fetching stock data for {symbol}: {e}")
    
    return None

def start_realtime_service():
    """Start the real-time service"""
    return realtime_service.start_service()

def stop_realtime_service():
    """Stop the real-time service"""
    realtime_service.stop_service()

def get_service_status():
    """Get the status of the real-time service"""
    return {
        'running': realtime_service.running,
        'cache_size': len(realtime_service.cache),
        'last_update': max(realtime_service.cache_timestamps.values()) if realtime_service.cache_timestamps else None
    }