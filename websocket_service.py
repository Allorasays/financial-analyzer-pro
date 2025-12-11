"""
WebSocket Service for Financial Analyzer Pro
Provides real-time data streaming functionality
"""

import streamlit as st
import json
import time
import threading
from datetime import datetime
import numpy as np

class WebSocketService:
    def __init__(self):
        self.connected = False
        self.subscriptions = set()
        self.data_buffer = {}
        self.thread = None
        self.running = False
        
    def connect(self):
        """Connect to WebSocket service"""
        if not self.connected:
            self.connected = True
            self.running = True
            self.thread = threading.Thread(target=self._simulate_data_stream, daemon=True)
            self.thread.start()
            return True
        return False
    
    def disconnect(self):
        """Disconnect from WebSocket service"""
        self.connected = False
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
    
    def subscribe(self, symbol):
        """Subscribe to real-time updates for a symbol"""
        self.subscriptions.add(symbol)
        return True
    
    def unsubscribe(self, symbol):
        """Unsubscribe from updates for a symbol"""
        self.subscriptions.discard(symbol)
        return True
    
    def _simulate_data_stream(self):
        """Simulate real-time data stream"""
        while self.running:
            try:
                for symbol in self.subscriptions:
                    # Simulate price updates
                    if symbol not in self.data_buffer:
                        self.data_buffer[symbol] = {
                            'price': 100.0 + (hash(symbol) % 1000),
                            'timestamp': datetime.now()
                        }
                    
                    # Generate realistic price movement
                    current_price = self.data_buffer[symbol]['price']
                    change = np.random.normal(0, 0.02) * current_price
                    new_price = max(current_price + change, 1.0)
                    
                    self.data_buffer[symbol] = {
                        'price': new_price,
                        'change': new_price - current_price,
                        'change_percent': ((new_price - current_price) / current_price) * 100,
                        'timestamp': datetime.now(),
                        'volume': np.random.randint(1000000, 10000000)
                    }
                
                time.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                print(f"Error in data stream: {e}")
                time.sleep(5)
    
    def get_latest_data(self, symbol):
        """Get latest data for a symbol"""
        return self.data_buffer.get(symbol, None)
    
    def get_all_data(self):
        """Get all subscribed data"""
        return self.data_buffer.copy()

# Global WebSocket service instance
websocket_service = WebSocketService()

def start_real_time_mode():
    """Start real-time mode"""
    return websocket_service.connect()

def stop_real_time_mode():
    """Stop real-time mode"""
    websocket_service.disconnect()

def get_real_time_data(symbol=None):
    """Get real-time data for a symbol or all symbols"""
    if symbol:
        return websocket_service.get_latest_data(symbol)
    else:
        return websocket_service.get_all_data()

def subscribe_to_symbol(symbol):
    """Subscribe to real-time updates for a symbol"""
    return websocket_service.subscribe(symbol)

def unsubscribe_from_symbol(symbol):
    """Unsubscribe from updates for a symbol"""
    return websocket_service.unsubscribe(symbol)

def is_real_time_active():
    """Check if real-time mode is active"""
    return websocket_service.connected

def get_subscriptions():
    """Get list of subscribed symbols"""
    return list(websocket_service.subscriptions)

def get_connection_status():
    """Get WebSocket connection status"""
    return {
        'connected': websocket_service.connected,
        'subscriptions': len(websocket_service.subscriptions),
        'data_points': len(websocket_service.data_buffer)
    }

def simulate_market_data(symbols):
    """Simulate market data for multiple symbols"""
    data = {}
    for symbol in symbols:
        base_price = 100.0 + (hash(symbol) % 1000)
        change = np.random.normal(0, 0.02) * base_price
        new_price = max(base_price + change, 1.0)
        
        data[symbol] = {
            'price': new_price,
            'change': change,
            'change_percent': (change / base_price) * 100,
            'timestamp': datetime.now(),
            'volume': np.random.randint(1000000, 10000000)
        }
    
    return data

def get_market_sentiment():
    """Get simulated market sentiment"""
    return {
        'overall_sentiment': np.random.choice(['Bullish', 'Bearish', 'Neutral']),
        'fear_greed_index': np.random.randint(0, 100),
        'market_volatility': np.random.uniform(0.1, 0.5),
        'timestamp': datetime.now()
    }

def get_news_sentiment():
    """Get simulated news sentiment"""
    return {
        'positive_news': np.random.randint(0, 10),
        'negative_news': np.random.randint(0, 10),
        'neutral_news': np.random.randint(0, 15),
        'sentiment_score': np.random.uniform(-1, 1),
        'timestamp': datetime.now()
    }