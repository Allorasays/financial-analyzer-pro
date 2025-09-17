"""
WebSocket Service for Real-Time Financial Data Streaming
Provides live price updates and market data streaming
"""

import json
import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Dict, List, Callable, Any
import threading
import logging
from dataclasses import dataclass
import time

# Optional imports with graceful fallbacks
try:
    import asyncio
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class WebSocketMessage:
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    timestamp: str
    message_type: str = "price_update"

class WebSocketService:
    """WebSocket service for real-time data streaming"""
    
    def __init__(self):
        self.connections = {}
        self.subscribers = {}
        self.running = False
        self.thread = None
        self.websockets_available = WEBSOCKETS_AVAILABLE
        
    def start_websocket_server(self, host: str = "localhost", port: int = 8765):
        """Start WebSocket server in a separate thread"""
        if not self.websockets_available:
            logger.warning("WebSockets not available - skipping server start")
            return
            
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(
            target=self._run_websocket_server,
            args=(host, port),
            daemon=True
        )
        self.thread.start()
        logger.info(f"WebSocket server started on {host}:{port}")
    
    def _run_websocket_server(self, host: str, port: int):
        """Run WebSocket server"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        start_server = websockets.serve(
            self._handle_connection,
            host,
            port,
            ping_interval=20,
            ping_timeout=10
        )
        
        loop.run_until_complete(start_server)
        loop.run_forever()
    
    async def _handle_connection(self, websocket, path):
        """Handle WebSocket connection"""
        client_id = f"client_{int(time.time())}"
        self.connections[client_id] = websocket
        
        try:
            logger.info(f"Client {client_id} connected")
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._process_message(client_id, data, websocket)
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        "error": "Invalid JSON format",
                        "timestamp": datetime.now().isoformat()
                    }))
                except Exception as e:
                    logger.error(f"Error processing message from {client_id}: {str(e)}")
                    await websocket.send(json.dumps({
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }))
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_id} disconnected")
        finally:
            if client_id in self.connections:
                del self.connections[client_id]
            if client_id in self.subscribers:
                del self.subscribers[client_id]
    
    async def _process_message(self, client_id: str, data: Dict[str, Any], websocket):
        """Process incoming WebSocket message"""
        message_type = data.get("type", "unknown")
        
        if message_type == "subscribe":
            symbols = data.get("symbols", [])
            self.subscribers[client_id] = symbols
            await websocket.send(json.dumps({
                "type": "subscription_confirmed",
                "symbols": symbols,
                "timestamp": datetime.now().isoformat()
            }))
            logger.info(f"Client {client_id} subscribed to {symbols}")
            
        elif message_type == "unsubscribe":
            if client_id in self.subscribers:
                del self.subscribers[client_id]
            await websocket.send(json.dumps({
                "type": "unsubscription_confirmed",
                "timestamp": datetime.now().isoformat()
            }))
            logger.info(f"Client {client_id} unsubscribed")
            
        elif message_type == "ping":
            await websocket.send(json.dumps({
                "type": "pong",
                "timestamp": datetime.now().isoformat()
            }))
            
        else:
            await websocket.send(json.dumps({
                "error": f"Unknown message type: {message_type}",
                "timestamp": datetime.now().isoformat()
            }))
    
    async def broadcast_price_update(self, symbol: str, price_data: Dict[str, Any]):
        """Broadcast price update to all subscribers"""
        if not self.connections:
            return
            
        message = {
            "type": "price_update",
            "symbol": symbol,
            "data": price_data,
            "timestamp": datetime.now().isoformat()
        }
        
        message_json = json.dumps(message)
        disconnected_clients = []
        
        for client_id, websocket in self.connections.items():
            try:
                # Only send to clients subscribed to this symbol
                if (client_id in self.subscribers and 
                    symbol in self.subscribers[client_id]):
                    await websocket.send(message_json)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.append(client_id)
            except Exception as e:
                logger.error(f"Error sending to client {client_id}: {str(e)}")
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            if client_id in self.connections:
                del self.connections[client_id]
            if client_id in self.subscribers:
                del self.subscribers[client_id]
    
    def stop_websocket_server(self):
        """Stop WebSocket server"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        logger.info("WebSocket server stopped")

class WebSocketClient:
    """WebSocket client for connecting to real-time data"""
    
    def __init__(self, uri: str = "ws://localhost:8765"):
        self.uri = uri
        self.websocket = None
        self.running = False
        self.callbacks = {}
        
    async def connect(self):
        """Connect to WebSocket server"""
        try:
            self.websocket = await websockets.connect(self.uri)
            self.running = True
            logger.info(f"Connected to WebSocket server at {self.uri}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket server: {str(e)}")
            return False
    
    async def subscribe(self, symbols: List[str]):
        """Subscribe to price updates for symbols"""
        if not self.websocket:
            return False
            
        message = {
            "type": "subscribe",
            "symbols": symbols
        }
        
        try:
            await self.websocket.send(json.dumps(message))
            logger.info(f"Subscribed to {symbols}")
            return True
        except Exception as e:
            logger.error(f"Failed to subscribe: {str(e)}")
            return False
    
    async def unsubscribe(self):
        """Unsubscribe from all updates"""
        if not self.websocket:
            return False
            
        message = {
            "type": "unsubscribe"
        }
        
        try:
            await self.websocket.send(json.dumps(message))
            logger.info("Unsubscribed from all updates")
            return True
        except Exception as e:
            logger.error(f"Failed to unsubscribe: {str(e)}")
            return False
    
    def add_callback(self, message_type: str, callback: Callable):
        """Add callback for specific message type"""
        if message_type not in self.callbacks:
            self.callbacks[message_type] = []
        self.callbacks[message_type].append(callback)
    
    async def listen(self):
        """Listen for incoming messages"""
        if not self.websocket:
            return
            
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    message_type = data.get("type", "unknown")
                    
                    # Call registered callbacks
                    if message_type in self.callbacks:
                        for callback in self.callbacks[message_type]:
                            try:
                                callback(data)
                            except Exception as e:
                                logger.error(f"Callback error: {str(e)}")
                                
                except json.JSONDecodeError:
                    logger.error("Invalid JSON received")
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed")
        except Exception as e:
            logger.error(f"WebSocket error: {str(e)}")
        finally:
            self.running = False
    
    async def disconnect(self):
        """Disconnect from WebSocket server"""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        self.running = False
        logger.info("Disconnected from WebSocket server")

# Streamlit integration for real-time updates
class StreamlitRealTimeUpdater:
    """Streamlit integration for real-time updates"""
    
    def __init__(self):
        self.websocket_client = None
        self.update_interval = 5  # seconds
        self.last_update = {}
        
    def start_real_time_updates(self, symbols: List[str], update_callback: Callable = None):
        """Start real-time updates for symbols"""
        if not update_callback:
            update_callback = self._default_update_callback
            
        # Store callback in session state
        st.session_state.realtime_callback = update_callback
        st.session_state.realtime_symbols = symbols
        
        # Start WebSocket client in background
        if 'websocket_thread' not in st.session_state:
            st.session_state.websocket_thread = threading.Thread(
                target=self._run_websocket_client,
                daemon=True
            )
            st.session_state.websocket_thread.start()
    
    def _run_websocket_client(self):
        """Run WebSocket client in background thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def run_client():
            self.websocket_client = WebSocketClient()
            
            if await self.websocket_client.connect():
                symbols = st.session_state.get('realtime_symbols', [])
                await self.websocket_client.subscribe(symbols)
                
                # Add callback for price updates
                self.websocket_client.add_callback("price_update", self._handle_price_update)
                
                # Listen for messages
                await self.websocket_client.listen()
        
        loop.run_until_complete(run_client())
    
    def _handle_price_update(self, data: Dict[str, Any]):
        """Handle price update from WebSocket"""
        symbol = data.get("symbol")
        price_data = data.get("data", {})
        
        if symbol and price_data:
            # Store in session state for Streamlit to access
            if 'realtime_data' not in st.session_state:
                st.session_state.realtime_data = {}
            
            st.session_state.realtime_data[symbol] = {
                **price_data,
                'last_updated': datetime.now().isoformat()
            }
            
            # Call user callback if provided
            if 'realtime_callback' in st.session_state:
                try:
                    st.session_state.realtime_callback(symbol, price_data)
                except Exception as e:
                    logger.error(f"Callback error: {str(e)}")
    
    def _default_update_callback(self, symbol: str, data: Dict[str, Any]):
        """Default update callback"""
        logger.info(f"Price update for {symbol}: {data}")
    
    def get_latest_data(self, symbol: str) -> Dict[str, Any]:
        """Get latest real-time data for symbol"""
        if 'realtime_data' in st.session_state and symbol in st.session_state.realtime_data:
            return st.session_state.realtime_data[symbol]
        return {}
    
    def stop_real_time_updates(self):
        """Stop real-time updates"""
        if self.websocket_client:
            asyncio.run(self.websocket_client.disconnect())
        self.websocket_client = None

# Global instances
websocket_service = WebSocketService()
streamlit_updater = StreamlitRealTimeUpdater()

# Streamlit helper functions
def start_real_time_mode(symbols: List[str]):
    """Start real-time mode for given symbols"""
    streamlit_updater.start_real_time_updates(symbols)

def get_real_time_data(symbol: str) -> Dict[str, Any]:
    """Get real-time data for symbol"""
    return streamlit_updater.get_latest_data(symbol)

def stop_real_time_mode():
    """Stop real-time mode"""
    streamlit_updater.stop_real_time_updates()
