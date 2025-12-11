#!/usr/bin/env python3
"""
WebSocket connection manager for real-time updates
"""

from fastapi import WebSocket
from typing import List, Dict, Set
import json
import logging
import asyncio

logger = logging.getLogger(__name__)

class ConnectionManager:
    def __init__(self):
        """Initialize connection manager"""
        self.active_connections: List[WebSocket] = []
        self.user_connections: Dict[int, List[WebSocket]] = {}
        self.user_subscriptions: Dict[int, Set[str]] = {}
        self.connection_lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket, user_id: int):
        """Accept WebSocket connection and register user"""
        await websocket.accept()
        
        async with self.connection_lock:
            self.active_connections.append(websocket)
            
            if user_id not in self.user_connections:
                self.user_connections[user_id] = []
                self.user_subscriptions[user_id] = set()
            
            self.user_connections[user_id].append(websocket)
        
        logger.info(f"User {user_id} connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket, user_id: int):
        """Remove WebSocket connection"""
        try:
            self.active_connections.remove(websocket)
            
            if user_id in self.user_connections:
                self.user_connections[user_id].remove(websocket)
                
                # Clean up if no more connections for this user
                if not self.user_connections[user_id]:
                    del self.user_connections[user_id]
                    if user_id in self.user_subscriptions:
                        del self.user_subscriptions[user_id]
            
            logger.info(f"User {user_id} disconnected. Total connections: {len(self.active_connections)}")
        except ValueError:
            # Connection already removed
            pass
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to specific WebSocket connection"""
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
    
    async def send_to_user(self, user_id: int, message: str):
        """Send message to all connections of a specific user"""
        if user_id in self.user_connections:
            for websocket in self.user_connections[user_id].copy():
                try:
                    await websocket.send_text(message)
                except Exception as e:
                    logger.error(f"Error sending to user {user_id}: {e}")
                    # Remove broken connection
                    self.disconnect(websocket, user_id)
    
    async def broadcast(self, message: str):
        """Broadcast message to all connected clients"""
        for websocket in self.active_connections.copy():
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")
                # Remove broken connection
                self.active_connections.remove(websocket)
    
    async def broadcast_market_data(self, market_data: dict):
        """Broadcast market data to all connected clients"""
        message = json.dumps({
            "type": "market_data",
            "data": market_data,
            "timestamp": asyncio.get_event_loop().time()
        })
        await self.broadcast(message)
    
    async def broadcast_portfolio_update(self, user_id: int):
        """Broadcast portfolio update to specific user"""
        message = json.dumps({
            "type": "portfolio_update",
            "message": "Portfolio updated",
            "timestamp": asyncio.get_event_loop().time()
        })
        await self.send_to_user(user_id, message)
    
    async def broadcast_price_alert(self, user_id: int, symbol: str, price: float, alert_type: str):
        """Broadcast price alert to specific user"""
        message = json.dumps({
            "type": "price_alert",
            "data": {
                "symbol": symbol,
                "price": price,
                "alert_type": alert_type
            },
            "timestamp": asyncio.get_event_loop().time()
        })
        await self.send_to_user(user_id, message)
    
    async def subscribe_user(self, user_id: int, symbols: List[str]):
        """Subscribe user to specific symbol updates"""
        if user_id not in self.user_subscriptions:
            self.user_subscriptions[user_id] = set()
        
        self.user_subscriptions[user_id].update(symbols)
        
        # Send confirmation
        message = json.dumps({
            "type": "subscription_confirmed",
            "data": {"symbols": list(symbols)},
            "timestamp": asyncio.get_event_loop().time()
        })
        await self.send_to_user(user_id, message)
    
    async def broadcast_symbol_update(self, symbol: str, price_data: dict):
        """Broadcast symbol update to subscribed users"""
        message = json.dumps({
            "type": "symbol_update",
            "data": {
                "symbol": symbol,
                "price_data": price_data
            },
            "timestamp": asyncio.get_event_loop().time()
        })
        
        # Send to users subscribed to this symbol
        for user_id, subscriptions in self.user_subscriptions.items():
            if symbol in subscriptions:
                await self.send_to_user(user_id, message)
    
    def get_connection_stats(self) -> dict:
        """Get connection statistics"""
        return {
            "total_connections": len(self.active_connections),
            "active_users": len(self.user_connections),
            "user_connections": {
                str(user_id): len(connections) 
                for user_id, connections in self.user_connections.items()
            },
            "subscriptions": {
                str(user_id): list(symbols) 
                for user_id, symbols in self.user_subscriptions.items()
            }
        }

