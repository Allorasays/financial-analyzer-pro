#!/usr/bin/env python3
"""
Real-time service for live portfolio updates and symbol-specific subscriptions
"""

import asyncio
import json
import logging
from typing import Dict, List, Set, Optional
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
from cache_service import CacheService
from websocket_manager import ConnectionManager
from portfolio_manager import EnhancedPortfolioManager

logger = logging.getLogger(__name__)

class RealTimeService:
    def __init__(self):
        self.cache_service = CacheService()
        self.websocket_manager = ConnectionManager()
        self.portfolio_manager = EnhancedPortfolioManager()
        
        # Symbol tracking
        self.tracked_symbols: Set[str] = set()
        self.symbol_subscribers: Dict[str, Set[int]] = {}  # symbol -> set of user_ids
        
        # Portfolio tracking
        self.tracked_portfolios: Set[int] = set()  # user_ids
        
        # Background tasks
        self.price_update_task = None
        self.portfolio_update_task = None
        
        # Update intervals
        self.price_update_interval = 30  # seconds
        self.portfolio_update_interval = 60  # seconds
    
    async def start_background_tasks(self):
        """Start all background real-time tasks"""
        if self.price_update_task is None:
            self.price_update_task = asyncio.create_task(self._price_update_loop())
            logger.info("Price update task started")
        
        if self.portfolio_update_task is None:
            self.portfolio_update_task = asyncio.create_task(self._portfolio_update_loop())
            logger.info("Portfolio update task started")
    
    async def stop_background_tasks(self):
        """Stop all background real-time tasks"""
        if self.price_update_task:
            self.price_update_task.cancel()
            try:
                await self.price_update_task
            except asyncio.CancelledError:
                pass
            self.price_update_task = None
        
        if self.portfolio_update_task:
            self.portfolio_update_task.cancel()
            try:
                await self.portfolio_update_task
            except asyncio.CancelledError:
                pass
            self.portfolio_update_task = None
        
        logger.info("Real-time service background tasks stopped")
    
    async def subscribe_to_symbol(self, user_id: int, symbol: str):
        """Subscribe user to symbol updates"""
        try:
            symbol = symbol.upper()
            
            # Add symbol to tracked symbols
            self.tracked_symbols.add(symbol)
            
            # Add user to symbol subscribers
            if symbol not in self.symbol_subscribers:
                self.symbol_subscribers[symbol] = set()
            self.symbol_subscribers[symbol].add(user_id)
            
            # Send confirmation
            await self.websocket_manager.send_to_user(user_id, json.dumps({
                "type": "subscription_confirmed",
                "data": {
                    "symbol": symbol,
                    "message": f"Subscribed to {symbol} updates"
                },
                "timestamp": datetime.now().timestamp()
            }))
            
            logger.info(f"User {user_id} subscribed to {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to subscribe user {user_id} to {symbol}: {e}")
    
    async def unsubscribe_from_symbol(self, user_id: int, symbol: str):
        """Unsubscribe user from symbol updates"""
        try:
            symbol = symbol.upper()
            
            # Remove user from symbol subscribers
            if symbol in self.symbol_subscribers:
                self.symbol_subscribers[symbol].discard(user_id)
                
                # If no more subscribers, remove symbol from tracking
                if not self.symbol_subscribers[symbol]:
                    del self.symbol_subscribers[symbol]
                    self.tracked_symbols.discard(symbol)
            
            # Send confirmation
            await self.websocket_manager.send_to_user(user_id, json.dumps({
                "type": "unsubscription_confirmed",
                "data": {
                    "symbol": symbol,
                    "message": f"Unsubscribed from {symbol} updates"
                },
                "timestamp": datetime.now().timestamp()
            }))
            
            logger.info(f"User {user_id} unsubscribed from {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to unsubscribe user {user_id} from {symbol}: {e}")
    
    async def subscribe_to_portfolio(self, user_id: int):
        """Subscribe user to portfolio updates"""
        try:
            self.tracked_portfolios.add(user_id)
            
            # Send confirmation
            await self.websocket_manager.send_to_user(user_id, json.dumps({
                "type": "portfolio_subscription_confirmed",
                "data": {
                    "message": "Subscribed to portfolio updates"
                },
                "timestamp": datetime.now().timestamp()
            }))
            
            logger.info(f"User {user_id} subscribed to portfolio updates")
            
        except Exception as e:
            logger.error(f"Failed to subscribe user {user_id} to portfolio updates: {e}")
    
    async def unsubscribe_from_portfolio(self, user_id: int):
        """Unsubscribe user from portfolio updates"""
        try:
            self.tracked_portfolios.discard(user_id)
            
            # Send confirmation
            await self.websocket_manager.send_to_user(user_id, json.dumps({
                "type": "portfolio_unsubscription_confirmed",
                "data": {
                    "message": "Unsubscribed from portfolio updates"
                },
                "timestamp": datetime.now().timestamp()
            }))
            
            logger.info(f"User {user_id} unsubscribed from portfolio updates")
            
        except Exception as e:
            logger.error(f"Failed to unsubscribe user {user_id} from portfolio updates: {e}")
    
    async def _price_update_loop(self):
        """Background task to update symbol prices"""
        while True:
            try:
                if self.tracked_symbols:
                    await self._update_symbol_prices()
                await asyncio.sleep(self.price_update_interval)
            except Exception as e:
                logger.error(f"Price update loop error: {e}")
                await asyncio.sleep(self.price_update_interval)
    
    async def _portfolio_update_loop(self):
        """Background task to update portfolio values"""
        while True:
            try:
                if self.tracked_portfolios:
                    await self._update_portfolio_values()
                await asyncio.sleep(self.portfolio_update_interval)
            except Exception as e:
                logger.error(f"Portfolio update loop error: {e}")
                await asyncio.sleep(self.portfolio_update_interval)
    
    async def _update_symbol_prices(self):
        """Update prices for all tracked symbols"""
        try:
            for symbol in list(self.tracked_symbols):
                price_data = await self._get_symbol_price_data(symbol)
                if price_data:
                    await self._broadcast_symbol_update(symbol, price_data)
                    
        except Exception as e:
            logger.error(f"Symbol price update error: {e}")
    
    async def _update_portfolio_values(self):
        """Update portfolio values for all tracked users"""
        try:
            for user_id in list(self.tracked_portfolios):
                portfolio_data = await self._get_portfolio_data(user_id)
                if portfolio_data:
                    await self._broadcast_portfolio_update(user_id, portfolio_data)
                    
        except Exception as e:
            logger.error(f"Portfolio value update error: {e}")
    
    async def _get_symbol_price_data(self, symbol: str) -> Optional[Dict]:
        """Get current price data for a symbol"""
        try:
            # Check cache first
            cache_key = f"symbol_data_{symbol}"
            cached_data = await self.cache_service.get(cache_key)
            if cached_data:
                return cached_data
            
            # Fetch from yfinance
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2d")
            
            if hist is not None and not hist.empty:
                current_price = float(hist['Close'].iloc[-1])
                previous_price = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
                change = current_price - previous_price
                change_percent = (change / previous_price * 100) if previous_price != 0 else 0
                
                price_data = {
                    "symbol": symbol,
                    "price": current_price,
                    "change": change,
                    "change_percent": change_percent,
                    "volume": int(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns else 0,
                    "timestamp": datetime.now().timestamp()
                }
                
                # Cache for 30 seconds
                await self.cache_service.set(cache_key, price_data, 30)
                return price_data
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get price data for {symbol}: {e}")
            return None
    
    async def _get_portfolio_data(self, user_id: int) -> Optional[Dict]:
        """Get current portfolio data for a user"""
        try:
            # Check cache first
            cache_key = f"portfolio_data_{user_id}"
            cached_data = await self.cache_service.get(cache_key)
            if cached_data:
                return cached_data
            
            # Get portfolio summary from enhanced portfolio manager
            portfolio_summary = self.portfolio_manager.get_portfolio_summary(user_id)
            
            if portfolio_summary:
                portfolio_data = {
                    "user_id": user_id,
                    "total_value": portfolio_summary.get("total_value", 0),
                    "total_cost": portfolio_summary.get("total_cost", 0),
                    "total_gain_loss": portfolio_summary.get("total_gain_loss", 0),
                    "total_gain_loss_pct": portfolio_summary.get("total_gain_loss_pct", 0),
                    "positions_count": len(portfolio_summary.get("positions", [])),
                    "timestamp": datetime.now().timestamp()
                }
                
                # Cache for 1 minute
                await self.cache_service.set(cache_key, portfolio_data, 60)
                return portfolio_data
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get portfolio data for user {user_id}: {e}")
            return None
    
    async def _broadcast_symbol_update(self, symbol: str, price_data: Dict):
        """Broadcast symbol update to subscribed users"""
        try:
            if symbol in self.symbol_subscribers:
                message = json.dumps({
                    "type": "symbol_update",
                    "data": price_data,
                    "timestamp": datetime.now().timestamp()
                })
                
                for user_id in self.symbol_subscribers[symbol]:
                    await self.websocket_manager.send_to_user(user_id, message)
                
                logger.debug(f"Symbol update broadcasted for {symbol} to {len(self.symbol_subscribers[symbol])} users")
                
        except Exception as e:
            logger.error(f"Failed to broadcast symbol update for {symbol}: {e}")
    
    async def _broadcast_portfolio_update(self, user_id: int, portfolio_data: Dict):
        """Broadcast portfolio update to user"""
        try:
            message = json.dumps({
                "type": "portfolio_update",
                "data": portfolio_data,
                "timestamp": datetime.now().timestamp()
            })
            
            await self.websocket_manager.send_to_user(user_id, message)
            logger.debug(f"Portfolio update broadcasted for user {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to broadcast portfolio update for user {user_id}: {e}")
    
    async def get_real_time_stats(self) -> Dict:
        """Get real-time service statistics"""
        try:
            return {
                "tracked_symbols": len(self.tracked_symbols),
                "tracked_portfolios": len(self.tracked_portfolios),
                "symbol_subscribers": {
                    symbol: len(subscribers) 
                    for symbol, subscribers in self.symbol_subscribers.items()
                },
                "price_update_interval": self.price_update_interval,
                "portfolio_update_interval": self.portfolio_update_interval,
                "price_update_task_running": self.price_update_task is not None,
                "portfolio_update_task_running": self.portfolio_update_task is not None
            }
            
        except Exception as e:
            logger.error(f"Failed to get real-time stats: {e}")
            return {"error": str(e)}
    
    async def force_update_symbol(self, symbol: str):
        """Force update for a specific symbol"""
        try:
            price_data = await self._get_symbol_price_data(symbol)
            if price_data:
                await self._broadcast_symbol_update(symbol, price_data)
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to force update symbol {symbol}: {e}")
            return False
    
    async def force_update_portfolio(self, user_id: int):
        """Force update for a specific portfolio"""
        try:
            portfolio_data = await self._get_portfolio_data(user_id)
            if portfolio_data:
                await self._broadcast_portfolio_update(user_id, portfolio_data)
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to force update portfolio for user {user_id}: {e}")
            return False
