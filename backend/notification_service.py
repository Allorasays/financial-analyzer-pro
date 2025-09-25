#!/usr/bin/env python3
"""
Notification Service for push notifications and alerts
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from database import get_db
from models import User, Position, Transaction
from cache_service import CacheService
from websocket_manager import ConnectionManager
import yfinance as yf

logger = logging.getLogger(__name__)

class NotificationService:
    def __init__(self):
        self.cache_service = CacheService()
        self.websocket_manager = ConnectionManager()
        self.active_alerts = {}  # user_id -> list of alerts
        self.alert_check_interval = 60  # Check alerts every 60 seconds
        self.background_task = None
    
    async def start_background_tasks(self):
        """Start background notification tasks"""
        if self.background_task is None:
            self.background_task = asyncio.create_task(self._alert_monitor())
            logger.info("Notification service background tasks started")
    
    async def stop_background_tasks(self):
        """Stop background notification tasks"""
        if self.background_task:
            self.background_task.cancel()
            try:
                await self.background_task
            except asyncio.CancelledError:
                pass
            self.background_task = None
            logger.info("Notification service background tasks stopped")
    
    async def create_price_alert(
        self, 
        user_id: int, 
        symbol: str, 
        alert_type: str, 
        target_price: float,
        db: Session
    ) -> bool:
        """Create a price alert for a user"""
        try:
            alert_id = f"{user_id}_{symbol}_{alert_type}_{target_price}"
            
            alert_data = {
                "id": alert_id,
                "user_id": user_id,
                "symbol": symbol,
                "alert_type": alert_type,
                "target_price": target_price,
                "created_at": datetime.now().isoformat(),
                "is_active": True
            }
            
            # Store alert in cache
            await self.cache_service.set(f"alert_{alert_id}", alert_data, 86400)  # 24 hours
            
            # Add to active alerts
            if user_id not in self.active_alerts:
                self.active_alerts[user_id] = []
            self.active_alerts[user_id].append(alert_data)
            
            logger.info(f"Price alert created for user {user_id}: {symbol} {alert_type} {target_price}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create price alert: {e}")
            return False
    
    async def create_portfolio_alert(
        self,
        user_id: int,
        alert_type: str,
        target_value: float,
        db: Session
    ) -> bool:
        """Create a portfolio-level alert"""
        try:
            alert_id = f"{user_id}_portfolio_{alert_type}_{target_value}"
            
            alert_data = {
                "id": alert_id,
                "user_id": user_id,
                "alert_type": alert_type,
                "target_value": target_value,
                "created_at": datetime.now().isoformat(),
                "is_active": True
            }
            
            # Store alert in cache
            await self.cache_service.set(f"alert_{alert_id}", alert_data, 86400)
            
            # Add to active alerts
            if user_id not in self.active_alerts:
                self.active_alerts[user_id] = []
            self.active_alerts[user_id].append(alert_data)
            
            logger.info(f"Portfolio alert created for user {user_id}: {alert_type} {target_value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create portfolio alert: {e}")
            return False
    
    async def get_user_alerts(self, user_id: int) -> List[Dict]:
        """Get all active alerts for a user"""
        try:
            if user_id in self.active_alerts:
                return self.active_alerts[user_id]
            
            # Try to load from cache
            alerts = []
            cache_keys = await self.cache_service.get(f"user_alerts_{user_id}")
            if cache_keys:
                for key in cache_keys:
                    alert_data = await self.cache_service.get(f"alert_{key}")
                    if alert_data:
                        alerts.append(alert_data)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Failed to get user alerts: {e}")
            return []
    
    async def delete_alert(self, user_id: int, alert_id: str) -> bool:
        """Delete an alert"""
        try:
            # Remove from cache
            await self.cache_service.delete(f"alert_{alert_id}")
            
            # Remove from active alerts
            if user_id in self.active_alerts:
                self.active_alerts[user_id] = [
                    alert for alert in self.active_alerts[user_id] 
                    if alert["id"] != alert_id
                ]
            
            logger.info(f"Alert {alert_id} deleted for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete alert: {e}")
            return False
    
    async def _alert_monitor(self):
        """Background task to monitor alerts"""
        while True:
            try:
                await self._check_price_alerts()
                await self._check_portfolio_alerts()
                await asyncio.sleep(self.alert_check_interval)
            except Exception as e:
                logger.error(f"Alert monitor error: {e}")
                await asyncio.sleep(self.alert_check_interval)
    
    async def _check_price_alerts(self):
        """Check price alerts for all users"""
        try:
            for user_id, alerts in self.active_alerts.items():
                for alert in alerts:
                    if alert.get("symbol") and alert.get("is_active", True):
                        await self._check_single_price_alert(alert)
        except Exception as e:
            logger.error(f"Price alert check error: {e}")
    
    async def _check_single_price_alert(self, alert: Dict):
        """Check a single price alert"""
        try:
            symbol = alert["symbol"]
            target_price = alert["target_price"]
            alert_type = alert["alert_type"]
            user_id = alert["user_id"]
            
            # Get current price
            current_price = await self._get_current_price(symbol)
            if current_price is None:
                return
            
            # Check if alert should trigger
            should_trigger = False
            if alert_type == "PRICE_ABOVE" and current_price > target_price:
                should_trigger = True
            elif alert_type == "PRICE_BELOW" and current_price < target_price:
                should_trigger = True
            elif alert_type == "PRICE_CHANGE_UP" and current_price > target_price:
                should_trigger = True
            elif alert_type == "PRICE_CHANGE_DOWN" and current_price < target_price:
                should_trigger = True
            
            if should_trigger:
                await self._trigger_price_alert(alert, current_price)
                
        except Exception as e:
            logger.error(f"Single price alert check error: {e}")
    
    async def _check_portfolio_alerts(self):
        """Check portfolio alerts for all users"""
        try:
            for user_id, alerts in self.active_alerts.items():
                for alert in alerts:
                    if alert.get("alert_type", "").startswith("PORTFOLIO_") and alert.get("is_active", True):
                        await self._check_single_portfolio_alert(alert)
        except Exception as e:
            logger.error(f"Portfolio alert check error: {e}")
    
    async def _check_single_portfolio_alert(self, alert: Dict):
        """Check a single portfolio alert"""
        try:
            user_id = alert["user_id"]
            alert_type = alert["alert_type"]
            target_value = alert["target_value"]
            
            # Get portfolio summary
            portfolio_summary = await self._get_portfolio_summary(user_id)
            if not portfolio_summary:
                return
            
            current_value = portfolio_summary.get("total_value", 0)
            current_gain_loss = portfolio_summary.get("total_gain_loss", 0)
            current_gain_loss_pct = portfolio_summary.get("total_gain_loss_pct", 0)
            
            # Check if alert should trigger
            should_trigger = False
            if alert_type == "PORTFOLIO_VALUE_ABOVE" and current_value > target_value:
                should_trigger = True
            elif alert_type == "PORTFOLIO_VALUE_BELOW" and current_value < target_value:
                should_trigger = True
            elif alert_type == "PORTFOLIO_GAIN_ABOVE" and current_gain_loss > target_value:
                should_trigger = True
            elif alert_type == "PORTFOLIO_GAIN_BELOW" and current_gain_loss < target_value:
                should_trigger = True
            elif alert_type == "PORTFOLIO_GAIN_PCT_ABOVE" and current_gain_loss_pct > target_value:
                should_trigger = True
            elif alert_type == "PORTFOLIO_GAIN_PCT_BELOW" and current_gain_loss_pct < target_value:
                should_trigger = True
            
            if should_trigger:
                await self._trigger_portfolio_alert(alert, portfolio_summary)
                
        except Exception as e:
            logger.error(f"Single portfolio alert check error: {e}")
    
    async def _trigger_price_alert(self, alert: Dict, current_price: float):
        """Trigger a price alert"""
        try:
            user_id = alert["user_id"]
            symbol = alert["symbol"]
            alert_type = alert["alert_type"]
            target_price = alert["target_price"]
            
            # Create notification message
            message = f"{symbol} {alert_type.replace('_', ' ').lower()} {current_price:.2f} (target: {target_price:.2f})"
            
            # Send WebSocket notification
            await self.websocket_manager.broadcast_price_alert(
                user_id, symbol, current_price, alert_type
            )
            
            # Send push notification (if browser supports it)
            await self._send_push_notification(user_id, f"Price Alert: {symbol}", message)
            
            # Deactivate alert
            alert["is_active"] = False
            await self.cache_service.set(f"alert_{alert['id']}", alert, 86400)
            
            logger.info(f"Price alert triggered for user {user_id}: {message}")
            
        except Exception as e:
            logger.error(f"Failed to trigger price alert: {e}")
    
    async def _trigger_portfolio_alert(self, alert: Dict, portfolio_summary: Dict):
        """Trigger a portfolio alert"""
        try:
            user_id = alert["user_id"]
            alert_type = alert["alert_type"]
            target_value = alert["target_value"]
            
            # Create notification message
            message = f"Portfolio {alert_type.replace('_', ' ').lower()} {portfolio_summary.get('total_value', 0):.2f} (target: {target_value:.2f})"
            
            # Send WebSocket notification
            await self.websocket_manager.send_to_user(user_id, json.dumps({
                "type": "portfolio_alert",
                "data": {
                    "alert_type": alert_type,
                    "target_value": target_value,
                    "current_value": portfolio_summary.get("total_value", 0)
                },
                "timestamp": datetime.now().timestamp()
            }))
            
            # Send push notification
            await self._send_push_notification(user_id, "Portfolio Alert", message)
            
            # Deactivate alert
            alert["is_active"] = False
            await self.cache_service.set(f"alert_{alert['id']}", alert, 86400)
            
            logger.info(f"Portfolio alert triggered for user {user_id}: {message}")
            
        except Exception as e:
            logger.error(f"Failed to trigger portfolio alert: {e}")
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        try:
            # Check cache first
            cache_key = f"price_{symbol}"
            cached_price = await self.cache_service.get(cache_key)
            if cached_price:
                return cached_price
            
            # Fetch from yfinance
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            if hist is not None and not hist.empty:
                current_price = float(hist['Close'].iloc[-1])
                # Cache for 1 minute
                await self.cache_service.set(cache_key, current_price, 60)
                return current_price
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get current price for {symbol}: {e}")
            return None
    
    async def _get_portfolio_summary(self, user_id: int) -> Optional[Dict]:
        """Get portfolio summary for a user"""
        try:
            # Check cache first
            cache_key = f"portfolio_summary_{user_id}"
            cached_summary = await self.cache_service.get(cache_key)
            if cached_summary:
                return cached_summary
            
            # This would integrate with the enhanced portfolio manager
            # For now, return a mock summary
            summary = {
                "total_value": 10000.0,
                "total_cost": 9500.0,
                "total_gain_loss": 500.0,
                "total_gain_loss_pct": 5.26
            }
            
            # Cache for 2 minutes
            await self.cache_service.set(cache_key, summary, 120)
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get portfolio summary for user {user_id}: {e}")
            return None
    
    async def _send_push_notification(self, user_id: int, title: str, message: str):
        """Send push notification to user"""
        try:
            # This would integrate with browser push notification API
            # For now, we'll send via WebSocket
            await self.websocket_manager.send_to_user(user_id, json.dumps({
                "type": "push_notification",
                "data": {
                    "title": title,
                    "message": message,
                    "timestamp": datetime.now().isoformat()
                }
            }))
            
            logger.info(f"Push notification sent to user {user_id}: {title}")
            
        except Exception as e:
            logger.error(f"Failed to send push notification: {e}")
    
    async def get_notification_stats(self) -> Dict:
        """Get notification service statistics"""
        try:
            total_alerts = sum(len(alerts) for alerts in self.active_alerts.values())
            active_users = len(self.active_alerts)
            
            return {
                "total_alerts": total_alerts,
                "active_users": active_users,
                "alert_check_interval": self.alert_check_interval,
                "background_task_running": self.background_task is not None
            }
            
        except Exception as e:
            logger.error(f"Failed to get notification stats: {e}")
            return {"error": str(e)}

