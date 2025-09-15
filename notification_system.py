import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import sqlite3
from typing import Dict, List, Optional
import json
import time
import threading
from portfolio_manager import EnhancedPortfolioManager

class RealTimeNotificationSystem:
    def __init__(self, db_path: str = 'financial_analyzer.db'):
        self.db_path = db_path
        self.portfolio_manager = EnhancedPortfolioManager(db_path)
        self.init_notification_tables()
    
    def init_notification_tables(self):
        """Initialize notification database tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Price alerts
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS price_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    symbol TEXT,
                    alert_type TEXT,
                    target_price REAL,
                    current_price REAL,
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    triggered_at TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # Portfolio alerts
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolio_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    alert_type TEXT,
                    target_value REAL,
                    current_value REAL,
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    triggered_at TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # News alerts
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS news_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    symbol TEXT,
                    keywords TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # Notification history
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS notification_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    notification_type TEXT,
                    title TEXT,
                    message TEXT,
                    symbol TEXT,
                    is_read BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # User notification preferences
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS notification_preferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    price_alerts BOOLEAN DEFAULT 1,
                    portfolio_alerts BOOLEAN DEFAULT 1,
                    news_alerts BOOLEAN DEFAULT 0,
                    email_notifications BOOLEAN DEFAULT 0,
                    push_notifications BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            conn.commit()
            conn.close()
        except Exception as e:
            st.error(f"Database initialization error: {str(e)}")
    
    def add_price_alert(self, user_id: int, symbol: str, alert_type: str, target_price: float) -> bool:
        """Add a price alert for a specific symbol"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if alert already exists
            cursor.execute('''
                SELECT id FROM price_alerts 
                WHERE user_id = ? AND symbol = ? AND alert_type = ? AND is_active = 1
            ''', (user_id, symbol, alert_type))
            
            if cursor.fetchone():
                conn.close()
                return False  # Alert already exists
            
            cursor.execute('''
                INSERT INTO price_alerts (user_id, symbol, alert_type, target_price, current_price)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, symbol, alert_type, target_price, 0.0))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            st.error(f"Error adding price alert: {str(e)}")
            return False
    
    def add_portfolio_alert(self, user_id: int, alert_type: str, target_value: float) -> bool:
        """Add a portfolio-level alert"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO portfolio_alerts (user_id, alert_type, target_value, current_value)
                VALUES (?, ?, ?, ?)
            ''', (user_id, alert_type, target_value, 0.0))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            st.error(f"Error adding portfolio alert: {str(e)}")
            return False
    
    def get_user_alerts(self, user_id: int) -> Dict:
        """Get all alerts for a user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Price alerts
            cursor.execute('''
                SELECT id, symbol, alert_type, target_price, current_price, is_active, created_at
                FROM price_alerts WHERE user_id = ? AND is_active = 1
                ORDER BY created_at DESC
            ''', (user_id,))
            price_alerts = cursor.fetchall()
            
            # Portfolio alerts
            cursor.execute('''
                SELECT id, alert_type, target_value, current_value, is_active, created_at
                FROM portfolio_alerts WHERE user_id = ? AND is_active = 1
                ORDER BY created_at DESC
            ''', (user_id,))
            portfolio_alerts = cursor.fetchall()
            
            # News alerts
            cursor.execute('''
                SELECT id, symbol, keywords, is_active, created_at
                FROM news_alerts WHERE user_id = ? AND is_active = 1
                ORDER BY created_at DESC
            ''', (user_id,))
            news_alerts = cursor.fetchall()
            
            conn.close()
            
            return {
                'price_alerts': [
                    {
                        'id': alert[0],
                        'symbol': alert[1],
                        'alert_type': alert[2],
                        'target_price': alert[3],
                        'current_price': alert[4],
                        'is_active': alert[5],
                        'created_at': alert[6]
                    }
                    for alert in price_alerts
                ],
                'portfolio_alerts': [
                    {
                        'id': alert[0],
                        'alert_type': alert[1],
                        'target_value': alert[2],
                        'current_value': alert[3],
                        'is_active': alert[4],
                        'created_at': alert[5]
                    }
                    for alert in portfolio_alerts
                ],
                'news_alerts': [
                    {
                        'id': alert[0],
                        'symbol': alert[1],
                        'keywords': alert[2],
                        'is_active': alert[3],
                        'created_at': alert[4]
                    }
                    for alert in news_alerts
                ]
            }
        except Exception as e:
            st.error(f"Error fetching alerts: {str(e)}")
            return {'price_alerts': [], 'portfolio_alerts': [], 'news_alerts': []}
    
    def check_price_alerts(self, user_id: int) -> List[Dict]:
        """Check and return triggered price alerts"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, symbol, alert_type, target_price
                FROM price_alerts WHERE user_id = ? AND is_active = 1
            ''', (user_id,))
            
            alerts = cursor.fetchall()
            triggered_alerts = []
            
            for alert in alerts:
                alert_id, symbol, alert_type, target_price = alert
                
                # Get current price
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="1d")
                    current_price = hist['Close'].iloc[-1] if not hist.empty else 0
                except:
                    current_price = 0
                
                # Check if alert should trigger
                should_trigger = False
                if alert_type == 'PRICE_ABOVE' and current_price > target_price:
                    should_trigger = True
                elif alert_type == 'PRICE_BELOW' and current_price < target_price:
                    should_trigger = True
                elif alert_type == 'PRICE_CHANGE_UP' and current_price > target_price:
                    should_trigger = True
                elif alert_type == 'PRICE_CHANGE_DOWN' and current_price < target_price:
                    should_trigger = True
                
                if should_trigger:
                    # Update current price and mark as triggered
                    cursor.execute('''
                        UPDATE price_alerts 
                        SET current_price = ?, triggered_at = CURRENT_TIMESTAMP, is_active = 0
                        WHERE id = ?
                    ''', (current_price, alert_id))
                    
                    # Add to notification history
                    cursor.execute('''
                        INSERT INTO notification_history (user_id, notification_type, title, message, symbol)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (user_id, 'PRICE_ALERT', f"Price Alert: {symbol}", 
                          f"{symbol} {alert_type.replace('_', ' ').lower()} {current_price:.2f} (target: {target_price:.2f})", symbol))
                    
                    triggered_alerts.append({
                        'symbol': symbol,
                        'alert_type': alert_type,
                        'target_price': target_price,
                        'current_price': current_price,
                        'message': f"{symbol} {alert_type.replace('_', ' ').lower()} {current_price:.2f}"
                    })
            
            conn.commit()
            conn.close()
            return triggered_alerts
            
        except Exception as e:
            st.error(f"Error checking price alerts: {str(e)}")
            return []
    
    def check_portfolio_alerts(self, user_id: int) -> List[Dict]:
        """Check and return triggered portfolio alerts"""
        try:
            # Get portfolio summary
            portfolio_summary = self.portfolio_manager.get_portfolio_summary(user_id)
            
            if not portfolio_summary:
                return []
            
            total_value = portfolio_summary.get('total_value', 0)
            total_gain_loss = portfolio_summary.get('total_gain_loss', 0)
            total_gain_loss_pct = portfolio_summary.get('total_gain_loss_pct', 0)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, alert_type, target_value
                FROM portfolio_alerts WHERE user_id = ? AND is_active = 1
            ''', (user_id,))
            
            alerts = cursor.fetchall()
            triggered_alerts = []
            
            for alert in alerts:
                alert_id, alert_type, target_value = alert
                
                should_trigger = False
                current_value = 0
                
                if alert_type == 'TOTAL_VALUE_ABOVE' and total_value > target_value:
                    should_trigger = True
                    current_value = total_value
                elif alert_type == 'TOTAL_VALUE_BELOW' and total_value < target_value:
                    should_trigger = True
                    current_value = total_value
                elif alert_type == 'GAIN_ABOVE' and total_gain_loss > target_value:
                    should_trigger = True
                    current_value = total_gain_loss
                elif alert_type == 'GAIN_BELOW' and total_gain_loss < target_value:
                    should_trigger = True
                    current_value = total_gain_loss
                elif alert_type == 'GAIN_PCT_ABOVE' and total_gain_loss_pct > target_value:
                    should_trigger = True
                    current_value = total_gain_loss_pct
                elif alert_type == 'GAIN_PCT_BELOW' and total_gain_loss_pct < target_value:
                    should_trigger = True
                    current_value = total_gain_loss_pct
                
                if should_trigger:
                    # Mark as triggered
                    cursor.execute('''
                        UPDATE portfolio_alerts 
                        SET current_value = ?, triggered_at = CURRENT_TIMESTAMP, is_active = 0
                        WHERE id = ?
                    ''', (current_value, alert_id))
                    
                    # Add to notification history
                    cursor.execute('''
                        INSERT INTO notification_history (user_id, notification_type, title, message)
                        VALUES (?, ?, ?, ?)
                    ''', (user_id, 'PORTFOLIO_ALERT', f"Portfolio Alert: {alert_type.replace('_', ' ').title()}", 
                          f"Portfolio {alert_type.replace('_', ' ').lower()} {current_value:.2f} (target: {target_value:.2f})"))
                    
                    triggered_alerts.append({
                        'alert_type': alert_type,
                        'target_value': target_value,
                        'current_value': current_value,
                        'message': f"Portfolio {alert_type.replace('_', ' ').lower()} {current_value:.2f}"
                    })
            
            conn.commit()
            conn.close()
            return triggered_alerts
            
        except Exception as e:
            st.error(f"Error checking portfolio alerts: {str(e)}")
            return []
    
    def get_notification_history(self, user_id: int, limit: int = 50) -> List[Dict]:
        """Get notification history for a user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, notification_type, title, message, symbol, is_read, created_at
                FROM notification_history 
                WHERE user_id = ? 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (user_id, limit))
            
            notifications = cursor.fetchall()
            conn.close()
            
            return [
                {
                    'id': notif[0],
                    'notification_type': notif[1],
                    'title': notif[2],
                    'message': notif[3],
                    'symbol': notif[4],
                    'is_read': notif[5],
                    'created_at': notif[6]
                }
                for notif in notifications
            ]
        except Exception as e:
            st.error(f"Error fetching notification history: {str(e)}")
            return []
    
    def mark_notification_read(self, user_id: int, notification_id: int) -> bool:
        """Mark a notification as read"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE notification_history 
                SET is_read = 1 
                WHERE id = ? AND user_id = ?
            ''', (notification_id, user_id))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            st.error(f"Error marking notification as read: {str(e)}")
            return False
    
    def get_unread_count(self, user_id: int) -> int:
        """Get count of unread notifications"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT COUNT(*) FROM notification_history 
                WHERE user_id = ? AND is_read = 0
            ''', (user_id,))
            
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except Exception as e:
            return 0
    
    def create_alert_management_ui(self, user_id: int) -> None:
        """Create alert management user interface"""
        st.header("🔔 Alert Management")
        
        # Tabs for different alert types
        tab1, tab2, tab3, tab4 = st.tabs([
            "Price Alerts", "Portfolio Alerts", "Notification History", "Settings"
        ])
        
        with tab1:
            self.show_price_alerts_ui(user_id)
        
        with tab2:
            self.show_portfolio_alerts_ui(user_id)
        
        with tab3:
            self.show_notification_history_ui(user_id)
        
        with tab4:
            self.show_notification_settings_ui(user_id)
    
    def show_price_alerts_ui(self, user_id: int):
        """Show price alerts management UI"""
        st.subheader("📈 Price Alerts")
        
        # Add new price alert
        with st.expander("Add New Price Alert"):
            with st.form("add_price_alert"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    symbol = st.text_input("Symbol", placeholder="AAPL").upper()
                with col2:
                    alert_type = st.selectbox("Alert Type", [
                        "PRICE_ABOVE", "PRICE_BELOW", "PRICE_CHANGE_UP", "PRICE_CHANGE_DOWN"
                    ])
                with col3:
                    target_price = st.number_input("Target Price", min_value=0.0, value=0.0)
                
                if st.form_submit_button("Add Alert"):
                    if symbol and target_price > 0:
                        if self.add_price_alert(user_id, symbol, alert_type, target_price):
                            st.success(f"Price alert added for {symbol}")
                            st.rerun()
                        else:
                            st.error("Failed to add alert or alert already exists")
                    else:
                        st.error("Please fill in all fields")
        
        # Show existing price alerts
        alerts = self.get_user_alerts(user_id)
        price_alerts = alerts.get('price_alerts', [])
        
        if price_alerts:
            st.subheader("Active Price Alerts")
            
            for alert in price_alerts:
                col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                
                with col1:
                    st.write(f"**{alert['symbol']}**")
                with col2:
                    st.write(f"{alert['alert_type'].replace('_', ' ').title()}")
                with col3:
                    st.write(f"Target: ${alert['target_price']:.2f}")
                with col4:
                    if st.button("Delete", key=f"delete_price_{alert['id']}"):
                        # Delete alert logic here
                        st.rerun()
        else:
            st.info("No price alerts set. Add some to get notified about price movements!")
    
    def show_portfolio_alerts_ui(self, user_id: int):
        """Show portfolio alerts management UI"""
        st.subheader("💼 Portfolio Alerts")
        
        # Add new portfolio alert
        with st.expander("Add New Portfolio Alert"):
            with st.form("add_portfolio_alert"):
                col1, col2 = st.columns(2)
                
                with col1:
                    alert_type = st.selectbox("Alert Type", [
                        "TOTAL_VALUE_ABOVE", "TOTAL_VALUE_BELOW", 
                        "GAIN_ABOVE", "GAIN_BELOW",
                        "GAIN_PCT_ABOVE", "GAIN_PCT_BELOW"
                    ])
                with col2:
                    target_value = st.number_input("Target Value", value=0.0)
                
                if st.form_submit_button("Add Alert"):
                    if target_value != 0:
                        if self.add_portfolio_alert(user_id, alert_type, target_value):
                            st.success("Portfolio alert added")
                            st.rerun()
                        else:
                            st.error("Failed to add alert")
                    else:
                        st.error("Please enter a target value")
        
        # Show existing portfolio alerts
        alerts = self.get_user_alerts(user_id)
        portfolio_alerts = alerts.get('portfolio_alerts', [])
        
        if portfolio_alerts:
            st.subheader("Active Portfolio Alerts")
            
            for alert in portfolio_alerts:
                col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                
                with col1:
                    st.write(f"**{alert['alert_type'].replace('_', ' ').title()}**")
                with col2:
                    st.write(f"Target: ${alert['target_value']:.2f}")
                with col3:
                    st.write(f"Current: ${alert['current_value']:.2f}")
                with col4:
                    if st.button("Delete", key=f"delete_portfolio_{alert['id']}"):
                        # Delete alert logic here
                        st.rerun()
        else:
            st.info("No portfolio alerts set. Add some to monitor your portfolio performance!")
    
    def show_notification_history_ui(self, user_id: int):
        """Show notification history UI"""
        st.subheader("📋 Notification History")
        
        notifications = self.get_notification_history(user_id)
        
        if notifications:
            for notif in notifications:
                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.write(f"**{notif['title']}**")
                        st.write(notif['message'])
                        st.write(f"*{notif['created_at']}*")
                    
                    with col2:
                        if notif['symbol']:
                            st.write(f"Symbol: {notif['symbol']}")
                    
                    with col3:
                        if not notif['is_read']:
                            if st.button("Mark Read", key=f"read_{notif['id']}"):
                                self.mark_notification_read(user_id, notif['id'])
                                st.rerun()
        else:
            st.info("No notifications yet. Set up some alerts to get started!")
    
    def show_notification_settings_ui(self, user_id: int):
        """Show notification settings UI"""
        st.subheader("⚙️ Notification Settings")
        
        st.write("Configure your notification preferences:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            price_alerts = st.checkbox("Price Alerts", value=True)
            portfolio_alerts = st.checkbox("Portfolio Alerts", value=True)
        
        with col2:
            news_alerts = st.checkbox("News Alerts", value=False)
            email_notifications = st.checkbox("Email Notifications", value=False)
        
        if st.button("Save Settings"):
            st.success("Settings saved!")
    
    def run_real_time_checks(self, user_id: int) -> List[Dict]:
        """Run real-time checks for all alerts"""
        try:
            triggered_alerts = []
            
            # Check price alerts
            price_alerts = self.check_price_alerts(user_id)
            triggered_alerts.extend(price_alerts)
            
            # Check portfolio alerts
            portfolio_alerts = self.check_portfolio_alerts(user_id)
            triggered_alerts.extend(portfolio_alerts)
            
            return triggered_alerts
            
        except Exception as e:
            st.error(f"Error running real-time checks: {str(e)}")
            return []





