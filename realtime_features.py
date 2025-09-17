import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import time
import asyncio
import json
import threading
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class RealTimeDataManager:
    """Real-time data management with caching and notifications"""
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        self.notifications = []
        self.watchlist = []
        self.price_alerts = {}
        
    def get_cached_data(self, symbol: str, period: str = "1d") -> Optional[pd.DataFrame]:
        """Get cached data if still valid"""
        cache_key = f"{symbol}_{period}"
        if cache_key in self.cache:
            data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return data
        return None
    
    def cache_data(self, symbol: str, data: pd.DataFrame, period: str = "1d"):
        """Cache data with timestamp"""
        cache_key = f"{symbol}_{period}"
        self.cache[cache_key] = (data, time.time())
    
    def get_real_time_data(self, symbol: str, period: str = "1d") -> Optional[pd.DataFrame]:
        """Get real-time data with caching"""
        # Check cache first
        cached_data = self.get_cached_data(symbol, period)
        if cached_data is not None:
            return cached_data
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            if not data.empty:
                self.cache_data(symbol, data, period)
                return data
        except Exception as e:
            st.error(f"Error fetching real-time data for {symbol}: {str(e)}")
        
        return None
    
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
    
    def check_price_alerts(self):
        """Check all price alerts and trigger notifications"""
        triggered_alerts = []
        
        for alert_id, alert in self.price_alerts.items():
            if alert['triggered']:
                continue
                
            data = self.get_real_time_data(alert['symbol'], "1d")
            if data is not None and len(data) > 0:
                current_price = data['Close'].iloc[-1]
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
                    
                    notification = {
                        'type': 'price_alert',
                        'symbol': alert['symbol'],
                        'message': f"Price alert triggered: {alert['symbol']} is {condition} ${target_price:.2f} (Current: ${current_price:.2f})",
                        'timestamp': datetime.now(),
                        'alert_id': alert_id
                    }
                    
                    self.notifications.append(notification)
                    triggered_alerts.append(alert)
        
        return triggered_alerts
    
    def get_market_overview_realtime(self) -> Dict:
        """Get real-time market overview"""
        symbols = ['^GSPC', '^IXIC', '^DJI', '^VIX', '^RUT']
        overview = {}
        
        for symbol in symbols:
            data = self.get_real_time_data(symbol, "2d")
            if data is not None and len(data) >= 2:
                current_price = data['Close'].iloc[-1]
                previous_price = data['Close'].iloc[-2]
                change = current_price - previous_price
                change_percent = (change / previous_price) * 100
                
                overview[symbol] = {
                    'price': current_price,
                    'change': change,
                    'change_percent': change_percent,
                    'volume': data['Volume'].iloc[-1] if 'Volume' in data.columns else 0,
                    'last_updated': datetime.now()
                }
        
        return overview
    
    def get_trending_stocks(self, limit: int = 10) -> List[Dict]:
        """Get trending stocks with real-time data"""
        popular_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'AMD', 'INTC']
        trending = []
        
        for symbol in popular_symbols[:limit]:
            data = self.get_real_time_data(symbol, "2d")
            if data is not None and len(data) >= 2:
                current_price = data['Close'].iloc[-1]
                previous_price = data['Close'].iloc[-2]
                change_percent = ((current_price - previous_price) / previous_price) * 100
                volume = data['Volume'].iloc[-1] if 'Volume' in data.columns else 0
                
                trending.append({
                    'symbol': symbol,
                    'price': current_price,
                    'change_percent': change_percent,
                    'volume': volume,
                    'last_updated': datetime.now()
                })
        
        # Sort by absolute change percentage
        trending.sort(key=lambda x: abs(x['change_percent']), reverse=True)
        return trending
    
    def get_volatility_ranking(self, symbols: List[str]) -> List[Dict]:
        """Get volatility ranking for given symbols"""
        volatility_data = []
        
        for symbol in symbols:
            data = self.get_real_time_data(symbol, "1mo")
            if data is not None and len(data) > 20:
                returns = data['Close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility
                
                volatility_data.append({
                    'symbol': symbol,
                    'volatility': volatility,
                    'current_price': data['Close'].iloc[-1],
                    'last_updated': datetime.now()
                })
        
        # Sort by volatility (highest first)
        volatility_data.sort(key=lambda x: x['volatility'], reverse=True)
        return volatility_data

class NotificationManager:
    """Manage notifications and alerts"""
    
    def __init__(self):
        self.notifications = []
        self.max_notifications = 100
    
    def add_notification(self, notification: Dict):
        """Add a new notification"""
        self.notifications.insert(0, notification)  # Add to beginning
        
        # Keep only the latest notifications
        if len(self.notifications) > self.max_notifications:
            self.notifications = self.notifications[:self.max_notifications]
    
    def get_recent_notifications(self, limit: int = 10) -> List[Dict]:
        """Get recent notifications"""
        return self.notifications[:limit]
    
    def clear_notifications(self):
        """Clear all notifications"""
        self.notifications = []
    
    def mark_as_read(self, notification_id: str):
        """Mark a notification as read"""
        for notification in self.notifications:
            if notification.get('id') == notification_id:
                notification['read'] = True
                break

def create_realtime_dashboard():
    """Create real-time dashboard with live updates"""
    st.markdown("""
    <style>
    .realtime-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .notification-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #007bff;
    }
    .alert-card {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #ffc107;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize managers
    if 'rt_manager' not in st.session_state:
        st.session_state.rt_manager = RealTimeDataManager()
    if 'notif_manager' not in st.session_state:
        st.session_state.notif_manager = NotificationManager()
    
    rt_manager = st.session_state.rt_manager
    notif_manager = st.session_state.notif_manager
    
    st.header("📡 Real-time Market Dashboard")
    
    # Auto-refresh toggle
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        auto_refresh = st.checkbox("Auto Refresh", value=True)
    with col2:
        refresh_interval = st.selectbox("Refresh Interval", [5, 10, 30, 60], index=1)
    with col3:
        if st.button("Refresh Now"):
            st.rerun()
    
    # Market overview
    st.subheader("📈 Live Market Overview")
    
    with st.spinner("Fetching real-time market data..."):
        market_data = rt_manager.get_market_overview_realtime()
    
    if market_data:
        col1, col2, col3, col4, col5 = st.columns(5)
        
        indices = [
            ('^GSPC', 'S&P 500', col1),
            ('^IXIC', 'NASDAQ', col2),
            ('^DJI', 'DOW', col3),
            ('^VIX', 'VIX', col4),
            ('^RUT', 'RUSSELL 2000', col5)
        ]
        
        for symbol, name, col in indices:
            with col:
                if symbol in market_data:
                    data = market_data[symbol]
                    change_color = "🟢" if data['change'] >= 0 else "🔴"
                    st.metric(
                        name,
                        f"${data['price']:.2f}",
                        f"{change_color} {data['change_percent']:+.2f}%"
                    )
                    st.caption(f"Updated: {data['last_updated'].strftime('%H:%M:%S')}")
    
    # Trending stocks
    st.subheader("🔥 Trending Stocks")
    
    trending_stocks = rt_manager.get_trending_stocks(10)
    
    if trending_stocks:
        trending_data = []
        for stock in trending_stocks:
            trending_data.append({
                'Symbol': stock['symbol'],
                'Price': f"${stock['price']:.2f}",
                'Change': f"{stock['change_percent']:+.2f}%",
                'Volume': f"{stock['volume']:,}",
                'Updated': stock['last_updated'].strftime('%H:%M:%S')
            })
        
        trending_df = pd.DataFrame(trending_data)
        st.dataframe(trending_df, use_container_width=True)
    
    # Price alerts
    st.subheader("⚠️ Price Alerts")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("**Add Price Alert**")
        alert_symbol = st.text_input("Symbol", value="AAPL")
        alert_price = st.number_input("Target Price", min_value=0.01, value=150.00, step=0.01)
        alert_condition = st.selectbox("Condition", ["above", "below"])
        
        if st.button("Add Alert"):
            alert_id = rt_manager.add_price_alert(alert_symbol, alert_price, alert_condition)
            st.success(f"Alert added for {alert_symbol} {alert_condition} ${alert_price:.2f}")
    
    with col2:
        st.write("**Active Alerts**")
        if rt_manager.price_alerts:
            for alert_id, alert in rt_manager.price_alerts.items():
                if not alert['triggered']:
                    st.write(f"• {alert['symbol']} {alert['condition']} ${alert['target_price']:.2f}")
        else:
            st.info("No active alerts")
    
    # Check alerts
    if st.button("Check Alerts"):
        triggered_alerts = rt_manager.check_price_alerts()
        if triggered_alerts:
            st.success(f"✅ {len(triggered_alerts)} alerts triggered!")
            for alert in triggered_alerts:
                st.warning(f"🚨 {alert['symbol']} is {alert['condition']} ${alert['target_price']:.2f}")
        else:
            st.info("No alerts triggered")
    
    # Notifications
    st.subheader("🔔 Notifications")
    
    recent_notifications = notif_manager.get_recent_notifications(10)
    
    if recent_notifications:
        for notification in recent_notifications:
            st.markdown(f"""
            <div class="notification-card">
                <strong>{notification['type'].replace('_', ' ').title()}</strong><br>
                {notification['message']}<br>
                <small>{notification['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No notifications")
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

def create_volatility_monitor():
    """Create volatility monitoring dashboard"""
    st.header("📊 Volatility Monitor")
    
    if 'rt_manager' not in st.session_state:
        st.session_state.rt_manager = RealTimeDataManager()
    
    rt_manager = st.session_state.rt_manager
    
    # Symbol input
    col1, col2 = st.columns([1, 3])
    with col1:
        symbols_input = st.text_input("Symbols (comma-separated)", value="AAPL,MSFT,GOOGL,AMZN,TSLA")
    with col2:
        st.write("")  # Spacer
    
    if st.button("Analyze Volatility"):
        symbols = [s.strip().upper() for s in symbols_input.split(',')]
        
        with st.spinner("Analyzing volatility..."):
            volatility_data = rt_manager.get_volatility_ranking(symbols)
        
        if volatility_data:
            st.success(f"✅ Volatility analysis complete for {len(volatility_data)} symbols")
            
            # Create volatility chart
            vol_df = pd.DataFrame(volatility_data)
            
            fig = px.bar(
                vol_df,
                x='symbol',
                y='volatility',
                title="Volatility Ranking",
                labels={'volatility': 'Volatility (%)', 'symbol': 'Symbol'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Volatility table
            st.subheader("Volatility Details")
            display_df = vol_df.copy()
            display_df['volatility'] = display_df['volatility'].apply(lambda x: f"{x:.2f}%")
            display_df['current_price'] = display_df['current_price'].apply(lambda x: f"${x:.2f}")
            display_df['last_updated'] = display_df['last_updated'].apply(lambda x: x.strftime('%H:%M:%S'))
            
            st.dataframe(display_df, use_container_width=True)
        else:
            st.warning("No volatility data available")

def create_live_portfolio_tracker():
    """Create live portfolio tracking"""
    st.header("💼 Live Portfolio Tracker")
    
    if 'rt_manager' not in st.session_state:
        st.session_state.rt_manager = RealTimeDataManager()
    
    rt_manager = st.session_state.rt_manager
    
    # Portfolio management
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Add Position")
        symbol = st.text_input("Symbol", value="AAPL")
        shares = st.number_input("Shares", min_value=1, value=10)
        price = st.number_input("Purchase Price", min_value=0.01, value=150.00, step=0.01)
        
        if st.button("Add Position"):
            if 'portfolio' not in st.session_state:
                st.session_state.portfolio = []
            
            # Get current price
            current_data = rt_manager.get_real_time_data(symbol, "1d")
            if current_data is not None:
                current_price = current_data['Close'].iloc[-1]
                position = {
                    'symbol': symbol,
                    'shares': shares,
                    'purchase_price': price,
                    'current_price': current_price,
                    'value': shares * current_price,
                    'cost_basis': shares * price,
                    'pnl': (current_price - price) * shares,
                    'pnl_percent': ((current_price - price) / price) * 100,
                    'last_updated': datetime.now()
                }
                st.session_state.portfolio.append(position)
                st.success(f"Added {shares} shares of {symbol}")
            else:
                st.error(f"Could not fetch current price for {symbol}")
    
    with col2:
        st.subheader("Portfolio Summary")
        if 'portfolio' in st.session_state and st.session_state.portfolio:
            # Update current prices
            for position in st.session_state.portfolio:
                current_data = rt_manager.get_real_time_data(position['symbol'], "1d")
                if current_data is not None:
                    current_price = current_data['Close'].iloc[-1]
                    position['current_price'] = current_price
                    position['value'] = position['shares'] * current_price
                    position['pnl'] = (current_price - position['purchase_price']) * position['shares']
                    position['pnl_percent'] = ((current_price - position['purchase_price']) / position['purchase_price']) * 100
                    position['last_updated'] = datetime.now()
            
            total_value = sum(pos['value'] for pos in st.session_state.portfolio)
            total_cost = sum(pos['cost_basis'] for pos in st.session_state.portfolio)
            total_pnl = total_value - total_cost
            total_pnl_percent = (total_pnl / total_cost) * 100 if total_cost > 0 else 0
            
            st.metric("Total Value", f"${total_value:,.2f}")
            st.metric("Total P&L", f"${total_pnl:,.2f}", f"{total_pnl_percent:+.2f}%")
            st.metric("Positions", len(st.session_state.portfolio))
        else:
            st.info("No positions in portfolio")
    
    # Portfolio table
    if 'portfolio' in st.session_state and st.session_state.portfolio:
        st.subheader("Live Portfolio Positions")
        
        portfolio_data = []
        for position in st.session_state.portfolio:
            portfolio_data.append({
                'Symbol': position['symbol'],
                'Shares': position['shares'],
                'Purchase Price': f"${position['purchase_price']:.2f}",
                'Current Price': f"${position['current_price']:.2f}",
                'Value': f"${position['value']:,.2f}",
                'P&L': f"${position['pnl']:,.2f}",
                'P&L %': f"{position['pnl_percent']:+.2f}%",
                'Updated': position['last_updated'].strftime('%H:%M:%S')
            })
        
        portfolio_df = pd.DataFrame(portfolio_data)
        st.dataframe(portfolio_df, use_container_width=True)
        
        if st.button("Clear Portfolio"):
            st.session_state.portfolio = []
            st.rerun()

# Main application
def main():
    st.set_page_config(
        page_title="Real-time Financial Dashboard",
        page_icon="📡",
        layout="wide"
    )
    
    st.title("📡 Real-time Financial Dashboard")
    
    tab1, tab2, tab3 = st.tabs(["📊 Market Overview", "📈 Volatility Monitor", "💼 Portfolio Tracker"])
    
    with tab1:
        create_realtime_dashboard()
    
    with tab2:
        create_volatility_monitor()
    
    with tab3:
        create_live_portfolio_tracker()

if __name__ == "__main__":
    main()
