"""
Real-Time Dashboard Components for Financial Analyzer Pro
Provides live market data, portfolio tracking, and price alerts
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import json
from typing import Dict, List, Any
import numpy as np

from realtime_data_service import realtime_service, get_cached_market_overview, get_cached_live_price
from websocket_service import start_real_time_mode, get_real_time_data, stop_real_time_mode

class RealTimeDashboard:
    """Real-time dashboard with live updates and alerts"""
    
    def __init__(self):
        self.auto_refresh_interval = 5  # seconds
        self.last_refresh = {}
        
    def display_market_overview_realtime(self):
        """Display real-time market overview with auto-refresh"""
        st.subheader("📈 Live Market Overview")
        
        # Auto-refresh toggle
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            auto_refresh = st.checkbox("🔄 Auto-refresh (5s)", value=True)
        with col2:
            if st.button("🔄 Refresh Now"):
                st.cache_data.clear()
        with col3:
            if st.button("⏸️ Pause"):
                auto_refresh = False
        
        # Get market data
        market_data = get_cached_market_overview()
        
        if not market_data:
            st.warning("⚠️ Unable to fetch market data")
            return
        
        # Display market indices
        indices = [
            ('^GSPC', 'S&P 500', '📊'),
            ('^IXIC', 'NASDAQ', '💻'),
            ('^DJI', 'DOW', '🏭'),
            ('^VIX', 'VIX', '😰'),
            ('^RUT', 'Russell 2000', '📈')
        ]
        
        # Create columns for indices
        cols = st.columns(len(indices))
        
        for i, (symbol, name, icon) in enumerate(indices):
            with cols[i]:
                if symbol in market_data:
                    data = market_data[symbol]
                    change_color = "🟢" if data['change'] >= 0 else "🔴"
                    
                    st.metric(
                        f"{icon} {name}",
                        f"${data['price']:.2f}",
                        f"{change_color} {data['change_percent']:+.2f}%",
                        help=f"Source: {data['data_source']} | Updated: {data['last_updated']}"
                    )
                else:
                    st.metric(f"{icon} {name}", "N/A", "N/A")
        
        # Market summary
        st.markdown("---")
        self._display_market_summary(market_data)
        
        # Auto-refresh logic
        if auto_refresh:
            time.sleep(self.auto_refresh_interval)
            st.rerun()
    
    def _display_market_summary(self, market_data: Dict[str, Any]):
        """Display market summary statistics"""
        if not market_data:
            return
            
        # Calculate summary stats
        changes = [data['change_percent'] for data in market_data.values() if 'change_percent' in data]
        
        if changes:
            avg_change = np.mean(changes)
            positive_count = sum(1 for c in changes if c > 0)
            total_count = len(changes)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Average Change", f"{avg_change:+.2f}%")
            with col2:
                st.metric("Advancing", f"{positive_count}/{total_count}")
            with col3:
                st.metric("Declining", f"{total_count - positive_count}/{total_count}")
            with col4:
                market_sentiment = "🟢 Bullish" if avg_change > 0 else "🔴 Bearish" if avg_change < 0 else "⚪ Neutral"
                st.metric("Sentiment", market_sentiment)
    
    def display_live_stock_tracker(self, symbols: List[str]):
        """Display live stock price tracker"""
        st.subheader("📊 Live Stock Tracker")
        
        # Symbol input
        col1, col2 = st.columns([3, 1])
        with col1:
            new_symbol = st.text_input("Add Symbol", placeholder="e.g., AAPL, TSLA, MSFT")
        with col2:
            if st.button("Add") and new_symbol:
                if new_symbol.upper() not in symbols:
                    symbols.append(new_symbol.upper())
                    st.rerun()
        
        if not symbols:
            st.info("Add symbols to track live prices")
            return
        
        # Real-time mode toggle
        col1, col2 = st.columns([2, 1])
        with col1:
            realtime_mode = st.checkbox("🔴 Real-time Mode", value=False)
        with col2:
            if st.button("Clear All"):
                symbols.clear()
                st.rerun()
        
        # Start real-time mode if enabled
        if realtime_mode:
            start_real_time_mode(symbols)
        
        # Display live prices
        if symbols:
            self._display_live_prices(symbols, realtime_mode)
    
    def _display_live_prices(self, symbols: List[str], realtime_mode: bool = False):
        """Display live prices for symbols"""
        # Create columns for symbols
        cols = st.columns(len(symbols))
        
        for i, symbol in enumerate(symbols):
            with cols[i]:
                try:
                    if realtime_mode:
                        # Get real-time data
                        realtime_data = get_real_time_data(symbol)
                        if realtime_data:
                            price = realtime_data.get('price', 0)
                            change = realtime_data.get('change', 0)
                            change_percent = realtime_data.get('change_percent', 0)
                            last_updated = realtime_data.get('last_updated', 'Unknown')
                            data_source = "WebSocket"
                        else:
                            # Fallback to cached data
                            live_data = get_cached_live_price(symbol)
                            price = live_data.get('price', 0)
                            change = live_data.get('change', 0)
                            change_percent = live_data.get('change_percent', 0)
                            last_updated = live_data.get('last_updated', 'Unknown')
                            data_source = live_data.get('data_source', 'Unknown')
                    else:
                        # Use cached data
                        live_data = get_cached_live_price(symbol)
                        price = live_data.get('price', 0)
                        change = live_data.get('change', 0)
                        change_percent = live_data.get('change_percent', 0)
                        last_updated = live_data.get('last_updated', 'Unknown')
                        data_source = live_data.get('data_source', 'Unknown')
                    
                    if price > 0:
                        change_color = "🟢" if change >= 0 else "🔴"
                        
                        st.metric(
                            symbol,
                            f"${price:.2f}",
                            f"{change_color} {change_percent:+.2f}%",
                            help=f"Source: {data_source} | Updated: {last_updated}"
                        )
                    else:
                        st.metric(symbol, "N/A", "N/A")
                        
                except Exception as e:
                    st.error(f"Error fetching {symbol}: {str(e)}")
    
    def display_portfolio_realtime(self, portfolio: List[Dict[str, Any]]):
        """Display real-time portfolio tracking"""
        st.subheader("💼 Live Portfolio Tracking")
        
        if not portfolio:
            st.info("No positions in portfolio")
            return
        
        # Real-time portfolio toggle
        col1, col2 = st.columns([2, 1])
        with col1:
            realtime_portfolio = st.checkbox("🔴 Live Portfolio Updates", value=False)
        with col2:
            if st.button("🔄 Update Portfolio"):
                st.cache_data.clear()
        
        # Calculate real-time portfolio metrics
        total_value = 0
        total_cost = 0
        updated_positions = []
        
        for position in portfolio:
            symbol = position['symbol']
            shares = position['shares']
            cost_basis = position['cost_basis']
            
            try:
                if realtime_portfolio:
                    # Get real-time price
                    realtime_data = get_real_time_data(symbol)
                    if realtime_data and realtime_data.get('price', 0) > 0:
                        current_price = realtime_data['price']
                        data_source = "WebSocket"
                    else:
                        live_data = get_cached_live_price(symbol)
                        current_price = live_data.get('price', position.get('current_price', 0))
                        data_source = live_data.get('data_source', 'Cached')
                else:
                    live_data = get_cached_live_price(symbol)
                    current_price = live_data.get('price', position.get('current_price', 0))
                    data_source = live_data.get('data_source', 'Cached')
                
                if current_price > 0:
                    current_value = shares * current_price
                    pnl = current_value - cost_basis
                    pnl_percent = (pnl / cost_basis) * 100 if cost_basis > 0 else 0
                    
                    updated_positions.append({
                        **position,
                        'current_price': current_price,
                        'current_value': current_value,
                        'pnl': pnl,
                        'pnl_percent': pnl_percent,
                        'data_source': data_source
                    })
                    
                    total_value += current_value
                    total_cost += cost_basis
                    
            except Exception as e:
                st.warning(f"Error updating {symbol}: {str(e)}")
                updated_positions.append(position)
                total_value += position.get('value', 0)
                total_cost += cost_basis
        
        # Display portfolio summary
        total_pnl = total_value - total_cost
        total_pnl_percent = (total_pnl / total_cost) * 100 if total_cost > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Value", f"${total_value:,.2f}")
        with col2:
            st.metric("Total P&L", f"${total_pnl:,.2f}", f"{total_pnl_percent:+.2f}%")
        with col3:
            st.metric("Positions", len(updated_positions))
        with col4:
            if realtime_portfolio:
                st.metric("Mode", "🔴 Live")
            else:
                st.metric("Mode", "📊 Cached")
        
        # Display updated positions
        if updated_positions:
            st.subheader("📋 Portfolio Positions")
            
            # Create DataFrame for display
            df_data = []
            for pos in updated_positions:
                df_data.append({
                    'Symbol': pos['symbol'],
                    'Shares': pos['shares'],
                    'Cost Basis': f"${pos['cost_basis']:,.2f}",
                    'Current Price': f"${pos['current_price']:,.2f}",
                    'Current Value': f"${pos['current_value']:,.2f}",
                    'P&L': f"${pos['pnl']:,.2f}",
                    'P&L %': f"{pos['pnl_percent']:+.2f}%",
                    'Source': pos.get('data_source', 'Unknown')
                })
            
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)
            
            # Portfolio performance chart
            self._display_portfolio_chart(updated_positions)
        
        # Auto-refresh for real-time mode
        if realtime_portfolio:
            time.sleep(self.auto_refresh_interval)
            st.rerun()
    
    def _display_portfolio_chart(self, positions: List[Dict[str, Any]]):
        """Display portfolio performance chart"""
        if not positions:
            return
            
        # Create pie chart for portfolio allocation
        symbols = [pos['symbol'] for pos in positions]
        values = [pos['current_value'] for pos in positions]
        pnl_values = [pos['pnl'] for pos in positions]
        
        # Allocation pie chart
        fig_allocation = go.Figure(data=[go.Pie(
            labels=symbols,
            values=values,
            hole=0.3,
            textinfo='label+percent',
            textposition='auto'
        )])
        
        fig_allocation.update_layout(
            title="Portfolio Allocation",
            height=400
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(fig_allocation, use_container_width=True)
        
        # P&L bar chart
        with col2:
            fig_pnl = go.Figure(data=[go.Bar(
                x=symbols,
                y=pnl_values,
                marker_color=['green' if pnl >= 0 else 'red' for pnl in pnl_values],
                text=[f"${pnl:,.0f}" for pnl in pnl_values],
                textposition='auto'
            )])
            
            fig_pnl.update_layout(
                title="P&L by Position",
                xaxis_title="Symbol",
                yaxis_title="P&L ($)",
                height=400
            )
            
            st.plotly_chart(fig_pnl, use_container_width=True)
    
    def display_price_alerts(self):
        """Display price alerts management"""
        st.subheader("🔔 Price Alerts")
        
        # Add new alert
        with st.expander("➕ Add Price Alert", expanded=False):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                alert_symbol = st.text_input("Symbol", placeholder="AAPL")
            with col2:
                alert_price = st.number_input("Target Price", min_value=0.01, value=150.00, step=0.01)
            with col3:
                alert_condition = st.selectbox("Condition", ["above", "below"])
            
            if st.button("Add Alert") and alert_symbol:
                alert_id = realtime_service.add_price_alert(
                    alert_symbol.upper(), 
                    alert_price, 
                    alert_condition
                )
                st.success(f"Alert added for {alert_symbol.upper()}")
        
        # Display active alerts
        if realtime_service.price_alerts:
            st.subheader("📋 Active Alerts")
            
            alert_data = []
            for alert_id, alert in realtime_service.price_alerts.items():
                if not alert['triggered']:
                    alert_data.append({
                        'Symbol': alert['symbol'],
                        'Target Price': f"${alert['target_price']:.2f}",
                        'Condition': alert['condition'],
                        'Created': alert['created_at'].strftime('%Y-%m-%d %H:%M:%S'),
                        'Status': 'Active'
                    })
            
            if alert_data:
                df_alerts = pd.DataFrame(alert_data)
                st.dataframe(df_alerts, use_container_width=True)
            else:
                st.info("No active alerts")
        
        # Check for triggered alerts
        if st.button("🔍 Check Alerts"):
            triggered_alerts = realtime_service.check_price_alerts()
            
            if triggered_alerts:
                st.success(f"🎉 {len(triggered_alerts)} alert(s) triggered!")
                
                for alert in triggered_alerts:
                    st.alert(f"🚨 {alert['symbol']} {alert['condition']} ${alert['target_price']:.2f} (Current: ${alert['triggered_price']:.2f})")
            else:
                st.info("No alerts triggered")
    
    def display_data_source_status(self):
        """Display data source status and health"""
        st.subheader("🔧 Data Source Status")
        
        status = realtime_service.get_data_source_status()
        
        for provider, info in status.items():
            with st.expander(f"{info['name']} ({provider})", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Status:** {'🟢 Active' if info['is_active'] else '🔴 Inactive'}")
                    st.write(f"**Rate Limit:** {info['rate_limit']} req/min")
                    st.write(f"**Requests Made:** {info['requests_made']}")
                
                with col2:
                    st.write(f"**API Key:** {'✅ Present' if info['has_api_key'] else '❌ Missing'}")
                    st.write(f"**Last Request:** {datetime.fromtimestamp(info['last_request']).strftime('%H:%M:%S') if info['last_request'] > 0 else 'Never'}")
                
                # Rate limit progress bar
                if info['rate_limit'] > 0:
                    progress = min(info['requests_made'] / info['rate_limit'], 1.0)
                    st.progress(progress)
                    st.caption(f"{info['requests_made']}/{info['rate_limit']} requests used")

# Global dashboard instance
realtime_dashboard = RealTimeDashboard()

# Streamlit integration functions
def display_realtime_market_overview():
    """Display real-time market overview"""
    realtime_dashboard.display_market_overview_realtime()

def display_live_stock_tracker(symbols: List[str] = None):
    """Display live stock tracker"""
    if symbols is None:
        symbols = st.session_state.get('tracked_symbols', [])
    realtime_dashboard.display_live_stock_tracker(symbols)

def display_portfolio_realtime(portfolio: List[Dict[str, Any]] = None):
    """Display real-time portfolio tracking"""
    if portfolio is None:
        portfolio = st.session_state.get('portfolio', [])
    realtime_dashboard.display_portfolio_realtime(portfolio)

def display_price_alerts():
    """Display price alerts management"""
    realtime_dashboard.display_price_alerts()

def display_data_source_status():
    """Display data source status"""
    realtime_dashboard.display_data_source_status()
