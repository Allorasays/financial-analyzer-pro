#!/usr/bin/env python3
"""
Financial Analyzer Pro - Day 5: Watchlist System Enhanced
Advanced watchlist management with price alerts, categories, and performance tracking
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import warnings
import json
import time

warnings.filterwarnings('ignore')

# ML imports with graceful fallbacks
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="Financial Analyzer Pro - Day 5 Watchlist",
    page_icon="👀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with watchlist theme
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #8e44ad 0%, #9b59b6 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .watchlist-card {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .alert-card {
        background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .category-card {
        background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stock-item {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #8e44ad;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        text-align: center;
    }
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
    .alert-message {
        background: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #ffeaa7;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = []
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []
if 'watchlist_categories' not in st.session_state:
    st.session_state.watchlist_categories = ['Tech', 'Healthcare', 'Finance', 'Energy', 'Consumer']
if 'price_alerts' not in st.session_state:
    st.session_state.price_alerts = []
if 'watchlist_performance' not in st.session_state:
    st.session_state.watchlist_performance = {}

# Simple cache for performance
class SimpleCache:
    def __init__(self, max_size=30):
        self.cache = {}
        self.max_size = max_size
    
    def get(self, key):
        return self.cache.get(key)
    
    def set(self, key, value):
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[key] = value
    
    def clear(self):
        self.cache.clear()

cache = SimpleCache()

def get_stock_data(symbol, period="1d"):
    """Get current stock data with enhanced error handling"""
    cache_key = f"stock_{symbol}_{period}"
    cached_data = cache.get(cache_key)
    
    if cached_data is not None:
        return cached_data, None
    
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, timeout=10)
        if data.empty:
            return None, f"No data available for {symbol}"
        
        cache.set(cache_key, data)
        return data, None
    except Exception as e:
        return None, f"Error fetching data for {symbol}: {str(e)}"

def get_stock_info(symbol):
    """Get comprehensive stock information"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        return {
            'symbol': symbol.upper(),
            'name': info.get('longName', symbol.upper()),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'dividend_yield': info.get('dividendYield', 0),
            'beta': info.get('beta', 0),
            'website': info.get('website', ''),
            'description': info.get('longBusinessSummary', 'No description available')
        }
    except Exception as e:
        return {
            'symbol': symbol.upper(),
            'name': symbol.upper(),
            'sector': 'Unknown',
            'industry': 'Unknown',
            'market_cap': 0,
            'pe_ratio': 0,
            'dividend_yield': 0,
            'beta': 0,
            'website': '',
            'description': 'Information not available'
        }

def add_to_watchlist(symbol, category='General', notes=''):
    """Add stock to watchlist"""
    # Check if already in watchlist
    for item in st.session_state.watchlist:
        if item['symbol'] == symbol.upper():
            return False, "Stock already in watchlist"
    
    # Get stock info
    stock_info = get_stock_info(symbol)
    data, error = get_stock_data(symbol, "1d")
    
    if data is not None and not data.empty:
        current_price = data['Close'].iloc[-1]
        previous_price = data['Open'].iloc[-1] if len(data) > 0 else current_price
        change = current_price - previous_price
        change_percent = (change / previous_price * 100) if previous_price != 0 else 0
    else:
        current_price = 0
        change = 0
        change_percent = 0
    
    watchlist_item = {
        'id': f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'symbol': symbol.upper(),
        'name': stock_info['name'],
        'sector': stock_info['sector'],
        'industry': stock_info['industry'],
        'category': category,
        'current_price': current_price,
        'change': change,
        'change_percent': change_percent,
        'market_cap': stock_info['market_cap'],
        'pe_ratio': stock_info['pe_ratio'],
        'dividend_yield': stock_info['dividend_yield'],
        'beta': stock_info['beta'],
        'notes': notes,
        'date_added': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M')
    }
    
    st.session_state.watchlist.append(watchlist_item)
    return True, f"Added {symbol.upper()} to watchlist"

def remove_from_watchlist(symbol):
    """Remove stock from watchlist"""
    for i, item in enumerate(st.session_state.watchlist):
        if item['symbol'] == symbol.upper():
            st.session_state.watchlist.pop(i)
            return True, f"Removed {symbol.upper()} from watchlist"
    return False, "Stock not found in watchlist"

def update_watchlist_prices():
    """Update all watchlist stock prices"""
    for item in st.session_state.watchlist:
        data, error = get_stock_data(item['symbol'], "1d")
        if data is not None and not data.empty:
            current_price = data['Close'].iloc[-1]
            previous_price = data['Open'].iloc[-1] if len(data) > 0 else current_price
            change = current_price - previous_price
            change_percent = (change / previous_price * 100) if previous_price != 0 else 0
            
            item['current_price'] = current_price
            item['change'] = change
            item['change_percent'] = change_percent
            item['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M')

def create_price_alert(symbol, alert_type, target_price, notes=''):
    """Create a price alert for a stock"""
    alert_id = f"alert_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    alert = {
        'id': alert_id,
        'symbol': symbol.upper(),
        'alert_type': alert_type,  # 'above' or 'below'
        'target_price': target_price,
        'notes': notes,
        'created_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'status': 'active',
        'triggered': False
    }
    
    st.session_state.price_alerts.append(alert)
    return True, f"Price alert created for {symbol.upper()}"

def check_price_alerts():
    """Check if any price alerts should be triggered"""
    triggered_alerts = []
    
    for alert in st.session_state.price_alerts:
        if alert['status'] == 'active' and not alert['triggered']:
            data, error = get_stock_data(alert['symbol'], "1d")
            if data is not None and not data.empty:
                current_price = data['Close'].iloc[-1]
                
                should_trigger = False
                if alert['alert_type'] == 'above' and current_price >= alert['target_price']:
                    should_trigger = True
                elif alert['alert_type'] == 'below' and current_price <= alert['target_price']:
                    should_trigger = True
                
                if should_trigger:
                    alert['triggered'] = True
                    alert['triggered_date'] = datetime.now().strftime('%Y-%m-%d %H:%M')
                    alert['triggered_price'] = current_price
                    triggered_alerts.append(alert)
    
    return triggered_alerts

def calculate_watchlist_performance():
    """Calculate watchlist performance metrics"""
    if not st.session_state.watchlist:
        return {
            'total_stocks': 0,
            'gainers': 0,
            'losers': 0,
            'unchanged': 0,
            'avg_change': 0,
            'best_performer': None,
            'worst_performer': None
        }
    
    gainers = sum(1 for item in st.session_state.watchlist if item['change_percent'] > 0)
    losers = sum(1 for item in st.session_state.watchlist if item['change_percent'] < 0)
    unchanged = sum(1 for item in st.session_state.watchlist if item['change_percent'] == 0)
    
    avg_change = np.mean([item['change_percent'] for item in st.session_state.watchlist])
    
    best_performer = max(st.session_state.watchlist, key=lambda x: x['change_percent']) if st.session_state.watchlist else None
    worst_performer = min(st.session_state.watchlist, key=lambda x: x['change_percent']) if st.session_state.watchlist else None
    
    return {
        'total_stocks': len(st.session_state.watchlist),
        'gainers': gainers,
        'losers': losers,
        'unchanged': unchanged,
        'avg_change': avg_change,
        'best_performer': best_performer,
        'worst_performer': worst_performer
    }

def create_watchlist_chart(watchlist_data):
    """Create watchlist performance chart"""
    if not watchlist_data:
        return None
    
    try:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Price Changes', 'Sector Distribution', 'Performance by Category', 'Market Cap Distribution'),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
    
        symbols = [item['symbol'] for item in watchlist_data]
        changes = [item['change_percent'] for item in watchlist_data]
        colors = ['green' if change > 0 else 'red' if change < 0 else 'gray' for change in changes]
        
        # Price changes bar chart
        fig.add_trace(go.Bar(
            x=symbols,
            y=changes,
            name="Price Change %",
            marker_color=colors
        ), row=1, col=1)
        
        # Sector distribution pie chart
        sectors = {}
        for item in watchlist_data:
            sector = item['sector']
            sectors[sector] = sectors.get(sector, 0) + 1
        
        if sectors:
            fig.add_trace(go.Pie(
                labels=list(sectors.keys()),
                values=list(sectors.values()),
                name="Sector Distribution"
            ), row=1, col=2)
        
        # Performance by category
        categories = {}
        for item in watchlist_data:
            category = item['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(item['change_percent'])
        
        category_avg = {cat: np.mean(changes) for cat, changes in categories.items()}
        
        fig.add_trace(go.Bar(
            x=list(category_avg.keys()),
            y=list(category_avg.values()),
            name="Avg Performance by Category"
        ), row=2, col=1)
        
        # Market cap distribution (simplified)
        market_caps = [item['market_cap'] for item in watchlist_data if item['market_cap'] > 0]
        if market_caps:
            fig.add_trace(go.Bar(
                x=symbols[:len(market_caps)],
                y=market_caps,
                name="Market Cap"
            ), row=2, col=2)
        
        fig.update_layout(
            title="Watchlist Performance Dashboard",
            height=600,
            showlegend=True
        )
        
        return fig
    except Exception as e:
        st.error(f"Chart creation failed: {str(e)}")
        return None

def display_watchlist_management():
    """Display comprehensive watchlist management interface"""
    st.markdown("""
    <div class="watchlist-card">
        <h2>👀 Watchlist System - Day 5 Enhanced</h2>
        <p>Advanced watchlist management with price alerts, categories, and performance tracking</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Watchlist view selection
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        view_mode = st.selectbox("View Mode", ["All Stocks", "By Category", "Performance", "Alerts"], index=0)
    
    with col2:
        if st.button("🔄 Update Prices", type="primary"):
            with st.spinner("Updating watchlist prices..."):
                update_watchlist_prices()
                st.success("Prices updated!")
                st.rerun()
    
    with col3:
        if st.button("🔔 Check Alerts"):
            triggered_alerts = check_price_alerts()
            if triggered_alerts:
                st.warning(f"🚨 {len(triggered_alerts)} alerts triggered!")
                for alert in triggered_alerts:
                    st.write(f"**{alert['symbol']}** {alert['alert_type']} ${alert['target_price']:.2f} - Current: ${alert['triggered_price']:.2f}")
            else:
                st.info("No alerts triggered")
    
    # Add stock to watchlist
    with st.expander("➕ Add Stock to Watchlist", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            symbol = st.text_input("Stock Symbol", value="AAPL", key="add_watchlist_symbol", placeholder="e.g., AAPL")
        with col2:
            category = st.selectbox("Category", st.session_state.watchlist_categories + ['Custom'], key="add_watchlist_category")
            if category == 'Custom':
                category = st.text_input("Custom Category", key="custom_category")
        with col3:
            notes = st.text_input("Notes", key="add_watchlist_notes", placeholder="Optional notes")
        with col4:
            if st.button("Add to Watchlist", type="primary"):
                if symbol:
                    success, message = add_to_watchlist(symbol, category, notes)
                    if success:
                        st.success(f"✅ {message}")
                        st.rerun()
                    else:
                        st.error(f"❌ {message}")
                else:
                    st.error("❌ Please enter a stock symbol")
    
    # Add custom category
    with st.expander("📁 Manage Categories", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            new_category = st.text_input("Add New Category", key="new_category")
            if st.button("Add Category"):
                if new_category and new_category not in st.session_state.watchlist_categories:
                    st.session_state.watchlist_categories.append(new_category)
                    st.success(f"✅ Added category: {new_category}")
                    st.rerun()
                elif new_category in st.session_state.watchlist_categories:
                    st.error("❌ Category already exists")
        
        with col2:
            st.write("**Current Categories:**")
            for category in st.session_state.watchlist_categories:
                st.write(f"• {category}")
    
    # Display watchlist based on view mode
    if st.session_state.watchlist:
        # Watchlist performance summary
        performance = calculate_watchlist_performance()
        
        st.subheader("📊 Watchlist Performance Summary")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Stocks", performance['total_stocks'])
        with col2:
            st.metric("Gainers", performance['gainers'])
        with col3:
            st.metric("Losers", performance['losers'])
        with col4:
            st.metric("Avg Change", f"{performance['avg_change']:+.2f}%")
        with col5:
            if performance['best_performer']:
                st.metric("Best Performer", f"{performance['best_performer']['symbol']} ({performance['best_performer']['change_percent']:+.2f}%)")
        
        # Display stocks based on view mode
        if view_mode == "All Stocks":
            st.subheader("📋 All Watchlist Stocks")
            for i, item in enumerate(st.session_state.watchlist):
                with st.container():
                    col1, col2, col3, col4, col5, col6, col7 = st.columns([2, 1, 1, 1, 1, 1, 1])
                    
                    with col1:
                        st.write(f"**{item['symbol']}** - {item['name']}")
                        st.caption(f"{item['sector']} | {item['category']}")
                        if item['notes']:
                            st.caption(f"📝 {item['notes']}")
                    
                    with col2:
                        st.write(f"**${item['current_price']:.2f}**")
                    
                    with col3:
                        change_color = "🟢" if item['change_percent'] > 0 else "🔴" if item['change_percent'] < 0 else "⚪"
                        st.write(f"{change_color} **{item['change_percent']:+.2f}%**")
                    
                    with col4:
                        st.write(f"**{item['sector']}**")
                    
                    with col5:
                        st.write(f"**{item['category']}**")
                    
                    with col6:
                        if item['pe_ratio'] > 0:
                            st.write(f"P/E: {item['pe_ratio']:.1f}")
                    
                    with col7:
                        if st.button("❌", key=f"remove_watchlist_{i}", help="Remove from watchlist"):
                            remove_from_watchlist(item['symbol'])
                            st.success(f"Removed {item['symbol']}")
                            st.rerun()
        
        elif view_mode == "By Category":
            st.subheader("📁 Watchlist by Category")
            categories = {}
            for item in st.session_state.watchlist:
                if item['category'] not in categories:
                    categories[item['category']] = []
                categories[item['category']].append(item)
            
            for category, items in categories.items():
                with st.expander(f"📁 {category} ({len(items)} stocks)", expanded=True):
                    for item in items:
                        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                        
                        with col1:
                            st.write(f"**{item['symbol']}** - {item['name']}")
                            st.caption(f"{item['sector']} | {item['notes'] if item['notes'] else 'No notes'}")
                        
                        with col2:
                            st.write(f"${item['current_price']:.2f}")
                        
                        with col3:
                            change_color = "🟢" if item['change_percent'] > 0 else "🔴" if item['change_percent'] < 0 else "⚪"
                            st.write(f"{change_color} {item['change_percent']:+.2f}%")
                        
                        with col4:
                            if st.button("❌", key=f"remove_cat_{item['symbol']}", help="Remove"):
                                remove_from_watchlist(item['symbol'])
                                st.rerun()
        
        elif view_mode == "Performance":
            st.subheader("📈 Watchlist Performance Analytics")
            
            # Performance chart
            chart = create_watchlist_chart(st.session_state.watchlist)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
            
            # Top performers
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🏆 Top Performers")
                top_performers = sorted(st.session_state.watchlist, key=lambda x: x['change_percent'], reverse=True)[:5]
                for i, item in enumerate(top_performers, 1):
                    st.write(f"{i}. **{item['symbol']}**: {item['change_percent']:+.2f}%")
            
            with col2:
                st.subheader("📉 Underperformers")
                underperformers = sorted(st.session_state.watchlist, key=lambda x: x['change_percent'])[:5]
                for i, item in enumerate(underperformers, 1):
                    st.write(f"{i}. **{item['symbol']}**: {item['change_percent']:+.2f}%")
        
        elif view_mode == "Alerts":
            st.subheader("🔔 Price Alerts Management")
            
            # Create new alert
            with st.expander("➕ Create Price Alert", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    alert_symbol = st.selectbox("Stock", [item['symbol'] for item in st.session_state.watchlist], key="alert_symbol")
                with col2:
                    alert_type = st.selectbox("Alert Type", ["above", "below"], key="alert_type")
                with col3:
                    target_price = st.number_input("Target Price", min_value=0.01, value=100.0, step=0.01, key="target_price")
                with col4:
                    alert_notes = st.text_input("Notes", key="alert_notes", placeholder="Optional")
                
                if st.button("Create Alert", type="primary"):
                    if alert_symbol:
                        success, message = create_price_alert(alert_symbol, alert_type, target_price, alert_notes)
                        if success:
                            st.success(f"✅ {message}")
                            st.rerun()
                        else:
                            st.error(f"❌ {message}")
            
            # Display alerts
            if st.session_state.price_alerts:
                for i, alert in enumerate(st.session_state.price_alerts):
                    with st.container():
                        col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
                        
                        with col1:
                            status_color = "🟢" if alert['status'] == 'active' else "🔴"
                            triggered_color = "🚨" if alert['triggered'] else "⏳"
                            st.write(f"{status_color} {triggered_color} **{alert['symbol']}** {alert['alert_type']} ${alert['target_price']:.2f}")
                            st.caption(f"Created: {alert['created_date']}")
                            if alert['notes']:
                                st.caption(f"📝 {alert['notes']}")
                        
                        with col2:
                            st.write(f"**${alert['target_price']:.2f}**")
                        
                        with col3:
                            st.write(f"**{alert['alert_type']}**")
                        
                        with col4:
                            if alert['triggered']:
                                st.write(f"🚨 Triggered: {alert['triggered_date']}")
                                st.write(f"Price: ${alert['triggered_price']:.2f}")
                            else:
                                st.write("⏳ Pending")
                        
                        with col5:
                            if st.button("❌", key=f"remove_alert_{i}", help="Remove alert"):
                                st.session_state.price_alerts.pop(i)
                                st.success(f"Removed alert for {alert['symbol']}")
                                st.rerun()
            else:
                st.info("No price alerts created yet")
    
    else:
        st.info("📝 No stocks in watchlist. Add some stocks to get started!")
        
        # Show sample watchlist suggestion
        st.subheader("💡 Sample Watchlist Suggestion")
        sample_stocks = [
            {"symbol": "AAPL", "category": "Tech", "notes": "iPhone maker"},
            {"symbol": "MSFT", "category": "Tech", "notes": "Cloud leader"},
            {"symbol": "JNJ", "category": "Healthcare", "notes": "Pharmaceuticals"},
            {"symbol": "JPM", "category": "Finance", "notes": "Banking"},
            {"symbol": "XOM", "category": "Energy", "notes": "Oil & gas"}
        ]
        
        if st.button("🚀 Add Sample Watchlist"):
            for stock in sample_stocks:
                add_to_watchlist(stock['symbol'], stock['category'], stock['notes'])
            st.success("Sample watchlist added! Refresh to see your stocks.")
            st.rerun()

def main():
    """Main application"""
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📊 Financial Analyzer Pro - Day 5</h1>
        <p>Watchlist System Enhanced with Price Alerts, Categories & Performance Tracking</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Status
    st.markdown("""
    <div class="success-message">
        <h4>🚀 Day 5: Watchlist System Enhanced</h4>
        <p>✅ Add Stocks to Watchlist | ✅ Price Alerts & Notifications | ✅ Custom Categories | ✅ Performance Tracking</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Settings & Navigation")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Navigate",
        ["👀 Watchlist Management", "💼 Portfolio Management", "📈 Stock Analysis", "📊 Market Overview"],
        index=0
    )
    
    # System status
    st.sidebar.subheader("📊 System Status")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("ML Status", "🟢 Available" if SKLEARN_AVAILABLE else "🟡 Limited")
    with col2:
        st.metric("Watchlist", f"{len(st.session_state.watchlist)} stocks")
    
    # Cache info
    st.sidebar.subheader("📊 Cache Status")
    st.sidebar.metric("Cache Size", f"{len(cache.cache)}/30")
    if st.sidebar.button("Clear Cache"):
        cache.clear()
        st.sidebar.success("Cache cleared!")
    
    # Main content based on navigation
    if page == "👀 Watchlist Management":
        display_watchlist_management()
    
    elif page == "💼 Portfolio Management":
        st.subheader("💼 Portfolio Management")
        st.info("Portfolio management features from Day 4 are available. This focuses on Day 5 watchlist features.")
        
        # Quick portfolio summary
        if st.session_state.portfolio:
            st.write(f"**Portfolio Positions:** {len(st.session_state.portfolio)}")
            total_value = sum(item.get('position_value', 0) for item in st.session_state.portfolio)
            st.write(f"**Total Value:** ${total_value:,.2f}")
        else:
            st.info("No positions in portfolio")
    
    elif page == "📈 Stock Analysis":
        st.subheader("📈 Stock Analysis")
        st.info("Stock analysis features from previous days are available.")
        
        # Quick stock lookup
        col1, col2 = st.columns([2, 1])
        with col1:
            symbol = st.text_input("Enter Stock Symbol", value="AAPL")
        with col2:
            if st.button("Quick Lookup"):
                with st.spinner("Fetching data..."):
                    data, error = get_stock_data(symbol, "1d")
                    if data is not None:
                        current_price = data['Close'].iloc[-1]
                        st.success(f"{symbol}: ${current_price:.2f}")
                    else:
                        st.error(f"Error: {error}")
    
    elif page == "📊 Market Overview":
        st.subheader("📊 Market Overview")
        st.info("Market overview features from previous days are available.")
        
        # Quick market data
        if st.button("Get Market Data"):
            indices = ['^GSPC', '^IXIC', '^DJI']
            col1, col2, col3 = st.columns(3)
            
            for i, symbol in enumerate(indices):
                with [col1, col2, col3][i]:
                    data, error = get_stock_data(symbol, "1d")
                    if data is not None:
                        price = data['Close'].iloc[-1]
                        st.metric(symbol, f"${price:.2f}")
                    else:
                        st.error("Error")

if __name__ == "__main__":
    main()
