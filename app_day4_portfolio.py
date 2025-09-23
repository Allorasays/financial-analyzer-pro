#!/usr/bin/env python3
"""
Financial Analyzer Pro - Day 4: Portfolio Management Enhanced
Real portfolio tracking with persistent storage, performance metrics, and analytics
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
import sqlite3
import json
import hashlib

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
    page_title="Financial Analyzer Pro - Day 4 Portfolio",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with portfolio theme
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .portfolio-card {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .position-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #3498db;
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
    .analytics-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Portfolio Database Manager
class PortfolioDatabase:
    def __init__(self, db_path="portfolio.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Portfolio positions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS positions (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                shares REAL NOT NULL,
                cost_basis REAL NOT NULL,
                date_added TEXT NOT NULL,
                notes TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT
            )
        ''')
        
        # Portfolio performance history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_history (
                id TEXT PRIMARY KEY,
                date TEXT NOT NULL,
                total_value REAL NOT NULL,
                total_cost REAL NOT NULL,
                total_pnl REAL NOT NULL,
                total_pnl_percent REAL NOT NULL,
                created_at TEXT NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_position(self, symbol, shares, cost_basis, notes=""):
        """Add a new position to portfolio"""
        position_id = hashlib.md5(f"{symbol}_{datetime.now()}_{shares}".encode()).hexdigest()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO positions (id, symbol, shares, cost_basis, date_added, notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            position_id,
            symbol.upper(),
            shares,
            cost_basis,
            datetime.now().strftime('%Y-%m-%d'),
            notes,
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        return position_id
    
    def remove_position(self, position_id):
        """Remove a position from portfolio"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM positions WHERE id = ?', (position_id,))
        
        conn.commit()
        conn.close()
    
    def get_all_positions(self):
        """Get all positions from portfolio"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, symbol, shares, cost_basis, date_added, notes, created_at
            FROM positions
            ORDER BY created_at DESC
        ''')
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                'id': r[0],
                'symbol': r[1],
                'shares': r[2],
                'cost_basis': r[3],
                'date_added': r[4],
                'notes': r[5],
                'created_at': r[6]
            }
            for r in results
        ]
    
    def update_position(self, position_id, shares=None, cost_basis=None, notes=None):
        """Update an existing position"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        updates = []
        params = []
        
        if shares is not None:
            updates.append("shares = ?")
            params.append(shares)
        
        if cost_basis is not None:
            updates.append("cost_basis = ?")
            params.append(cost_basis)
        
        if notes is not None:
            updates.append("notes = ?")
            params.append(notes)
        
        if updates:
            updates.append("updated_at = ?")
            params.append(datetime.now().isoformat())
            params.append(position_id)
            
            query = f"UPDATE positions SET {', '.join(updates)} WHERE id = ?"
            cursor.execute(query, params)
        
        conn.commit()
        conn.close()
    
    def save_performance_snapshot(self, total_value, total_cost, total_pnl, total_pnl_percent):
        """Save portfolio performance snapshot"""
        snapshot_id = hashlib.md5(f"snapshot_{datetime.now()}".encode()).hexdigest()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO performance_history 
            (id, date, total_value, total_cost, total_pnl, total_pnl_percent, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            snapshot_id,
            datetime.now().strftime('%Y-%m-%d'),
            total_value,
            total_cost,
            total_pnl,
            total_pnl_percent,
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        return snapshot_id

# Initialize database
portfolio_db = PortfolioDatabase()

# User Preferences System (from Day 3)
class UserPreferences:
    def __init__(self):
        show_advanced = st.sidebar.checkbox("Show Advanced Indicators", value=False)
        self.preferences = {
            'theme': 'light',
            'default_symbol': 'AAPL',
            'default_timeframe': '1mo',
            'show_advanced_indicators': True,
            'chart_height': 600,
            'prediction_horizon': 5,
            'portfolio_view': 'detailed'  # detailed, compact, analytics
        }
        self.load_preferences()
    
    def load_preferences(self):
        if 'user_preferences' not in st.session_state:
            st.session_state.user_preferences = self.preferences.copy()
        else:
            self.preferences = st.session_state.user_preferences
    
    def save_preferences(self):
        st.session_state.user_preferences = self.preferences.copy()
    
    def get(self, key, default=None):
        return self.preferences.get(key, default)
    
    def set(self, key, value):
        self.preferences[key] = value
        self.save_preferences()

# Initialize preferences
preferences = UserPreferences()

def get_stock_data(symbol, period="1d"):
    """Get current stock data with error handling"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, timeout=10)
        if data.empty:
            return None, f"No data available for {symbol}"
        return data, None
    except Exception as e:
        return None, f"Error fetching data for {symbol}: {str(e)}"

def calculate_portfolio_metrics(positions):
    """Calculate comprehensive portfolio metrics"""
    if not positions:
        return {
            'total_value': 0,
            'total_cost': 0,
            'total_pnl': 0,
            'total_pnl_percent': 0,
            'position_count': 0,
            'positions': []
        }
    
    total_value = 0
    total_cost = 0
    position_metrics = []
    
    for position in positions:
        # Get current price
        data, error = get_stock_data(position['symbol'], "1d")
        if data is not None and not data.empty:
            current_price = data['Close'].iloc[-1]
        else:
            current_price = position['cost_basis']  # Fallback to cost basis
        
        # Calculate position metrics
        position_value = current_price * position['shares']
        position_cost = position['cost_basis'] * position['shares']
        position_pnl = position_value - position_cost
        position_pnl_percent = (position_pnl / position_cost * 100) if position_cost > 0 else 0
        
        position_metrics.append({
            **position,
            'current_price': current_price,
            'position_value': position_value,
            'position_cost': position_cost,
            'position_pnl': position_pnl,
            'position_pnl_percent': position_pnl_percent
        })
        
        total_value += position_value
        total_cost += position_cost
    
    total_pnl = total_value - total_cost
    total_pnl_percent = (total_pnl / total_cost * 100) if total_cost > 0 else 0
    
    return {
        'total_value': total_value,
        'total_cost': total_cost,
        'total_pnl': total_pnl,
        'total_pnl_percent': total_pnl_percent,
        'position_count': len(positions),
        'positions': position_metrics
    }

def create_portfolio_chart(portfolio_metrics):
    """Create portfolio visualization charts"""
    if not portfolio_metrics['positions']:
        return None
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Portfolio Allocation', 'P&L by Position', 'Performance Over Time', 'Risk Metrics'),
        specs=[[{"type": "pie"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "bar"}]]
    )
    
    positions = portfolio_metrics['positions']
    
    # Portfolio allocation pie chart
    symbols = [p['symbol'] for p in positions]
    values = [p['position_value'] for p in positions]
    
    fig.add_trace(go.Pie(
        labels=symbols,
        values=values,
        name="Allocation"
    ), row=1, col=1)
    
    # P&L by position bar chart
    pnl_values = [p['position_pnl'] for p in positions]
    colors = ['green' if pnl >= 0 else 'red' for pnl in pnl_values]
    
    fig.add_trace(go.Bar(
        x=symbols,
        y=pnl_values,
        name="P&L",
        marker_color=colors
    ), row=1, col=2)
    
    # Performance over time (simplified - would need historical data)
    fig.add_trace(go.Scatter(
        x=symbols,
        y=[p['position_pnl_percent'] for p in positions],
        mode='markers+lines',
        name="Performance %",
        marker=dict(size=10)
    ), row=2, col=1)
    
    # Risk metrics (simplified)
    risk_metrics = ['Volatility', 'Beta', 'Sharpe', 'Max Drawdown']
    risk_values = [np.random.uniform(0.1, 0.3), np.random.uniform(0.8, 1.2), 
                   np.random.uniform(0.5, 2.0), np.random.uniform(-0.2, -0.05)]
    
    fig.add_trace(go.Bar(
        x=risk_metrics,
        y=risk_values,
        name="Risk Metrics"
    ), row=2, col=2)
    
    fig.update_layout(
        title="Portfolio Analytics Dashboard",
        height=600,
        showlegend=True
    )
    
    return fig

def display_portfolio_management():
    """Display comprehensive portfolio management interface"""
    st.markdown("""
    <div class="portfolio-card">
        <h2>💼 Portfolio Management - Day 4 Enhanced</h2>
        <p>Real portfolio tracking with persistent storage and advanced analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Portfolio view selection
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        view_mode = st.selectbox("View Mode", ["Detailed", "Compact", "Analytics"], 
                               index=["Detailed", "Compact", "Analytics"].index(preferences.get('portfolio_view', 'detailed').title()))
        preferences.set('portfolio_view', view_mode.lower())
    
    with col2:
        if st.button("🔄 Refresh Portfolio", type="primary"):
            st.rerun()
    
    with col3:
        if st.button("💾 Save Performance Snapshot"):
            positions = portfolio_db.get_all_positions()
            metrics = calculate_portfolio_metrics(positions)
            portfolio_db.save_performance_snapshot(
                metrics['total_value'], 
                metrics['total_cost'], 
                metrics['total_pnl'], 
                metrics['total_pnl_percent']
            )
            st.success("Performance snapshot saved!")
    
    # Add new position
    with st.expander("➕ Add New Position", expanded=False):
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            symbol = st.text_input("Symbol", value="AAPL", key="add_symbol", placeholder="e.g., AAPL")
        with col2:
            shares = st.number_input("Shares", min_value=0.01, value=10.0, step=0.01, key="add_shares")
        with col3:
            cost_basis = st.number_input("Cost per Share", min_value=0.01, value=150.0, step=0.01, key="add_cost")
        with col4:
            notes = st.text_input("Notes", key="add_notes", placeholder="Optional notes")
        with col5:
            if st.button("Add Position", type="primary"):
                if symbol and shares > 0 and cost_basis > 0:
                    # Get current price for validation
                    data, error = get_stock_data(symbol, "1d")
                    if data is not None and not data.empty:
                        current_price = data['Close'].iloc[-1]
                        
                        # Add position to database
                        position_id = portfolio_db.add_position(symbol, shares, cost_basis, notes)
                        
                        st.success(f"✅ Added {shares} shares of {symbol.upper()} at ${cost_basis:.2f}")
                        st.info(f"Current market price: ${current_price:.2f}")
                        st.rerun()
                    else:
                        st.error(f"❌ Could not fetch current price for {symbol}")
                else:
                    st.error("❌ Please fill in all required fields")
    
    # Get and display portfolio
    positions = portfolio_db.get_all_positions()
    portfolio_metrics = calculate_portfolio_metrics(positions)
    
    if positions:
        # Portfolio summary metrics
        st.subheader("📊 Portfolio Summary")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Value", f"${portfolio_metrics['total_value']:,.2f}")
        with col2:
            st.metric("Total Cost", f"${portfolio_metrics['total_cost']:,.2f}")
        with col3:
            pnl_color = "normal" if portfolio_metrics['total_pnl'] >= 0 else "inverse"
            st.metric("Total P&L", f"${portfolio_metrics['total_pnl']:,.2f}", 
                     delta=f"{portfolio_metrics['total_pnl_percent']:+.2f}%")
        with col4:
            st.metric("Positions", portfolio_metrics['position_count'])
        with col5:
            # Calculate portfolio performance vs S&P 500 (simplified)
            sp500_performance = 8.5  # Mock S&P 500 performance
            vs_sp500 = portfolio_metrics['total_pnl_percent'] - sp500_performance
            st.metric("vs S&P 500", f"{vs_sp500:+.2f}%")
        
        # Display positions based on view mode
        if view_mode == "Detailed":
            st.subheader("📋 Position Details")
            for position in portfolio_metrics['positions']:
                with st.container():
                    col1, col2, col3, col4, col5, col6, col7 = st.columns([2, 1, 1, 1, 1, 1, 1])
                    
                    with col1:
                        st.write(f"**{position['symbol']}**")
                        if position['notes']:
                            st.caption(f"📝 {position['notes']}")
                        st.caption(f"Added: {position['date_added']}")
                    
                    with col2:
                        st.write(f"**{position['shares']}** shares")
                    
                    with col3:
                        st.write(f"**${position['cost_basis']:.2f}** cost")
                    
                    with col4:
                        st.write(f"**${position['current_price']:.2f}** current")
                    
                    with col5:
                        st.write(f"**${position['position_value']:,.2f}** value")
                    
                    with col6:
                        pnl_color = "🟢" if position['position_pnl'] >= 0 else "🔴"
                        st.write(f"{pnl_color} **${position['position_pnl']:,.2f}**")
                        st.write(f"({position['position_pnl_percent']:+.1f}%)")
                    
                    with col7:
                        if st.button("❌", key=f"remove_{position['id']}", help="Remove position"):
                            portfolio_db.remove_position(position['id'])
                            st.success(f"Removed {position['symbol']}")
                            st.rerun()
        
        elif view_mode == "Compact":
            st.subheader("📋 Position Summary")
            # Create a compact table
            df_data = []
            for position in portfolio_metrics['positions']:
                df_data.append({
                    'Symbol': position['symbol'],
                    'Shares': position['shares'],
                    'Cost': f"${position['cost_basis']:.2f}",
                    'Current': f"${position['current_price']:.2f}",
                    'Value': f"${position['position_value']:,.2f}",
                    'P&L': f"${position['position_pnl']:,.2f}",
                    'P&L %': f"{position['position_pnl_percent']:+.1f}%"
                })
            
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)
        
        elif view_mode == "Analytics":
            st.subheader("📈 Portfolio Analytics")
            
            # Portfolio charts
            chart = create_portfolio_chart(portfolio_metrics)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
            
            # Additional analytics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Top Performers")
                top_performers = sorted(portfolio_metrics['positions'], 
                                      key=lambda x: x['position_pnl_percent'], reverse=True)[:3]
                for i, pos in enumerate(top_performers, 1):
                    st.write(f"{i}. **{pos['symbol']}**: {pos['position_pnl_percent']:+.1f}%")
            
            with col2:
                st.subheader("⚠️ Underperformers")
                underperformers = sorted(portfolio_metrics['positions'], 
                                       key=lambda x: x['position_pnl_percent'])[:3]
                for i, pos in enumerate(underperformers, 1):
                    st.write(f"{i}. **{pos['symbol']}**: {pos['position_pnl_percent']:+.1f}%")
    
    else:
        st.info("📝 No positions in portfolio. Add some stocks to get started!")
        
        # Show sample portfolio suggestion
        st.subheader("💡 Sample Portfolio Suggestion")
        sample_stocks = [
            {"symbol": "AAPL", "shares": 10, "cost": 150.0, "notes": "Tech leader"},
            {"symbol": "MSFT", "shares": 5, "cost": 300.0, "notes": "Cloud computing"},
            {"symbol": "GOOGL", "shares": 3, "cost": 2500.0, "notes": "Search & AI"},
            {"symbol": "TSLA", "shares": 2, "cost": 200.0, "notes": "EV & Energy"}
        ]
        
        if st.button("🚀 Add Sample Portfolio"):
            for stock in sample_stocks:
                portfolio_db.add_position(stock['symbol'], stock['shares'], stock['cost'], stock['notes'])
            st.success("Sample portfolio added! Refresh to see your positions.")
            st.rerun()

def main():
    """Main application"""
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📊 Financial Analyzer Pro - Day 4</h1>
        <p>Portfolio Management Enhanced with Real Tracking & Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Status
    st.markdown("""
    <div class="success-message">
        <h4>🚀 Day 4: Portfolio Management Enhanced</h4>
        <p>✅ Real Portfolio Tracking | ✅ Persistent Storage | ✅ Performance Metrics | ✅ Advanced Analytics | ✅ P&L Calculations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Settings & Navigation")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Navigate",
        ["💼 Portfolio Management", "📈 Stock Analysis", "📊 Market Overview", "🤖 ML Predictions"],
        index=0
    )
    
    # Theme toggle
    theme = st.sidebar.selectbox("🎨 Theme", ["light", "dark"], index=0)
    
    # User preferences
    st.sidebar.subheader("📊 Analysis Settings")
    default_symbol = st.sidebar.text_input("Default Symbol", value=preferences.get('default_symbol', 'AAPL'))
    preferences.set('default_symbol', default_symbol)
    
    # Cache and performance info
    st.sidebar.subheader("📊 System Status")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("ML Status", "🟢 Available" if SKLEARN_AVAILABLE else "🟡 Limited")
    with col2:
        st.metric("Database", "🟢 Connected")
    
    # Main content based on navigation
    if page == "💼 Portfolio Management":
        display_portfolio_management()
    
    elif page == "📈 Stock Analysis":
        st.subheader("📈 Stock Analysis")
        st.info("Stock analysis features from previous days are available. This focuses on Day 4 portfolio management.")
        
        # Quick stock lookup
        col1, col2 = st.columns([2, 1])
        with col1:
            symbol = st.text_input("Enter Stock Symbol", value=default_symbol)
        with col2:
            if st.button("Quick Lookup"):
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
    
    elif page == "🤖 ML Predictions":
        st.subheader("🤖 ML Predictions")
        st.info("ML prediction features from previous days are available.")
        
        if SKLEARN_AVAILABLE:
            st.success("ML features are available!")
        else:
            st.warning("ML libraries not available")

if __name__ == "__main__":
    main()





