import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import sqlite3
from typing import Dict, List, Optional
import os
import time
from auth_system import init_session_state, show_login_page, show_user_menu, require_auth
from advanced_analytics import AdvancedFinancialAnalytics
from portfolio_manager import EnhancedPortfolioManager
from notification_system import RealTimeNotificationSystem

# Page config
st.set_page_config(
    page_title="Financial Analyzer Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .positive {
        color: #28a745;
        font-weight: bold;
    }
    .negative {
        color: #dc3545;
        font-weight: bold;
    }
    .user-welcome {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
    .analytics-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .notification-badge {
        background: #ff4444;
        color: white;
        border-radius: 50%;
        padding: 2px 6px;
        font-size: 12px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class FinancialAnalyzer:
    def __init__(self):
        self.init_database()
        self.analytics = AdvancedFinancialAnalytics()
        self.portfolio_manager = EnhancedPortfolioManager()
        self.notification_system = RealTimeNotificationSystem()
    
    def init_database(self):
        """Initialize SQLite database for user data"""
        try:
            conn = sqlite3.connect('financial_analyzer.db')
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE,
                    email TEXT UNIQUE,
                    password_hash TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
        except Exception as e:
            st.error(f"Database initialization error: {str(e)}")
    
    def get_market_data(self, symbol: str, period: str = "1mo") -> Optional[pd.DataFrame]:
        """Get real market data using yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            if data.empty:
                return None
            return data
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate technical analysis indicators"""
        if data.empty:
            return {}
        
        try:
            # Simple Moving Averages
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = data['Close'].ewm(span=12).mean()
            exp2 = data['Close'].ewm(span=26).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9).mean()
            histogram = macd - signal
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            data['BB_Middle'] = data['Close'].rolling(window=bb_period).mean()
            bb_std_dev = data['Close'].rolling(window=bb_period).std()
            data['BB_Upper'] = data['BB_Middle'] + (bb_std_dev * bb_std)
            data['BB_Lower'] = data['BB_Middle'] - (bb_std_dev * bb_std)
            
            return {
                'rsi': rsi.iloc[-1] if not rsi.empty and not pd.isna(rsi.iloc[-1]) else None,
                'macd': macd.iloc[-1] if not macd.empty and not pd.isna(macd.iloc[-1]) else None,
                'macd_signal': signal.iloc[-1] if not signal.empty and not pd.isna(signal.iloc[-1]) else None,
                'macd_histogram': histogram.iloc[-1] if not histogram.empty and not pd.isna(histogram.iloc[-1]) else None,
                'sma_20': data['SMA_20'].iloc[-1] if not data['SMA_20'].empty and not pd.isna(data['SMA_20'].iloc[-1]) else None,
                'sma_50': data['SMA_50'].iloc[-1] if not data['SMA_50'].empty and not pd.isna(data['SMA_50'].iloc[-1]) else None,
                'bb_upper': data['BB_Upper'].iloc[-1] if not data['BB_Upper'].empty and not pd.isna(data['BB_Upper'].iloc[-1]) else None,
                'bb_middle': data['BB_Middle'].iloc[-1] if not data['BB_Middle'].empty and not pd.isna(data['BB_Middle'].iloc[-1]) else None,
                'bb_lower': data['BB_Lower'].iloc[-1] if not data['BB_Lower'].empty and not pd.isna(data['BB_Lower'].iloc[-1]) else None
            }
        except Exception as e:
            st.error(f"Error calculating technical indicators: {str(e)}")
            return {}
    
    def get_market_overview(self) -> Dict:
        """Get market overview data"""
        symbols = ['^GSPC', '^IXIC', '^DJI', '^VIX']  # S&P 500, NASDAQ, DOW, VIX
        overview = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                hist = ticker.history(period="2d")
                
                if not hist.empty and len(hist) >= 2:
                    current_price = hist['Close'].iloc[-1]
                    previous_price = hist['Close'].iloc[-2]
                    change = current_price - previous_price
                    change_percent = (change / previous_price) * 100
                    
                    overview[symbol] = {
                        'price': current_price,
                        'change': change,
                        'change_percent': change_percent,
                        'name': info.get('longName', symbol)
                    }
            except Exception as e:
                st.warning(f"Could not fetch {symbol}: {str(e)}")
        
        return overview
    
    def create_price_chart(self, data: pd.DataFrame, symbol: str) -> go.Figure:
        """Create interactive price chart with technical indicators"""
        fig = go.Figure()
        
        # Price line
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Price',
            line=dict(color='#667eea', width=2)
        ))
        
        # Moving averages
        if 'SMA_20' in data.columns and not data['SMA_20'].isna().all():
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['SMA_20'],
                mode='lines',
                name='SMA 20',
                line=dict(color='orange', width=1, dash='dash')
            ))
        
        if 'SMA_50' in data.columns and not data['SMA_50'].isna().all():
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['SMA_50'],
                mode='lines',
                name='SMA 50',
                line=dict(color='red', width=1, dash='dash')
            ))
        
        # Bollinger Bands
        if all(col in data.columns for col in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
            if not data['BB_Upper'].isna().all():
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['BB_Upper'],
                    mode='lines',
                    name='BB Upper',
                    line=dict(color='gray', width=1),
                    showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['BB_Lower'],
                    mode='lines',
                    name='BB Lower',
                    line=dict(color='gray', width=1),
                    fill='tonexty',
                    fillcolor='rgba(128,128,128,0.1)',
                    showlegend=False
                ))
        
        fig.update_layout(
            title=f"{symbol} Price Chart",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode='x unified',
            height=500,
            showlegend=True
        )
        
        return fig

@require_auth
def show_dashboard():
    """Show personalized dashboard with all features"""
    st.markdown("""
    <div class="main-header">
        <h1>📊 Financial Analyzer Pro</h1>
        <p>Advanced Financial Analysis Platform with AI-Powered Insights & Real-time Notifications</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Welcome message
    if st.session_state.authenticated and st.session_state.user_data:
        user_id = st.session_state.user_data['user_id']
        
        # Check for notifications
        notification_count = st.session_state.notification_system.get_unread_count(user_id)
        
        st.markdown(f"""
        <div class="user-welcome">
            <h3>Welcome back, {st.session_state.user_data['username']}!</h3>
            <p>Here's your personalized financial dashboard with advanced analytics and real-time notifications</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show notification count
        if notification_count > 0:
            st.markdown(f"""
            <div style="background: #ff4444; color: white; padding: 10px; border-radius: 5px; margin: 10px 0;">
                🔔 You have {notification_count} unread notifications
            </div>
            """, unsafe_allow_html=True)
    
    analyzer = FinancialAnalyzer()
    
    # Market Overview
    st.subheader("Market Overview")
    with st.spinner("Loading market data..."):
        overview = analyzer.get_market_overview()
    
    if overview:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if '^GSPC' in overview:
                data = overview['^GSPC']
                st.metric(
                    "S&P 500",
                    f"${data['price']:.2f}",
                    f"{data['change']:+.2f} ({data['change_percent']:+.2f}%)"
                )
        
        with col2:
            if '^IXIC' in overview:
                data = overview['^IXIC']
                st.metric(
                    "NASDAQ",
                    f"${data['price']:.2f}",
                    f"{data['change']:+.2f} ({data['change_percent']:+.2f}%)"
                )
        
        with col3:
            if '^DJI' in overview:
                data = overview['^DJI']
                st.metric(
                    "DOW",
                    f"${data['price']:.2f}",
                    f"{data['change']:+.2f} ({data['change_percent']:+.2f}%)"
                )
        
        with col4:
            if '^VIX' in overview:
                data = overview['^VIX']
                st.metric(
                    "VIX",
                    f"{data['price']:.2f}",
                    f"{data['change']:+.2f} ({data['change_percent']:+.2f}%)"
                )
    else:
        st.warning("Unable to load market data. Please try again later.")
    
    # Portfolio Summary
    if st.session_state.authenticated and st.session_state.user_data:
        user_id = st.session_state.user_data['user_id']
        
        st.subheader("💼 Portfolio Summary")
        portfolio_summary = analyzer.portfolio_manager.get_portfolio_summary(user_id)
        
        if portfolio_summary and portfolio_summary.get('positions'):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Value", f"${portfolio_summary['total_value']:,.2f}")
            with col2:
                st.metric("Total Cost", f"${portfolio_summary['total_cost']:,.2f}")
            with col3:
                st.metric("Gain/Loss", f"${portfolio_summary['total_gain_loss']:+,.2f}")
            with col4:
                st.metric("Gain/Loss %", f"{portfolio_summary['total_gain_loss_pct']:+.2f}%")
            
            # Portfolio performance chart
            analyzer.portfolio_manager.create_portfolio_charts(portfolio_summary)
        else:
            st.info("No positions in your portfolio. Add some stocks to get started!")
    
    # Advanced Analytics Section
    st.markdown("""
    <div class="analytics-card">
        <h3>🔬 Advanced Analytics</h3>
        <p>Get AI-powered investment insights with DCF valuation, risk analysis, and recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Analysis with Advanced Features
    st.subheader("Quick Stock Analysis with Advanced Analytics")
    symbol = st.text_input("Enter stock symbol", value="AAPL").upper()
    
    if st.button("Analyze with Advanced Analytics"):
        if symbol:
            with st.spinner(f"Running comprehensive analysis for {symbol}..."):
                # Basic analysis
                data = analyzer.get_market_data(symbol, "3mo")
                if data is not None and not data.empty:
                    # Current price info
                    current_price = data['Close'].iloc[-1]
                    previous_price = data['Close'].iloc[-2]
                    change = current_price - previous_price
                    change_percent = (change / previous_price) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Price", f"${current_price:.2f}")
                    with col2:
                        st.metric("Change", f"${change:+.2f}")
                    with col3:
                        st.metric("Change %", f"{change_percent:+.2f}%")
                    
                    # Technical indicators
                    indicators = analyzer.calculate_technical_indicators(data)
                    
                    st.subheader("Technical Analysis")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if indicators.get('rsi'):
                            st.metric("RSI (14)", f"{indicators['rsi']:.2f}")
                    with col2:
                        if indicators.get('macd'):
                            st.metric("MACD", f"{indicators['macd']:.4f}")
                    with col3:
                        if indicators.get('sma_20'):
                            st.metric("SMA 20", f"${indicators['sma_20']:.2f}")
                    
                    # Price chart
                    st.subheader("Price Chart")
                    fig = analyzer.create_price_chart(data, symbol)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Advanced Analytics
                    st.subheader("🔬 Advanced Financial Analytics")
                    analyzer.analytics.create_analytics_dashboard(symbol)
                else:
                    st.error(f"Could not fetch data for {symbol}")

@require_auth
def show_enhanced_portfolio():
    """Show enhanced portfolio management page"""
    st.header("💼 Enhanced Portfolio Management")
    
    if st.session_state.authenticated and st.session_state.user_data:
        user_id = st.session_state.user_data['user_id']
        
        # Portfolio management tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "Positions", "Performance", "Alerts", "Goals"
        ])
        
        with tab1:
            # Add new position
            st.subheader("Add New Position")
            with st.form("add_position"):
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    symbol = st.text_input("Symbol", placeholder="AAPL").upper()
                with col2:
                    shares = st.number_input("Shares", min_value=0.0, value=1.0)
                with col3:
                    purchase_price = st.number_input("Purchase Price", min_value=0.0, value=0.0)
                with col4:
                    purchase_date = st.date_input("Purchase Date", value=datetime.now().date())
                with col5:
                    transaction_type = st.selectbox("Type", ["BUY", "SELL"])
                
                fees = st.number_input("Fees", min_value=0.0, value=0.0)
                notes = st.text_input("Notes (optional)", placeholder="Investment thesis, etc.")
                
                if st.form_submit_button("Add Position"):
                    if symbol and shares > 0 and purchase_price > 0:
                        if analyzer.portfolio_manager.add_position(
                            user_id, symbol, shares, purchase_price, 
                            purchase_date, transaction_type, fees, notes
                        ):
                            st.success(f"Added {shares} shares of {symbol} at ${purchase_price}")
                            st.rerun()
                        else:
                            st.error("Failed to add position")
                    else:
                        st.error("Please fill in all required fields")
            
            # Display portfolio
            st.subheader("Your Portfolio")
            portfolio_summary = analyzer.portfolio_manager.get_portfolio_summary(user_id)
            
            if portfolio_summary and portfolio_summary.get('positions'):
                # Portfolio charts
                analyzer.portfolio_manager.create_portfolio_charts(portfolio_summary)
            else:
                st.info("No positions in your portfolio. Add some stocks to get started!")
        
        with tab2:
            st.subheader("Portfolio Performance")
            
            portfolio_summary = analyzer.portfolio_manager.get_portfolio_summary(user_id)
            
            if portfolio_summary and portfolio_summary.get('positions'):
                # Performance metrics
                performance_metrics = portfolio_summary.get('performance_metrics', {})
                
                if performance_metrics:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Daily Return", f"{performance_metrics.get('daily_return', 0):.2f}%")
                        st.metric("Weekly Return", f"{performance_metrics.get('weekly_return', 0):.2f}%")
                    with col2:
                        st.metric("Monthly Return", f"{performance_metrics.get('monthly_return', 0):.2f}%")
                        st.metric("Volatility", f"{performance_metrics.get('volatility', 0):.2f}%")
                    with col3:
                        st.metric("Sharpe Ratio", f"{performance_metrics.get('sharpe_ratio', 0):.2f}")
                        st.metric("Max Drawdown", f"{performance_metrics.get('max_drawdown', 0):.2f}%")
                
                # Diversification metrics
                diversification = portfolio_summary.get('diversification', {})
                if diversification:
                    st.subheader("Portfolio Diversification")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Number of Positions", diversification.get('num_positions', 0))
                    with col2:
                        st.metric("Largest Position", f"{diversification.get('max_position_weight', 0):.1f}%")
                    with col3:
                        st.metric("Concentration Index", f"{diversification.get('concentration_index', 0):.3f}")
            else:
                st.info("No portfolio data available for performance analysis")
        
        with tab3:
            st.subheader("Portfolio Alerts")
            analyzer.notification_system.show_alert_management_ui(user_id)
        
        with tab4:
            st.subheader("Portfolio Goals")
            st.info("Portfolio goals feature coming soon!")
    
    else:
        st.error("Please log in to view your portfolio")

@require_auth
def show_notifications():
    """Show notifications and alerts page"""
    st.header("🔔 Notifications & Alerts")
    
    if st.session_state.authenticated and st.session_state.user_data:
        user_id = st.session_state.user_data['user_id']
        
        # Run real-time checks
        if st.button("Check for New Alerts"):
            with st.spinner("Checking for triggered alerts..."):
                triggered_alerts = analyzer.notification_system.run_real_time_checks(user_id)
                
                if triggered_alerts:
                    st.success(f"Found {len(triggered_alerts)} triggered alerts!")
                    for alert in triggered_alerts:
                        st.warning(f"🚨 {alert['message']}")
                else:
                    st.info("No new alerts triggered")
        
        # Show alert management
        analyzer.notification_system.create_alert_management_ui(user_id)
    
    else:
        st.error("Please log in to view notifications")

@require_auth
def show_advanced_analytics():
    """Show dedicated advanced analytics page"""
    st.header("🔬 Advanced Financial Analytics")
    
    st.markdown("""
    <div class="analytics-card">
        <h3>AI-Powered Investment Analysis</h3>
        <p>Get comprehensive financial analysis including DCF valuation, risk assessment, and investment recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    analyzer = FinancialAnalyzer()
    
    symbol = st.text_input("Enter stock symbol for advanced analysis", value="AAPL").upper()
    
    if st.button("Run Advanced Analysis", type="primary"):
        if symbol:
            analyzer.analytics.create_analytics_dashboard(symbol)
        else:
            st.error("Please enter a stock symbol")

@require_auth
def show_stock_analysis():
    """Show stock analysis page with advanced features"""
    st.header("🔍 Advanced Stock Analysis")
    
    analyzer = FinancialAnalyzer()
    
    symbol = st.text_input("Enter stock symbol", value="AAPL").upper()
    period = st.selectbox("Time period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"])
    
    if st.button("Run Comprehensive Analysis"):
        if symbol:
            with st.spinner(f"Running comprehensive analysis for {symbol}..."):
                data = analyzer.get_market_data(symbol, period)
                if data is not None and not data.empty:
                    # Price information
                    current_price = data['Close'].iloc[-1]
                    high_52w = data['High'].max()
                    low_52w = data['Low'].min()
                    volume = data['Volume'].iloc[-1]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Current Price", f"${current_price:.2f}")
                    with col2:
                        st.metric("52W High", f"${high_52w:.2f}")
                    with col3:
                        st.metric("52W Low", f"${low_52w:.2f}")
                    with col4:
                        st.metric("Volume", f"{volume:,}")
                    
                    # Technical indicators
                    indicators = analyzer.calculate_technical_indicators(data)
                    
                    st.subheader("Technical Indicators")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Momentum Indicators**")
                        if indicators.get('rsi'):
                            st.write(f"RSI (14): {indicators['rsi']:.2f}")
                        if indicators.get('macd'):
                            st.write(f"MACD: {indicators['macd']:.4f}")
                        if indicators.get('macd_signal'):
                            st.write(f"MACD Signal: {indicators['macd_signal']:.4f}")
                    
                    with col2:
                        st.write("**Moving Averages**")
                        if indicators.get('sma_20'):
                            st.write(f"SMA 20: ${indicators['sma_20']:.2f}")
                        if indicators.get('sma_50'):
                            st.write(f"SMA 50: ${indicators['sma_50']:.2f}")
                    
                    # Price chart
                    st.subheader("Price Chart with Technical Indicators")
                    fig = analyzer.create_price_chart(data, symbol)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Advanced Analytics
                    st.subheader("🔬 Advanced Financial Analytics")
                    analyzer.analytics.create_analytics_dashboard(symbol)
                else:
                    st.error(f"Could not fetch data for {symbol}")

@require_auth
def show_watchlist():
    """Show watchlist page"""
    st.header("⭐ Watchlist")
    
    if st.session_state.authenticated and st.session_state.user_data:
        user_id = st.session_state.user_data['user_id']
        
        # Add to watchlist
        st.subheader("Add to Watchlist")
        with st.form("add_watchlist"):
            symbol = st.text_input("Symbol", placeholder="AAPL").upper()
            if st.form_submit_button("Add to Watchlist"):
                if symbol:
                    try:
                        conn = sqlite3.connect('financial_analyzer.db')
                        cursor = conn.cursor()
                        
                        # Check if already in watchlist
                        cursor.execute('''
                            SELECT id FROM watchlists WHERE user_id = ? AND symbol = ?
                        ''', (user_id, symbol))
                        
                        if cursor.fetchone():
                            st.warning(f"{symbol} is already in your watchlist")
                        else:
                            cursor.execute('''
                                INSERT INTO watchlists (user_id, symbol)
                                VALUES (?, ?)
                            ''', (user_id, symbol))
                            
                            conn.commit()
                            st.success(f"Added {symbol} to watchlist")
                            st.rerun()
                        
                        conn.close()
                    except Exception as e:
                        st.error(f"Error adding to watchlist: {str(e)}")
        
        # Display watchlist
        st.subheader("Your Watchlist")
        
        try:
            conn = sqlite3.connect('financial_analyzer.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT symbol, added_at FROM watchlists WHERE user_id = ? ORDER BY added_at DESC
            ''', (user_id,))
            
            watchlist_items = cursor.fetchall()
            conn.close()
            
            if watchlist_items:
                watchlist_data = []
                analyzer = FinancialAnalyzer()
                
                for symbol, added_at in watchlist_items:
                    data = analyzer.get_market_data(symbol, "1d")
                    
                    if data is not None and not data.empty:
                        current_price = data['Close'].iloc[-1]
                        previous_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
                        change = current_price - previous_price
                        change_percent = (change / previous_price) * 100 if previous_price > 0 else 0
                        
                        watchlist_data.append({
                            'Symbol': symbol,
                            'Current Price': f"${current_price:.2f}",
                            'Change': f"${change:+.2f}",
                            'Change %': f"{change_percent:+.2f}%",
                            'Added': added_at
                        })
                    else:
                        watchlist_data.append({
                            'Symbol': symbol,
                            'Current Price': "N/A",
                            'Change': "N/A",
                            'Change %': "N/A",
                            'Added': added_at
                        })
                
                if watchlist_data:
                    df = pd.DataFrame(watchlist_data)
                    st.dataframe(df, use_container_width=True)
            else:
                st.info("Your watchlist is empty. Add some stocks to monitor!")
                
        except Exception as e:
            st.error(f"Error loading watchlist: {str(e)}")
    else:
        st.error("Please log in to view your watchlist")

@require_auth
def show_market_overview():
    """Show market overview page"""
    st.header("🌍 Market Overview")
    
    analyzer = FinancialAnalyzer()
    
    # Real-time market data
    with st.spinner("Loading market data..."):
        overview = analyzer.get_market_overview()
    
    if overview:
        for symbol, data in overview.items():
            with st.expander(f"{data['name']} ({symbol})"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Price", f"${data['price']:.2f}")
                with col2:
                    st.metric("Change", f"${data['change']:+.2f}")
                with col3:
                    st.metric("Change %", f"{data['change_percent']:+.2f}%")
    else:
        st.warning("Unable to load market data. Please try again later.")

def main():
    # Initialize session state
    init_session_state()
    
    # Initialize analyzer
    analyzer = FinancialAnalyzer()
    
    # Show user menu in sidebar
    show_user_menu()
    
    # Main navigation
    if st.session_state.authenticated:
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox("Choose a page", [
            "Dashboard", "Stock Analysis", "Advanced Analytics", "Portfolio", "Watchlist", "Notifications", "Market Overview"
        ])
        
        if page == "Dashboard":
            show_dashboard()
        elif page == "Stock Analysis":
            show_stock_analysis()
        elif page == "Advanced Analytics":
            show_advanced_analytics()
        elif page == "Portfolio":
            show_enhanced_portfolio()
        elif page == "Watchlist":
            show_watchlist()
        elif page == "Notifications":
            show_notifications()
        elif page == "Market Overview":
            show_market_overview()
    else:
        show_login_page()

if __name__ == "__main__":
    main()








