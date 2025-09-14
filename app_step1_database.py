"""
Financial Analyzer Pro - Step 1: Database Foundation
Enhanced version with SQLite database integration
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our database module
from database import DatabaseManager, init_database

# Initialize database
init_database()
db = DatabaseManager()

# Page configuration
st.set_page_config(
    page_title="Financial Analyzer Pro - Step 1",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .step-indicator {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        border-left: 5px solid #667eea;
    }
    .database-status {
        background: #e8f5e8;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Financial Analyzer Pro</h1>', unsafe_allow_html=True)
    
    # Step indicator
    st.markdown("""
    <div class="step-indicator">
        <h3>🚀 Step 1: Database Foundation</h3>
        <p>✅ SQLite database integrated | ✅ User management ready | ✅ Portfolio system ready | ✅ Preferences system ready</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Database status
    db_stats = db.get_database_stats()
    st.markdown(f"""
    <div class="database-status">
        <h4>🗄️ Database Status</h4>
        <p><strong>Users:</strong> {db_stats.get('users_count', 0)} | 
        <strong>Portfolios:</strong> {db_stats.get('portfolios_count', 0)} | 
        <strong>Watchlists:</strong> {db_stats.get('watchlists_count', 0)} | 
        <strong>Positions:</strong> {db_stats.get('portfolio_positions_count', 0)}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📊 Financial Analyzer Pro")
    st.sidebar.markdown("**Step 1: Database Foundation**")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Choose Analysis Type",
        ["📈 Quick Analysis", "🔍 Detailed Analysis", "📊 Portfolio Analysis", "⚙️ Database Management"]
    )
    
    if page == "📈 Quick Analysis":
        quick_analysis_page()
    elif page == "🔍 Detailed Analysis":
        detailed_analysis_page()
    elif page == "📊 Portfolio Analysis":
        portfolio_analysis_page()
    elif page == "⚙️ Database Management":
        database_management_page()

def quick_analysis_page():
    """Quick stock analysis page"""
    st.header("📈 Quick Stock Analysis")
    
    # Stock input
    col1, col2 = st.columns([2, 1])
    with col1:
        symbol = st.text_input("Enter Stock Symbol", value="AAPL", placeholder="e.g., AAPL, MSFT, GOOGL")
    with col2:
        timeframe = st.selectbox("Timeframe", ["1mo", "3mo", "6mo", "1y", "2y", "5y"])
    
    if st.button("Analyze Stock", type="primary"):
        if symbol:
            analyze_stock(symbol, timeframe)
        else:
            st.error("Please enter a stock symbol")

def detailed_analysis_page():
    """Detailed analysis page"""
    st.header("🔍 Detailed Financial Analysis")
    
    # Stock input
    col1, col2 = st.columns([2, 1])
    with col1:
        symbol = st.text_input("Enter Stock Symbol", value="AAPL", placeholder="e.g., AAPL, MSFT, GOOGL", key="detailed_symbol")
    with col2:
        timeframe = st.selectbox("Timeframe", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], key="detailed_timeframe")
    
    if st.button("Perform Detailed Analysis", type="primary"):
        if symbol:
            perform_detailed_analysis(symbol, timeframe)
        else:
            st.error("Please enter a stock symbol")

def portfolio_analysis_page():
    """Portfolio analysis page"""
    st.header("📊 Portfolio Analysis")
    
    # Portfolio input
    st.subheader("Portfolio Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Add Stock to Portfolio**")
        symbol = st.text_input("Symbol", placeholder="AAPL")
        shares = st.number_input("Shares", min_value=0.0, value=10.0)
        purchase_price = st.number_input("Purchase Price", min_value=0.0, value=150.0)
    
    with col2:
        st.write("**Portfolio Summary**")
        st.info("Portfolio analysis will be available in Step 2 with user authentication")
    
    if st.button("Add to Portfolio", type="primary"):
        if symbol and shares > 0:
            st.success(f"Added {shares} shares of {symbol} at ${purchase_price:.2f}")
            st.info("Portfolio persistence will be available in Step 2")

def database_management_page():
    """Database management page"""
    st.header("⚙️ Database Management")
    
    # Database stats
    st.subheader("📊 Database Statistics")
    stats = db.get_database_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Users", stats.get('users_count', 0))
    with col2:
        st.metric("Portfolios", stats.get('portfolios_count', 0))
    with col3:
        st.metric("Positions", stats.get('portfolio_positions_count', 0))
    with col4:
        st.metric("Watchlists", stats.get('watchlists_count', 0))
    
    # Test database operations
    st.subheader("🧪 Test Database Operations")
    
    if st.button("Create Test User", type="primary"):
        success = db.create_user("testuser", "test@example.com", "testpass123", "Test User")
        if success:
            st.success("✅ Test user created successfully!")
        else:
            st.error("❌ Failed to create test user (may already exist)")
    
    if st.button("Test Authentication"):
        user = db.authenticate_user("testuser", "testpass123")
        if user:
            st.success(f"✅ Authentication successful! Welcome {user['full_name']}")
        else:
            st.error("❌ Authentication failed")
    
    # Database schema info
    st.subheader("📋 Database Schema")
    st.info("""
    **Tables Created:**
    - **users**: User accounts and profiles
    - **user_preferences**: User-specific settings
    - **portfolios**: User portfolios
    - **portfolio_positions**: Stock positions
    - **watchlists**: User watchlists
    - **analysis_templates**: Saved analysis configurations
    - **user_sessions**: Session management
    """)

def analyze_stock(symbol, timeframe):
    """Perform quick stock analysis"""
    try:
        # Get stock data
        with st.spinner(f"Fetching data for {symbol}..."):
            stock = yf.Ticker(symbol)
            data = stock.history(period=timeframe)
            
            if data.empty:
                st.error(f"No data found for {symbol}")
                return
        
        # Basic metrics
        current_price = data['Close'].iloc[-1]
        prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
        change = current_price - prev_price
        change_pct = (change / prev_price) * 100 if prev_price != 0 else 0
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", f"${current_price:.2f}")
        with col2:
            st.metric("Change", f"${change:.2f}")
        with col3:
            st.metric("Change %", f"{change_pct:.2f}%")
        with col4:
            st.metric("Volume", f"{data['Volume'].iloc[-1]:,}")
        
        # Price chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#667eea', width=2)
        ))
        
        fig.update_layout(
            title=f"{symbol} Price Chart ({timeframe})",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Basic analysis
        st.subheader("📊 Quick Analysis")
        
        # Price trend
        price_trend = "📈 Bullish" if change > 0 else "📉 Bearish" if change < 0 else "➡️ Neutral"
        st.write(f"**Price Trend:** {price_trend}")
        
        # Volatility
        volatility = data['Close'].pct_change().std() * 100
        st.write(f"**Volatility:** {volatility:.2f}%")
        
        # Support/Resistance
        support = data['Low'].min()
        resistance = data['High'].max()
        st.write(f"**Support:** ${support:.2f} | **Resistance:** ${resistance:.2f}")
        
    except Exception as e:
        st.error(f"Error analyzing {symbol}: {str(e)}")

def perform_detailed_analysis(symbol, timeframe):
    """Perform detailed financial analysis"""
    try:
        # Get stock data
        with st.spinner(f"Performing detailed analysis for {symbol}..."):
            stock = yf.Ticker(symbol)
            data = stock.history(period=timeframe)
            
            if data.empty:
                st.error(f"No data found for {symbol}")
                return
        
        # Calculate technical indicators
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        
        # RSI calculation
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD calculation
        exp1 = data['Close'].ewm(span=12).mean()
        exp2 = data['Close'].ewm(span=26).mean()
        data['MACD'] = exp1 - exp2
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        
        # Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
        data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
        
        # Display current indicators
        current_price = data['Close'].iloc[-1]
        current_rsi = data['RSI'].iloc[-1]
        current_macd = data['MACD'].iloc[-1]
        current_macd_signal = data['MACD_Signal'].iloc[-1]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", f"${current_price:.2f}")
        with col2:
            st.metric("RSI", f"{current_rsi:.1f}")
        with col3:
            st.metric("MACD", f"{current_macd:.2f}")
        with col4:
            st.metric("MACD Signal", f"{current_macd_signal:.2f}")
        
        # Technical indicators chart
        fig = go.Figure()
        
        # Price and moving averages
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#667eea', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['SMA_20'],
            mode='lines',
            name='SMA 20',
            line=dict(color='orange', width=1)
        ))
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['SMA_50'],
            mode='lines',
            name='SMA 50',
            line=dict(color='red', width=1)
        ))
        
        # Bollinger Bands
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['BB_Upper'],
            mode='lines',
            name='BB Upper',
            line=dict(color='gray', width=1, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['BB_Lower'],
            mode='lines',
            name='BB Lower',
            line=dict(color='gray', width=1, dash='dash'),
            fill='tonexty'
        ))
        
        fig.update_layout(
            title=f"{symbol} Technical Analysis ({timeframe})",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # RSI chart
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=data.index,
            y=data['RSI'],
            mode='lines',
            name='RSI',
            line=dict(color='purple', width=2)
        ))
        
        # RSI levels
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
        
        fig_rsi.update_layout(
            title="RSI (Relative Strength Index)",
            xaxis_title="Date",
            yaxis_title="RSI",
            height=300,
            yaxis=dict(range=[0, 100])
        )
        
        st.plotly_chart(fig_rsi, use_container_width=True)
        
        # Analysis summary
        st.subheader("📊 Analysis Summary")
        
        # RSI analysis
        if current_rsi > 70:
            rsi_signal = "🔴 Overbought - Consider selling"
        elif current_rsi < 30:
            rsi_signal = "🟢 Oversold - Consider buying"
        else:
            rsi_signal = "🟡 Neutral - Hold position"
        
        st.write(f"**RSI Signal:** {rsi_signal}")
        
        # MACD analysis
        if current_macd > current_macd_signal:
            macd_signal = "🟢 Bullish - MACD above signal"
        else:
            macd_signal = "🔴 Bearish - MACD below signal"
        
        st.write(f"**MACD Signal:** {macd_signal}")
        
        # Moving average analysis
        if current_price > data['SMA_20'].iloc[-1]:
            ma_signal = "🟢 Bullish - Price above SMA 20"
        else:
            ma_signal = "🔴 Bearish - Price below SMA 20"
        
        st.write(f"**Moving Average Signal:** {ma_signal}")
        
    except Exception as e:
        st.error(f"Error performing detailed analysis: {str(e)}")

if __name__ == "__main__":
    main()
