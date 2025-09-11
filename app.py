import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import yfinance as yf
import time
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Page configuration
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
        font-size: 3rem;
        color: #667eea;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Performance monitoring
def performance_monitor(func):
    """Decorator to monitor function performance"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            if execution_time > 1.0:
                st.warning(f"⚠️ {func.__name__} took {execution_time:.2f}s to execute")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            st.error(f"❌ {func.__name__} failed after {execution_time:.2f}s: {e}")
            raise
    return wrapper

@performance_monitor
def get_stock_data(ticker: str, period: str = "1y"):
    """Get stock data using yfinance"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        info = stock.info
        
        if hist.empty:
            st.error(f"No data found for {ticker}")
            return None, None
            
        return hist, info
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None, None

@performance_monitor
def calculate_technical_indicators(df):
    """Calculate basic technical indicators"""
    try:
        # Simple Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        return df
    except Exception as e:
        st.error(f"Error calculating indicators: {str(e)}")
        return df

def main():
    st.markdown('<h1 class="main-header">📈 Financial Analyzer Pro</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔧 Settings")
    
    # Stock selection
    ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL", help="e.g., AAPL, MSFT, GOOGL")
    period = st.sidebar.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
    
    if st.sidebar.button("📊 Analyze Stock"):
        if ticker:
            analyze_stock(ticker, period)
        else:
            st.error("Please enter a stock ticker")

def analyze_stock(ticker: str, period: str):
    """Main analysis function"""
    st.markdown(f"## 📊 Analyzing {ticker.upper()}")
    
    # Get data
    with st.spinner(f"Fetching data for {ticker}..."):
        hist, info = get_stock_data(ticker, period)
    
    if hist is None or info is None:
        return
    
    # Calculate indicators
    hist = calculate_technical_indicators(hist)
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = hist['Close'].iloc[-1]
    prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
    change = current_price - prev_close
    change_pct = (change / prev_close) * 100 if prev_close != 0 else 0
    
    with col1:
        st.metric("Current Price", f"${current_price:.2f}")
    with col2:
        st.metric("Change", f"${change:.2f}", f"{change_pct:.2f}%")
    with col3:
        st.metric("Volume", f"{hist['Volume'].iloc[-1]:,}")
    with col4:
        market_cap = info.get('marketCap', 0)
        if market_cap:
            st.metric("Market Cap", f"${market_cap/1e9:.1f}B")
    
    # Price chart
    st.subheader("📈 Price Chart")
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='#667eea', width=2)
    ))
    
    # Add moving averages
    if 'SMA_20' in hist.columns:
        fig.add_trace(go.Scatter(
            x=hist.index,
            y=hist['SMA_20'],
            mode='lines',
            name='SMA 20',
            line=dict(color='orange', width=1)
        ))
    
    if 'SMA_50' in hist.columns:
        fig.add_trace(go.Scatter(
            x=hist.index,
            y=hist['SMA_50'],
            mode='lines',
            name='SMA 50',
            line=dict(color='red', width=1)
        ))
    
    # Add Bollinger Bands
    if 'BB_Upper' in hist.columns and 'BB_Lower' in hist.columns:
        fig.add_trace(go.Scatter(
            x=hist.index,
            y=hist['BB_Upper'],
            mode='lines',
            name='BB Upper',
            line=dict(color='gray', width=1, dash='dash'),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=hist.index,
            y=hist['BB_Lower'],
            mode='lines',
            name='BB Lower',
            line=dict(color='gray', width=1, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(128,128,128,0.1)',
            showlegend=False
        ))
    
    fig.update_layout(
        title=f"{ticker.upper()} Stock Price",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Technical Analysis
    st.subheader("🔍 Technical Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Moving Averages**")
        if 'SMA_20' in hist.columns and 'SMA_50' in hist.columns:
            sma_20 = hist['SMA_20'].iloc[-1]
            sma_50 = hist['SMA_50'].iloc[-1]
            
            if current_price > sma_20 > sma_50:
                st.success("🟢 Bullish: Price above both moving averages")
            elif current_price < sma_20 < sma_50:
                st.error("🔴 Bearish: Price below both moving averages")
            else:
                st.warning("🟡 Mixed signals from moving averages")
    
    with col2:
        st.markdown("**RSI (Relative Strength Index)**")
        if 'RSI' in hist.columns:
            rsi = hist['RSI'].iloc[-1]
            st.metric("RSI", f"{rsi:.1f}")
            
            if rsi > 70:
                st.error("🔴 Overbought (>70)")
            elif rsi < 30:
                st.success("🟢 Oversold (<30)")
            else:
                st.info("🟡 Neutral (30-70)")
    
    # Volume analysis
    st.subheader("📊 Volume Analysis")
    volume_fig = go.Figure()
    volume_fig.add_trace(go.Bar(
        x=hist.index,
        y=hist['Volume'],
        name='Volume',
        marker_color='lightblue'
    ))
    
    volume_fig.update_layout(
        title="Trading Volume",
        xaxis_title="Date",
        yaxis_title="Volume",
        height=300
    )
    
    st.plotly_chart(volume_fig, use_container_width=True)
    
    # Recent data table
    st.subheader("📋 Recent Data")
    recent_data = hist.tail(10)[['Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI']].round(2)
    st.dataframe(recent_data, use_container_width=True)
    
    # Download data
    st.subheader("💾 Download Data")
    csv = hist.to_csv()
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"{ticker}_data.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()
