import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Financial Analyzer Pro",
    page_icon="📈",
    layout="wide"
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
</style>
""", unsafe_allow_html=True)

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

def main():
    st.markdown('<h1 class="main-header">📈 Financial Analyzer Pro</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔧 Settings")
    
    # Stock selection
    ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL", help="e.g., AAPL, MSFT, GOOGL")
    period = st.sidebar.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
    
    if st.sidebar.button("📊 Analyze Stock") or ticker:
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
    
    # Add simple moving average
    hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist['SMA_20'],
        mode='lines',
        name='SMA 20',
        line=dict(color='orange', width=1)
    ))
    
    fig.update_layout(
        title=f"{ticker.upper()} Stock Price",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Volume chart
    st.subheader("📊 Volume")
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
    recent_data = hist.tail(10)[['Close', 'Volume', 'SMA_20']].round(2)
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
