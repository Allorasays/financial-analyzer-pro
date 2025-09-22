#!/usr/bin/env python3
"""
Minimal Financial Analyzer - Guaranteed to work on Render
Simple, stable version with basic functionality
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Financial Analyzer Pro",
    page_icon="📊",
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
</style>
""", unsafe_allow_html=True)

def get_stock_data(symbol, period="1mo"):
    """Get stock data from yfinance"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def create_price_chart(data, symbol):
    """Create a simple price chart"""
    if data is None or data.empty:
        return None
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name=f'{symbol} Price',
        line=dict(color='#667eea', width=2)
    ))
    
    fig.update_layout(
        title=f'{symbol} Stock Price',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        height=400,
        showlegend=True
    )
    
    return fig

def main():
    # Header
    st.markdown('<div class="main-header"><h1>📊 Financial Analyzer Pro</h1><p>Simple & Reliable Financial Analysis</p></div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("📈 Analysis Options")
    
    # Stock symbol input
    symbol = st.sidebar.text_input("Enter Stock Symbol", value="AAPL", help="Enter a valid stock ticker symbol (e.g., AAPL, MSFT, GOOGL)")
    
    # Time period selection
    period = st.sidebar.selectbox(
        "Select Time Period",
        ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"],
        index=2
    )
    
    # Analysis type
    analysis_type = st.sidebar.selectbox(
        "Analysis Type",
        ["Price Chart", "Volume Analysis", "Basic Metrics"]
    )
    
    if st.sidebar.button("🔍 Analyze Stock", type="primary"):
        with st.spinner(f"Fetching data for {symbol}..."):
            # Get stock data
            data = get_stock_data(symbol, period)
            
            if data is not None and not data.empty:
                st.success(f"✅ Successfully loaded data for {symbol}")
                
                # Display basic info
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    current_price = data['Close'].iloc[-1]
                    st.metric("Current Price", f"${current_price:.2f}")
                
                with col2:
                    price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
                    st.metric("Daily Change", f"${price_change:.2f}")
                
                with col3:
                    price_change_pct = (price_change / data['Close'].iloc[-2]) * 100
                    st.metric("Change %", f"{price_change_pct:.2f}%")
                
                with col4:
                    volume = data['Volume'].iloc[-1]
                    st.metric("Volume", f"{volume:,}")
                
                # Analysis based on type
                if analysis_type == "Price Chart":
                    st.subheader("📈 Price Chart")
                    chart = create_price_chart(data, symbol)
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
                
                elif analysis_type == "Volume Analysis":
                    st.subheader("📊 Volume Analysis")
                    volume_fig = go.Figure()
                    volume_fig.add_trace(go.Bar(
                        x=data.index,
                        y=data['Volume'],
                        name='Volume',
                        marker_color='#764ba2'
                    ))
                    volume_fig.update_layout(
                        title=f'{symbol} Trading Volume',
                        xaxis_title='Date',
                        yaxis_title='Volume',
                        height=400
                    )
                    st.plotly_chart(volume_fig, use_container_width=True)
                
                elif analysis_type == "Basic Metrics":
                    st.subheader("📋 Basic Metrics")
                    
                    # Calculate basic metrics
                    high_52w = data['High'].max()
                    low_52w = data['Low'].min()
                    avg_volume = data['Volume'].mean()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("52-Week High", f"${high_52w:.2f}")
                        st.metric("52-Week Low", f"${low_52w:.2f}")
                    
                    with col2:
                        st.metric("Average Volume", f"{avg_volume:,.0f}")
                        st.metric("Data Points", len(data))
                
                # Data table
                st.subheader("📊 Raw Data")
                st.dataframe(data.tail(10), use_container_width=True)
                
            else:
                st.error(f"❌ Could not fetch data for {symbol}. Please check the symbol and try again.")
    
    # Footer
    st.markdown("---")
    st.markdown("**Financial Analyzer Pro** - Simple, Reliable, Fast")
    st.markdown("*Built with Streamlit and yfinance*")

if __name__ == "__main__":
    main()
