#!/usr/bin/env python3
"""
Minimal test version of Financial Analyzer for Render deployment
This version has minimal dependencies and should work reliably
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
    page_title="Financial Analyzer Pro - Test",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple CSS
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
</style>
""", unsafe_allow_html=True)

def get_stock_data(symbol, period="1y"):
    """Get stock data from Yahoo Finance"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        return data, None
    except Exception as e:
        return None, str(e)

def calculate_basic_indicators(data):
    """Calculate basic technical indicators"""
    if data is None or len(data) == 0:
        return data
    
    # Simple Moving Averages
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    return data

def main():
    """Main application"""
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("📊 Financial Analyzer Pro - Test Version")
    st.markdown("**Real-time Financial Analysis & Portfolio Management**")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("🔧 Analysis Settings")
    
    # Stock symbol input
    symbol = st.sidebar.text_input("Stock Symbol", value="AAPL", help="Enter a stock symbol (e.g., AAPL, MSFT, GOOGL)")
    period = st.sidebar.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
    
    # Analysis button
    if st.sidebar.button("🚀 Analyze Stock", type="primary"):
        with st.spinner("Fetching data and analyzing..."):
            # Get data
            data, error = get_stock_data(symbol, period)
            
            if error:
                st.error(f"❌ Error fetching data: {error}")
                return
            
            if data is None or len(data) == 0:
                st.error("❌ No data available for this symbol")
                return
            
            # Calculate indicators
            data = calculate_basic_indicators(data)
            
            # Display current price
            current_price = data['Close'].iloc[-1]
            previous_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
            change = current_price - previous_price
            change_percent = (change / previous_price) * 100
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Price", f"${current_price:.2f}")
            with col2:
                st.metric("Change", f"${change:.2f}")
            with col3:
                st.metric("Change %", f"{change_percent:.2f}%")
            with col4:
                rsi = data['RSI'].iloc[-1] if 'RSI' in data.columns else 0
                st.metric("RSI", f"{rsi:.1f}")
            
            # Price chart
            st.subheader("📈 Price Chart")
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='#1f77b4', width=2)
            ))
            
            if 'SMA_20' in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['SMA_20'],
                    mode='lines',
                    name='SMA 20',
                    line=dict(color='orange', width=1)
                ))
            
            if 'SMA_50' in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['SMA_50'],
                    mode='lines',
                    name='SMA 50',
                    line=dict(color='red', width=1)
                ))
            
            fig.update_layout(
                title=f"{symbol} Stock Price",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # RSI chart
            if 'RSI' in data.columns:
                st.subheader("📊 RSI Indicator")
                rsi_fig = go.Figure()
                
                rsi_fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['RSI'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='purple', width=2)
                ))
                
                # Add RSI levels
                rsi_fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
                rsi_fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
                
                rsi_fig.update_layout(
                    title="RSI (Relative Strength Index)",
                    xaxis_title="Date",
                    yaxis_title="RSI",
                    yaxis=dict(range=[0, 100]),
                    height=300
                )
                
                st.plotly_chart(rsi_fig, use_container_width=True)
            
            # Data summary
            st.subheader("📋 Data Summary")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Symbol:** {symbol}")
                st.write(f"**Period:** {period}")
                st.write(f"**Data Points:** {len(data)}")
                st.write(f"**Date Range:** {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
            
            with col2:
                st.write(f"**High:** ${data['High'].max():.2f}")
                st.write(f"**Low:** ${data['Low'].min():.2f}")
                st.write(f"**Volume (Avg):** {data['Volume'].mean():,.0f}")
                st.write(f"**Volatility:** {data['Close'].pct_change().std() * 100:.2f}%")
    
    # Footer
    st.markdown("---")
    st.markdown("**Financial Analyzer Pro - Test Version** | Built with Streamlit")
    st.markdown("*This is a minimal test version for deployment verification*")

if __name__ == "__main__":
    main()
