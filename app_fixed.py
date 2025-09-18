import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional

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
</style>
""", unsafe_allow_html=True)

def get_market_data(symbol: str, period: str = "1mo") -> Optional[pd.DataFrame]:
    """Get market data using yfinance"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        if data.empty:
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def get_market_overview() -> Dict:
    """Get market overview data"""
    symbols = ['^GSPC', '^IXIC', '^DJI', '^VIX']
    overview = {}
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
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
                    'name': symbol
                }
        except Exception as e:
            st.warning(f"Could not fetch {symbol}: {str(e)}")
    
    return overview

def calculate_technical_indicators(data: pd.DataFrame) -> Dict:
    """Calculate technical indicators"""
    if data.empty:
        return {}
    
    try:
        # Simple Moving Averages
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        
        # RSI (simplified)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # MACD (simplified)
        exp1 = data['Close'].ewm(span=12).mean()
        exp2 = data['Close'].ewm(span=26).mean()
        macd = exp1 - exp2
        
        return {
            'rsi': rsi.iloc[-1] if not rsi.empty and not pd.isna(rsi.iloc[-1]) else None,
            'macd': macd.iloc[-1] if not macd.empty and not pd.isna(macd.iloc[-1]) else None,
            'sma_20': data['SMA_20'].iloc[-1] if not data['SMA_20'].empty and not pd.isna(data['SMA_20'].iloc[-1]) else None,
            'sma_50': data['SMA_50'].iloc[-1] if not data['SMA_50'].empty and not pd.isna(data['SMA_50'].iloc[-1]) else None
        }
    except Exception as e:
        st.error(f"Error calculating technical indicators: {str(e)}")
        return {}

def create_simple_chart(data: pd.DataFrame, symbol: str) -> go.Figure:
    """Create a simple, fast chart"""
    fig = go.Figure()
    
    # Price line only
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='Price',
        line=dict(color='#667eea', width=2)
    ))
    
    fig.update_layout(
        title=f"{symbol} Price Chart",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=400,
        showlegend=False
    )
    
    return fig

def main():
    st.markdown("""
    <div class="main-header">
        <h1>📈 Financial Analyzer Pro</h1>
        <p>Fixed Version - No Caching Issues</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "Dashboard", "Stock Analysis", "Quick Lookup"
    ])
    
    if page == "Dashboard":
        st.header("📊 Dashboard")
        
        # Market Overview
        st.subheader("Market Overview")
        with st.spinner("Loading market data..."):
            overview = get_market_overview()
        
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
        
        # Quick Analysis
        st.subheader("Quick Stock Analysis")
        symbol = st.text_input("Enter stock symbol", value="AAPL").upper()
        
        if st.button("Analyze Stock"):
            if symbol:
                with st.spinner(f"Analyzing {symbol}..."):
                    data = get_market_data(symbol, "1mo")
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
                        
                        # Simple technical indicators
                        indicators = calculate_technical_indicators(data)
                        
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
                        
                        # Simple price chart
                        st.subheader("Price Chart")
                        fig = create_simple_chart(data, symbol)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(f"Could not fetch data for {symbol}")
    
    elif page == "Stock Analysis":
        st.header("🔍 Stock Analysis")
        
        symbol = st.text_input("Enter stock symbol", value="AAPL").upper()
        period = st.selectbox("Time period", ["1mo", "3mo", "6mo", "1y"])
        
        if st.button("Analyze"):
            if symbol:
                with st.spinner(f"Analyzing {symbol}..."):
                    data = get_market_data(symbol, period)
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
                        indicators = calculate_technical_indicators(data)
                        
                        st.subheader("Technical Indicators")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Momentum Indicators**")
                            if indicators.get('rsi'):
                                st.write(f"RSI (14): {indicators['rsi']:.2f}")
                            if indicators.get('macd'):
                                st.write(f"MACD: {indicators['macd']:.4f}")
                        
                        with col2:
                            st.write("**Moving Averages**")
                            if indicators.get('sma_20'):
                                st.write(f"SMA 20: ${indicators['sma_20']:.2f}")
                            if indicators.get('sma_50'):
                                st.write(f"SMA 50: ${indicators['sma_50']:.2f}")
                        
                        # Price chart
                        st.subheader("Price Chart")
                        fig = create_simple_chart(data, symbol)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(f"Could not fetch data for {symbol}")
    
    elif page == "Quick Lookup":
        st.header("⚡ Quick Stock Lookup")
        
        symbol = st.text_input("Enter stock symbol", value="AAPL").upper()
        
        if st.button("Get Quote"):
            if symbol:
                with st.spinner(f"Getting quote for {symbol}..."):
                    data = get_market_data(symbol, "5d")
                    if data is not None and not data.empty:
                        current_price = data['Close'].iloc[-1]
                        previous_price = data['Close'].iloc[-2]
                        change = current_price - previous_price
                        change_percent = (change / previous_price) * 100
                        
                        st.success(f"**{symbol}**: ${current_price:.2f} ({change:+.2f}, {change_percent:+.2f}%)")
                    else:
                        st.error(f"Could not fetch data for {symbol}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"App startup error: {str(e)}")
        st.write("Please check the logs for more details.")






