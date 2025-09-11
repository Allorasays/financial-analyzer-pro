import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

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

def generate_mock_data(ticker: str, days: int = 30):
    """Generate mock stock data for demonstration"""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Generate realistic price data
    base_price = 100 + hash(ticker) % 200  # Different base price per ticker
    prices = []
    current_price = base_price
    
    for i in range(days):
        # Random walk with slight upward bias
        change = random.uniform(-0.05, 0.08)
        current_price *= (1 + change)
        prices.append(current_price)
    
    # Generate volume data
    volumes = [random.randint(1000000, 10000000) for _ in range(days)]
    
    df = pd.DataFrame({
        'Date': dates,
        'Close': prices,
        'Volume': volumes
    })
    
    # Calculate moving averages
    df['SMA_20'] = df['Close'].rolling(window=min(20, days)).mean()
    df['SMA_50'] = df['Close'].rolling(window=min(50, days)).mean()
    
    return df

def main():
    st.markdown('<h1 class="main-header">📈 Financial Analyzer Pro</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="metric-card">
        <h3>🚀 Welcome to Financial Analyzer Pro!</h3>
        <p>This is a demonstration version with mock data. Enter a stock ticker below to see sample analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔧 Settings")
    
    # Stock selection
    ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL", help="e.g., AAPL, MSFT, GOOGL")
    period = st.sidebar.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
    
    if st.sidebar.button("📊 Analyze Stock") or ticker:
        if ticker:
            analyze_stock(ticker, period)
        else:
            st.error("Please enter a stock ticker")

def analyze_stock(ticker: str, period: str):
    """Main analysis function with mock data"""
    st.markdown(f"## 📊 Analyzing {ticker.upper()}")
    
    # Convert period to days
    period_days = {
        "1mo": 30,
        "3mo": 90,
        "6mo": 180,
        "1y": 365,
        "2y": 730
    }
    
    days = period_days.get(period, 365)
    
    # Generate mock data
    with st.spinner(f"Generating sample data for {ticker}..."):
        hist = generate_mock_data(ticker, days)
    
    if hist.empty:
        st.error(f"Could not generate data for {ticker}")
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
        market_cap = current_price * random.randint(1000000000, 10000000000)
        st.metric("Market Cap", f"${market_cap/1e9:.1f}B")
    
    # Price chart using Streamlit's built-in chart
    st.subheader("📈 Price Chart")
    
    # Create a simple line chart
    chart_data = hist[['Close', 'SMA_20', 'SMA_50']].fillna(method='bfill')
    st.line_chart(chart_data)
    
    # Volume chart
    st.subheader("📊 Volume")
    volume_data = hist[['Volume']]
    st.bar_chart(volume_data)
    
    # Technical Analysis
    st.subheader("🔍 Technical Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Moving Averages**")
        sma_20 = hist['SMA_20'].iloc[-1]
        sma_50 = hist['SMA_50'].iloc[-1]
        
        if current_price > sma_20 > sma_50:
            st.success("🟢 Bullish: Price above both moving averages")
        elif current_price < sma_20 < sma_50:
            st.error("🔴 Bearish: Price below both moving averages")
        else:
            st.warning("🟡 Mixed signals from moving averages")
    
    with col2:
        st.markdown("**Price Trend**")
        recent_prices = hist['Close'].tail(5)
        if recent_prices.iloc[-1] > recent_prices.iloc[0]:
            st.success("🟢 Uptrend: Recent price increase")
        else:
            st.error("🔴 Downtrend: Recent price decrease")
    
    # Recent data table
    st.subheader("📋 Recent Data")
    recent_data = hist.tail(10)[['Close', 'Volume', 'SMA_20', 'SMA_50']].round(2)
    st.dataframe(recent_data, use_container_width=True)
    
    # Download data
    st.subheader("💾 Download Data")
    csv = hist.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"{ticker}_data.csv",
        mime="text/csv"
    )
    
    # Info about mock data
    st.info("ℹ️ **Note:** This is demonstration data. In production, this would connect to real market data sources.")

if __name__ == "__main__":
    main()
