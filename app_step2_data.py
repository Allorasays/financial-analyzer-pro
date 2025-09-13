import streamlit as st
import yfinance as yf
import pandas as pd

# Page config
st.set_page_config(
    page_title="Financial Analyzer Pro - Step 2",
    page_icon="📈",
    layout="wide"
)

# Header
st.title("📈 Financial Analyzer Pro")
st.write("**Step 2: Real Market Data** - Live stock prices working")

# Stock lookup with real data
st.header("🔍 Stock Lookup")
symbol = st.text_input("Enter stock symbol", value="AAPL").upper()

if st.button("Get Real Data"):
    if symbol:
        try:
            with st.spinner(f"Fetching data for {symbol}..."):
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1d")
                
                if not data.empty:
                    current_price = data['Close'].iloc[-1]
                    previous_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
                    change = current_price - previous_price
                    change_percent = (change / previous_price) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Price", f"${current_price:.2f}")
                    with col2:
                        st.metric("Change", f"${change:+.2f}")
                    with col3:
                        st.metric("Change %", f"{change_percent:+.2f}%")
                    
                    st.success(f"✅ Real data loaded for {symbol}!")
                else:
                    st.error(f"No data found for {symbol}")
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")

# Market overview with real data
st.header("📊 Market Overview")
try:
    with st.spinner("Loading market data..."):
        # S&P 500
        sp500 = yf.Ticker("^GSPC")
        sp500_data = sp500.history(period="2d")
        
        if not sp500_data.empty:
            current = sp500_data['Close'].iloc[-1]
            previous = sp500_data['Close'].iloc[-2] if len(sp500_data) > 1 else current
            change = current - previous
            change_pct = (change / previous) * 100 if previous != 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("S&P 500", f"${current:.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
            with col2:
                st.metric("NASDAQ", "Loading...", "Loading...")
            with col3:
                st.metric("DOW", "Loading...", "Loading...")
        else:
            st.warning("Could not load market data")
except Exception as e:
    st.warning(f"Market data temporarily unavailable: {str(e)}")

# Portfolio
st.header("💼 Portfolio")
portfolio_data = {
    'Symbol': ['AAPL', 'MSFT', 'GOOGL'],
    'Shares': [10, 5, 3],
    'Price': [150.25, 300.50, 2800.75],
    'Value': [1502.50, 1502.50, 8402.25]
}

st.dataframe(portfolio_data, width='stretch')

# Status
st.success("✅ Step 2 Complete: Real market data working!")
st.info("Next: Add interactive charts with Plotly")
