import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Financial Analyzer Pro - Step 3",
    page_icon="📈",
    layout="wide"
)

# Header
st.title("📈 Financial Analyzer Pro")
st.write("**Step 3: Interactive Charts** - Plotly visualizations working")

# Stock lookup with charts
st.header("🔍 Stock Analysis")
symbol = st.text_input("Enter stock symbol", value="AAPL").upper()
period = st.selectbox("Time period", ["1mo", "3mo", "6mo", "1y"])

if st.button("Analyze with Chart"):
    if symbol:
        try:
            with st.spinner(f"Analyzing {symbol}..."):
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period)
                
                if not data.empty:
                    # Price metrics
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
                    
                    # Create interactive chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data['Close'],
                        mode='lines',
                        name='Price',
                        line=dict(color='#667eea', width=2)
                    ))
                    
                    fig.update_layout(
                        title=f"{symbol} Price Chart ({period})",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    st.success(f"✅ Chart created for {symbol}!")
                else:
                    st.error(f"No data found for {symbol}")
        except Exception as e:
            st.error(f"Error analyzing {symbol}: {str(e)}")

# Market overview
st.header("📊 Market Overview")
try:
    with st.spinner("Loading market data..."):
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
st.success("✅ Step 3 Complete: Interactive charts working!")
st.info("Next: Add database functionality")
