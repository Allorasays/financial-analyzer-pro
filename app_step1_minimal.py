import streamlit as st

# Page config
st.set_page_config(
    page_title="Financial Analyzer Pro - Step 1",
    page_icon="📈",
    layout="wide"
)

# Header
st.title("📈 Financial Analyzer Pro")
st.write("**Step 1: Minimal Version** - Basic functionality working")

# Stock lookup
st.header("🔍 Stock Lookup")
symbol = st.text_input("Enter stock symbol", value="AAPL").upper()

if st.button("Get Info"):
    if symbol:
        st.success(f"✅ Symbol: {symbol}")
        st.info("This is a minimal version. Real data will be added in Step 2.")

# Market overview
st.header("📊 Market Overview")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("S&P 500", "4,500.00", "+54.20")

with col2:
    st.metric("NASDAQ", "14,200.00", "+112.50")

with col3:
    st.metric("DOW", "35,000.00", "+525.00")

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
st.success("✅ Step 1 Complete: Minimal Streamlit app working!")
st.info("Next: Add real market data with yfinance")







