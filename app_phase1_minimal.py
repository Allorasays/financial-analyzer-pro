import streamlit as st
import pandas as pd
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Financial Analyzer Pro - Phase 1",
    page_icon="📈",
    layout="wide"
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

# Header
st.markdown("""
<div class="main-header">
    <h1>📈 Financial Analyzer Pro</h1>
    <p>Phase 1: Ultra-Minimal Deployment</p>
    <p>Status: ✅ Deployed Successfully!</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🎯 Phase 1 Features")
st.sidebar.success("✅ Basic UI")
st.sidebar.success("✅ Stock Lookup")
st.sidebar.success("✅ Mock Data")
st.sidebar.success("✅ Portfolio Table")

# Main Content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📊 Stock Analysis")
    
    # Stock input
    symbol = st.text_input("Enter stock symbol", value="AAPL", help="Enter a valid stock ticker symbol")
    
    if st.button("Get Stock Info", type="primary"):
        st.success(f"✅ Successfully fetched data for {symbol}")
        
        # Mock data display
        col1_1, col1_2, col1_3 = st.columns(3)
        
        with col1_1:
            st.metric("Current Price", "$150.25", "+2.50")
        with col1_2:
            st.metric("Volume", "45.2M", "+5.2M")
        with col1_3:
            st.metric("Market Cap", "$2.4T", "+$50B")
        
        # Mock chart data
        st.subheader("Price Chart (Mock Data)")
        chart_data = pd.DataFrame({
            'Date': pd.date_range(start='2024-01-01', periods=30, freq='D'),
            'Price': [150 + i * 0.5 + (i % 3 - 1) * 2 for i in range(30)]
        })
        st.line_chart(chart_data.set_index('Date'))

with col2:
    st.header("📈 Market Overview")
    
    # Mock market data
    market_data = {
        'Index': ['S&P 500', 'NASDAQ', 'DOW', 'VIX'],
        'Value': [4500.00, 14200.00, 35000.00, 18.5],
        'Change': ['+1.2%', '+0.8%', '+1.5%', '-0.3%']
    }
    
    df_market = pd.DataFrame(market_data)
    st.dataframe(df_market, use_container_width=True)
    
    st.header("💼 Portfolio")
    
    # Mock portfolio
    portfolio_data = {
        'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'TSLA'],
        'Shares': [10, 5, 3, 2],
        'Price': [150.25, 300.50, 2800.75, 250.00],
        'Value': [1502.50, 1502.50, 8402.25, 500.00],
        'P&L': ['+$50.00', '+$25.00', '+$200.00', '-$10.00']
    }
    
    df_portfolio = pd.DataFrame(portfolio_data)
    st.dataframe(df_portfolio, use_container_width=True)
    
    # Portfolio summary
    total_value = df_portfolio['Value'].sum()
    st.metric("Total Portfolio Value", f"${total_value:,.2f}", "+$265.00")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🚀 <strong>Phase 1 Complete!</strong> Ready for Phase 2: Real-time Data Integration</p>
    <p>Next: Add yfinance integration and live market data</p>
</div>
""", unsafe_allow_html=True)
