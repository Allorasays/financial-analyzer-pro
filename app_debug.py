import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import time
import sys
import os
from typing import Dict, List, Optional

# Debug information
st.write("Python version:", sys.version)
st.write("Working directory:", os.getcwd())
st.write("Environment variables:", {k: v for k, v in os.environ.items() if 'PORT' in k or 'STREAMLIT' in k})

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

# Simple cached functions
@st.cache_data(ttl=300)
def get_simple_data(symbol: str) -> Optional[Dict]:
    """Get simple stock data"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="5d")
        if hist.empty:
            return None
        
        current = hist['Close'].iloc[-1]
        previous = hist['Close'].iloc[-2]
        change = current - previous
        change_pct = (change / previous) * 100
        
        return {
            'price': current,
            'change': change,
            'change_pct': change_pct
        }
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def main():
    st.markdown("""
    <div class="main-header">
        <h1>📈 Financial Analyzer Pro</h1>
        <p>Debug Version - Testing Deployment</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.success("✅ App is running successfully!")
    
    # Simple stock lookup
    st.header("🔍 Stock Lookup")
    symbol = st.text_input("Enter stock symbol", value="AAPL").upper()
    
    if st.button("Get Quote"):
        if symbol:
            with st.spinner("Fetching data..."):
                data = get_simple_data(symbol)
                if data:
                    st.success(f"**{symbol}**: ${data['price']:.2f} ({data['change']:+.2f}, {data['change_pct']:+.2f}%)")
                else:
                    st.error(f"Could not fetch data for {symbol}")
    
    # Simple market overview
    st.header("📊 Market Overview")
    st.info("This is a simplified version for testing deployment.")
    
    # Test basic functionality
    st.header("🧪 Functionality Test")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("S&P 500", "4,500.00", "+54.20 (+1.22%)")
    with col2:
        st.metric("NASDAQ", "14,200.00", "+112.50 (+0.80%)")
    with col3:
        st.metric("DOW", "35,000.00", "+525.00 (+1.52%)")
    
    st.success("🎉 All basic functionality is working!")

if __name__ == "__main__":
    main()






