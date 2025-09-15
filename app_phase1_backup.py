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

def get_market_data(symbol: str, period: str = "1mo"):
    """Get market data using yfinance"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def main():
    # Header
    st.markdown('<div class="main-header"><h1>📈 Financial Analyzer Pro</h1><p>Simplified Deployment Version</p></div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.selectbox("Choose Analysis", ["📈 Stock Analysis", "📊 Market Overview", "💼 Portfolio"])
    
    if page == "📈 Stock Analysis":
        stock_analysis_page()
    elif page == "📊 Market Overview":
        market_overview_page()
    elif page == "💼 Portfolio":
        portfolio_page()

def stock_analysis_page():
    """Stock analysis page"""
    st.header("📈 Stock Analysis")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        symbol = st.text_input("Enter Stock Symbol", value="AAPL", placeholder="e.g., AAPL, MSFT, GOOGL")
    with col2:
        timeframe = st.selectbox("Timeframe", ["1mo", "3mo", "6mo", "1y", "2y", "5y"])
    
    if st.button("Analyze Stock", type="primary"):
        if symbol:
            analyze_stock(symbol, timeframe)
        else:
            st.error("Please enter a stock symbol")

def analyze_stock(symbol, timeframe):
    """Perform stock analysis"""
    with st.spinner(f"Fetching data for {symbol}..."):
        data = get_market_data(symbol, timeframe)
        
        if data is not None and not data.empty:
            # Basic metrics
            current_price = data['Close'].iloc[-1]
            prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
            change = current_price - prev_price
            change_pct = (change / prev_price) * 100 if prev_price != 0 else 0
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Price", f"${current_price:.2f}")
            with col2:
                st.metric("Change", f"${change:.2f}")
            with col3:
                st.metric("Change %", f"{change_pct:.2f}%")
            with col4:
                st.metric("Volume", f"{data['Volume'].iloc[-1]:,}")
            
            # Price chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='#667eea', width=2)
            ))
            
            fig.update_layout(
                title=f"{symbol} Price Chart ({timeframe})",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Basic analysis
            st.subheader("📊 Quick Analysis")
            price_trend = "📈 Bullish" if change > 0 else "📉 Bearish" if change < 0 else "➡️ Neutral"
            st.write(f"**Price Trend:** {price_trend}")
            
            volatility = data['Close'].pct_change().std() * 100
            st.write(f"**Volatility:** {volatility:.2f}%")
            
            support = data['Low'].min()
            resistance = data['High'].max()
            st.write(f"**Support:** ${support:.2f} | **Resistance:** ${resistance:.2f}")

def market_overview_page():
    """Market overview page"""
    st.header("📊 Market Overview")
    
    # Major indices
    indices = {
        'S&P 500': '^GSPC',
        'NASDAQ': '^IXIC',
        'DOW': '^DJI'
    }
    
    col1, col2, col3 = st.columns(3)
    
    for i, (name, symbol) in enumerate(indices.items()):
        with [col1, col2, col3][i]:
            try:
                data = get_market_data(symbol, "1d")
                if data is not None and not data.empty:
                    current = data['Close'].iloc[-1]
                    prev = data['Close'].iloc[-2] if len(data) > 1 else current
                    change = current - prev
                    change_pct = (change / prev) * 100 if prev != 0 else 0
                    
                    st.metric(
                        name,
                        f"${current:.2f}",
                        f"{change:+.2f} ({change_pct:+.2f}%)"
                    )
                else:
                    st.metric(name, "N/A", "Error")
            except:
                st.metric(name, "N/A", "Error")

def portfolio_page():
    """Portfolio page"""
    st.header("💼 Portfolio")
    
    # Sample portfolio data
    portfolio_data = {
        'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'TSLA'],
        'Shares': [10, 5, 3, 2],
        'Purchase Price': [150.25, 300.50, 2800.75, 200.00],
        'Current Price': [175.30, 320.15, 2900.20, 185.50],
        'Value': [1753.00, 1600.75, 8700.60, 371.00]
    }
    
    df = pd.DataFrame(portfolio_data)
    df['P&L'] = df['Value'] - (df['Shares'] * df['Purchase Price'])
    df['P&L %'] = (df['P&L'] / (df['Shares'] * df['Purchase Price'])) * 100
    
    st.dataframe(df, use_container_width=True)
    
    total_value = df['Value'].sum()
    total_cost = (df['Shares'] * df['Purchase Price']).sum()
    total_pnl = total_value - total_cost
    total_pnl_pct = (total_pnl / total_cost) * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Value", f"${total_value:,.2f}")
    with col2:
        st.metric("Total P&L", f"${total_pnl:,.2f}")
    with col3:
        st.metric("P&L %", f"{total_pnl_pct:.2f}%")

if __name__ == "__main__":
    main()
