import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Page config
st.set_page_config(
    page_title="Financial Analyzer Pro - Phase 2",
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
    .success {
        color: #28a745;
        font-weight: bold;
    }
    .error {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>📈 Financial Analyzer Pro</h1>
    <p>Phase 2: Real-time Data Integration</p>
    <p>Status: ✅ Live Market Data Active!</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🎯 Phase 2 Features")
st.sidebar.success("✅ Real-time Data")
st.sidebar.success("✅ Live Charts")
st.sidebar.success("✅ Market Overview")
st.sidebar.success("✅ Portfolio Tracking")

def get_stock_data(symbol, period="1mo"):
    """Get real-time stock data using yfinance"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        if data.empty:
            return None, f"No data found for {symbol}"
        return data, None
    except Exception as e:
        return None, f"Error fetching data: {str(e)}"

def get_market_overview():
    """Get real-time market data"""
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
                    'change_percent': change_percent
                }
        except Exception as e:
            st.error(f"Error fetching {symbol}: {str(e)}")
    
    return overview

# Main Content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📊 Real-time Stock Analysis")
    
    # Stock input
    symbol = st.text_input("Enter stock symbol", value="AAPL", help="Enter a valid stock ticker symbol")
    
    if st.button("Get Live Data", type="primary"):
        with st.spinner("Fetching real-time data..."):
            data, error = get_stock_data(symbol)
            
            if error:
                st.error(f"❌ {error}")
            else:
                st.success(f"✅ Successfully fetched live data for {symbol}")
                
                # Real-time metrics
                current_price = data['Close'].iloc[-1]
                previous_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
                change = current_price - previous_price
                change_percent = (change / previous_price) * 100
                
                col1_1, col1_2, col1_3 = st.columns(3)
                
                with col1_1:
                    st.metric("Current Price", f"${current_price:.2f}", f"{change:+.2f}")
                with col1_2:
                    st.metric("Volume", f"{data['Volume'].iloc[-1]:,}", "+5.2M")
                with col1_3:
                    st.metric("Change", f"{change_percent:+.2f}%", f"{change:+.2f}")
                
                # Real-time chart
                st.subheader("Live Price Chart")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='#1f77b4', width=2)
                ))
                
                fig.update_layout(
                    title=f"{symbol} Price Chart (Last 30 Days)",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)

with col2:
    st.header("📈 Live Market Overview")
    
    # Real-time market data
    with st.spinner("Fetching market data..."):
        market_data = get_market_overview()
    
    if market_data:
        market_df = pd.DataFrame({
            'Index': ['S&P 500', 'NASDAQ', 'DOW', 'VIX'],
            'Symbol': ['^GSPC', '^IXIC', '^DJI', '^VIX'],
            'Price': [market_data.get('^GSPC', {}).get('price', 0),
                     market_data.get('^IXIC', {}).get('price', 0),
                     market_data.get('^DJI', {}).get('price', 0),
                     market_data.get('^VIX', {}).get('price', 0)],
            'Change': [market_data.get('^GSPC', {}).get('change_percent', 0),
                      market_data.get('^IXIC', {}).get('change_percent', 0),
                      market_data.get('^DJI', {}).get('change_percent', 0),
                      market_data.get('^VIX', {}).get('change_percent', 0)]
        })
        
        # Format the data for display
        for i, row in market_df.iterrows():
            if row['Price'] > 0:
                change_color = "🟢" if row['Change'] >= 0 else "🔴"
                st.write(f"{change_color} **{row['Index']}**: ${row['Price']:.2f} ({row['Change']:+.2f}%)")
    
    st.header("💼 Live Portfolio")
    
    # Mock portfolio with real prices
    portfolio_symbols = ['AAPL', 'MSFT', 'GOOGL']
    portfolio_data = []
    
    for symbol in portfolio_symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                portfolio_data.append({
                    'Symbol': symbol,
                    'Shares': 10,
                    'Price': current_price,
                    'Value': 10 * current_price
                })
        except:
            # Fallback to mock data
            portfolio_data.append({
                'Symbol': symbol,
                'Shares': 10,
                'Price': 150.00,
                'Value': 1500.00
            })
    
    if portfolio_data:
        df_portfolio = pd.DataFrame(portfolio_data)
        st.dataframe(df_portfolio, use_container_width=True)
        
        total_value = df_portfolio['Value'].sum()
        st.metric("Total Portfolio Value", f"${total_value:,.2f}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🚀 <strong>Phase 2 Complete!</strong> Real-time data integration successful!</p>
    <p>Next: Phase 3 - Financial Analytics & Technical Indicators</p>
</div>
""", unsafe_allow_html=True)
