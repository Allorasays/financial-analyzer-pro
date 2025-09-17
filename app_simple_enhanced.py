import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="Financial Analyzer Pro",
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
</style>
""", unsafe_allow_html=True)

def get_stock_data(symbol, period="1mo"):
    """Get stock data using yfinance"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def create_simple_chart(data, symbol):
    """Create a simple price chart"""
    fig = go.Figure()
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
        height=400
    )
    
    return fig

def main():
    st.markdown("""
    <div class="main-header">
        <h1>📈 Financial Analyzer Pro</h1>
        <p>Enhanced Financial Analysis Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "Dashboard", "Stock Analysis", "Market Overview"
    ])
    
    if page == "Dashboard":
        st.header("📊 Dashboard")
        
        # Market Overview
        st.subheader("Market Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("S&P 500", "4,523.45", "+1.2%")
        with col2:
            st.metric("NASDAQ", "14,234.67", "+0.8%")
        with col3:
            st.metric("DOW", "35,123.89", "-0.3%")
        
        # Quick Analysis
        st.subheader("Quick Stock Analysis")
        symbol = st.text_input("Enter stock symbol", value="AAPL").upper()
        
        if st.button("Analyze Stock"):
            if symbol:
                with st.spinner(f"Analyzing {symbol}..."):
                    data = get_stock_data(symbol, "1mo")
                    if data is not None and not data.empty:
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
                        
                        # Price chart
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
                    data = get_stock_data(symbol, period)
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
                        
                        # Price chart
                        st.subheader("Price Chart")
                        fig = create_simple_chart(data, symbol)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(f"Could not fetch data for {symbol}")
    
    elif page == "Market Overview":
        st.header("🌍 Market Overview")
        
        # Sample market data
        st.subheader("Major Indices")
        
        market_data = {
            'Index': ['S&P 500', 'NASDAQ', 'DOW', 'VIX'],
            'Price': [4523.45, 14234.67, 35123.89, 18.45],
            'Change': [52.34, 112.67, -105.23, -1.23],
            'Change %': [1.17, 0.80, -0.30, -6.25]
        }
        
        df = pd.DataFrame(market_data)
        st.dataframe(df, use_container_width=True)
        
        # Sample portfolio
        st.subheader("Sample Portfolio")
        portfolio_data = {
            'Symbol': ['AAPL', 'MSFT', 'GOOGL'],
            'Shares': [10, 5, 3],
            'Current Price': [150.25, 300.50, 2800.75],
            'Total Value': [1502.50, 1502.50, 8402.25],
            'Gain/Loss': [52.50, -25.00, 150.75]
        }
        
        df_portfolio = pd.DataFrame(portfolio_data)
        st.dataframe(df_portfolio, use_container_width=True)

if __name__ == "__main__":
    main()








