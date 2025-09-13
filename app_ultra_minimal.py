import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

# Page config
st.set_page_config(
    page_title="Financial Analyzer Pro",
    page_icon="📈",
    layout="wide"
)

def main():
    st.title("📈 Financial Analyzer Pro")
    st.write("Welcome to Financial Analyzer Pro!")
    
    # Simple stock lookup
    symbol = st.text_input("Enter stock symbol", value="AAPL").upper()
    
    if st.button("Get Stock Data"):
        if symbol:
            try:
                with st.spinner("Fetching data..."):
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period="1mo")
                    
                    if not data.empty:
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
                        
                        # Simple chart
                        st.line_chart(data['Close'])
                    else:
                        st.error(f"No data found for {symbol}")
            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")
    
    # Market overview
    st.header("Market Overview")
    try:
        with st.spinner("Loading market data..."):
            # S&P 500
            sp500 = yf.Ticker("^GSPC")
            sp500_data = sp500.history(period="2d")
            
            if not sp500_data.empty:
                current = sp500_data['Close'].iloc[-1]
                previous = sp500_data['Close'].iloc[-2]
                change = current - previous
                change_pct = (change / previous) * 100
                
                st.metric("S&P 500", f"${current:.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
    except Exception as e:
        st.warning("Could not load market data")
    
    # Portfolio mockup
    st.header("Portfolio")
    portfolio_data = {
        'Symbol': ['AAPL', 'MSFT', 'GOOGL'],
        'Shares': [10, 5, 3],
        'Price': [150.25, 300.50, 2800.75],
        'Value': [1502.50, 1502.50, 8402.25]
    }
    
    df = pd.DataFrame(portfolio_data)
    st.dataframe(df, use_container_width=True)
    
    total_value = df['Value'].sum()
    st.metric("Total Portfolio Value", f"${total_value:,.2f}")

if __name__ == "__main__":
    main()