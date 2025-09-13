import streamlit as st

st.title("📈 Financial Analyzer Pro")
st.write("Welcome to Financial Analyzer Pro!")

st.header("Stock Lookup")
symbol = st.text_input("Enter stock symbol", value="AAPL")

if st.button("Get Info"):
    st.write(f"Symbol: {symbol}")
    st.write("This is a minimal version for testing deployment.")
    st.write("Full features will be available in the complete version.")

st.header("Market Overview")
st.write("S&P 500: 4,500.00 (+1.2%)")
st.write("NASDAQ: 14,200.00 (+0.8%)")
st.write("DOW: 35,000.00 (+1.5%)")

st.header("Portfolio")
portfolio_data = {
    'Symbol': ['AAPL', 'MSFT', 'GOOGL'],
    'Shares': [10, 5, 3],
    'Price': [150.25, 300.50, 2800.75],
    'Value': [1502.50, 1502.50, 8402.25]
}

st.dataframe(portfolio_data, width='stretch')