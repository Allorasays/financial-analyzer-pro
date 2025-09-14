import streamlit as st

st.title("📈 Financial Analyzer Pro - Step 1")
st.write("This is the most basic version to confirm Streamlit is working.")

st.header("Basic Dashboard")
st.write("If you can see this, the basic app is working!")

# Simple metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("S&P 500", "4,523.45", "+1.2%")
with col2:
    st.metric("NASDAQ", "14,234.67", "+0.8%")
with col3:
    st.metric("DOW", "35,123.89", "-0.3%")

st.success("✅ Step 1 Complete: Basic Streamlit app is working!")



