import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Financial Analyzer Pro",
    page_icon="📈",
    layout="wide"
)

# Main app
def main():
    st.title("📈 Financial Analyzer Pro")
    
    st.markdown("""
    ## Welcome to Financial Analyzer Pro!
    
    This is a working Streamlit application deployed on Render.
    
    ### Features:
    - ✅ **Real-time Analysis** - Coming soon
    - ✅ **Portfolio Management** - Coming soon  
    - ✅ **Technical Indicators** - Coming soon
    - ✅ **Machine Learning** - Coming soon
    
    ### Status: 🟢 **LIVE AND WORKING**
    
    The app is successfully deployed and running!
    """)
    
    # Simple interactive element
    st.subheader("Test Input")
    user_input = st.text_input("Enter your name:", value="User")
    
    if st.button("Say Hello"):
        st.success(f"Hello, {user_input}! The app is working perfectly.")
    
    # Simple chart
    st.subheader("Sample Chart")
    import pandas as pd
    import numpy as np
    
    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['A', 'B', 'C']
    )
    
    st.line_chart(chart_data)
    
    st.info("🎉 **Deployment Successful!** This basic version is working on Render.")

if __name__ == "__main__":
    main()
