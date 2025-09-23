import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Financial Analyzer Pro - Day 4",
    page_icon="💼",
    layout="wide"
)

# Initialize session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = []

def get_stock_price(symbol):
    """Get current stock price"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1d", timeout=10)
        if not data.empty:
            return data['Close'].iloc[-1]
        return None
    except:
        return None

def main():
    st.title("📊 Financial Analyzer Pro - Day 4 Portfolio")
    st.markdown("**Portfolio Management with Real Tracking**")
    
    # Add position form
    st.subheader("➕ Add New Position")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        symbol = st.text_input("Symbol", value="AAPL", key="add_symbol")
    with col2:
        shares = st.number_input("Shares", min_value=0.01, value=10.0, key="add_shares")
    with col3:
        cost_basis = st.number_input("Cost per Share", min_value=0.01, value=150.0, key="add_cost")
    with col4:
        if st.button("Add Position", type="primary"):
            if symbol and shares > 0 and cost_basis > 0:
                # Get current price
                current_price = get_stock_price(symbol)
                if current_price:
                    position = {
                        'symbol': symbol.upper(),
                        'shares': shares,
                        'cost_basis': cost_basis,
                        'current_price': current_price,
                        'date_added': datetime.now().strftime('%Y-%m-%d')
                    }
                    st.session_state.portfolio.append(position)
                    st.success(f"✅ Added {shares} shares of {symbol.upper()}")
                    st.rerun()
                else:
                    st.error(f"❌ Could not fetch price for {symbol}")
            else:
                st.error("❌ Please fill in all fields")
    
    # Display portfolio
    if st.session_state.portfolio:
        st.subheader("📊 Your Portfolio")
        
        total_value = 0
        total_cost = 0
        
        for i, position in enumerate(st.session_state.portfolio):
            # Get current price
            current_price = get_stock_price(position['symbol'])
            if current_price:
                position['current_price'] = current_price
            
            # Calculate metrics
            position_value = position['current_price'] * position['shares']
            position_cost = position['cost_basis'] * position['shares']
            position_pnl = position_value - position_cost
            position_pnl_percent = (position_pnl / position_cost * 100) if position_cost > 0 else 0
            
            col1, col2, col3, col4, col5, col6 = st.columns([2, 1, 1, 1, 1, 1])
            
            with col1:
                st.write(f"**{position['symbol']}**")
                st.caption(f"Added: {position['date_added']}")
            
            with col2:
                st.write(f"{position['shares']} shares")
            
            with col3:
                st.write(f"${position['cost_basis']:.2f} cost")
            
            with col4:
                st.write(f"${position['current_price']:.2f} current")
            
            with col5:
                pnl_color = "🟢" if position_pnl >= 0 else "🔴"
                st.write(f"{pnl_color} ${position_pnl:.2f}")
                st.write(f"({position_pnl_percent:+.1f}%)")
            
            with col6:
                if st.button("❌", key=f"remove_{i}"):
                    st.session_state.portfolio.pop(i)
                    st.rerun()
            
            total_value += position_value
            total_cost += position_cost
        
        # Portfolio summary
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Value", f"${total_value:,.2f}")
        with col2:
            st.metric("Total Cost", f"${total_cost:,.2f}")
        with col3:
            total_pnl = total_value - total_cost
            st.metric("Total P&L", f"${total_pnl:,.2f}")
        with col4:
            total_pnl_percent = (total_pnl / total_cost * 100) if total_cost > 0 else 0
            st.metric("P&L %", f"{total_pnl_percent:+.2f}%")
    
    else:
        st.info("📝 No positions in portfolio. Add some stocks to get started!")
        
        # Sample portfolio
        if st.button("🚀 Add Sample Portfolio"):
            sample_positions = [
                {"symbol": "AAPL", "shares": 10, "cost_basis": 150.0},
                {"symbol": "MSFT", "shares": 5, "cost_basis": 300.0},
                {"symbol": "GOOGL", "shares": 3, "cost_basis": 2500.0}
            ]
            
            for pos in sample_positions:
                current_price = get_stock_price(pos['symbol'])
                if current_price:
                    position = {
                        'symbol': pos['symbol'],
                        'shares': pos['shares'],
                        'cost_basis': pos['cost_basis'],
                        'current_price': current_price,
                        'date_added': datetime.now().strftime('%Y-%m-%d')
                    }
                    st.session_state.portfolio.append(position)
            
            st.success("Sample portfolio added!")
            st.rerun()

if __name__ == "__main__":
    main()




