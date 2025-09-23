#!/usr/bin/env python3
"""
Portfolio Management UI Components
Day 4 Implementation: Portfolio Management Interface
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from portfolio_manager import PortfolioManager, generate_portfolio_report
import yfinance as yf

def get_current_prices(symbols: list) -> dict:
    """Get current prices for multiple symbols"""
    prices = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d")
            if not data.empty:
                prices[symbol] = data['Close'].iloc[-1]
            else:
                prices[symbol] = 100.0  # Fallback price
        except:
            prices[symbol] = 100.0  # Fallback price
    return prices

def show_portfolio_management():
    """Main portfolio management interface"""
    st.header("💼 Portfolio Management")
    
    # Initialize portfolio manager
    portfolio_manager = PortfolioManager()
    
    # Sidebar controls
    st.sidebar.subheader("📊 Portfolio Controls")
    
    # Portfolio selection
    portfolios = portfolio_manager.get_portfolios()
    portfolio_names = [f"{p['name']} ({p['id'][:8]}...)" for p in portfolios]
    portfolio_names.insert(0, "Create New Portfolio")
    
    selected_portfolio = st.sidebar.selectbox("Select Portfolio", portfolio_names)
    
    if selected_portfolio == "Create New Portfolio":
        show_create_portfolio_form(portfolio_manager)
    else:
        # Get selected portfolio ID
        selected_index = portfolio_names.index(selected_portfolio) - 1
        portfolio_id = portfolios[selected_index]['id']
        portfolio_name = portfolios[selected_index]['name']
        
        st.subheader(f"📈 {portfolio_name}")
        
        # Portfolio actions
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("➕ Add Position", key="add_pos"):
                st.session_state.show_add_position = True
        with col2:
            if st.button("📊 View Performance", key="view_perf"):
                st.session_state.show_performance = True
        with col3:
            if st.button("🗑️ Delete Portfolio", key="delete_port", type="secondary"):
                st.session_state.show_delete_confirm = True
        
        # Show portfolio content
        show_portfolio_overview(portfolio_manager, portfolio_id, portfolio_name)
        
        # Show add position form if requested
        if st.session_state.get('show_add_position', False):
            show_add_position_form(portfolio_manager, portfolio_id)
        
        # Show performance analysis if requested
        if st.session_state.get('show_performance', False):
            show_portfolio_performance(portfolio_manager, portfolio_id, portfolio_name)
        
        # Show delete confirmation if requested
        if st.session_state.get('show_delete_confirm', False):
            show_delete_confirmation(portfolio_manager, portfolio_id, portfolio_name)

def show_create_portfolio_form(portfolio_manager: PortfolioManager):
    """Show form to create new portfolio"""
    st.subheader("🆕 Create New Portfolio")
    
    with st.form("create_portfolio"):
        name = st.text_input("Portfolio Name", placeholder="e.g., Growth Portfolio, Retirement Fund")
        description = st.text_area("Description (Optional)", placeholder="Brief description of this portfolio...")
        
        submitted = st.form_submit_button("Create Portfolio", type="primary")
        
        if submitted:
            if name:
                portfolio_id = portfolio_manager.create_portfolio(name, description)
                st.success(f"✅ Portfolio '{name}' created successfully!")
                st.rerun()
            else:
                st.error("Please enter a portfolio name")

def show_portfolio_overview(portfolio_manager: PortfolioManager, portfolio_id: str, portfolio_name: str):
    """Show portfolio overview with positions and metrics"""
    # Get current positions
    positions = portfolio_manager.get_positions(portfolio_id)
    
    if not positions:
        st.info("📝 No positions in this portfolio. Add some positions to get started!")
        return
    
    # Get current prices
    symbols = [pos['symbol'] for pos in positions]
    current_prices = get_current_prices(symbols)
    
    # Update prices in database
    portfolio_manager.update_position_prices(portfolio_id, current_prices)
    
    # Calculate metrics
    metrics = portfolio_manager.calculate_portfolio_metrics(portfolio_id, current_prices)
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Value", f"${metrics['total_value']:,.2f}")
    with col2:
        st.metric("Total Cost", f"${metrics['total_cost']:,.2f}")
    with col3:
        st.metric("Total P&L", f"${metrics['total_pnl']:,.2f}")
    with col4:
        st.metric("P&L %", f"{metrics['total_pnl_percent']:.2f}%")
    
    # Positions table
    st.subheader("📊 Current Positions")
    
    positions_data = []
    for pos in metrics['positions']:
        positions_data.append({
            'Symbol': pos['symbol'],
            'Quantity': f"{pos['quantity']:.2f}",
            'Purchase Price': f"${pos['purchase_price']:.2f}",
            'Current Price': f"${pos['current_price']:.2f}",
            'Cost Basis': f"${pos['cost_basis']:,.2f}",
            'Current Value': f"${pos['current_value']:,.2f}",
            'P&L': f"${pos['pnl']:,.2f}",
            'P&L %': f"{pos['pnl_percent']:.2f}%",
            'Weight': f"{pos['weight']:.1f}%"
        })
    
    df = pd.DataFrame(positions_data)
    st.dataframe(df, use_container_width=True)
    
    # Portfolio allocation chart
    if len(positions_data) > 1:
        st.subheader("🥧 Portfolio Allocation")
        
        fig = px.pie(
            values=[pos['current_value'] for pos in metrics['positions']],
            names=[pos['symbol'] for pos in metrics['positions']],
            title="Portfolio Allocation by Value"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent transactions
    st.subheader("📋 Recent Transactions")
    transactions = portfolio_manager.get_transactions(portfolio_id)[:10]  # Last 10
    
    if transactions:
        trans_data = []
        for trans in transactions:
            trans_data.append({
                'Date': trans['date'],
                'Symbol': trans['symbol'],
                'Type': trans['type'],
                'Quantity': f"{trans['quantity']:.2f}",
                'Price': f"${trans['price']:.2f}",
                'Total': f"${trans['quantity'] * trans['price']:,.2f}",
                'Notes': trans['notes']
            })
        
        trans_df = pd.DataFrame(trans_data)
        st.dataframe(trans_df, use_container_width=True)
    else:
        st.info("No transactions yet")

def show_add_position_form(portfolio_manager: PortfolioManager, portfolio_id: str):
    """Show form to add new position"""
    st.subheader("➕ Add New Position")
    
    with st.form("add_position"):
        col1, col2 = st.columns(2)
        
        with col1:
            symbol = st.text_input("Stock Symbol", placeholder="e.g., AAPL, MSFT").upper()
            quantity = st.number_input("Quantity", min_value=0.01, value=1.0, step=0.01)
        
        with col2:
            purchase_price = st.number_input("Purchase Price ($)", min_value=0.01, value=100.0, step=0.01)
            purchase_date = st.date_input("Purchase Date", value=datetime.now().date())
        
        notes = st.text_area("Notes (Optional)", placeholder="Any additional notes about this position...")
        
        col1, col2 = st.columns(2)
        with col1:
            submitted = st.form_submit_button("Add Position", type="primary")
        with col2:
            if st.form_submit_button("Cancel"):
                st.session_state.show_add_position = False
                st.rerun()
        
        if submitted:
            if symbol and quantity > 0 and purchase_price > 0:
                try:
                    position_id = portfolio_manager.add_position(
                        portfolio_id, symbol, quantity, purchase_price, 
                        purchase_date.isoformat(), notes
                    )
                    st.success(f"✅ Added {quantity} shares of {symbol} at ${purchase_price:.2f}")
                    st.session_state.show_add_position = False
                    st.rerun()
                except Exception as e:
                    st.error(f"Error adding position: {str(e)}")
            else:
                st.error("Please fill in all required fields")

def show_portfolio_performance(portfolio_manager: PortfolioManager, portfolio_id: str, portfolio_name: str):
    """Show detailed portfolio performance analysis"""
    st.subheader(f"📈 Performance Analysis: {portfolio_name}")
    
    # Get current positions and prices
    positions = portfolio_manager.get_positions(portfolio_id)
    symbols = [pos['symbol'] for pos in positions]
    current_prices = get_current_prices(symbols)
    
    # Calculate current metrics
    metrics = portfolio_manager.calculate_portfolio_metrics(portfolio_id, current_prices)
    
    # Save current performance snapshot
    portfolio_manager.save_portfolio_performance(portfolio_id, metrics)
    
    # Get performance history
    performance_history = portfolio_manager.get_portfolio_performance_history(portfolio_id, 30)
    
    if performance_history:
        # Performance chart
        st.subheader("📊 Portfolio Value Over Time")
        
        df_perf = pd.DataFrame(performance_history)
        df_perf['date'] = pd.to_datetime(df_perf['date'])
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Portfolio Value', 'P&L Percentage'),
            vertical_spacing=0.1
        )
        
        # Portfolio value
        fig.add_trace(
            go.Scatter(x=df_perf['date'], y=df_perf['total_value'], 
                      name='Total Value', line=dict(color='blue')),
            row=1, col=1
        )
        
        # P&L percentage
        fig.add_trace(
            go.Scatter(x=df_perf['date'], y=df_perf['total_pnl_percent'], 
                      name='P&L %', line=dict(color='green')),
            row=2, col=1
        )
        
        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics
        st.subheader("📊 Performance Metrics")
        
        if len(performance_history) > 1:
            first_value = performance_history[0]['total_value']
            last_value = performance_history[-1]['total_value']
            total_return = ((last_value - first_value) / first_value * 100) if first_value > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("30-Day Return", f"{total_return:.2f}%")
            with col2:
                st.metric("Current Value", f"${metrics['total_value']:,.2f}")
            with col3:
                st.metric("Total P&L", f"${metrics['total_pnl']:,.2f}")
            with col4:
                st.metric("P&L %", f"{metrics['total_pnl_percent']:.2f}%")
    
    # Risk metrics
    st.subheader("⚠️ Risk Analysis")
    risk_metrics = calculate_portfolio_risk_metrics(metrics['positions'])
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Portfolio Volatility", f"{risk_metrics.get('portfolio_volatility', 0):.2f}")
    with col2:
        st.metric("Portfolio Beta", f"{risk_metrics.get('portfolio_beta', 0):.2f}")
    with col3:
        st.metric("Concentration Risk", f"{risk_metrics.get('concentration_risk', 0):.1%}")
    with col4:
        st.metric("Diversification", f"{risk_metrics.get('diversification_ratio', 0):.1f}/10")

def show_delete_confirmation(portfolio_manager: PortfolioManager, portfolio_id: str, portfolio_name: str):
    """Show portfolio deletion confirmation"""
    st.warning(f"⚠️ Are you sure you want to delete '{portfolio_name}'?")
    st.error("This action cannot be undone and will delete all positions and transaction history.")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Yes, Delete Portfolio", type="primary"):
            portfolio_manager.delete_portfolio(portfolio_id)
            st.success("✅ Portfolio deleted successfully!")
            st.session_state.show_delete_confirm = False
            st.rerun()
    with col2:
        if st.button("❌ Cancel"):
            st.session_state.show_delete_confirm = False
            st.rerun()

def calculate_portfolio_risk_metrics(positions: list) -> dict:
    """Calculate portfolio risk metrics"""
    if not positions:
        return {}
    
    # Calculate position weights
    total_value = sum(pos['current_value'] for pos in positions)
    weights = [pos['current_value'] / total_value for pos in positions] if total_value > 0 else []
    
    # Calculate weighted average metrics
    weighted_volatility = sum(w * 0.2 for w in weights)  # Simplified volatility
    portfolio_beta = sum(w * 1.0 for w in weights)  # Simplified beta
    
    return {
        'portfolio_volatility': weighted_volatility,
        'portfolio_beta': portfolio_beta,
        'concentration_risk': max(weights) if weights else 0,
        'diversification_ratio': len(positions) / 10.0  # Simplified diversification
    }
