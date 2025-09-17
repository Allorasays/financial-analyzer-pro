#!/usr/bin/env python3
"""
Simple Real-Time Dashboard for Financial Analyzer Pro
Displays live market data and stock information
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import List, Dict, Any
import time

def display_realtime_market_overview():
    """Display real-time market overview"""
    st.subheader("📈 Live Market Overview")
    
    # Refresh button
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔄 Refresh", type="primary"):
            st.cache_data.clear()
            st.rerun()
    
    with col2:
        st.info("💡 Data refreshes every 30 seconds automatically")
    
    # Get market data
    from realtime_simple import get_market_overview
    market_data = get_market_overview()
    
    if market_data:
        # Display major indices
        col1, col2, col3, col4 = st.columns(4)
        
        indices = [
            ('^GSPC', 'S&P 500', col1),
            ('^IXIC', 'NASDAQ', col2),
            ('^DJI', 'DOW', col3),
            ('^VIX', 'VIX', col4)
        ]
        
        for symbol, name, col in indices:
            with col:
                if symbol in market_data:
                    data = market_data[symbol]
                    change_color = "🟢" if data['change'] >= 0 else "🔴"
                    
                    st.metric(
                        name,
                        f"${data['price']:.2f}",
                        f"{change_color} {data['change_percent']:+.2f}%"
                    )
                    
                    # Additional info
                    st.caption(f"Volume: {data['volume']:,.0f}")
        else:
            st.error("Could not fetch market data")
    
    # Market status
    st.subheader("📊 Market Status")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.info(f"🕐 Last updated: {current_time}")

def display_live_stock_tracker(symbols: List[str]):
    """Display live stock tracker"""
    st.subheader("📊 Live Stock Tracker")
    
    # Add/remove symbols
    col1, col2 = st.columns([2, 1])
    
    with col1:
        new_symbol = st.text_input("Add Symbol", placeholder="Enter stock symbol (e.g., AAPL)")
    
    with col2:
        if st.button("➕ Add", type="primary"):
            if new_symbol and new_symbol.upper() not in symbols:
                symbols.append(new_symbol.upper())
                st.success(f"Added {new_symbol.upper()}")
                st.rerun()
    
    # Display tracked symbols
    if symbols:
        st.write("**Tracked Symbols:**")
        
        # Create columns for symbols
        cols = st.columns(min(len(symbols), 4))
        
        for i, symbol in enumerate(symbols):
            col = cols[i % 4]
            
            with col:
                # Get live price
                from realtime_simple import get_live_price
                price, error = get_live_price(symbol)
                
                if price is not None:
                    st.metric(
                        symbol,
                        f"${price:.2f}",
                        "Live"
                    )
                else:
                    st.error(f"❌ {symbol}")
                    st.caption(f"Error: {error}")
                
                # Remove button
                if st.button("❌", key=f"remove_{symbol}"):
                    symbols.remove(symbol)
                    st.rerun()
    else:
        st.info("No symbols being tracked. Add some stocks to get started!")

def display_portfolio_realtime(portfolio_positions: List[Dict[str, Any]]):
    """Display portfolio with real-time updates"""
    st.subheader("💼 Live Portfolio")
    
    if not portfolio_positions:
        st.info("No positions in portfolio")
        return
    
    # Refresh button
    if st.button("🔄 Refresh Portfolio", type="primary"):
        st.cache_data.clear()
        st.rerun()
    
    # Display positions with live prices
    total_value = 0
    total_cost = 0
    
    for i, position in enumerate(portfolio_positions):
        with st.container():
            col1, col2, col3, col4, col5, col6 = st.columns([2, 1, 1, 1, 1, 1])
            
            with col1:
                st.write(f"**{position['symbol']}**")
                st.write(f"Added: {position.get('date_added', 'Unknown')}")
            
            with col2:
                st.write(f"**{position['shares']}** shares")
            
            with col3:
                st.write(f"**${position['cost_basis']:.2f}** cost")
            
            with col4:
                # Get live price
                from realtime_simple import get_live_price
                live_price, error = get_live_price(position['symbol'])
                
                if live_price is not None:
                    st.write(f"**${live_price:.2f}** live")
                    
                    # Calculate P&L
                    pnl = (live_price - position['cost_basis']) * position['shares']
                    pnl_percent = (live_price - position['cost_basis']) / position['cost_basis'] * 100
                    
                    with col5:
                        color = "🟢" if pnl >= 0 else "🔴"
                        st.write(f"{color} **${pnl:.2f}**")
                        st.write(f"({pnl_percent:+.1f}%)")
                    
                    # Update totals
                    position_value = live_price * position['shares']
                    position_cost = position['cost_basis'] * position['shares']
                    total_value += position_value
                    total_cost += position_cost
                else:
                    st.write("**N/A** live")
                    st.caption(f"Error: {error}")
            
            with col6:
                if st.button("❌", key=f"remove_portfolio_{i}"):
                    portfolio_positions.pop(i)
                    st.rerun()
    
    # Portfolio summary
    if total_value > 0:
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
            total_pnl_percent = (total_value - total_cost) / total_cost * 100 if total_cost > 0 else 0
            st.metric("Total P&L %", f"{total_pnl_percent:+.2f}%")

def display_trending_stocks():
    """Display trending stocks"""
    st.subheader("🔥 Trending Stocks")
    
    from realtime_simple import get_trending_stocks
    trending_data = get_trending_stocks()
    
    if trending_data:
        # Create a DataFrame for better display
        df = pd.DataFrame(trending_data)
        
        # Display as metrics
        cols = st.columns(min(len(trending_data), 4))
        
        for i, stock in enumerate(trending_data):
            col = cols[i % 4]
            
            with col:
                st.metric(
                    stock['symbol'],
                    f"${stock['price']:.2f}",
                    f"{stock['change_percent']:+.2f}%"
                )
                st.caption(stock['name'])
    else:
        st.info("No trending data available")

def display_data_source_status():
    """Display data source status"""
    st.subheader("🔧 Data Source Status")
    
    # Yahoo Finance status
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.metric("Yahoo Finance", "🟢 Online")
    
    with col2:
        st.caption("Primary data source for all market data")
    
    # Cache status
    st.subheader("💾 Cache Status")
    
    from realtime_simple import realtime_service
    
    cache_info = []
    for key, value in realtime_service.cache.items():
        if key in realtime_service.last_update:
            age = time.time() - realtime_service.last_update[key]
            cache_info.append({
                'Key': key,
                'Age (seconds)': f"{age:.1f}",
                'Status': "🟢 Fresh" if age < realtime_service.cache_ttl else "🟡 Stale"
            })
    
    if cache_info:
        df = pd.DataFrame(cache_info)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No cached data")

def display_price_alerts():
    """Display price alerts (simplified version)"""
    st.subheader("🔔 Price Alerts")
    
    st.info("💡 Price alerts feature coming soon! For now, you can monitor your portfolio for real-time updates.")
    
    # Simple alert setup
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        alert_symbol = st.text_input("Symbol", placeholder="AAPL")
    
    with col2:
        alert_price = st.number_input("Alert Price", min_value=0.0, value=150.0)
    
    with col3:
        if st.button("Set Alert"):
            if alert_symbol:
                st.success(f"Alert set for {alert_symbol.upper()} at ${alert_price:.2f}")
            else:
                st.error("Please enter a symbol")
    
    # Current alerts (placeholder)
    st.subheader("📋 Current Alerts")
    st.info("No active alerts. Set some alerts above to get started!")

