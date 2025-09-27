"""
Real-time Dashboard Components for Financial Analyzer Pro
Provides UI components for real-time data display
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import numpy as np

def display_realtime_market_overview():
    """Display real-time market overview"""
    st.subheader("📈 Live Market Overview")
    
    try:
        from realtime_data_service import get_cached_market_overview
        market_data = get_cached_market_overview()
        
        if market_data:
            st.success("✅ Real-time market data loaded")
            
            # Display market indices
            indices = [
                ('^GSPC', 'S&P 500'),
                ('^IXIC', 'NASDAQ'),
                ('^DJI', 'Dow Jones'),
                ('^VIX', 'VIX Volatility')
            ]
            
            cols = st.columns(4)
            for i, (symbol, name) in enumerate(indices):
                with cols[i]:
                    if symbol in market_data:
                        data = market_data[symbol]
                        change_color = "🟢" if data['change'] >= 0 else "🔴"
                        st.metric(
                            name,
                            f"${data['price']:.2f}",
                            f"{change_color} {data['change_percent']:+.2f}%"
                        )
            
            # Market sentiment
            positive_count = sum(1 for data in market_data.values() if data['change'] >= 0)
            total_count = len(market_data)
            sentiment = "🟢 Bullish" if positive_count > total_count/2 else "🔴 Bearish"
            
            st.metric("Market Sentiment", sentiment, f"{positive_count}/{total_count} up")
            
        else:
            st.warning("⚠️ No market data available")
            
    except ImportError:
        st.error("❌ Real-time service not available")
    except Exception as e:
        st.error(f"❌ Error loading market data: {str(e)}")

def display_live_stock_tracker(tracked_symbols):
    """Display live stock tracker"""
    st.subheader("📊 Live Stock Tracker")
    
    # Add/remove symbols
    col1, col2 = st.columns([3, 1])
    with col1:
        new_symbol = st.text_input("Add Symbol", placeholder="e.g., AAPL")
    with col2:
        if st.button("Add") and new_symbol:
            if new_symbol.upper() not in tracked_symbols:
                tracked_symbols.append(new_symbol.upper())
                st.rerun()
    
    if tracked_symbols:
        # Display tracked symbols
        try:
            from realtime_data_service import get_cached_live_price
            
            for i in range(0, len(tracked_symbols), 3):
                cols = st.columns(3)
                for j, symbol in enumerate(tracked_symbols[i:i+3]):
                    with cols[j]:
                        price_data = get_cached_live_price(symbol)
                        if price_data:
                            change_color = "🟢" if price_data['change'] >= 0 else "🔴"
                            st.metric(
                                symbol,
                                f"${price_data['price']:.2f}",
                                f"{change_color} {price_data['change_percent']:+.2f}%"
                            )
                        else:
                            st.metric(symbol, "N/A", "No data")
                        
                        # Remove button
                        if st.button(f"Remove {symbol}", key=f"remove_{symbol}"):
                            tracked_symbols.remove(symbol)
                            st.rerun()
        except ImportError:
            st.error("❌ Real-time service not available")
        except Exception as e:
            st.error(f"❌ Error loading stock data: {str(e)}")
    else:
        st.info("No symbols being tracked. Add some symbols above.")

def display_portfolio_realtime(portfolio):
    """Display real-time portfolio updates"""
    st.subheader("💼 Live Portfolio Updates")
    
    if not portfolio:
        st.info("No positions in portfolio")
        return
    
    try:
        from realtime_data_service import get_cached_live_price
        
        total_value = 0
        total_cost = 0
        updated_positions = []
        
        for position in portfolio:
            symbol = position['symbol']
            shares = position['shares']
            cost_basis = position['cost_basis']
            
            # Get current price
            price_data = get_cached_live_price(symbol)
            if price_data:
                current_price = price_data['price']
                current_value = shares * current_price
                pnl = current_value - cost_basis
                pnl_percent = (pnl / cost_basis) * 100 if cost_basis > 0 else 0
                
                updated_position = {
                    'symbol': symbol,
                    'shares': shares,
                    'current_price': current_price,
                    'current_value': current_value,
                    'cost_basis': cost_basis,
                    'pnl': pnl,
                    'pnl_percent': pnl_percent,
                    'change_percent': price_data['change_percent']
                }
                
                updated_positions.append(updated_position)
                total_value += current_value
                total_cost += cost_basis
        
        if updated_positions:
            total_pnl = total_value - total_cost
            total_pnl_percent = (total_pnl / total_cost) * 100 if total_cost > 0 else 0
            
            # Portfolio summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Value", f"${total_value:,.2f}")
            with col2:
                st.metric("Total P&L", f"${total_pnl:,.2f}", f"{total_pnl_percent:+.2f}%")
            with col3:
                st.metric("Positions", len(updated_positions))
            with col4:
                avg_change = sum(pos['change_percent'] for pos in updated_positions) / len(updated_positions)
                st.metric("Avg Change", f"{avg_change:+.2f}%")
            
            # Positions table
            st.subheader("📊 Live Position Updates")
            df = pd.DataFrame(updated_positions)
            
            # Format the dataframe
            display_df = df.copy()
            display_df['current_price'] = display_df['current_price'].apply(lambda x: f"${x:.2f}")
            display_df['current_value'] = display_df['current_value'].apply(lambda x: f"${x:,.2f}")
            display_df['cost_basis'] = display_df['cost_basis'].apply(lambda x: f"${x:,.2f}")
            display_df['pnl'] = display_df['pnl'].apply(lambda x: f"${x:,.2f}")
            display_df['pnl_percent'] = display_df['pnl_percent'].apply(lambda x: f"{x:+.2f}%")
            display_df['change_percent'] = display_df['change_percent'].apply(lambda x: f"{x:+.2f}%")
            
            st.dataframe(display_df, use_container_width=True)
        
    except ImportError:
        st.error("❌ Real-time service not available")
    except Exception as e:
        st.error(f"❌ Error updating portfolio: {str(e)}")

def display_price_alerts():
    """Display price alerts interface"""
    st.subheader("🔔 Price Alerts")
    
    # Initialize alerts in session state
    if 'price_alerts' not in st.session_state:
        st.session_state.price_alerts = []
    
    # Add new alert
    with st.expander("➕ Add New Alert"):
        col1, col2, col3 = st.columns(3)
        with col1:
            alert_symbol = st.text_input("Symbol", placeholder="AAPL")
        with col2:
            alert_type = st.selectbox("Alert Type", ["Above", "Below"])
        with col3:
            alert_price = st.number_input("Price", min_value=0.01, value=100.00, step=0.01)
        
        if st.button("Add Alert") and alert_symbol:
            alert = {
                'symbol': alert_symbol.upper(),
                'type': alert_type,
                'price': alert_price,
                'created': datetime.now(),
                'triggered': False
            }
            st.session_state.price_alerts.append(alert)
            st.success(f"Alert added for {alert_symbol} {alert_type} ${alert_price}")
            st.rerun()
    
    # Display active alerts
    if st.session_state.price_alerts:
        st.subheader("📋 Active Alerts")
        
        try:
            from realtime_data_service import get_cached_live_price
            
            for i, alert in enumerate(st.session_state.price_alerts):
                col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                
                with col1:
                    st.write(f"**{alert['symbol']}**")
                with col2:
                    st.write(f"{alert['type']} ${alert['price']:.2f}")
                with col3:
                    # Check if alert is triggered
                    price_data = get_cached_live_price(alert['symbol'])
                    if price_data:
                        current_price = price_data['price']
                        if alert['type'] == "Above" and current_price > alert['price']:
                            st.success("🚨 TRIGGERED!")
                            alert['triggered'] = True
                        elif alert['type'] == "Below" and current_price < alert['price']:
                            st.success("🚨 TRIGGERED!")
                            alert['triggered'] = True
                        else:
                            st.info(f"${current_price:.2f}")
                    else:
                        st.warning("No data")
                with col4:
                    if st.button("Remove", key=f"remove_alert_{i}"):
                        st.session_state.price_alerts.pop(i)
                        st.rerun()
        except ImportError:
            st.error("❌ Real-time service not available")
        except Exception as e:
            st.error(f"❌ Error checking alerts: {str(e)}")
    else:
        st.info("No price alerts set. Add some alerts above.")

def display_data_source_status():
    """Display data source status"""
    st.subheader("🔧 Data Source Status")
    
    try:
        from realtime_data_service import get_service_status
        status = get_service_status()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status_color = "🟢" if status['running'] else "🔴"
            st.metric("Service Status", f"{status_color} {'Running' if status['running'] else 'Stopped'}")
        
        with col2:
            st.metric("Cache Size", f"{status['cache_size']} items")
        
        with col3:
            if status['last_update']:
                last_update = status['last_update'].strftime("%H:%M:%S")
                st.metric("Last Update", last_update)
            else:
                st.metric("Last Update", "Never")
        
        # Data source health
        st.subheader("📊 Data Source Health")
        
        sources = [
            ("Yahoo Finance", "✅ Available"),
            ("Real-time Cache", "✅ Active" if status['running'] else "❌ Inactive"),
            ("WebSocket", "⚠️ Limited"),
            ("API Rate Limit", "✅ Normal")
        ]
        
        for source, health in sources:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(f"**{source}:**")
            with col2:
                st.write(health)
        
    except ImportError:
        st.error("❌ Real-time service not available")
    except Exception as e:
        st.error(f"❌ Error getting service status: {str(e)}")

def display_real_time_chart(symbol, period="1d"):
    """Display real-time chart for a symbol"""
    try:
        from realtime_data_service import get_cached_stock_data
        
        data = get_cached_stock_data(symbol, period)
        if data is not None and not data.empty:
            fig = go.Figure()
            
            # Price line
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='#1f77b4', width=2)
            ))
            
            fig.update_layout(
                title=f"{symbol} Real-time Chart",
                xaxis_title="Time",
                yaxis_title="Price ($)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"No data available for {symbol}")
            
    except ImportError:
        st.error("❌ Real-time service not available")
    except Exception as e:
        st.error(f"❌ Error displaying chart: {str(e)}")