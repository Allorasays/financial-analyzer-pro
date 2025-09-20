#!/usr/bin/env python3
"""
Enhanced Financial Analyzer Pro - Stable Version for Render
Includes portfolio management, technical analysis, and ML predictions
with robust error handling and graceful fallbacks
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import warnings
import json
import io
warnings.filterwarnings('ignore')

# ML imports with graceful fallbacks
try:
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="Financial Analyzer Pro - Enhanced",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS
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
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .portfolio-item {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #007bff;
    }
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = []
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']

def get_stock_data(symbol, period="1y"):
    """Get stock data from Yahoo Finance with error handling"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        if data.empty:
            return None, f"No data available for {symbol}"
        return data, None
    except Exception as e:
        return None, f"Error fetching data for {symbol}: {str(e)}"

def calculate_technical_indicators(data):
    """Calculate comprehensive technical indicators"""
    if data is None or len(data) == 0:
        return data
    
    # Simple Moving Averages
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    
    # Exponential Moving Averages
    data['EMA_12'] = data['Close'].ewm(span=12).mean()
    data['EMA_26'] = data['Close'].ewm(span=26).mean()
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
    data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
    
    # Bollinger Bands
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    bb_std = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
    data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
    
    # Stochastic Oscillator
    low_min = data['Low'].rolling(window=14).min()
    high_max = data['High'].rolling(window=14).max()
    data['Stoch_K'] = 100 * ((data['Close'] - low_min) / (high_max - low_min))
    data['Stoch_D'] = data['Stoch_K'].rolling(window=3).mean()
    
    # Volume indicators
    data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
    data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
    
    # Price momentum
    data['Price_Change'] = data['Close'].pct_change()
    data['Price_Range'] = (data['High'] - data['Low']) / data['Close']
    
    return data

def calculate_financial_ratios(data, symbol):
    """Calculate financial ratios and metrics"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        ratios = {}
        
        # Price ratios
        if 'trailingPE' in info and info['trailingPE']:
            ratios['P/E Ratio'] = round(info['trailingPE'], 2)
        if 'priceToBook' in info and info['priceToBook']:
            ratios['P/B Ratio'] = round(info['priceToBook'], 2)
        if 'priceToSalesTrailing12Months' in info and info['priceToSalesTrailing12Months']:
            ratios['P/S Ratio'] = round(info['priceToSalesTrailing12Months'], 2)
        
        # Profitability ratios
        if 'returnOnEquity' in info and info['returnOnEquity']:
            ratios['ROE'] = f"{info['returnOnEquity'] * 100:.2f}%"
        if 'returnOnAssets' in info and info['returnOnAssets']:
            ratios['ROA'] = f"{info['returnOnAssets'] * 100:.2f}%"
        
        # Margin ratios
        if 'grossMargins' in info and info['grossMargins']:
            ratios['Gross Margin'] = f"{info['grossMargins'] * 100:.2f}%"
        if 'operatingMargins' in info and info['operatingMargins']:
            ratios['Operating Margin'] = f"{info['operatingMargins'] * 100:.2f}%"
        if 'profitMargins' in info and info['profitMargins']:
            ratios['Net Margin'] = f"{info['profitMargins'] * 100:.2f}%"
        
        # Growth ratios
        if 'revenueGrowth' in info and info['revenueGrowth']:
            ratios['Revenue Growth'] = f"{info['revenueGrowth'] * 100:.2f}%"
        if 'earningsGrowth' in info and info['earningsGrowth']:
            ratios['Earnings Growth'] = f"{info['earningsGrowth'] * 100:.2f}%"
        
        # Debt ratios
        if 'debtToEquity' in info and info['debtToEquity']:
            ratios['Debt/Equity'] = round(info['debtToEquity'], 2)
        if 'currentRatio' in info and info['currentRatio']:
            ratios['Current Ratio'] = round(info['currentRatio'], 2)
        
        return ratios
    except Exception as e:
        st.warning(f"Could not fetch financial ratios: {str(e)}")
        return {}

def predict_price_ml(data, symbol, periods=5):
    """Predict future prices using machine learning"""
    if not SKLEARN_AVAILABLE:
        return None, "ML library not available"
    
    try:
        # More robust feature selection
        basic_features = ['Close', 'Volume']
        
        # Add technical indicators if they exist and have data
        technical_features = []
        if 'RSI' in data.columns and not data['RSI'].isna().all():
            technical_features.append('RSI')
        if 'SMA_20' in data.columns and not data['SMA_20'].isna().all():
            technical_features.append('SMA_20')
        if 'SMA_50' in data.columns and not data['SMA_50'].isna().all():
            technical_features.append('SMA_50')
        if 'MACD' in data.columns and not data['MACD'].isna().all():
            technical_features.append('MACD')
        if 'MACD_Signal' in data.columns and not data['MACD_Signal'].isna().all():
            technical_features.append('MACD_Signal')
        
        # Add price-based features
        data['Price_Change'] = data['Close'].pct_change()
        data['Price_Range'] = (data['High'] - data['Low']) / data['Close']
        data['Volume_Change'] = data['Volume'].pct_change()
        
        price_features = ['Price_Change', 'Price_Range', 'Volume_Change']
        
        # Combine all available features
        all_features = basic_features + technical_features + price_features
        
        # Filter features that exist in the data and have sufficient non-NaN values
        available_features = []
        for feature in all_features:
            if feature in data.columns:
                non_nan_count = data[feature].notna().sum()
                if non_nan_count >= 10:  # Require at least 10 non-NaN values
                    available_features.append(feature)
        
        if len(available_features) < 2:
            return None, f"Insufficient features for prediction (need ≥2, got {len(available_features)})"
        
        # Create lagged features
        df_ml = data[available_features].dropna()
        if len(df_ml) < 10:  # Reduced from 30 to 10
            return None, f"Insufficient data for prediction (need ≥10, got {len(df_ml)})"
        
        # Create target variable (future price)
        df_ml['Target'] = df_ml['Close'].shift(-periods)
        df_ml = df_ml.dropna()
        
        if len(df_ml) < 5:  # Reduced from 20 to 5
            return None, f"Insufficient data after creating target (need ≥5, got {len(df_ml)})"
        
        # Ensure we have valid features
        feature_cols = [col for col in available_features if col != 'Close' and col in df_ml.columns]
        if len(feature_cols) < 1:
            return None, "No valid features for prediction"
        
        # Prepare features and target
        X = df_ml[feature_cols]
        y = df_ml['Target']
        
        # Split data
        split_idx = max(1, int(len(df_ml) * 0.8))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Predict future prices
        last_features = X.iloc[-1:].values
        future_prices = []
        current_price = data['Close'].iloc[-1]
        
        for i in range(periods):
            pred_price = model.predict(last_features)[0]
            future_prices.append(pred_price)
            # Update features for next prediction (simplified)
            if len(last_features[0]) > 0:
                last_features[0][0] = pred_price  # Update price feature
        
        # Create prediction dates
        last_date = data.index[-1]
        prediction_dates = [last_date + timedelta(days=i+1) for i in range(periods)]
        
        return {
            'predictions': future_prices,
            'dates': prediction_dates,
            'current_price': current_price,
            'accuracy': r2,
            'mse': mse,
            'model_type': 'Linear Regression',
            'features_used': len(feature_cols),
            'data_points': len(df_ml)
        }, None
        
    except Exception as e:
        return None, f"Prediction error: {str(e)}"

def get_market_overview():
    """Get market overview for major indices"""
    try:
        indices = {
            '^GSPC': 'S&P 500',
            '^IXIC': 'NASDAQ',
            '^DJI': 'DOW',
            '^VIX': 'VIX'
        }
        
        market_data = {}
        for symbol, name in indices.items():
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1d")
                if not data.empty:
                    current_price = data['Close'].iloc[-1]
                    previous_price = data['Open'].iloc[-1]
                    change = current_price - previous_price
                    change_percent = (change / previous_price) * 100
                    
                    market_data[symbol] = {
                        'name': name,
                        'price': current_price,
                        'change': change,
                        'change_percent': change_percent
                    }
            except Exception as e:
                st.warning(f"Could not fetch {name}: {str(e)}")
                continue
        
        return market_data
    except Exception as e:
        st.error(f"Error fetching market overview: {str(e)}")
        return {}

def display_portfolio():
    """Display portfolio management interface"""
    st.subheader("💼 Portfolio Management")
    
    # Add new position
    with st.expander("➕ Add New Position", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            symbol = st.text_input("Symbol", value="AAPL", key="add_symbol")
        with col2:
            shares = st.number_input("Shares", min_value=0.0, value=10.0, key="add_shares")
        with col3:
            cost_basis = st.number_input("Cost per Share", min_value=0.0, value=150.0, key="add_cost")
        with col4:
            if st.button("Add Position", type="primary"):
                if symbol and shares > 0 and cost_basis > 0:
                    # Get current price
                    data, error = get_stock_data(symbol, "1d")
                    if data is not None and not data.empty:
                        current_price = data['Close'].iloc[-1]
                        
                        position = {
                            'symbol': symbol.upper(),
                            'shares': shares,
                            'cost_basis': cost_basis,
                            'current_price': current_price,
                            'date_added': datetime.now().strftime("%Y-%m-%d")
                        }
                        
                        st.session_state.portfolio.append(position)
                        st.success(f"✅ Added {shares} shares of {symbol.upper()} at ${cost_basis:.2f}")
                        st.rerun()
                    else:
                        st.error(f"❌ Could not fetch current price for {symbol}")
                else:
                    st.error("❌ Please fill in all fields")
    
    # Display portfolio
    if st.session_state.portfolio:
        st.subheader("📊 Current Portfolio")
        
        total_value = 0
        total_cost = 0
        
        for i, position in enumerate(st.session_state.portfolio):
            with st.container():
                col1, col2, col3, col4, col5, col6 = st.columns([2, 1, 1, 1, 1, 1])
                
                with col1:
                    st.write(f"**{position['symbol']}**")
                    st.write(f"Added: {position['date_added']}")
                
                with col2:
                    st.write(f"**{position['shares']}** shares")
                
                with col3:
                    st.write(f"**${position['cost_basis']:.2f}** cost")
                
                with col4:
                    st.write(f"**${position['current_price']:.2f}** current")
                
                with col5:
                    pnl = (position['current_price'] - position['cost_basis']) * position['shares']
                    pnl_percent = (position['current_price'] - position['cost_basis']) / position['cost_basis'] * 100
                    color = "🟢" if pnl >= 0 else "🔴"
                    st.write(f"{color} **${pnl:.2f}**")
                    st.write(f"({pnl_percent:+.1f}%)")
                
                with col6:
                    if st.button("❌", key=f"remove_{i}"):
                        st.session_state.portfolio.pop(i)
                        st.rerun()
                
                # Calculate totals
                position_value = position['current_price'] * position['shares']
                position_cost = position['cost_basis'] * position['shares']
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
            total_pnl_percent = (total_value - total_cost) / total_cost * 100 if total_cost > 0 else 0
            st.metric("Total P&L %", f"{total_pnl_percent:+.2f}%")
    else:
        st.info("📝 No positions in portfolio. Add some stocks to get started!")

def main():
    """Main application"""
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("📊 Financial Analyzer Pro - Enhanced")
    st.markdown("**Advanced Financial Analysis & Portfolio Management**")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("🔧 Analysis Settings")
    
    # Analysis type selection
    analysis_type = st.sidebar.selectbox(
        "Analysis Type",
        ["📈 Stock Analysis", "💼 Portfolio Management", "📊 Market Overview", "🤖 ML Predictions"],
        index=0
    )
    
    if analysis_type == "📈 Stock Analysis":
        # Stock symbol input
        symbol = st.sidebar.text_input("Stock Symbol", value="AAPL", help="Enter a stock symbol (e.g., AAPL, MSFT, GOOGL)")
        period = st.sidebar.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
        
        # Analysis button
        if st.sidebar.button("🚀 Analyze Stock", type="primary"):
            with st.spinner("Fetching data and analyzing..."):
                # Get data
                data, error = get_stock_data(symbol, period)
                
                if error:
                    st.error(f"❌ {error}")
                    return
                
                if data is None or len(data) == 0:
                    st.error("❌ No data available for this symbol")
                    return
                
                # Calculate indicators
                data = calculate_technical_indicators(data)
                
                # Display current price
                current_price = data['Close'].iloc[-1]
                previous_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
                change = current_price - previous_price
                change_percent = (change / previous_price) * 100
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Current Price", f"${current_price:.2f}")
                with col2:
                    st.metric("Change", f"${change:.2f}")
                with col3:
                    st.metric("Change %", f"{change_percent:.2f}%")
                with col4:
                    rsi = data['RSI'].iloc[-1] if 'RSI' in data.columns else 0
                    st.metric("RSI", f"{rsi:.1f}")
                
                # Price chart
                st.subheader("📈 Price Chart")
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='#1f77b4', width=2)
                ))
                
                if 'SMA_20' in data.columns:
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data['SMA_20'],
                        mode='lines',
                        name='SMA 20',
                        line=dict(color='orange', width=1)
                    ))
                
                if 'SMA_50' in data.columns:
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data['SMA_50'],
                        mode='lines',
                        name='SMA 50',
                        line=dict(color='red', width=1)
                    ))
                
                # Bollinger Bands
                if 'BB_Upper' in data.columns and 'BB_Lower' in data.columns:
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data['BB_Upper'],
                        mode='lines',
                        name='BB Upper',
                        line=dict(color='gray', width=1, dash='dash'),
                        showlegend=False
                    ))
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data['BB_Lower'],
                        mode='lines',
                        name='BB Lower',
                        line=dict(color='gray', width=1, dash='dash'),
                        fill='tonexty',
                        fillcolor='rgba(128,128,128,0.1)',
                        showlegend=False
                    ))
                
                fig.update_layout(
                    title=f"{symbol} Stock Price",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Technical indicators
                col1, col2 = st.columns(2)
                
                with col1:
                    # RSI chart
                    if 'RSI' in data.columns:
                        st.subheader("📊 RSI Indicator")
                        rsi_fig = go.Figure()
                        
                        rsi_fig.add_trace(go.Scatter(
                            x=data.index,
                            y=data['RSI'],
                            mode='lines',
                            name='RSI',
                            line=dict(color='purple', width=2)
                        ))
                        
                        # Add RSI levels
                        rsi_fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
                        rsi_fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
                        
                        rsi_fig.update_layout(
                            title="RSI (Relative Strength Index)",
                            xaxis_title="Date",
                            yaxis_title="RSI",
                            yaxis=dict(range=[0, 100]),
                            height=300
                        )
                        
                        st.plotly_chart(rsi_fig, use_container_width=True)
                
                with col2:
                    # MACD chart
                    if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
                        st.subheader("📊 MACD Indicator")
                        macd_fig = go.Figure()
                        
                        macd_fig.add_trace(go.Scatter(
                            x=data.index,
                            y=data['MACD'],
                            mode='lines',
                            name='MACD',
                            line=dict(color='blue', width=2)
                        ))
                        
                        macd_fig.add_trace(go.Scatter(
                            x=data.index,
                            y=data['MACD_Signal'],
                            mode='lines',
                            name='Signal',
                            line=dict(color='red', width=2)
                        ))
                        
                        macd_fig.add_trace(go.Bar(
                            x=data.index,
                            y=data['MACD_Histogram'],
                            name='Histogram',
                            marker_color='gray'
                        ))
                        
                        macd_fig.update_layout(
                            title="MACD (Moving Average Convergence Divergence)",
                            xaxis_title="Date",
                            yaxis_title="MACD",
                            height=300
                        )
                        
                        st.plotly_chart(macd_fig, use_container_width=True)
                
                # Financial ratios
                st.subheader("📋 Financial Ratios")
                ratios = calculate_financial_ratios(data, symbol)
                
                if ratios:
                    col1, col2, col3, col4 = st.columns(4)
                    ratio_items = list(ratios.items())
                    
                    for i, (key, value) in enumerate(ratio_items):
                        col = [col1, col2, col3, col4][i % 4]
                        with col:
                            st.metric(key, value)
                else:
                    st.info("Financial ratios not available for this symbol")
                
                # Data summary
                st.subheader("📊 Data Summary")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Symbol:** {symbol}")
                    st.write(f"**Period:** {period}")
                    st.write(f"**Data Points:** {len(data)}")
                    st.write(f"**Date Range:** {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
                
                with col2:
                    st.write(f"**High:** ${data['High'].max():.2f}")
                    st.write(f"**Low:** ${data['Low'].min():.2f}")
                    st.write(f"**Volume (Avg):** {data['Volume'].mean():,.0f}")
                    st.write(f"**Volatility:** {data['Close'].pct_change().std() * 100:.2f}%")
    
    elif analysis_type == "💼 Portfolio Management":
        display_portfolio()
    
    elif analysis_type == "📊 Market Overview":
        st.subheader("📊 Market Overview")
        
        if st.button("🔄 Refresh Market Data", type="primary"):
            with st.spinner("Fetching market data..."):
                market_data = get_market_overview()
                
                if market_data:
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
                else:
                    st.error("Could not fetch market data")
    
    elif analysis_type == "🤖 ML Predictions":
        st.subheader("🤖 Machine Learning Predictions")
        
        if not SKLEARN_AVAILABLE:
            st.error("❌ Machine learning libraries not available")
            st.info("💡 ML features require scikit-learn. The app will work with basic features.")
        else:
            col1, col2 = st.columns([1, 3])
            with col1:
                symbol = st.text_input("Stock Symbol", value="AAPL", key="ml_symbol")
            with col2:
                period = st.selectbox("Time Period", ["6mo", "1y", "2y", "5y"], index=1, key="ml_period")
            
            if st.button("🚀 Run ML Analysis", type="primary"):
                with st.spinner("Running machine learning analysis..."):
                    data, error = get_stock_data(symbol, period)
                    
                    if error:
                        st.error(f"❌ {error}")
                    else:
                        st.success(f"✅ ML analysis complete for {symbol}")
                        
                        # Calculate indicators
                        data = calculate_technical_indicators(data)
                        
                        # Run ML prediction
                        prediction_result, pred_error = predict_price_ml(data, symbol, periods=5)
                        
                        if prediction_result:
                            st.subheader("📈 Price Predictions")
                            
                            # Display prediction metrics
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Model Type", prediction_result['model_type'])
                            with col2:
                                st.metric("Accuracy (R²)", f"{prediction_result['accuracy']:.3f}")
                            with col3:
                                st.metric("Features Used", prediction_result['features_used'])
                            with col4:
                                st.metric("Data Points", prediction_result['data_points'])
                            
                            # Create prediction chart
                            pred_fig = go.Figure()
                            
                            # Historical data
                            pred_fig.add_trace(go.Scatter(
                                x=data.index[-30:],  # Last 30 days
                                y=data['Close'].iloc[-30:],
                                mode='lines',
                                name='Historical Price',
                                line=dict(color='blue', width=2)
                            ))
                            
                            # Predictions
                            pred_fig.add_trace(go.Scatter(
                                x=prediction_result['dates'],
                                y=prediction_result['predictions'],
                                mode='lines+markers',
                                name='Predicted Price',
                                line=dict(color='red', width=2, dash='dash'),
                                marker=dict(size=8)
                            ))
                            
                            # Current price line
                            pred_fig.add_hline(
                                y=prediction_result['current_price'],
                                line_dash="dot",
                                line_color="green",
                                annotation_text=f"Current: ${prediction_result['current_price']:.2f}"
                            )
                            
                            pred_fig.update_layout(
                                title=f"{symbol} Price Predictions (Next 5 Days)",
                                xaxis_title="Date",
                                yaxis_title="Price ($)",
                                hovermode='x unified',
                                height=500
                            )
                            
                            st.plotly_chart(pred_fig, use_container_width=True)
                            
                            # Prediction details
                            st.subheader("📋 Prediction Details")
                            for i, (date, price) in enumerate(zip(prediction_result['dates'], prediction_result['predictions'])):
                                change = price - prediction_result['current_price']
                                change_percent = (change / prediction_result['current_price']) * 100
                                st.write(f"**Day {i+1}** ({date.strftime('%Y-%m-%d')}): ${price:.2f} ({change_percent:+.2f}%)")
                        else:
                            st.error(f"❌ {pred_error}")
    
    # Footer
    st.markdown("---")
    st.markdown("**Financial Analyzer Pro - Enhanced** | Built with Streamlit")
    st.markdown("*Advanced financial analysis with portfolio management and ML predictions*")

if __name__ == "__main__":
    main()


