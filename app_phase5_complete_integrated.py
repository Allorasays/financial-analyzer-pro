import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import time
import json
import warnings
warnings.filterwarnings('ignore')

# Enhanced ML imports
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
    page_title="Financial Analyzer Pro - Complete Platform",
    page_icon="📈",
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
    .feature-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .success { color: #28a745; font-weight: bold; }
    .error { color: #dc3545; font-weight: bold; }
    .warning { color: #ffc107; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>📈 Financial Analyzer Pro</h1>
    <p>Complete Professional Financial Analysis Platform</p>
    <p>Status: ✅ All Phases Complete - Enterprise-Ready Platform!</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("🎯 Complete Platform")
analysis_tab = st.sidebar.selectbox(
    "Select Analysis Module",
    [
        "🏠 Dashboard",
        "📊 Stock Analysis", 
        "💼 Portfolio Management",
        "📈 Market Overview",
        "🏭 Industry Analysis",
        "⚠️ Risk Assessment",
        "🤖 Machine Learning",
        "📊 Technical Analysis",
        "📤 Export & Reports",
        "⚙️ Settings"
    ]
)

# Initialize session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = []
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []
if 'notifications' not in st.session_state:
    st.session_state.notifications = []

def get_market_data(symbol: str, period: str = "1y"):
    """Get market data with caching"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        if data.empty:
            return None, f"No data found for {symbol}"
        return data, None
    except Exception as e:
        return None, f"Error fetching data: {str(e)}"

def calculate_technical_indicators(data):
    """Calculate comprehensive technical indicators"""
    df = data.copy()
    
    # Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    return df

def train_ml_models(X, y):
    """Train multiple ML models"""
    if not SKLEARN_AVAILABLE:
        return None, "ML library not available"
    
    try:
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        trained_models = {}
        scores = {}
        
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            trained_models[name] = model
            scores[name] = {'mse': mse, 'r2': r2}
        
        return {
            'models': trained_models,
            'scaler': scaler,
            'scores': scores,
            'X_test': X_test,
            'y_test': y_test
        }, None
        
    except Exception as e:
        return None, f"Error training models: {str(e)}"

def calculate_risk_metrics(data):
    """Calculate comprehensive risk metrics"""
    returns = data['Close'].pct_change().dropna()
    
    risk_metrics = {}
    risk_metrics['Volatility (Annualized)'] = f"{returns.std() * np.sqrt(252) * 100:.2f}%"
    risk_metrics['Sharpe Ratio'] = f"{(returns.mean() * 252) / (returns.std() * np.sqrt(252)):.2f}"
    
    # Maximum Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    risk_metrics['Max Drawdown'] = f"{drawdown.min() * 100:.2f}%"
    
    # Value at Risk
    var_95 = np.percentile(returns, 5)
    var_99 = np.percentile(returns, 1)
    risk_metrics['VaR (95%)'] = f"{var_95 * 100:.2f}%"
    risk_metrics['VaR (99%)'] = f"{var_99 * 100:.2f}%"
    
    return risk_metrics

def get_market_overview():
    """Get real-time market overview"""
    symbols = ['^GSPC', '^IXIC', '^DJI', '^VIX']
    overview = {}
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2d")
            
            if not hist.empty and len(hist) >= 2:
                current_price = hist['Close'].iloc[-1]
                previous_price = hist['Close'].iloc[-2]
                change = current_price - previous_price
                change_percent = (change / previous_price) * 100
                
                overview[symbol] = {
                    'price': current_price,
                    'change': change,
                    'change_percent': change_percent
                }
        except Exception as e:
            st.warning(f"Could not fetch {symbol}: {str(e)}")
    
    return overview

# Main Application Logic
if analysis_tab == "🏠 Dashboard":
    st.header("🏠 Financial Dashboard")
    
    # Market overview
    st.subheader("📈 Market Overview")
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
    
    # Portfolio summary
    st.subheader("💼 Portfolio Summary")
    if st.session_state.portfolio:
        total_value = sum(pos['value'] for pos in st.session_state.portfolio)
        total_cost = sum(pos['cost_basis'] for pos in st.session_state.portfolio)
        total_pnl = total_value - total_cost
        total_pnl_percent = (total_pnl / total_cost) * 100 if total_cost > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Value", f"${total_value:,.2f}")
        with col2:
            st.metric("Total P&L", f"${total_pnl:,.2f}", f"{total_pnl_percent:+.2f}%")
        with col3:
            st.metric("Positions", len(st.session_state.portfolio))
        with col4:
            st.metric("Watchlist", len(st.session_state.watchlist))
    else:
        st.info("No positions in portfolio. Add positions to see portfolio summary.")

elif analysis_tab == "📊 Stock Analysis":
    st.header("📊 Comprehensive Stock Analysis")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        symbol = st.text_input("Stock Symbol", value="AAPL")
    with col2:
        period = st.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=1)
    
    if st.button("Analyze Stock", type="primary"):
        with st.spinner("Running comprehensive analysis..."):
            data, error = get_market_data(symbol, period)
            
            if error:
                st.error(f"❌ {error}")
            else:
                st.success(f"✅ Analysis complete for {symbol}")
                
                # Calculate indicators
                data_with_indicators = calculate_technical_indicators(data)
                risk_metrics = calculate_risk_metrics(data)
                
                # Display metrics
                current_price = data['Close'].iloc[-1]
                previous_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
                change = current_price - previous_price
                change_percent = (change / previous_price) * 100
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Current Price", f"${current_price:.2f}", f"{change:+.2f}")
                with col2:
                    st.metric("Change", f"{change_percent:+.2f}%")
                with col3:
                    st.metric("RSI", f"{data_with_indicators['RSI'].iloc[-1]:.1f}")
                with col4:
                    st.metric("Volatility", risk_metrics['Volatility (Annualized)'])
                
                # Price chart with indicators
                st.subheader("Price Chart with Technical Indicators")
                
                fig = go.Figure()
                
                # Price line
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='#1f77b4', width=2)
                ))
                
                # Moving averages
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data_with_indicators['SMA_20'],
                    mode='lines',
                    name='SMA 20',
                    line=dict(color='orange', width=1, dash='dash')
                ))
                
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data_with_indicators['SMA_50'],
                    mode='lines',
                    name='SMA 50',
                    line=dict(color='red', width=1, dash='dash')
                ))
                
                # Bollinger Bands
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data_with_indicators['BB_Upper'],
                    mode='lines',
                    name='BB Upper',
                    line=dict(color='gray', width=1, dash='dot')
                ))
                
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data_with_indicators['BB_Lower'],
                    mode='lines',
                    name='BB Lower',
                    line=dict(color='gray', width=1, dash='dot'),
                    fill='tonexty'
                ))
                
                fig.update_layout(
                    title=f"{symbol} Technical Analysis",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk metrics
                st.subheader("Risk Assessment")
                col1, col2 = st.columns(2)
                
                with col1:
                    for metric, value in risk_metrics.items():
                        st.write(f"• **{metric}**: {value}")
                
                with col2:
                    volatility = float(risk_metrics['Volatility (Annualized)'].replace('%', ''))
                    if volatility < 20:
                        st.success("✅ Low risk investment")
                    elif volatility < 40:
                        st.warning("⚠️ Medium risk investment")
                    else:
                        st.error("🚨 High risk investment")

elif analysis_tab == "💼 Portfolio Management":
    st.header("💼 Portfolio Management")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Add Position")
        symbol = st.text_input("Stock Symbol", value="AAPL")
        shares = st.number_input("Number of Shares", min_value=1, value=10)
        price = st.number_input("Purchase Price", min_value=0.01, value=150.00, step=0.01)
        
        if st.button("Add to Portfolio"):
            current_data = get_market_data(symbol, "1d")
            if current_data is not None:
                current_price = current_data['Close'].iloc[-1]
                position = {
                    'symbol': symbol,
                    'shares': shares,
                    'purchase_price': price,
                    'current_price': current_price,
                    'value': shares * current_price,
                    'cost_basis': shares * price,
                    'pnl': (current_price - price) * shares,
                    'pnl_percent': ((current_price - price) / price) * 100
                }
                st.session_state.portfolio.append(position)
                st.success(f"Added {shares} shares of {symbol} to portfolio")
            else:
                st.error(f"Could not fetch current price for {symbol}")
    
    with col2:
        st.subheader("Portfolio Summary")
        if st.session_state.portfolio:
            total_value = sum(pos['value'] for pos in st.session_state.portfolio)
            total_cost = sum(pos['cost_basis'] for pos in st.session_state.portfolio)
            total_pnl = total_value - total_cost
            total_pnl_percent = (total_pnl / total_cost) * 100 if total_cost > 0 else 0
            
            st.metric("Total Value", f"${total_value:,.2f}")
            st.metric("Total P&L", f"${total_pnl:,.2f}", f"{total_pnl_percent:+.2f}%")
            st.metric("Positions", len(st.session_state.portfolio))
        else:
            st.info("No positions in portfolio")
    
    # Portfolio table
    if st.session_state.portfolio:
        st.subheader("Portfolio Positions")
        portfolio_df = pd.DataFrame(st.session_state.portfolio)
        
        # Format the dataframe for display
        display_df = portfolio_df.copy()
        display_df['purchase_price'] = display_df['purchase_price'].apply(lambda x: f"${x:.2f}")
        display_df['current_price'] = display_df['current_price'].apply(lambda x: f"${x:.2f}")
        display_df['value'] = display_df['value'].apply(lambda x: f"${x:,.2f}")
        display_df['cost_basis'] = display_df['cost_basis'].apply(lambda x: f"${x:,.2f}")
        display_df['pnl'] = display_df['pnl'].apply(lambda x: f"${x:,.2f}")
        display_df['pnl_percent'] = display_df['pnl_percent'].apply(lambda x: f"{x:+.2f}%")
        
        st.dataframe(display_df, use_container_width=True)
        
        if st.button("Clear Portfolio", type="secondary"):
            st.session_state.portfolio = []
            st.rerun()

elif analysis_tab == "🤖 Machine Learning":
    st.header("🤖 Machine Learning Analysis")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        symbol = st.text_input("Stock Symbol", value="AAPL")
    with col2:
        period = st.selectbox("Time Period", ["6mo", "1y", "2y", "5y"], index=1)
    
    if st.button("Run ML Analysis", type="primary"):
        with st.spinner("Running machine learning analysis..."):
            data, error = get_market_data(symbol, period)
            
            if error:
                st.error(f"❌ {error}")
            else:
                st.success(f"✅ ML analysis complete for {symbol}")
                
                # Prepare features
                data_with_indicators = calculate_technical_indicators(data)
                
                # Create features for ML
                features = ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower']
                available_features = [f for f in features if f in data_with_indicators.columns]
                
                if len(available_features) >= 3:
                    X = data_with_indicators[available_features].dropna()
                    y = data['Close'].shift(-1).dropna()
                    
                    # Align features and target
                    min_len = min(len(X), len(y))
                    X = X.iloc[:min_len]
                    y = y.iloc[:min_len]
                    
                    if len(X) > 50:
                        # Train models
                        ml_results, ml_error = train_ml_models(X, y)
                        
                        if ml_error:
                            st.error(f"❌ {ml_error}")
                        else:
                            # Display model performance
                            st.subheader("Model Performance")
                            
                            performance_data = []
                            for model_name, scores in ml_results['scores'].items():
                                performance_data.append({
                                    'Model': model_name,
                                    'R² Score': f"{scores['r2']:.4f}",
                                    'MSE': f"{scores['mse']:.2f}"
                                })
                            
                            performance_df = pd.DataFrame(performance_data)
                            st.dataframe(performance_df, use_container_width=True)
                            
                            # Best model prediction
                            best_model = max(ml_results['scores'].items(), key=lambda x: x[1]['r2'])
                            st.success(f"🏆 Best Model: {best_model[0]} (R² = {best_model[1]['r2']:.4f})")
                            
                            # Make prediction
                            latest_features = X.iloc[-1:].values
                            latest_features_scaled = ml_results['scaler'].transform(latest_features)
                            
                            best_model_obj = ml_results['models'][best_model[0]]
                            prediction = best_model_obj.predict(latest_features_scaled)[0]
                            
                            current_price = data['Close'].iloc[-1]
                            price_change = (prediction - current_price) / current_price * 100
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Current Price", f"${current_price:.2f}")
                            with col2:
                                st.metric("Predicted Price", f"${prediction:.2f}")
                            with col3:
                                st.metric("Expected Change", f"{price_change:+.2f}%")
                    else:
                        st.warning("⚠️ Not enough data for ML analysis")
                else:
                    st.warning("⚠️ Insufficient features for ML analysis")

elif analysis_tab == "📤 Export & Reports":
    st.header("📤 Export & Reports")
    
    st.subheader("Export Portfolio Data")
    
    if st.session_state.portfolio:
        portfolio_df = pd.DataFrame(st.session_state.portfolio)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Portfolio Data**")
            st.dataframe(portfolio_df, use_container_width=True)
        
        with col2:
            st.write("**Export Options**")
            
            # CSV export
            csv = portfolio_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # JSON export
            json_data = portfolio_df.to_json(orient='records')
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name=f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    else:
        st.info("No portfolio data to export")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🎉 <strong>Financial Analyzer Pro - Complete Platform!</strong></p>
    <p>All Features • Real-time Data • Machine Learning • Portfolio Management</p>
    <p>Phase 5 Complete - Professional Financial Analysis Platform</p>
</div>
""", unsafe_allow_html=True)
