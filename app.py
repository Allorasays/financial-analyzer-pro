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
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Phase 3 imports
from database_manager import DatabaseManager
from auth_system import AuthSystem
from portfolio_persistence import PortfolioPersistence

# Page config - moved to top to avoid issues
st.set_page_config(
    page_title="Financial Analyzer Pro - Research & Analysis",
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
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .positive {
        color: #28a745;
        font-weight: bold;
    }
    .negative {
        color: #dc3545;
        font-weight: bold;
    }
    .ml-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .prediction-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .risk-card {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .phase-indicator {
        background: #e8f5e8;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        border-left: 5px solid #28a745;
    }
    .ml-status {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 5px solid #ffc107;
    }
    .user-welcome {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
    .database-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize systems
db = DatabaseManager()
auth = AuthSystem()
portfolio_persistence = PortfolioPersistence()

def get_market_data(symbol: str, period: str = "1mo"):
    """Get market data using yfinance with enhanced error handling"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        if data.empty:
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def calculate_technical_indicators(data):
    """Calculate comprehensive technical indicators"""
    if data.empty:
        return data
    
    try:
        # Moving Averages
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['EMA_12'] = data['Close'].ewm(span=12).mean()
        data['EMA_26'] = data['Close'].ewm(span=26).mean()
        
        # RSI Calculation
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
        data['BB_Width'] = data['BB_Upper'] - data['BB_Lower']
        data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
        
        # Stochastic Oscillator
        low_14 = data['Low'].rolling(window=14).min()
        high_14 = data['High'].rolling(window=14).max()
        data['Stoch_K'] = 100 * ((data['Close'] - low_14) / (high_14 - low_14))
        data['Stoch_D'] = data['Stoch_K'].rolling(window=3).mean()
        
        # Volume indicators
        data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
        
        # Price momentum
        data['Momentum'] = data['Close'].pct_change(periods=10)
        data['Rate_of_Change'] = ((data['Close'] - data['Close'].shift(10)) / data['Close'].shift(10)) * 100
        
        # ML features
        data['Price_Change'] = data['Close'].pct_change()
        data['Volatility'] = data['Price_Change'].rolling(window=20).std()
        data['Volume_Change'] = data['Volume'].pct_change()
        
        return data
    except Exception as e:
        st.error(f"Error calculating technical indicators: {str(e)}")
        return data

def predict_price_ml(data, symbol, periods=5):
    """Predict future prices using machine learning"""
    if not SKLEARN_AVAILABLE:
        return None, "ML library not available"
    
    try:
        # Prepare features for ML
        features = ['Close', 'Volume', 'RSI', 'MACD', 'BB_Position', 'Stoch_K', 'Volatility']
        available_features = [f for f in features if f in data.columns]
        
        if len(available_features) < 2:
            return None, "Insufficient features for prediction"
        
        # Create lagged features
        df_ml = data[available_features].dropna()
        if len(df_ml) < 30:
            return None, "Insufficient data for prediction"
        
        # Create target variable (future price)
        df_ml['Target'] = df_ml['Close'].shift(-periods)
        df_ml = df_ml.dropna()
        
        if len(df_ml) < 20:
            return None, "Insufficient data after creating target"
        
        # Ensure we have valid features
        feature_cols = [col for col in available_features if col != 'Close' and col in df_ml.columns]
        if len(feature_cols) < 1:
            return None, "No valid features for prediction"
        
        # Prepare features and target
        X = df_ml[feature_cols]
        y = df_ml['Target']
        
        # Split data
        split_idx = int(len(df_ml) * 0.8)
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
            'model_type': 'Linear Regression'
        }, None
        
    except Exception as e:
        return None, f"Prediction error: {str(e)}"

def calculate_risk_score(data, symbol):
    """Calculate comprehensive risk score (0-100)"""
    try:
        risk_factors = {}
        
        # Volatility risk (0-30 points)
        volatility = data['Close'].pct_change().std() * np.sqrt(252)  # Annualized
        volatility_risk = min(30, volatility * 100)
        risk_factors['volatility'] = volatility_risk
        
        # RSI risk (0-20 points)
        if 'RSI' in data.columns:
            current_rsi = data['RSI'].iloc[-1]
            if current_rsi > 80 or current_rsi < 20:
                rsi_risk = 20
            elif current_rsi > 70 or current_rsi < 30:
                rsi_risk = 10
            else:
                rsi_risk = 0
        else:
            rsi_risk = 0
        risk_factors['rsi'] = rsi_risk
        
        # Volume risk (0-15 points)
        if 'Volume_Ratio' in data.columns:
            avg_volume_ratio = data['Volume_Ratio'].mean()
            if avg_volume_ratio > 2.0:
                volume_risk = 15
            elif avg_volume_ratio > 1.5:
                volume_risk = 10
            else:
                volume_risk = 0
        else:
            volume_risk = 0
        risk_factors['volume'] = volume_risk
        
        # Trend risk (0-20 points)
        if len(data) >= 20:
            recent_trend = (data['Close'].iloc[-1] - data['Close'].iloc[-20]) / data['Close'].iloc[-20]
            if abs(recent_trend) > 0.2:  # 20% change
                trend_risk = 20
            elif abs(recent_trend) > 0.1:  # 10% change
                trend_risk = 10
            else:
                trend_risk = 0
        else:
            trend_risk = 0
        risk_factors['trend'] = trend_risk
        
        # Bollinger Bands risk (0-15 points)
        if 'BB_Position' in data.columns:
            bb_position = data['BB_Position'].iloc[-1]
            if bb_position > 0.9 or bb_position < 0.1:
                bb_risk = 15
            elif bb_position > 0.8 or bb_position < 0.2:
                bb_risk = 10
            else:
                bb_risk = 0
        else:
            bb_risk = 0
        risk_factors['bollinger'] = bb_risk
        
        # Total risk score
        total_risk = sum(risk_factors.values())
        risk_level = "Low" if total_risk < 30 else "Medium" if total_risk < 60 else "High"
        
        return {
            'total_score': total_risk,
            'risk_level': risk_level,
            'factors': risk_factors,
            'recommendation': get_risk_recommendation(total_risk, risk_level)
        }, None
        
    except Exception as e:
        return None, f"Risk calculation error: {str(e)}"

def get_risk_recommendation(score, level):
    """Get risk-based investment recommendation"""
    if level == "Low":
        return "🟢 Conservative investment recommended"
    elif level == "Medium":
        return "🟡 Moderate risk tolerance required"
    else:
        return "🔴 High risk - consider carefully"

def get_market_overview():
    """Get comprehensive market overview data"""
    symbols = {
        'S&P 500': '^GSPC',
        'NASDAQ': '^IXIC', 
        'DOW': '^DJI',
        'VIX': '^VIX',
        'Russell 2000': '^RUT',
        'Gold': 'GC=F',
        'Oil': 'CL=F',
        '10Y Treasury': '^TNX'
    }
    
    overview = {}
    
    for name, symbol in symbols.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2d")
            
            if not hist.empty and len(hist) >= 2:
                current_price = hist['Close'].iloc[-1]
                previous_price = hist['Close'].iloc[-2]
                change = current_price - previous_price
                change_percent = (change / previous_price) * 100
                
                overview[name] = {
                    'symbol': symbol,
                    'price': current_price,
                    'change': change,
                    'change_percent': change_percent,
                    'volume': hist['Volume'].iloc[-1] if 'Volume' in hist.columns else 0
                }
        except Exception as e:
            st.warning(f"Could not fetch {name}: {str(e)}")
    
    return overview

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📊 Financial Analyzer Pro</h1>
        <p>Advanced Financial Research & Analysis Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Phase indicator
    if auth.is_authenticated():
        st.markdown("""
        <div class="phase-indicator">
            <h3>🚀 Phase 3 Features Active</h3>
            <p>✅ User Authentication | ✅ Portfolio Persistence | ✅ User Preferences | ✅ Database Integration | ✅ Personalized Experience</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="phase-indicator">
            <h3>🔍 Research Mode</h3>
            <p>✅ ML Stock Analysis | ✅ Anomaly Detection | ✅ Risk Assessment | ✅ Market Overview | 🔐 Sign in for Portfolio Management</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Handle optional authentication
    if hasattr(st.session_state, 'show_login') and st.session_state.show_login:
        auth.show_login_page()
        return
    
    if hasattr(st.session_state, 'show_register') and st.session_state.show_register:
        auth.show_register_page()
        return
    
    # User welcome
    if auth.is_authenticated():
        user = auth.get_current_user()
        st.markdown(f"""
        <div class="user-welcome">
            <h4>👤 Welcome back, {user['full_name'] or user['username']}!</h4>
            <p>Your personalized financial analysis platform is ready.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="user-welcome">
            <h4>🔍 Financial Research Platform</h4>
            <p>Analyze stocks, detect anomalies, and assess risk. Sign in to save portfolios and access personalized features.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Database status
    db_stats = db.get_database_stats()
    st.markdown(f"""
    <div class="database-card">
        <h4>🗄️ Database Status</h4>
        <p><strong>Users:</strong> {db_stats.get('users_count', 0)} | 
        <strong>Portfolios:</strong> {db_stats.get('portfolios_count', 0)} | 
        <strong>Watchlists:</strong> {db_stats.get('watchlists_count', 0)} | 
        <strong>Active Sessions:</strong> {db_stats.get('active_sessions_count', 0)}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ML Status
    ml_status = "🟢 Full ML Features Available" if SKLEARN_AVAILABLE else "🟡 Limited ML Features (scikit-learn not available)"
    st.markdown(f"""
    <div class="ml-status">
        <h4>🤖 Machine Learning Status</h4>
        <p><strong>Status:</strong> {ml_status}</p>
        <p><strong>Available Features:</strong> Price Predictions, Anomaly Detection, Risk Scoring</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Optional authentication header
    auth.show_optional_auth_header()
    
    # Main navigation
    st.sidebar.title("📊 Analysis Tools")
    
    # Different navigation options based on authentication status
    if auth.is_authenticated():
        page = st.sidebar.selectbox("Choose Analysis", [
            "📈 ML Stock Analysis", 
            "🔍 Anomaly Detection", 
            "📊 Risk Assessment",
            "💼 Portfolio Manager",
            "👀 Watchlist Manager",
            "📊 Market Overview",
            "⚙️ Settings"
        ])
    else:
        page = st.sidebar.selectbox("Choose Analysis", [
            "📈 ML Stock Analysis", 
            "🔍 Anomaly Detection", 
            "📊 Risk Assessment",
            "📊 Market Overview",
            "💼 Portfolio Manager (Sign In Required)",
            "👀 Watchlist Manager (Sign In Required)",
            "⚙️ Settings (Sign In Required)"
        ])
    
    # Route to appropriate page
    if page == "📈 ML Stock Analysis":
        ml_stock_analysis_page()
    elif page == "🔍 Anomaly Detection":
        anomaly_detection_page()
    elif page == "📊 Risk Assessment":
        risk_assessment_page()
    elif page == "📊 Market Overview":
        market_overview_page()
    elif page == "💼 Portfolio Manager" or page == "💼 Portfolio Manager (Sign In Required)":
        portfolio_persistence.show_portfolio_manager()
    elif page == "👀 Watchlist Manager" or page == "👀 Watchlist Manager (Sign In Required)":
        portfolio_persistence.show_watchlist_manager()
    elif page == "⚙️ Settings" or page == "⚙️ Settings (Sign In Required)":
        if auth.is_authenticated():
            auth.show_settings_page()
        else:
            auth.show_auth_prompt("Settings")

def ml_stock_analysis_page():
    """ML-powered stock analysis"""
    st.header("📈 ML Stock Analysis")
    
    # Get user preferences
    preferences = auth.get_user_preferences()
    default_symbols = preferences.get('default_symbols', ['AAPL', 'MSFT', 'GOOGL'])
    default_timeframe = preferences.get('default_timeframe', '1mo')
    
    col1, col2 = st.columns([2, 1])
    with col1:
        symbol = st.text_input("Enter Stock Symbol", value=default_symbols[0] if default_symbols else "AAPL", 
                              placeholder="e.g., AAPL, MSFT, GOOGL")
    with col2:
        timeframe = st.selectbox("Timeframe", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], 
                                index=["1mo", "3mo", "6mo", "1y", "2y", "5y"].index(default_timeframe))
    
    if st.button("Analyze with ML", type="primary"):
        if symbol:
            with st.spinner(f"Performing ML analysis for {symbol}..."):
                data = get_market_data(symbol, timeframe)
                
                if data is not None and not data.empty:
                    # Calculate technical indicators
                    data = calculate_technical_indicators(data)
                    
                    # Basic metrics
                    current_price = data['Close'].iloc[-1]
                    prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
                    change = current_price - prev_price
                    change_pct = (change / prev_price) * 100 if prev_price != 0 else 0
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Current Price", f"${current_price:.2f}")
                    with col2:
                        st.metric("Change", f"${change:.2f}")
                    with col3:
                        st.metric("Change %", f"{change_pct:.2f}%")
                    with col4:
                        st.metric("Volume", f"{data['Volume'].iloc[-1]:,}")
                    
                    # ML Predictions
                    st.subheader("🤖 ML Price Predictions")
                    predictions, error = predict_price_ml(data, symbol, periods=5)
                    
                    if predictions:
                        st.markdown(f"""
                        <div class="prediction-card">
                            <h4>📈 Price Predictions (Next 5 Days)</h4>
                            <p><strong>Model:</strong> {predictions['model_type']}</p>
                            <p><strong>Accuracy (R²):</strong> {predictions['accuracy']:.3f}</p>
                            <p><strong>Current Price:</strong> ${predictions['current_price']:.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show predictions
                        pred_df = pd.DataFrame({
                            'Date': predictions['dates'],
                            'Predicted Price': [f"${p:.2f}" for p in predictions['predictions']],
                            'Change from Current': [f"{((p - predictions['current_price']) / predictions['current_price'] * 100):+.2f}%" 
                                                  for p in predictions['predictions']]
                        })
                        st.dataframe(pred_df, use_container_width=True)
                    else:
                        st.error(f"Prediction failed: {error}")
                    
                    # Risk Assessment
                    st.subheader("📊 Risk Assessment")
                    risk_data, risk_error = calculate_risk_score(data, symbol)
                    
                    if risk_data:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Risk Score", f"{risk_data['total_score']:.1f}/100")
                        with col2:
                            st.metric("Risk Level", risk_data['risk_level'])
                        with col3:
                            st.metric("Recommendation", risk_data['recommendation'])
                    else:
                        st.error(f"Risk assessment failed: {risk_error}")

def anomaly_detection_page():
    """Anomaly detection analysis"""
    st.header("🔍 Anomaly Detection")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        symbol = st.text_input("Enter Stock Symbol", value="AAPL", placeholder="e.g., AAPL, MSFT, GOOGL", key="anomaly_symbol")
    with col2:
        timeframe = st.selectbox("Timeframe", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], key="anomaly_timeframe")
    
    if st.button("Detect Anomalies", type="primary"):
        if symbol:
            with st.spinner(f"Detecting anomalies for {symbol}..."):
                data = get_market_data(symbol, timeframe)
                
                if data is not None and not data.empty:
                    data = calculate_technical_indicators(data)
                    
                    # Detect anomalies
                    if SCIPY_AVAILABLE:
                        # Z-score method
                        price_changes = data['Close'].pct_change().dropna()
                        
                        if len(price_changes) >= 10:
                            z_scores = np.abs(stats.zscore(price_changes))
                            threshold = 2.5
                            anomalies = z_scores > threshold
                            
                            anomaly_count = anomalies.sum()
                            
                            st.subheader("🔍 Anomaly Detection Results")
                            st.metric("Price Anomalies", anomaly_count)
                            
                            if anomaly_count > 0:
                                st.info(f"Found {anomaly_count} price anomalies using Z-score method (threshold: {threshold})")
                            else:
                                st.success("No significant price anomalies detected")
                        else:
                            st.warning("Insufficient data for anomaly detection")
                    else:
                        st.warning("Statistical library not available for anomaly detection")

def risk_assessment_page():
    """Risk assessment analysis"""
    st.header("📊 Risk Assessment")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        symbol = st.text_input("Enter Stock Symbol", value="AAPL", placeholder="e.g., AAPL, MSFT, GOOGL", key="risk_symbol")
    with col2:
        timeframe = st.selectbox("Timeframe", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], key="risk_timeframe")
    
    if st.button("Assess Risk", type="primary"):
        if symbol:
            with st.spinner(f"Assessing risk for {symbol}..."):
                data = get_market_data(symbol, timeframe)
                
                if data is not None and not data.empty:
                    data = calculate_technical_indicators(data)
                    
                    # Calculate risk score
                    risk_data, error = calculate_risk_score(data, symbol)
                    
                    if risk_data:
                        st.subheader("📊 Risk Assessment Results")
                        
                        # Risk score display
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Overall Risk Score", f"{risk_data['total_score']:.1f}/100")
                        with col2:
                            risk_color = "🟢" if risk_data['total_score'] < 30 else "🟡" if risk_data['total_score'] < 60 else "🔴"
                            st.metric("Risk Level", f"{risk_color} {risk_data['risk_level']}")
                        with col3:
                            st.metric("Recommendation", risk_data['recommendation'])
                        
                        # Risk factors
                        st.subheader("🔍 Risk Factors Analysis")
                        risk_factors = risk_data['factors']
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Individual Risk Factors:**")
                            for factor, score in risk_factors.items():
                                st.write(f"• {factor.title()}: {score:.1f}/100")
                        
                        with col2:
                            # Risk factors chart
                            fig = px.bar(x=list(risk_factors.keys()), 
                                       y=list(risk_factors.values()),
                                       title="Risk Factors Breakdown",
                                       labels={'x': 'Risk Factor', 'y': 'Score'})
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(f"Risk assessment failed: {error}")

def market_overview_page():
    """Enhanced market overview"""
    st.header("📊 Market Overview")
    
    with st.spinner("Fetching market data..."):
        overview = get_market_overview()
    
    if overview:
        # Major indices
        st.subheader("📈 Major Indices")
        indices = ['S&P 500', 'NASDAQ', 'DOW', 'Russell 2000']
        cols = st.columns(len(indices))
        
        for i, index in enumerate(indices):
            if index in overview:
                data = overview[index]
                with cols[i]:
                    st.metric(
                        index,
                        f"${data['price']:.2f}",
                        f"{data['change']:+.2f} ({data['change_percent']:+.2f}%)"
                    )
        
        # Volatility and commodities
        st.subheader("📊 Volatility & Commodities")
        volatility_commodities = ['VIX', 'Gold', 'Oil', '10Y Treasury']
        cols = st.columns(len(volatility_commodities))
        
        for i, item in enumerate(volatility_commodities):
            if item in overview:
                data = overview[item]
                with cols[i]:
                    st.metric(
                        item,
                        f"${data['price']:.2f}",
                        f"{data['change']:+.2f} ({data['change_percent']:+.2f}%)"
                    )
        
        # Market sentiment analysis
        st.subheader("🎯 AI Market Sentiment Analysis")
        
        # Calculate overall market sentiment
        positive_count = sum(1 for data in overview.values() if data['change_percent'] > 0)
        total_count = len(overview)
        sentiment_score = (positive_count / total_count) * 100 if total_count > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Bullish Indices", f"{positive_count}/{total_count}")
        with col2:
            st.metric("Sentiment Score", f"{sentiment_score:.1f}%")
        with col3:
            sentiment = "🟢 Bullish" if sentiment_score > 60 else "🔴 Bearish" if sentiment_score < 40 else "🟡 Neutral"
            st.metric("Market Sentiment", sentiment)

if __name__ == "__main__":
    main()
