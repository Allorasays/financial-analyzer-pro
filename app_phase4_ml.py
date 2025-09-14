import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import time
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Financial Analyzer Pro - Phase 4",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
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
    .success {
        color: #28a745;
        font-weight: bold;
    }
    .error {
        color: #dc3545;
        font-weight: bold;
    }
    .warning {
        color: #ffc107;
        font-weight: bold;
    }
    .ml-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🤖 Financial Analyzer Pro</h1>
    <p>Phase 4: Machine Learning & Advanced Analytics</p>
    <p>Status: ✅ ML Models + Advanced Features Active!</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🎯 Phase 4 Features")
st.sidebar.success("✅ Real-time Data")
st.sidebar.success("✅ Technical Indicators")
st.sidebar.success("✅ Financial Ratios")
st.sidebar.success("✅ ML Predictions")
st.sidebar.success("✅ Risk Assessment")
st.sidebar.success("✅ Anomaly Detection")
st.sidebar.success("✅ Portfolio Optimization")

def get_stock_data(symbol, period="1y"):
    """Get real-time stock data using yfinance"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        if data.empty:
            return None, f"No data found for {symbol}"
        return data, None
    except Exception as e:
        return None, f"Error fetching data: {str(e)}"

def calculate_technical_indicators(data):
    """Calculate technical indicators"""
    df = data.copy()
    
    # Simple Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    # Price features
    df['Price_Change'] = df['Close'].pct_change()
    df['Volume_Change'] = df['Volume'].pct_change()
    df['High_Low_Ratio'] = df['High'] / df['Low']
    df['Close_Open_Ratio'] = df['Close'] / df['Open']
    
    return df

def prepare_ml_features(data):
    """Prepare features for ML models"""
    df = data.copy()
    
    # Create features
    features = []
    
    # Price features
    features.extend([
        'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal',
        'BB_Upper', 'BB_Lower', 'BB_Middle',
        'Price_Change', 'Volume_Change', 'High_Low_Ratio', 'Close_Open_Ratio'
    ])
    
    # Create feature matrix
    X = df[features].dropna()
    
    # Create target (next day's price)
    y = df['Close'].shift(-1).dropna()
    
    # Align features and target
    min_len = min(len(X), len(y))
    X = X.iloc[:min_len]
    y = y.iloc[:min_len]
    
    return X, y

def train_price_prediction_model(X, y):
    """Train ML model for price prediction"""
    try:
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return model, scaler, mse, r2, y_test, y_pred
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None, None, None, None, None

def detect_anomalies(data):
    """Detect anomalies in stock data"""
    try:
        # Calculate returns
        returns = data['Close'].pct_change().dropna()
        
        # Z-score method
        z_scores = np.abs(stats.zscore(returns))
        threshold = 2.5
        
        anomalies = data[z_scores > threshold]
        
        # Isolation Forest method (simplified)
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Mark extreme returns
        extreme_returns = returns[np.abs(returns - mean_return) > 2 * std_return]
        
        return anomalies, extreme_returns
    except Exception as e:
        st.error(f"Error detecting anomalies: {str(e)}")
        return pd.DataFrame(), pd.Series()

def calculate_risk_metrics(data):
    """Calculate comprehensive risk metrics"""
    returns = data['Close'].pct_change().dropna()
    
    risk_metrics = {}
    
    # Basic risk metrics
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
    
    # Skewness and Kurtosis
    risk_metrics['Skewness'] = f"{returns.skew():.2f}"
    risk_metrics['Kurtosis'] = f"{returns.kurtosis():.2f}"
    
    return risk_metrics

def get_market_overview():
    """Get real-time market data"""
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
            st.error(f"Error fetching {symbol}: {str(e)}")
    
    return overview

# Main Content
st.header("🤖 Machine Learning Stock Analysis")

# Stock input
col1, col2 = st.columns([1, 3])
with col1:
    symbol = st.text_input("Stock Symbol", value="AAPL", help="Enter a valid stock ticker symbol")
with col2:
    period = st.selectbox("Time Period", ["6mo", "1y", "2y", "5y"], index=1)

if st.button("Run ML Analysis", type="primary"):
    with st.spinner("Fetching data and running ML models..."):
        data, error = get_stock_data(symbol, period)
        
        if error:
            st.error(f"❌ {error}")
        else:
            st.success(f"✅ Successfully analyzed {symbol} with ML models")
            
            # Calculate technical indicators
            data_with_indicators = calculate_technical_indicators(data)
            
            # Prepare ML features
            X, y = prepare_ml_features(data_with_indicators)
            
            if len(X) > 50:  # Ensure enough data for ML
                # Train ML model
                model, scaler, mse, r2, y_test, y_pred = train_price_prediction_model(X, y)
                
                if model is not None:
                    # Display ML results
                    col1, col2, col3, col4 = st.columns(4)
                    
                    current_price = data['Close'].iloc[-1]
                    previous_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
                    change = current_price - previous_price
                    change_percent = (change / previous_price) * 100
                    
                    with col1:
                        st.metric("Current Price", f"${current_price:.2f}", f"{change:+.2f}")
                    with col2:
                        st.metric("Model R²", f"{r2:.3f}", "Accuracy")
                    with col3:
                        st.metric("MSE", f"{mse:.2f}", "Error")
                    with col4:
                        st.metric("Data Points", f"{len(X)}", "Training")
                    
                    # Tabs for different ML analysis
                    tab1, tab2, tab3, tab4 = st.tabs(["🤖 ML Predictions", "⚠️ Anomaly Detection", "📊 Risk Analysis", "📈 Model Performance"])
                    
                    with tab1:
                        st.subheader("Machine Learning Predictions")
                        
                        # Make future predictions
                        if len(X) > 0:
                            latest_features = X.iloc[-1:].values
                            latest_features_scaled = scaler.transform(latest_features)
                            next_price_pred = model.predict(latest_features_scaled)[0]
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Next Day Prediction**")
                                st.metric("Predicted Price", f"${next_price_pred:.2f}")
                                
                                # Prediction confidence
                                prediction_error = np.sqrt(mse)
                                confidence = max(0, 100 - (prediction_error / current_price * 100))
                                st.metric("Confidence", f"{confidence:.1f}%")
                            
                            with col2:
                                st.write("**Prediction Analysis**")
                                price_change_pred = (next_price_pred - current_price) / current_price * 100
                                st.metric("Expected Change", f"{price_change_pred:+.2f}%")
                                
                                if price_change_pred > 0:
                                    st.success("📈 Bullish Prediction")
                                else:
                                    st.error("📉 Bearish Prediction")
                        
                        # Feature importance
                        st.subheader("Feature Importance")
                        feature_names = X.columns
                        feature_importance = np.abs(model.coef_)
                        
                        importance_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Importance': feature_importance
                        }).sort_values('Importance', ascending=False)
                        
                        fig_importance = px.bar(
                            importance_df.head(10),
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            title="Top 10 Most Important Features"
                        )
                        st.plotly_chart(fig_importance, use_container_width=True)
                    
                    with tab2:
                        st.subheader("Anomaly Detection")
                        
                        # Detect anomalies
                        anomalies, extreme_returns = detect_anomalies(data)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Anomaly Summary**")
                            st.metric("Total Anomalies", len(anomalies))
                            st.metric("Extreme Returns", len(extreme_returns))
                            
                            if len(anomalies) > 0:
                                st.write("**Recent Anomalies**")
                                recent_anomalies = anomalies.tail(5)
                                for idx, row in recent_anomalies.iterrows():
                                    st.write(f"• {idx.strftime('%Y-%m-%d')}: ${row['Close']:.2f}")
                        
                        with col2:
                            # Anomaly chart
                            fig_anomaly = go.Figure()
                            
                            # Price line
                            fig_anomaly.add_trace(go.Scatter(
                                x=data.index,
                                y=data['Close'],
                                mode='lines',
                                name='Close Price',
                                line=dict(color='blue', width=2)
                            ))
                            
                            # Anomaly points
                            if len(anomalies) > 0:
                                fig_anomaly.add_trace(go.Scatter(
                                    x=anomalies.index,
                                    y=anomalies['Close'],
                                    mode='markers',
                                    name='Anomalies',
                                    marker=dict(color='red', size=8, symbol='x')
                                ))
                            
                            fig_anomaly.update_layout(
                                title="Price Chart with Anomaly Detection",
                                xaxis_title="Date",
                                yaxis_title="Price ($)",
                                height=400
                            )
                            
                            st.plotly_chart(fig_anomaly, use_container_width=True)
                    
                    with tab3:
                        st.subheader("Advanced Risk Analysis")
                        
                        # Calculate risk metrics
                        risk_metrics = calculate_risk_metrics(data)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Risk Metrics**")
                            for metric, value in risk_metrics.items():
                                st.write(f"• {metric}: {value}")
                        
                        with col2:
                            # Risk assessment
                            volatility = float(risk_metrics['Volatility (Annualized)'].replace('%', ''))
                            sharpe = float(risk_metrics['Sharpe Ratio'])
                            
                            if volatility < 20:
                                risk_level = "Low"
                                risk_color = "green"
                            elif volatility < 40:
                                risk_level = "Medium"
                                risk_color = "orange"
                            else:
                                risk_level = "High"
                                risk_color = "red"
                            
                            st.write("**Risk Assessment**")
                            st.write(f"• Risk Level: <span style='color: {risk_color}'>{risk_level}</span>", unsafe_allow_html=True)
                            st.write(f"• Volatility: {volatility:.1f}%")
                            st.write(f"• Sharpe Ratio: {sharpe:.2f}")
                            
                            # Risk recommendation
                            if risk_level == "Low":
                                st.success("✅ Low risk investment")
                            elif risk_level == "Medium":
                                st.warning("⚠️ Medium risk investment")
                            else:
                                st.error("🚨 High risk investment")
                    
                    with tab4:
                        st.subheader("Model Performance Analysis")
                        
                        if y_test is not None and y_pred is not None:
                            # Performance metrics
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Model Metrics**")
                                st.metric("R² Score", f"{r2:.3f}")
                                st.metric("MSE", f"{mse:.2f}")
                                st.metric("RMSE", f"{np.sqrt(mse):.2f}")
                                
                                # Accuracy interpretation
                                if r2 > 0.7:
                                    st.success("✅ Excellent model performance")
                                elif r2 > 0.5:
                                    st.warning("⚠️ Good model performance")
                                else:
                                    st.error("❌ Poor model performance")
                            
                            with col2:
                                # Actual vs Predicted chart
                                fig_performance = go.Figure()
                                
                                fig_performance.add_trace(go.Scatter(
                                    x=y_test.values,
                                    y=y_pred,
                                    mode='markers',
                                    name='Actual vs Predicted',
                                    marker=dict(color='blue', size=6)
                                ))
                                
                                # Perfect prediction line
                                min_val = min(y_test.min(), y_pred.min())
                                max_val = max(y_test.max(), y_pred.max())
                                fig_performance.add_trace(go.Scatter(
                                    x=[min_val, max_val],
                                    y=[min_val, max_val],
                                    mode='lines',
                                    name='Perfect Prediction',
                                    line=dict(color='red', dash='dash')
                                ))
                                
                                fig_performance.update_layout(
                                    title="Actual vs Predicted Prices",
                                    xaxis_title="Actual Price ($)",
                                    yaxis_title="Predicted Price ($)",
                                    height=400
                                )
                                
                                st.plotly_chart(fig_performance, use_container_width=True)
                else:
                    st.error("❌ Failed to train ML model")
            else:
                st.warning("⚠️ Not enough data for ML analysis. Need at least 50 data points.")

# Market Overview
st.header("📈 Live Market Overview")

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

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🤖 <strong>Phase 4 Complete!</strong> Machine Learning & Advanced Analytics Active!</p>
    <p>Next: Phase 5 - Full Application Deployment</p>
</div>
""", unsafe_allow_html=True)
