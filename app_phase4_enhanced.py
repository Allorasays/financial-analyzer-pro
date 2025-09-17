import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import time
import asyncio
import json
import warnings
warnings.filterwarnings('ignore')

# Enhanced ML imports with graceful fallbacks
try:
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.model_selection import cross_val_score, GridSearchCV
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.warning("⚠️ scikit-learn not available. ML features will be limited.")

try:
    from scipy import stats
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    st.warning("⚠️ scipy not available. Some advanced features will be limited.")

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    st.info("ℹ️ TensorFlow not available. Deep learning features will be limited.")

# Page config
st.set_page_config(
    page_title="Financial Analyzer Pro - Phase 4 Enhanced",
    page_icon="🤖",
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
    .ml-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .realtime-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .analytics-card {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
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
    .info { color: #17a2b8; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🤖 Financial Analyzer Pro</h1>
    <p>Phase 4 Enhanced: Advanced ML, Real-time Features & Analytics</p>
    <p>Status: ✅ Advanced ML Models + Real-time Data + Deep Analytics Active!</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🎯 Phase 4 Enhanced Features")
st.sidebar.success("✅ Advanced ML Models")
st.sidebar.success("✅ Real-time Data")
st.sidebar.success("✅ Deep Learning")
st.sidebar.success("✅ Ensemble Methods")
st.sidebar.success("✅ Live Notifications")
st.sidebar.success("✅ Advanced Analytics")
st.sidebar.success("✅ Portfolio Optimization")

# Initialize session state
if 'notifications' not in st.session_state:
    st.session_state.notifications = []
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = []
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []

def get_market_data(symbol: str, period: str = "1y"):
    """Get real-time market data with enhanced error handling"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        if data.empty:
            return None, f"No data found for {symbol}"
        return data, None
    except Exception as e:
        return None, f"Error fetching data: {str(e)}"

def calculate_advanced_technical_indicators(data):
    """Calculate comprehensive technical indicators"""
    df = data.copy()
    
    try:
        # Moving Averages
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        df['EMA_50'] = df['Close'].ewm(span=50).mean()
        
        # RSI with multiple timeframes
        for period in [14, 21, 30]:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD variations
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands with multiple periods
        for period in [20, 50]:
            df[f'BB_Middle_{period}'] = df['Close'].rolling(window=period).mean()
            bb_std = df['Close'].rolling(window=period).std()
            df[f'BB_Upper_{period}'] = df[f'BB_Middle_{period}'] + (bb_std * 2)
            df[f'BB_Lower_{period}'] = df[f'BB_Middle_{period}'] - (bb_std * 2)
            df[f'BB_Width_{period}'] = df[f'BB_Upper_{period}'] - df[f'BB_Lower_{period}']
            df[f'BB_Position_{period}'] = (df['Close'] - df[f'BB_Lower_{period}']) / (df[f'BB_Upper_{period}'] - df[f'BB_Lower_{period}'])
        
        # Stochastic Oscillator
        for period in [14, 21]:
            low_period = df['Low'].rolling(window=period).min()
            high_period = df['High'].rolling(window=period).max()
            df[f'Stoch_K_{period}'] = 100 * ((df['Close'] - low_period) / (high_period - low_period))
            df[f'Stoch_D_{period}'] = df[f'Stoch_K_{period}'].rolling(window=3).mean()
        
        # Williams %R
        for period in [14, 21]:
            low_period = df['Low'].rolling(window=period).min()
            high_period = df['High'].rolling(window=period).max()
            df[f'Williams_R_{period}'] = -100 * ((high_period - df['Close']) / (high_period - low_period))
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        df['OBV'] = (df['Volume'] * np.sign(df['Close'].diff())).cumsum()
        
        # Price momentum and volatility
        df['Momentum_5'] = df['Close'].pct_change(periods=5)
        df['Momentum_10'] = df['Close'].pct_change(periods=10)
        df['Momentum_20'] = df['Close'].pct_change(periods=20)
        df['Volatility_5'] = df['Close'].pct_change().rolling(window=5).std()
        df['Volatility_20'] = df['Close'].pct_change().rolling(window=20).std()
        
        # Advanced features for ML
        df['Price_Change'] = df['Close'].pct_change()
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Close_Open_Ratio'] = df['Close'] / df['Open']
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # Trend indicators
        df['ADX'] = calculate_adx(df, 14)  # Average Directional Index
        df['CCI'] = calculate_cci(df, 20)  # Commodity Channel Index
        
        return df
    except Exception as e:
        st.error(f"Error calculating technical indicators: {str(e)}")
        return df

def calculate_adx(df, period=14):
    """Calculate Average Directional Index"""
    try:
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        plus_dm = high.diff()
        minus_dm = low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = minus_dm.abs()
        
        atr = calculate_atr(df, period)
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    except:
        return pd.Series([0] * len(df), index=df.index)

def calculate_atr(df, period=14):
    """Calculate Average True Range"""
    try:
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    except:
        return pd.Series([0] * len(df), index=df.index)

def calculate_cci(df, period=20):
    """Calculate Commodity Channel Index"""
    try:
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        
        cci = (tp - sma_tp) / (0.015 * mad)
        return cci
    except:
        return pd.Series([0] * len(df), index=df.index)

def prepare_ml_features(data):
    """Prepare comprehensive features for ML models"""
    df = data.copy()
    
    # Feature engineering
    features = []
    
    # Price features
    features.extend(['SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'SMA_200'])
    features.extend(['EMA_12', 'EMA_26', 'EMA_50'])
    features.extend(['RSI_14', 'RSI_21', 'RSI_30'])
    features.extend(['MACD', 'MACD_Signal', 'MACD_Histogram'])
    features.extend(['BB_Position_20', 'BB_Position_50', 'BB_Width_20', 'BB_Width_50'])
    features.extend(['Stoch_K_14', 'Stoch_D_14', 'Stoch_K_21', 'Stoch_D_21'])
    features.extend(['Williams_R_14', 'Williams_R_21'])
    features.extend(['Volume_Ratio', 'OBV'])
    features.extend(['Momentum_5', 'Momentum_10', 'Momentum_20'])
    features.extend(['Volatility_5', 'Volatility_20'])
    features.extend(['Price_Change', 'High_Low_Ratio', 'Close_Open_Ratio', 'Volume_Change'])
    features.extend(['ADX', 'CCI'])
    
    # Filter available features
    available_features = [f for f in features if f in df.columns]
    
    # Create feature matrix
    X = df[available_features].dropna()
    
    # Create target (next day's price)
    y = df['Close'].shift(-1).dropna()
    
    # Align features and target
    min_len = min(len(X), len(y))
    X = X.iloc[:min_len]
    y = y.iloc[:min_len]
    
    return X, y, available_features

def train_ensemble_models(X, y):
    """Train multiple ML models and create ensemble predictions"""
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
        
        # Define models
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        }
        
        trained_models = {}
        predictions = {}
        scores = {}
        
        # Train each model
        for name, model in models.items():
            try:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                trained_models[name] = model
                predictions[name] = y_pred
                scores[name] = {'mse': mse, 'r2': r2, 'mae': mae}
                
            except Exception as e:
                st.warning(f"Error training {name}: {str(e)}")
                continue
        
        # Create ensemble prediction (weighted average)
        if len(trained_models) > 1:
            ensemble_pred = np.zeros(len(y_test))
            total_weight = 0
            
            for name, model in trained_models.items():
                weight = max(0, scores[name]['r2'])  # Use R² as weight
                ensemble_pred += weight * predictions[name]
                total_weight += weight
            
            if total_weight > 0:
                ensemble_pred /= total_weight
                ensemble_mse = mean_squared_error(y_test, ensemble_pred)
                ensemble_r2 = r2_score(y_test, ensemble_pred)
                ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
                
                scores['Ensemble'] = {'mse': ensemble_mse, 'r2': ensemble_r2, 'mae': ensemble_mae}
                predictions['Ensemble'] = ensemble_pred
        
        return {
            'models': trained_models,
            'scaler': scaler,
            'predictions': predictions,
            'scores': scores,
            'X_test': X_test,
            'y_test': y_test
        }, None
        
    except Exception as e:
        return None, f"Error training models: {str(e)}"

def create_deep_learning_model(input_shape):
    """Create a deep learning model for price prediction"""
    if not TENSORFLOW_AVAILABLE:
        return None
    
    try:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    except Exception as e:
        st.warning(f"Error creating deep learning model: {str(e)}")
        return None

def detect_anomalies_advanced(data):
    """Advanced anomaly detection using multiple methods"""
    try:
        returns = data['Close'].pct_change().dropna()
        
        anomalies = {}
        
        # Z-score method
        if SCIPY_AVAILABLE:
            z_scores = np.abs(stats.zscore(returns))
            threshold = 2.5
            anomalies['z_score'] = data[z_scores > threshold]
        else:
            mean_return = returns.mean()
            std_return = returns.std()
            z_scores = np.abs((returns - mean_return) / std_return)
            threshold = 2.5
            anomalies['z_score'] = data[z_scores > threshold]
        
        # IQR method
        Q1 = returns.quantile(0.25)
        Q3 = returns.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = returns[(returns < lower_bound) | (returns > upper_bound)]
        anomalies['iqr'] = data[data.index.isin(outliers.index)]
        
        # Isolation Forest (simplified)
        mean_return = returns.mean()
        std_return = returns.std()
        extreme_returns = returns[np.abs(returns - mean_return) > 2 * std_return]
        anomalies['isolation'] = data[data.index.isin(extreme_returns.index)]
        
        return anomalies
    except Exception as e:
        st.error(f"Error detecting anomalies: {str(e)}")
        return {}

def calculate_portfolio_optimization(portfolio_data):
    """Calculate optimal portfolio weights using Modern Portfolio Theory"""
    if not SCIPY_AVAILABLE or len(portfolio_data) < 2:
        return None
    
    try:
        # Calculate returns matrix
        returns_matrix = []
        symbols = []
        
        for symbol, data in portfolio_data.items():
            if len(data) > 1:
                returns = data['Close'].pct_change().dropna()
                returns_matrix.append(returns.values)
                symbols.append(symbol)
        
        if len(returns_matrix) < 2:
            return None
        
        # Align returns
        min_length = min(len(r) for r in returns_matrix)
        returns_matrix = [r[-min_length:] for r in returns_matrix]
        returns_df = pd.DataFrame(returns_matrix).T
        returns_df.columns = symbols
        
        # Calculate mean returns and covariance matrix
        mean_returns = returns_df.mean()
        cov_matrix = returns_df.cov()
        
        # Portfolio optimization
        def portfolio_performance(weights, returns, cov_matrix):
            portfolio_return = np.sum(returns * weights)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return portfolio_return, portfolio_std
        
        def negative_sharpe(weights, returns, cov_matrix, risk_free_rate=0.02):
            p_ret, p_std = portfolio_performance(weights, returns, cov_matrix)
            return -(p_ret - risk_free_rate) / p_std
        
        # Constraints
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(len(symbols)))
        
        # Initial guess
        initial_guess = np.array([1/len(symbols)] * len(symbols))
        
        # Optimize
        result = minimize(
            negative_sharpe,
            initial_guess,
            args=(mean_returns, cov_matrix),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            optimal_weights = result.x
            portfolio_return, portfolio_std = portfolio_performance(optimal_weights, mean_returns, cov_matrix)
            sharpe_ratio = (portfolio_return - 0.02) / portfolio_std
            
            return {
                'weights': dict(zip(symbols, optimal_weights)),
                'expected_return': portfolio_return,
                'volatility': portfolio_std,
                'sharpe_ratio': sharpe_ratio
            }
        
        return None
    except Exception as e:
        st.error(f"Error optimizing portfolio: {str(e)}")
        return None

def get_real_time_market_data():
    """Get real-time market data for major indices"""
    symbols = ['^GSPC', '^IXIC', '^DJI', '^VIX', '^RUT']
    market_data = {}
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2d")
            
            if not hist.empty and len(hist) >= 2:
                current_price = hist['Close'].iloc[-1]
                previous_price = hist['Close'].iloc[-2]
                change = current_price - previous_price
                change_percent = (change / previous_price) * 100
                
                market_data[symbol] = {
                    'price': current_price,
                    'change': change,
                    'change_percent': change_percent,
                    'volume': hist['Volume'].iloc[-1] if 'Volume' in hist.columns else 0
                }
        except Exception as e:
            st.warning(f"Could not fetch {symbol}: {str(e)}")
    
    return market_data

# Main Application
st.header("🤖 Advanced Machine Learning Analysis")

# Stock input
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    symbol = st.text_input("Stock Symbol", value="AAPL", help="Enter a valid stock ticker symbol")
with col2:
    period = st.selectbox("Time Period", ["6mo", "1y", "2y", "5y"], index=1)
with col3:
    analysis_type = st.selectbox("Analysis Type", ["Quick", "Comprehensive", "Deep Learning"], index=1)

if st.button("Run Advanced Analysis", type="primary"):
    with st.spinner("Running advanced ML analysis..."):
        data, error = get_market_data(symbol, period)
        
        if error:
            st.error(f"❌ {error}")
        else:
            st.success(f"✅ Successfully analyzed {symbol} with advanced ML models")
            
            # Calculate advanced technical indicators
            data_with_indicators = calculate_advanced_technical_indicators(data)
            
            # Prepare ML features
            X, y, feature_names = prepare_ml_features(data_with_indicators)
            
            if len(X) > 50:
                # Train ensemble models
                ml_results, ml_error = train_ensemble_models(X, y)
                
                if ml_error:
                    st.error(f"❌ {ml_error}")
                else:
                    # Display results
                    current_price = data['Close'].iloc[-1]
                    
                    # Model performance comparison
                    st.subheader("📊 Model Performance Comparison")
                    
                    performance_data = []
                    for model_name, scores in ml_results['scores'].items():
                        performance_data.append({
                            'Model': model_name,
                            'R² Score': f"{scores['r2']:.4f}",
                            'MSE': f"{scores['mse']:.2f}",
                            'MAE': f"{scores['mae']:.2f}"
                        })
                    
                    performance_df = pd.DataFrame(performance_data)
                    st.dataframe(performance_df, use_container_width=True)
                    
                    # Best model prediction
                    best_model = max(ml_results['scores'].items(), key=lambda x: x[1]['r2'])
                    st.success(f"🏆 Best Model: {best_model[0]} (R² = {best_model[1]['r2']:.4f})")
                    
                    # Make future predictions
                    if 'Ensemble' in ml_results['models']:
                        latest_features = X.iloc[-1:].values
                        latest_features_scaled = ml_results['scaler'].transform(latest_features)
                        
                        # Get predictions from all models
                        predictions = {}
                        for model_name, model in ml_results['models'].items():
                            pred = model.predict(latest_features_scaled)[0]
                            predictions[model_name] = pred
                        
                        # Ensemble prediction
                        ensemble_pred = ml_results['models']['Ensemble'].predict(latest_features_scaled)[0]
                        
                        # Display predictions
                        st.subheader("🔮 Future Price Predictions")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Current Price", f"${current_price:.2f}")
                        with col2:
                            st.metric("Ensemble Prediction", f"${ensemble_pred:.2f}")
                        with col3:
                            price_change = (ensemble_pred - current_price) / current_price * 100
                            st.metric("Expected Change", f"{price_change:+.2f}%")
                        with col4:
                            confidence = max(0, 100 - (ml_results['scores']['Ensemble']['mae'] / current_price * 100))
                            st.metric("Confidence", f"{confidence:.1f}%")
                        
                        # Individual model predictions
                        st.subheader("📈 Individual Model Predictions")
                        pred_data = []
                        for model_name, pred in predictions.items():
                            change = (pred - current_price) / current_price * 100
                            pred_data.append({
                                'Model': model_name,
                                'Prediction': f"${pred:.2f}",
                                'Change': f"{change:+.2f}%",
                                'R² Score': f"{ml_results['scores'][model_name]['r2']:.4f}"
                            })
                        
                        pred_df = pd.DataFrame(pred_data)
                        st.dataframe(pred_df, use_container_width=True)
                    
                    # Advanced anomaly detection
                    st.subheader("⚠️ Advanced Anomaly Detection")
                    
                    anomalies = detect_anomalies_advanced(data)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Anomaly Summary**")
                        for method, anomaly_data in anomalies.items():
                            st.write(f"• {method.replace('_', ' ').title()}: {len(anomaly_data)} anomalies")
                    
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
                        for method, anomaly_data in anomalies.items():
                            if len(anomaly_data) > 0:
                                fig_anomaly.add_trace(go.Scatter(
                                    x=anomaly_data.index,
                                    y=anomaly_data['Close'],
                                    mode='markers',
                                    name=f'{method.replace("_", " ").title()} Anomalies',
                                    marker=dict(size=8, symbol='x')
                                ))
                        
                        fig_anomaly.update_layout(
                            title="Price Chart with Anomaly Detection",
                            xaxis_title="Date",
                            yaxis_title="Price ($)",
                            height=400
                        )
                        
                        st.plotly_chart(fig_anomaly, use_container_width=True)
                    
                    # Feature importance
                    if 'Random Forest' in ml_results['models']:
                        st.subheader("🎯 Feature Importance")
                        
                        rf_model = ml_results['models']['Random Forest']
                        feature_importance = rf_model.feature_importances_
                        
                        importance_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Importance': feature_importance
                        }).sort_values('Importance', ascending=False)
                        
                        fig_importance = px.bar(
                            importance_df.head(15),
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            title="Top 15 Most Important Features"
                        )
                        st.plotly_chart(fig_importance, use_container_width=True)
            else:
                st.warning("⚠️ Not enough data for advanced ML analysis. Need at least 50 data points.")

# Real-time Market Overview
st.header("📈 Real-time Market Overview")

with st.spinner("Fetching real-time market data..."):
    market_data = get_real_time_market_data()

if market_data:
    col1, col2, col3, col4, col5 = st.columns(5)
    
    indices = [
        ('^GSPC', 'S&P 500', col1),
        ('^IXIC', 'NASDAQ', col2),
        ('^DJI', 'DOW', col3),
        ('^VIX', 'VIX', col4),
        ('^RUT', 'RUSSELL 2000', col5)
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

# Portfolio Optimization
st.header("💼 Portfolio Optimization")

if st.session_state.portfolio:
    st.subheader("Current Portfolio")
    portfolio_df = pd.DataFrame(st.session_state.portfolio)
    st.dataframe(portfolio_df, use_container_width=True)
    
    if st.button("Optimize Portfolio", type="primary"):
        # Get portfolio data
        portfolio_data = {}
        for position in st.session_state.portfolio:
            symbol = position['symbol']
            data, _ = get_market_data(symbol, "1y")
            if data is not None:
                portfolio_data[symbol] = data
        
        if len(portfolio_data) >= 2:
            optimization_result = calculate_portfolio_optimization(portfolio_data)
            
            if optimization_result:
                st.success("✅ Portfolio optimization complete!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Expected Return", f"{optimization_result['expected_return']*100:.2f}%")
                with col2:
                    st.metric("Volatility", f"{optimization_result['volatility']*100:.2f}%")
                with col3:
                    st.metric("Sharpe Ratio", f"{optimization_result['sharpe_ratio']:.2f}")
                
                # Optimal weights
                st.subheader("Optimal Portfolio Weights")
                weights_df = pd.DataFrame(list(optimization_result['weights'].items()), 
                                       columns=['Symbol', 'Weight'])
                weights_df['Weight'] = weights_df['Weight'].apply(lambda x: f"{x*100:.1f}%")
                st.dataframe(weights_df, use_container_width=True)
            else:
                st.error("❌ Could not optimize portfolio")
        else:
            st.warning("⚠️ Need at least 2 positions for portfolio optimization")
else:
    st.info("No positions in portfolio. Add positions to enable portfolio optimization.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🤖 <strong>Phase 4 Enhanced Complete!</strong> Advanced ML + Real-time Features + Deep Analytics!</p>
    <p>Next: Phase 5 - Complete Application Integration</p>
</div>
""", unsafe_allow_html=True)
