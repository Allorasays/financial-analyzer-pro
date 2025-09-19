import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import warnings
import json
import io
import time
import os
import sqlite3
import hashlib
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

# Page config - optimized for Render
st.set_page_config(
    page_title="Financial Analyzer Pro - Enhanced",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# User Preferences System
class UserPreferences:
    def __init__(self):
        self.preferences = {
            'theme': 'light',
            'default_symbol': 'AAPL',
            'default_timeframe': '1mo',
            'show_advanced_indicators': True,
            'chart_height': 600,
            'prediction_horizon': 5
        }
        self.load_preferences()
    
    def load_preferences(self):
        """Load user preferences from session state"""
        if 'user_preferences' not in st.session_state:
            st.session_state.user_preferences = self.preferences.copy()
        else:
            self.preferences = st.session_state.user_preferences
    
    def save_preferences(self):
        """Save user preferences to session state"""
        st.session_state.user_preferences = self.preferences.copy()
    
    def get(self, key, default=None):
        return self.preferences.get(key, default)
    
    def set(self, key, value):
        self.preferences[key] = value
        self.save_preferences()

# ML Prediction Accuracy Tracker
class MLAccuracyTracker:
    def __init__(self, db_path="ml_accuracy.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                prediction_date TEXT NOT NULL,
                predicted_price REAL NOT NULL,
                actual_price REAL,
                prediction_horizon INTEGER NOT NULL,
                model_type TEXT NOT NULL,
                features_used INTEGER NOT NULL,
                r2_score REAL,
                created_at TEXT NOT NULL,
                updated_at TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_prediction(self, symbol, prediction_data, horizon):
        """Store a new prediction"""
        prediction_id = hashlib.md5(f"{symbol}_{datetime.now()}_{horizon}".encode()).hexdigest()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions 
            (id, symbol, prediction_date, predicted_price, prediction_horizon, model_type, features_used, r2_score, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            prediction_id,
            symbol,
            prediction_data['dates'][0].strftime('%Y-%m-%d'),
            prediction_data['predictions'][0],
            horizon,
            prediction_data['model_type'],
            prediction_data['features_used'],
            prediction_data.get('r2_score', 0),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        return prediction_id
    
    def calculate_accuracy_metrics(self, symbol):
        """Calculate accuracy metrics for a symbol"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT predicted_price, actual_price, prediction_horizon
            FROM predictions 
            WHERE symbol = ? AND actual_price IS NOT NULL
        ''', (symbol,))
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return {
                'total_predictions': 0,
                'accuracy_percentage': 0.0,
                'mae': 0.0,
                'rmse': 0.0,
                'recent_accuracy': 0.0
            }
        
        predicted_prices = [r[0] for r in results]
        actual_prices = [r[1] for r in results]
        
        # Calculate metrics
        total_predictions = len(results)
        
        # Accuracy within 5% of actual price
        correct_predictions = sum(1 for p, a in zip(predicted_prices, actual_prices) 
                                if abs(p - a) / a < 0.05)
        accuracy_percentage = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
        
        # MAE and RMSE
        mae = np.mean([abs(p - a) for p, a in zip(predicted_prices, actual_prices)])
        rmse = np.sqrt(np.mean([(p - a) ** 2 for p, a in zip(predicted_prices, actual_prices)]))
        
        # Recent accuracy (last 5 predictions)
        recent_results = results[-5:] if len(results) >= 5 else results
        recent_correct = sum(1 for r in recent_results 
                           if abs(r[0] - r[1]) / r[1] < 0.05)
        recent_accuracy = (recent_correct / len(recent_results) * 100) if recent_results else 0
        
        return {
            'total_predictions': total_predictions,
            'accuracy_percentage': accuracy_percentage,
            'mae': mae,
            'rmse': rmse,
            'recent_accuracy': recent_accuracy
        }

# Smart Cache System
class SmartCache:
    def __init__(self, max_size=50, ttl=300):
        self.cache = {}
        self.timestamps = {}
        self.max_size = max_size
        self.ttl = ttl  # Time to live in seconds
    
    def get(self, key):
        if key in self.cache:
            if time.time() - self.timestamps[key] < self.ttl:
                return self.cache[key]
            else:
                del self.cache[key]
                del self.timestamps[key]
        return None
    
    def set(self, key, value):
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
            del self.cache[oldest_key]
            del self.timestamps[oldest_key]
        
        self.cache[key] = value
        self.timestamps[key] = time.time()
    
    def clear(self):
        self.cache.clear()
        self.timestamps.clear()
    
    def get_stats(self):
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'ttl': self.ttl
        }

# Initialize components
preferences = UserPreferences()
accuracy_tracker = MLAccuracyTracker()
cache = SmartCache()

# Dynamic CSS based on theme
def get_theme_css(theme):
    if theme == 'dark':
        return """
        <style>
            .main-header {
                background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
                padding: 2rem;
                border-radius: 10px;
                color: white;
                text-align: center;
                margin-bottom: 2rem;
            }
            .metric-card {
                background: #2c3e50;
                color: white;
                padding: 1.5rem;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.3);
                margin: 1rem 0;
            }
            .prediction-card {
                background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
                padding: 1.5rem;
                border-radius: 10px;
                color: white;
                margin: 1rem 0;
            }
            .analytics-card {
                background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
                color: white;
                padding: 1.5rem;
                border-radius: 10px;
                margin: 1rem 0;
                text-align: center;
            }
            .indicator-card {
                background: #34495e;
                color: white;
                padding: 1rem;
                border-radius: 8px;
                border-left: 4px solid #3498db;
                margin: 0.5rem 0;
            }
            .success-message {
                background: #d4edda;
                color: #155724;
                padding: 1rem;
                border-radius: 10px;
                margin-bottom: 1rem;
                border-left: 5px solid #28a745;
            }
            .info-message {
                background: #d1ecf1;
                color: #0c5460;
                padding: 1rem;
                border-radius: 10px;
                margin: 1rem 0;
                border-left: 5px solid #17a2b8;
            }
        </style>
        """
    else:
        return """
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
            .prediction-card {
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                padding: 1.5rem;
                border-radius: 10px;
                color: white;
                margin: 1rem 0;
            }
            .analytics-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 1.5rem;
                border-radius: 10px;
                margin: 1rem 0;
                text-align: center;
            }
            .indicator-card {
                background: #f8f9fa;
                padding: 1rem;
                border-radius: 8px;
                border-left: 4px solid #667eea;
                margin: 0.5rem 0;
            }
            .success-message {
                background: #d4edda;
                color: #155724;
                padding: 1rem;
                border-radius: 10px;
                margin-bottom: 1rem;
                border-left: 5px solid #28a745;
            }
            .info-message {
                background: #d1ecf1;
                color: #0c5460;
                padding: 1rem;
                border-radius: 10px;
                margin: 1rem 0;
                border-left: 5px solid #17a2b8;
            }
        </style>
        """

# Apply theme
current_theme = preferences.get('theme', 'light')
st.markdown(get_theme_css(current_theme), unsafe_allow_html=True)

def get_market_data(symbol: str, period: str = "1mo"):
    """Get market data with smart caching and robust fallback"""
    cache_key = f"market_data_{symbol}_{period}"
    cached_data = cache.get(cache_key)
    
    if cached_data is not None:
        return cached_data
    
    try:
        # Try with timeout
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, timeout=10)
        
        if data is not None and not data.empty:
            cache.set(cache_key, data)
            return data
    except Exception as e:
        st.warning(f"Yahoo Finance API failed for {symbol}: {str(e)}")
    
    # Fallback to 1-day data
    try:
        data = ticker.history(period="1d", timeout=5)
        if data is not None and not data.empty:
            cache.set(cache_key, data)
            return data
    except Exception as e:
        st.warning(f"1-day fallback failed for {symbol}: {str(e)}")
    
    # Final fallback to demo data
    st.info(f"Using demo data for {symbol} (API unavailable)")
    
    period_days = {
        "1mo": 30, "3mo": 90, "6mo": 180, 
        "1y": 365, "2y": 730, "5y": 1825
    }.get(period, 30)
    
    dates = pd.date_range(start=datetime.now() - timedelta(days=period_days), end=datetime.now(), freq='D')
    np.random.seed(hash(symbol) % 2**32)
    
    symbol_prices = {
        'AAPL': 150, 'MSFT': 300, 'GOOGL': 2500, 'AMZN': 3000,
        'TSLA': 200, 'META': 300, 'NVDA': 400, 'NFLX': 400
    }
    base_price = symbol_prices.get(symbol.upper(), 100 + (hash(symbol) % 1000))
    
    price_changes = np.random.normal(0, 0.02, len(dates))
    prices = [base_price]
    
    for change in price_changes[1:]:
        prices.append(prices[-1] * (1 + change))
    
    data = pd.DataFrame({
        'Open': [p * (1 + np.random.normal(0, 0.01)) for p in prices],
        'High': [p * (1 + abs(np.random.normal(0, 0.02))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.02))) for p in prices],
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    
    cache.set(cache_key, data)
    return data

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
    """Enhanced ML prediction with accuracy tracking"""
    if not SKLEARN_AVAILABLE:
        return None, "ML library not available"
    
    try:
        # Start with basic features
        basic_features = ['Close', 'Volume']
        enhanced_features = ['RSI', 'SMA_20', 'MACD', 'BB_Position']
        
        # Check which features are available
        available_features = []
        for feature in basic_features + enhanced_features:
            if feature in data.columns and not data[feature].isna().all():
                available_features.append(feature)
        
        # Fallback to basic features if needed
        if len(available_features) < 2:
            basic_available = [f for f in basic_features if f in data.columns]
            if len(basic_available) >= 2:
                available_features = basic_available
            else:
                return None, "Insufficient features for prediction"
        
        # Create ML dataset
        df_ml = data[available_features].copy()
        df_ml = df_ml.fillna(method='ffill').fillna(method='bfill')
        df_ml = df_ml.dropna()
        
        if len(df_ml) < 20:
            return None, "Insufficient data for prediction"
        
        # Create target variable
        df_ml['Target'] = df_ml['Close'].shift(-periods)
        df_ml = df_ml.dropna()
        
        if len(df_ml) < 10:
            return None, "Insufficient data after creating target"
        
        # Prepare features and target
        feature_cols = [col for col in available_features if col != 'Close']
        if len(feature_cols) == 0:
            return None, "No valid features for prediction"
        
        X = df_ml[feature_cols]
        y = df_ml['Target']
        
        # Check for invalid values
        if X.isna().any().any() or np.isinf(X).any().any():
            return None, "Invalid data values in features"
        
        if y.isna().any() or np.isinf(y).any():
            return None, "Invalid data values in target"
        
        # Train model
        model = LinearRegression()
        model.fit(X, y)
        
        # Make predictions
        last_features = X.iloc[-1:].values
        future_prices = []
        current_price = data['Close'].iloc[-1]
        
        for i in range(periods):
            pred_price = model.predict(last_features)[0]
            future_prices.append(pred_price)
            
            # Update features for next prediction
            if len(last_features[0]) > 0 and 'Volume' in feature_cols:
                vol_idx = feature_cols.index('Volume') if 'Volume' in feature_cols else 0
                last_features[0][vol_idx] = last_features[0][vol_idx] * 0.99
        
        # Create prediction dates
        last_date = data.index[-1]
        prediction_dates = [last_date + timedelta(days=i+1) for i in range(periods)]
        
        # Calculate R² score safely
        try:
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred) if len(y) > 1 else 0
        except:
            r2 = 0
        
        prediction_data = {
            'predictions': future_prices,
            'dates': prediction_dates,
            'current_price': current_price,
            'model_type': 'Enhanced Linear Regression',
            'features_used': len(feature_cols),
            'r2_score': r2,
            'accuracy': r2  # For backward compatibility
        }
        
        # Store prediction for accuracy tracking
        accuracy_tracker.store_prediction(symbol, prediction_data, periods)
        
        return prediction_data, None
        
    except Exception as e:
        return None, f"Prediction error: {str(e)}"

def create_enhanced_chart(data, symbol, theme='light'):
    """Create enhanced chart with theme support"""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(f'{symbol} - Price Chart', 'RSI', 'MACD'),
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # Price chart with candlesticks
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name=symbol
    ), row=1, col=1)
    
    # Moving averages
    if 'SMA_20' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['SMA_20'],
            mode='lines',
            name='SMA 20',
            line=dict(color='orange', width=2)
        ), row=1, col=1)
    
    if 'SMA_50' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['SMA_50'],
            mode='lines',
            name='SMA 50',
            line=dict(color='blue', width=2)
        ), row=1, col=1)
    
    # Bollinger Bands
    if 'BB_Upper' in data.columns and 'BB_Lower' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['BB_Upper'],
            mode='lines',
            name='BB Upper',
            line=dict(color='gray', width=1, dash='dash'),
            showlegend=False
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['BB_Lower'],
            mode='lines',
            name='BB Lower',
            line=dict(color='gray', width=1, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(128,128,128,0.1)',
            showlegend=False
        ), row=1, col=1)
    
    # RSI
    if 'RSI' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['RSI'],
            mode='lines',
            name='RSI',
            line=dict(color='purple', width=2)
        ), row=2, col=1)
        
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
    
    # MACD
    if 'MACD' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['MACD'],
            mode='lines',
            name='MACD',
            line=dict(color='blue', width=2)
        ), row=3, col=1)
        
        if 'MACD_Signal' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['MACD_Signal'],
                mode='lines',
                name='MACD Signal',
                line=dict(color='red', width=2)
            ), row=3, col=1)
    
    # Update layout
    fig.update_layout(
        title=f'{symbol} - Enhanced Technical Analysis',
        height=preferences.get('chart_height', 600),
        showlegend=True,
        hovermode='x unified'
    )
    
    # Update axes
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    
    return fig

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

def create_candlestick_chart(data, symbol):
    """Create professional candlestick chart"""
    fig = go.Figure(data=go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name=symbol
    ))
    
    # Add moving averages
    if 'SMA_20' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['SMA_20'],
            mode='lines',
            name='SMA 20',
            line=dict(color='orange', width=2)
        ))
    
    if 'SMA_50' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['SMA_50'],
            mode='lines',
            name='SMA 50',
            line=dict(color='blue', width=2)
        ))
    
    # Add Bollinger Bands
    if all(col in data.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
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
        title=f'{symbol} - Price Chart with Technical Indicators',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        height=600,
        showlegend=True
    )
    
    return fig

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📊 Financial Analyzer Pro - Enhanced</h1>
        <p>Advanced Analytics with User Experience & ML Accuracy Tracking</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Status
    st.markdown("""
    <div class="success-message">
        <h4>🚀 Day 1-3: Enhanced Features</h4>
        <p>✅ Smart Caching | ✅ ML Accuracy Tracking | ✅ Theme Toggle | ✅ User Preferences | ✅ Enhanced Analytics | ✅ Robust Data Fallback</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with user preferences
    st.sidebar.title("⚙️ Settings & Preferences")
    
    # Theme toggle
    theme = st.sidebar.selectbox("🎨 Theme", ["light", "dark"], index=0 if preferences.get('theme') == 'light' else 1)
    if theme != preferences.get('theme'):
        preferences.set('theme', theme)
        st.rerun()
    
    # User preferences
    st.sidebar.subheader("📊 Analysis Settings")
    default_symbol = st.sidebar.text_input("Default Symbol", value=preferences.get('default_symbol', 'AAPL'))
    preferences.set('default_symbol', default_symbol)
    
    default_timeframe = st.sidebar.selectbox("Default Timeframe", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], 
                                           index=["1mo", "3mo", "6mo", "1y", "2y", "5y"].index(preferences.get('default_timeframe', '1mo')))
    preferences.set('default_timeframe', default_timeframe)
    
    show_advanced = st.sidebar.checkbox("Show Advanced Indicators", value=preferences.get('show_advanced_indicators', True))
    preferences.set('show_advanced_indicators', show_advanced)
    
    prediction_horizon = st.sidebar.slider("Prediction Horizon (days)", 1, 10, preferences.get('prediction_horizon', 5))
    preferences.set('prediction_horizon', prediction_horizon)
    
    chart_height = st.sidebar.slider("Chart Height", 400, 800, preferences.get('chart_height', 600))
    preferences.set('chart_height', chart_height)
    
    # Cache stats
    cache_stats = cache.get_stats()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Cache Size", f"{cache_stats['size']}/{cache_stats['max_size']}")
    with col2:
        st.metric("ML Status", "🟢 Available" if SKLEARN_AVAILABLE else "🟡 Limited")
    with col3:
        if st.button("Clear Cache"):
            cache.clear()
            st.success("Cache cleared!")
    
    # Main navigation
    st.sidebar.title("📊 Analysis Tools")
    page = st.sidebar.selectbox("Choose Analysis", [
        "📈 Enhanced ML Analysis", 
        "🔍 Anomaly Detection", 
        "📊 Risk Assessment",
        "📊 Market Overview",
        "📈 Technical Charts",
        "📊 ML Accuracy Tracking"
    ])
    
    # Route to appropriate page
    if page == "📈 Enhanced ML Analysis":
        enhanced_ml_analysis_page()
    elif page == "🔍 Anomaly Detection":
        anomaly_detection_page()
    elif page == "📊 Risk Assessment":
        risk_assessment_page()
    elif page == "📊 Market Overview":
        market_overview_page()
    elif page == "📈 Technical Charts":
        technical_charts_page()
    elif page == "📊 ML Accuracy Tracking":
        ml_accuracy_tracking_page()

def enhanced_ml_analysis_page():
    """Enhanced ML-powered stock analysis with accuracy tracking"""
    st.header("📈 Enhanced ML Stock Analysis")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        symbol = st.text_input("Enter Stock Symbol", value=preferences.get('default_symbol', 'AAPL'), 
                              placeholder="e.g., AAPL, MSFT, GOOGL", key="enhanced_symbol")
    with col2:
        timeframe = st.selectbox("Timeframe", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], 
                               index=["1mo", "3mo", "6mo", "1y", "2y", "5y"].index(preferences.get('default_timeframe', '1mo')))
    
    if st.button("Analyze with Enhanced ML", type="primary"):
        if symbol:
            with st.spinner(f"Performing enhanced ML analysis for {symbol}..."):
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
                    
                    # Enhanced Technical Indicators (conditional on user preference)
                    if show_advanced:
                        st.subheader("📊 Enhanced Technical Indicators")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.markdown("**Momentum Indicators**")
                            if 'RSI' in data.columns:
                                rsi = data['RSI'].iloc[-1]
                                rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
                                st.metric("RSI", f"{rsi:.1f}", help=f"Status: {rsi_status}")
                        
                        with col2:
                            st.markdown("**Trend Indicators**")
                            if 'SMA_20' in data.columns:
                                sma20 = data['SMA_20'].iloc[-1]
                                st.metric("SMA 20", f"${sma20:.2f}")
                            
                            if 'MACD' in data.columns:
                                macd = data['MACD'].iloc[-1]
                                st.metric("MACD", f"{macd:.3f}")
                        
                        with col3:
                            st.markdown("**Volatility Indicators**")
                            if 'BB_Position' in data.columns:
                                bb_pos = data['BB_Position'].iloc[-1]
                                bb_status = "Upper Band" if bb_pos > 0.8 else "Lower Band" if bb_pos < 0.2 else "Middle"
                                st.metric("BB Position", f"{bb_pos:.2f}", help=f"Status: {bb_status}")
                        
                        with col4:
                            st.markdown("**Volume Indicators**")
                            if 'Volume' in data.columns:
                                vol = data['Volume'].iloc[-1]
                                st.metric("Current Volume", f"{vol:,}")
                    
                    # Enhanced ML Predictions with accuracy tracking
                    st.subheader("🤖 Enhanced ML Price Predictions & Accuracy Tracking")
                    predictions, error = predict_price_ml(data, symbol, periods=prediction_horizon)
                    
                    if predictions:
                        st.markdown(f"""
                        <div class="prediction-card">
                            <h4>📈 Price Predictions (Next {prediction_horizon} Days)</h4>
                            <p><strong>Model:</strong> {predictions['model_type']}</p>
                            <p><strong>Features Used:</strong> {predictions['features_used']}</p>
                            <p><strong>R² Score:</strong> {predictions['accuracy']:.3f}</p>
                            <p><strong>Current Price:</strong> ${predictions['current_price']:.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        pred_df = pd.DataFrame({
                            'Date': predictions['dates'],
                            'Predicted Price': [f"${p:.2f}" for p in predictions['predictions']],
                            'Change from Current': [f"{((p - predictions['current_price']) / predictions['current_price'] * 100):+.2f}%" 
                                                  for p in predictions['predictions']]
                        })
                        st.dataframe(pred_df, use_container_width=True)
                        
                        # ML Accuracy Tracking
                        st.subheader("📊 ML Prediction Accuracy Tracking")
                        accuracy_metrics = accuracy_tracker.calculate_accuracy_metrics(symbol)
                        
                        if accuracy_metrics['total_predictions'] > 0:
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Predictions", accuracy_metrics['total_predictions'])
                            with col2:
                                st.metric("Overall Accuracy", f"{accuracy_metrics['accuracy_percentage']:.1f}%")
                            with col3:
                                st.metric("Recent Accuracy", f"{accuracy_metrics['recent_accuracy']:.1f}%")
                            with col4:
                                st.metric("MAE", f"${accuracy_metrics['mae']:.2f}")
                        else:
                            st.info("No previous predictions found. Accuracy tracking will start after more predictions are made.")
                    else:
                        st.error(f"Prediction failed: {error}")
                    
                    # Enhanced Chart
                    st.subheader("📈 Enhanced Technical Analysis Chart")
                    fig = create_enhanced_chart(data, symbol, current_theme)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.success(f"Enhanced analysis completed successfully for {symbol}")
                else:
                    st.error(f"No data available for {symbol}")

def ml_accuracy_tracking_page():
    """ML accuracy tracking and analysis"""
    st.header("📊 ML Accuracy Tracking")
    
    # Symbol selection
    symbol = st.text_input("Enter Stock Symbol", value=preferences.get('default_symbol', 'AAPL'), 
                          placeholder="e.g., AAPL, MSFT, GOOGL", key="accuracy_symbol")
    
    if st.button("View Accuracy Metrics", type="primary"):
        if symbol:
            with st.spinner(f"Calculating accuracy metrics for {symbol}..."):
                accuracy_metrics = accuracy_tracker.calculate_accuracy_metrics(symbol)
                
                if accuracy_metrics['total_predictions'] > 0:
                    st.subheader(f"📊 Accuracy Metrics for {symbol}")
                    
                    # Overall metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Predictions", accuracy_metrics['total_predictions'])
                    with col2:
                        st.metric("Overall Accuracy", f"{accuracy_metrics['accuracy_percentage']:.1f}%")
                    with col3:
                        st.metric("Recent Accuracy", f"{accuracy_metrics['recent_accuracy']:.1f}%")
                    with col4:
                        st.metric("MAE", f"${accuracy_metrics['mae']:.2f}")
                    
                    # Additional metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("RMSE", f"${accuracy_metrics['rmse']:.2f}")
                    with col2:
                        accuracy_color = "🟢" if accuracy_metrics['accuracy_percentage'] > 70 else "🟡" if accuracy_metrics['accuracy_percentage'] > 50 else "🔴"
                        st.metric("Performance", f"{accuracy_color} {accuracy_metrics['accuracy_percentage']:.1f}%")
                    
                    # Accuracy trend chart
                    st.subheader("📈 Accuracy Trend Over Time")
                    st.info("Accuracy tracking data will be displayed here as more predictions are made.")
                    
                else:
                    st.info(f"No prediction history found for {symbol}. Make some predictions first to see accuracy metrics.")
    
    # Global accuracy overview
    st.subheader("🌍 Global ML Performance Overview")
    st.info("Global accuracy metrics will be displayed here as the system collects more prediction data.")

def ml_stock_analysis_page():
    """ML-powered stock analysis (legacy)"""
    st.header("📈 ML Stock Analysis")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        symbol = st.text_input("Enter Stock Symbol", value="AAPL", placeholder="e.g., AAPL, MSFT, GOOGL")
    with col2:
        timeframe = st.selectbox("Timeframe", ["1mo", "3mo", "6mo", "1y", "2y", "5y"])
    
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
                    
                    # Technical Indicators
                    st.subheader("📊 Technical Indicators")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        if 'RSI' in data.columns:
                            rsi = data['RSI'].iloc[-1]
                            st.metric("RSI", f"{rsi:.1f}")
                    
                    with col2:
                        if 'MACD' in data.columns:
                            macd = data['MACD'].iloc[-1]
                            st.metric("MACD", f"{macd:.3f}")
                    
                    with col3:
                        if 'BB_Position' in data.columns:
                            bb_pos = data['BB_Position'].iloc[-1]
                            st.metric("BB Position", f"{bb_pos:.2f}")
                    
                    with col4:
                        if 'Stoch_K' in data.columns:
                            stoch = data['Stoch_K'].iloc[-1]
                            st.metric("Stochastic %K", f"{stoch:.1f}")
                    
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
                                
                                # Show anomaly dates
                                anomaly_dates = data.index[anomalies]
                                anomaly_prices = data['Close'][anomalies]
                                
                                anomaly_df = pd.DataFrame({
                                    'Date': anomaly_dates,
                                    'Price': anomaly_prices,
                                    'Z-Score': z_scores[anomalies]
                                })
                                st.dataframe(anomaly_df, use_container_width=True)
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

def technical_charts_page():
    """Technical charts and analysis"""
    st.header("📈 Technical Charts")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        symbol = st.text_input("Enter Stock Symbol", value="AAPL", placeholder="e.g., AAPL, MSFT, GOOGL", key="chart_symbol")
    with col2:
        timeframe = st.selectbox("Timeframe", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], key="chart_timeframe")
    
    if st.button("Generate Chart", type="primary"):
        if symbol:
            with st.spinner(f"Generating chart for {symbol}..."):
                data = get_market_data(symbol, timeframe)
                
                if data is not None and not data.empty:
                    data = calculate_technical_indicators(data)
                    
                    # Create candlestick chart
                    fig = create_candlestick_chart(data, symbol)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Additional technical analysis
                    st.subheader("📊 Technical Analysis Summary")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        if 'RSI' in data.columns:
                            rsi = data['RSI'].iloc[-1]
                            rsi_signal = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
                            st.metric("RSI", f"{rsi:.1f}", rsi_signal)
                    
                    with col2:
                        if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
                            macd = data['MACD'].iloc[-1]
                            macd_signal = data['MACD_Signal'].iloc[-1]
                            macd_trend = "Bullish" if macd > macd_signal else "Bearish"
                            st.metric("MACD", f"{macd:.3f}", macd_trend)
                    
                    with col3:
                        if 'BB_Position' in data.columns:
                            bb_pos = data['BB_Position'].iloc[-1]
                            bb_signal = "Near Upper" if bb_pos > 0.8 else "Near Lower" if bb_pos < 0.2 else "Middle"
                            st.metric("BB Position", f"{bb_pos:.2f}", bb_signal)
                    
                    with col4:
                        if 'Stoch_K' in data.columns:
                            stoch = data['Stoch_K'].iloc[-1]
                            stoch_signal = "Overbought" if stoch > 80 else "Oversold" if stoch < 20 else "Neutral"
                            st.metric("Stochastic", f"{stoch:.1f}", stoch_signal)

if __name__ == "__main__":
    main()
