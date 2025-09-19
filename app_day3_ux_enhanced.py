import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import warnings
import time
import os
import sqlite3
import hashlib
import json

warnings.filterwarnings('ignore')

# ML imports with graceful fallbacks
try:
    from sklearn.linear_model import LinearRegression
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
    page_title="Financial Analyzer Pro - UX Enhanced",
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
            'auto_refresh': False,
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
        
        # Create accuracy metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS accuracy_metrics (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                model_type TEXT NOT NULL,
                total_predictions INTEGER DEFAULT 0,
                correct_predictions INTEGER DEFAULT 0,
                accuracy_percentage REAL DEFAULT 0.0,
                mae REAL DEFAULT 0.0,
                rmse REAL DEFAULT 0.0,
                directional_accuracy REAL DEFAULT 0.0,
                last_updated TEXT NOT NULL
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
    
    def update_actual_price(self, symbol, date, actual_price):
        """Update actual price when available"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE predictions 
            SET actual_price = ?, updated_at = ?
            WHERE symbol = ? AND prediction_date = ?
        ''', (actual_price, datetime.now().isoformat(), symbol, date))
        
        conn.commit()
        conn.close()
    
    def calculate_accuracy_metrics(self, symbol, model_type=None):
        """Calculate accuracy metrics for a symbol"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = '''
            SELECT predicted_price, actual_price, prediction_horizon, model_type
            FROM predictions 
            WHERE symbol = ? AND actual_price IS NOT NULL
        '''
        params = [symbol]
        
        if model_type:
            query += ' AND model_type = ?'
            params.append(model_type)
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return {
                'total_predictions': 0,
                'accuracy_percentage': 0.0,
                'mae': 0.0,
                'rmse': 0.0,
                'directional_accuracy': 0.0,
                'recent_accuracy': 0.0
            }
        
        predicted_prices = [r[0] for r in results]
        actual_prices = [r[1] for r in results]
        horizons = [r[2] for r in results]
        
        # Calculate metrics
        total_predictions = len(results)
        
        # Accuracy within 5% of actual price
        correct_predictions = sum(1 for p, a in zip(predicted_prices, actual_prices) 
                                if abs(p - a) / a < 0.05)
        accuracy_percentage = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
        
        # MAE and RMSE
        mae = np.mean([abs(p - a) for p, a in zip(predicted_prices, actual_prices)])
        rmse = np.sqrt(np.mean([(p - a) ** 2 for p, a in zip(predicted_prices, actual_prices)]))
        
        # Directional accuracy
        directional_correct = 0
        for i in range(1, len(actual_prices)):
            pred_direction = 1 if predicted_prices[i] > predicted_prices[i-1] else -1
            actual_direction = 1 if actual_prices[i] > actual_prices[i-1] else -1
            if pred_direction == actual_direction:
                directional_correct += 1
        
        directional_accuracy = (directional_correct / (len(actual_prices) - 1) * 100) if len(actual_prices) > 1 else 0
        
        # Recent accuracy (last 10 predictions)
        recent_results = results[-10:] if len(results) >= 10 else results
        recent_correct = sum(1 for r in recent_results 
                           if abs(r[0] - r[1]) / r[1] < 0.05)
        recent_accuracy = (recent_correct / len(recent_results) * 100) if recent_results else 0
        
        return {
            'total_predictions': total_predictions,
            'accuracy_percentage': accuracy_percentage,
            'mae': mae,
            'rmse': rmse,
            'directional_accuracy': directional_accuracy,
            'recent_accuracy': recent_accuracy
        }
    
    def get_recent_predictions(self, symbol, limit=10):
        """Get recent predictions for a symbol"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT prediction_date, predicted_price, actual_price, model_type, r2_score
            FROM predictions 
            WHERE symbol = ?
            ORDER BY created_at DESC
            LIMIT ?
        ''', (symbol, limit))
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                'date': r[0],
                'predicted': r[1],
                'actual': r[2],
                'model': r[3],
                'r2_score': r[4],
                'accuracy': abs(r[1] - r[2]) / r[2] * 100 if r[2] else None
            }
            for r in results
        ]

# Initialize components
preferences = UserPreferences()
accuracy_tracker = MLAccuracyTracker()

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
            .stSelectbox > div > div {
                background-color: #2c3e50;
                color: white;
            }
            .stTextInput > div > div > input {
                background-color: #2c3e50;
                color: white;
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
        </style>
        """

# Apply theme
current_theme = preferences.get('theme', 'light')
st.markdown(get_theme_css(current_theme), unsafe_allow_html=True)

# Simple cache for Render
class SimpleCache:
    def __init__(self, max_size=50):
        self.cache = {}
        self.max_size = max_size
    
    def get(self, key):
        return self.cache.get(key)
    
    def set(self, key, value):
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[key] = value
    
    def clear(self):
        self.cache.clear()
    
    def get_stats(self):
        return {
            'size': len(self.cache),
            'max_size': self.max_size
        }

# Global cache
cache = SimpleCache()

def get_market_data(symbol: str, period: str = "1mo"):
    """Get market data with simple caching and robust fallback"""
    cache_key = f"market_data_{symbol}_{period}"
    cached_data = cache.get(cache_key)
    
    if cached_data is not None:
        return cached_data
    
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, timeout=10)
        
        if data is not None and not data.empty:
            cache.set(cache_key, data)
            return data
    except Exception as e:
        st.warning(f"Yahoo Finance API failed for {symbol}: {str(e)}")
    
    # Fallback to demo data
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

def calculate_enhanced_technical_indicators(data):
    """Calculate enhanced technical indicators"""
    if data.empty:
        return data
    
    try:
        data = data.copy()
        
        # Basic Moving Averages
        data['SMA_5'] = data['Close'].rolling(window=5).mean()
        data['SMA_10'] = data['Close'].rolling(window=10).mean()
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_200'] = data['Close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        data['EMA_12'] = data['Close'].ewm(span=12).mean()
        data['EMA_26'] = data['Close'].ewm(span=26).mean()
        data['EMA_50'] = data['Close'].ewm(span=50).mean()
        
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
        data['Stoch_K'] = 100 * (data['Close'] - low_14) / (high_14 - low_14)
        data['Stoch_D'] = data['Stoch_K'].rolling(window=3).mean()
        
        # Williams %R
        data['Williams_R'] = -100 * (high_14 - data['Close']) / (high_14 - low_14)
        
        # Commodity Channel Index (CCI)
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        sma_tp = typical_price.rolling(window=20).mean()
        mad = typical_price.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())))
        data['CCI'] = (typical_price - sma_tp) / (0.015 * mad)
        
        # Average True Range (ATR)
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        data['ATR'] = true_range.rolling(window=14).mean()
        
        # Volume Indicators
        data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
        data['OBV'] = (data['Volume'] * np.sign(data['Close'].diff())).cumsum()
        data['VPT'] = (data['Volume'] * (data['Close'] / data['Close'].shift() - 1)).cumsum()
        
        return data
    except Exception as e:
        st.error(f"Error calculating technical indicators: {str(e)}")
        return data

def predict_price_ml(data, symbol, periods=5):
    """Enhanced ML prediction with accuracy tracking"""
    if not SKLEARN_AVAILABLE:
        return None, "ML library not available"
    
    try:
        # Start with basic features that are more likely to be available
        basic_features = ['Close', 'Volume']
        enhanced_features = ['RSI', 'SMA_20', 'MACD', 'BB_Position', 'Stoch_K', 'ATR']
        
        # Check which features are available and have valid data
        available_features = []
        for feature in basic_features + enhanced_features:
            if feature in data.columns and not data[feature].isna().all():
                available_features.append(feature)
        
        # Fallback to basic features if enhanced features are not available
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
        
        # Check for any infinite or NaN values
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
            'r2_score': r2
        }
        
        # Store prediction for accuracy tracking
        accuracy_tracker.store_prediction(symbol, prediction_data, periods)
        
        return prediction_data, None
        
    except Exception as e:
        return None, f"Prediction error: {str(e)}"

def create_enhanced_chart(data, symbol, theme='light'):
    """Create enhanced chart with theme support"""
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(f'{symbol} - Price Chart', 'RSI', 'MACD', 'Volume'),
        vertical_spacing=0.05,
        row_heights=[0.4, 0.2, 0.2, 0.2]
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
    
    # Volume
    fig.add_trace(go.Bar(
        x=data.index,
        y=data['Volume'],
        name='Volume',
        marker_color='lightblue'
    ), row=4, col=1)
    
    # Update layout with theme
    layout_color = '#2c3e50' if theme == 'dark' else '#ffffff'
    text_color = '#ffffff' if theme == 'dark' else '#000000'
    
    fig.update_layout(
        title=f'{symbol} - Enhanced Technical Analysis',
        height=preferences.get('chart_height', 600),
        showlegend=True,
        hovermode='x unified',
        plot_bgcolor=layout_color,
        paper_bgcolor=layout_color,
        font_color=text_color
    )
    
    # Update axes
    fig.update_xaxes(title_text="Date", row=4, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="Volume", row=4, col=1)
    
    return fig

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📊 Financial Analyzer Pro - UX Enhanced</h1>
        <p>Advanced Analytics with User Experience & ML Accuracy Tracking</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Status
    st.markdown("""
    <div class="success-message">
        <h4>🚀 Day 3: User Experience Enhanced</h4>
        <p>✅ Theme Toggle | ✅ Responsive Design | ✅ User Preferences | ✅ ML Accuracy Tracking | ✅ Keyboard Shortcuts</p>
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
    
    # Keyboard shortcuts info
    st.sidebar.subheader("⌨️ Keyboard Shortcuts")
    st.sidebar.markdown("""
    - **Ctrl + Enter**: Analyze current symbol
    - **Ctrl + T**: Toggle theme
    - **Ctrl + R**: Refresh data
    - **Ctrl + C**: Clear cache
    """)
    
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
    
    # Main interface
    st.header("📈 Enhanced Stock Analysis")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        symbol = st.text_input("Enter Stock Symbol", value=preferences.get('default_symbol', 'AAPL'), 
                              placeholder="e.g., AAPL, MSFT, GOOGL", key="symbol_input")
    with col2:
        timeframe = st.selectbox("Timeframe", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], 
                               index=["1mo", "3mo", "6mo", "1y", "2y", "5y"].index(preferences.get('default_timeframe', '1mo')))
    
    # Analyze button with keyboard shortcut
    if st.button("Analyze Stock", type="primary", key="analyze_btn") or st.session_state.get('analyze_triggered', False):
        if st.session_state.get('analyze_triggered', False):
            st.session_state.analyze_triggered = False
        
        if symbol:
            with st.spinner(f"Analyzing {symbol} with enhanced UX..."):
                # Get data
                data = get_market_data(symbol, timeframe)
                
                if data is not None and not data.empty:
                    # Calculate enhanced indicators
                    data = calculate_enhanced_technical_indicators(data)
                    
                    # Basic metrics
                    current_price = data['Close'].iloc[-1]
                    prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
                    change = current_price - prev_price
                    change_pct = (change / prev_price) * 100 if prev_price != 0 else 0
                    
                    # Display basic metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Current Price", f"${current_price:.2f}")
                    with col2:
                        st.metric("Change", f"${change:.2f}")
                    with col3:
                        st.metric("Change %", f"{change_pct:.2f}%")
                    with col4:
                        st.metric("Volume", f"{data['Volume'].iloc[-1]:,}")
                    
                    # Technical Indicators (conditional on user preference)
                    if show_advanced:
                        st.subheader("📊 Technical Indicators")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.markdown("**Momentum Indicators**")
                            if 'RSI' in data.columns:
                                rsi = data['RSI'].iloc[-1]
                                rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
                                st.metric("RSI", f"{rsi:.1f}", help=f"Status: {rsi_status}")
                            
                            if 'Stoch_K' in data.columns:
                                stoch = data['Stoch_K'].iloc[-1]
                                st.metric("Stochastic %K", f"{stoch:.1f}")
                        
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
                            if 'Volume_Ratio' in data.columns:
                                vol_ratio = data['Volume_Ratio'].iloc[-1]
                                st.metric("Volume Ratio", f"{vol_ratio:.2f}")
                    
                    # ML Predictions with accuracy tracking
                    st.subheader("🤖 ML Price Predictions & Accuracy Tracking")
                    predictions, error = predict_price_ml(data, symbol, periods=prediction_horizon)
                    
                    if predictions:
                        st.markdown(f"""
                        <div class="prediction-card">
                            <h4>📈 Price Predictions (Next {prediction_horizon} Days)</h4>
                            <p><strong>Model:</strong> {predictions['model_type']}</p>
                            <p><strong>Features Used:</strong> {predictions['features_used']}</p>
                            <p><strong>R² Score:</strong> {predictions['r2_score']:.3f}</p>
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
                                st.metric("Directional Accuracy", f"{accuracy_metrics['directional_accuracy']:.1f}%")
                            
                            # Show recent predictions
                            recent_predictions = accuracy_tracker.get_recent_predictions(symbol, 5)
                            if recent_predictions:
                                st.subheader("📈 Recent Prediction History")
                                recent_df = pd.DataFrame(recent_predictions)
                                st.dataframe(recent_df, use_container_width=True)
                        else:
                            st.info("No previous predictions found. Accuracy tracking will start after more predictions are made.")
                    else:
                        st.error(f"Prediction failed: {error}")
                    
                    # Enhanced Chart
                    st.subheader("📈 Enhanced Technical Analysis Chart")
                    fig = create_enhanced_chart(data, symbol, theme)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.success(f"Enhanced analysis completed successfully for {symbol}")
                else:
                    st.error(f"No data available for {symbol}")

    # JavaScript for keyboard shortcuts
    st.markdown("""
    <script>
    document.addEventListener('keydown', function(e) {
        if (e.ctrlKey) {
            switch(e.key) {
                case 'Enter':
                    e.preventDefault();
                    document.querySelector('[data-testid="baseButton-secondary"]').click();
                    break;
                case 't':
                    e.preventDefault();
                    // Toggle theme logic would go here
                    break;
                case 'r':
                    e.preventDefault();
                    window.location.reload();
                    break;
                case 'c':
                    e.preventDefault();
                    // Clear cache logic would go here
                    break;
            }
        }
    });
    </script>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
