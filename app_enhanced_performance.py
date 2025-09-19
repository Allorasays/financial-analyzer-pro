import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import warnings
import time
import os
import sqlite3
import hashlib
import logging
from typing import Optional, Dict, Any, List
import threading
from functools import wraps

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
    page_title="Financial Analyzer Pro Enhanced",
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
    .prediction-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
    }
    .warning-message {
        background: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #ffeaa7;
        margin: 1rem 0;
    }
    .info-message {
        background: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #bee5eb;
        margin: 1rem 0;
    }
    .performance-metric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem;
    }
    .loading-container {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced Smart Cache with TTL and Memory Management
class SmartCache:
    def __init__(self, max_size: int = 200, default_ttl: int = 300):
        self.cache = {}
        self.timestamps = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.lock = threading.Lock()
        self.hit_count = 0
        self.miss_count = 0
    
    def _is_expired(self, key: str) -> bool:
        if key not in self.timestamps:
            return True
        return time.time() - self.timestamps[key] > self.default_ttl
    
    def _evict_oldest(self):
        if not self.cache:
            return
        oldest_key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
        del self.cache[oldest_key]
        del self.timestamps[oldest_key]
    
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key in self.cache and not self._is_expired(key):
                self.hit_count += 1
                return self.cache[key]
            else:
                if key in self.cache:
                    del self.cache[key]
                    del self.timestamps[key]
                self.miss_count += 1
                return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        with self.lock:
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_oldest()
            
            self.cache[key] = value
            self.timestamps[key] = time.time()
            if ttl:
                self.default_ttl = ttl
    
    def clear(self) -> None:
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
            self.hit_count = 0
            self.miss_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            total_requests = self.hit_count + self.miss_count
            hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_count': self.hit_count,
                'miss_count': self.miss_count,
                'hit_rate': hit_rate,
                'ttl': self.default_ttl
            }

# Error Recovery System
class ErrorRecovery:
    @staticmethod
    def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None
                current_delay = delay
                
                for attempt in range(max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        if attempt < max_retries:
                            st.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {current_delay:.1f}s...")
                            time.sleep(current_delay)
                            current_delay *= backoff
                        else:
                            st.error(f"All {max_retries + 1} attempts failed. Last error: {str(e)}")
                
                return None
            return wrapper
        return decorator
    
    @staticmethod
    def fallback_data(symbol: str, period: str = "1mo") -> pd.DataFrame:
        """Generate realistic fallback data when APIs fail"""
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
        
        return pd.DataFrame({
            'Open': [p * (1 + np.random.normal(0, 0.01)) for p in prices],
            'High': [p * (1 + abs(np.random.normal(0, 0.02))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.02))) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)

# Prediction Accuracy Tracking System
class PredictionTracker:
    def __init__(self, db_path: str = "predictions.db"):
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
                accuracy_score REAL,
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
                last_updated TEXT NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_prediction(self, symbol: str, prediction_data: Dict, horizon: int) -> str:
        prediction_id = hashlib.md5(f"{symbol}_{datetime.now()}_{horizon}".encode()).hexdigest()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions 
            (id, symbol, prediction_date, predicted_price, prediction_horizon, model_type, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            prediction_id,
            symbol,
            prediction_data['dates'][0].strftime('%Y-%m-%d'),
            prediction_data['predictions'][0],
            horizon,
            prediction_data['model_type'],
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        return prediction_id
    
    def update_actual_price(self, symbol: str, date: str, actual_price: float):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE predictions 
            SET actual_price = ?, updated_at = ?
            WHERE symbol = ? AND prediction_date = ?
        ''', (actual_price, datetime.now().isoformat(), symbol, date))
        
        conn.commit()
        conn.close()
    
    def calculate_accuracy_metrics(self, symbol: str, model_type: str = None) -> Dict:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = '''
            SELECT predicted_price, actual_price, model_type
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
                'directional_accuracy': 0.0
            }
        
        predicted_prices = [r[0] for r in results]
        actual_prices = [r[1] for r in results]
        
        # Calculate metrics
        total_predictions = len(results)
        correct_predictions = sum(1 for p, a in zip(predicted_prices, actual_prices) 
                                if abs(p - a) / a < 0.05)  # Within 5%
        
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
        
        return {
            'total_predictions': total_predictions,
            'accuracy_percentage': (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0,
            'mae': mae,
            'rmse': rmse,
            'directional_accuracy': directional_accuracy
        }
    
    def get_recent_predictions(self, symbol: str, limit: int = 10) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT prediction_date, predicted_price, actual_price, model_type, accuracy_score
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
                'accuracy': r[4]
            }
            for r in results
        ]

# Global instances
cache = SmartCache(max_size=200, default_ttl=300)
prediction_tracker = PredictionTracker()

# Enhanced data fetching with error recovery
@ErrorRecovery.retry_on_failure(max_retries=3, delay=1.0, backoff=2.0)
def get_market_data(symbol: str, period: str = "1mo"):
    """Get market data with smart caching and robust fallback"""
    cache_key = f"market_data_{symbol}_{period}"
    cached_data = cache.get(cache_key)
    
    if cached_data is not None:
        return cached_data
    
    # Try yfinance with timeout
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, timeout=10)
        
        if data is not None and not data.empty:
            cache.set(cache_key, data, ttl=300)  # 5 minutes
            return data
    except Exception as e:
        st.warning(f"Yahoo Finance API failed for {symbol}: {str(e)}")
    
    # Fallback to demo data
    st.info(f"Using demo data for {symbol} (API unavailable)")
    data = ErrorRecovery.fallback_data(symbol, period)
    cache.set(cache_key, data, ttl=120)  # 2 minutes for demo data
    return data

def calculate_technical_indicators(data):
    """Calculate enhanced technical indicators"""
    if data.empty:
        return data
    
    try:
        data = data.copy()
        
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
        data['Stoch_K'] = 100 * (data['Close'] - low_14) / (high_14 - low_14)
        data['Stoch_D'] = data['Stoch_K'].rolling(window=3).mean()
        
        # Volume indicators
        data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
        
        return data
    except Exception as e:
        st.error(f"Error calculating technical indicators: {str(e)}")
        return data

def predict_price_ml(data, symbol, periods=5):
    """Enhanced ML prediction with accuracy tracking"""
    if not SKLEARN_AVAILABLE:
        return None, "ML library not available"
    
    try:
        # Enhanced features
        features = ['Close', 'Volume', 'RSI', 'SMA_20', 'MACD', 'BB_Position']
        available_features = [f for f in features if f in data.columns]
        
        if len(available_features) < 2:
            return None, "Insufficient features for prediction"
        
        # Prepare data
        df_ml = data[available_features].dropna()
        if len(df_ml) < 20:
            return None, "Insufficient data for prediction"
        
        # Create target
        df_ml['Target'] = df_ml['Close'].shift(-periods)
        df_ml = df_ml.dropna()
        
        if len(df_ml) < 10:
            return None, "Insufficient data after creating target"
        
        # Features and target
        feature_cols = [col for col in available_features if col != 'Close']
        X = df_ml[feature_cols]
        y = df_ml['Target']
        
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
            if len(last_features[0]) > 0:
                last_features[0][0] = pred_price
        
        # Create prediction dates
        last_date = data.index[-1]
        prediction_dates = [last_date + timedelta(days=i+1) for i in range(periods)]
        
        prediction_data = {
            'predictions': future_prices,
            'dates': prediction_dates,
            'current_price': current_price,
            'model_type': 'Enhanced Linear Regression',
            'features_used': len(feature_cols),
            'r2_score': r2_score(y, model.predict(X)) if len(y) > 1 else 0
        }
        
        # Store prediction for accuracy tracking
        prediction_tracker.store_prediction(symbol, prediction_data, periods)
        
        return prediction_data, None
        
    except Exception as e:
        return None, f"Prediction error: {str(e)}"

def create_enhanced_candlestick_chart(data, symbol):
    """Create enhanced candlestick chart with technical indicators"""
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
        title=f'{symbol} - Enhanced Price Chart',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        height=600,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📊 Financial Analyzer Pro Enhanced</h1>
        <p>Advanced Financial Research & Analysis Platform with Performance Optimization</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Status
    st.markdown("""
    <div class="success-message">
        <h4>🚀 Enhanced Performance Version</h4>
        <p>✅ Smart Caching | ✅ Error Recovery | ✅ ML Analysis | ✅ Prediction Tracking | ✅ Loading States</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Performance metrics
    cache_stats = cache.get_stats()
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="performance-metric">
            <h4>Cache Hit Rate</h4>
            <h2>{cache_stats['hit_rate']:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="performance-metric">
            <h4>Cache Size</h4>
            <h2>{cache_stats['size']}/{cache_stats['max_size']}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="performance-metric">
            <h4>ML Status</h4>
            <h2>🟢 Available</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        if st.button("Clear Cache", type="secondary"):
            cache.clear()
            st.success("Cache cleared!")
            st.rerun()
    
    # Main interface
    st.sidebar.title("📊 Analysis Tools")
    
    # Stock analysis
    st.header("📈 Enhanced Stock Analysis")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        symbol = st.text_input("Enter Stock Symbol", value="AAPL", placeholder="e.g., AAPL, MSFT, GOOGL")
    with col2:
        timeframe = st.selectbox("Timeframe", ["1mo", "3mo", "6mo", "1y", "2y", "5y"])
    
    if st.button("Analyze Stock", type="primary"):
        if symbol:
            with st.spinner(f"Analyzing {symbol} with enhanced features..."):
                # Get data with loading indicator
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Fetching market data...")
                progress_bar.progress(20)
                data = get_market_data(symbol, timeframe)
                
                status_text.text("Calculating technical indicators...")
                progress_bar.progress(40)
                data = calculate_technical_indicators(data)
                
                status_text.text("Generating ML predictions...")
                progress_bar.progress(60)
                predictions, error = predict_price_ml(data, symbol, periods=5)
                
                status_text.text("Creating visualizations...")
                progress_bar.progress(80)
                
                if data is not None and not data.empty:
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
                    
                    # Enhanced Technical Indicators
                    st.subheader("📊 Enhanced Technical Indicators")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        if 'RSI' in data.columns:
                            rsi = data['RSI'].iloc[-1]
                            rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
                            st.metric("RSI", f"{rsi:.1f}", help=f"Status: {rsi_status}")
                    
                    with col2:
                        if 'MACD' in data.columns:
                            macd = data['MACD'].iloc[-1]
                            st.metric("MACD", f"{macd:.3f}")
                    
                    with col3:
                        if 'BB_Position' in data.columns:
                            bb_pos = data['BB_Position'].iloc[-1]
                            bb_status = "Upper Band" if bb_pos > 0.8 else "Lower Band" if bb_pos < 0.2 else "Middle"
                            st.metric("BB Position", f"{bb_pos:.2f}", help=f"Status: {bb_status}")
                    
                    with col4:
                        if 'Stoch_K' in data.columns:
                            stoch = data['Stoch_K'].iloc[-1]
                            st.metric("Stochastic %K", f"{stoch:.1f}")
                    
                    # ML Predictions with accuracy tracking
                    st.subheader("🤖 Enhanced ML Price Predictions")
                    if predictions:
                        st.markdown(f"""
                        <div class="prediction-card">
                            <h4>📈 Price Predictions (Next 5 Days)</h4>
                            <p><strong>Model:</strong> {predictions['model_type']}</p>
                            <p><strong>Features Used:</strong> {predictions['features_used']}</p>
                            <p><strong>R² Score:</strong> {predictions['r2_score']:.3f}</p>
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
                        
                        # Show accuracy metrics
                        accuracy_metrics = prediction_tracker.calculate_accuracy_metrics(symbol)
                        if accuracy_metrics['total_predictions'] > 0:
                            st.subheader("📊 Prediction Accuracy")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Predictions", accuracy_metrics['total_predictions'])
                            with col2:
                                st.metric("Accuracy %", f"{accuracy_metrics['accuracy_percentage']:.1f}%")
                            with col3:
                                st.metric("MAE", f"${accuracy_metrics['mae']:.2f}")
                            with col4:
                                st.metric("Directional Accuracy", f"{accuracy_metrics['directional_accuracy']:.1f}%")
                    else:
                        st.error(f"Prediction failed: {error}")
                    
                    # Enhanced Chart
                    st.subheader("📈 Enhanced Price Chart")
                    fig = create_enhanced_candlestick_chart(data, symbol)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    progress_bar.progress(100)
                    status_text.text("Analysis complete!")
                    
                    st.success(f"Enhanced analysis completed successfully for {symbol}")
                else:
                    st.error(f"No data available for {symbol}")

if __name__ == "__main__":
    main()