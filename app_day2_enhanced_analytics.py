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
    page_title="Financial Analyzer Pro - Enhanced Analytics",
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
    .trend-up {
        color: #28a745;
        font-weight: bold;
    }
    .trend-down {
        color: #dc3545;
        font-weight: bold;
    }
    .trend-neutral {
        color: #6c757d;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

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
    """Calculate enhanced technical indicators for Day 2"""
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
        
        # On-Balance Volume (OBV)
        data['OBV'] = (data['Volume'] * np.sign(data['Close'].diff())).cumsum()
        
        # Volume Price Trend (VPT)
        data['VPT'] = (data['Volume'] * (data['Close'] / data['Close'].shift() - 1)).cumsum()
        
        return data
    except Exception as e:
        st.error(f"Error calculating technical indicators: {str(e)}")
        return data

def calculate_trend_analysis(data):
    """Calculate trend analysis indicators"""
    if data.empty:
        return {}
    
    try:
        # Trend Direction
        sma_20 = data['SMA_20'].iloc[-1] if 'SMA_20' in data.columns else data['Close'].iloc[-1]
        sma_50 = data['SMA_50'].iloc[-1] if 'SMA_50' in data.columns else data['Close'].iloc[-1]
        current_price = data['Close'].iloc[-1]
        
        # Short-term trend (5 vs 20 SMA)
        sma_5 = data['SMA_5'].iloc[-1] if 'SMA_5' in data.columns else current_price
        short_trend = "Up" if sma_5 > sma_20 else "Down" if sma_5 < sma_20 else "Sideways"
        
        # Medium-term trend (20 vs 50 SMA)
        medium_trend = "Up" if sma_20 > sma_50 else "Down" if sma_20 < sma_50 else "Sideways"
        
        # Long-term trend (50 vs 200 SMA)
        sma_200 = data['SMA_200'].iloc[-1] if 'SMA_200' in data.columns else sma_50
        long_trend = "Up" if sma_50 > sma_200 else "Down" if sma_50 < sma_200 else "Sideways"
        
        # Trend Strength
        price_change_5d = ((current_price - data['Close'].iloc[-5]) / data['Close'].iloc[-5] * 100) if len(data) >= 5 else 0
        price_change_20d = ((current_price - data['Close'].iloc[-20]) / data['Close'].iloc[-20] * 100) if len(data) >= 20 else 0
        
        # Support and Resistance
        recent_high = data['High'].rolling(window=20).max().iloc[-1]
        recent_low = data['Low'].rolling(window=20).min().iloc[-1]
        
        support_level = recent_low
        resistance_level = recent_high
        
        # Distance from support/resistance
        distance_to_support = ((current_price - support_level) / support_level * 100)
        distance_to_resistance = ((resistance_level - current_price) / current_price * 100)
        
        return {
            'short_trend': short_trend,
            'medium_trend': medium_trend,
            'long_trend': long_trend,
            'trend_strength_5d': price_change_5d,
            'trend_strength_20d': price_change_20d,
            'support_level': support_level,
            'resistance_level': resistance_level,
            'distance_to_support': distance_to_support,
            'distance_to_resistance': distance_to_resistance
        }
    except Exception as e:
        st.error(f"Error calculating trend analysis: {str(e)}")
        return {}

def calculate_volume_analysis(data):
    """Calculate volume analysis indicators"""
    if data.empty:
        return {}
    
    try:
        current_volume = data['Volume'].iloc[-1]
        avg_volume_20 = data['Volume'].rolling(window=20).mean().iloc[-1]
        volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1
        
        # Volume trend
        volume_trend_5d = data['Volume'].rolling(window=5).mean().iloc[-1]
        volume_trend_20d = data['Volume'].rolling(window=20).mean().iloc[-1]
        volume_trend = "Increasing" if volume_trend_5d > volume_trend_20d else "Decreasing"
        
        # Price-Volume relationship
        price_change = data['Close'].pct_change().iloc[-1]
        volume_change = data['Volume'].pct_change().iloc[-1]
        
        if price_change > 0 and volume_change > 0:
            pv_relationship = "Bullish"
        elif price_change < 0 and volume_change > 0:
            pv_relationship = "Bearish"
        elif price_change > 0 and volume_change < 0:
            pv_relationship = "Weak Bullish"
        elif price_change < 0 and volume_change < 0:
            pv_relationship = "Weak Bearish"
        else:
            pv_relationship = "Neutral"
        
        # OBV trend
        obv_trend = "Up" if data['OBV'].iloc[-1] > data['OBV'].iloc[-5] else "Down" if len(data) >= 5 else "Neutral"
        
        return {
            'current_volume': current_volume,
            'avg_volume_20': avg_volume_20,
            'volume_ratio': volume_ratio,
            'volume_trend': volume_trend,
            'pv_relationship': pv_relationship,
            'obv_trend': obv_trend
        }
    except Exception as e:
        st.error(f"Error calculating volume analysis: {str(e)}")
        return {}

def calculate_sentiment_indicators(data):
    """Calculate market sentiment indicators"""
    if data.empty:
        return {}
    
    try:
        # Fear & Greed Index simulation
        rsi = data['RSI'].iloc[-1] if 'RSI' in data.columns else 50
        bb_position = data['BB_Position'].iloc[-1] if 'BB_Position' in data.columns else 0.5
        
        # RSI-based sentiment
        if rsi > 70:
            rsi_sentiment = "Extreme Greed"
        elif rsi > 60:
            rsi_sentiment = "Greed"
        elif rsi > 40:
            rsi_sentiment = "Neutral"
        elif rsi > 30:
            rsi_sentiment = "Fear"
        else:
            rsi_sentiment = "Extreme Fear"
        
        # Bollinger Bands sentiment
        if bb_position > 0.8:
            bb_sentiment = "Overbought"
        elif bb_position < 0.2:
            bb_sentiment = "Oversold"
        else:
            bb_sentiment = "Neutral"
        
        # Volatility analysis
        returns = data['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility
        
        if volatility > 30:
            volatility_level = "High"
        elif volatility > 15:
            volatility_level = "Medium"
        else:
            volatility_level = "Low"
        
        # Momentum
        momentum_5d = ((data['Close'].iloc[-1] - data['Close'].iloc[-5]) / data['Close'].iloc[-5] * 100) if len(data) >= 5 else 0
        momentum_20d = ((data['Close'].iloc[-1] - data['Close'].iloc[-20]) / data['Close'].iloc[-20] * 100) if len(data) >= 20 else 0
        
        # Overall sentiment score (0-100)
        sentiment_score = 50  # Start neutral
        
        # Adjust based on RSI
        if rsi > 50:
            sentiment_score += (rsi - 50) * 0.5
        else:
            sentiment_score -= (50 - rsi) * 0.5
        
        # Adjust based on BB position
        sentiment_score += (bb_position - 0.5) * 20
        
        # Adjust based on momentum
        sentiment_score += momentum_5d * 0.1
        
        sentiment_score = max(0, min(100, sentiment_score))
        
        if sentiment_score > 70:
            overall_sentiment = "Bullish"
        elif sentiment_score > 30:
            overall_sentiment = "Neutral"
        else:
            overall_sentiment = "Bearish"
        
        return {
            'rsi_sentiment': rsi_sentiment,
            'bb_sentiment': bb_sentiment,
            'volatility_level': volatility_level,
            'volatility': volatility,
            'momentum_5d': momentum_5d,
            'momentum_20d': momentum_20d,
            'sentiment_score': sentiment_score,
            'overall_sentiment': overall_sentiment
        }
    except Exception as e:
        st.error(f"Error calculating sentiment indicators: {str(e)}")
        return {}

def predict_price_ml(data, symbol, periods=5):
    """Enhanced ML prediction with robust error handling"""
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
            # Try with just Close and Volume
            basic_available = [f for f in basic_features if f in data.columns]
            if len(basic_available) >= 2:
                available_features = basic_available
            else:
                return None, "Insufficient features for prediction"
        
        # Create ML dataset
        df_ml = data[available_features].copy()
        
        # Fill any remaining NaN values with forward fill, then backward fill
        df_ml = df_ml.fillna(method='ffill').fillna(method='bfill')
        
        # Remove any rows that still have NaN values
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
            
            # Update features for next prediction (simple approach)
            if len(last_features[0]) > 0 and 'Volume' in feature_cols:
                # Update volume with a simple trend
                vol_idx = feature_cols.index('Volume') if 'Volume' in feature_cols else 0
                last_features[0][vol_idx] = last_features[0][vol_idx] * 0.99  # Slight decrease
        
        # Create prediction dates
        last_date = data.index[-1]
        prediction_dates = [last_date + timedelta(days=i+1) for i in range(periods)]
        
        # Calculate R² score safely
        try:
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred) if len(y) > 1 else 0
        except:
            r2 = 0
        
        return {
            'predictions': future_prices,
            'dates': prediction_dates,
            'current_price': current_price,
            'model_type': 'Enhanced Linear Regression',
            'features_used': len(feature_cols),
            'r2_score': r2
        }, None
        
    except Exception as e:
        return None, f"Prediction error: {str(e)}"

def create_enhanced_chart(data, symbol):
    """Create enhanced chart with multiple indicators"""
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
        
        if 'MACD_Histogram' in data.columns:
            fig.add_trace(go.Bar(
                x=data.index,
                y=data['MACD_Histogram'],
                name='MACD Histogram',
                marker_color='gray'
            ), row=3, col=1)
    
    # Volume
    fig.add_trace(go.Bar(
        x=data.index,
        y=data['Volume'],
        name='Volume',
        marker_color='lightblue'
    ), row=4, col=1)
    
    # Update layout
    fig.update_layout(
        title=f'{symbol} - Enhanced Technical Analysis',
        height=800,
        showlegend=True,
        hovermode='x unified'
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
        <h1>📊 Financial Analyzer Pro - Enhanced Analytics</h1>
        <p>Advanced Technical Analysis with Trend, Volume & Sentiment Indicators</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Status
    st.markdown("""
    <div class="success-message">
        <h4>🚀 Day 2: Enhanced Analytics</h4>
        <p>✅ Advanced Technical Indicators | ✅ Trend Analysis | ✅ Volume Analysis | ✅ Market Sentiment | ✅ Enhanced Visualizations</p>
    </div>
    """, unsafe_allow_html=True)
    
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
            with st.spinner(f"Analyzing {symbol} with enhanced analytics..."):
                # Get data
                data = get_market_data(symbol, timeframe)
                
                if data is not None and not data.empty:
                    # Calculate enhanced indicators
                    data = calculate_enhanced_technical_indicators(data)
                    
                    # Calculate analysis
                    trend_analysis = calculate_trend_analysis(data)
                    volume_analysis = calculate_volume_analysis(data)
                    sentiment_indicators = calculate_sentiment_indicators(data)
                    
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
                    
                    # Enhanced Technical Indicators
                    st.subheader("📊 Enhanced Technical Indicators")
                    
                    # Create columns for indicators
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
                        
                        if 'Williams_R' in data.columns:
                            williams = data['Williams_R'].iloc[-1]
                            st.metric("Williams %R", f"{williams:.1f}")
                    
                    with col2:
                        st.markdown("**Trend Indicators**")
                        if 'SMA_20' in data.columns:
                            sma20 = data['SMA_20'].iloc[-1]
                            st.metric("SMA 20", f"${sma20:.2f}")
                        
                        if 'SMA_50' in data.columns:
                            sma50 = data['SMA_50'].iloc[-1]
                            st.metric("SMA 50", f"${sma50:.2f}")
                        
                        if 'MACD' in data.columns:
                            macd = data['MACD'].iloc[-1]
                            st.metric("MACD", f"{macd:.3f}")
                    
                    with col3:
                        st.markdown("**Volatility Indicators**")
                        if 'BB_Position' in data.columns:
                            bb_pos = data['BB_Position'].iloc[-1]
                            bb_status = "Upper Band" if bb_pos > 0.8 else "Lower Band" if bb_pos < 0.2 else "Middle"
                            st.metric("BB Position", f"{bb_pos:.2f}", help=f"Status: {bb_status}")
                        
                        if 'ATR' in data.columns:
                            atr = data['ATR'].iloc[-1]
                            st.metric("ATR", f"{atr:.2f}")
                        
                        if 'CCI' in data.columns:
                            cci = data['CCI'].iloc[-1]
                            st.metric("CCI", f"{cci:.1f}")
                    
                    with col4:
                        st.markdown("**Volume Indicators**")
                        if 'Volume_Ratio' in data.columns:
                            vol_ratio = data['Volume_Ratio'].iloc[-1]
                            st.metric("Volume Ratio", f"{vol_ratio:.2f}")
                        
                        if 'OBV' in data.columns:
                            obv = data['OBV'].iloc[-1]
                            st.metric("OBV", f"{obv:,.0f}")
                        
                        if 'VPT' in data.columns:
                            vpt = data['VPT'].iloc[-1]
                            st.metric("VPT", f"{vpt:,.0f}")
                    
                    # Trend Analysis
                    if trend_analysis:
                        st.subheader("📈 Trend Analysis")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("**Trend Direction**")
                            short_trend = trend_analysis.get('short_trend', 'Unknown')
                            medium_trend = trend_analysis.get('medium_trend', 'Unknown')
                            long_trend = trend_analysis.get('long_trend', 'Unknown')
                            
                            trend_class = "trend-up" if short_trend == "Up" else "trend-down" if short_trend == "Down" else "trend-neutral"
                            st.markdown(f"<div class='indicator-card'><strong>Short-term:</strong> <span class='{trend_class}'>{short_trend}</span></div>", unsafe_allow_html=True)
                            
                            trend_class = "trend-up" if medium_trend == "Up" else "trend-down" if medium_trend == "Down" else "trend-neutral"
                            st.markdown(f"<div class='indicator-card'><strong>Medium-term:</strong> <span class='{trend_class}'>{medium_trend}</span></div>", unsafe_allow_html=True)
                            
                            trend_class = "trend-up" if long_trend == "Up" else "trend-down" if long_trend == "Down" else "trend-neutral"
                            st.markdown(f"<div class='indicator-card'><strong>Long-term:</strong> <span class='{trend_class}'>{long_trend}</span></div>", unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("**Support & Resistance**")
                            support = trend_analysis.get('support_level', 0)
                            resistance = trend_analysis.get('resistance_level', 0)
                            dist_support = trend_analysis.get('distance_to_support', 0)
                            dist_resistance = trend_analysis.get('distance_to_resistance', 0)
                            
                            st.metric("Support Level", f"${support:.2f}")
                            st.metric("Resistance Level", f"${resistance:.2f}")
                            st.metric("Distance to Support", f"{dist_support:.1f}%")
                            st.metric("Distance to Resistance", f"{dist_resistance:.1f}%")
                        
                        with col3:
                            st.markdown("**Trend Strength**")
                            strength_5d = trend_analysis.get('trend_strength_5d', 0)
                            strength_20d = trend_analysis.get('trend_strength_20d', 0)
                            
                            st.metric("5-Day Change", f"{strength_5d:.2f}%")
                            st.metric("20-Day Change", f"{strength_20d:.2f}%")
                    
                    # Volume Analysis
                    if volume_analysis:
                        st.subheader("📊 Volume Analysis")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("**Volume Metrics**")
                            current_vol = volume_analysis.get('current_volume', 0)
                            avg_vol = volume_analysis.get('avg_volume_20', 0)
                            vol_ratio = volume_analysis.get('volume_ratio', 1)
                            
                            st.metric("Current Volume", f"{current_vol:,}")
                            st.metric("20-Day Average", f"{avg_vol:,}")
                            st.metric("Volume Ratio", f"{vol_ratio:.2f}")
                        
                        with col2:
                            st.markdown("**Volume Trends**")
                            vol_trend = volume_analysis.get('volume_trend', 'Unknown')
                            pv_relationship = volume_analysis.get('pv_relationship', 'Unknown')
                            obv_trend = volume_analysis.get('obv_trend', 'Unknown')
                            
                            st.markdown(f"<div class='indicator-card'><strong>Volume Trend:</strong> {vol_trend}</div>", unsafe_allow_html=True)
                            st.markdown(f"<div class='indicator-card'><strong>Price-Volume:</strong> {pv_relationship}</div>", unsafe_allow_html=True)
                            st.markdown(f"<div class='indicator-card'><strong>OBV Trend:</strong> {obv_trend}</div>", unsafe_allow_html=True)
                    
                    # Sentiment Analysis
                    if sentiment_indicators:
                        st.subheader("😊 Market Sentiment")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("**Sentiment Indicators**")
                            rsi_sentiment = sentiment_indicators.get('rsi_sentiment', 'Unknown')
                            bb_sentiment = sentiment_indicators.get('bb_sentiment', 'Unknown')
                            overall_sentiment = sentiment_indicators.get('overall_sentiment', 'Unknown')
                            
                            st.markdown(f"<div class='indicator-card'><strong>RSI Sentiment:</strong> {rsi_sentiment}</div>", unsafe_allow_html=True)
                            st.markdown(f"<div class='indicator-card'><strong>BB Sentiment:</strong> {bb_sentiment}</div>", unsafe_allow_html=True)
                            
                            sentiment_class = "trend-up" if overall_sentiment == "Bullish" else "trend-down" if overall_sentiment == "Bearish" else "trend-neutral"
                            st.markdown(f"<div class='indicator-card'><strong>Overall:</strong> <span class='{sentiment_class}'>{overall_sentiment}</span></div>", unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("**Volatility & Momentum**")
                            volatility = sentiment_indicators.get('volatility', 0)
                            volatility_level = sentiment_indicators.get('volatility_level', 'Unknown')
                            momentum_5d = sentiment_indicators.get('momentum_5d', 0)
                            
                            st.metric("Volatility", f"{volatility:.1f}%")
                            st.metric("Volatility Level", volatility_level)
                            st.metric("5-Day Momentum", f"{momentum_5d:.2f}%")
                        
                        with col3:
                            st.markdown("**Sentiment Score**")
                            sentiment_score = sentiment_indicators.get('sentiment_score', 50)
                            
                            # Create a progress bar for sentiment score
                            st.progress(sentiment_score / 100)
                            st.metric("Sentiment Score", f"{sentiment_score:.0f}/100")
                    
                    # ML Predictions
                    st.subheader("🤖 Enhanced ML Price Predictions")
                    predictions, error = predict_price_ml(data, symbol, periods=5)
                    
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
                        
                        pred_df = pd.DataFrame({
                            'Date': predictions['dates'],
                            'Predicted Price': [f"${p:.2f}" for p in predictions['predictions']],
                            'Change from Current': [f"{((p - predictions['current_price']) / predictions['current_price'] * 100):+.2f}%" 
                                                  for p in predictions['predictions']]
                        })
                        st.dataframe(pred_df, use_container_width=True)
                    else:
                        st.error(f"Prediction failed: {error}")
                    
                    # Enhanced Chart
                    st.subheader("📈 Enhanced Technical Analysis Chart")
                    fig = create_enhanced_chart(data, symbol)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.success(f"Enhanced analysis completed successfully for {symbol}")
                else:
                    st.error(f"No data available for {symbol}")

if __name__ == "__main__":
    main()
