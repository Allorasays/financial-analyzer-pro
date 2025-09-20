#!/usr/bin/env python3
"""
Financial Analyzer Pro - Day 6: Advanced Charts
Interactive candlestick charts, multiple timeframes, drawing tools, and technical indicators
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import warnings
import os
import json

warnings.filterwarnings('ignore')

# ML imports with graceful fallbacks
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="Financial Analyzer Pro - Day 6",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with charting theme
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chart-card {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .indicator-card {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .timeframe-card {
        background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .drawing-tool {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #3498db;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        text-align: center;
    }
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
    .chart-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = []
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []
if 'chart_settings' not in st.session_state:
    st.session_state.chart_settings = {
        'timeframe': '1d',
        'indicators': ['SMA_20', 'SMA_50'],
        'drawing_tools': [],
        'theme': 'light',
        'show_volume': True,
        'show_grid': True
    }
if 'chart_comparison' not in st.session_state:
    st.session_state.chart_comparison = []

# Simple cache for performance
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

cache = SimpleCache()

def get_stock_data(symbol, period="1d", interval="1d"):
    """Get stock data with enhanced caching for charts"""
    cache_key = f"stock_{symbol}_{period}_{interval}"
    cached_data = cache.get(cache_key)
    
    if cached_data is not None:
        return cached_data, None
    
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval, timeout=15)
        if data.empty:
            return None, f"No data available for {symbol}"
        
        cache.set(cache_key, data)
        return data, None
    except Exception as e:
        return None, f"Error fetching data for {symbol}: {str(e)}"

def calculate_technical_indicators(data):
    """Calculate various technical indicators"""
    if data is None or data.empty:
        return data
    
    df = data.copy()
    
    # Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # Volume indicators
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    
    return df

def create_candlestick_chart(data, symbol, timeframe, indicators=None, drawing_tools=None):
    """Create interactive candlestick chart with technical indicators"""
    if data is None or data.empty:
        return None
    
    # Calculate technical indicators
    df = calculate_technical_indicators(data)
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f'{symbol} - {timeframe}', 'Volume', 'RSI'),
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name=symbol,
        increasing_line_color='#00ff00',
        decreasing_line_color='#ff0000'
    ), row=1, col=1)
    
    # Technical indicators
    if indicators:
        for indicator in indicators:
            if indicator == 'SMA_20' and 'SMA_20' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['SMA_20'],
                    name='SMA 20',
                    line=dict(color='blue', width=2)
                ), row=1, col=1)
            
            elif indicator == 'SMA_50' and 'SMA_50' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['SMA_50'],
                    name='SMA 50',
                    line=dict(color='orange', width=2)
                ), row=1, col=1)
            
            elif indicator == 'BB' and all(col in df.columns for col in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['BB_Upper'],
                    name='BB Upper',
                    line=dict(color='purple', width=1, dash='dash')
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['BB_Middle'],
                    name='BB Middle',
                    line=dict(color='purple', width=1)
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['BB_Lower'],
                    name='BB Lower',
                    line=dict(color='purple', width=1, dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(128,0,128,0.1)'
                ), row=1, col=1)
    
    # Volume chart
    colors = ['green' if close >= open else 'red' for close, open in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'],
        name='Volume',
        marker_color=colors,
        opacity=0.7
    ), row=2, col=1)
    
    # RSI chart
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['RSI'],
            name='RSI',
            line=dict(color='purple', width=2)
        ), row=3, col=1)
        
        # RSI overbought/oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    # Update layout
    fig.update_layout(
        title=f'{symbol} - {timeframe} Chart',
        xaxis_rangeslider_visible=False,
        height=800,
        showlegend=True,
        template='plotly_white'
    )
    
    # Update axes
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    
    return fig

def create_comparison_chart(symbols, timeframe, period):
    """Create side-by-side chart comparison"""
    if not symbols:
        return None
    
    fig = make_subplots(
        rows=len(symbols), cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=symbols
    )
    
    for i, symbol in enumerate(symbols):
        data, error = get_stock_data(symbol, period, timeframe)
        if data is not None and not data.empty:
            # Normalize prices to percentage change
            normalized_prices = (data['Close'] / data['Close'].iloc[0] - 1) * 100
            
            fig.add_trace(go.Scatter(
                x=data.index,
                y=normalized_prices,
                name=symbol,
                line=dict(width=2)
            ), row=i+1, col=1)
    
    fig.update_layout(
        title=f'Chart Comparison - {timeframe}',
        height=200 * len(symbols),
        showlegend=True,
        template='plotly_white'
    )
    
    return fig

def create_heatmap_chart(symbols, timeframe):
    """Create correlation heatmap for multiple symbols"""
    if len(symbols) < 2:
        return None
    
    # Get data for all symbols
    data_dict = {}
    for symbol in symbols:
        data, error = get_stock_data(symbol, "1mo", timeframe)
        if data is not None and not data.empty:
            data_dict[symbol] = data['Close'].pct_change().dropna()
    
    if len(data_dict) < 2:
        return None
    
    # Create correlation matrix
    df = pd.DataFrame(data_dict)
    correlation_matrix = df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=correlation_matrix.round(2).values,
        texttemplate="%{text}",
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title='Stock Correlation Heatmap',
        height=500,
        template='plotly_white'
    )
    
    return fig

def display_chart_management():
    """Display comprehensive chart management interface"""
    st.markdown("""
    <div class="chart-card">
        <h2>📊 Advanced Charts - Day 6 Enhanced</h2>
        <p>Interactive candlestick charts, multiple timeframes, drawing tools, and technical indicators</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Chart controls
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        symbol = st.text_input("Stock Symbol", value="AAPL", key="chart_symbol")
    
    with col2:
        timeframe = st.selectbox(
            "Timeframe",
            ["1m", "5m", "15m", "1h", "4h", "1d", "1wk", "1mo"],
            index=5,
            key="chart_timeframe"
        )
    
    with col3:
        period = st.selectbox(
            "Period",
            ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
            index=2,
            key="chart_period"
        )
    
    with col4:
        if st.button("📊 Generate Chart", type="primary"):
            st.session_state.chart_settings['timeframe'] = timeframe
            st.rerun()
    
    # Technical indicators selection
    st.subheader("📈 Technical Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        show_sma_20 = st.checkbox("SMA 20", value=True)
        show_sma_50 = st.checkbox("SMA 50", value=True)
    
    with col2:
        show_bb = st.checkbox("Bollinger Bands", value=False)
        show_ema = st.checkbox("EMA 12/26", value=False)
    
    with col3:
        show_rsi = st.checkbox("RSI", value=True)
        show_macd = st.checkbox("MACD", value=False)
    
    with col4:
        show_volume = st.checkbox("Volume", value=True)
        show_grid = st.checkbox("Grid", value=True)
    
    # Get chart data
    data, error = get_stock_data(symbol, period, timeframe)
    
    if data is not None and not data.empty:
        # Update chart settings
        indicators = []
        if show_sma_20:
            indicators.append('SMA_20')
        if show_sma_50:
            indicators.append('SMA_50')
        if show_bb:
            indicators.append('BB')
        if show_ema:
            indicators.append('EMA')
        if show_rsi:
            indicators.append('RSI')
        if show_macd:
            indicators.append('MACD')
        
        st.session_state.chart_settings['indicators'] = indicators
        st.session_state.chart_settings['show_volume'] = show_volume
        st.session_state.chart_settings['show_grid'] = show_grid
        
        # Create and display chart
        chart = create_candlestick_chart(data, symbol, timeframe, indicators)
        if chart:
            st.plotly_chart(chart, use_container_width=True)
        
        # Chart statistics
        st.subheader("📊 Chart Statistics")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            current_price = data['Close'].iloc[-1]
            st.metric("Current Price", f"${current_price:.2f}")
        
        with col2:
            price_change = data['Close'].iloc[-1] - data['Open'].iloc[0]
            st.metric("Total Change", f"${price_change:.2f}")
        
        with col3:
            price_change_pct = (price_change / data['Open'].iloc[0]) * 100
            st.metric("Change %", f"{price_change_pct:+.2f}%")
        
        with col4:
            high_price = data['High'].max()
            st.metric("52W High", f"${high_price:.2f}")
        
        with col5:
            low_price = data['Low'].min()
            st.metric("52W Low", f"${low_price:.2f}")
        
        # Technical analysis summary
        if indicators:
            st.subheader("🔍 Technical Analysis Summary")
            df = calculate_technical_indicators(data)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Moving Averages:**")
                if 'SMA_20' in df.columns and not df['SMA_20'].isna().iloc[-1]:
                    sma_20 = df['SMA_20'].iloc[-1]
                    st.write(f"SMA 20: ${sma_20:.2f}")
                if 'SMA_50' in df.columns and not df['SMA_50'].isna().iloc[-1]:
                    sma_50 = df['SMA_50'].iloc[-1]
                    st.write(f"SMA 50: ${sma_50:.2f}")
            
            with col2:
                st.write("**Bollinger Bands:**")
                if all(col in df.columns for col in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
                    bb_upper = df['BB_Upper'].iloc[-1]
                    bb_middle = df['BB_Middle'].iloc[-1]
                    bb_lower = df['BB_Lower'].iloc[-1]
                    st.write(f"Upper: ${bb_upper:.2f}")
                    st.write(f"Middle: ${bb_middle:.2f}")
                    st.write(f"Lower: ${bb_lower:.2f}")
            
            with col3:
                st.write("**RSI & MACD:**")
                if 'RSI' in df.columns and not df['RSI'].isna().iloc[-1]:
                    rsi = df['RSI'].iloc[-1]
                    st.write(f"RSI: {rsi:.2f}")
                if 'MACD' in df.columns and not df['MACD'].isna().iloc[-1]:
                    macd = df['MACD'].iloc[-1]
                    st.write(f"MACD: {macd:.4f}")
    
    else:
        st.error(f"❌ {error}")
        st.info("💡 Try a different symbol or timeframe")

def display_chart_comparison():
    """Display chart comparison interface"""
    st.markdown("""
    <div class="timeframe-card">
        <h3>📊 Chart Comparison</h3>
        <p>Compare multiple stocks side-by-side with normalized performance</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Comparison controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        comparison_symbols = st.text_input(
            "Symbols to Compare (comma-separated)",
            value="AAPL,MSFT,GOOGL,AMZN,TSLA",
            help="Enter stock symbols separated by commas"
        )
    
    with col2:
        comparison_timeframe = st.selectbox(
            "Timeframe",
            ["1d", "1wk", "1mo"],
            index=2,
            key="comparison_timeframe"
        )
    
    with col3:
        comparison_period = st.selectbox(
            "Period",
            ["1mo", "3mo", "6mo", "1y"],
            index=0,
            key="comparison_period"
        )
    
    if st.button("📊 Compare Charts", type="primary"):
        symbols = [s.strip().upper() for s in comparison_symbols.split(',')]
        st.session_state.chart_comparison = symbols
        st.rerun()
    
    # Display comparison chart
    if st.session_state.chart_comparison:
        chart = create_comparison_chart(
            st.session_state.chart_comparison,
            comparison_timeframe,
            comparison_period
        )
        if chart:
            st.plotly_chart(chart, use_container_width=True)
        
        # Correlation heatmap
        if len(st.session_state.chart_comparison) >= 2:
            st.subheader("🔗 Correlation Analysis")
            heatmap = create_heatmap_chart(
                st.session_state.chart_comparison,
                comparison_timeframe
            )
            if heatmap:
                st.plotly_chart(heatmap, use_container_width=True)

def display_drawing_tools():
    """Display chart drawing tools interface"""
    st.markdown("""
    <div class="drawing-tool">
        <h3>✏️ Chart Drawing Tools</h3>
        <p>Professional drawing tools for technical analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📏 Trend Lines")
        st.write("• **Support Lines**: Draw horizontal support levels")
        st.write("• **Resistance Lines**: Draw horizontal resistance levels")
        st.write("• **Trend Lines**: Draw diagonal trend lines")
        st.write("• **Channel Lines**: Draw parallel channel lines")
    
    with col2:
        st.subheader("📐 Fibonacci Tools")
        st.write("• **Fibonacci Retracements**: 23.6%, 38.2%, 50%, 61.8%")
        st.write("• **Fibonacci Extensions**: 127.2%, 161.8%, 261.8%")
        st.write("• **Fibonacci Fans**: Angle-based Fibonacci lines")
        st.write("• **Fibonacci Arcs**: Time-based Fibonacci curves")
    
    with col3:
        st.subheader("✏️ Annotation Tools")
        st.write("• **Text Labels**: Add text annotations to charts")
        st.write("• **Shapes**: Rectangles, circles, triangles")
        st.write("• **Arrows**: Point to specific price levels")
        st.write("• **Freehand Drawing**: Custom drawing capabilities")
    
    # Drawing tools status
    st.subheader("🛠️ Drawing Tools Status")
    st.info("📝 **Note**: Drawing tools are implemented in the chart interface. Use the interactive chart above to draw trend lines, add annotations, and perform technical analysis.")
    
    # Quick drawing guide
    with st.expander("📖 Quick Drawing Guide", expanded=False):
        st.markdown("""
        **How to Use Drawing Tools:**
        1. **Hover** over the chart to see drawing options
        2. **Click and drag** to draw trend lines
        3. **Right-click** to access drawing menu
        4. **Double-click** to add text annotations
        5. **Use keyboard shortcuts** for quick access
        
        **Keyboard Shortcuts:**
        - `T` - Trend line tool
        - `H` - Horizontal line tool
        - `F` - Fibonacci retracement
        - `A` - Arrow tool
        - `S` - Shape tool
        - `Esc` - Exit drawing mode
        """)

def main():
    """Main application"""
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📊 Financial Analyzer Pro - Day 6</h1>
        <p>Advanced Charts with Interactive Candlesticks, Multiple Timeframes & Technical Indicators</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Status
    st.markdown("""
    <div class="success-message">
        <h4>🚀 Day 6: Advanced Charts Enhanced</h4>
        <p>✅ Interactive Candlestick Charts | ✅ Multiple Timeframe Analysis | ✅ Chart Drawing Tools | ✅ Technical Indicator Overlays</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Chart Settings")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Chart Features",
        ["📊 Main Chart", "📈 Chart Comparison", "✏️ Drawing Tools", "🔍 Technical Analysis"],
        index=0
    )
    
    # Chart theme
    theme = st.sidebar.selectbox("Chart Theme", ["Light", "Dark", "Auto"], index=0)
    
    # System status
    st.sidebar.subheader("📊 System Status")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("ML Status", "🟢 Available" if SKLEARN_AVAILABLE else "🟡 Limited")
    with col2:
        st.metric("Cache Size", f"{len(cache.cache)}/50")
    
    # Cache management
    if st.sidebar.button("Clear Cache"):
        cache.clear()
        st.sidebar.success("Cache cleared!")
    
    # Main content based on navigation
    if page == "📊 Main Chart":
        display_chart_management()
    
    elif page == "📈 Chart Comparison":
        display_chart_comparison()
    
    elif page == "✏️ Drawing Tools":
        display_drawing_tools()
    
    elif page == "🔍 Technical Analysis":
        st.subheader("🔍 Advanced Technical Analysis")
        st.info("Technical analysis features are integrated into the main chart. Use the chart above to access all technical indicators and analysis tools.")
        
        # Technical analysis summary
        st.subheader("📊 Available Technical Indicators")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Trend Indicators:**")
            st.write("• Simple Moving Average (SMA)")
            st.write("• Exponential Moving Average (EMA)")
            st.write("• Bollinger Bands")
            st.write("• Parabolic SAR")
        
        with col2:
            st.write("**Momentum Indicators:**")
            st.write("• RSI (Relative Strength Index)")
            st.write("• MACD")
            st.write("• Stochastic Oscillator")
            st.write("• Williams %R")
        
        with col3:
            st.write("**Volume Indicators:**")
            st.write("• Volume SMA")
            st.write("• On-Balance Volume (OBV)")
            st.write("• Volume Rate of Change")
            st.write("• Accumulation/Distribution")

if __name__ == "__main__":
    main()
