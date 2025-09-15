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

# Page config
st.set_page_config(
    page_title="Financial Analyzer Pro - Phase 1",
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
    .technical-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .portfolio-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .market-card {
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
</style>
""", unsafe_allow_html=True)

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
        return {}
    
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
        
        return data
    except Exception as e:
        st.error(f"Error calculating technical indicators: {str(e)}")
        return data

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

def calculate_portfolio_metrics(portfolio_data):
    """Calculate advanced portfolio performance metrics"""
    if portfolio_data.empty:
        return {}
    
    try:
        # Basic metrics
        total_value = portfolio_data['Value'].sum()
        total_cost = (portfolio_data['Shares'] * portfolio_data['Purchase Price']).sum()
        total_pnl = total_value - total_cost
        total_pnl_pct = (total_pnl / total_cost) * 100 if total_cost > 0 else 0
        
        # Individual position metrics
        portfolio_data['P&L'] = portfolio_data['Value'] - (portfolio_data['Shares'] * portfolio_data['Purchase Price'])
        portfolio_data['P&L %'] = (portfolio_data['P&L'] / (portfolio_data['Shares'] * portfolio_data['Purchase Price'])) * 100
        
        # Risk metrics
        returns = portfolio_data['P&L %'] / 100
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        sharpe_ratio = (returns.mean() * 252) / (volatility * np.sqrt(252)) if volatility > 0 else 0
        
        # Diversification metrics
        weights = portfolio_data['Value'] / total_value if total_value > 0 else portfolio_data['Value']
        herfindahl_index = (weights ** 2).sum()
        diversification_ratio = 1 - herfindahl_index
        
        # Best and worst performers
        best_performer = portfolio_data.loc[portfolio_data['P&L %'].idxmax()] if not portfolio_data.empty else None
        worst_performer = portfolio_data.loc[portfolio_data['P&L %'].idxmin()] if not portfolio_data.empty else None
        
        return {
            'total_value': total_value,
            'total_cost': total_cost,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'diversification_ratio': diversification_ratio,
            'best_performer': best_performer,
            'worst_performer': worst_performer,
            'position_count': len(portfolio_data)
        }
    except Exception as e:
        st.error(f"Error calculating portfolio metrics: {str(e)}")
        return {}

def create_candlestick_chart(data, symbol):
    """Create professional candlestick chart with technical indicators"""
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name=symbol,
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ))
    
    # Moving averages
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
        title=f"{symbol} Price Chart with Technical Indicators",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=500,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    return fig

def create_volume_chart(data, symbol):
    """Create volume analysis chart"""
    fig = go.Figure()
    
    # Volume bars
    colors = ['green' if data['Close'].iloc[i] >= data['Open'].iloc[i] else 'red' 
              for i in range(len(data))]
    
    fig.add_trace(go.Bar(
        x=data.index,
        y=data['Volume'],
        name='Volume',
        marker_color=colors,
        opacity=0.7
    ))
    
    # Volume moving average
    if 'Volume_SMA' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Volume_SMA'],
            mode='lines',
            name='Volume SMA 20',
            line=dict(color='blue', width=2)
        ))
    
    fig.update_layout(
        title=f"{symbol} Volume Analysis",
        xaxis_title="Date",
        yaxis_title="Volume",
        height=300,
        showlegend=True
    )
    
    return fig

def create_technical_indicators_chart(data, symbol):
    """Create technical indicators subplot chart"""
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('RSI', 'MACD', 'Stochastic'),
        vertical_spacing=0.1,
        row_heights=[0.4, 0.3, 0.3]
    )
    
    # RSI
    if 'RSI' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['RSI'],
            mode='lines',
            name='RSI',
            line=dict(color='purple', width=2)
        ), row=1, col=1)
        
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", 
                     annotation_text="Overbought (70)", row=1, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", 
                     annotation_text="Oversold (30)", row=1, col=1)
    
    # MACD
    if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['MACD'],
            mode='lines',
            name='MACD',
            line=dict(color='blue', width=2)
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['MACD_Signal'],
            mode='lines',
            name='MACD Signal',
            line=dict(color='red', width=2)
        ), row=2, col=1)
        
        # MACD Histogram
        if 'MACD_Histogram' in data.columns:
            colors = ['green' if x >= 0 else 'red' for x in data['MACD_Histogram']]
            fig.add_trace(go.Bar(
                x=data.index,
                y=data['MACD_Histogram'],
                name='MACD Histogram',
                marker_color=colors,
                opacity=0.7
            ), row=2, col=1)
    
    # Stochastic
    if 'Stoch_K' in data.columns and 'Stoch_D' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Stoch_K'],
            mode='lines',
            name='Stoch %K',
            line=dict(color='orange', width=2)
        ), row=3, col=1)
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Stoch_D'],
            mode='lines',
            name='Stoch %D',
            line=dict(color='blue', width=2)
        ), row=3, col=1)
        
        # Stochastic levels
        fig.add_hline(y=80, line_dash="dash", line_color="red", 
                     annotation_text="Overbought (80)", row=3, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="green", 
                     annotation_text="Oversold (20)", row=3, col=1)
    
    fig.update_layout(
        title=f"{symbol} Technical Indicators",
        height=600,
        showlegend=True
    )
    
    return fig

def export_data(data, format_type, filename):
    """Export data in specified format"""
    if format_type == "CSV":
        return data.to_csv(index=True)
    elif format_type == "JSON":
        return data.to_json(orient='records', date_format='iso')
    else:
        return None

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📈 Financial Analyzer Pro</h1>
        <p>Phase 1 Enhanced - Advanced Technical Analysis & Portfolio Management</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Phase indicator
    st.markdown("""
    <div class="phase-indicator">
        <h3>🚀 Phase 1 Features Active</h3>
        <p>✅ Advanced Technical Indicators | ✅ Professional Charts | ✅ Enhanced Portfolio Metrics | ✅ Market Overview | ✅ Data Export</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.selectbox("Choose Analysis", [
        "📈 Advanced Stock Analysis", 
        "📊 Enhanced Market Overview", 
        "💼 Portfolio Analytics",
        "🔧 Technical Indicators"
    ])
    
    if page == "📈 Advanced Stock Analysis":
        advanced_stock_analysis_page()
    elif page == "📊 Enhanced Market Overview":
        enhanced_market_overview_page()
    elif page == "💼 Portfolio Analytics":
        portfolio_analytics_page()
    elif page == "🔧 Technical Indicators":
        technical_indicators_page()

def advanced_stock_analysis_page():
    """Advanced stock analysis with technical indicators"""
    st.header("📈 Advanced Stock Analysis")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        symbol = st.text_input("Enter Stock Symbol", value="AAPL", placeholder="e.g., AAPL, MSFT, GOOGL")
    with col2:
        timeframe = st.selectbox("Timeframe", ["1mo", "3mo", "6mo", "1y", "2y", "5y"])
    
    if st.button("Analyze Stock", type="primary"):
        if symbol:
            with st.spinner(f"Performing advanced analysis for {symbol}..."):
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
                    
                    # Technical indicators summary
                    if 'RSI' in data.columns:
                        current_rsi = data['RSI'].iloc[-1]
                        rsi_signal = "🔴 Overbought" if current_rsi > 70 else "🟢 Oversold" if current_rsi < 30 else "🟡 Neutral"
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("RSI", f"{current_rsi:.1f}", rsi_signal)
                        with col2:
                            if 'MACD' in data.columns:
                                current_macd = data['MACD'].iloc[-1]
                                st.metric("MACD", f"{current_macd:.2f}")
                        with col3:
                            if 'BB_Position' in data.columns:
                                bb_pos = data['BB_Position'].iloc[-1]
                                bb_signal = "🔴 Near Upper" if bb_pos > 0.8 else "🟢 Near Lower" if bb_pos < 0.2 else "🟡 Middle"
                                st.metric("BB Position", f"{bb_pos:.2f}", bb_signal)
                        with col4:
                            if 'Stoch_K' in data.columns:
                                stoch_k = data['Stoch_K'].iloc[-1]
                                st.metric("Stochastic %K", f"{stoch_k:.1f}")
                    
                    # Charts
                    st.subheader("📊 Price Chart with Technical Indicators")
                    candlestick_fig = create_candlestick_chart(data, symbol)
                    st.plotly_chart(candlestick_fig, use_container_width=True)
                    
                    st.subheader("📊 Volume Analysis")
                    volume_fig = create_volume_chart(data, symbol)
                    st.plotly_chart(volume_fig, use_container_width=True)
                    
                    st.subheader("📊 Technical Indicators")
                    indicators_fig = create_technical_indicators_chart(data, symbol)
                    st.plotly_chart(indicators_fig, use_container_width=True)
                    
                    # Export functionality
                    st.subheader("📥 Export Data")
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        export_format = st.selectbox("Format", ["CSV", "JSON"])
                    with col2:
                        if st.button("Download Data"):
                            csv_data = export_data(data, export_format, f"{symbol}_analysis")
                            st.download_button(
                                label=f"Download {export_format}",
                                data=csv_data,
                                file_name=f"{symbol}_analysis.{export_format.lower()}",
                                mime="text/csv" if export_format == "CSV" else "application/json"
                            )

def enhanced_market_overview_page():
    """Enhanced market overview with more indices and sectors"""
    st.header("📊 Enhanced Market Overview")
    
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
        st.subheader("🎯 Market Sentiment Analysis")
        
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
        
        # VIX analysis
        if 'VIX' in overview:
            vix_value = overview['VIX']['price']
            if vix_value < 20:
                vix_signal = "🟢 Low Volatility - Market Calm"
            elif vix_value > 30:
                vix_signal = "🔴 High Volatility - Market Stress"
            else:
                vix_signal = "🟡 Moderate Volatility"
            
            st.markdown(f"""
            <div class="market-card">
                <h4>📊 VIX Analysis</h4>
                <p><strong>Current VIX:</strong> {vix_value:.2f}</p>
                <p><strong>Signal:</strong> {vix_signal}</p>
            </div>
            """, unsafe_allow_html=True)

def portfolio_analytics_page():
    """Enhanced portfolio analytics with advanced metrics"""
    st.header("💼 Portfolio Analytics")
    
    # Sample portfolio data (in real app, this would come from database)
    portfolio_data = pd.DataFrame({
        'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'NVDA'],
        'Shares': [10, 5, 3, 2, 1, 4],
        'Purchase Price': [150.25, 300.50, 2800.75, 200.00, 3200.00, 400.00],
        'Current Price': [175.30, 320.15, 2900.20, 185.50, 3500.00, 450.00],
        'Value': [1753.00, 1600.75, 8700.60, 371.00, 3500.00, 1800.00]
    })
    
    # Calculate advanced metrics
    metrics = calculate_portfolio_metrics(portfolio_data)
    
    if metrics:
        # Portfolio summary
        st.subheader("📊 Portfolio Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Value", f"${metrics['total_value']:,.2f}")
        with col2:
            st.metric("Total P&L", f"${metrics['total_pnl']:,.2f}")
        with col3:
            st.metric("P&L %", f"{metrics['total_pnl_pct']:.2f}%")
        with col4:
            st.metric("Positions", f"{metrics['position_count']}")
        
        # Risk metrics
        st.subheader("📈 Risk & Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Volatility", f"{metrics['volatility']:.2f}")
        with col2:
            st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        with col3:
            st.metric("Diversification", f"{metrics['diversification_ratio']:.2f}")
        with col4:
            risk_level = "🟢 Low" if metrics['volatility'] < 0.2 else "🟡 Medium" if metrics['volatility'] < 0.4 else "🔴 High"
            st.metric("Risk Level", risk_level)
        
        # Best and worst performers
        if metrics['best_performer'] is not None and metrics['worst_performer'] is not None:
            st.subheader("🏆 Top Performers")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="portfolio-card">
                    <h4>🥇 Best Performer</h4>
                    <p><strong>{metrics['best_performer']['Symbol']}</strong></p>
                    <p>P&L: ${metrics['best_performer']['P&L']:.2f} ({metrics['best_performer']['P&L %']:.2f}%)</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="portfolio-card">
                    <h4>🥉 Worst Performer</h4>
                    <p><strong>{metrics['worst_performer']['Symbol']}</strong></p>
                    <p>P&L: ${metrics['worst_performer']['P&L']:.2f} ({metrics['worst_performer']['P&L %']:.2f}%)</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Portfolio allocation chart
        st.subheader("📊 Portfolio Allocation")
        fig = px.pie(portfolio_data, values='Value', names='Symbol', 
                    title="Portfolio Allocation by Value")
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance chart
        st.subheader("📈 Performance Analysis")
        portfolio_data['P&L %'] = (portfolio_data['P&L'] / (portfolio_data['Shares'] * portfolio_data['Purchase Price'])) * 100
        
        fig = px.bar(portfolio_data, x='Symbol', y='P&L %', 
                    title="Individual Position Performance",
                    color='P&L %', color_continuous_scale='RdYlGn')
        st.plotly_chart(fig, use_container_width=True)
        
        # Export portfolio data
        st.subheader("📥 Export Portfolio Data")
        col1, col2 = st.columns([1, 3])
        with col1:
            export_format = st.selectbox("Format", ["CSV", "JSON"], key="portfolio_export")
        with col2:
            if st.button("Download Portfolio Data"):
                csv_data = export_data(portfolio_data, export_format, "portfolio_analysis")
                st.download_button(
                    label=f"Download {export_format}",
                    data=csv_data,
                    file_name=f"portfolio_analysis.{export_format.lower()}",
                    mime="text/csv" if export_format == "CSV" else "application/json"
                )

def technical_indicators_page():
    """Dedicated technical indicators analysis page"""
    st.header("🔧 Technical Indicators Analysis")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        symbol = st.text_input("Enter Stock Symbol", value="AAPL", placeholder="e.g., AAPL, MSFT, GOOGL", key="tech_symbol")
    with col2:
        timeframe = st.selectbox("Timeframe", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], key="tech_timeframe")
    
    if st.button("Analyze Technical Indicators", type="primary"):
        if symbol:
            with st.spinner(f"Calculating technical indicators for {symbol}..."):
                data = get_market_data(symbol, timeframe)
                
                if data is not None and not data.empty:
                    data = calculate_technical_indicators(data)
                    
                    # Current indicator values
                    st.subheader("📊 Current Indicator Values")
                    
                    if 'RSI' in data.columns:
                        current_rsi = data['RSI'].iloc[-1]
                        rsi_signal = "🔴 Overbought" if current_rsi > 70 else "🟢 Oversold" if current_rsi < 30 else "🟡 Neutral"
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.markdown(f"""
                            <div class="technical-card">
                                <h4>RSI (14)</h4>
                                <p><strong>Value:</strong> {current_rsi:.1f}</p>
                                <p><strong>Signal:</strong> {rsi_signal}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            if 'MACD' in data.columns:
                                current_macd = data['MACD'].iloc[-1]
                                macd_signal = data['MACD_Signal'].iloc[-1]
                                macd_trend = "🟢 Bullish" if current_macd > macd_signal else "🔴 Bearish"
                                st.markdown(f"""
                                <div class="technical-card">
                                    <h4>MACD</h4>
                                    <p><strong>MACD:</strong> {current_macd:.2f}</p>
                                    <p><strong>Signal:</strong> {macd_signal:.2f}</p>
                                    <p><strong>Trend:</strong> {macd_trend}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        with col3:
                            if 'BB_Position' in data.columns:
                                bb_pos = data['BB_Position'].iloc[-1]
                                bb_signal = "🔴 Near Upper" if bb_pos > 0.8 else "🟢 Near Lower" if bb_pos < 0.2 else "🟡 Middle"
                                st.markdown(f"""
                                <div class="technical-card">
                                    <h4>Bollinger Bands</h4>
                                    <p><strong>Position:</strong> {bb_pos:.2f}</p>
                                    <p><strong>Signal:</strong> {bb_signal}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        with col4:
                            if 'Stoch_K' in data.columns and 'Stoch_D' in data.columns:
                                stoch_k = data['Stoch_K'].iloc[-1]
                                stoch_d = data['Stoch_D'].iloc[-1]
                                stoch_signal = "🔴 Overbought" if stoch_k > 80 else "🟢 Oversold" if stoch_k < 20 else "🟡 Neutral"
                                st.markdown(f"""
                                <div class="technical-card">
                                    <h4>Stochastic</h4>
                                    <p><strong>%K:</strong> {stoch_k:.1f}</p>
                                    <p><strong>%D:</strong> {stoch_d:.1f}</p>
                                    <p><strong>Signal:</strong> {stoch_signal}</p>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Technical indicators chart
                    st.subheader("📊 Technical Indicators Chart")
                    indicators_fig = create_technical_indicators_chart(data, symbol)
                    st.plotly_chart(indicators_fig, use_container_width=True)
                    
                    # Indicator summary
                    st.subheader("📋 Technical Analysis Summary")
                    
                    # Calculate overall signal
                    signals = []
                    if 'RSI' in data.columns:
                        rsi = data['RSI'].iloc[-1]
                        if rsi > 70:
                            signals.append("RSI: Overbought")
                        elif rsi < 30:
                            signals.append("RSI: Oversold")
                    
                    if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
                        macd = data['MACD'].iloc[-1]
                        macd_sig = data['MACD_Signal'].iloc[-1]
                        if macd > macd_sig:
                            signals.append("MACD: Bullish")
                        else:
                            signals.append("MACD: Bearish")
                    
                    if 'BB_Position' in data.columns:
                        bb_pos = data['BB_Position'].iloc[-1]
                        if bb_pos > 0.8:
                            signals.append("BB: Near Upper Band")
                        elif bb_pos < 0.2:
                            signals.append("BB: Near Lower Band")
                    
                    if signals:
                        st.write("**Current Signals:**")
                        for signal in signals:
                            st.write(f"• {signal}")
                    else:
                        st.write("**Current Signals:** Mixed signals - no clear trend")

if __name__ == "__main__":
    main()
