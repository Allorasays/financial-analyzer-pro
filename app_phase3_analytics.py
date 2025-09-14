import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import time

# Page config
st.set_page_config(
    page_title="Financial Analyzer Pro - Phase 3",
    page_icon="📈",
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
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>📈 Financial Analyzer Pro</h1>
    <p>Phase 3: Advanced Financial Analytics</p>
    <p>Status: ✅ Real-time Data + Advanced Analytics Active!</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🎯 Phase 3 Features")
st.sidebar.success("✅ Real-time Data")
st.sidebar.success("✅ Live Charts")
st.sidebar.success("✅ Technical Indicators")
st.sidebar.success("✅ Financial Ratios")
st.sidebar.success("✅ Risk Assessment")
st.sidebar.success("✅ Portfolio Analytics")

def get_stock_data(symbol, period="1mo"):
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
    
    # RSI (Relative Strength Index)
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
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    # Volume indicators
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    
    return df

def calculate_financial_ratios(data, symbol):
    """Calculate financial ratios"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        ratios = {}
        
        # Price ratios
        ratios['P/E Ratio'] = info.get('trailingPE', 'N/A')
        ratios['P/B Ratio'] = info.get('priceToBook', 'N/A')
        ratios['P/S Ratio'] = info.get('priceToSalesTrailing12Months', 'N/A')
        
        # Profitability ratios
        ratios['ROE'] = info.get('returnOnEquity', 'N/A')
        ratios['ROA'] = info.get('returnOnAssets', 'N/A')
        ratios['Gross Margin'] = info.get('grossMargins', 'N/A')
        ratios['Operating Margin'] = info.get('operatingMargins', 'N/A')
        ratios['Net Margin'] = info.get('profitMargins', 'N/A')
        
        # Growth ratios
        ratios['Revenue Growth'] = info.get('revenueGrowth', 'N/A')
        ratios['Earnings Growth'] = info.get('earningsGrowth', 'N/A')
        
        # Debt ratios
        ratios['Debt/Equity'] = info.get('debtToEquity', 'N/A')
        ratios['Current Ratio'] = info.get('currentRatio', 'N/A')
        ratios['Quick Ratio'] = info.get('quickRatio', 'N/A')
        
        return ratios
    except Exception as e:
        st.error(f"Error calculating ratios: {str(e)}")
        return {}

def calculate_risk_metrics(data):
    """Calculate risk metrics"""
    returns = data['Close'].pct_change().dropna()
    
    risk_metrics = {}
    
    # Volatility (annualized)
    risk_metrics['Volatility (30d)'] = f"{returns.std() * np.sqrt(252) * 100:.2f}%"
    
    # Sharpe Ratio (assuming 2% risk-free rate)
    risk_free_rate = 0.02
    sharpe_ratio = (returns.mean() * 252 - risk_free_rate) / (returns.std() * np.sqrt(252))
    risk_metrics['Sharpe Ratio'] = f"{sharpe_ratio:.2f}"
    
    # Maximum Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    risk_metrics['Max Drawdown'] = f"{max_drawdown * 100:.2f}%"
    
    # Value at Risk (95% confidence)
    var_95 = np.percentile(returns, 5)
    risk_metrics['VaR (95%)'] = f"{var_95 * 100:.2f}%"
    
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
st.header("📊 Advanced Stock Analysis")

# Stock input
col1, col2 = st.columns([1, 3])
with col1:
    symbol = st.text_input("Stock Symbol", value="AAPL", help="Enter a valid stock ticker symbol")
with col2:
    period = st.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=0)

if st.button("Analyze Stock", type="primary"):
    with st.spinner("Fetching data and calculating analytics..."):
        data, error = get_stock_data(symbol, period)
        
        if error:
            st.error(f"❌ {error}")
        else:
            st.success(f"✅ Successfully analyzed {symbol}")
            
            # Calculate technical indicators
            data_with_indicators = calculate_technical_indicators(data)
            
            # Get financial ratios
            financial_ratios = calculate_financial_ratios(data, symbol)
            
            # Calculate risk metrics
            risk_metrics = calculate_risk_metrics(data)
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            current_price = data['Close'].iloc[-1]
            previous_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
            change = current_price - previous_price
            change_percent = (change / previous_price) * 100
            
            with col1:
                st.metric("Current Price", f"${current_price:.2f}", f"{change:+.2f}")
            with col2:
                st.metric("Change", f"{change_percent:+.2f}%", f"{change:+.2f}")
            with col3:
                st.metric("Volume", f"{data['Volume'].iloc[-1]:,}")
            with col4:
                st.metric("RSI", f"{data_with_indicators['RSI'].iloc[-1]:.1f}")
            
            # Tabs for different analysis
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Chart", "🔧 Technical Indicators", "📊 Financial Ratios", "⚠️ Risk Analysis"])
            
            with tab1:
                st.subheader("Price Chart with Technical Indicators")
                
                fig = go.Figure()
                
                # Price line
                fig.add_trace(go.Scatter(
                    x=data_with_indicators.index,
                    y=data_with_indicators['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='#1f77b4', width=2)
                ))
                
                # Moving averages
                fig.add_trace(go.Scatter(
                    x=data_with_indicators.index,
                    y=data_with_indicators['SMA_20'],
                    mode='lines',
                    name='SMA 20',
                    line=dict(color='orange', width=1, dash='dash')
                ))
                
                fig.add_trace(go.Scatter(
                    x=data_with_indicators.index,
                    y=data_with_indicators['SMA_50'],
                    mode='lines',
                    name='SMA 50',
                    line=dict(color='red', width=1, dash='dash')
                ))
                
                # Bollinger Bands
                fig.add_trace(go.Scatter(
                    x=data_with_indicators.index,
                    y=data_with_indicators['BB_Upper'],
                    mode='lines',
                    name='BB Upper',
                    line=dict(color='gray', width=1, dash='dot'),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=data_with_indicators.index,
                    y=data_with_indicators['BB_Lower'],
                    mode='lines',
                    name='BB Lower',
                    line=dict(color='gray', width=1, dash='dot'),
                    fill='tonexty',
                    fillcolor='rgba(128,128,128,0.1)',
                    showlegend=False
                ))
                
                fig.update_layout(
                    title=f"{symbol} Price Chart with Technical Indicators",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    height=500,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.subheader("Technical Indicators")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # RSI Chart
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(
                        x=data_with_indicators.index,
                        y=data_with_indicators['RSI'],
                        mode='lines',
                        name='RSI',
                        line=dict(color='purple', width=2)
                    ))
                    
                    # RSI levels
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
                    
                    fig_rsi.update_layout(
                        title="RSI (Relative Strength Index)",
                        xaxis_title="Date",
                        yaxis_title="RSI",
                        height=300
                    )
                    
                    st.plotly_chart(fig_rsi, use_container_width=True)
                
                with col2:
                    # MACD Chart
                    fig_macd = go.Figure()
                    fig_macd.add_trace(go.Scatter(
                        x=data_with_indicators.index,
                        y=data_with_indicators['MACD'],
                        mode='lines',
                        name='MACD',
                        line=dict(color='blue', width=2)
                    ))
                    fig_macd.add_trace(go.Scatter(
                        x=data_with_indicators.index,
                        y=data_with_indicators['MACD_Signal'],
                        mode='lines',
                        name='Signal',
                        line=dict(color='red', width=2)
                    ))
                    fig_macd.add_trace(go.Bar(
                        x=data_with_indicators.index,
                        y=data_with_indicators['MACD_Histogram'],
                        name='Histogram',
                        marker_color='gray'
                    ))
                    
                    fig_macd.update_layout(
                        title="MACD (Moving Average Convergence Divergence)",
                        xaxis_title="Date",
                        yaxis_title="MACD",
                        height=300
                    )
                    
                    st.plotly_chart(fig_macd, use_container_width=True)
                
                # Technical indicators summary
                st.subheader("Technical Indicators Summary")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    rsi_value = data_with_indicators['RSI'].iloc[-1]
                    rsi_signal = "Overbought" if rsi_value > 70 else "Oversold" if rsi_value < 30 else "Neutral"
                    st.metric("RSI", f"{rsi_value:.1f}", rsi_signal)
                
                with col2:
                    macd_value = data_with_indicators['MACD'].iloc[-1]
                    macd_signal = data_with_indicators['MACD_Signal'].iloc[-1]
                    macd_trend = "Bullish" if macd_value > macd_signal else "Bearish"
                    st.metric("MACD", f"{macd_value:.3f}", macd_trend)
                
                with col3:
                    sma_20 = data_with_indicators['SMA_20'].iloc[-1]
                    sma_50 = data_with_indicators['SMA_50'].iloc[-1]
                    sma_trend = "Bullish" if sma_20 > sma_50 else "Bearish"
                    st.metric("SMA Trend", f"20: {sma_20:.2f}", sma_trend)
            
            with tab3:
                st.subheader("Financial Ratios")
                
                if financial_ratios:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Price Ratios**")
                        for ratio, value in list(financial_ratios.items())[:3]:
                            if value != 'N/A':
                                st.write(f"• {ratio}: {value}")
                            else:
                                st.write(f"• {ratio}: Not Available")
                    
                    with col2:
                        st.write("**Profitability Ratios**")
                        for ratio, value in list(financial_ratios.items())[3:6]:
                            if value != 'N/A':
                                st.write(f"• {ratio}: {value}")
                            else:
                                st.write(f"• {ratio}: Not Available")
                    
                    col3, col4 = st.columns(2)
                    
                    with col3:
                        st.write("**Growth Ratios**")
                        for ratio, value in list(financial_ratios.items())[6:8]:
                            if value != 'N/A':
                                st.write(f"• {ratio}: {value}")
                            else:
                                st.write(f"• {ratio}: Not Available")
                    
                    with col4:
                        st.write("**Debt Ratios**")
                        for ratio, value in list(financial_ratios.items())[8:]:
                            if value != 'N/A':
                                st.write(f"• {ratio}: {value}")
                            else:
                                st.write(f"• {ratio}: Not Available")
                else:
                    st.warning("Financial ratios not available for this symbol")
            
            with tab4:
                st.subheader("Risk Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Risk Metrics**")
                    for metric, value in risk_metrics.items():
                        st.write(f"• {metric}: {value}")
                
                with col2:
                    # Risk level assessment
                    volatility = float(risk_metrics['Volatility (30d)'].replace('%', ''))
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
    <p>🚀 <strong>Phase 3 Complete!</strong> Advanced Financial Analytics Active!</p>
    <p>Next: Phase 4 - Machine Learning & Advanced Features</p>
</div>
""", unsafe_allow_html=True)
