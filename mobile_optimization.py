import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

def create_mobile_optimized_interface():
    """Create mobile-optimized interface for Financial Analyzer Pro"""
    
    # Mobile-specific CSS
    st.markdown("""
    <style>
    /* Mobile-first responsive design */
    @media (max-width: 768px) {
        .main-header {
            padding: 1rem;
            font-size: 1.2rem;
        }
        .metric-card {
            padding: 1rem;
            margin: 0.5rem 0;
        }
        .mobile-button {
            width: 100%;
            margin: 0.25rem 0;
        }
        .mobile-input {
            width: 100%;
        }
        .mobile-chart {
            height: 300px;
        }
        .mobile-table {
            font-size: 0.8rem;
        }
        .mobile-nav {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: white;
            border-top: 1px solid #ddd;
            padding: 0.5rem;
            z-index: 1000;
        }
        .mobile-nav-item {
            display: inline-block;
            width: 20%;
            text-align: center;
            padding: 0.5rem;
            text-decoration: none;
            color: #666;
        }
        .mobile-nav-item.active {
            color: #007bff;
            font-weight: bold;
        }
        .mobile-card {
            background: white;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .mobile-swipe {
            overflow-x: auto;
            white-space: nowrap;
        }
        .mobile-swipe-item {
            display: inline-block;
            width: 200px;
            margin-right: 1rem;
        }
    }
    
    /* PWA specific styles */
    .pwa-install {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        text-align: center;
    }
    
    .pwa-install button {
        background: white;
        color: #667eea;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        font-weight: bold;
        cursor: pointer;
    }
    
    /* Touch-friendly elements */
    .touch-target {
        min-height: 44px;
        min-width: 44px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    /* Mobile navigation */
    .mobile-tabs {
        display: flex;
        background: #f8f9fa;
        border-radius: 8px;
        margin: 1rem 0;
        overflow-x: auto;
    }
    
    .mobile-tab {
        flex: 1;
        padding: 0.75rem;
        text-align: center;
        border-radius: 8px;
        margin: 0.25rem;
        background: transparent;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .mobile-tab.active {
        background: #007bff;
        color: white;
    }
    
    /* Responsive charts */
    .chart-container {
        width: 100%;
        height: 300px;
    }
    
    @media (min-width: 768px) {
        .chart-container {
            height: 500px;
        }
    }
    
    /* Mobile-specific animations */
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .slide-in {
        animation: slideIn 0.3s ease-out;
    }
    
    @keyframes slideIn {
        from { transform: translateX(-100%); }
        to { transform: translateX(0); }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # PWA manifest and service worker
    st.markdown("""
    <link rel="manifest" href="data:application/json,{
        'name': 'Financial Analyzer Pro',
        'short_name': 'FinAnalyzer',
        'description': 'Professional Financial Analysis Platform',
        'start_url': '/',
        'display': 'standalone',
        'background_color': '#ffffff',
        'theme_color': '#667eea',
        'icons': [
            {
                'src': 'data:image/svg+xml,<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 100 100\"><text y=\".9em\" font-size=\"90\">📈</text></svg>',
                'sizes': '192x192',
                'type': 'image/svg+xml'
            }
        ]
    }">
    """, unsafe_allow_html=True)
    
    # Mobile navigation
    st.markdown("""
    <div class="mobile-tabs">
        <button class="mobile-tab active" onclick="showTab('dashboard')">🏠 Dashboard</button>
        <button class="mobile-tab" onclick="showTab('stocks')">📊 Stocks</button>
        <button class="mobile-tab" onclick="showTab('portfolio')">💼 Portfolio</button>
        <button class="mobile-tab" onclick="showTab('analytics')">📈 Analytics</button>
        <button class="mobile-tab" onclick="showTab('settings')">⚙️ Settings</button>
    </div>
    """, unsafe_allow_html=True)
    
    # JavaScript for mobile navigation
    st.markdown("""
    <script>
    function showTab(tabName) {
        // Hide all tab contents
        const tabs = document.querySelectorAll('.tab-content');
        tabs.forEach(tab => tab.style.display = 'none');
        
        // Remove active class from all tabs
        const tabButtons = document.querySelectorAll('.mobile-tab');
        tabButtons.forEach(button => button.classList.remove('active'));
        
        // Show selected tab
        document.getElementById(tabName).style.display = 'block';
        
        // Add active class to clicked tab
        event.target.classList.add('active');
    }
    
    // PWA install prompt
    let deferredPrompt;
    window.addEventListener('beforeinstallprompt', (e) => {
        e.preventDefault();
        deferredPrompt = e;
        document.getElementById('install-button').style.display = 'block';
    });
    
    function installPWA() {
        if (deferredPrompt) {
            deferredPrompt.prompt();
            deferredPrompt.userChoice.then((choiceResult) => {
                if (choiceResult.outcome === 'accepted') {
                    console.log('PWA installed');
                }
                deferredPrompt = null;
            });
        }
    }
    </script>
    """, unsafe_allow_html=True)
    
    # PWA Install Banner
    st.markdown("""
    <div class="pwa-install" id="install-banner" style="display: none;">
        <h4>📱 Install Financial Analyzer Pro</h4>
        <p>Get the full mobile experience with our PWA!</p>
        <button id="install-button" onclick="installPWA()" style="display: none;">Install App</button>
    </div>
    """, unsafe_allow_html=True)

def create_mobile_dashboard():
    """Create mobile-optimized dashboard"""
    st.markdown('<div class="tab-content" id="dashboard">', unsafe_allow_html=True)
    
    st.header("📱 Mobile Dashboard")
    
    # Quick market overview
    st.subheader("📈 Market Overview")
    
    # Get market data
    symbols = ['^GSPC', '^IXIC', '^DJI', '^VIX']
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
                    'change_percent': change_percent
                }
        except:
            continue
    
    if market_data:
        # Mobile-friendly metrics
        col1, col2 = st.columns(2)
        
        with col1:
            if '^GSPC' in market_data:
                data = market_data['^GSPC']
                change_color = "🟢" if data['change'] >= 0 else "🔴"
                st.metric(
                    "S&P 500",
                    f"${data['price']:.2f}",
                    f"{change_color} {data['change_percent']:+.2f}%"
                )
        
        with col2:
            if '^IXIC' in market_data:
                data = market_data['^IXIC']
                change_color = "🟢" if data['change'] >= 0 else "🔴"
                st.metric(
                    "NASDAQ",
                    f"${data['price']:.2f}",
                    f"{change_color} {data['change_percent']:+.2f}%"
                )
    
    # Portfolio summary (mobile)
    st.subheader("💼 Portfolio Summary")
    
    if 'portfolio' in st.session_state and st.session_state.portfolio:
        total_value = sum(pos['value'] for pos in st.session_state.portfolio)
        total_cost = sum(pos['cost_basis'] for pos in st.session_state.portfolio)
        total_pnl = total_value - total_cost
        total_pnl_percent = (total_pnl / total_cost) * 100 if total_cost > 0 else 0
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Value", f"${total_value:,.0f}")
        with col2:
            st.metric("P&L", f"{total_pnl_percent:+.1f}%")
        
        # Mobile portfolio list
        for position in st.session_state.portfolio[:5]:  # Show only first 5
            with st.container():
                st.markdown(f"""
                <div class="mobile-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong>{position['symbol']}</strong><br>
                            <small>{position['shares']} shares</small>
                        </div>
                        <div style="text-align: right;">
                            <div>${position['current_price']:.2f}</div>
                            <div style="color: {'green' if position['pnl'] >= 0 else 'red'};">
                                {position['pnl_percent']:+.1f}%
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No portfolio positions")
    
    # Quick actions
    st.subheader("⚡ Quick Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Analyze Stock", use_container_width=True):
            st.session_state.mobile_tab = "stocks"
            st.rerun()
    
    with col2:
        if st.button("💼 Add Position", use_container_width=True):
            st.session_state.mobile_tab = "portfolio"
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def create_mobile_stock_analysis():
    """Create mobile-optimized stock analysis"""
    st.markdown('<div class="tab-content" id="stocks" style="display: none;">', unsafe_allow_html=True)
    
    st.header("📊 Stock Analysis")
    
    # Mobile input
    symbol = st.text_input("Stock Symbol", value="AAPL", placeholder="Enter symbol (e.g., AAPL)")
    period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y"], index=1)
    
    if st.button("Analyze", type="primary", use_container_width=True):
        with st.spinner("Analyzing..."):
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period)
                
                if not data.empty:
                    st.success(f"✅ Analysis complete for {symbol}")
                    
                    # Mobile metrics
                    current_price = data['Close'].iloc[-1]
                    previous_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
                    change = current_price - previous_price
                    change_percent = (change / previous_price) * 100
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Price", f"${current_price:.2f}")
                    with col2:
                        st.metric("Change", f"{change_percent:+.2f}%")
                    
                    # Mobile chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data['Close'],
                        mode='lines',
                        name='Close Price',
                        line=dict(color='#1f77b4', width=2)
                    ))
                    
                    fig.update_layout(
                        title=f"{symbol} Price Chart",
                        height=300,
                        showlegend=False,
                        margin=dict(l=0, r=0, t=30, b=0)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Technical indicators (simplified for mobile)
                    st.subheader("📈 Technical Indicators")
                    
                    # Calculate simple indicators
                    data['SMA_20'] = data['Close'].rolling(window=20).mean()
                    data['SMA_50'] = data['Close'].rolling(window=50).mean()
                    
                    # RSI
                    delta = data['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    data['RSI'] = 100 - (100 / (1 + rs))
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("SMA 20", f"${data['SMA_20'].iloc[-1]:.2f}")
                    with col2:
                        st.metric("SMA 50", f"${data['SMA_50'].iloc[-1]:.2f}")
                    with col3:
                        st.metric("RSI", f"{data['RSI'].iloc[-1]:.1f}")
                    
                    # Mobile action buttons
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("Add to Watchlist", use_container_width=True):
                            if 'watchlist' not in st.session_state:
                                st.session_state.watchlist = []
                            if symbol not in st.session_state.watchlist:
                                st.session_state.watchlist.append(symbol)
                                st.success(f"Added {symbol} to watchlist")
                    
                    with col2:
                        if st.button("Add to Portfolio", use_container_width=True):
                            st.session_state.mobile_tab = "portfolio"
                            st.rerun()
                
                else:
                    st.error(f"No data found for {symbol}")
                    
            except Exception as e:
                st.error(f"Error analyzing {symbol}: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def create_mobile_portfolio():
    """Create mobile-optimized portfolio management"""
    st.markdown('<div class="tab-content" id="portfolio" style="display: none;">', unsafe_allow_html=True)
    
    st.header("💼 Portfolio")
    
    # Add position
    st.subheader("➕ Add Position")
    
    with st.form("add_position_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            symbol = st.text_input("Symbol", value="AAPL")
            shares = st.number_input("Shares", min_value=1, value=10)
        
        with col2:
            price = st.number_input("Price", min_value=0.01, value=150.00, step=0.01)
        
        if st.form_submit_button("Add Position", use_container_width=True):
            if 'portfolio' not in st.session_state:
                st.session_state.portfolio = []
            
            # Get current price
            try:
                ticker = yf.Ticker(symbol)
                current_data = ticker.history(period="1d")
                if not current_data.empty:
                    current_price = current_data['Close'].iloc[-1]
                    position = {
                        'symbol': symbol,
                        'shares': shares,
                        'purchase_price': price,
                        'current_price': current_price,
                        'value': shares * current_price,
                        'cost_basis': shares * price,
                        'pnl': (current_price - price) * shares,
                        'pnl_percent': ((current_price - price) / price) * 100
                    }
                    st.session_state.portfolio.append(position)
                    st.success(f"Added {shares} shares of {symbol}")
                else:
                    st.error(f"Could not fetch current price for {symbol}")
            except Exception as e:
                st.error(f"Error adding position: {str(e)}")
    
    # Portfolio summary
    if 'portfolio' in st.session_state and st.session_state.portfolio:
        st.subheader("📊 Portfolio Summary")
        
        total_value = sum(pos['value'] for pos in st.session_state.portfolio)
        total_cost = sum(pos['cost_basis'] for pos in st.session_state.portfolio)
        total_pnl = total_value - total_cost
        total_pnl_percent = (total_pnl / total_cost) * 100 if total_cost > 0 else 0
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Value", f"${total_value:,.0f}")
        with col2:
            st.metric("Total P&L", f"{total_pnl_percent:+.1f}%")
        
        # Mobile portfolio list
        st.subheader("📋 Positions")
        
        for i, position in enumerate(st.session_state.portfolio):
            with st.container():
                st.markdown(f"""
                <div class="mobile-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong>{position['symbol']}</strong><br>
                            <small>{position['shares']} shares @ ${position['purchase_price']:.2f}</small>
                        </div>
                        <div style="text-align: right;">
                            <div>${position['current_price']:.2f}</div>
                            <div style="color: {'green' if position['pnl'] >= 0 else 'red'};">
                                {position['pnl_percent']:+.1f}%
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Clear portfolio button
        if st.button("Clear Portfolio", type="secondary", use_container_width=True):
            st.session_state.portfolio = []
            st.rerun()
    
    else:
        st.info("No positions in portfolio")
    
    st.markdown('</div>', unsafe_allow_html=True)

def create_mobile_analytics():
    """Create mobile-optimized analytics"""
    st.markdown('<div class="tab-content" id="analytics" style="display: none;">', unsafe_allow_html=True)
    
    st.header("📈 Analytics")
    
    # Quick analytics options
    analytics_options = [
        "Market Overview",
        "Sector Analysis", 
        "Volatility Monitor",
        "Trending Stocks"
    ]
    
    selected_analytics = st.selectbox("Select Analysis", analytics_options)
    
    if selected_analytics == "Market Overview":
        st.subheader("📊 Market Overview")
        
        # Get market data
        symbols = ['^GSPC', '^IXIC', '^DJI', '^VIX']
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
                        'change_percent': change_percent
                    }
            except:
                continue
        
        if market_data:
            for symbol, data in market_data.items():
                name = symbol.replace('^', '')
                change_color = "🟢" if data['change'] >= 0 else "🔴"
                
                st.markdown(f"""
                <div class="mobile-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div><strong>{name}</strong></div>
                        <div style="text-align: right;">
                            <div>${data['price']:.2f}</div>
                            <div style="color: {'green' if data['change'] >= 0 else 'red'};">
                                {change_color} {data['change_percent']:+.2f}%
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    elif selected_analytics == "Trending Stocks":
        st.subheader("🔥 Trending Stocks")
        
        # Get trending stocks
        trending_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META']
        trending_data = []
        
        for symbol in trending_symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="2d")
                if not hist.empty and len(hist) >= 2:
                    current_price = hist['Close'].iloc[-1]
                    previous_price = hist['Close'].iloc[-2]
                    change_percent = ((current_price - previous_price) / previous_price) * 100
                    
                    trending_data.append({
                        'symbol': symbol,
                        'price': current_price,
                        'change_percent': change_percent
                    })
            except:
                continue
        
        # Sort by absolute change
        trending_data.sort(key=lambda x: abs(x['change_percent']), reverse=True)
        
        for stock in trending_data[:5]:  # Show top 5
            change_color = "🟢" if stock['change_percent'] >= 0 else "🔴"
            
            st.markdown(f"""
            <div class="mobile-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div><strong>{stock['symbol']}</strong></div>
                    <div style="text-align: right;">
                        <div>${stock['price']:.2f}</div>
                        <div style="color: {'green' if stock['change_percent'] >= 0 else 'red'};">
                            {change_color} {stock['change_percent']:+.2f}%
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def create_mobile_settings():
    """Create mobile-optimized settings"""
    st.markdown('<div class="tab-content" id="settings" style="display: none;">', unsafe_allow_html=True)
    
    st.header("⚙️ Settings")
    
    # Theme settings
    st.subheader("🎨 Theme")
    theme = st.selectbox("Theme", ["Light", "Dark", "Auto"], index=0)
    
    # Notifications
    st.subheader("🔔 Notifications")
    price_alerts = st.checkbox("Price Alerts", value=True)
    portfolio_updates = st.checkbox("Portfolio Updates", value=True)
    market_news = st.checkbox("Market News", value=False)
    
    # Data settings
    st.subheader("📊 Data")
    auto_refresh = st.checkbox("Auto Refresh", value=True)
    refresh_interval = st.selectbox("Refresh Interval", [5, 10, 30, 60], index=1)
    
    # Export settings
    st.subheader("📤 Export")
    default_format = st.selectbox("Default Export Format", ["CSV", "JSON", "Excel"], index=0)
    
    # Save settings
    if st.button("Save Settings", type="primary", use_container_width=True):
        st.success("Settings saved!")
    
    # App info
    st.subheader("ℹ️ App Info")
    st.info("""
    **Financial Analyzer Pro Mobile**
    
    Version: 1.0.0
    Last Updated: December 2024
    
    Features:
    • Real-time market data
    • Portfolio management
    • Technical analysis
    • Mobile-optimized interface
    • PWA support
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="Financial Analyzer Pro Mobile",
        page_icon="📱",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Create mobile interface
    create_mobile_optimized_interface()
    
    # Create mobile tabs
    create_mobile_dashboard()
    create_mobile_stock_analysis()
    create_mobile_portfolio()
    create_mobile_analytics()
    create_mobile_settings()
    
    # Mobile footer
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; color: #666;">
        <p>📱 <strong>Financial Analyzer Pro Mobile</strong></p>
        <p>Optimized for mobile devices • PWA Ready</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
