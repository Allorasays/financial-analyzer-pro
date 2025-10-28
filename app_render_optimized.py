#!/usr/bin/env python3
"""
Render-optimized version of Financial Analyzer Pro
This version is specifically optimized for Render deployment
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import numpy as np
import warnings
import json
import io
import base64
import zipfile
import time
import threading
import asyncio
import hashlib
import secrets
import sqlite3
import os
from datetime import datetime, timedelta
from collections import defaultdict, deque

warnings.filterwarnings('ignore')

# Export and reporting imports with graceful fallbacks
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.graphics.shapes import Drawing, Rect
    from reportlab.graphics.charts.lineplots import LinePlot
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.chart import LineChart, Reference
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

# ML imports with graceful fallbacks
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
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
    page_title="Financial Analyzer Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for better mobile and desktop experience
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 12px;
        border-radius: 4px;
        margin: 10px 0;
    }
    .error-message {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 12px;
        border-radius: 4px;
        margin: 10px 0;
    }
    .warning-message {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 12px;
        border-radius: 4px;
        margin: 10px 0;
    }
    @media (max-width: 768px) {
        .main-header {
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .metric-card {
            padding: 1rem;
            margin-bottom: 0.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = []
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
if 'authenticated_user' not in st.session_state:
    st.session_state.authenticated_user = None
if 'user_preferences' not in st.session_state:
    st.session_state.user_preferences = {}
if 'business_metrics' not in st.session_state:
    st.session_state.business_metrics = {}
if 'auth_system_ready' not in st.session_state:
    st.session_state.auth_system_ready = False

# Render-optimized database path
def get_db_path():
    """Get database path optimized for Render"""
    if os.getenv('RENDER'):
        # On Render, use a writable directory
        return '/tmp/users.db'
    else:
        # Local development
        return 'users.db'

def get_stock_data(symbol, period="1y"):
    """Get stock data from Yahoo Finance with error handling"""
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
    """Calculate technical indicators"""
    if data is None or len(data) < 20:
        return {}
    
    indicators = {}
    
    # Simple Moving Averages
    indicators['SMA_20'] = data['Close'].rolling(window=20).mean()
    indicators['SMA_50'] = data['Close'].rolling(window=50).mean()
    
    # Exponential Moving Averages
    indicators['EMA_12'] = data['Close'].ewm(span=12).mean()
    indicators['EMA_26'] = data['Close'].ewm(span=26).mean()
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    indicators['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    indicators['MACD'] = indicators['EMA_12'] - indicators['EMA_26']
    indicators['MACD_Signal'] = indicators['MACD'].ewm(span=9).mean()
    
    return indicators

def get_financial_ratios(symbol):
    """Get financial ratios with error handling"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        ratios = {}
        ratio_keys = {
            'trailingPE': 'P/E Ratio',
            'priceToBook': 'P/B Ratio',
            'priceToSalesTrailing12Months': 'P/S Ratio',
            'returnOnEquity': 'ROE',
            'returnOnAssets': 'ROA',
            'grossMargins': 'Gross Margin',
            'operatingMargins': 'Operating Margin',
            'profitMargins': 'Net Margin',
            'debtToEquity': 'Debt/Equity',
            'currentRatio': 'Current Ratio'
        }
        
        for key, name in ratio_keys.items():
            if key in info and info[key] is not None:
                ratios[name] = info[key]
        
        return ratios
    except Exception as e:
        st.warning(f"Could not fetch financial ratios: {str(e)}")
        return {}

def predict_price_ml(data):
    """ML price prediction with error handling"""
    if not SKLEARN_AVAILABLE or data is None or len(data) < 50:
        return None, "ML prediction not available"
    
    try:
        # Prepare features
        features = []
        for i in range(5, len(data)):
            feature_set = [
                data['Close'].iloc[i-5:i].mean(),  # 5-day average
                data['Volume'].iloc[i-5:i].mean(),  # 5-day volume average
                data['Close'].iloc[i-1],  # Previous close
                data['High'].iloc[i-1] - data['Low'].iloc[i-1],  # Daily range
                data['Close'].iloc[i-1] / data['Close'].iloc[i-5] - 1  # 5-day return
            ]
            features.append(feature_set)
        
        X = np.array(features[:-1])
        y = data['Close'].iloc[6:].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'Neural Network': MLPRegressor(hidden_layer_sizes=(50, 25), random_state=42, max_iter=500)
        }
        
        predictions = {}
        for name, model in models.items():
            try:
                model.fit(X_scaled, y)
                last_features = scaler.transform([X[-1]])
                pred = model.predict(last_features)[0]
                predictions[name] = pred
            except Exception as e:
                predictions[name] = f"Error: {str(e)}"
        
        return predictions, "ML predictions completed"
    except Exception as e:
        return None, f"ML prediction error: {str(e)}"

# Render-optimized Authentication System
class RenderUserAuthentication:
    """Render-optimized user authentication system"""
    def __init__(self, db_path=None):
        self.db_path = db_path or get_db_path()
        self._init_database()
    
    def _init_database(self):
        """Initialize database with Render compatibility"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                )
            ''')
            
            # User portfolios table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_portfolios (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    symbol TEXT NOT NULL,
                    shares REAL NOT NULL,
                    cost_basis REAL NOT NULL,
                    current_price REAL NOT NULL,
                    date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # User preferences table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_preferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    preference_key TEXT NOT NULL,
                    preference_value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id),
                    UNIQUE(user_id, preference_key)
                )
            ''')
            
            conn.commit()
            conn.close()
        except Exception as e:
            st.error(f"Database initialization error: {str(e)}")
    
    def hash_password(self, password):
        """Hash password with salt"""
        salt = secrets.token_hex(16)
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return salt + password_hash.hex()
    
    def verify_password(self, password, password_hash):
        """Verify password against hash"""
        try:
            salt = password_hash[:32]
            stored_hash = password_hash[32:]
            password_hash_check = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            return password_hash_check.hex() == stored_hash
        except:
            return False
    
    def register_user(self, username, email, password):
        """Register a new user"""
        try:
            if not username or not email or not password:
                return False, "All fields are required"
            
            if len(password) < 6:
                return False, "Password must be at least 6 characters"
            
            password_hash = self.hash_password(password)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                (username, email, password_hash)
            )
            
            user_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return True, f"User {username} registered successfully"
            
        except sqlite3.IntegrityError:
            return False, "Username or email already exists"
        except Exception as e:
            return False, f"Registration error: {str(e)}"
    
    def authenticate_user(self, username, password):
        """Authenticate user login"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT id, username, email, password_hash FROM users WHERE username = ? AND is_active = 1",
                (username,)
            )
            
            user = cursor.fetchone()
            conn.close()
            
            if user and self.verify_password(password, user[3]):
                return True, {
                    'user_id': user[0],
                    'username': user[1],
                    'email': user[2]
                }
            else:
                return False, "Invalid username or password"
                
        except Exception as e:
            return False, f"Authentication error: {str(e)}"
    
    def save_user_portfolio(self, user_id, portfolio_data):
        """Save user's portfolio to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Clear existing portfolio
            cursor.execute("DELETE FROM user_portfolios WHERE user_id = ?", (user_id,))
            
            # Insert new portfolio
            for position in portfolio_data:
                cursor.execute(
                    "INSERT INTO user_portfolios (user_id, symbol, shares, cost_basis, current_price) VALUES (?, ?, ?, ?, ?)",
                    (user_id, position['symbol'], position['shares'], position['cost_basis'], position['current_price'])
                )
            
            conn.commit()
            conn.close()
            return True, "Portfolio saved successfully"
            
        except Exception as e:
            return False, f"Error saving portfolio: {str(e)}"
    
    def load_user_portfolio(self, user_id):
        """Load user's portfolio from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT symbol, shares, cost_basis, current_price, date_added FROM user_portfolios WHERE user_id = ?",
                (user_id,)
            )
            
            portfolio_data = []
            for row in cursor.fetchall():
                portfolio_data.append({
                    'symbol': row[0],
                    'shares': row[1],
                    'cost_basis': row[2],
                    'current_price': row[3],
                    'date_added': row[4]
                })
            
            conn.close()
            return portfolio_data
            
        except Exception as e:
            st.error(f"Error loading portfolio: {str(e)}")
            return []

# Global authentication instance
auth_system = None

def main():
    """Main application optimized for Render"""
    global auth_system
    
    # Initialize authentication system
    if auth_system is None:
        try:
            auth_system = RenderUserAuthentication()
            st.session_state.auth_system_ready = True
        except Exception as e:
            st.error(f"Authentication system error: {str(e)}")
            auth_system = None
            st.session_state.auth_system_ready = False
    
    # Header
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("📊 Financial Analyzer Pro - Enhanced")
    st.markdown("**Advanced Financial Analysis & Portfolio Management**")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("🔧 Analysis Settings")
    
    # Authentication Section
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔐 Account")
    
    # Check if auth system is ready
    if not st.session_state.get('auth_system_ready', False):
        st.sidebar.error("⚠️ Authentication system not ready")
        return
    
    if st.session_state.authenticated_user:
        # User is logged in
        user_info = st.session_state.authenticated_user
        st.sidebar.success(f"👤 {user_info['username']}")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("👤 Profile"):
                st.session_state.show_profile = True
        with col2:
            if st.button("🚪 Logout"):
                # Save current data before logout
                if st.session_state.portfolio and auth_system:
                    auth_system.save_user_portfolio(user_info['user_id'], st.session_state.portfolio)
                
                # Clear session
                st.session_state.authenticated_user = None
                st.session_state.portfolio = []
                st.session_state.user_preferences = {}
                st.session_state.business_metrics = {}
                st.rerun()
    else:
        # User is not logged in - show login/register options
        auth_tab = st.sidebar.selectbox("Choose Action", ["Login", "Register"], key="auth_tab")
        
        if auth_tab == "Login":
            with st.sidebar.form("login_form"):
                st.markdown("**🔑 Sign In**")
                username = st.text_input("Username", key="login_username")
                password = st.text_input("Password", type="password", key="login_password")
                
                if st.form_submit_button("Sign In", type="primary"):
                    if username and password and auth_system:
                        success, result = auth_system.authenticate_user(username, password)
                        if success:
                            st.session_state.authenticated_user = result
                            # Load user's saved data
                            user_id = result['user_id']
                            st.session_state.portfolio = auth_system.load_user_portfolio(user_id)
                            st.success("✅ Signed in successfully!")
                            st.rerun()
                        else:
                            st.error(f"❌ {result}")
                    else:
                        st.error("Please enter username and password")
        
        elif auth_tab == "Register":
            with st.sidebar.form("register_form"):
                st.markdown("**📝 Create Account**")
                new_username = st.text_input("Username", key="reg_username")
                new_email = st.text_input("Email", key="reg_email")
                new_password = st.text_input("Password", type="password", key="reg_password")
                confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm")
                
                if st.form_submit_button("Create Account", type="primary"):
                    if new_username and new_email and new_password and confirm_password and auth_system:
                        if new_password == confirm_password:
                            success, result = auth_system.register_user(new_username, new_email, new_password)
                            if success:
                                st.success(f"✅ {result}")
                                st.info("You can now sign in with your credentials")
                            else:
                                st.error(f"❌ {result}")
                        else:
                            st.error("Passwords do not match")
                    else:
                        st.error("Please fill in all fields")
    
    st.sidebar.markdown("---")
    
    # Analysis type selection
    analysis_type = st.sidebar.selectbox(
        "Analysis Type",
        ["📈 Stock Analysis", "💼 Portfolio Management", "📊 Market Overview", "👤 User Profile"],
        index=0
    )
    
    if analysis_type == "📈 Stock Analysis":
        # Stock Analysis Section
        st.subheader("📈 Stock Analysis")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            symbol = st.text_input("Stock Symbol", value="AAPL", help="Enter a stock symbol (e.g., AAPL, MSFT, GOOGL)")
            period = st.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
        
        with col2:
            st.info("💡 **Tip**: Use the sidebar to switch between different analysis types")
        
        if st.button("🔍 Analyze Stock", type="primary"):
            with st.spinner("Fetching stock data..."):
                data = get_stock_data(symbol.upper(), period)
                
                if data is not None:
                    st.success(f"✅ Successfully fetched data for {symbol.upper()}")
                    
                    # Price Chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price', line=dict(color='#667eea')))
                    
                    fig.update_layout(
                        title=f"{symbol.upper()} Stock Price ({period})",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        height=400,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Current Price Info
                    current_price = data['Close'].iloc[-1]
                    prev_price = data['Close'].iloc[-2]
                    change = current_price - prev_price
                    change_percent = (change / prev_price) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Price", f"${current_price:.2f}")
                    with col2:
                        st.metric("Daily Change", f"${change:.2f}")
                    with col3:
                        st.metric("Change %", f"{change_percent:+.2f}%")
                    
                    # Technical Indicators
                    indicators = calculate_technical_indicators(data)
                    if indicators:
                        st.subheader("📊 Technical Indicators")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Moving Averages
                            fig_ma = go.Figure()
                            fig_ma.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
                            fig_ma.add_trace(go.Scatter(x=data.index, y=indicators['SMA_20'], mode='lines', name='SMA 20'))
                            fig_ma.add_trace(go.Scatter(x=data.index, y=indicators['SMA_50'], mode='lines', name='SMA 50'))
                            
                            fig_ma.update_layout(title="Moving Averages", height=300)
                            st.plotly_chart(fig_ma, use_container_width=True)
                        
                        with col2:
                            # RSI
                            fig_rsi = go.Figure()
                            fig_rsi.add_trace(go.Scatter(x=data.index, y=indicators['RSI'], mode='lines', name='RSI'))
                            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                            
                            fig_rsi.update_layout(title="RSI", height=300, yaxis=dict(range=[0, 100]))
                            st.plotly_chart(fig_rsi, use_container_width=True)
                    
                    # Financial Ratios
                    ratios = get_financial_ratios(symbol.upper())
                    if ratios:
                        st.subheader("📋 Financial Ratios")
                        
                        ratio_cols = st.columns(3)
                        for i, (name, value) in enumerate(ratios.items()):
                            with ratio_cols[i % 3]:
                                if isinstance(value, (int, float)):
                                    if 'Ratio' in name or 'Margin' in name:
                                        st.metric(name, f"{value:.2f}")
                                    else:
                                        st.metric(name, f"{value:.2%}")
                                else:
                                    st.metric(name, str(value))
                    
                    # ML Predictions
                    if SKLEARN_AVAILABLE:
                        st.subheader("🤖 ML Price Predictions")
                        with st.spinner("Running ML models..."):
                            predictions, message = predict_price_ml(data)
                            
                            if predictions:
                                pred_cols = st.columns(2)
                                for i, (model, pred) in enumerate(predictions.items()):
                                    with pred_cols[i % 2]:
                                        if isinstance(pred, (int, float)):
                                            st.metric(f"{model} Prediction", f"${pred:.2f}")
                                        else:
                                            st.metric(f"{model}", pred)
                            else:
                                st.warning(message)
                    else:
                        st.info("🤖 ML predictions require scikit-learn (not available in this deployment)")
                
                else:
                    st.error(f"❌ Could not fetch data for {symbol.upper()}")
    
    elif analysis_type == "💼 Portfolio Management":
        # Portfolio Management Section
        st.subheader("💼 Portfolio Management")
        
        # Add new position
        with st.expander("➕ Add New Position", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                symbol = st.text_input("Symbol", placeholder="e.g., AAPL")
            with col2:
                shares = st.number_input("Shares", min_value=0.0, value=1.0)
            with col3:
                cost_basis = st.number_input("Cost Basis ($)", min_value=0.0, value=100.0)
            
            if st.button("Add Position"):
                if symbol and shares and cost_basis:
                    # Get current price
                    data = get_stock_data(symbol.upper(), "1d")
                    if data is not None and not data.empty:
                        current_price = data['Close'].iloc[-1]
                        
                        position = {
                            'symbol': symbol.upper(),
                            'shares': shares,
                            'cost_basis': cost_basis,
                            'current_price': current_price,
                            'date_added': datetime.now().strftime("%Y-%m-%d")
                        }
                        
                        st.session_state.portfolio.append(position)
                        st.success(f"✅ Added {shares} shares of {symbol.upper()}")
                        
                        # Auto-save if user is signed in
                        if st.session_state.authenticated_user and auth_system:
                            user_id = st.session_state.authenticated_user['user_id']
                            auth_system.save_user_portfolio(user_id, st.session_state.portfolio)
                            st.info("💾 Portfolio auto-saved")
                        
                        st.rerun()
                    else:
                        st.error(f"❌ Could not fetch current price for {symbol}")
                else:
                    st.error("❌ Please fill in all fields")
        
        # Display portfolio
        if st.session_state.portfolio:
            st.subheader("📊 Your Portfolio")
            
            total_value = 0
            total_cost = 0
            
            for i, position in enumerate(st.session_state.portfolio):
                col1, col2, col3, col4, col5, col6 = st.columns([2, 1, 1, 1, 1, 1])
                
                with col1:
                    st.write(f"**{position['symbol']}**")
                    st.caption(f"Added: {position['date_added']}")
                
                with col2:
                    st.write(f"{position['shares']} shares")
                
                with col3:
                    st.write(f"${position['cost_basis']:.2f}")
                
                with col4:
                    st.write(f"${position['current_price']:.2f}")
                
                with col5:
                    pnl = (position['current_price'] - position['cost_basis']) * position['shares']
                    pnl_percent = (position['current_price'] / position['cost_basis'] - 1) * 100
                    color = "🟢" if pnl >= 0 else "🔴"
                    st.write(f"{color} **${pnl:.2f}**")
                    st.write(f"({pnl_percent:+.1f}%)")
                
                with col6:
                    if st.button("❌", key=f"remove_{i}"):
                        st.session_state.portfolio.pop(i)
                        
                        # Auto-save if user is signed in
                        if st.session_state.authenticated_user and auth_system:
                            user_id = st.session_state.authenticated_user['user_id']
                            auth_system.save_user_portfolio(user_id, st.session_state.portfolio)
                            st.info("💾 Portfolio auto-saved")
                        
                        st.rerun()
                
                # Calculate totals
                position_value = position['current_price'] * position['shares']
                position_cost = position['cost_basis'] * position['shares']
                total_value += position_value
                total_cost += position_cost
            
            # Portfolio Summary
            st.subheader("📈 Portfolio Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Value", f"${total_value:,.2f}")
            with col2:
                st.metric("Total Cost", f"${total_cost:,.2f}")
            with col3:
                total_pnl = total_value - total_cost
                st.metric("Total P&L", f"${total_pnl:,.2f}")
            with col4:
                total_pnl_percent = (total_value - total_cost) / total_cost * 100 if total_cost > 0 else 0
                st.metric("Total P&L %", f"{total_pnl_percent:+.2f}%")
        else:
            st.info("📝 No positions in portfolio. Add some stocks to get started!")
    
    elif analysis_type == "📊 Market Overview":
        # Market Overview Section
        st.subheader("📊 Market Overview")
        
        indices_config = {
            '^GSPC': 'S&P 500',
            '^IXIC': 'NASDAQ',
            '^DJI': 'DOW',
            '^VIX': 'VIX'
        }
        
        # Fetch market data
        with st.spinner("Fetching market data..."):
            market_data = {}
            for symbol, name in indices_config.items():
                try:
                    data = get_stock_data(symbol, "1d")
                    if data is not None and not data.empty:
                        current_price = data['Close'].iloc[-1]
                        prev_price = data['Open'].iloc[-1]
                        change = current_price - prev_price
                        change_percent = (change / prev_price) * 100
                        
                        market_data[symbol] = {
                            'name': name,
                            'price': current_price,
                            'change': change,
                            'change_percent': change_percent
                        }
                except Exception as e:
                    st.warning(f"Could not fetch {name} data: {str(e)}")
        
        # Display market data
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
                    else:
                        st.error(f"❌ {name} data unavailable")
            
            # Market summary
            st.subheader("📈 Market Summary")
            
            # Calculate overall market direction
            total_change = sum(data['change_percent'] for data in market_data.values() if data)
            avg_change = total_change / len(market_data) if market_data else 0
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                market_direction = "📈 Bullish" if avg_change > 0 else "📉 Bearish" if avg_change < 0 else "➡️ Neutral"
                st.metric("Market Direction", market_direction)
            
            with col2:
                st.metric("Average Change", f"{avg_change:+.2f}%")
            
            with col3:
                st.metric("Data Sources", "Yahoo Finance")
        
        else:
            st.error("❌ Could not fetch market data")
    
    elif analysis_type == "👤 User Profile":
        # User Profile Section
        if st.session_state.authenticated_user:
            user_info = st.session_state.authenticated_user
            st.subheader(f"👤 Profile: {user_info['username']}")
            
            # User Information
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Email:** {user_info['email']}")
                st.info(f"**User ID:** {user_info['user_id']}")
            
            with col2:
                st.subheader("📊 Account Features")
                st.markdown("""
                - ✅ **Personal Portfolio**: Auto-saved across sessions
                - ✅ **Data Persistence**: All your analysis data is saved
                - ✅ **User Preferences**: Customize your dashboard
                - ✅ **Secure Authentication**: Your data is protected
                """)
        else:
            st.subheader("👤 User Profile")
            st.info("🔐 Please sign in to access your profile and personal metrics")
            st.markdown("""
            **Features available when signed in:**
            - 💼 **Personal Portfolio**: Auto-saved across sessions
            - 📊 **Business Metrics**: Track your personal business KPIs
            - ⚙️ **Preferences**: Customize your dashboard
            - 📈 **Data Persistence**: All your analysis data is saved
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("**Financial Analyzer Pro - Enhanced** | Built with Streamlit")
    st.markdown("*Advanced financial analysis with portfolio management and authentication*")

if __name__ == "__main__":
    main()
