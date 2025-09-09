"""
User Authentication and Portfolio Management System
"""
import streamlit as st
import hashlib
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd

class UserAuth:
    """User authentication and session management"""
    
    def __init__(self):
        self.users_file = "users.json"
        self.sessions_file = "sessions.json"
        self.portfolios_file = "portfolios.json"
        self.load_data()
    
    def load_data(self):
        """Load user data from files"""
        # Load users
        if os.path.exists(self.users_file):
            with open(self.users_file, 'r') as f:
                self.users = json.load(f)
        else:
            self.users = {}
        
        # Load sessions
        if os.path.exists(self.sessions_file):
            with open(self.sessions_file, 'r') as f:
                self.sessions = json.load(f)
        else:
            self.sessions = {}
        
        # Load portfolios
        if os.path.exists(self.portfolios_file):
            with open(self.portfolios_file, 'r') as f:
                self.portfolios = json.load(f)
        else:
            self.portfolios = {}
    
    def save_data(self):
        """Save user data to files"""
        with open(self.users_file, 'w') as f:
            json.dump(self.users, f, indent=2)
        with open(self.sessions_file, 'w') as f:
            json.dump(self.sessions, f, indent=2)
        with open(self.portfolios_file, 'w') as f:
            json.dump(self.portfolios, f, indent=2)
    
    def hash_password(self, password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def register_user(self, username: str, email: str, password: str) -> bool:
        """Register a new user"""
        if username in self.users:
            return False
        
        hashed_password = self.hash_password(password)
        self.users[username] = {
            'email': email,
            'password': hashed_password,
            'created_at': datetime.now().isoformat(),
            'last_login': None
        }
        
        # Create default portfolio
        self.portfolios[username] = {
            'stocks': [],
            'watchlist': [],
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        }
        
        self.save_data()
        return True
    
    def authenticate_user(self, username: str, password: str) -> bool:
        """Authenticate user login"""
        if username not in self.users:
            return False
        
        hashed_password = self.hash_password(password)
        if self.users[username]['password'] == hashed_password:
            self.users[username]['last_login'] = datetime.now().isoformat()
            self.save_data()
            return True
        return False
    
    def create_session(self, username: str) -> str:
        """Create a new session for user"""
        session_id = hashlib.sha256(f"{username}{datetime.now()}".encode()).hexdigest()
        self.sessions[session_id] = {
            'username': username,
            'created_at': datetime.now().isoformat(),
            'expires_at': (datetime.now() + timedelta(hours=24)).isoformat()
        }
        self.save_data()
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[str]:
        """Validate session and return username if valid"""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        expires_at = datetime.fromisoformat(session['expires_at'])
        
        if datetime.now() > expires_at:
            del self.sessions[session_id]
            self.save_data()
            return None
        
        return session['username']
    
    def logout(self, session_id: str):
        """Logout user by removing session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            self.save_data()

class PortfolioManager:
    """Portfolio and watchlist management"""
    
    def __init__(self, auth_system: UserAuth):
        self.auth = auth_system
    
    def add_stock_to_portfolio(self, username: str, ticker: str, shares: int, purchase_price: float, purchase_date: str = None) -> bool:
        """Add stock to user's portfolio"""
        if username not in self.auth.portfolios:
            return False
        
        if purchase_date is None:
            purchase_date = datetime.now().strftime('%Y-%m-%d')
        
        stock_entry = {
            'ticker': ticker.upper(),
            'shares': shares,
            'purchase_price': purchase_price,
            'purchase_date': purchase_date,
            'added_at': datetime.now().isoformat()
        }
        
        self.auth.portfolios[username]['stocks'].append(stock_entry)
        self.auth.portfolios[username]['last_updated'] = datetime.now().isoformat()
        self.auth.save_data()
        return True
    
    def remove_stock_from_portfolio(self, username: str, ticker: str) -> bool:
        """Remove stock from user's portfolio"""
        if username not in self.auth.portfolios:
            return False
        
        portfolio = self.auth.portfolios[username]['stocks']
        self.auth.portfolios[username]['stocks'] = [
            stock for stock in portfolio if stock['ticker'] != ticker.upper()
        ]
        self.auth.portfolios[username]['last_updated'] = datetime.now().isoformat()
        self.auth.save_data()
        return True
    
    def add_to_watchlist(self, username: str, ticker: str) -> bool:
        """Add stock to user's watchlist"""
        if username not in self.auth.portfolios:
            return False
        
        if ticker.upper() not in self.auth.portfolios[username]['watchlist']:
            self.auth.portfolios[username]['watchlist'].append(ticker.upper())
            self.auth.portfolios[username]['last_updated'] = datetime.now().isoformat()
            self.auth.save_data()
        return True
    
    def remove_from_watchlist(self, username: str, ticker: str) -> bool:
        """Remove stock from user's watchlist"""
        if username not in self.auth.portfolios:
            return False
        
        if ticker.upper() in self.auth.portfolios[username]['watchlist']:
            self.auth.portfolios[username]['watchlist'].remove(ticker.upper())
            self.auth.portfolios[username]['last_updated'] = datetime.now().isoformat()
            self.auth.save_data()
        return True
    
    def get_portfolio(self, username: str) -> Dict:
        """Get user's portfolio"""
        if username not in self.auth.portfolios:
            return {'stocks': [], 'watchlist': []}
        
        return self.auth.portfolios[username]
    
    def calculate_portfolio_value(self, username: str, current_prices: Dict[str, float]) -> Dict:
        """Calculate portfolio value with current prices"""
        portfolio = self.get_portfolio(username)
        total_value = 0
        total_cost = 0
        stock_values = []
        
        for stock in portfolio['stocks']:
            ticker = stock['ticker']
            shares = stock['shares']
            purchase_price = stock['purchase_price']
            current_price = current_prices.get(ticker, 0)
            
            cost = shares * purchase_price
            value = shares * current_price
            gain_loss = value - cost
            gain_loss_pct = (gain_loss / cost * 100) if cost > 0 else 0
            
            stock_values.append({
                'ticker': ticker,
                'shares': shares,
                'purchase_price': purchase_price,
                'current_price': current_price,
                'cost': cost,
                'value': value,
                'gain_loss': gain_loss,
                'gain_loss_pct': gain_loss_pct
            })
            
            total_value += value
            total_cost += cost
        
        total_gain_loss = total_value - total_cost
        total_gain_loss_pct = (total_gain_loss / total_cost * 100) if total_cost > 0 else 0
        
        return {
            'stocks': stock_values,
            'watchlist': portfolio['watchlist'],
            'total_value': total_value,
            'total_cost': total_cost,
            'total_gain_loss': total_gain_loss,
            'total_gain_loss_pct': total_gain_loss_pct,
            'last_updated': datetime.now().isoformat()
        }

# Initialize authentication system
@st.cache_resource
def get_auth_system():
    return UserAuth()

@st.cache_resource
def get_portfolio_manager():
    return PortfolioManager(get_auth_system())

def show_login_form():
    """Display login/register form"""
    st.markdown("### 🔐 User Authentication")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            submit = st.form_submit_button("Login")
            
            if submit:
                auth = get_auth_system()
                if auth.authenticate_user(username, password):
                    session_id = auth.create_session(username)
                    st.session_state['session_id'] = session_id
                    st.session_state['username'] = username
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")
    
    with tab2:
        with st.form("register_form"):
            new_username = st.text_input("Username", key="register_username")
            email = st.text_input("Email", key="register_email")
            new_password = st.text_input("Password", type="password", key="register_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
            submit = st.form_submit_button("Register")
            
            if submit:
                if new_password != confirm_password:
                    st.error("❌ Passwords do not match")
                elif len(new_password) < 6:
                    st.error("❌ Password must be at least 6 characters")
                else:
                    auth = get_auth_system()
                    if auth.register_user(new_username, email, new_password):
                        st.success("✅ Registration successful! Please login.")
                    else:
                        st.error("❌ Username already exists")

def show_user_dashboard():
    """Display user dashboard with portfolio"""
    if 'username' not in st.session_state:
        return
    
    username = st.session_state['username']
    portfolio_manager = get_portfolio_manager()
    
    st.markdown(f"### 👋 Welcome, {username}!")
    
    # Portfolio overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 View Portfolio"):
            st.session_state['current_page'] = 'portfolio'
            st.rerun()
    
    with col2:
        if st.button("👀 Watchlist"):
            st.session_state['current_page'] = 'watchlist'
            st.rerun()
    
    with col3:
        if st.button("➕ Add Stock"):
            st.session_state['current_page'] = 'add_stock'
            st.rerun()
    
    # Quick portfolio summary
    portfolio = portfolio_manager.get_portfolio(username)
    if portfolio['stocks']:
        st.markdown("#### 📈 Portfolio Summary")
        st.write(f"**Stocks held:** {len(portfolio['stocks'])}")
        st.write(f"**Watchlist:** {len(portfolio['watchlist'])}")
    else:
        st.info("💡 Add stocks to your portfolio to get started!")

def show_portfolio_page():
    """Display portfolio management page"""
    if 'username' not in st.session_state:
        return
    
    username = st.session_state['username']
    portfolio_manager = get_portfolio_manager()
    
    st.markdown("### 📊 Your Portfolio")
    
    # Add stock form
    with st.expander("➕ Add New Stock"):
        with st.form("add_stock_form"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                ticker = st.text_input("Ticker Symbol", placeholder="AAPL")
            with col2:
                shares = st.number_input("Shares", min_value=1, value=1)
            with col3:
                purchase_price = st.number_input("Purchase Price", min_value=0.01, value=0.01, step=0.01)
            with col4:
                purchase_date = st.date_input("Purchase Date", value=datetime.now().date())
            
            if st.form_submit_button("Add to Portfolio"):
                if ticker:
                    success = portfolio_manager.add_stock_to_portfolio(
                        username, ticker, shares, purchase_price, purchase_date.strftime('%Y-%m-%d')
                    )
                    if success:
                        st.success(f"✅ Added {ticker} to portfolio!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to add stock")
    
    # Display portfolio
    portfolio = portfolio_manager.get_portfolio(username)
    
    if portfolio['stocks']:
        st.markdown("#### 📈 Current Holdings")
        
        # Create portfolio DataFrame
        portfolio_data = []
        for stock in portfolio['stocks']:
            portfolio_data.append({
                'Ticker': stock['ticker'],
                'Shares': stock['shares'],
                'Purchase Price': f"${stock['purchase_price']:.2f}",
                'Purchase Date': stock['purchase_date'],
                'Added': stock['added_at'][:10]
            })
        
        df = pd.DataFrame(portfolio_data)
        st.dataframe(df, use_container_width=True)
        
        # Remove stock option
        st.markdown("#### 🗑️ Remove Stock")
        remove_ticker = st.selectbox("Select stock to remove", [""] + [stock['ticker'] for stock in portfolio['stocks']])
        if st.button("Remove Stock") and remove_ticker:
            portfolio_manager.remove_stock_from_portfolio(username, remove_ticker)
            st.success(f"✅ Removed {remove_ticker} from portfolio!")
            st.rerun()
    else:
        st.info("💡 Your portfolio is empty. Add some stocks to get started!")

def show_watchlist_page():
    """Display watchlist management page"""
    if 'username' not in st.session_state:
        return
    
    username = st.session_state['username']
    portfolio_manager = get_portfolio_manager()
    
    st.markdown("### 👀 Your Watchlist")
    
    # Add to watchlist
    with st.expander("➕ Add to Watchlist"):
        with st.form("add_watchlist_form"):
            ticker = st.text_input("Ticker Symbol", placeholder="AAPL", key="watchlist_ticker")
            if st.form_submit_button("Add to Watchlist"):
                if ticker:
                    success = portfolio_manager.add_to_watchlist(username, ticker)
                    if success:
                        st.success(f"✅ Added {ticker} to watchlist!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to add to watchlist")
    
    # Display watchlist
    portfolio = portfolio_manager.get_portfolio(username)
    
    if portfolio['watchlist']:
        st.markdown("#### 📋 Watchlist")
        
        # Create watchlist DataFrame
        watchlist_data = []
        for ticker in portfolio['watchlist']:
            watchlist_data.append({'Ticker': ticker})
        
        df = pd.DataFrame(watchlist_data)
        st.dataframe(df, use_container_width=True)
        
        # Remove from watchlist
        st.markdown("#### 🗑️ Remove from Watchlist")
        remove_ticker = st.selectbox("Select ticker to remove", [""] + portfolio['watchlist'], key="remove_watchlist")
        if st.button("Remove from Watchlist") and remove_ticker:
            portfolio_manager.remove_from_watchlist(username, remove_ticker)
            st.success(f"✅ Removed {remove_ticker} from watchlist!")
            st.rerun()
    else:
        st.info("💡 Your watchlist is empty. Add some stocks to watch!")

def logout_user():
    """Logout current user"""
    if 'session_id' in st.session_state:
        auth = get_auth_system()
        auth.logout(st.session_state['session_id'])
    
    # Clear session state
    for key in ['session_id', 'username', 'current_page']:
        if key in st.session_state:
            del st.session_state[key]
    
    st.success("✅ Logged out successfully!")
    st.rerun()
