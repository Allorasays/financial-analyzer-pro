"""
Authentication System for Financial Analyzer Pro - Phase 3
Handles user login, registration, session management, and security
"""

import streamlit as st
import re
from datetime import datetime, timedelta
from typing import Optional, Dict
from database_manager import DatabaseManager

class AuthSystem:
    def __init__(self):
        self.db = DatabaseManager()
        self.init_session_state()
    
    def init_session_state(self):
        """Initialize session state variables"""
        if 'user' not in st.session_state:
            st.session_state.user = None
        if 'session_token' not in st.session_state:
            st.session_state.session_token = None
        if 'is_authenticated' not in st.session_state:
            st.session_state.is_authenticated = False
        if 'is_guest' not in st.session_state:
            st.session_state.is_guest = True  # Default to guest mode
        if 'login_attempts' not in st.session_state:
            st.session_state.login_attempts = 0
        if 'last_attempt' not in st.session_state:
            st.session_state.last_attempt = None
    
    def validate_email(self, email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def validate_password(self, password: str) -> Dict[str, any]:
        """Validate password strength"""
        result = {
            'is_valid': True,
            'errors': [],
            'strength': 'weak'
        }
        
        if len(password) < 8:
            result['is_valid'] = False
            result['errors'].append("Password must be at least 8 characters long")
        
        if not re.search(r'[A-Z]', password):
            result['errors'].append("Password must contain at least one uppercase letter")
        
        if not re.search(r'[a-z]', password):
            result['errors'].append("Password must contain at least one lowercase letter")
        
        if not re.search(r'\d', password):
            result['errors'].append("Password must contain at least one number")
        
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            result['errors'].append("Password must contain at least one special character")
        
        # Calculate strength
        if len(password) >= 12 and len(result['errors']) <= 1:
            result['strength'] = 'strong'
        elif len(password) >= 10 and len(result['errors']) <= 2:
            result['strength'] = 'medium'
        
        if len(result['errors']) > 2:
            result['is_valid'] = False
        
        return result
    
    def validate_username(self, username: str) -> Dict[str, any]:
        """Validate username"""
        result = {
            'is_valid': True,
            'errors': []
        }
        
        if len(username) < 3:
            result['is_valid'] = False
            result['errors'].append("Username must be at least 3 characters long")
        
        if len(username) > 20:
            result['is_valid'] = False
            result['errors'].append("Username must be less than 20 characters")
        
        if not re.match(r'^[a-zA-Z0-9_]+$', username):
            result['is_valid'] = False
            result['errors'].append("Username can only contain letters, numbers, and underscores")
        
        return result
    
    def show_login_page(self):
        """Display login page"""
        st.markdown("""
        <div style="max-width: 400px; margin: 0 auto; padding: 2rem; background: white; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h2 style="text-align: center; color: #333; margin-bottom: 2rem;">🔐 Sign In</h2>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            username_email = st.text_input("Username or Email", placeholder="Enter your username or email")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                login_clicked = st.form_submit_button("Sign In", type="primary", use_container_width=True)
            with col2:
                register_clicked = st.form_submit_button("Create Account", use_container_width=True)
            with col3:
                guest_clicked = st.form_submit_button("Continue as Guest", use_container_width=True)
        
        if login_clicked:
            self.handle_login(username_email, password)
        
        if register_clicked:
            st.session_state.show_register = True
            st.rerun()
        
        if guest_clicked:
            self.enable_guest_mode()
            st.session_state.show_login = False
            st.session_state.show_register = False
            st.rerun()
    
    def show_register_page(self):
        """Display registration page"""
        st.markdown("""
        <div style="max-width: 400px; margin: 0 auto; padding: 2rem; background: white; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h2 style="text-align: center; color: #333; margin-bottom: 2rem;">📝 Create Account</h2>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("register_form"):
            username = st.text_input("Username", placeholder="Choose a username")
            email = st.text_input("Email", placeholder="Enter your email address")
            full_name = st.text_input("Full Name", placeholder="Enter your full name")
            password = st.text_input("Password", type="password", placeholder="Create a strong password")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                register_clicked = st.form_submit_button("Create Account", type="primary", use_container_width=True)
            with col2:
                back_clicked = st.form_submit_button("Back to Login", use_container_width=True)
        
        if register_clicked:
            self.handle_registration(username, email, full_name, password, confirm_password)
        
        if back_clicked:
            st.session_state.show_register = False
            st.rerun()
    
    def handle_login(self, username_email: str, password: str):
        """Handle user login"""
        # Rate limiting
        now = datetime.now()
        if st.session_state.last_attempt:
            time_diff = (now - st.session_state.last_attempt).seconds
            if time_diff < 60 and st.session_state.login_attempts >= 3:
                st.error("Too many login attempts. Please wait 1 minute before trying again.")
                return
        
        if not username_email or not password:
            st.error("Please enter both username/email and password")
            return
        
        # Authenticate user
        user = self.db.authenticate_user(username_email, password)
        
        if user:
            # Create session
            session_token = self.db.create_session(user['id'])
            
            if session_token:
                # Update session state
                st.session_state.user = user
                st.session_state.session_token = session_token
                st.session_state.is_authenticated = True
                st.session_state.login_attempts = 0
                st.session_state.last_attempt = None
                
                st.success(f"Welcome back, {user['full_name'] or user['username']}!")
                st.rerun()
            else:
                st.error("Login failed. Please try again.")
        else:
            st.session_state.login_attempts += 1
            st.session_state.last_attempt = now
            st.error("Invalid username/email or password")
    
    def handle_registration(self, username: str, email: str, full_name: str, password: str, confirm_password: str):
        """Handle user registration"""
        # Validate inputs
        if not all([username, email, full_name, password, confirm_password]):
            st.error("Please fill in all fields")
            return
        
        # Validate email
        if not self.validate_email(email):
            st.error("Please enter a valid email address")
            return
        
        # Validate username
        username_validation = self.validate_username(username)
        if not username_validation['is_valid']:
            for error in username_validation['errors']:
                st.error(error)
            return
        
        # Validate password
        password_validation = self.validate_password(password)
        if not password_validation['is_valid']:
            for error in password_validation['errors']:
                st.error(error)
            return
        
        # Check password confirmation
        if password != confirm_password:
            st.error("Passwords do not match")
            return
        
        # Create user
        if self.db.create_user(username, email, password, full_name):
            st.success("Account created successfully! Please sign in.")
            st.session_state.show_register = False
            st.rerun()
        else:
            st.error("Username or email already exists. Please choose different credentials.")
    
    def show_user_menu(self):
        """Display user menu in sidebar"""
        if st.session_state.is_authenticated and st.session_state.user:
            user = st.session_state.user
            
            st.sidebar.markdown("---")
            st.sidebar.markdown(f"**👤 Welcome, {user['full_name'] or user['username']}**")
            
            # User menu
            menu_option = st.sidebar.selectbox(
                "Account Menu",
                ["Dashboard", "My Portfolios", "My Watchlists", "Settings", "Sign Out"]
            )
            
            if menu_option == "Sign Out":
                self.logout()
            elif menu_option == "Settings":
                self.show_settings_page()
            elif menu_option == "My Portfolios":
                st.session_state.current_page = "My Portfolios"
            elif menu_option == "My Watchlists":
                st.session_state.current_page = "My Watchlists"
            else:
                st.session_state.current_page = "Dashboard"
    
    def show_settings_page(self):
        """Display user settings page"""
        st.header("⚙️ Account Settings")
        
        if st.session_state.is_authenticated and st.session_state.user:
            user_id = st.session_state.user['id']
            
            # Get current preferences
            preferences = self.db.get_user_preferences(user_id)
            
            with st.form("settings_form"):
                st.subheader("Personal Information")
                full_name = st.text_input("Full Name", value=st.session_state.user.get('full_name', ''))
                email = st.text_input("Email", value=st.session_state.user.get('email', ''), disabled=True)
                username = st.text_input("Username", value=st.session_state.user.get('username', ''), disabled=True)
                
                st.subheader("Preferences")
                theme = st.selectbox("Theme", ["light", "dark"], index=0 if preferences.get('theme') == 'light' else 1)
                default_timeframe = st.selectbox("Default Timeframe", 
                    ["1mo", "3mo", "6mo", "1y", "2y", "5y"], 
                    index=["1mo", "3mo", "6mo", "1y", "2y", "5y"].index(preferences.get('default_timeframe', '1mo')))
                
                default_symbols = st.text_area("Default Symbols (one per line)", 
                    value='\n'.join(preferences.get('default_symbols', [])),
                    help="Enter one stock symbol per line")
                
                if st.form_submit_button("Save Settings", type="primary"):
                    # Update preferences
                    new_preferences = {
                        'theme': theme,
                        'default_timeframe': default_timeframe,
                        'default_symbols': [s.strip().upper() for s in default_symbols.split('\n') if s.strip()],
                        'notification_settings': preferences.get('notification_settings', {}),
                        'dashboard_layout': preferences.get('dashboard_layout', {})
                    }
                    
                    if self.db.update_user_preferences(user_id, new_preferences):
                        st.success("Settings saved successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to save settings. Please try again.")
    
    def logout(self):
        """Handle user logout"""
        # Clear session state
        st.session_state.user = None
        st.session_state.session_token = None
        st.session_state.is_authenticated = False
        st.session_state.current_page = None
        
        st.success("You have been signed out successfully!")
        st.rerun()
    
    def require_auth(self, func):
        """Decorator to require authentication for functions"""
        def wrapper(*args, **kwargs):
            if not st.session_state.is_authenticated:
                st.warning("Please sign in to access this feature.")
                return None
            return func(*args, **kwargs)
        return wrapper
    
    def get_current_user(self) -> Optional[Dict]:
        """Get current authenticated user"""
        if st.session_state.is_authenticated and st.session_state.user:
            return st.session_state.user
        return None
    
    def is_authenticated(self) -> bool:
        """Check if user is authenticated"""
        return st.session_state.is_authenticated
    
    def get_user_preferences(self) -> Dict:
        """Get current user preferences"""
        if st.session_state.is_authenticated and st.session_state.user:
            return self.db.get_user_preferences(st.session_state.user['id'])
        return {
            'theme': 'light',
            'default_timeframe': '1mo',
            'default_symbols': ['AAPL', 'MSFT', 'GOOGL'],
            'notification_settings': {},
            'dashboard_layout': {}
        }
    
    def enable_guest_mode(self):
        """Enable guest mode for research without login"""
        st.session_state.is_guest = True
        st.session_state.is_authenticated = False
        st.session_state.user = None
        st.session_state.session_token = None
    
    def is_guest_mode(self) -> bool:
        """Check if user is in guest mode"""
        return st.session_state.is_guest and not st.session_state.is_authenticated
    
    def require_auth_for_feature(self, feature_name: str) -> bool:
        """Check if user needs to be authenticated for a specific feature"""
        auth_required_features = [
            'portfolio_manager', 'watchlist_manager', 'save_portfolio', 
            'save_watchlist', 'user_preferences', 'export_data'
        ]
        return feature_name in auth_required_features
    
    def show_auth_prompt(self, feature_name: str):
        """Show authentication prompt for features that require login"""
        st.warning(f"🔐 **{feature_name}** requires an account. Please sign in to access this feature.")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("Sign In", key=f"signin_{feature_name}"):
                st.session_state.show_login = True
                st.rerun()
        with col2:
            if st.button("Create Account", key=f"register_{feature_name}"):
                st.session_state.show_register = True
                st.rerun()
        with col3:
            if st.button("Continue as Guest", key=f"guest_{feature_name}"):
                st.session_state.show_login = False
                st.session_state.show_register = False
                st.rerun()
    
    def show_optional_auth_header(self):
        """Show optional authentication header in sidebar"""
        if self.is_guest_mode():
            st.sidebar.markdown("---")
            st.sidebar.markdown("""
            <div style="background: #e3f2fd; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
                <h4>👤 Guest Mode</h4>
                <p>You're browsing as a guest. Sign in to save portfolios and access personalized features.</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.sidebar.columns(2)
            with col1:
                if st.button("Sign In", key="header_signin"):
                    st.session_state.show_login = True
                    st.rerun()
            with col2:
                if st.button("Register", key="header_register"):
                    st.session_state.show_register = True
                    st.rerun()
        else:
            # Show user menu for authenticated users
            self.show_user_menu()
