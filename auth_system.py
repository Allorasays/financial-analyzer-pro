<<<<<<< Updated upstream
import streamlit as st
import sqlite3
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import re

class AuthSystem:
    def __init__(self, db_path: str = 'financial_analyzer.db'):
        self.db_path = db_path
        self.init_auth_tables()
    
    def init_auth_tables(self):
        """Initialize authentication-related database tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Users table (if not exists)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                )
            ''')
            
            # Sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    session_token TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    is_active BOOLEAN DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            conn.commit()
            conn.close()
        except Exception as e:
            st.error(f"Database initialization error: {str(e)}")
    
    def hash_password(self, password: str, salt: str) -> str:
        """Hash password with salt"""
        return hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000).hex()
    
    def generate_salt(self) -> str:
        """Generate a random salt"""
        return secrets.token_hex(16)
    
    def validate_email(self, email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def register_user(self, username: str, email: str, password: str) -> Dict[str, Any]:
        """Register a new user"""
        try:
            # Validate inputs
            if not username or len(username) < 3:
                return {'success': False, 'message': 'Username must be at least 3 characters long'}
            
            if not self.validate_email(email):
                return {'success': False, 'message': 'Invalid email format'}
            
            if len(password) < 8:
                return {'success': False, 'message': 'Password must be at least 8 characters long'}
            
            # Check if user already exists
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT id FROM users WHERE username = ? OR email = ?', (username, email))
            if cursor.fetchone():
                conn.close()
                return {'success': False, 'message': 'Username or email already exists'}
            
            # Create new user
            salt = self.generate_salt()
            password_hash = self.hash_password(password, salt)
            
            cursor.execute('''
                INSERT INTO users (username, email, password_hash, salt)
                VALUES (?, ?, ?, ?)
            ''', (username, email, password_hash, salt))
            
            user_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return {'success': True, 'message': 'User registered successfully', 'user_id': user_id}
            
        except Exception as e:
            return {'success': False, 'message': f'Registration failed: {str(e)}'}
    
    def login_user(self, username: str, password: str) -> Dict[str, Any]:
        """Login user and create session"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get user data
            cursor.execute('''
                SELECT id, username, email, password_hash, salt, is_active
                FROM users WHERE username = ? OR email = ?
            ''', (username, username))
            
            user = cursor.fetchone()
            if not user:
                conn.close()
                return {'success': False, 'message': 'Invalid username or password'}
            
            user_id, db_username, email, db_password_hash, salt, is_active = user
            
            if not is_active:
                conn.close()
                return {'success': False, 'message': 'Account is deactivated'}
            
            # Verify password
            password_hash = self.hash_password(password, salt)
            if password_hash != db_password_hash:
                conn.close()
                return {'success': False, 'message': 'Invalid username or password'}
            
            # Create session
            session_token = secrets.token_urlsafe(32)
            expires_at = datetime.now() + timedelta(days=30)
            
            cursor.execute('''
                INSERT INTO user_sessions (user_id, session_token, expires_at)
                VALUES (?, ?, ?)
            ''', (user_id, session_token, expires_at))
            
            # Update last login
            cursor.execute('''
                UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?
            ''', (user_id,))
            
            conn.commit()
            conn.close()
            
            return {
                'success': True,
                'message': 'Login successful',
                'user_id': user_id,
                'username': db_username,
                'email': email,
                'session_token': session_token
            }
            
        except Exception as e:
            return {'success': False, 'message': f'Login failed: {str(e)}'}
    
    def logout_user(self, session_token: str) -> bool:
        """Logout user by deactivating session"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('UPDATE user_sessions SET is_active = 0 WHERE session_token = ?', (session_token,))
            conn.commit()
            conn.close()
            
            return True
        except Exception as e:
            return False

def init_session_state():
    """Initialize session state for authentication"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_data' not in st.session_state:
        st.session_state.user_data = None
    if 'auth_system' not in st.session_state:
        st.session_state.auth_system = AuthSystem()

def show_login_page():
    """Show login/signup page"""
    st.markdown("""
    <div class="main-header">
        <h1>🔐 Financial Analyzer Pro</h1>
        <p>Sign in to access your personalized financial dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        st.subheader("Login to Your Account")
        
        with st.form("login_form"):
            username = st.text_input("Username or Email", placeholder="Enter your username or email")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submit_button = st.form_submit_button("Login", use_container_width=True)
            
            if submit_button:
                if username and password:
                    result = st.session_state.auth_system.login_user(username, password)
                    if result['success']:
                        st.session_state.authenticated = True
                        st.session_state.user_data = result
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error(result['message'])
                else:
                    st.error("Please fill in all fields")
    
    with tab2:
        st.subheader("Create New Account")
        
        with st.form("signup_form"):
            new_username = st.text_input("Username", placeholder="Choose a username (min 3 characters)")
            new_email = st.text_input("Email", placeholder="Enter your email address")
            new_password = st.text_input("Password", type="password", placeholder="Create a strong password")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
            submit_button = st.form_submit_button("Sign Up", use_container_width=True)
            
            if submit_button:
                if new_username and new_email and new_password and confirm_password:
                    if new_password != confirm_password:
                        st.error("Passwords do not match")
                    else:
                        result = st.session_state.auth_system.register_user(new_username, new_email, new_password)
                        if result['success']:
                            st.success("Account created successfully! Please login.")
                        else:
                            st.error(result['message'])
                else:
                    st.error("Please fill in all fields")

def show_user_menu():
    """Show user menu in sidebar"""
    if st.session_state.authenticated and st.session_state.user_data:
        user_data = st.session_state.user_data
        
        with st.sidebar:
            st.markdown("---")
            st.markdown(f"**Welcome, {user_data['username']}!**")
            
            if st.button("🚪 Logout", use_container_width=True):
                st.session_state.auth_system.logout_user(user_data['session_token'])
                st.session_state.authenticated = False
                st.session_state.user_data = None
                st.success("Logged out successfully!")
                st.rerun()

def require_auth(func):
    """Decorator to require authentication for pages"""
    def wrapper(*args, **kwargs):
        if not st.session_state.authenticated:
            show_login_page()
            return
        return func(*args, **kwargs)
=======
import streamlit as st
import sqlite3
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import re

class AuthSystem:
    def __init__(self, db_path: str = 'financial_analyzer.db'):
        self.db_path = db_path
        self.init_auth_tables()
    
    def init_auth_tables(self):
        """Initialize authentication-related database tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Users table (if not exists)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                )
            ''')
            
            # Sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    session_token TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    is_active BOOLEAN DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            conn.commit()
            conn.close()
        except Exception as e:
            st.error(f"Database initialization error: {str(e)}")
    
    def hash_password(self, password: str, salt: str) -> str:
        """Hash password with salt"""
        return hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000).hex()
    
    def generate_salt(self) -> str:
        """Generate a random salt"""
        return secrets.token_hex(16)
    
    def validate_email(self, email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def register_user(self, username: str, email: str, password: str) -> Dict[str, Any]:
        """Register a new user"""
        try:
            # Validate inputs
            if not username or len(username) < 3:
                return {'success': False, 'message': 'Username must be at least 3 characters long'}
            
            if not self.validate_email(email):
                return {'success': False, 'message': 'Invalid email format'}
            
            if len(password) < 8:
                return {'success': False, 'message': 'Password must be at least 8 characters long'}
            
            # Check if user already exists
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT id FROM users WHERE username = ? OR email = ?', (username, email))
            if cursor.fetchone():
                conn.close()
                return {'success': False, 'message': 'Username or email already exists'}
            
            # Create new user
            salt = self.generate_salt()
            password_hash = self.hash_password(password, salt)
            
            cursor.execute('''
                INSERT INTO users (username, email, password_hash, salt)
                VALUES (?, ?, ?, ?)
            ''', (username, email, password_hash, salt))
            
            user_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return {'success': True, 'message': 'User registered successfully', 'user_id': user_id}
            
        except Exception as e:
            return {'success': False, 'message': f'Registration failed: {str(e)}'}
    
    def login_user(self, username: str, password: str) -> Dict[str, Any]:
        """Login user and create session"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get user data
            cursor.execute('''
                SELECT id, username, email, password_hash, salt, is_active
                FROM users WHERE username = ? OR email = ?
            ''', (username, username))
            
            user = cursor.fetchone()
            if not user:
                conn.close()
                return {'success': False, 'message': 'Invalid username or password'}
            
            user_id, db_username, email, db_password_hash, salt, is_active = user
            
            if not is_active:
                conn.close()
                return {'success': False, 'message': 'Account is deactivated'}
            
            # Verify password
            password_hash = self.hash_password(password, salt)
            if password_hash != db_password_hash:
                conn.close()
                return {'success': False, 'message': 'Invalid username or password'}
            
            # Create session
            session_token = secrets.token_urlsafe(32)
            expires_at = datetime.now() + timedelta(days=30)
            
            cursor.execute('''
                INSERT INTO user_sessions (user_id, session_token, expires_at)
                VALUES (?, ?, ?)
            ''', (user_id, session_token, expires_at))
            
            # Update last login
            cursor.execute('''
                UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?
            ''', (user_id,))
            
            conn.commit()
            conn.close()
            
            return {
                'success': True,
                'message': 'Login successful',
                'user_id': user_id,
                'username': db_username,
                'email': email,
                'session_token': session_token
            }
            
        except Exception as e:
            return {'success': False, 'message': f'Login failed: {str(e)}'}
    
    def logout_user(self, session_token: str) -> bool:
        """Logout user by deactivating session"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('UPDATE user_sessions SET is_active = 0 WHERE session_token = ?', (session_token,))
            conn.commit()
            conn.close()
            
            return True
        except Exception as e:
            return False

def init_session_state():
    """Initialize session state for authentication"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_data' not in st.session_state:
        st.session_state.user_data = None
    if 'auth_system' not in st.session_state:
        st.session_state.auth_system = AuthSystem()

def show_login_page():
    """Show login/signup page"""
    st.markdown("""
    <div class="main-header">
        <h1>🔐 Financial Analyzer Pro</h1>
        <p>Sign in to access your personalized financial dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        st.subheader("Login to Your Account")
        
        with st.form("login_form"):
            username = st.text_input("Username or Email", placeholder="Enter your username or email")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submit_button = st.form_submit_button("Login", use_container_width=True)
            
            if submit_button:
                if username and password:
                    result = st.session_state.auth_system.login_user(username, password)
                    if result['success']:
                        st.session_state.authenticated = True
                        st.session_state.user_data = result
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error(result['message'])
                else:
                    st.error("Please fill in all fields")
    
    with tab2:
        st.subheader("Create New Account")
        
        with st.form("signup_form"):
            new_username = st.text_input("Username", placeholder="Choose a username (min 3 characters)")
            new_email = st.text_input("Email", placeholder="Enter your email address")
            new_password = st.text_input("Password", type="password", placeholder="Create a strong password")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
            submit_button = st.form_submit_button("Sign Up", use_container_width=True)
            
            if submit_button:
                if new_username and new_email and new_password and confirm_password:
                    if new_password != confirm_password:
                        st.error("Passwords do not match")
                    else:
                        result = st.session_state.auth_system.register_user(new_username, new_email, new_password)
                        if result['success']:
                            st.success("Account created successfully! Please login.")
                        else:
                            st.error(result['message'])
                else:
                    st.error("Please fill in all fields")

def show_user_menu():
    """Show user menu in sidebar"""
    if st.session_state.authenticated and st.session_state.user_data:
        user_data = st.session_state.user_data
        
        with st.sidebar:
            st.markdown("---")
            st.markdown(f"**Welcome, {user_data['username']}!**")
            
            if st.button("🚪 Logout", use_container_width=True):
                st.session_state.auth_system.logout_user(user_data['session_token'])
                st.session_state.authenticated = False
                st.session_state.user_data = None
                st.success("Logged out successfully!")
                st.rerun()

def require_auth(func):
    """Decorator to require authentication for pages"""
    def wrapper(*args, **kwargs):
        if not st.session_state.authenticated:
            show_login_page()
            return
        return func(*args, **kwargs)
>>>>>>> Stashed changes
    return wrapper