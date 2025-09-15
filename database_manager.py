"""
Database Manager for Financial Analyzer Pro - Phase 3
Handles SQLite database operations, user management, and data persistence
"""

import sqlite3
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json

class DatabaseManager:
    def __init__(self, db_path: str = "financial_analyzer.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with all required tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    full_name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    salt TEXT NOT NULL
                )
            ''')
            
            # User preferences
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id INTEGER PRIMARY KEY,
                    theme TEXT DEFAULT 'light',
                    default_timeframe TEXT DEFAULT '1mo',
                    default_symbols TEXT DEFAULT '["AAPL", "MSFT", "GOOGL"]',
                    notification_settings TEXT DEFAULT '{}',
                    dashboard_layout TEXT DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            ''')
            
            # Portfolios
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolios (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    is_public BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            ''')
            
            # Portfolio positions
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolio_positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    portfolio_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    shares REAL NOT NULL,
                    purchase_price REAL NOT NULL,
                    purchase_date TIMESTAMP NOT NULL,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (portfolio_id) REFERENCES portfolios(id) ON DELETE CASCADE
                )
            ''')
            
            # Watchlists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS watchlists (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    symbols TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            ''')
            
            # User sessions
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    session_token TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    is_active BOOLEAN DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            ''')
            
            # Analysis templates
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_templates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    template_data TEXT NOT NULL,
                    is_public BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            ''')
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Database initialization error: {str(e)}")
            return False
    
    def hash_password(self, password: str, salt: str = None) -> Tuple[str, str]:
        """Hash password with salt"""
        if salt is None:
            salt = secrets.token_hex(16)
        
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
        return password_hash.hex(), salt
    
    def create_user(self, username: str, email: str, password: str, full_name: str = None) -> bool:
        """Create a new user account"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if user already exists
            cursor.execute("SELECT id FROM users WHERE username = ? OR email = ?", (username, email))
            if cursor.fetchone():
                conn.close()
                return False
            
            # Hash password
            password_hash, salt = self.hash_password(password)
            
            # Insert user
            cursor.execute('''
                INSERT INTO users (username, email, password_hash, full_name, salt)
                VALUES (?, ?, ?, ?, ?)
            ''', (username, email, password_hash, full_name, salt))
            
            user_id = cursor.lastrowid
            
            # Create default preferences
            cursor.execute('''
                INSERT INTO user_preferences (user_id)
                VALUES (?)
            ''', (user_id,))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"User creation error: {str(e)}")
            return False
    
    def authenticate_user(self, username_or_email: str, password: str) -> Optional[Dict]:
        """Authenticate user login"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Find user by username or email
            cursor.execute('''
                SELECT id, username, email, password_hash, full_name, salt, is_active
                FROM users WHERE (username = ? OR email = ?) AND is_active = 1
            ''', (username_or_email, username_or_email))
            
            user = cursor.fetchone()
            if not user:
                conn.close()
                return None
            
            user_id, username, email, stored_hash, full_name, salt, is_active = user
            
            # Verify password
            password_hash, _ = self.hash_password(password, salt)
            if password_hash != stored_hash:
                conn.close()
                return None
            
            # Update last login
            cursor.execute('''
                UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?
            ''', (user_id,))
            
            conn.commit()
            conn.close()
            
            return {
                'id': user_id,
                'username': username,
                'email': email,
                'full_name': full_name,
                'is_active': is_active
            }
            
        except Exception as e:
            print(f"Authentication error: {str(e)}")
            return None
    
    def create_session(self, user_id: int) -> str:
        """Create a new user session"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Generate session token
            session_token = secrets.token_urlsafe(32)
            expires_at = datetime.now() + timedelta(days=30)
            
            # Insert session
            cursor.execute('''
                INSERT INTO user_sessions (user_id, session_token, expires_at)
                VALUES (?, ?, ?)
            ''', (user_id, session_token, expires_at))
            
            conn.commit()
            conn.close()
            
            return session_token
            
        except Exception as e:
            print(f"Session creation error: {str(e)}")
            return None
    
    def validate_session(self, session_token: str) -> Optional[Dict]:
        """Validate user session"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT u.id, u.username, u.email, u.full_name, s.expires_at
                FROM users u
                JOIN user_sessions s ON u.id = s.user_id
                WHERE s.session_token = ? AND s.is_active = 1 AND s.expires_at > CURRENT_TIMESTAMP
            ''', (session_token,))
            
            session = cursor.fetchone()
            if not session:
                conn.close()
                return None
            
            user_id, username, email, full_name, expires_at = session
            
            conn.close()
            
            return {
                'id': user_id,
                'username': username,
                'email': email,
                'full_name': full_name,
                'expires_at': expires_at
            }
            
        except Exception as e:
            print(f"Session validation error: {str(e)}")
            return None
    
    def get_user_preferences(self, user_id: int) -> Dict:
        """Get user preferences"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT theme, default_timeframe, default_symbols, notification_settings, dashboard_layout
                FROM user_preferences WHERE user_id = ?
            ''', (user_id,))
            
            prefs = cursor.fetchone()
            conn.close()
            
            if prefs:
                theme, timeframe, symbols, notifications, layout = prefs
                return {
                    'theme': theme,
                    'default_timeframe': timeframe,
                    'default_symbols': json.loads(symbols) if symbols else [],
                    'notification_settings': json.loads(notifications) if notifications else {},
                    'dashboard_layout': json.loads(layout) if layout else {}
                }
            else:
                return {
                    'theme': 'light',
                    'default_timeframe': '1mo',
                    'default_symbols': ['AAPL', 'MSFT', 'GOOGL'],
                    'notification_settings': {},
                    'dashboard_layout': {}
                }
                
        except Exception as e:
            print(f"Get preferences error: {str(e)}")
            return {}
    
    def update_user_preferences(self, user_id: int, preferences: Dict) -> bool:
        """Update user preferences"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE user_preferences SET
                    theme = ?, default_timeframe = ?, default_symbols = ?,
                    notification_settings = ?, dashboard_layout = ?, updated_at = CURRENT_TIMESTAMP
                WHERE user_id = ?
            ''', (
                preferences.get('theme', 'light'),
                preferences.get('default_timeframe', '1mo'),
                json.dumps(preferences.get('default_symbols', [])),
                json.dumps(preferences.get('notification_settings', {})),
                json.dumps(preferences.get('dashboard_layout', {})),
                user_id
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Update preferences error: {str(e)}")
            return False
    
    def create_portfolio(self, user_id: int, name: str, description: str = None, is_public: bool = False) -> Optional[int]:
        """Create a new portfolio"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO portfolios (user_id, name, description, is_public)
                VALUES (?, ?, ?, ?)
            ''', (user_id, name, description, is_public))
            
            portfolio_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return portfolio_id
            
        except Exception as e:
            print(f"Portfolio creation error: {str(e)}")
            return None
    
    def get_user_portfolios(self, user_id: int) -> List[Dict]:
        """Get all portfolios for a user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, name, description, is_public, created_at, updated_at
                FROM portfolios WHERE user_id = ? ORDER BY updated_at DESC
            ''', (user_id,))
            
            portfolios = []
            for row in cursor.fetchall():
                portfolios.append({
                    'id': row[0],
                    'name': row[1],
                    'description': row[2],
                    'is_public': bool(row[3]),
                    'created_at': row[4],
                    'updated_at': row[5]
                })
            
            conn.close()
            return portfolios
            
        except Exception as e:
            print(f"Get portfolios error: {str(e)}")
            return []
    
    def add_portfolio_position(self, portfolio_id: int, symbol: str, shares: float, 
                             purchase_price: float, purchase_date: str, notes: str = None) -> bool:
        """Add a position to a portfolio"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO portfolio_positions (portfolio_id, symbol, shares, purchase_price, purchase_date, notes)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (portfolio_id, symbol, shares, purchase_price, purchase_date, notes))
            
            # Update portfolio updated_at
            cursor.execute('''
                UPDATE portfolios SET updated_at = CURRENT_TIMESTAMP WHERE id = ?
            ''', (portfolio_id,))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Add position error: {str(e)}")
            return False
    
    def get_portfolio_positions(self, portfolio_id: int) -> List[Dict]:
        """Get all positions in a portfolio"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, symbol, shares, purchase_price, purchase_date, notes, created_at
                FROM portfolio_positions WHERE portfolio_id = ? ORDER BY created_at DESC
            ''', (portfolio_id,))
            
            positions = []
            for row in cursor.fetchall():
                positions.append({
                    'id': row[0],
                    'symbol': row[1],
                    'shares': row[2],
                    'purchase_price': row[3],
                    'purchase_date': row[4],
                    'notes': row[5],
                    'created_at': row[6]
                })
            
            conn.close()
            return positions
            
        except Exception as e:
            print(f"Get positions error: {str(e)}")
            return []
    
    def create_watchlist(self, user_id: int, name: str, symbols: List[str]) -> Optional[int]:
        """Create a new watchlist"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO watchlists (user_id, name, symbols)
                VALUES (?, ?, ?)
            ''', (user_id, name, json.dumps(symbols)))
            
            watchlist_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return watchlist_id
            
        except Exception as e:
            print(f"Watchlist creation error: {str(e)}")
            return None
    
    def get_user_watchlists(self, user_id: int) -> List[Dict]:
        """Get all watchlists for a user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, name, symbols, created_at, updated_at
                FROM watchlists WHERE user_id = ? ORDER BY updated_at DESC
            ''', (user_id,))
            
            watchlists = []
            for row in cursor.fetchall():
                watchlists.append({
                    'id': row[0],
                    'name': row[1],
                    'symbols': json.loads(row[2]) if row[2] else [],
                    'created_at': row[3],
                    'updated_at': row[4]
                })
            
            conn.close()
            return watchlists
            
        except Exception as e:
            print(f"Get watchlists error: {str(e)}")
            return []
    
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            stats = {}
            
            # Count users
            cursor.execute("SELECT COUNT(*) FROM users")
            stats['users_count'] = cursor.fetchone()[0]
            
            # Count portfolios
            cursor.execute("SELECT COUNT(*) FROM portfolios")
            stats['portfolios_count'] = cursor.fetchone()[0]
            
            # Count positions
            cursor.execute("SELECT COUNT(*) FROM portfolio_positions")
            stats['portfolio_positions_count'] = cursor.fetchone()[0]
            
            # Count watchlists
            cursor.execute("SELECT COUNT(*) FROM watchlists")
            stats['watchlists_count'] = cursor.fetchone()[0]
            
            # Count sessions
            cursor.execute("SELECT COUNT(*) FROM user_sessions WHERE is_active = 1")
            stats['active_sessions_count'] = cursor.fetchone()[0]
            
            conn.close()
            return stats
            
        except Exception as e:
            print(f"Database stats error: {str(e)}")
            return {}
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE user_sessions SET is_active = 0 
                WHERE expires_at < CURRENT_TIMESTAMP AND is_active = 1
            ''')
            
            cleaned = cursor.rowcount
            conn.commit()
            conn.close()
            
            return cleaned
            
        except Exception as e:
            print(f"Session cleanup error: {str(e)}")
            return 0
