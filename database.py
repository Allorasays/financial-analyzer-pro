"""
Database module for Financial Analyzer Pro
Handles SQLite database operations for user data, portfolios, and preferences
"""

import sqlite3
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import streamlit as st

class DatabaseManager:
    """Manages SQLite database operations for Financial Analyzer Pro"""
    
    def __init__(self, db_path: str = "financial_analyzer.db"):
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self) -> sqlite3.Connection:
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn
    
    def init_database(self):
        """Initialize database with required tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    full_name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    subscription_type TEXT DEFAULT 'free'
                )
            """)
            
            # User preferences table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    default_timeframe TEXT DEFAULT '1y',
                    favorite_indicators TEXT DEFAULT '["RSI", "MACD", "SMA_20"]',
                    risk_tolerance TEXT DEFAULT 'Medium',
                    theme TEXT DEFAULT 'Light',
                    notifications_enabled BOOLEAN DEFAULT 1,
                    email_alerts BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                )
            """)
            
            # Portfolios table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS portfolios (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    is_default BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                )
            """)
            
            # Portfolio positions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    portfolio_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    shares REAL NOT NULL,
                    purchase_price REAL NOT NULL,
                    purchase_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (portfolio_id) REFERENCES portfolios (id) ON DELETE CASCADE
                )
            """)
            
            # Watchlists table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS watchlists (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    symbols TEXT DEFAULT '[]',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                )
            """)
            
            # Analysis templates table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis_templates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    template_data TEXT NOT NULL,
                    is_public BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                )
            """)
            
            # User sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    session_token TEXT UNIQUE NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                )
            """)
            
            conn.commit()
    
    def hash_password(self, password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return self.hash_password(password) == hashed
    
    # User management methods
    def create_user(self, username: str, email: str, password: str, full_name: str = None) -> bool:
        """Create a new user"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                password_hash = self.hash_password(password)
                
                cursor.execute("""
                    INSERT INTO users (username, email, password_hash, full_name)
                    VALUES (?, ?, ?, ?)
                """, (username, email, password_hash, full_name))
                
                user_id = cursor.lastrowid
                
                # Create default preferences for new user
                cursor.execute("""
                    INSERT INTO user_preferences (user_id)
                    VALUES (?)
                """, (user_id,))
                
                # Create default portfolio
                cursor.execute("""
                    INSERT INTO portfolios (user_id, name, description, is_default)
                    VALUES (?, ?, ?, ?)
                """, (user_id, "My Portfolio", "Default portfolio", 1))
                
                # Create default watchlist
                cursor.execute("""
                    INSERT INTO watchlists (user_id, name)
                    VALUES (?, ?)
                """, (user_id, "My Watchlist"))
                
                conn.commit()
                return True
        except sqlite3.IntegrityError:
            return False
        except Exception as e:
            st.error(f"Error creating user: {str(e)}")
            return False
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict]:
        """Authenticate user and return user data"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                password_hash = self.hash_password(password)
                
                cursor.execute("""
                    SELECT id, username, email, full_name, subscription_type, created_at
                    FROM users 
                    WHERE username = ? AND password_hash = ? AND is_active = 1
                """, (username, password_hash))
                
                user = cursor.fetchone()
                if user:
                    # Update last login
                    cursor.execute("""
                        UPDATE users SET last_login = CURRENT_TIMESTAMP
                        WHERE id = ?
                    """, (user['id'],))
                    conn.commit()
                    
                    return dict(user)
                return None
        except Exception as e:
            st.error(f"Error authenticating user: {str(e)}")
            return None
    
    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        """Get user by ID"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, username, email, full_name, subscription_type, created_at, last_login
                    FROM users WHERE id = ? AND is_active = 1
                """, (user_id,))
                
                user = cursor.fetchone()
                return dict(user) if user else None
        except Exception as e:
            st.error(f"Error getting user: {str(e)}")
            return None
    
    # User preferences methods
    def get_user_preferences(self, user_id: int) -> Dict:
        """Get user preferences"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM user_preferences WHERE user_id = ?
                """, (user_id,))
                
                prefs = cursor.fetchone()
                if prefs:
                    prefs_dict = dict(prefs)
                    # Parse JSON fields
                    prefs_dict['favorite_indicators'] = json.loads(prefs_dict['favorite_indicators'])
                    return prefs_dict
                return {}
        except Exception as e:
            st.error(f"Error getting preferences: {str(e)}")
            return {}
    
    def update_user_preferences(self, user_id: int, preferences: Dict) -> bool:
        """Update user preferences"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Convert list to JSON string
                if 'favorite_indicators' in preferences:
                    preferences['favorite_indicators'] = json.dumps(preferences['favorite_indicators'])
                
                # Check if preferences exist
                cursor.execute("SELECT id FROM user_preferences WHERE user_id = ?", (user_id,))
                if cursor.fetchone():
                    # Update existing preferences
                    set_clause = ", ".join([f"{key} = ?" for key in preferences.keys()])
                    values = list(preferences.values()) + [user_id]
                    cursor.execute(f"""
                        UPDATE user_preferences 
                        SET {set_clause}, updated_at = CURRENT_TIMESTAMP
                        WHERE user_id = ?
                    """, values)
                else:
                    # Create new preferences
                    columns = list(preferences.keys()) + ['user_id']
                    placeholders = ", ".join(["?" for _ in columns])
                    values = list(preferences.values()) + [user_id]
                    cursor.execute(f"""
                        INSERT INTO user_preferences ({", ".join(columns)})
                        VALUES ({placeholders})
                    """, values)
                
                conn.commit()
                return True
        except Exception as e:
            st.error(f"Error updating preferences: {str(e)}")
            return False
    
    # Portfolio methods
    def get_user_portfolios(self, user_id: int) -> List[Dict]:
        """Get all portfolios for a user"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM portfolios WHERE user_id = ? ORDER BY created_at DESC
                """, (user_id,))
                
                portfolios = cursor.fetchall()
                return [dict(portfolio) for portfolio in portfolios]
        except Exception as e:
            st.error(f"Error getting portfolios: {str(e)}")
            return []
    
    def create_portfolio(self, user_id: int, name: str, description: str = None) -> bool:
        """Create a new portfolio"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO portfolios (user_id, name, description)
                    VALUES (?, ?, ?)
                """, (user_id, name, description))
                conn.commit()
                return True
        except Exception as e:
            st.error(f"Error creating portfolio: {str(e)}")
            return False
    
    def get_portfolio_positions(self, portfolio_id: int) -> List[Dict]:
        """Get all positions in a portfolio"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM portfolio_positions WHERE portfolio_id = ? ORDER BY created_at DESC
                """, (portfolio_id,))
                
                positions = cursor.fetchall()
                return [dict(position) for position in positions]
        except Exception as e:
            st.error(f"Error getting portfolio positions: {str(e)}")
            return []
    
    def add_position_to_portfolio(self, portfolio_id: int, symbol: str, shares: float, 
                                purchase_price: float, notes: str = None) -> bool:
        """Add a position to a portfolio"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO portfolio_positions (portfolio_id, symbol, shares, purchase_price, notes)
                    VALUES (?, ?, ?, ?, ?)
                """, (portfolio_id, symbol, shares, purchase_price, notes))
                conn.commit()
                return True
        except Exception as e:
            st.error(f"Error adding position: {str(e)}")
            return False
    
    def remove_position_from_portfolio(self, position_id: int) -> bool:
        """Remove a position from a portfolio"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM portfolio_positions WHERE id = ?", (position_id,))
                conn.commit()
                return True
        except Exception as e:
            st.error(f"Error removing position: {str(e)}")
            return False
    
    # Watchlist methods
    def get_user_watchlists(self, user_id: int) -> List[Dict]:
        """Get all watchlists for a user"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM watchlists WHERE user_id = ? ORDER BY created_at DESC
                """, (user_id,))
                
                watchlists = cursor.fetchall()
                result = []
                for watchlist in watchlists:
                    watchlist_dict = dict(watchlist)
                    watchlist_dict['symbols'] = json.loads(watchlist_dict['symbols'])
                    result.append(watchlist_dict)
                return result
        except Exception as e:
            st.error(f"Error getting watchlists: {str(e)}")
            return []
    
    def update_watchlist(self, watchlist_id: int, symbols: List[str]) -> bool:
        """Update watchlist symbols"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE watchlists 
                    SET symbols = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (json.dumps(symbols), watchlist_id))
                conn.commit()
                return True
        except Exception as e:
            st.error(f"Error updating watchlist: {str(e)}")
            return False
    
    # Session management
    def create_session(self, user_id: int) -> str:
        """Create a new user session"""
        import secrets
        session_token = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(days=7)  # 7-day session
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO user_sessions (user_id, session_token, expires_at)
                    VALUES (?, ?, ?)
                """, (user_id, session_token, expires_at))
                conn.commit()
                return session_token
        except Exception as e:
            st.error(f"Error creating session: {str(e)}")
            return None
    
    def validate_session(self, session_token: str) -> Optional[Dict]:
        """Validate session token and return user data"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT u.id, u.username, u.email, u.full_name, u.subscription_type
                    FROM users u
                    JOIN user_sessions s ON u.id = s.user_id
                    WHERE s.session_token = ? AND s.expires_at > CURRENT_TIMESTAMP AND u.is_active = 1
                """, (session_token,))
                
                user = cursor.fetchone()
                return dict(user) if user else None
        except Exception as e:
            st.error(f"Error validating session: {str(e)}")
            return None
    
    def delete_session(self, session_token: str) -> bool:
        """Delete a session"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM user_sessions WHERE session_token = ?", (session_token,))
                conn.commit()
                return True
        except Exception as e:
            st.error(f"Error deleting session: {str(e)}")
            return False
    
    # Utility methods
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                stats = {}
                tables = ['users', 'portfolios', 'portfolio_positions', 'watchlists', 'analysis_templates']
                
                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    stats[f"{table}_count"] = cursor.fetchone()[0]
                
                return stats
        except Exception as e:
            st.error(f"Error getting database stats: {str(e)}")
            return {}

# Global database instance
db = DatabaseManager()

# Convenience functions for easy access
def get_db() -> DatabaseManager:
    """Get database instance"""
    return db

def init_database():
    """Initialize database (call this at app startup)"""
    db.init_database()
