"""
Enhanced User Authentication System for FinancialAnalyzerApp
Supports mobile app with social login, portfolio management, and subscription tiers
"""

import sqlite3
import bcrypt
import jwt
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from fastapi import HTTPException, status
from pydantic import BaseModel, EmailStr
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 30

# Subscription tiers
SUBSCRIPTION_TIERS = {
    "free": {
        "name": "Free",
        "max_portfolios": 1,
        "max_positions": 10,
        "max_watchlists": 1,
        "max_watchlist_items": 10,
        "real_time_quotes": False,
        "api_access": False,
        "price_alerts": False
    },
    "premium": {
        "name": "Premium",
        "max_portfolios": 5,
        "max_positions": 100,
        "max_watchlists": 10,
        "max_watchlist_items": 100,
        "real_time_quotes": True,
        "api_access": False,
        "price_alerts": True
    },
    "pro": {
        "name": "Pro",
        "max_portfolios": 20,
        "max_positions": 500,
        "max_watchlists": 50,
        "max_watchlist_items": 500,
        "real_time_quotes": True,
        "api_access": True,
        "price_alerts": True
    },
    "enterprise": {
        "name": "Enterprise",
        "max_portfolios": -1,  # Unlimited
        "max_positions": -1,   # Unlimited
        "max_watchlists": -1,  # Unlimited
        "max_watchlist_items": -1,  # Unlimited
        "real_time_quotes": True,
        "api_access": True,
        "price_alerts": True
    }
}

# Pydantic models
class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: str
    email: str
    username: str
    first_name: Optional[str]
    last_name: Optional[str]
    subscription_tier: str
    created_at: datetime
    is_active: bool

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str
    expires_in: int

class PortfolioCreate(BaseModel):
    name: str
    description: Optional[str] = None

class PortfolioPosition(BaseModel):
    ticker: str
    shares: float
    cost_basis: float
    purchase_date: str

class WatchlistCreate(BaseModel):
    name: str

class PriceAlertCreate(BaseModel):
    ticker: str
    alert_type: str  # 'price_above', 'price_below', 'percentage_change'
    target_value: float

class DatabaseManager:
    """Enhanced database manager for mobile app features"""
    
    def __init__(self, db_path: str = "financial_analyzer.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with enhanced schema for mobile app"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            
            # Users table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    first_name TEXT,
                    last_name TEXT,
                    subscription_tier TEXT DEFAULT 'free',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    last_login TIMESTAMP,
                    email_verified BOOLEAN DEFAULT 0,
                    verification_token TEXT,
                    reset_token TEXT,
                    reset_token_expires TIMESTAMP
                )
            """)
            
            # User sessions for refresh tokens
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    refresh_token TEXT UNIQUE NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    device_info TEXT,
                    ip_address TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                )
            """)
            
            # Portfolios table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolios (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    is_default BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                )
            """)
            
            # Portfolio positions
            conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_positions (
                    id TEXT PRIMARY KEY,
                    portfolio_id TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    shares REAL NOT NULL,
                    cost_basis REAL NOT NULL,
                    purchase_date DATE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (portfolio_id) REFERENCES portfolios (id) ON DELETE CASCADE
                )
            """)
            
            # Watchlists
            conn.execute("""
                CREATE TABLE IF NOT EXISTS watchlists (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                )
            """)
            
            # Watchlist items
            conn.execute("""
                CREATE TABLE IF NOT EXISTS watchlist_items (
                    id TEXT PRIMARY KEY,
                    watchlist_id TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (watchlist_id) REFERENCES watchlists (id) ON DELETE CASCADE
                )
            """)
            
            # Price alerts
            conn.execute("""
                CREATE TABLE IF NOT EXISTS price_alerts (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    target_value REAL NOT NULL,
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    triggered_at TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                )
            """)
            
            # User preferences
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    preference_key TEXT NOT NULL,
                    preference_value TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
                    UNIQUE(user_id, preference_key)
                )
            """)
            
            # User activity logs
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_activity_logs (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    activity_type TEXT NOT NULL,
                    activity_data TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                )
            """)
            
            # Create indexes for better performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_portfolios_user_id ON portfolios(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_positions_portfolio_id ON portfolio_positions(portfolio_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_watchlists_user_id ON watchlists(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_user_id ON price_alerts(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON user_sessions(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_activity_user_id ON user_activity_logs(user_id)")
            
            conn.commit()

class UserService:
    """Enhanced user service for mobile app"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    def create_user(self, user_data: UserCreate) -> UserResponse:
        """Create new user with enhanced validation"""
        user_id = str(uuid.uuid4())
        password_hash = self.hash_password(user_data.password)
        
        with sqlite3.connect(self.db.db_path) as conn:
            try:
                conn.execute("""
                    INSERT INTO users (id, email, username, password_hash, first_name, last_name)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    user_id,
                    user_data.email.lower(),
                    user_data.username.lower(),
                    password_hash,
                    user_data.first_name,
                    user_data.last_name
                ))
                conn.commit()
                
                # Create default portfolio
                portfolio_id = str(uuid.uuid4())
                conn.execute("""
                    INSERT INTO portfolios (id, user_id, name, is_default)
                    VALUES (?, ?, ?, ?)
                """, (portfolio_id, user_id, "My Portfolio", 1))
                
                # Create default watchlist
                watchlist_id = str(uuid.uuid4())
                conn.execute("""
                    INSERT INTO watchlists (id, user_id, name)
                    VALUES (?, ?, ?)
                """, (watchlist_id, user_id, "My Watchlist"))
                
                conn.commit()
                
                return self.get_user_by_id(user_id)
                
            except sqlite3.IntegrityError as e:
                if "email" in str(e):
                    raise HTTPException(status_code=400, detail="Email already registered")
                elif "username" in str(e):
                    raise HTTPException(status_code=400, detail="Username already taken")
                else:
                    raise HTTPException(status_code=400, detail="User creation failed")
    
    def authenticate_user(self, email: str, password: str) -> Optional[UserResponse]:
        """Authenticate user and return user data"""
        with sqlite3.connect(self.db.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, email, username, password_hash, first_name, last_name,
                       subscription_tier, created_at, is_active
                FROM users WHERE email = ? AND is_active = 1
            """, (email.lower(),))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            user_id, email, username, password_hash, first_name, last_name, \
            subscription_tier, created_at, is_active = row
            
            if not self.verify_password(password, password_hash):
                return None
            
            # Update last login
            conn.execute("""
                UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?
            """, (user_id,))
            conn.commit()
            
            return UserResponse(
                id=user_id,
                email=email,
                username=username,
                first_name=first_name,
                last_name=last_name,
                subscription_tier=subscription_tier,
                created_at=datetime.fromisoformat(created_at),
                is_active=bool(is_active)
            )
    
    def get_user_by_id(self, user_id: str) -> Optional[UserResponse]:
        """Get user by ID"""
        with sqlite3.connect(self.db.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, email, username, first_name, last_name,
                       subscription_tier, created_at, is_active
                FROM users WHERE id = ? AND is_active = 1
            """, (user_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            user_id, email, username, first_name, last_name, \
            subscription_tier, created_at, is_active = row
            
            return UserResponse(
                id=user_id,
                email=email,
                username=username,
                first_name=first_name,
                last_name=last_name,
                subscription_tier=subscription_tier,
                created_at=datetime.fromisoformat(created_at),
                is_active=bool(is_active)
            )
    
    def update_subscription_tier(self, user_id: str, tier: str) -> bool:
        """Update user subscription tier"""
        if tier not in SUBSCRIPTION_TIERS:
            return False
        
        with sqlite3.connect(self.db.db_path) as conn:
            conn.execute("""
                UPDATE users SET subscription_tier = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (tier, user_id))
            conn.commit()
            return True
    
    def get_user_limits(self, user_id: str) -> Dict[str, Any]:
        """Get user's current limits based on subscription tier"""
        user = self.get_user_by_id(user_id)
        if not user:
            return SUBSCRIPTION_TIERS["free"]
        
        return SUBSCRIPTION_TIERS.get(user.subscription_tier, SUBSCRIPTION_TIERS["free"])

class TokenService:
    """Enhanced token service for mobile app"""
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire, "type": "access"})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    def create_refresh_token(self, data: dict) -> str:
        """Create refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode.update({"exp": expire, "type": "refresh"})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    def verify_token(self, token: str, token_type: str = "access") -> Optional[dict]:
        """Verify token and return payload"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            if payload.get("type") != token_type:
                return None
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.JWTError:
            return None
    
    def store_refresh_token(self, user_id: str, refresh_token: str, device_info: str = None, ip_address: str = None) -> str:
        """Store refresh token in database"""
        session_id = str(uuid.uuid4())
        expires_at = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        
        with sqlite3.connect("financial_analyzer.db") as conn:
            conn.execute("""
                INSERT INTO user_sessions (id, user_id, refresh_token, expires_at, device_info, ip_address)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (session_id, user_id, refresh_token, expires_at.isoformat(), device_info, ip_address))
            conn.commit()
        
        return session_id
    
    def revoke_refresh_token(self, refresh_token: str) -> bool:
        """Revoke refresh token"""
        with sqlite3.connect("financial_analyzer.db") as conn:
            cursor = conn.execute("DELETE FROM user_sessions WHERE refresh_token = ?", (refresh_token,))
            conn.commit()
            return cursor.rowcount > 0

# Initialize services
db_manager = DatabaseManager()
user_service = UserService(db_manager)
token_service = TokenService()

# Export for use in main application
__all__ = [
    'UserCreate', 'UserLogin', 'UserResponse', 'TokenResponse',
    'PortfolioCreate', 'PortfolioPosition', 'WatchlistCreate', 'PriceAlertCreate',
    'DatabaseManager', 'UserService', 'TokenService',
    'SUBSCRIPTION_TIERS', 'db_manager', 'user_service', 'token_service'
]




