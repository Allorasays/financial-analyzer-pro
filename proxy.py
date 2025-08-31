from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import sqlite3
import bcrypt
import jwt
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import ta
import json
import os
from dotenv import load_dotenv
import time
from collections import defaultdict, deque
import threading
from contextlib import contextmanager
import csv
import io
from fastapi.responses import StreamingResponse
import zipfile
from datetime import datetime, timedelta


load_dotenv()

app = FastAPI(
    title="Financial Analyzer Pro API",
    description="Advanced API for financial data analysis, portfolios, and ML predictions",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Rate Limiting Configuration
RATE_LIMIT_CONFIG = {
    "default": {
        "requests": 100,  # requests per window
        "window": 3600,   # window in seconds (1 hour)
    },
    "auth": {
        "requests": 10,    # 10 login attempts per hour
        "window": 3600,
    },
    "market_data": {
        "requests": 300,   # 300 market data requests per hour
        "window": 3600,
    },
    "ml_predictions": {
        "requests": 50,    # 50 ML predictions per hour (more expensive)
        "window": 3600,
    },
    "portfolio": {
        "requests": 200,   # 200 portfolio operations per hour
        "window": 3600,
    },
    "technical_analysis": {
        "requests": 150,   # 150 technical analysis requests per hour
        "window": 3600,
    }
}

# Rate limiting storage (in production, use Redis)
rate_limit_storage = defaultdict(lambda: defaultdict(deque))

class RateLimitExceeded(HTTPException):
    def __init__(self, retry_after: int):
        super().__init__(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "message": "Too many requests. Please try again later.",
                "retry_after": retry_after
            }
        )

def get_client_identifier(request: Request) -> str:
    """Get client identifier for rate limiting"""
    # Try to get user ID from JWT token first
    auth_header = request.headers.get("authorization")
    if auth_header and auth_header.startswith("Bearer "):
        try:
            token = auth_header.split(" ")[1]
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return f"user:{payload.get('sub', 'unknown')}"
        except:
            pass
    
    # Fallback to IP address
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return f"ip:{forwarded_for.split(',')[0].strip()}"
    
    return f"ip:{request.client.host if request.client else 'unknown'}"

def check_rate_limit(client_id: str, endpoint_type: str = "default") -> None:
    """Check if client has exceeded rate limit for endpoint type"""
    config = RATE_LIMIT_CONFIG.get(endpoint_type, RATE_LIMIT_CONFIG["default"])
    max_requests = config["requests"]
    window = config["window"]
    
    current_time = time.time()
    client_requests = rate_limit_storage[client_id][endpoint_type]
    
    # Remove expired requests
    while client_requests and current_time - client_requests[0] > window:
        client_requests.popleft()
    
    # Check if limit exceeded
    if len(client_requests) >= max_requests:
        # Calculate retry after time
        oldest_request = client_requests[0]
        retry_after = int(window - (current_time - oldest_request))
        raise RateLimitExceeded(retry_after)
    
    # Add current request
    client_requests.append(current_time)

# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Middleware for rate limiting"""
    try:
        # Determine endpoint type based on path
        path = request.url.path
        if path.startswith("/api/auth/"):
            endpoint_type = "auth"
        elif path.startswith("/api/market/"):
            endpoint_type = "market_data"
        elif path.startswith("/api/ml/"):
            endpoint_type = "ml_predictions"
        elif path.startswith("/api/portfolio") or path.startswith("/api/watchlist"):
            endpoint_type = "portfolio"
        elif path.startswith("/api/technical/"):
            endpoint_type = "technical_analysis"
        else:
            endpoint_type = "default"
        
        # Get client identifier and check rate limit
        client_id = get_client_identifier(request)
        check_rate_limit(client_id, endpoint_type)
        
        # Continue with the request
        response = await call_next(request)
        return response
        
    except RateLimitExceeded as e:
        return JSONResponse(
            status_code=429,
            content=e.detail,
            headers={"Retry-After": str(e.detail["retry_after"])}
        )
    except Exception as e:
        # If rate limiting fails, continue with the request
        return await call_next(request)

# Database Manager Class
class DatabaseManager:
    def __init__(self, db_path: str = 'financial_analyzer.db'):
        self.db_path = db_path
        self.lock = threading.Lock()
        self.init_db()
    
    def get_connection(self):
        """Get a database connection with proper configuration"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable row factory for named access
        conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign key constraints
        conn.execute("PRAGMA journal_mode = WAL")  # Enable WAL mode for better concurrency
        return conn
    
    @contextmanager
    def get_db_cursor(self):
        """Context manager for database operations"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def init_db(self):
        """Initialize database with all required tables"""
        with self.get_db_cursor() as cursor:
            # Users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    first_name TEXT,
                    last_name TEXT,
                    phone TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    last_login TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # User preferences table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_preferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER UNIQUE NOT NULL,
                    default_currency TEXT DEFAULT 'USD',
                    timezone TEXT DEFAULT 'UTC',
                    notification_enabled BOOLEAN DEFAULT 1,
                    email_notifications BOOLEAN DEFAULT 1,
                    push_notifications BOOLEAN DEFAULT 1,
                    risk_tolerance TEXT DEFAULT 'MODERATE',
                    investment_goals TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # Portfolios table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolios (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    ticker TEXT NOT NULL,
                    shares REAL NOT NULL,
                    avg_price REAL NOT NULL,
                    purchase_date DATE,
                    notes TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id),
                    UNIQUE(user_id, ticker)
                )
            ''')
            
            # Portfolio transactions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolio_transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    portfolio_id INTEGER NOT NULL,
                    transaction_type TEXT NOT NULL CHECK(transaction_type IN ('BUY', 'SELL', 'DIVIDEND', 'SPLIT')),
                    shares REAL NOT NULL,
                    price_per_share REAL NOT NULL,
                    transaction_date DATE NOT NULL,
                    fees REAL DEFAULT 0,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (portfolio_id) REFERENCES portfolios (id)
                )
            ''')
            
            # Watchlists table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS watchlists (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    ticker TEXT NOT NULL,
                    price_alert_high REAL,
                    price_alert_low REAL,
                    notes TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id),
                    UNIQUE(user_id, ticker)
                )
            ''')
            
            # Market data cache table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_data_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticker, data_type)
                )
            ''')
            
            # API usage logs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS api_usage_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    endpoint TEXT NOT NULL,
                    request_method TEXT NOT NULL,
                    response_status INTEGER,
                    response_time_ms INTEGER,
                    user_agent TEXT,
                    ip_address TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # Rate limiting logs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rate_limit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    client_id TEXT NOT NULL,
                    endpoint_type TEXT NOT NULL,
                    request_count INTEGER NOT NULL,
                    window_start TIMESTAMP NOT NULL,
                    violation_count INTEGER DEFAULT 0,
                    last_violation TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # User sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    session_token TEXT UNIQUE NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    ip_address TEXT,
                    user_agent TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_portfolios_user_ticker ON portfolios(user_id, ticker)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_watchlists_user_ticker ON watchlists(user_id, ticker)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_market_data_cache_ticker_type ON market_data_cache(ticker, data_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_api_usage_logs_user_endpoint ON api_usage_logs(user_id, endpoint)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_rate_limit_logs_client_type ON rate_limit_logs(client_id, endpoint_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_sessions_token ON user_sessions(session_token)')
    
    def log_api_usage(self, user_id: int = None, endpoint: str = "", request_method: str = "", 
                      response_status: int = 0, response_time_ms: int = 0, 
                      user_agent: str = "", ip_address: str = ""):
        """Log API usage for monitoring and analytics"""
        try:
            with self.get_db_cursor() as cursor:
                cursor.execute('''
                    INSERT INTO api_usage_logs 
                    (user_id, endpoint, request_method, response_time_ms, user_agent, ip_address)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (user_id, endpoint, request_method, response_status, response_time_ms, user_agent, ip_address))
        except Exception as e:
            print(f"Error logging API usage: {e}")
    
    def log_rate_limit_violation(self, client_id: str, endpoint_type: str):
        """Log rate limit violations for monitoring"""
        try:
            with self.get_db_cursor() as cursor:
                cursor.execute('''
                    INSERT OR REPLACE INTO rate_limit_logs 
                    (client_id, endpoint_type, request_count, window_start, violation_count, last_violation)
                    VALUES (?, ?, 0, ?, 
                        COALESCE((SELECT violation_count + 1 FROM rate_limit_logs 
                                 WHERE client_id = ? AND endpoint_type = ?), 1),
                        CURRENT_TIMESTAMP)
                ''', (client_id, endpoint_type, datetime.now().isoformat(), client_id, endpoint_type))
        except Exception as e:
            print(f"Error logging rate limit violation: {e}")
    
    def get_user_by_username(self, username: str):
        """Get user by username"""
        try:
            with self.get_db_cursor() as cursor:
                cursor.execute('''
                    SELECT u.*, up.default_currency, up.timezone, up.risk_tolerance, up.investment_goals
                    FROM users u
                    LEFT JOIN user_preferences up ON u.id = up.user_id
                    WHERE u.username = ? AND u.is_active = 1
                ''', (username,))
                return cursor.fetchone()
        except Exception as e:
            print(f"Error getting user: {e}")
            return None
    
    def create_user(self, username: str, email: str, password_hash: str, 
                   first_name: str = None, last_name: str = None, phone: str = None):
        """Create a new user with preferences"""
        try:
            with self.get_db_cursor() as cursor:
                # Insert user
                cursor.execute('''
                    INSERT INTO users (username, email, password_hash, first_name, last_name, phone)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (username, email, password_hash, first_name, last_name, phone))
                
                user_id = cursor.lastrowid
                
                # Create default user preferences
                cursor.execute('''
                    INSERT INTO user_preferences (user_id)
                    VALUES (?)
                ''', (user_id,))
                
                return user_id
        except Exception as e:
            print(f"Error creating user: {e}")
            raise e
    
    def update_user_last_login(self, user_id: int):
        """Update user's last login timestamp"""
        try:
            with self.get_db_cursor() as cursor:
                cursor.execute('''
                    UPDATE users SET last_login = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (user_id,))
        except Exception as e:
            print(f"Error updating last login: {e}")
    
    def get_user_portfolio(self, user_id: int):
        """Get user's complete portfolio with transaction history"""
        try:
            with self.get_db_cursor() as cursor:
                # Get portfolio items
                cursor.execute('''
                    SELECT p.*, 
                           COALESCE(SUM(CASE WHEN pt.transaction_type = 'BUY' THEN pt.shares ELSE 0 END), 0) as total_bought,
                           COALESCE(SUM(CASE WHEN pt.transaction_type = 'SELL' THEN pt.shares ELSE 0 END), 0) as total_sold
                    FROM portfolios p
                    LEFT JOIN portfolio_transactions pt ON p.id = pt.portfolio_id
                    WHERE p.user_id = ? AND p.is_active = 1
                    GROUP BY p.id
                ''', (user_id,))
                
                portfolio_items = cursor.fetchall()
                
                # Get transaction history
                cursor.execute('''
                    SELECT pt.*, p.ticker
                    FROM portfolio_transactions pt
                    JOIN portfolios p ON pt.portfolio_id = p.id
                    WHERE p.user_id = ?
                    ORDER BY pt.transaction_date DESC
                ''', (user_id,))
                
                transactions = cursor.fetchall()
                
                return {
                    'portfolio': portfolio_items,
                    'transactions': transactions
                }
        except Exception as e:
            print(f"Error getting portfolio: {e}")
            return None
    
    def add_portfolio_transaction(self, user_id: int, ticker: str, transaction_type: str, 
                                shares: float, price_per_share: float, transaction_date: str, 
                                fees: float = 0, notes: str = ""):
        """Add a portfolio transaction and update portfolio"""
        try:
            with self.get_db_cursor() as cursor:
                # Get or create portfolio item
                cursor.execute('''
                    SELECT id, shares, avg_price FROM portfolios 
                    WHERE user_id = ? AND ticker = ?
                ''', (user_id,))
                
                portfolio_item = cursor.fetchone()
                
                if portfolio_item:
                    portfolio_id = portfolio_item[0]
                    current_shares = portfolio_item[1]
                    current_avg_price = portfolio_item[2]
                    
                    if transaction_type == 'BUY':
                        new_shares = current_shares + shares
                        new_avg_price = ((current_shares * current_avg_price) + (shares * price_per_share)) / new_shares
                    elif transaction_type == 'SELL':
                        new_shares = current_shares - shares
                        new_avg_price = current_avg_price  # Keep same average price
                    else:
                        new_shares = current_shares
                        new_avg_price = current_avg_price
                    
                    # Update portfolio
                    cursor.execute('''
                        UPDATE portfolios 
                        SET shares = ?, avg_price = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                    ''', (new_shares, new_avg_price, portfolio_id))
                    
                    # Deactivate portfolio item if no shares left
                    if new_shares <= 0:
                        cursor.execute('''
                            UPDATE portfolios SET is_active = 0 WHERE id = ?
                        ''', (portfolio_id,))
                else:
                    # Create new portfolio item for BUY transactions
                    if transaction_type == 'BUY':
                        cursor.execute('''
                            INSERT INTO portfolios (user_id, ticker, shares, avg_price, purchase_date)
                            VALUES (?, ?, ?, ?, ?)
                        ''', (user_id, ticker, shares, price_per_share, transaction_date))
                        portfolio_id = cursor.lastrowid
                    else:
                        raise Exception(f"Cannot {transaction_type} shares for ticker {ticker} - not in portfolio")
                
                # Add transaction record
                cursor.execute('''
                    INSERT INTO portfolio_transactions 
                    (portfolio_id, transaction_type, shares, price_per_share, transaction_date, fees, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (portfolio_id, transaction_type, shares, price_per_share, transaction_date, fees, notes))
                
                return True
        except Exception as e:
            print(f"Error adding portfolio transaction: {e}")
            raise e
    
    def cache_market_data(self, ticker: str, data_type: str, data_json: str, cache_duration_minutes: int = 5):
        """Cache market data to reduce API calls"""
        try:
            expires_at = datetime.now() + timedelta(minutes=cache_duration_minutes)
            with self.get_db_cursor() as cursor:
                cursor.execute('''
                    INSERT OR REPLACE INTO market_data_cache 
                    (ticker, data_type, data_json, expires_at)
                    VALUES (?, ?, ?, ?)
                ''', (ticker, data_type, data_json, expires_at.isoformat()))
        except Exception as e:
            print(f"Error caching market data: {e}")
    
    def get_cached_market_data(self, ticker: str, data_type: str):
        """Get cached market data if not expired"""
        try:
            with self.get_db_cursor() as cursor:
                cursor.execute('''
                    SELECT data_json FROM market_data_cache 
                    WHERE ticker = ? AND data_type = ? AND expires_at > CURRENT_TIMESTAMP
                ''', (ticker, data_type, data_type))
                
                result = cursor.fetchone()
                if result:
                    return json.loads(result[0])
                return None
        except Exception as e:
            print(f"Error getting cached market data: {e}")
            return None
    
    def cleanup_expired_data(self):
        """Clean up expired cache and old logs"""
        try:
            with self.get_db_cursor() as cursor:
                # Clean expired market data cache
                cursor.execute('DELETE FROM market_data_cache WHERE expires_at <= CURRENT_TIMESTAMP')
                
                # Clean old API usage logs (keep last 30 days)
                cursor.execute('''
                    DELETE FROM api_usage_logs 
                    WHERE created_at <= datetime('now', '-30 days')
                ''')
                
                # Clean old rate limit logs (keep last 7 days)
                cursor.execute('''
                    DELETE FROM rate_limit_logs 
                    WHERE created_at <= datetime('now', '-7 days')
                ''')
                
                # Clean old user sessions (keep last 7 days)
                cursor.execute('''
                    DELETE FROM user_sessions 
                    WHERE expires_at <= CURRENT_TIMESTAMP
                ''')
                
                print("Database cleanup completed")
        except Exception as e:
            print(f"Error during database cleanup: {e}")
    
    def export_portfolio_csv(self, user_id: int) -> str:
        """Export portfolio to CSV format"""
        try:
            with self.get_db_cursor() as cursor:
                # Get portfolio data with current market prices
                cursor.execute('''
                    SELECT 
                        p.ticker,
                        p.shares,
                        p.avg_price,
                        p.purchase_date,
                        p.notes,
                        p.added_at,
                        p.updated_at
                    FROM portfolios p
                    WHERE p.user_id = ? AND p.is_active = 1
                    ORDER BY p.ticker
                ''', (user_id,))
                
                portfolio_data = cursor.fetchall()
                
                # Create CSV content
                output = io.StringIO()
                writer = csv.writer(output)
                
                # Write header
                writer.writerow([
                    'Ticker', 'Shares', 'Average Price', 'Purchase Date', 
                    'Notes', 'Added Date', 'Last Updated'
                ])
                
                # Write data
                for row in portfolio_data:
                    writer.writerow([
                        row['ticker'],
                        row['shares'],
                        row['avg_price'],
                        row['purchase_date'] or 'N/A',
                        row['notes'] or 'N/A',
                        row['added_at'],
                        row['updated_at']
                    ])
                
                return output.getvalue()
        except Exception as e:
            print(f"Error exporting portfolio CSV: {e}")
            return ""
    
    def export_transactions_csv(self, user_id: int, start_date: str = None, end_date: str = None) -> str:
        """Export transaction history to CSV format"""
        try:
            with self.get_db_cursor() as cursor:
                # Build query with optional date filters
                query = '''
                    SELECT 
                        pt.transaction_type,
                        p.ticker,
                        pt.shares,
                        pt.price_per_share,
                        pt.transaction_date,
                        pt.fees,
                        pt.notes,
                        pt.created_at
                    FROM portfolio_transactions pt
                    JOIN portfolios p ON pt.portfolio_id = p.id
                    WHERE p.user_id = ?
                '''
                params = [user_id]
                
                if start_date:
                    query += " AND pt.transaction_date >= ?"
                    params.append(start_date)
                
                if end_date:
                    query += " AND pt.transaction_date <= ?"
                    params.append(end_date)
                
                query += " ORDER BY pt.transaction_date DESC"
                
                cursor.execute(query, params)
                transactions = cursor.fetchall()
                
                # Create CSV content
                output = io.StringIO()
                writer = csv.writer(output)
                
                # Write header
                writer.writerow([
                    'Transaction Type', 'Ticker', 'Shares', 'Price per Share',
                    'Transaction Date', 'Fees', 'Notes', 'Created At'
                ])
                
                # Write data
                for row in transactions:
                    writer.writerow([
                        row['transaction_type'],
                        row['ticker'],
                        row['shares'],
                        row['price_per_share'],
                        row['transaction_date'],
                        row['fees'] or 0,
                        row['notes'] or 'N/A',
                        row['created_at']
                    ])
                
                return output.getvalue()
        except Exception as e:
            print(f"Error exporting transactions CSV: {e}")
            return ""
    
    def export_watchlist_csv(self, user_id: int) -> str:
        """Export watchlist to CSV format"""
        try:
            with self.get_db_cursor() as cursor:
                cursor.execute('''
                    SELECT 
                        ticker,
                        price_alert_high,
                        price_alert_low,
                        notes,
                        added_at,
                        updated_at
                    FROM watchlists
                    WHERE user_id = ? AND is_active = 1
                    ORDER BY ticker
                ''', (user_id,))
                
                watchlist_data = cursor.fetchall()
                
                # Create CSV content
                output = io.StringIO()
                writer = csv.writer(output)
                
                # Write header
                writer.writerow([
                    'Ticker', 'High Alert Price', 'Low Alert Price', 
                    'Notes', 'Added Date', 'Last Updated'
                ])
                
                # Write data
                for row in watchlist_data:
                    writer.writerow([
                        row['ticker'],
                        row['price_alert_high'] or 'N/A',
                        row['price_alert_low'] or 'N/A',
                        row['notes'] or 'N/A',
                        row['added_at'],
                        row['updated_at']
                    ])
                
                return output.getvalue()
        except Exception as e:
            print(f"Error exporting watchlist CSV: {e}")
            return ""
    
    def export_portfolio_summary_json(self, user_id: int) -> dict:
        """Export portfolio summary as JSON"""
        try:
            with self.get_db_cursor() as cursor:
                # Get portfolio summary
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total_positions,
                        SUM(p.shares * p.avg_price) as total_cost,
                        SUM(p.shares) as total_shares
                    FROM portfolios p
                    WHERE p.user_id = ? AND p.is_active = 1
                ''', (user_id,))
                
                summary = cursor.fetchone()
                
                # Get portfolio by sector (if available)
                cursor.execute('''
                    SELECT 
                        p.ticker,
                        p.shares,
                        p.avg_price,
                        p.purchase_date
                    FROM portfolios p
                    WHERE p.user_id = ? AND p.is_active = 1
                    ORDER BY p.ticker
                ''', (user_id,))
                
                positions = cursor.fetchall()
                
                return {
                    "export_date": datetime.now().isoformat(),
                    "user_id": user_id,
                    "summary": {
                        "total_positions": summary['total_positions'],
                        "total_cost": summary['total_cost'] or 0,
                        "total_shares": summary['total_shares'] or 0
                    },
                    "positions": [
                        {
                            "ticker": pos['ticker'],
                            "shares": pos['shares'],
                            "average_price": pos['avg_price'],
                            "purchase_date": pos['purchase_date']
                        }
                        for pos in positions
                    ]
                }
        except Exception as e:
            print(f"Error exporting portfolio summary: {e}")
            return {}
    
    def export_user_activity_logs(self, user_id: int, days: int = 30) -> str:
        """Export user activity logs to CSV"""
        try:
            with self.get_db_cursor() as cursor:
                cursor.execute('''
                    SELECT 
                        endpoint,
                        request_method,
                        response_status,
                        response_time_ms,
                        user_agent,
                        ip_address,
                        created_at
                    FROM api_usage_logs
                    WHERE user_id = ? 
                    AND created_at >= datetime('now', '-{} days')
                    ORDER BY created_at DESC
                '''.format(days), (user_id,))
                
                logs = cursor.fetchall()
                
                # Create CSV content
                output = io.StringIO()
                writer = csv.writer(output)
                
                # Write header
                writer.writerow([
                    'Endpoint', 'Method', 'Status', 'Response Time (ms)',
                    'User Agent', 'IP Address', 'Timestamp'
                ])
                
                # Write data
                for row in logs:
                    writer.writerow([
                        row['endpoint'],
                        row['request_method'],
                        row['response_status'],
                        row['response_time_ms'],
                        row['user_agent'] or 'N/A',
                        row['ip_address'] or 'N/A',
                        row['created_at']
                    ])
                
                return output.getvalue()
        except Exception as e:
            print(f"Error exporting activity logs: {e}")
            return ""

# Initialize database manager
db_manager = DatabaseManager()

# Models
class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class PortfolioItem(BaseModel):
    ticker: str
    shares: float
    avg_price: float

class WatchlistItem(BaseModel):
    ticker: str

# Authentication functions
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Real-time market data functions
def get_real_time_data(ticker: str) -> Dict[str, Any]:
    """Get real-time market data using yfinance"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="1d", interval="1m")
        
        if hist.empty:
            raise Exception("No data available")
        
        current_price = hist['Close'].iloc[-1]
        prev_close = info.get('previousClose', current_price)
        change = current_price - prev_close
        change_pct = (change / prev_close) * 100 if prev_close else 0
        
        return {
            "ticker": ticker.upper(),
            "current_price": round(current_price, 2),
            "previous_close": round(prev_close, 2),
            "change": round(change, 2),
            "change_pct": round(change_pct, 2),
            "volume": int(hist['Volume'].iloc[-1]),
            "market_cap": info.get('marketCap', 0),
            "pe_ratio": info.get('trailingPE', 0),
            "dividend_yield": info.get('dividendYield', 0),
            "beta": info.get('beta', 0),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching real-time data: {str(e)}")

def get_technical_indicators(ticker: str, period: str = "1y") -> Dict[str, Any]:
    """Calculate advanced technical indicators"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        
        if hist.empty:
            raise Exception("No historical data available")
        
        # Calculate technical indicators
        close = hist['Close']
        high = hist['High']
        low = hist['Low']
        volume = hist['Volume']
        
        # Moving averages
        sma_20 = ta.trend.sma_indicator(close, window=20)
        sma_50 = ta.trend.sma_indicator(close, window=50)
        sma_200 = ta.trend.sma_indicator(close, window=200)
        
        # RSI
        rsi = ta.momentum.rsi(close, window=14)
        
        # MACD
        macd = ta.trend.macd(close)
        macd_signal = ta.trend.macd_signal(close)
        macd_histogram = ta.trend.macd_diff(close)
        
        # Bollinger Bands
        bb_upper = ta.volatility.bollinger_hband(close)
        bb_lower = ta.volatility.bollinger_lband(close)
        bb_middle = ta.volatility.bollinger_mavg(close)
        
        # Stochastic
        stoch_k = ta.momentum.stoch(high, low, close)
        stoch_d = ta.momentum.stoch_signal(high, low, close)
        
        # Volume indicators
        volume_sma = ta.volume.volume_sma(close, volume)
        
        # ATR (Average True Range)
        atr = ta.volatility.average_true_range(high, low, close)
        
        # Get latest values
        latest_data = {
            "ticker": ticker.upper(),
            "current_price": round(close.iloc[-1], 2),
            "sma_20": round(sma_20.iloc[-1], 2),
            "sma_50": round(sma_50.iloc[-1], 2),
            "sma_200": round(sma_200.iloc[-1], 2),
            "rsi": round(rsi.iloc[-1], 2),
            "macd": round(macd.iloc[-1], 4),
            "macd_signal": round(macd_signal.iloc[-1], 4),
            "macd_histogram": round(macd_histogram.iloc[-1], 4),
            "bb_upper": round(bb_upper.iloc[-1], 2),
            "bb_lower": round(bb_lower.iloc[-1], 2),
            "bb_middle": round(bb_middle.iloc[-1], 2),
            "stoch_k": round(stoch_k.iloc[-1], 2),
            "stoch_d": round(stoch_d.iloc[-1], 2),
            "atr": round(atr.iloc[-1], 2),
            "volume_sma": round(volume_sma.iloc[-1], 0),
            "timestamp": datetime.now().isoformat()
        }
        
        # Technical analysis signals
        signals = {
            "trend": "Bullish" if close.iloc[-1] > sma_50.iloc[-1] > sma_200.iloc[-1] else "Bearish",
            "rsi_signal": "Oversold" if rsi.iloc[-1] < 30 else "Overbought" if rsi.iloc[-1] > 70 else "Neutral",
            "macd_signal": "Bullish" if macd.iloc[-1] > macd_signal.iloc[-1] else "Bearish",
            "bb_position": "Upper" if close.iloc[-1] > bb_upper.iloc[-1] else "Lower" if close.iloc[-1] < bb_lower.iloc[-1] else "Middle"
        }
        
        latest_data["signals"] = signals
        
        return latest_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating technical indicators: {str(e)}")

def get_ml_predictions(ticker: str, days_ahead: int = 30) -> Dict[str, Any]:
    """Generate machine learning price predictions"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2y")
        
        if hist.empty or len(hist) < 100:
            raise Exception("Insufficient historical data for ML predictions")
        
        # Prepare features
        df = hist.copy()
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        df['MACD'] = ta.trend.macd(df['Close'])
        df['BB_Upper'] = ta.volatility.bollinger_hband(df['Close'])
        df['BB_Lower'] = ta.volatility.bollinger_lband(df['Close'])
        
        # Create lag features
        for i in range(1, 6):
            df[f'Close_Lag_{i}'] = df['Close'].shift(i)
            df[f'Volume_Lag_{i}'] = df['Volume'].shift(i)
        
        # Drop NaN values
        df = df.dropna()
        
        # Prepare features and target
        feature_columns = ['Close', 'Volume', 'Returns', 'Volatility', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'BB_Upper', 'BB_Lower']
        for i in range(1, 6):
            feature_columns.extend([f'Close_Lag_{i}', f'Volume_Lag_{i}'])
        
        X = df[feature_columns]
        y = df['Close'].shift(-1).dropna()
        
        # Align X and y
        X = X.iloc[:-1]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        latest_features = scaler.transform(X.iloc[-1:])
        prediction = model.predict(latest_features)[0]
        
        # Calculate confidence interval (simplified)
        confidence = model.score(X_test_scaled, y_test)
        
        # Generate future predictions
        future_predictions = []
        current_features = latest_features.copy()
        
        for day in range(1, days_ahead + 1):
            pred = model.predict(current_features)[0]
            future_predictions.append({
                "day": day,
                "predicted_price": round(pred, 2),
                "date": (datetime.now() + timedelta(days=day)).strftime("%Y-%m-%d")
            })
            
            # Update features for next prediction (simplified)
            current_features[0, 0] = pred  # Update Close price
        
        return {
            "ticker": ticker.upper(),
            "current_price": round(df['Close'].iloc[-1], 2),
            "predicted_price_1d": round(prediction, 2),
            "confidence_score": round(confidence, 3),
            "future_predictions": future_predictions,
            "model_accuracy": round(confidence * 100, 1),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating ML predictions: {str(e)}")

# API Endpoints (Rate limiting handled by middleware)
@app.post("/api/auth/register")
async def register_user(user: UserCreate):
    """Register a new user"""
    try:
        # Check if user already exists
        existing_user = db_manager.get_user_by_username(user.username)
        if existing_user:
            raise HTTPException(status_code=400, detail="Username or email already exists")
        
        # Hash password and create user
        hashed_password = hash_password(user.password)
        user_id = db_manager.create_user(user.username, user.email, hashed_password)
        
        # Create access token
        access_token = create_access_token(data={"sub": user.username})
        
        return {
            "message": "User registered successfully",
            "access_token": access_token,
            "token_type": "bearer"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration error: {str(e)}")

@app.post("/api/auth/login")
async def login_user(user: UserLogin):
    """Login user"""
    try:
        # Get user from database
        user_data = db_manager.get_user_by_username(user.username)
        
        if not user_data or not verify_password(user.password, user_data['password_hash']):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Update last login
        db_manager.update_user_last_login(user_data['id'])
        
        # Create access token
        access_token = create_access_token(data={"sub": user.username})
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "username": user_data['username']
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Login error: {str(e)}")

@app.get("/api/market/realtime/{ticker}")
async def get_realtime_data(ticker: str):
    """Get real-time market data for a ticker"""
    return get_real_time_data(ticker)

@app.get("/api/technical/{ticker}")
async def get_technical_analysis(ticker: str, period: str = "1y"):
    """Get technical analysis indicators"""
    return get_technical_indicators(ticker, period)

@app.get("/api/ml/predictions/{ticker}")
async def get_ml_predictions_endpoint(ticker: str, days: int = 30):
    """Get machine learning price predictions"""
    return get_ml_predictions(ticker, days)

@app.get("/api/portfolio", dependencies=[Depends(verify_token)])
async def get_user_portfolio(username: str = Depends(verify_token)):
    """Get user's portfolio"""
    try:
        conn = sqlite3.connect('financial_analyzer.db')
        cursor = conn.cursor()
        
        # Get user ID
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        user_id = cursor.fetchone()[0]
        
        # Get portfolio
        cursor.execute("""
            SELECT ticker, shares, avg_price, added_at 
            FROM portfolios 
            WHERE user_id = ?
        """, (user_id,))
        
        portfolio_items = []
        for row in cursor.fetchall():
            ticker, shares, avg_price, added_at = row
            
            # Get current price
            try:
                current_data = get_real_time_data(ticker)
                current_price = current_data["current_price"]
                total_value = shares * current_price
                total_cost = shares * avg_price
                gain_loss = total_value - total_cost
                gain_loss_pct = (gain_loss / total_cost) * 100 if total_cost > 0 else 0
                
                portfolio_items.append({
                    "ticker": ticker,
                    "shares": shares,
                    "avg_price": avg_price,
                    "current_price": current_price,
                    "total_value": round(total_value, 2),
                    "total_cost": round(total_cost, 2),
                    "gain_loss": round(gain_loss, 2),
                    "gain_loss_pct": round(gain_loss_pct, 2),
                    "added_at": added_at
                })
            except:
                # If can't get current price, use stored data
                portfolio_items.append({
                    "ticker": ticker,
                    "shares": shares,
                    "avg_price": avg_price,
                    "current_price": avg_price,
                    "total_value": shares * avg_price,
                    "total_cost": shares * avg_price,
                    "gain_loss": 0,
                    "gain_loss_pct": 0,
                    "added_at": added_at
                })
        
        conn.close()
        
        # Calculate portfolio summary
        total_portfolio_value = sum(item["total_value"] for item in portfolio_items)
        total_portfolio_cost = sum(item["total_cost"] for item in portfolio_items)
        total_gain_loss = total_portfolio_value - total_portfolio_cost
        total_gain_loss_pct = (total_gain_loss / total_portfolio_cost) * 100 if total_portfolio_cost > 0 else 0
        
        return {
            "portfolio": portfolio_items,
            "summary": {
                "total_value": round(total_portfolio_value, 2),
                "total_cost": round(total_portfolio_cost, 2),
                "total_gain_loss": round(total_gain_loss, 2),
                "total_gain_loss_pct": round(total_gain_loss_pct, 2),
                "num_positions": len(portfolio_items)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching portfolio: {str(e)}")

@app.post("/api/portfolio/add", dependencies=[Depends(verify_token)])
async def add_to_portfolio(item: PortfolioItem, username: str = Depends(verify_token)):
    """Add stock to portfolio"""
    try:
        conn = sqlite3.connect('financial_analyzer.db')
        cursor = conn.cursor()
        
        # Get user ID
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        user_id = cursor.fetchone()[0]
        
        # Check if ticker already exists in portfolio
        cursor.execute("""
            SELECT shares, avg_price FROM portfolios 
            WHERE user_id = ? AND ticker = ?
        """, (user_id, item.ticker))
        
        existing = cursor.fetchone()
        
        if existing:
            # Update existing position
            old_shares, old_avg_price = existing
            new_shares = old_shares + item.shares
            new_avg_price = ((old_shares * old_avg_price) + (item.shares * item.avg_price)) / new_shares
            
            cursor.execute("""
                UPDATE portfolios 
                SET shares = ?, avg_price = ? 
                WHERE user_id = ? AND ticker = ?
            """, (new_shares, new_avg_price, user_id, item.ticker))
        else:
            # Add new position
            cursor.execute("""
                INSERT INTO portfolios (user_id, ticker, shares, avg_price)
                VALUES (?, ?, ?, ?)
            """, (user_id, item.ticker, item.shares, item.avg_price))
        
        conn.commit()
        conn.close()
        
        return {"message": f"Added {item.shares} shares of {item.ticker} to portfolio"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding to portfolio: {str(e)}")

@app.get("/api/watchlist", dependencies=[Depends(verify_token)])
async def get_watchlist(username: str = Depends(verify_token)):
    """Get user's watchlist"""
    try:
        conn = sqlite3.connect('financial_analyzer.db')
        cursor = conn.cursor()
        
        # Get user ID
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        user_id = cursor.fetchone()[0]
        
        # Get watchlist
        cursor.execute("""
            SELECT ticker, added_at FROM watchlists 
            WHERE user_id = ?
        """, (user_id,))
        
        watchlist_items = []
        for row in cursor.fetchall():
            ticker, added_at = row
            
            try:
                current_data = get_real_time_data(ticker)
                watchlist_items.append({
                    "ticker": ticker,
                    "current_price": current_data["current_price"],
                    "change": current_data["change"],
                    "change_pct": current_data["change_pct"],
                    "added_at": added_at
                })
            except:
                watchlist_items.append({
                    "ticker": ticker,
                    "current_price": 0,
                    "change": 0,
                    "change_pct": 0,
                    "added_at": added_at
                })
        
        conn.close()
        
        return {"watchlist": watchlist_items}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching watchlist: {str(e)}")

@app.post("/api/watchlist/add", dependencies=[Depends(verify_token)])
async def add_to_watchlist(item: WatchlistItem, username: str = Depends(verify_token)):
    """Add stock to watchlist"""
    try:
        conn = sqlite3.connect('financial_analyzer.db')
        cursor = conn.cursor()
        
        # Get user ID
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        user_id = cursor.fetchone()[0]
        
        # Check if already in watchlist
        cursor.execute("""
            SELECT id FROM watchlists 
            WHERE user_id = ? AND ticker = ?
        """, (user_id, item.ticker))
        
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="Stock already in watchlist")
        
        # Add to watchlist
        cursor.execute("""
            INSERT INTO watchlists (user_id, ticker)
            VALUES (?, ?)
        """, (user_id, item.ticker))
        
        conn.commit()
        conn.close()
        
        return {"message": f"Added {item.ticker} to watchlist"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding to watchlist: {str(e)}")

@app.get("/api/market/overview")
async def get_market_overview():
    """Get market overview with real-time data"""
    try:
        # Major indices
        indices = ["^GSPC", "^IXIC", "^DJI", "^RUT"]
        indices_data = []
        
        for index in indices:
            try:
                data = get_real_time_data(index)
                indices_data.append({
                    "symbol": index.replace("^", ""),
                    "name": {"^GSPC": "S&P 500", "^IXIC": "NASDAQ", "^DJI": "DOW", "^RUT": "Russell 2000"}[index],
                    "value": data["current_price"],
                    "change": data["change"],
                    "change_pct": data["change_pct"]
                })
            except:
                continue
        
        # Trending stocks
        trending_tickers = ["TSLA", "NVDA", "META", "GOOGL", "AAPL", "MSFT"]
        trending_data = []
        
        for ticker in trending_tickers:
            try:
                data = get_real_time_data(ticker)
                trending_data.append({
                    "ticker": ticker,
                    "price": data["current_price"],
                    "change": data["change"],
                    "change_pct": data["change_pct"],
                    "volume": data["volume"]
                })
            except:
                continue
        
        return {
            "indices": indices_data,
            "trending_stocks": trending_data,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching market overview: {str(e)}")

@app.get("/api/rate-limits")
async def get_rate_limits():
    """Get current rate limit configuration"""
    return {
        "rate_limits": RATE_LIMIT_CONFIG,
        "description": "Rate limits are applied per client (user ID or IP address) per endpoint type",
        "note": "Rate limits are enforced to ensure fair usage and protect API resources"
    }

# Export Endpoints
@app.get("/api/export/portfolio/csv", dependencies=[Depends(verify_token)])
async def export_portfolio_csv(username: str = Depends(verify_token)):
    """Export portfolio to CSV format"""
    try:
        # Get user ID
        user_data = db_manager.get_user_by_username(username)
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Generate CSV content
        csv_content = db_manager.export_portfolio_csv(user_data['id'])
        if not csv_content:
            raise HTTPException(status_code=500, detail="Error generating CSV export")
        
        # Create streaming response
        return StreamingResponse(
            io.StringIO(csv_content),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=portfolio_{username}_{datetime.now().strftime('%Y%m%d')}.csv"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export error: {str(e)}")

@app.get("/api/export/transactions/csv", dependencies=[Depends(verify_token)])
async def export_transactions_csv(
    username: str = Depends(verify_token),
    start_date: str = None,
    end_date: str = None
):
    """Export transaction history to CSV format"""
    try:
        # Get user ID
        user_data = db_manager.get_user_by_username(username)
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Generate CSV content
        csv_content = db_manager.export_transactions_csv(user_data['id'], start_date, end_date)
        if not csv_content:
            raise HTTPException(status_code=500, detail="Error generating CSV export")
        
        # Create streaming response
        return StreamingResponse(
            io.StringIO(csv_content),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=transactions_{username}_{datetime.now().strftime('%Y%m%d')}.csv"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export error: {str(e)}")

@app.get("/api/export/watchlist/csv", dependencies=[Depends(verify_token)])
async def export_watchlist_csv(username: str = Depends(verify_token)):
    """Export watchlist to CSV format"""
    try:
        # Get user ID
        user_data = db_manager.get_user_by_username(username)
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Generate CSV content
        csv_content = db_manager.export_watchlist_csv(user_data['id'])
        if not csv_content:
            raise HTTPException(status_code=500, detail="Error generating CSV export")
        
        # Create streaming response
        return StreamingResponse(
            io.StringIO(csv_content),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=watchlist_{username}_{datetime.now().strftime('%Y%m%d')}.csv"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export error: {str(e)}")

@app.get("/api/export/portfolio/summary", dependencies=[Depends(verify_token)])
async def export_portfolio_summary(username: str = Depends(verify_token)):
    """Export portfolio summary as JSON"""
    try:
        # Get user ID
        user_data = db_manager.get_user_by_username(username)
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Generate summary
        summary = db_manager.export_portfolio_summary_json(user_data['id'])
        if not summary:
            raise HTTPException(status_code=500, detail="Error generating portfolio summary")
        
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export error: {str(e)}")

@app.get("/api/export/activity-logs/csv", dependencies=[Depends(verify_token)])
async def export_activity_logs_csv(
    username: str = Depends(verify_token),
    days: int = 30
):
    """Export user activity logs to CSV format"""
    try:
        # Get user ID
        user_data = db_manager.get_user_by_username(username)
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Validate days parameter
        if days < 1 or days > 365:
            raise HTTPException(status_code=400, detail="Days must be between 1 and 365")
        
        # Generate CSV content
        csv_content = db_manager.export_user_activity_logs(user_data['id'], days)
        if not csv_content:
            raise HTTPException(status_code=500, detail="Error generating CSV export")
        
        # Create streaming response
        return StreamingResponse(
            io.StringIO(csv_content),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=activity_logs_{username}_{datetime.now().strftime('%Y%m%d')}.csv"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export error: {str(e)}")

@app.get("/api/export/all", dependencies=[Depends(verify_token)])
async def export_all_data(username: str = Depends(verify_token)):
    """Export all user data as a ZIP file"""
    try:
        # Get user ID
        user_data = db_manager.get_user_by_username(username)
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Create ZIP file in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add portfolio CSV
            portfolio_csv = db_manager.export_portfolio_csv(user_data['id'])
            if portfolio_csv:
                zip_file.writestr(f"portfolio_{username}.csv", portfolio_csv)
            
            # Add transactions CSV
            transactions_csv = db_manager.export_transactions_csv(user_data['id'])
            if transactions_csv:
                zip_file.writestr(f"transactions_{username}.csv", transactions_csv)
            
            # Add watchlist CSV
            watchlist_csv = db_manager.export_watchlist_csv(user_data['id'])
            if watchlist_csv:
                zip_file.writestr(f"watchlist_{username}.csv", watchlist_csv)
            
            # Add portfolio summary JSON
            portfolio_summary = db_manager.export_portfolio_summary_json(user_data['id'])
            if portfolio_summary:
                zip_file.writestr(f"portfolio_summary_{username}.json", json.dumps(portfolio_summary, indent=2))
            
            # Add activity logs CSV
            activity_logs_csv = db_manager.export_user_activity_logs(user_data['id'], 30)
            if activity_logs_csv:
                zip_file.writestr(f"activity_logs_{username}.csv", activity_logs_csv)
            
            # Add README file
            readme_content = f"""Financial Analyzer Pro - Data Export
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Username: {username}

Files included:
- portfolio_{username}.csv: Current portfolio holdings
- transactions_{username}.csv: Complete transaction history
- watchlist_{username}.csv: Watchlist with price alerts
- portfolio_summary_{username}.json: Portfolio summary and statistics
- activity_logs_{username}.csv: API usage and activity logs

This export contains all your financial data from Financial Analyzer Pro.
"""
            zip_file.writestr("README.txt", readme_content)
        
        # Reset buffer position
        zip_buffer.seek(0)
        
        # Create streaming response
        return StreamingResponse(
            io.BytesIO(zip_buffer.getvalue()),
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename=financial_data_{username}_{datetime.now().strftime('%Y%m%d')}.zip"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export error: {str(e)}")

@app.post("/api/admin/cleanup", dependencies=[Depends(verify_token)])
async def cleanup_database(username: str = Depends(verify_token)):
    """Clean up expired data and old logs (Admin function)"""
    try:
        # Get user ID
        user_data = db_manager.get_user_by_username(username)
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")
        
        # For now, allow any authenticated user to run cleanup
        # In production, you might want to check admin privileges
        db_manager.cleanup_expired_data()
        
        return {
            "message": "Database cleanup completed successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup error: {str(e)}")

@app.get("/api/export/portfolio/performance", dependencies=[Depends(verify_token)])
async def export_portfolio_performance(username: str = Depends(verify_token)):
    """Export portfolio performance analysis"""
    try:
        # Get user ID
        user_data = db_manager.get_user_by_username(username)
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get portfolio data
        portfolio_data = db_manager.get_user_portfolio(user_data['id'])
        if not portfolio_data:
            raise HTTPException(status_code=404, detail="No portfolio data found")
        
        # Calculate performance metrics
        performance_data = {
            "export_date": datetime.now().isoformat(),
            "username": username,
            "portfolio_summary": {
                "total_positions": len(portfolio_data['portfolio']),
                "total_shares": sum(pos['shares'] for pos in portfolio_data['portfolio']),
                "total_cost": sum(pos['shares'] * pos['avg_price'] for pos in portfolio_data['portfolio'])
            },
            "positions": []
        }
        
        # Add current market data for each position
        for position in portfolio_data['portfolio']:
            try:
                market_data = get_real_time_data(position['ticker'])
                current_price = market_data['current_price']
                shares = position['shares']
                avg_price = position['avg_price']
                
                total_cost = shares * avg_price
                current_value = shares * current_price
                gain_loss = current_value - total_cost
                gain_loss_pct = (gain_loss / total_cost) * 100 if total_cost > 0 else 0
                
                performance_data["positions"].append({
                    "ticker": position['ticker'],
                    "shares": shares,
                    "average_price": avg_price,
                    "current_price": current_price,
                    "total_cost": total_cost,
                    "current_value": current_value,
                    "gain_loss": gain_loss,
                    "gain_loss_pct": gain_loss_pct,
                    "purchase_date": position.get('purchase_date', 'N/A')
                })
            except:
                # If market data unavailable, use stored data
                performance_data["positions"].append({
                    "ticker": position['ticker'],
                    "shares": position['shares'],
                    "average_price": position['avg_price'],
                    "current_price": position['avg_price'],
                    "total_cost": position['shares'] * position['avg_price'],
                    "current_value": position['shares'] * position['avg_price'],
                    "gain_loss": 0,
                    "gain_loss_pct": 0,
                    "purchase_date": position.get('purchase_date', 'N/A')
                })
        
        return performance_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Performance export error: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Financial Analyzer Pro API v2.0",
        "version": "2.0.0",
        "features": [
            "Real-time market data",
            "User authentication & portfolios",
            "Advanced technical analysis",
            "Machine learning predictions",
            "Watchlist management",
            "API rate limiting",
            "Data export functionality"
        ],
        "endpoints": {
            "auth": "/api/auth/register, /api/auth/login",
            "market": "/api/market/realtime/{ticker}, /api/market/overview",
            "technical": "/api/technical/{ticker}",
            "ml": "/api/ml/predictions/{ticker}",
            "portfolio": "/api/portfolio, /api/portfolio/add",
            "watchlist": "/api/watchlist, /api/watchlist/add",
            "export": "/api/export/portfolio/csv, /api/export/transactions/csv, /api/export/watchlist/csv, /api/export/portfolio/summary, /api/export/portfolio/performance, /api/export/activity-logs/csv, /api/export/all",
            "admin": "/api/admin/cleanup",
            "rate_limits": "/api/rate-limits"
        }
    }

# Error handlers
@app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content=exc.detail,
        headers={"Retry-After": str(exc.detail["retry_after"])}
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "status_code": 500}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
