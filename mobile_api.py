"""
Enhanced API Endpoints for Mobile App
Supports user authentication, portfolio management, and subscription features
"""

from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List, Dict, Any, Optional
import sqlite3
from datetime import datetime, timedelta
import uuid

# Import enhanced auth service
from enhanced_auth_service import (
    UserCreate, UserLogin, UserResponse, TokenResponse,
    PortfolioCreate, PortfolioPosition, WatchlistCreate, PriceAlertCreate,
    user_service, token_service, SUBSCRIPTION_TIERS
)

# Initialize FastAPI app
app = FastAPI(
    title="FinancialAnalyzerApp Mobile API",
    description="Enhanced API for mobile financial analysis app",
    version="2.0.0"
)

# CORS middleware for mobile app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> UserResponse:
    """Get current authenticated user"""
    token = credentials.credentials
    payload = token_service.verify_token(token, "access")
    
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user_id = payload.get("sub")
    user = user_service.get_user_by_id(user_id)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user

def check_subscription_limit(user: UserResponse, limit_type: str, current_count: int) -> bool:
    """Check if user has reached subscription limit"""
    limits = user_service.get_user_limits(user.id)
    max_limit = limits.get(f"max_{limit_type}", 0)
    
    if max_limit == -1:  # Unlimited
        return True
    
    return current_count < max_limit

# Authentication endpoints
@app.post("/api/auth/register", response_model=UserResponse)
async def register_user(user_data: UserCreate, request: Request):
    """Register new user"""
    try:
        user = user_service.create_user(user_data)
        
        # Log registration activity
        with sqlite3.connect("financial_analyzer.db") as conn:
            conn.execute("""
                INSERT INTO user_activity_logs (id, user_id, activity_type, ip_address, user_agent)
                VALUES (?, ?, ?, ?, ?)
            """, (
                str(uuid.uuid4()),
                user.id,
                "user_registration",
                request.client.host,
                request.headers.get("user-agent", "")
            ))
            conn.commit()
        
        return user
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/auth/login", response_model=TokenResponse)
async def login_user(login_data: UserLogin, request: Request):
    """Login user and return tokens"""
    user = user_service.authenticate_user(login_data.email, login_data.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create tokens
    access_token = token_service.create_access_token(data={"sub": user.id})
    refresh_token = token_service.create_refresh_token(data={"sub": user.id})
    
    # Store refresh token
    token_service.store_refresh_token(
        user.id,
        refresh_token,
        request.headers.get("user-agent", ""),
        request.client.host
    )
    
    # Log login activity
    with sqlite3.connect("financial_analyzer.db") as conn:
        conn.execute("""
            INSERT INTO user_activity_logs (id, user_id, activity_type, ip_address, user_agent)
            VALUES (?, ?, ?, ?, ?)
        """, (
            str(uuid.uuid4()),
            user.id,
            "user_login",
            request.client.host,
            request.headers.get("user-agent", "")
        ))
        conn.commit()
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=30 * 60  # 30 minutes
    )

@app.post("/api/auth/refresh", response_model=TokenResponse)
async def refresh_token(refresh_token: str, request: Request):
    """Refresh access token using refresh token"""
    payload = token_service.verify_token(refresh_token, "refresh")
    
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    user_id = payload.get("sub")
    user = user_service.get_user_by_id(user_id)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    # Create new access token
    access_token = token_service.create_access_token(data={"sub": user.id})
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,  # Keep same refresh token
        token_type="bearer",
        expires_in=30 * 60
    )

@app.post("/api/auth/logout")
async def logout_user(refresh_token: str, current_user: UserResponse = Depends(get_current_user)):
    """Logout user and revoke refresh token"""
    token_service.revoke_refresh_token(refresh_token)
    
    # Log logout activity
    with sqlite3.connect("financial_analyzer.db") as conn:
        conn.execute("""
            INSERT INTO user_activity_logs (id, user_id, activity_type)
            VALUES (?, ?, ?)
        """, (str(uuid.uuid4()), current_user.id, "user_logout"))
        conn.commit()
    
    return {"message": "Successfully logged out"}

# User profile endpoints
@app.get("/api/user/profile", response_model=UserResponse)
async def get_user_profile(current_user: UserResponse = Depends(get_current_user)):
    """Get current user profile"""
    return current_user

@app.put("/api/user/profile", response_model=UserResponse)
async def update_user_profile(
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    current_user: UserResponse = Depends(get_current_user)
):
    """Update user profile"""
    with sqlite3.connect("financial_analyzer.db") as conn:
        conn.execute("""
            UPDATE users SET first_name = ?, last_name = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (first_name, last_name, current_user.id))
        conn.commit()
    
    return user_service.get_user_by_id(current_user.id)

@app.get("/api/user/limits")
async def get_user_limits(current_user: UserResponse = Depends(get_current_user)):
    """Get user's subscription limits"""
    return user_service.get_user_limits(current_user.id)

# Portfolio endpoints
@app.get("/api/portfolios")
async def get_user_portfolios(current_user: UserResponse = Depends(get_current_user)):
    """Get user's portfolios"""
    with sqlite3.connect("financial_analyzer.db") as conn:
        cursor = conn.execute("""
            SELECT id, name, description, is_default, created_at, updated_at
            FROM portfolios WHERE user_id = ? ORDER BY is_default DESC, created_at ASC
        """, (current_user.id,))
        
        portfolios = []
        for row in cursor.fetchall():
            portfolios.append({
                "id": row[0],
                "name": row[1],
                "description": row[2],
                "is_default": bool(row[3]),
                "created_at": row[4],
                "updated_at": row[5]
            })
        
        return portfolios

@app.post("/api/portfolios")
async def create_portfolio(
    portfolio_data: PortfolioCreate,
    current_user: UserResponse = Depends(get_current_user)
):
    """Create new portfolio"""
    # Check portfolio limit
    with sqlite3.connect("financial_analyzer.db") as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM portfolios WHERE user_id = ?", (current_user.id,))
        current_count = cursor.fetchone()[0]
    
    if not check_subscription_limit(current_user, "portfolios", current_count):
        limits = user_service.get_user_limits(current_user.id)
        raise HTTPException(
            status_code=403,
            detail=f"Portfolio limit reached. Upgrade to {limits['name']} for more portfolios."
        )
    
    portfolio_id = str(uuid.uuid4())
    
    with sqlite3.connect("financial_analyzer.db") as conn:
        conn.execute("""
            INSERT INTO portfolios (id, user_id, name, description)
            VALUES (?, ?, ?, ?)
        """, (portfolio_id, current_user.id, portfolio_data.name, portfolio_data.description))
        conn.commit()
    
    return {"id": portfolio_id, "message": "Portfolio created successfully"}

@app.get("/api/portfolios/{portfolio_id}/positions")
async def get_portfolio_positions(
    portfolio_id: str,
    current_user: UserResponse = Depends(get_current_user)
):
    """Get portfolio positions"""
    with sqlite3.connect("financial_analyzer.db") as conn:
        # Verify portfolio ownership
        cursor = conn.execute("""
            SELECT id FROM portfolios WHERE id = ? AND user_id = ?
        """, (portfolio_id, current_user.id))
        
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Portfolio not found")
        
        # Get positions
        cursor = conn.execute("""
            SELECT id, ticker, shares, cost_basis, purchase_date, created_at
            FROM portfolio_positions WHERE portfolio_id = ?
            ORDER BY created_at DESC
        """, (portfolio_id,))
        
        positions = []
        for row in cursor.fetchall():
            positions.append({
                "id": row[0],
                "ticker": row[1],
                "shares": row[2],
                "cost_basis": row[3],
                "purchase_date": row[4],
                "created_at": row[5]
            })
        
        return positions

@app.post("/api/portfolios/{portfolio_id}/positions")
async def add_portfolio_position(
    portfolio_id: str,
    position_data: PortfolioPosition,
    current_user: UserResponse = Depends(get_current_user)
):
    """Add position to portfolio"""
    # Check position limit
    with sqlite3.connect("financial_analyzer.db") as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM portfolio_positions WHERE portfolio_id = ?", (portfolio_id,))
        current_count = cursor.fetchone()[0]
    
    if not check_subscription_limit(current_user, "positions", current_count):
        limits = user_service.get_user_limits(current_user.id)
        raise HTTPException(
            status_code=403,
            detail=f"Position limit reached. Upgrade to {limits['name']} for more positions."
        )
    
    position_id = str(uuid.uuid4())
    
    with sqlite3.connect("financial_analyzer.db") as conn:
        # Verify portfolio ownership
        cursor = conn.execute("""
            SELECT id FROM portfolios WHERE id = ? AND user_id = ?
        """, (portfolio_id, current_user.id))
        
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Portfolio not found")
        
        conn.execute("""
            INSERT INTO portfolio_positions (id, portfolio_id, ticker, shares, cost_basis, purchase_date)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (position_id, portfolio_id, position_data.ticker, position_data.shares, 
              position_data.cost_basis, position_data.purchase_date))
        conn.commit()
    
    return {"id": position_id, "message": "Position added successfully"}

# Watchlist endpoints
@app.get("/api/watchlists")
async def get_user_watchlists(current_user: UserResponse = Depends(get_current_user)):
    """Get user's watchlists"""
    with sqlite3.connect("financial_analyzer.db") as conn:
        cursor = conn.execute("""
            SELECT id, name, created_at FROM watchlists WHERE user_id = ?
            ORDER BY created_at ASC
        """, (current_user.id,))
        
        watchlists = []
        for row in cursor.fetchall():
            watchlists.append({
                "id": row[0],
                "name": row[1],
                "created_at": row[2]
            })
        
        return watchlists

@app.post("/api/watchlists")
async def create_watchlist(
    watchlist_data: WatchlistCreate,
    current_user: UserResponse = Depends(get_current_user)
):
    """Create new watchlist"""
    # Check watchlist limit
    with sqlite3.connect("financial_analyzer.db") as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM watchlists WHERE user_id = ?", (current_user.id,))
        current_count = cursor.fetchone()[0]
    
    if not check_subscription_limit(current_user, "watchlists", current_count):
        limits = user_service.get_user_limits(current_user.id)
        raise HTTPException(
            status_code=403,
            detail=f"Watchlist limit reached. Upgrade to {limits['name']} for more watchlists."
        )
    
    watchlist_id = str(uuid.uuid4())
    
    with sqlite3.connect("financial_analyzer.db") as conn:
        conn.execute("""
            INSERT INTO watchlists (id, user_id, name)
            VALUES (?, ?, ?)
        """, (watchlist_id, current_user.id, watchlist_data.name))
        conn.commit()
    
    return {"id": watchlist_id, "message": "Watchlist created successfully"}

@app.get("/api/watchlists/{watchlist_id}/items")
async def get_watchlist_items(
    watchlist_id: str,
    current_user: UserResponse = Depends(get_current_user)
):
    """Get watchlist items"""
    with sqlite3.connect("financial_analyzer.db") as conn:
        # Verify watchlist ownership
        cursor = conn.execute("""
            SELECT id FROM watchlists WHERE id = ? AND user_id = ?
        """, (watchlist_id, current_user.id))
        
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Watchlist not found")
        
        # Get items
        cursor = conn.execute("""
            SELECT id, ticker, added_at FROM watchlist_items WHERE watchlist_id = ?
            ORDER BY added_at DESC
        """, (watchlist_id,))
        
        items = []
        for row in cursor.fetchall():
            items.append({
                "id": row[0],
                "ticker": row[1],
                "added_at": row[2]
            })
        
        return items

@app.post("/api/watchlists/{watchlist_id}/items")
async def add_watchlist_item(
    watchlist_id: str,
    ticker: str,
    current_user: UserResponse = Depends(get_current_user)
):
    """Add item to watchlist"""
    # Check watchlist item limit
    with sqlite3.connect("financial_analyzer.db") as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM watchlist_items WHERE watchlist_id = ?", (watchlist_id,))
        current_count = cursor.fetchone()[0]
    
    if not check_subscription_limit(current_user, "watchlist_items", current_count):
        limits = user_service.get_user_limits(current_user.id)
        raise HTTPException(
            status_code=403,
            detail=f"Watchlist item limit reached. Upgrade to {limits['name']} for more items."
        )
    
    item_id = str(uuid.uuid4())
    
    with sqlite3.connect("financial_analyzer.db") as conn:
        # Verify watchlist ownership
        cursor = conn.execute("""
            SELECT id FROM watchlists WHERE id = ? AND user_id = ?
        """, (watchlist_id, current_user.id))
        
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Watchlist not found")
        
        # Check if ticker already exists
        cursor = conn.execute("""
            SELECT id FROM watchlist_items WHERE watchlist_id = ? AND ticker = ?
        """, (watchlist_id, ticker.upper()))
        
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="Ticker already in watchlist")
        
        conn.execute("""
            INSERT INTO watchlist_items (id, watchlist_id, ticker)
            VALUES (?, ?, ?)
        """, (item_id, watchlist_id, ticker.upper()))
        conn.commit()
    
    return {"id": item_id, "message": "Item added to watchlist successfully"}

# Price alerts endpoints
@app.get("/api/alerts")
async def get_user_alerts(current_user: UserResponse = Depends(get_current_user)):
    """Get user's price alerts"""
    limits = user_service.get_user_limits(current_user.id)
    if not limits.get("price_alerts", False):
        raise HTTPException(
            status_code=403,
            detail="Price alerts not available in your subscription tier"
        )
    
    with sqlite3.connect("financial_analyzer.db") as conn:
        cursor = conn.execute("""
            SELECT id, ticker, alert_type, target_value, is_active, created_at, triggered_at
            FROM price_alerts WHERE user_id = ? ORDER BY created_at DESC
        """, (current_user.id,))
        
        alerts = []
        for row in cursor.fetchall():
            alerts.append({
                "id": row[0],
                "ticker": row[1],
                "alert_type": row[2],
                "target_value": row[3],
                "is_active": bool(row[4]),
                "created_at": row[5],
                "triggered_at": row[6]
            })
        
        return alerts

@app.post("/api/alerts")
async def create_price_alert(
    alert_data: PriceAlertCreate,
    current_user: UserResponse = Depends(get_current_user)
):
    """Create price alert"""
    limits = user_service.get_user_limits(current_user.id)
    if not limits.get("price_alerts", False):
        raise HTTPException(
            status_code=403,
            detail="Price alerts not available in your subscription tier"
        )
    
    alert_id = str(uuid.uuid4())
    
    with sqlite3.connect("financial_analyzer.db") as conn:
        conn.execute("""
            INSERT INTO price_alerts (id, user_id, ticker, alert_type, target_value)
            VALUES (?, ?, ?, ?, ?)
        """, (alert_id, current_user.id, alert_data.ticker.upper(), 
              alert_data.alert_type, alert_data.target_value))
        conn.commit()
    
    return {"id": alert_id, "message": "Price alert created successfully"}

# Subscription endpoints
@app.get("/api/subscription/tiers")
async def get_subscription_tiers():
    """Get available subscription tiers"""
    return SUBSCRIPTION_TIERS

@app.post("/api/subscription/upgrade")
async def upgrade_subscription(
    tier: str,
    current_user: UserResponse = Depends(get_current_user)
):
    """Upgrade user subscription (placeholder for payment integration)"""
    if tier not in SUBSCRIPTION_TIERS:
        raise HTTPException(status_code=400, detail="Invalid subscription tier")
    
    # In production, integrate with payment processor (Stripe, PayPal, etc.)
    success = user_service.update_subscription_tier(current_user.id, tier)
    
    if success:
        return {"message": f"Successfully upgraded to {SUBSCRIPTION_TIERS[tier]['name']}"}
    else:
        raise HTTPException(status_code=400, detail="Failed to upgrade subscription")

# Health check endpoint
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
