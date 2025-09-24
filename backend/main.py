#!/usr/bin/env python3
"""
Financial Analyzer Pro - FastAPI Backend
Phase 1: Foundation & Real-Time Features

This FastAPI backend integrates with existing Streamlit functionality
while adding real-time WebSocket capabilities and enhanced performance.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
import asyncio
import json
import time
from typing import List, Dict, Optional
import logging

# Import existing functionality from current app
from portfolio_manager import PortfolioManager, EnhancedPortfolioManager
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Import new backend components
from database import get_db, engine, Base
from models import User, Portfolio, Position, Transaction
from schemas import UserCreate, UserResponse, PortfolioCreate, PortfolioResponse
from auth import create_access_token, verify_token, get_current_user
from cache_service import CacheService
from websocket_manager import ConnectionManager
from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="Financial Analyzer Pro - Enhanced Real-Time API",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8501"],  # React + Streamlit
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
cache_service = CacheService()
websocket_manager = ConnectionManager()
security = HTTPBearer()

# Create database tables
Base.metadata.create_all(bind=engine)

# Initialize existing portfolio manager
portfolio_manager = PortfolioManager()
enhanced_portfolio_manager = EnhancedPortfolioManager()

# Background task for real-time market data
market_data_task = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global market_data_task
    logger.info("Starting Financial Analyzer Pro API...")
    
    # Start background market data task
    market_data_task = asyncio.create_task(market_data_broadcaster())
    logger.info("Market data broadcaster started")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global market_data_task
    if market_data_task:
        market_data_task.cancel()
    logger.info("Financial Analyzer Pro API shutdown")

# Health check endpoints
@app.get("/health")
async def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": settings.api_version
    }

@app.get("/health/database")
async def database_health(db: Session = Depends(get_db)):
    """Database health check"""
    try:
        # Test database connection
        db.execute("SELECT 1")
        return {"database": "healthy", "status": "connected"}
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {"database": "unhealthy", "error": str(e)}

@app.get("/health/cache")
async def cache_health():
    """Redis cache health check"""
    try:
        await cache_service.set("health_check", "ok", 10)
        result = await cache_service.get("health_check")
        return {"cache": "healthy", "status": "connected"}
    except Exception as e:
        logger.error(f"Cache health check failed: {e}")
        return {"cache": "unhealthy", "error": str(e)}

# Authentication endpoints
@app.post("/auth/register", response_model=UserResponse)
async def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    try:
        # Check if user already exists
        existing_user = db.query(User).filter(User.email == user_data.email).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Create new user
        user = User(
            username=user_data.username,
            email=user_data.email,
            password_hash=user_data.password  # In production, hash this
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        
        # Create default portfolio
        portfolio_id = enhanced_portfolio_manager._get_or_create_user_portfolio(user.id)
        
        return UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            created_at=user.created_at
        )
    except Exception as e:
        logger.error(f"User registration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )

@app.post("/auth/login")
async def login_user(user_data: UserCreate, db: Session = Depends(get_db)):
    """Login user and return access token"""
    try:
        user = db.query(User).filter(User.email == user_data.email).first()
        if not user or user.password_hash != user_data.password:  # In production, verify hash
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        access_token = create_access_token(data={"sub": str(user.id)})
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": UserResponse(
                id=user.id,
                username=user.username,
                email=user.email,
                created_at=user.created_at
            )
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User login failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

# Market data endpoints (using existing yfinance integration)
@app.get("/api/market-data/{symbol}")
async def get_market_data(symbol: str, period: str = "1mo"):
    """Get market data for a symbol (using existing functionality)"""
    try:
        # Check cache first
        cache_key = f"market_data_{symbol}_{period}"
        cached_data = await cache_service.get(cache_key)
        if cached_data:
            return cached_data
        
        # Get data using existing yfinance integration
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        
        if data.empty:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No data found for symbol {symbol}"
            )
        
        # Convert to JSON-serializable format
        result = {
            "symbol": symbol,
            "period": period,
            "data": data.reset_index().to_dict("records"),
            "last_updated": datetime.now().isoformat()
        }
        
        # Cache for 5 minutes
        await cache_service.set(cache_key, result, 300)
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Market data fetch failed for {symbol}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch market data"
        )

@app.get("/api/market-overview")
async def get_market_overview():
    """Get market overview data (using existing functionality)"""
    try:
        cache_key = "market_overview"
        cached_data = await cache_service.get(cache_key)
        if cached_data:
            return cached_data
        
        # Use existing market overview logic
        symbols = ['^GSPC', '^IXIC', '^DJI', '^VIX']
        overview = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                hist = ticker.history(period="2d")
                
                if not hist.empty and len(hist) >= 2:
                    current_price = hist['Close'].iloc[-1]
                    previous_price = hist['Close'].iloc[-2]
                    change = current_price - previous_price
                    change_percent = (change / previous_price) * 100
                    
                    overview[symbol] = {
                        'price': float(current_price),
                        'change': float(change),
                        'change_percent': float(change_percent),
                        'name': info.get('longName', symbol)
                    }
            except Exception as e:
                logger.warning(f"Could not fetch {symbol}: {e}")
        
        # Cache for 2 minutes
        await cache_service.set(cache_key, overview, 120)
        
        return overview
    except Exception as e:
        logger.error(f"Market overview fetch failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch market overview"
        )

# Portfolio endpoints (using existing portfolio manager)
@app.get("/api/portfolio/{user_id}")
async def get_user_portfolio(user_id: int, current_user: User = Depends(get_current_user)):
    """Get user's portfolio summary (using existing enhanced portfolio manager)"""
    try:
        if current_user.id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        # Use existing enhanced portfolio manager
        portfolio_summary = enhanced_portfolio_manager.get_portfolio_summary(user_id)
        
        return portfolio_summary
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Portfolio fetch failed for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch portfolio"
        )

@app.post("/api/portfolio/{user_id}/positions")
async def add_position(
    user_id: int,
    symbol: str,
    shares: float,
    price: float,
    purchase_date: str,
    transaction_type: str = "BUY",
    fees: float = 0.0,
    notes: str = "",
    current_user: User = Depends(get_current_user)
):
    """Add position to user's portfolio (using existing portfolio manager)"""
    try:
        if current_user.id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        # Parse purchase date
        purchase_date_obj = datetime.strptime(purchase_date, "%Y-%m-%d").date()
        
        # Use existing enhanced portfolio manager
        success = enhanced_portfolio_manager.add_position(
            user_id, symbol, shares, price, purchase_date_obj, transaction_type, fees, notes
        )
        
        if success:
            # Broadcast portfolio update via WebSocket
            await websocket_manager.broadcast_portfolio_update(user_id)
            return {"message": "Position added successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to add position"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Add position failed for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add position"
        )

# WebSocket endpoint for real-time updates
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: int):
    """WebSocket endpoint for real-time updates"""
    await websocket_manager.connect(websocket, user_id)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            if message.get("type") == "ping":
                await websocket_manager.send_personal_message(
                    json.dumps({"type": "pong", "timestamp": time.time()}),
                    websocket
                )
            elif message.get("type") == "subscribe":
                # Subscribe to specific data streams
                await websocket_manager.subscribe_user(user_id, message.get("symbols", []))
                
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket, user_id)
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {e}")
        websocket_manager.disconnect(websocket, user_id)

# Background task for market data broadcasting
async def market_data_broadcaster():
    """Background task to broadcast real-time market data"""
    while True:
        try:
            # Get market overview data
            overview = await get_market_overview()
            
            # Broadcast to all connected users
            await websocket_manager.broadcast_market_data(overview)
            
            # Wait 30 seconds before next update
            await asyncio.sleep(30)
            
        except Exception as e:
            logger.error(f"Market data broadcaster error: {e}")
            await asyncio.sleep(30)

# Global markets endpoint (using existing functionality)
@app.get("/api/global-markets")
async def get_global_markets():
    """Get global market indices (using existing functionality)"""
    try:
        cache_key = "global_markets"
        cached_data = await cache_service.get(cache_key)
        if cached_data:
            return cached_data
        
        # Use existing global markets logic
        indices = [
            ('^FTSE', 'FTSE 100'),
            ('^GDAXI', 'DAX'),
            ('^FCHI', 'CAC 40'),
            ('^N225', 'Nikkei 225'),
            ('^HSI', 'Hang Seng'),
            ('000001.SS', 'SSE Composite'),
            ('^BSESN', 'BSE Sensex'),
            ('^AXJO', 'ASX 200'),
            ('^KS11', 'KOSPI'),
            ('^JKSE', 'Jakarta Composite')
        ]
        
        markets = []
        for sym, name in indices:
            try:
                ticker = yf.Ticker(sym)
                hist = ticker.history(period="2d")
                if hist is not None and not hist.empty and len(hist) >= 1:
                    current_price = float(hist['Close'].iloc[-1])
                    if len(hist) >= 2:
                        previous_price = float(hist['Close'].iloc[-2])
                    else:
                        previous_price = current_price
                    change = current_price - previous_price
                    change_percent = (change / previous_price * 100) if previous_price else 0.0
                    
                    markets.append({
                        'name': name,
                        'symbol': sym,
                        'price': current_price,
                        'change': change,
                        'change_percent': change_percent
                    })
            except Exception as e:
                logger.warning(f"Index fetch failed for {name} ({sym}): {e}")
        
        # Cache for 5 minutes
        await cache_service.set(cache_key, markets, 300)
        
        return markets
    except Exception as e:
        logger.error(f"Global markets fetch failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch global markets"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
