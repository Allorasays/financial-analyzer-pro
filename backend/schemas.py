#!/usr/bin/env python3
"""
Pydantic schemas for request/response validation
"""

from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime, date

# User schemas
class UserBase(BaseModel):
    username: str
    email: EmailStr

class UserCreate(UserBase):
    password: str

class UserResponse(UserBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

# Portfolio schemas
class PortfolioBase(BaseModel):
    name: str
    description: Optional[str] = None

class PortfolioCreate(PortfolioBase):
    pass

class PortfolioResponse(PortfolioBase):
    id: str
    user_id: int
    is_active: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

# Position schemas
class PositionBase(BaseModel):
    symbol: str
    quantity: float
    purchase_price: float
    purchase_date: date
    notes: Optional[str] = None

class PositionCreate(PositionBase):
    pass

class PositionResponse(PositionBase):
    id: str
    portfolio_id: str
    current_price: Optional[float]
    last_updated: Optional[datetime]
    created_at: datetime
    
    class Config:
        from_attributes = True

# Transaction schemas
class TransactionBase(BaseModel):
    symbol: str
    transaction_type: str
    quantity: float
    price: float
    transaction_date: date
    fees: float = 0.0
    notes: Optional[str] = None

class TransactionCreate(TransactionBase):
    pass

class TransactionResponse(TransactionBase):
    id: str
    portfolio_id: str
    created_at: datetime
    
    class Config:
        from_attributes = True

# Market data schemas
class MarketDataResponse(BaseModel):
    symbol: str
    period: str
    data: List[Dict[str, Any]]
    last_updated: str

class MarketOverviewResponse(BaseModel):
    symbol: str
    price: float
    change: float
    change_percent: float
    name: str

class GlobalMarketResponse(BaseModel):
    name: str
    symbol: str
    price: float
    change: float
    change_percent: float

# Portfolio summary schemas
class PositionMetrics(BaseModel):
    symbol: str
    quantity: float
    purchase_price: float
    current_price: float
    cost_basis: float
    current_value: float
    pnl: float
    pnl_percent: float
    weight: float

class PortfolioSummary(BaseModel):
    portfolio_id: str
    positions: List[PositionMetrics]
    total_value: float
    total_cost: float
    total_gain_loss: float
    total_gain_loss_pct: float
    diversification: Dict[str, Any]
    performance_metrics: Dict[str, Any]

# WebSocket message schemas
class WebSocketMessage(BaseModel):
    type: str
    data: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None

class WebSocketResponse(BaseModel):
    type: str
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: float

