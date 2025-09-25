#!/usr/bin/env python3
"""
Configuration settings for the FastAPI backend
"""

import os
from typing import Optional
from pydantic import BaseSettings

class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql://financial_user:secure_password@localhost:5432/financial_analyzer"
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    
    # API
    api_title: str = "Financial Analyzer Pro API"
    api_version: str = "1.0.0"
    debug: bool = True
    
    # Security
    secret_key: str = "your-secret-key-change-this-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # External APIs
    yahoo_finance_api_key: Optional[str] = None
    alpha_vantage_api_key: Optional[str] = None
    
    # WebSocket settings
    websocket_ping_interval: int = 30
    websocket_ping_timeout: int = 10
    
    # Cache settings
    market_data_cache_ttl: int = 300  # 5 minutes
    portfolio_cache_ttl: int = 120    # 2 minutes
    user_cache_ttl: int = 3600        # 1 hour
    
    class Config:
        env_file = ".env"

settings = Settings()
