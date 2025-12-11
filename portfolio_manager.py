#!/usr/bin/env python3
"""
Portfolio Management System for Financial Analyzer Pro
Day 4 Implementation: Real Portfolio Tracking
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import hashlib
from typing import Dict, List, Optional, Tuple
import streamlit as st
import plotly.graph_objects as go

class PortfolioManager:
    def __init__(self, db_path="portfolio.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize portfolio database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Portfolio table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolios (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                is_active BOOLEAN DEFAULT 1
            )
        ''')
        
        # Positions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS positions (
                id TEXT PRIMARY KEY,
                portfolio_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                quantity REAL NOT NULL,
                purchase_price REAL NOT NULL,
                purchase_date TEXT NOT NULL,
                current_price REAL,
                last_updated TEXT,
                notes TEXT,
                FOREIGN KEY (portfolio_id) REFERENCES portfolios (id)
            )
        ''')
        
        # Transactions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                id TEXT PRIMARY KEY,
                portfolio_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                transaction_type TEXT NOT NULL, -- 'BUY', 'SELL'
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                transaction_date TEXT NOT NULL,
                fees REAL DEFAULT 0,
                notes TEXT,
                FOREIGN KEY (portfolio_id) REFERENCES portfolios (id)
            )
        ''')
        
        # Portfolio performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_performance (
                id TEXT PRIMARY KEY,
                portfolio_id TEXT NOT NULL,
                date TEXT NOT NULL,
                total_value REAL NOT NULL,
                total_cost REAL NOT NULL,
                total_pnl REAL NOT NULL,
                total_pnl_percent REAL NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (portfolio_id) REFERENCES portfolios (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_portfolio(self, name: str, description: str = "") -> str:
        """Create a new portfolio"""
        portfolio_id = hashlib.md5(f"{name}_{datetime.now()}".encode()).hexdigest()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO portfolios (id, name, description, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (portfolio_id, name, description, datetime.now().isoformat(), datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        
        return portfolio_id
    
    def get_portfolios(self) -> List[Dict]:
        """Get all active portfolios"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, name, description, created_at, updated_at
            FROM portfolios 
            WHERE is_active = 1
            ORDER BY created_at DESC
        ''')
        
        portfolios = []
        for row in cursor.fetchall():
            portfolios.append({
                'id': row[0],
                'name': row[1],
                'description': row[2],
                'created_at': row[3],
                'updated_at': row[4]
            })
        
        conn.close()
        return portfolios
    
    def add_position(self, portfolio_id: str, symbol: str, quantity: float, 
                    purchase_price: float, purchase_date: str, notes: str = "") -> str:
        """Add a new position to portfolio"""
        position_id = hashlib.md5(f"{portfolio_id}_{symbol}_{datetime.now()}".encode()).hexdigest()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Add position
        cursor.execute('''
            INSERT INTO positions (id, portfolio_id, symbol, quantity, purchase_price, 
                                purchase_date, notes, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (position_id, portfolio_id, symbol, quantity, purchase_price, 
              purchase_date, notes, datetime.now().isoformat()))
        
        # Add transaction record
        transaction_id = hashlib.md5(f"{position_id}_buy_{datetime.now()}".encode()).hexdigest()
        cursor.execute('''
            INSERT INTO transactions (id, portfolio_id, symbol, transaction_type, 
                                   quantity, price, transaction_date, notes)
            VALUES (?, ?, ?, 'BUY', ?, ?, ?, ?)
        ''', (transaction_id, portfolio_id, symbol, quantity, purchase_price, 
              purchase_date, f"Initial purchase: {notes}"))
        
        # Update portfolio timestamp
        cursor.execute('''
            UPDATE portfolios SET updated_at = ? WHERE id = ?
        ''', (datetime.now().isoformat(), portfolio_id))
        
        conn.commit()
        conn.close()
        
        return position_id
    
    def remove_position(self, portfolio_id: str, symbol: str, quantity: float, 
                       sell_price: float, sell_date: str, notes: str = "") -> str:
        """Remove/sell position from portfolio"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get current position
        cursor.execute('''
            SELECT id, quantity FROM positions 
            WHERE portfolio_id = ? AND symbol = ?
        ''', (portfolio_id, symbol))
        
        position = cursor.fetchone()
        if not position:
            conn.close()
            raise ValueError(f"No position found for {symbol}")
        
        position_id, current_quantity = position
        
        if quantity > current_quantity:
            conn.close()
            raise ValueError(f"Cannot sell {quantity} shares, only {current_quantity} available")
        
        # Update position quantity
        new_quantity = current_quantity - quantity
        if new_quantity == 0:
            # Remove position completely
            cursor.execute('DELETE FROM positions WHERE id = ?', (position_id,))
        else:
            # Update quantity
            cursor.execute('''
                UPDATE positions SET quantity = ?, last_updated = ?
                WHERE id = ?
            ''', (new_quantity, datetime.now().isoformat(), position_id))
        
        # Add sell transaction
        transaction_id = hashlib.md5(f"{position_id}_sell_{datetime.now()}".encode()).hexdigest()
        cursor.execute('''
            INSERT INTO transactions (id, portfolio_id, symbol, transaction_type, 
                                   quantity, price, transaction_date, notes)
            VALUES (?, ?, ?, 'SELL', ?, ?, ?, ?)
        ''', (transaction_id, portfolio_id, symbol, quantity, sell_price, 
              sell_date, f"Sell transaction: {notes}"))
        
        # Update portfolio timestamp
        cursor.execute('''
            UPDATE portfolios SET updated_at = ? WHERE id = ?
        ''', (datetime.now().isoformat(), portfolio_id))
        
        conn.commit()
        conn.close()
        
        return transaction_id
    
    def get_positions(self, portfolio_id: str) -> List[Dict]:
        """Get all positions for a portfolio"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT symbol, quantity, purchase_price, purchase_date, 
                   current_price, last_updated, notes
            FROM positions 
            WHERE portfolio_id = ?
            ORDER BY symbol
        ''', (portfolio_id,))
        
        positions = []
        for row in cursor.fetchall():
            positions.append({
                'symbol': row[0],
                'quantity': row[1],
                'purchase_price': row[2],
                'purchase_date': row[3],
                'current_price': row[4],
                'last_updated': row[5],
                'notes': row[6]
            })
        
        conn.close()
        return positions
    
    def get_transactions(self, portfolio_id: str) -> List[Dict]:
        """Get all transactions for a portfolio"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT symbol, transaction_type, quantity, price, transaction_date, 
                   fees, notes
            FROM transactions 
            WHERE portfolio_id = ?
            ORDER BY transaction_date DESC
        ''', (portfolio_id,))
        
        transactions = []
        for row in cursor.fetchall():
            transactions.append({
                'symbol': row[0],
                'type': row[1],
                'quantity': row[2],
                'price': row[3],
                'date': row[4],
                'fees': row[5],
                'notes': row[6]
            })
        
        conn.close()
        return transactions
    
    def calculate_portfolio_metrics(self, portfolio_id: str, current_prices: Dict[str, float]) -> Dict:
        """Calculate comprehensive portfolio metrics"""
        positions = self.get_positions(portfolio_id)
        
        if not positions:
            return {
                'total_value': 0,
                'total_cost': 0,
                'total_pnl': 0,
                'total_pnl_percent': 0,
                'positions_count': 0,
                'positions': []
            }
        
        total_cost = 0
        total_value = 0
        position_metrics = []
        
        for pos in positions:
            symbol = pos['symbol']
            quantity = pos['quantity']
            purchase_price = pos['purchase_price']
            current_price = current_prices.get(symbol, purchase_price)
            
            cost_basis = quantity * purchase_price
            current_value = quantity * current_price
            pnl = current_value - cost_basis
            pnl_percent = (pnl / cost_basis * 100) if cost_basis > 0 else 0
            
            total_cost += cost_basis
            total_value += current_value
            
            position_metrics.append({
                'symbol': symbol,
                'quantity': quantity,
                'purchase_price': purchase_price,
                'current_price': current_price,
                'cost_basis': cost_basis,
                'current_value': current_value,
                'pnl': pnl,
                'pnl_percent': pnl_percent,
                'weight': 0  # Will be calculated after total_value is known
            })
        
        # Calculate weights
        for pos in position_metrics:
            pos['weight'] = (pos['current_value'] / total_value * 100) if total_value > 0 else 0
        
        total_pnl = total_value - total_cost
        total_pnl_percent = (total_pnl / total_cost * 100) if total_cost > 0 else 0
        
        return {
            'total_value': total_value,
            'total_cost': total_cost,
            'total_pnl': total_pnl,
            'total_pnl_percent': total_pnl_percent,
            'positions_count': len(positions),
            'positions': position_metrics
        }
    
    def update_position_prices(self, portfolio_id: str, current_prices: Dict[str, float]):
        """Update current prices for all positions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for symbol, price in current_prices.items():
            cursor.execute('''
                UPDATE positions 
                SET current_price = ?, last_updated = ?
                WHERE portfolio_id = ? AND symbol = ?
            ''', (price, datetime.now().isoformat(), portfolio_id, symbol))
        
        conn.commit()
        conn.close()
    
    def save_portfolio_performance(self, portfolio_id: str, metrics: Dict):
        """Save daily portfolio performance snapshot"""
        performance_id = hashlib.md5(f"{portfolio_id}_{datetime.now().date()}".encode()).hexdigest()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO portfolio_performance 
            (id, portfolio_id, date, total_value, total_cost, total_pnl, total_pnl_percent, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (performance_id, portfolio_id, datetime.now().date().isoformat(),
              metrics['total_value'], metrics['total_cost'], metrics['total_pnl'],
              metrics['total_pnl_percent'], datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def get_portfolio_performance_history(self, portfolio_id: str, days: int = 30) -> List[Dict]:
        """Get portfolio performance history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        start_date = (datetime.now() - timedelta(days=days)).date().isoformat()
        
        cursor.execute('''
            SELECT date, total_value, total_cost, total_pnl, total_pnl_percent
            FROM portfolio_performance 
            WHERE portfolio_id = ? AND date >= ?
            ORDER BY date ASC
        ''', (portfolio_id, start_date))
        
        history = []
        for row in cursor.fetchall():
            history.append({
                'date': row[0],
                'total_value': row[1],
                'total_cost': row[2],
                'total_pnl': row[3],
                'total_pnl_percent': row[4]
            })
        
        conn.close()
        return history
    
    def delete_portfolio(self, portfolio_id: str):
        """Delete a portfolio and all associated data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Delete all related data
        cursor.execute('DELETE FROM positions WHERE portfolio_id = ?', (portfolio_id,))
        cursor.execute('DELETE FROM transactions WHERE portfolio_id = ?', (portfolio_id,))
        cursor.execute('DELETE FROM portfolio_performance WHERE portfolio_id = ?', (portfolio_id,))
        cursor.execute('DELETE FROM portfolios WHERE id = ?', (portfolio_id,))
        
        conn.commit()
        conn.close()

# Portfolio Analysis Functions
def calculate_portfolio_risk_metrics(positions: List[Dict]) -> Dict:
    """Calculate portfolio risk metrics"""
    if not positions:
        return {}
    
    # Calculate position weights
    total_value = sum(pos['current_value'] for pos in positions)
    weights = [pos['current_value'] / total_value for pos in positions] if total_value > 0 else []
    
    # Calculate weighted average metrics
    weighted_volatility = sum(w * 0.2 for w in weights)  # Simplified volatility
    portfolio_beta = sum(w * 1.0 for w in weights)  # Simplified beta
    
    return {
        'portfolio_volatility': weighted_volatility,
        'portfolio_beta': portfolio_beta,
        'concentration_risk': max(weights) if weights else 0,
        'diversification_ratio': len(positions) / 10.0  # Simplified diversification
    }

def generate_portfolio_report(portfolio_id: str, portfolio_manager: PortfolioManager, 
                            current_prices: Dict[str, float]) -> Dict:
    """Generate comprehensive portfolio report"""
    metrics = portfolio_manager.calculate_portfolio_metrics(portfolio_id, current_prices)
    positions = metrics['positions']
    risk_metrics = calculate_portfolio_risk_metrics(positions)
    performance_history = portfolio_manager.get_portfolio_performance_history(portfolio_id, 30)
    
    return {
        'metrics': metrics,
        'risk_metrics': risk_metrics,
        'performance_history': performance_history,
        'generated_at': datetime.now().isoformat()
    }

class EnhancedPortfolioManager:
    """Compatibility layer used by enhanced apps. Manages a per-user default portfolio
    and provides summary plus simple chart rendering utilities.
    """
    def __init__(self, db_path: str = 'financial_analyzer.db'):
        # Mapping of user_id to a portfolio_id in the PortfolioManager database
        self.user_db_path = db_path
        # Use a dedicated DB for portfolio storage (existing schema)
        self.portfolio_manager = PortfolioManager('portfolio.db')
        self._init_user_mapping_table()

    def _init_user_mapping_table(self) -> None:
        try:
            conn = sqlite3.connect(self.user_db_path)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_portfolios (
                    user_id INTEGER PRIMARY KEY,
                    portfolio_id TEXT
                )
            ''')
            conn.commit()
            conn.close()
        except Exception as e:
            st.error(f"User portfolio mapping init error: {str(e)}")

    def _get_or_create_user_portfolio(self, user_id: int) -> str:
        try:
            conn = sqlite3.connect(self.user_db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT portfolio_id FROM user_portfolios WHERE user_id = ?', (user_id,))
            row = cursor.fetchone()
            if row and row[0]:
                conn.close()
                return row[0]
            # Create a default portfolio for the user
            name = f"User {user_id} Portfolio"
            portfolio_id = self.portfolio_manager.create_portfolio(name, "Default portfolio")
            cursor.execute('INSERT OR REPLACE INTO user_portfolios (user_id, portfolio_id) VALUES (?, ?)', (user_id, portfolio_id))
            conn.commit()
            conn.close()
            return portfolio_id
        except Exception as e:
            st.error(f"Error ensuring user portfolio: {str(e)}")
            # Fallback to a transient portfolio if mapping fails
            return self.portfolio_manager.create_portfolio(f"User {user_id} Portfolio", "Default portfolio")

    def add_position(self, user_id: int, symbol: str, shares: float, price: float,
                     date: datetime, transaction_type: str = 'BUY', fees: float = 0.0,
                     notes: str = "") -> bool:
        try:
            portfolio_id = self._get_or_create_user_portfolio(user_id)
            if transaction_type.upper() == 'SELL':
                self.portfolio_manager.remove_position(
                    portfolio_id, symbol, shares, price, date.isoformat(), notes
                )
            else:
                self.portfolio_manager.add_position(
                    portfolio_id, symbol, shares, price, date.isoformat(), notes
                )
            return True
        except Exception as e:
            st.error(f"Error adding position: {str(e)}")
            return False

    def get_portfolio_summary(self, user_id: int) -> Dict:
        try:
            portfolio_id = self._get_or_create_user_portfolio(user_id)
            # Get current prices for symbols in portfolio
            positions = self.portfolio_manager.get_positions(portfolio_id)
            symbols = [p['symbol'] for p in positions]
            current_prices = {s: self._fetch_price_with_fallback(s) for s in symbols}
            # Update stored prices
            self.portfolio_manager.update_position_prices(portfolio_id, current_prices)
            metrics = self.portfolio_manager.calculate_portfolio_metrics(portfolio_id, current_prices)
            # Augment with simple diversification metrics
            diversification = {}
            if metrics.get('positions'):
                weights = [pos['weight'] for pos in metrics['positions']]
                diversification = {
                    'num_positions': len(weights),
                    'max_position_weight': max(weights) if weights else 0,
                    'concentration_index': float(np.sqrt(np.sum(np.square(np.array(weights) / 100.0)))) if weights else 0,
                }
            return {
                'portfolio_id': portfolio_id,
                'positions': metrics.get('positions', []),
                'total_value': metrics.get('total_value', 0),
                'total_cost': metrics.get('total_cost', 0),
                'total_gain_loss': metrics.get('total_pnl', 0),
                'total_gain_loss_pct': metrics.get('total_pnl_percent', 0),
                'diversification': diversification,
                'performance_metrics': {},
            }
        except Exception as e:
            st.error(f"Error building portfolio summary: {str(e)}")
            return {}

    def create_portfolio_charts(self, portfolio_summary: Dict) -> None:
        try:
            positions = portfolio_summary.get('positions', [])
            if not positions:
                st.info("No positions to chart.")
                return
            # Allocation pie chart
            labels = [p['symbol'] for p in positions]
            values = [max(p.get('current_value', 0), 0) for p in positions]
            if sum(values) <= 0:
                st.info("No position value to chart.")
                return
            fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
            fig.update_layout(title_text="Portfolio Allocation")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating portfolio charts: {str(e)}")

    def _fetch_price_with_fallback(self, symbol: str) -> float:
        try:
            import yfinance as yf
            hist = yf.Ticker(symbol).history(period="1d")
            if hist is not None and not hist.empty:
                return float(hist['Close'].iloc[-1])
        except Exception:
            pass
        # Fallback demo price
        base = 100 + (hash(symbol) % 1000)
        np.random.seed(hash(symbol) % 2**32)
        return float(base * (1 + np.random.normal(0, 0.01)))