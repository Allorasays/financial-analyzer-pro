import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import sqlite3
from typing import Dict, List, Optional, Tuple
import json

class EnhancedPortfolioManager:
    def __init__(self, db_path: str = 'financial_analyzer.db'):
        self.db_path = db_path
        self.init_portfolio_tables()
    
    def init_portfolio_tables(self):
        """Initialize enhanced portfolio database tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Enhanced portfolios table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolios (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    symbol TEXT,
                    shares REAL,
                    purchase_price REAL,
                    purchase_date DATE,
                    transaction_type TEXT DEFAULT 'BUY',
                    fees REAL DEFAULT 0.0,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # Portfolio performance history
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolio_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    date DATE,
                    total_value REAL,
                    total_cost REAL,
                    total_gain_loss REAL,
                    total_gain_loss_pct REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # Portfolio alerts
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolio_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    symbol TEXT,
                    alert_type TEXT,
                    target_value REAL,
                    current_value REAL,
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # Portfolio goals
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolio_goals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    goal_name TEXT,
                    target_value REAL,
                    current_value REAL,
                    target_date DATE,
                    is_achieved BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            conn.commit()
            conn.close()
        except Exception as e:
            st.error(f"Database initialization error: {str(e)}")
    
    def add_position(self, user_id: int, symbol: str, shares: float, purchase_price: float, 
                    purchase_date: str, transaction_type: str = 'BUY', fees: float = 0.0, notes: str = '') -> bool:
        """Add a new position to portfolio"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO portfolios (user_id, symbol, shares, purchase_price, purchase_date, 
                                     transaction_type, fees, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, symbol, shares, purchase_price, purchase_date, transaction_type, fees, notes))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            st.error(f"Error adding position: {str(e)}")
            return False
    
    def get_portfolio_positions(self, user_id: int) -> List[Dict]:
        """Get all portfolio positions for a user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, symbol, shares, purchase_price, purchase_date, transaction_type, 
                       fees, notes, created_at
                FROM portfolios WHERE user_id = ? ORDER BY created_at DESC
            ''', (user_id,))
            
            positions = cursor.fetchall()
            conn.close()
            
            return [
                {
                    'id': pos[0],
                    'symbol': pos[1],
                    'shares': pos[2],
                    'purchase_price': pos[3],
                    'purchase_date': pos[4],
                    'transaction_type': pos[5],
                    'fees': pos[6],
                    'notes': pos[7],
                    'created_at': pos[8]
                }
                for pos in positions
            ]
        except Exception as e:
            st.error(f"Error fetching portfolio: {str(e)}")
            return []
    
    def get_portfolio_summary(self, user_id: int) -> Dict:
        """Get comprehensive portfolio summary"""
        try:
            positions = self.get_portfolio_positions(user_id)
            
            if not positions:
                return {
                    'total_value': 0,
                    'total_cost': 0,
                    'total_gain_loss': 0,
                    'total_gain_loss_pct': 0,
                    'positions': [],
                    'diversification': {},
                    'performance_metrics': {}
                }
            
            total_value = 0
            total_cost = 0
            position_data = []
            
            # Calculate current values
            for pos in positions:
                symbol = pos['symbol']
                shares = pos['shares']
                purchase_price = pos['purchase_price']
                transaction_type = pos['transaction_type']
                
                # Get current price
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="1d")
                    current_price = hist['Close'].iloc[-1] if not hist.empty else purchase_price
                except:
                    current_price = purchase_price
                
                # Calculate values
                cost_basis = shares * purchase_price
                current_value = shares * current_price
                gain_loss = current_value - cost_basis
                gain_loss_pct = (gain_loss / cost_basis) * 100 if cost_basis > 0 else 0
                
                # Adjust for transaction type
                if transaction_type == 'SELL':
                    current_value = -current_value
                    cost_basis = -cost_basis
                    gain_loss = -gain_loss
                
                position_data.append({
                    'symbol': symbol,
                    'shares': shares,
                    'purchase_price': purchase_price,
                    'current_price': current_price,
                    'cost_basis': cost_basis,
                    'current_value': current_value,
                    'gain_loss': gain_loss,
                    'gain_loss_pct': gain_loss_pct,
                    'transaction_type': transaction_type,
                    'notes': pos['notes']
                })
                
                total_value += current_value
                total_cost += cost_basis
            
            total_gain_loss = total_value - total_cost
            total_gain_loss_pct = (total_gain_loss / total_cost) * 100 if total_cost > 0 else 0
            
            # Calculate diversification
            diversification = self.calculate_diversification(position_data)
            
            # Calculate performance metrics
            performance_metrics = self.calculate_performance_metrics(user_id, total_value, total_cost)
            
            return {
                'total_value': total_value,
                'total_cost': total_cost,
                'total_gain_loss': total_gain_loss,
                'total_gain_loss_pct': total_gain_loss_pct,
                'positions': position_data,
                'diversification': diversification,
                'performance_metrics': performance_metrics
            }
            
        except Exception as e:
            st.error(f"Error calculating portfolio summary: {str(e)}")
            return {}
    
    def calculate_diversification(self, positions: List[Dict]) -> Dict:
        """Calculate portfolio diversification metrics"""
        try:
            if not positions:
                return {}
            
            # Calculate sector diversification (simplified)
            total_value = sum(abs(pos['current_value']) for pos in positions)
            
            # Group by symbol for now (in real implementation, would group by sector)
            symbol_weights = {}
            for pos in positions:
                symbol = pos['symbol']
                weight = abs(pos['current_value']) / total_value if total_value > 0 else 0
                symbol_weights[symbol] = symbol_weights.get(symbol, 0) + weight
            
            # Calculate concentration metrics
            weights = list(symbol_weights.values())
            max_weight = max(weights) if weights else 0
            num_positions = len(symbol_weights)
            
            # Herfindahl-Hirschman Index (concentration measure)
            hhi = sum(w**2 for w in weights)
            
            return {
                'num_positions': num_positions,
                'max_position_weight': max_weight * 100,
                'concentration_index': hhi,
                'symbol_weights': symbol_weights
            }
            
        except Exception as e:
            st.error(f"Error calculating diversification: {str(e)}")
            return {}
    
    def calculate_performance_metrics(self, user_id: int, current_value: float, current_cost: float) -> Dict:
        """Calculate portfolio performance metrics"""
        try:
            # Get historical performance data
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT date, total_value, total_cost, total_gain_loss_pct
                FROM portfolio_performance 
                WHERE user_id = ? 
                ORDER BY date DESC 
                LIMIT 30
            ''', (user_id,))
            
            history = cursor.fetchall()
            conn.close()
            
            if not history:
                return {
                    'daily_return': 0,
                    'weekly_return': 0,
                    'monthly_return': 0,
                    'volatility': 0,
                    'sharpe_ratio': 0,
                    'max_drawdown': 0
                }
            
            # Calculate returns
            values = [row[1] for row in history]
            returns = []
            
            for i in range(1, len(values)):
                if values[i-1] != 0:
                    daily_return = (values[i] - values[i-1]) / values[i-1]
                    returns.append(daily_return)
            
            if not returns:
                return {
                    'daily_return': 0,
                    'weekly_return': 0,
                    'monthly_return': 0,
                    'volatility': 0,
                    'sharpe_ratio': 0,
                    'max_drawdown': 0
                }
            
            # Calculate metrics
            daily_return = returns[-1] if returns else 0
            weekly_return = sum(returns[-7:]) if len(returns) >= 7 else sum(returns)
            monthly_return = sum(returns[-30:]) if len(returns) >= 30 else sum(returns)
            
            # Volatility (standard deviation of returns)
            volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
            
            # Sharpe ratio (simplified)
            risk_free_rate = 0.04  # 4% annual
            excess_return = np.mean(returns) * 252 - risk_free_rate
            sharpe_ratio = excess_return / volatility if volatility > 0 else 0
            
            # Maximum drawdown
            peak = values[0]
            max_drawdown = 0
            for value in values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
            return {
                'daily_return': daily_return * 100,
                'weekly_return': weekly_return * 100,
                'monthly_return': monthly_return * 100,
                'volatility': volatility * 100,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown * 100
            }
            
        except Exception as e:
            st.error(f"Error calculating performance metrics: {str(e)}")
            return {}
    
    def create_portfolio_charts(self, portfolio_data: Dict) -> None:
        """Create comprehensive portfolio visualization charts"""
        if not portfolio_data or not portfolio_data.get('positions'):
            st.warning("No portfolio data available for visualization")
            return
        
        positions = portfolio_data['positions']
        
        # 1. Portfolio Allocation Pie Chart
        st.subheader("📊 Portfolio Allocation")
        
        # Filter out zero values and group by symbol
        symbol_values = {}
        for pos in positions:
            symbol = pos['symbol']
            value = abs(pos['current_value'])
            if value > 0:
                symbol_values[symbol] = symbol_values.get(symbol, 0) + value
        
        if symbol_values:
            fig_pie = px.pie(
                values=list(symbol_values.values()),
                names=list(symbol_values.keys()),
                title="Portfolio Allocation by Symbol"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # 2. Performance Chart
        st.subheader("📈 Portfolio Performance")
        
        # Create performance data
        performance_data = []
        cumulative_value = 0
        cumulative_cost = 0
        
        for pos in positions:
            cumulative_value += pos['current_value']
            cumulative_cost += pos['cost_basis']
            performance_data.append({
                'Symbol': pos['symbol'],
                'Current Value': cumulative_value,
                'Cost Basis': cumulative_cost,
                'Gain/Loss': cumulative_value - cumulative_cost
            })
        
        if performance_data:
            df_perf = pd.DataFrame(performance_data)
            
            fig_perf = go.Figure()
            fig_perf.add_trace(go.Scatter(
                x=df_perf['Symbol'],
                y=df_perf['Current Value'],
                mode='lines+markers',
                name='Current Value',
                line=dict(color='blue', width=2)
            ))
            fig_perf.add_trace(go.Scatter(
                x=df_perf['Symbol'],
                y=df_perf['Cost Basis'],
                mode='lines+markers',
                name='Cost Basis',
                line=dict(color='red', width=2)
            ))
            
            fig_perf.update_layout(
                title="Portfolio Value vs Cost Basis",
                xaxis_title="Position",
                yaxis_title="Value ($)",
                height=400
            )
            
            st.plotly_chart(fig_perf, use_container_width=True)
        
        # 3. Individual Position Performance
        st.subheader("📋 Position Performance")
        
        position_data = []
        for pos in positions:
            position_data.append({
                'Symbol': pos['symbol'],
                'Shares': pos['shares'],
                'Purchase Price': f"${pos['purchase_price']:.2f}",
                'Current Price': f"${pos['current_price']:.2f}",
                'Current Value': f"${pos['current_value']:,.2f}",
                'Cost Basis': f"${pos['cost_basis']:,.2f}",
                'Gain/Loss': f"${pos['gain_loss']:+,.2f}",
                'Gain/Loss %': f"{pos['gain_loss_pct']:+.2f}%",
                'Type': pos['transaction_type']
            })
        
        df_positions = pd.DataFrame(position_data)
        st.dataframe(df_positions, use_container_width=True)
    
    def add_portfolio_alert(self, user_id: int, symbol: str, alert_type: str, target_value: float) -> bool:
        """Add a portfolio alert"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO portfolio_alerts (user_id, symbol, alert_type, target_value, current_value)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, symbol, alert_type, target_value, 0.0))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            st.error(f"Error adding alert: {str(e)}")
            return False
    
    def get_portfolio_alerts(self, user_id: int) -> List[Dict]:
        """Get portfolio alerts for a user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, symbol, alert_type, target_value, current_value, is_active, created_at
                FROM portfolio_alerts WHERE user_id = ? AND is_active = 1
                ORDER BY created_at DESC
            ''', (user_id,))
            
            alerts = cursor.fetchall()
            conn.close()
            
            return [
                {
                    'id': alert[0],
                    'symbol': alert[1],
                    'alert_type': alert[2],
                    'target_value': alert[3],
                    'current_value': alert[4],
                    'is_active': alert[5],
                    'created_at': alert[6]
                }
                for alert in alerts
            ]
        except Exception as e:
            st.error(f"Error fetching alerts: {str(e)}")
            return []
    
    def check_alerts(self, user_id: int) -> List[Dict]:
        """Check and return triggered alerts"""
        try:
            alerts = self.get_portfolio_alerts(user_id)
            triggered_alerts = []
            
            for alert in alerts:
                symbol = alert['symbol']
                target_value = alert['target_value']
                alert_type = alert['alert_type']
                
                # Get current price
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="1d")
                    current_price = hist['Close'].iloc[-1] if not hist.empty else 0
                except:
                    current_price = 0
                
                # Check if alert should trigger
                should_trigger = False
                if alert_type == 'PRICE_ABOVE' and current_price > target_value:
                    should_trigger = True
                elif alert_type == 'PRICE_BELOW' and current_price < target_value:
                    should_trigger = True
                elif alert_type == 'GAIN_ABOVE' and current_price > target_value:
                    should_trigger = True
                elif alert_type == 'LOSS_BELOW' and current_price < target_value:
                    should_trigger = True
                
                if should_trigger:
                    triggered_alerts.append({
                        'symbol': symbol,
                        'alert_type': alert_type,
                        'target_value': target_value,
                        'current_value': current_price,
                        'message': f"{symbol} {alert_type.replace('_', ' ').lower()} {current_price:.2f} (target: {target_value:.2f})"
                    })
            
            return triggered_alerts
            
        except Exception as e:
            st.error(f"Error checking alerts: {str(e)}")
            return []





