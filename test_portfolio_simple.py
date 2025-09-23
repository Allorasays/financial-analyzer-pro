#!/usr/bin/env python3
"""
Simple test script for Portfolio Management System
"""

from portfolio_manager import PortfolioManager
from datetime import datetime, timedelta
import os

def test_portfolio_system():
    """Test the portfolio management system"""
    print("Testing Portfolio Management System...")
    
    # Initialize portfolio manager
    portfolio_manager = PortfolioManager("test_portfolio.db")
    
    try:
        # Test 1: Create portfolio
        print("\n1. Testing portfolio creation...")
        portfolio_id = portfolio_manager.create_portfolio("Test Portfolio", "A test portfolio")
        print(f"Created portfolio: {portfolio_id}")
        
        # Test 2: Add positions
        print("\n2. Testing position addition...")
        position1_id = portfolio_manager.add_position(
            portfolio_id, "AAPL", 10, 150.0, 
            (datetime.now() - timedelta(days=30)).isoformat(), 
            "Initial purchase"
        )
        print(f"Added AAPL position: {position1_id}")
        
        # Test 3: Get positions
        print("\n3. Testing position retrieval...")
        positions = portfolio_manager.get_positions(portfolio_id)
        print(f"Retrieved {len(positions)} positions")
        
        # Test 4: Calculate metrics
        print("\n4. Testing portfolio metrics...")
        current_prices = {"AAPL": 160.0}
        metrics = portfolio_manager.calculate_portfolio_metrics(portfolio_id, current_prices)
        print(f"Portfolio metrics calculated:")
        print(f"   - Total Value: ${metrics['total_value']:,.2f}")
        print(f"   - Total Cost: ${metrics['total_cost']:,.2f}")
        print(f"   - Total P&L: ${metrics['total_pnl']:,.2f}")
        
        print("\nAll portfolio tests passed!")
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
        raise
    
    finally:
        # Cleanup
        if os.path.exists("test_portfolio.db"):
            os.remove("test_portfolio.db")
            print("\nCleaned up test database")

if __name__ == "__main__":
    test_portfolio_system()
