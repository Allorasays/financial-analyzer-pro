#!/usr/bin/env python3
"""
Test script for Portfolio Management System
Day 4 Implementation Testing
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
        portfolio_id = portfolio_manager.create_portfolio("Test Portfolio", "A test portfolio for development")
        print(f"Created portfolio: {portfolio_id}")
        
        # Test 2: Add positions
        print("\n2. Testing position addition...")
        position1_id = portfolio_manager.add_position(
            portfolio_id, "AAPL", 10, 150.0, 
            (datetime.now() - timedelta(days=30)).isoformat(), 
            "Initial purchase"
        )
        print(f"✅ Added AAPL position: {position1_id}")
        
        position2_id = portfolio_manager.add_position(
            portfolio_id, "MSFT", 5, 300.0, 
            (datetime.now() - timedelta(days=15)).isoformat(), 
            "Tech stock addition"
        )
        print(f"✅ Added MSFT position: {position2_id}")
        
        # Test 3: Get positions
        print("\n3. Testing position retrieval...")
        positions = portfolio_manager.get_positions(portfolio_id)
        print(f"✅ Retrieved {len(positions)} positions")
        for pos in positions:
            print(f"   - {pos['symbol']}: {pos['quantity']} shares @ ${pos['purchase_price']}")
        
        # Test 4: Calculate metrics
        print("\n4. Testing portfolio metrics...")
        current_prices = {"AAPL": 160.0, "MSFT": 320.0}
        metrics = portfolio_manager.calculate_portfolio_metrics(portfolio_id, current_prices)
        print(f"✅ Portfolio metrics calculated:")
        print(f"   - Total Value: ${metrics['total_value']:,.2f}")
        print(f"   - Total Cost: ${metrics['total_cost']:,.2f}")
        print(f"   - Total P&L: ${metrics['total_pnl']:,.2f}")
        print(f"   - P&L %: {metrics['total_pnl_percent']:.2f}%")
        
        # Test 5: Get transactions
        print("\n5. Testing transaction history...")
        transactions = portfolio_manager.get_transactions(portfolio_id)
        print(f"✅ Retrieved {len(transactions)} transactions")
        for trans in transactions:
            print(f"   - {trans['date']}: {trans['type']} {trans['quantity']} {trans['symbol']} @ ${trans['price']}")
        
        # Test 6: Sell position
        print("\n6. Testing position selling...")
        sell_transaction_id = portfolio_manager.remove_position(
            portfolio_id, "AAPL", 3, 165.0, 
            datetime.now().isoformat(), 
            "Partial sale"
        )
        print(f"✅ Sold 3 AAPL shares: {sell_transaction_id}")
        
        # Test 7: Updated metrics
        print("\n7. Testing updated metrics...")
        updated_metrics = portfolio_manager.calculate_portfolio_metrics(portfolio_id, current_prices)
        print(f"✅ Updated portfolio metrics:")
        print(f"   - Total Value: ${updated_metrics['total_value']:,.2f}")
        print(f"   - Total Cost: ${updated_metrics['total_cost']:,.2f}")
        print(f"   - Total P&L: ${updated_metrics['total_pnl']:,.2f}")
        print(f"   - P&L %: {updated_metrics['total_pnl_percent']:.2f}%")
        
        # Test 8: Performance tracking
        print("\n8. Testing performance tracking...")
        portfolio_manager.save_portfolio_performance(portfolio_id, updated_metrics)
        performance_history = portfolio_manager.get_portfolio_performance_history(portfolio_id, 30)
        print(f"✅ Saved performance snapshot, history length: {len(performance_history)}")
        
        # Test 9: Get portfolios
        print("\n9. Testing portfolio listing...")
        portfolios = portfolio_manager.get_portfolios()
        print(f"✅ Retrieved {len(portfolios)} portfolios")
        for portfolio in portfolios:
            print(f"   - {portfolio['name']}: {portfolio['id'][:8]}...")
        
        print("\n🎉 All portfolio tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        raise
    
    finally:
        # Cleanup
        if os.path.exists("test_portfolio.db"):
            os.remove("test_portfolio.db")
            print("\n🧹 Cleaned up test database")

if __name__ == "__main__":
    test_portfolio_system()
