#!/usr/bin/env python3
"""
Test script for Day 4 Portfolio Management
This script tests the core functionality without Streamlit to identify issues
"""

import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ Pandas imported successfully")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        print("✅ Plotly imported successfully")
    except ImportError as e:
        print(f"❌ Plotly import failed: {e}")
        return False
    
    try:
        import yfinance as yf
        print("✅ YFinance imported successfully")
    except ImportError as e:
        print(f"❌ YFinance import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error, r2_score
        print("✅ Scikit-learn imported successfully")
    except ImportError as e:
        print(f"❌ Scikit-learn import failed: {e}")
        return False
    
    return True

def test_yfinance():
    """Test yfinance functionality"""
    print("\nTesting yfinance...")
    
    try:
        ticker = yf.Ticker("AAPL")
        data = ticker.history(period="1d", timeout=10)
        if not data.empty:
            print(f"✅ YFinance working - AAPL price: ${data['Close'].iloc[-1]:.2f}")
            return True
        else:
            print("❌ YFinance returned empty data")
            return False
    except Exception as e:
        print(f"❌ YFinance test failed: {e}")
        return False

def test_portfolio_calculations():
    """Test portfolio calculation functions"""
    print("\nTesting portfolio calculations...")
    
    # Mock portfolio data
    positions = [
        {
            'symbol': 'AAPL',
            'shares': 10,
            'cost_basis': 150.0,
            'date_added': '2024-01-01',
            'notes': 'Test position'
        }
    ]
    
    try:
        # Test portfolio metrics calculation
        total_value = 0
        total_cost = 0
        
        for position in positions:
            # Get current price
            ticker = yf.Ticker(position['symbol'])
            data = ticker.history(period="1d", timeout=10)
            if not data.empty:
                current_price = data['Close'].iloc[-1]
            else:
                current_price = position['cost_basis']
            
            position_value = current_price * position['shares']
            position_cost = position['cost_basis'] * position['shares']
            
            total_value += position_value
            total_cost += position_cost
        
        total_pnl = total_value - total_cost
        total_pnl_percent = (total_pnl / total_cost * 100) if total_cost > 0 else 0
        
        print(f"✅ Portfolio calculations working")
        print(f"   Total Value: ${total_value:.2f}")
        print(f"   Total Cost: ${total_cost:.2f}")
        print(f"   Total P&L: ${total_pnl:.2f} ({total_pnl_percent:+.2f}%)")
        
        return True
    except Exception as e:
        print(f"❌ Portfolio calculations failed: {e}")
        return False

def test_plotly():
    """Test plotly functionality"""
    print("\nTesting Plotly...")
    
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        
        # Create a simple chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 4, 2], mode='lines'))
        fig.update_layout(title="Test Chart")
        
        print("✅ Plotly working - chart created successfully")
        return True
    except Exception as e:
        print(f"❌ Plotly test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Day 4 Portfolio Management Components")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    if not imports_ok:
        print("\n❌ Import tests failed. Please install missing dependencies:")
        print("pip install streamlit pandas plotly yfinance numpy scikit-learn")
        return
    
    # Test yfinance
    yfinance_ok = test_yfinance()
    
    # Test portfolio calculations
    portfolio_ok = test_portfolio_calculations()
    
    # Test plotly
    plotly_ok = test_plotly()
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"Imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"YFinance: {'✅ PASS' if yfinance_ok else '❌ FAIL'}")
    print(f"Portfolio: {'✅ PASS' if portfolio_ok else '❌ FAIL'}")
    print(f"Plotly: {'✅ PASS' if plotly_ok else '❌ FAIL'}")
    
    if all([imports_ok, yfinance_ok, portfolio_ok, plotly_ok]):
        print("\n🎉 All tests passed! Day 4 app should work correctly.")
        print("Run: streamlit run app_day4_simple.py")
    else:
        print("\n⚠️ Some tests failed. Please fix the issues before running the app.")

if __name__ == "__main__":
    main()





