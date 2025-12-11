#!/usr/bin/env python3
"""
Simple deployment test for Financial Analyzer Pro
Tests all critical imports and basic functionality
"""

import sys
import os

def test_imports():
    """Test all critical imports"""
    print("🔍 Testing critical imports...")
    
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
        print("✅ Plotly imported successfully")
    except ImportError as e:
        print(f"❌ Plotly import failed: {e}")
        return False
    
    try:
        import yfinance as yf
        print("✅ yfinance imported successfully")
    except ImportError as e:
        print(f"❌ yfinance import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import sqlite3
        print("✅ SQLite3 imported successfully")
    except ImportError as e:
        print(f"❌ SQLite3 import failed: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality"""
    print("\n🔍 Testing basic functionality...")
    
    try:
        import yfinance as yf
        import pandas as pd
        
        # Test yfinance data fetch
        ticker = yf.Ticker("AAPL")
        data = ticker.history(period="1d")
        
        if not data.empty:
            print("✅ yfinance data fetch successful")
        else:
            print("⚠️ yfinance returned empty data (might be market hours)")
        
        # Test pandas operations
        df = pd.DataFrame({'test': [1, 2, 3]})
        if len(df) == 3:
            print("✅ Pandas operations working")
        else:
            print("❌ Pandas operations failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_app_file():
    """Test if app.py can be imported"""
    print("\n🔍 Testing app.py import...")
    
    try:
        # Add current directory to path
        sys.path.insert(0, os.getcwd())
        
        # Try to import the app
        import app
        print("✅ app.py imported successfully")
        
        # Check if main function exists
        if hasattr(app, 'main'):
            print("✅ main() function found")
        else:
            print("⚠️ main() function not found")
        
        return True
        
    except Exception as e:
        print(f"❌ app.py import failed: {e}")
        return False

def test_environment():
    """Test environment variables"""
    print("\n🔍 Testing environment...")
    
    # Check Python version
    python_version = sys.version_info
    print(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check if we're in the right directory
    if os.path.exists("app.py"):
        print("✅ app.py found in current directory")
    else:
        print("❌ app.py not found in current directory")
        return False
    
    if os.path.exists("requirements.txt"):
        print("✅ requirements.txt found")
    else:
        print("❌ requirements.txt not found")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🚀 Financial Analyzer Pro - Deployment Test")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Test environment
    if not test_environment():
        all_tests_passed = False
    
    # Test imports
    if not test_imports():
        all_tests_passed = False
    
    # Test basic functionality
    if not test_basic_functionality():
        all_tests_passed = False
    
    # Test app file
    if not test_app_file():
        all_tests_passed = False
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED! Ready for deployment!")
        print("\n📋 Next steps:")
        print("1. Commit your changes: git add . && git commit -m 'Fix: Render deployment configuration'")
        print("2. Push to GitHub: git push origin main")
        print("3. Deploy on Render using the updated render.yaml")
    else:
        print("❌ Some tests failed. Please fix the issues before deploying.")
        print("\n🔧 Common fixes:")
        print("- Run: pip install -r requirements.txt")
        print("- Check that all files are in the correct directory")
        print("- Verify Python version compatibility")
    
    return all_tests_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

















