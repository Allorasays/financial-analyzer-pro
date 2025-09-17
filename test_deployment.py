#!/usr/bin/env python3
"""
Test script to verify the Financial Analyzer Pro deployment
"""

import sys
import subprocess
import os

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import streamlit as st
        print("✅ streamlit imported successfully")
    except ImportError as e:
        print(f"❌ streamlit import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
    except ImportError as e:
        print(f"❌ pandas import failed: {e}")
        return False
    
    try:
        import plotly.graph_objects as go
        print("✅ plotly imported successfully")
    except ImportError as e:
        print(f"❌ plotly import failed: {e}")
        return False
    
    try:
        import yfinance as yf
        print("✅ yfinance imported successfully")
    except ImportError as e:
        print(f"❌ yfinance import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ numpy imported successfully")
    except ImportError as e:
        print(f"❌ numpy import failed: {e}")
        return False
    
    return True

def test_app_file():
    """Test if the app file exists and is valid"""
    print("Testing app file...")
    
    app_files = ['app_minimal_deploy.py', 'app.py']
    app_file = None
    
    for file in app_files:
        if os.path.exists(file):
            app_file = file
            break
    
    if not app_file:
        print("❌ No app file found")
        return False
    
    print(f"✅ Found app file: {app_file}")
    
    # Test if the file can be imported
    try:
        with open(app_file, 'r') as f:
            content = f.read()
            compile(content, app_file, 'exec')
        print("✅ App file syntax is valid")
        return True
    except SyntaxError as e:
        print(f"❌ App file has syntax errors: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading app file: {e}")
        return False

def test_streamlit_command():
    """Test if streamlit command works"""
    print("Testing streamlit command...")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "streamlit", "--version"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print(f"✅ Streamlit version: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ Streamlit command failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ Streamlit command timed out")
        return False
    except Exception as e:
        print(f"❌ Error testing streamlit: {e}")
        return False

def main():
    print("🧪 Testing Financial Analyzer Pro deployment...")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("App File Test", test_app_file),
        ("Streamlit Command Test", test_streamlit_command)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}")
        print("-" * 30)
        if test_func():
            print(f"✅ {test_name} PASSED")
            passed += 1
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Deployment should work.")
        return 0
    else:
        print("⚠️ Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())






