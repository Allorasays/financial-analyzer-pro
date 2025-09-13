#!/usr/bin/env python3
"""
Debug script to test deployment locally and identify issues
"""

import sys
import subprocess
import os

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    required_modules = [
        'streamlit',
        'pandas', 
        'plotly',
        'yfinance',
        'numpy',
        'scikit-learn',
        'scipy',
        'requests'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    return len(failed_imports) == 0

def test_app_file():
    """Test if the main app file can be imported"""
    print("\nTesting app file...")
    
    try:
        # Test if we can import the main app
        import app_final_enhanced
        print("✅ app_final_enhanced imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import app_final_enhanced: {e}")
        return False
    except Exception as e:
        print(f"❌ Error importing app_final_enhanced: {e}")
        return False

def test_streamlit_command():
    """Test if streamlit command works"""
    print("\nTesting streamlit command...")
    
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

def test_database_creation():
    """Test if database can be created"""
    print("\nTesting database creation...")
    
    try:
        import sqlite3
        conn = sqlite3.connect('test_financial_analyzer.db')
        cursor = conn.cursor()
        
        # Test basic table creation
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                email TEXT UNIQUE,
                password_hash TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        
        # Clean up test database
        os.remove('test_financial_analyzer.db')
        
        print("✅ Database creation test passed")
        return True
    except Exception as e:
        print(f"❌ Database creation failed: {e}")
        return False

def main():
    print("🔍 Financial Analyzer Pro - Deployment Debug")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("App File Test", test_app_file),
        ("Streamlit Command Test", test_streamlit_command),
        ("Database Test", test_database_creation)
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
        print("🎉 All tests passed! The app should work.")
        return 0
    else:
        print("⚠️ Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

