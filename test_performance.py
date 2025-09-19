#!/usr/bin/env python3
"""
Performance test script for Financial Analyzer Pro Enhanced
"""

import time
import sys
import os
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing imports...")
    
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
        print("✅ Scikit-learn imported successfully")
    except ImportError as e:
        print(f"❌ Scikit-learn import failed: {e}")
        return False
    
    try:
        import sqlite3
        print("✅ SQLite3 imported successfully")
    except ImportError as e:
        print(f"❌ SQLite3 import failed: {e}")
        return False
    
    return True

def test_smart_cache():
    """Test the SmartCache functionality"""
    print("\n🧪 Testing SmartCache...")
    
    try:
        from app_enhanced_performance import SmartCache
        
        # Create cache instance
        cache = SmartCache(max_size=10, default_ttl=5)
        
        # Test basic operations
        cache.set("test_key", "test_value")
        value = cache.get("test_key")
        
        if value == "test_value":
            print("✅ Cache set/get working")
        else:
            print("❌ Cache set/get failed")
            return False
        
        # Test TTL
        cache.set("ttl_key", "ttl_value", ttl=1)
        time.sleep(2)
        expired_value = cache.get("ttl_key")
        
        if expired_value is None:
            print("✅ TTL expiration working")
        else:
            print("❌ TTL expiration failed")
            return False
        
        # Test cache stats
        stats = cache.get_stats()
        if 'size' in stats and 'hit_rate' in stats:
            print("✅ Cache stats working")
        else:
            print("❌ Cache stats failed")
            return False
        
        # Test cache eviction
        for i in range(15):  # More than max_size
            cache.set(f"key_{i}", f"value_{i}")
        
        stats = cache.get_stats()
        if stats['size'] <= 10:
            print("✅ Cache eviction working")
        else:
            print("❌ Cache eviction failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ SmartCache test failed: {e}")
        return False

def test_prediction_tracker():
    """Test the PredictionTracker functionality"""
    print("\n🧪 Testing PredictionTracker...")
    
    try:
        from app_enhanced_performance import PredictionTracker
        
        # Create tracker instance
        tracker = PredictionTracker("test_predictions.db")
        
        # Test storing prediction
        prediction_data = {
            'predictions': [100.0, 101.0, 102.0],
            'dates': [datetime.now()],
            'model_type': 'Test Model'
        }
        
        prediction_id = tracker.store_prediction("TEST", prediction_data, 1)
        
        if prediction_id:
            print("✅ Prediction storage working")
        else:
            print("❌ Prediction storage failed")
            return False
        
        # Test accuracy metrics calculation
        metrics = tracker.calculate_accuracy_metrics("TEST")
        
        if 'total_predictions' in metrics:
            print("✅ Accuracy metrics working")
        else:
            print("❌ Accuracy metrics failed")
            return False
        
        # Test recent predictions
        recent = tracker.get_recent_predictions("TEST", 5)
        
        if isinstance(recent, list):
            print("✅ Recent predictions working")
        else:
            print("❌ Recent predictions failed")
            return False
        
        # Clean up test database
        if os.path.exists("test_predictions.db"):
            os.remove("test_predictions.db")
        
        return True
        
    except Exception as e:
        print(f"❌ PredictionTracker test failed: {e}")
        return False

def test_error_recovery():
    """Test the ErrorRecovery functionality"""
    print("\n🧪 Testing ErrorRecovery...")
    
    try:
        from app_enhanced_performance import ErrorRecovery
        
        # Test fallback data generation
        fallback_data = ErrorRecovery.fallback_data("AAPL", "1mo")
        
        if not fallback_data.empty and 'Close' in fallback_data.columns:
            print("✅ Fallback data generation working")
        else:
            print("❌ Fallback data generation failed")
            return False
        
        # Test retry decorator
        call_count = 0
        
        @ErrorRecovery.retry_on_failure(max_retries=2, delay=0.1)
        def failing_function():
            nonlocal call_count
            call_count += 1
            raise Exception("Test error")
        
        result = failing_function()
        
        if call_count == 3 and result is None:  # 1 initial + 2 retries
            print("✅ Retry decorator working")
        else:
            print("❌ Retry decorator failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ ErrorRecovery test failed: {e}")
        return False

def test_data_fetching():
    """Test the enhanced data fetching"""
    print("\n🧪 Testing data fetching...")
    
    try:
        from app_enhanced_performance import get_market_data
        
        # Test with a known symbol
        start_time = time.time()
        data = get_market_data("AAPL", "1mo")
        fetch_time = time.time() - start_time
        
        if data is not None and not data.empty:
            print(f"✅ Data fetching working (took {fetch_time:.2f}s)")
        else:
            print("❌ Data fetching failed")
            return False
        
        # Test caching
        start_time = time.time()
        cached_data = get_market_data("AAPL", "1mo")
        cache_time = time.time() - start_time
        
        if cache_time < fetch_time:
            print(f"✅ Caching working (cached: {cache_time:.2f}s vs fresh: {fetch_time:.2f}s)")
        else:
            print("⚠️ Caching may not be working optimally")
        
        return True
        
    except Exception as e:
        print(f"❌ Data fetching test failed: {e}")
        return False

def test_technical_indicators():
    """Test technical indicators calculation"""
    print("\n🧪 Testing technical indicators...")
    
    try:
        from app_enhanced_performance import calculate_technical_indicators
        import pandas as pd
        import numpy as np
        
        # Create test data
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        test_data = pd.DataFrame({
            'Open': np.random.uniform(100, 110, len(dates)),
            'High': np.random.uniform(110, 120, len(dates)),
            'Low': np.random.uniform(90, 100, len(dates)),
            'Close': np.random.uniform(100, 110, len(dates)),
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        
        # Calculate indicators
        enhanced_data = calculate_technical_indicators(test_data)
        
        # Check for enhanced indicators
        enhanced_indicators = ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'BB_Upper', 'BB_Lower', 'Stoch_K', 'Stoch_D']
        found_indicators = [ind for ind in enhanced_indicators if ind in enhanced_data.columns]
        
        if len(found_indicators) >= 5:
            print(f"✅ Technical indicators working ({len(found_indicators)} indicators calculated)")
        else:
            print(f"❌ Technical indicators failed (only {len(found_indicators)} indicators found)")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Technical indicators test failed: {e}")
        return False

def run_performance_benchmark():
    """Run performance benchmark"""
    print("\n🧪 Running performance benchmark...")
    
    try:
        from app_enhanced_performance import get_market_data, calculate_technical_indicators, predict_price_ml
        
        # Benchmark data fetching
        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "META"]
        total_time = 0
        
        for symbol in symbols:
            start_time = time.time()
            data = get_market_data(symbol, "1mo")
            fetch_time = time.time() - start_time
            total_time += fetch_time
            print(f"  {symbol}: {fetch_time:.2f}s")
        
        avg_fetch_time = total_time / len(symbols)
        print(f"✅ Average data fetch time: {avg_fetch_time:.2f}s")
        
        # Benchmark technical indicators
        start_time = time.time()
        data = get_market_data("AAPL", "1mo")
        enhanced_data = calculate_technical_indicators(data)
        indicators_time = time.time() - start_time
        print(f"✅ Technical indicators calculation: {indicators_time:.2f}s")
        
        # Benchmark ML prediction
        start_time = time.time()
        predictions, error = predict_price_ml(enhanced_data, "AAPL", 5)
        ml_time = time.time() - start_time
        print(f"✅ ML prediction: {ml_time:.2f}s")
        
        total_benchmark_time = avg_fetch_time + indicators_time + ml_time
        print(f"✅ Total benchmark time: {total_benchmark_time:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        return False

def main():
    """Run all performance tests"""
    print("🚀 Financial Analyzer Pro Enhanced - Performance Test Suite")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("SmartCache Test", test_smart_cache),
        ("PredictionTracker Test", test_prediction_tracker),
        ("ErrorRecovery Test", test_error_recovery),
        ("Data Fetching Test", test_data_fetching),
        ("Technical Indicators Test", test_technical_indicators),
        ("Performance Benchmark", run_performance_benchmark)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print(f"\n{'='*60}")
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced performance version is ready!")
        return True
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)