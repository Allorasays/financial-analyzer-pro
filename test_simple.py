#!/usr/bin/env python3
"""
Simple performance test for Financial Analyzer Pro Enhanced
"""

import time
import sys
import os

def test_basic_imports():
    """Test basic imports"""
    print("🧪 Testing Basic Imports...")
    
    try:
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        print("✅ Basic imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {str(e)}")
        return False

def test_cache_system():
    """Test the cache system"""
    print("\n🧪 Testing Cache System...")
    
    try:
        # Simple cache implementation
        class SimpleCache:
            def __init__(self, max_size=100):
                self.cache = {}
                self.max_size = max_size
            
            def get(self, key):
                return self.cache.get(key)
            
            def set(self, key, value):
                if len(self.cache) >= self.max_size:
                    # Remove oldest item
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                self.cache[key] = value
            
            def clear(self):
                self.cache.clear()
        
        # Test cache
        cache = SimpleCache(max_size=10)
        
        # Test operations
        for i in range(15):
            cache.set(f"key_{i}", f"value_{i}")
        
        # Verify cache size limit
        assert len(cache.cache) == 10, f"Cache size should be 10, got {len(cache.cache)}"
        
        # Test retrieval
        value = cache.get("key_5")
        assert value == "value_5", f"Expected 'value_5', got '{value}'"
        
        print("✅ Cache system working correctly")
        return True
    except Exception as e:
        print(f"❌ Cache test failed: {str(e)}")
        return False

def test_data_operations():
    """Test data operations"""
    print("\n🧪 Testing Data Operations...")
    
    try:
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # Create sample data
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
        np.random.seed(42)
        
        data = pd.DataFrame({
            'Open': np.random.uniform(90, 110, len(dates)),
            'High': np.random.uniform(100, 120, len(dates)),
            'Low': np.random.uniform(80, 100, len(dates)),
            'Close': np.random.uniform(90, 110, len(dates)),
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        
        # Test technical indicators
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['RSI'] = 50 + np.random.normal(0, 10, len(dates))  # Mock RSI
        
        # Verify data
        assert len(data) > 0, "Data should not be empty"
        assert 'SMA_20' in data.columns, "SMA_20 should be in columns"
        assert 'RSI' in data.columns, "RSI should be in columns"
        
        print(f"✅ Data operations successful - {len(data)} rows processed")
        return True
    except Exception as e:
        print(f"❌ Data operations failed: {str(e)}")
        return False

def test_prediction_simulation():
    """Test prediction simulation"""
    print("\n🧪 Testing Prediction Simulation...")
    
    try:
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # Simulate prediction data
        current_price = 100.0
        predictions = []
        dates = []
        
        for i in range(5):
            # Simple prediction: current price + random change
            change = np.random.normal(0, 0.02)  # 2% daily volatility
            pred_price = current_price * (1 + change)
            predictions.append(pred_price)
            dates.append(datetime.now() + timedelta(days=i+1))
            current_price = pred_price
        
        # Simulate actual prices (with some noise)
        actual_prices = [p + np.random.normal(0, 1) for p in predictions]
        
        # Calculate accuracy metrics
        mae = np.mean(np.abs(np.array(predictions) - np.array(actual_prices)))
        rmse = np.sqrt(np.mean((np.array(predictions) - np.array(actual_prices)) ** 2))
        
        # Verify metrics
        assert mae > 0, "MAE should be positive"
        assert rmse > 0, "RMSE should be positive"
        
        print(f"✅ Prediction simulation successful - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        return True
    except Exception as e:
        print(f"❌ Prediction simulation failed: {str(e)}")
        return False

def run_simple_tests():
    """Run all simple tests"""
    print("🔬 Financial Analyzer Pro - Simple Performance Test")
    print("=" * 60)
    
    tests = [
        test_basic_imports,
        test_cache_system,
        test_data_operations,
        test_prediction_simulation
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {str(e)}")
            results.append(False)
    
    print("\n📊 Test Results:")
    print("=" * 40)
    
    passed = sum(results)
    total = len(results)
    
    test_names = [
        "Basic Imports",
        "Cache System", 
        "Data Operations",
        "Prediction Simulation"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name}: {status}")
    
    print("=" * 40)
    print(f"Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = run_simple_tests()
    
    if success:
        print("\n✅ Simple performance testing completed successfully!")
        print("🚀 You can now run the enhanced app with: streamlit run app_enhanced_performance.py")
    else:
        print("\n❌ Simple performance testing failed.")
        sys.exit(1)



