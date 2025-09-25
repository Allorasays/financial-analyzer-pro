#!/usr/bin/env python3
"""
Test script to diagnose analysis tools issues
"""

import sys
import os

def test_analysis_tools():
    """Test all analysis tools"""
    print("🔧 Testing Financial Analyzer Pro Analysis Tools")
    print("=" * 60)
    
    try:
        # Test 1: Import the app
        print("📦 Test 1: Importing app functions...")
        from app import get_market_data, predict_price_ml, calculate_technical_indicators
        print("✅ App functions imported successfully")
        
        # Test 2: Test data fetching
        print("\n📊 Test 2: Testing data fetching...")
        try:
            data = get_market_data('AAPL', '1mo', min_days=60)
            print(f"✅ Data fetch successful: {len(data)} days retrieved")
            print(f"   Columns: {list(data.columns)}")
        except Exception as e:
            print(f"❌ Data fetch failed: {str(e)}")
            return False
        
        # Test 3: Test technical indicators
        print("\n📈 Test 3: Testing technical indicators...")
        try:
            indicators = calculate_technical_indicators(data)
            print(f"✅ Indicators calculated successfully")
            print(f"   Enhanced columns: {list(indicators.columns)}")
        except Exception as e:
            print(f"❌ Technical indicators failed: {str(e)}")
            return False
        
        # Test 4: Test ML predictions
        print("\n🤖 Test 4: Testing ML predictions...")
        try:
            predictions, error = predict_price_ml(indicators, 'AAPL', 5)
            if predictions:
                print("✅ ML prediction successful!")
                print(f"   Model: {predictions['model_type']}")
                print(f"   Confidence: {predictions.get('confidence', 'N/A'):.1f}%")
                print(f"   Data points: {predictions.get('data_points', 'N/A')} days")
            else:
                print(f"❌ ML prediction failed: {error}")
                return False
        except Exception as e:
            print(f"❌ ML prediction error: {str(e)}")
            return False
        
        # Test 5: Test with different symbols
        print("\n🎯 Test 5: Testing with different symbols...")
        test_symbols = ['MSFT', 'GOOGL', 'TSLA']
        for symbol in test_symbols:
            try:
                data = get_market_data(symbol, '3mo', min_days=60)
                indicators = calculate_technical_indicators(data)
                predictions, error = predict_price_ml(indicators, symbol, 5)
                if predictions:
                    print(f"✅ {symbol}: SUCCESS")
                else:
                    print(f"❌ {symbol}: {error}")
            except Exception as e:
                print(f"❌ {symbol}: Error - {str(e)}")
        
        print("\n🎉 All analysis tools test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Critical error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_analysis_tools()
    if success:
        print("\n✅ Analysis tools are working correctly!")
    else:
        print("\n❌ Analysis tools have issues that need fixing!")
