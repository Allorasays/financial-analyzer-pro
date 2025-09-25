#!/usr/bin/env python3
"""
Test the Streamlit app directly
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_streamlit_functions():
    """Test the main Streamlit functions"""
    print("🧪 Testing Streamlit App Functions")
    print("=" * 50)
    
    try:
        # Import the main function
        from app import main
        
        print("✅ Main function imported successfully")
        
        # Test individual components
        from app import get_market_data, calculate_technical_indicators, predict_price_ml, create_candlestick_chart, get_global_markets_overview
        
        print("✅ All functions imported successfully")
        
        # Test data flow
        print("\n📊 Testing data flow...")
        
        # Test 1: Data fetching
        symbol = "AAPL"
        timeframe = "1mo"
        min_days = 60
        
        print(f"Testing with {symbol}, {timeframe}, min_days={min_days}")
        
        data = get_market_data(symbol, timeframe, min_days=min_days)
        print(f"✅ Data fetched: {len(data)} days")
        
        # Test 2: Technical indicators
        data_with_indicators = calculate_technical_indicators(data)
        print(f"✅ Indicators calculated: {len(data_with_indicators.columns)} columns")
        
        # Test 3: ML predictions
        predictions, error = predict_price_ml(data_with_indicators, symbol, 5)
        if predictions:
            print(f"✅ ML predictions: SUCCESS (confidence: {predictions.get('confidence', 'N/A'):.1f}%)")
        else:
            print(f"❌ ML predictions: {error}")
        
        # Test 4: Chart creation
        fig = create_candlestick_chart(data_with_indicators, symbol)
        print(f"✅ Chart created: {type(fig)}")
        
        # Test 5: Global markets
        markets = get_global_markets_overview()
        print(f"✅ Global markets: {len(markets)} indices")
        
        print("\n🎉 All Streamlit functions working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing Streamlit app: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_streamlit_functions()
    if success:
        print("\n✅ Streamlit app functions are working!")
    else:
        print("\n❌ Streamlit app has issues!")
