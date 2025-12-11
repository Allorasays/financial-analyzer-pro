"""
Test script for daily predictions job
Run this to test the prediction system locally
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from daily_predictions_job import make_daily_predictions
from prediction_validator import prediction_validator
from prediction_tracker import prediction_tracker

def test_predictions():
    """Test making predictions"""
    print("=" * 60)
    print("TESTING DAILY PREDICTIONS")
    print("=" * 60)
    
    # Make 3 test predictions (smaller for testing)
    results = make_daily_predictions(base_url="http://localhost:8000", count=3)
    
    print(f"\nResults: {results['successful']} successful, {results['failed']} failed")
    for ticker_result in results['tickers']:
        print(f"  {ticker_result['ticker']}: {ticker_result['status']}")
    
    return results

def test_validation():
    """Test validation"""
    print("\n" + "=" * 60)
    print("TESTING VALIDATION")
    print("=" * 60)
    
    # Get pending validations
    pending = prediction_tracker.get_pending_validations(max_days_past=30)
    print(f"Pending validations: {len(pending)}")
    
    if pending:
        # Validate first one as test
        print(f"\nValidating first pending: {pending[0]['ticker']} (ID: {pending[0]['id']})")
        result = prediction_validator.validate_single_prediction(pending[0]['id'])
        print(f"Result: {result.get('status')}")
        if result.get('validation'):
            val = result['validation']
            print(f"  Predicted: ${val.get('predicted_price'):.2f}")
            print(f"  Actual: ${val.get('actual_price'):.2f}")
            print(f"  Error: ${val.get('price_error'):.2f} ({val.get('price_error_pct'):.2f}%)")
            print(f"  Direction: {'✓ Correct' if val.get('direction_correct') else '✗ Wrong'}")
    else:
        print("No pending validations (need to wait for target dates to pass)")

def test_accuracy():
    """Test accuracy metrics"""
    print("\n" + "=" * 60)
    print("TESTING ACCURACY METRICS")
    print("=" * 60)
    
    metrics = prediction_tracker.calculate_accuracy_metrics(min_validations=1)
    
    if metrics.get('status') == 'insufficient_data':
        print(f"Not enough data: {metrics.get('message')}")
    else:
        print(f"Total Validations: {metrics.get('total_validations')}")
        print(f"Direction Accuracy: {metrics.get('direction_accuracy_pct', 0):.1f}%")
        print(f"Mean Absolute Error: ${metrics.get('mean_absolute_error', 0):.2f}")
        print(f"RMSE: ${metrics.get('rmse', 0):.2f}")

if __name__ == "__main__":
    print("Testing Daily Predictions System\n")
    
    # Test predictions
    test_predictions()
    
    # Test validation (if any pending)
    test_validation()
    
    # Test accuracy
    test_accuracy()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

