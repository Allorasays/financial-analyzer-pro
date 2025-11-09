#!/usr/bin/env python3
"""
ML Accuracy Evaluation Script for Financial Analyzer
Tests ML prediction accuracy with current data sources
"""

import sys
import os
import time
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urljoin, urlparse
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

DEFAULT_BASE_URL = os.getenv("ML_BASE_URL", "https://moneta-backend-api.onrender.com/")
DEFAULT_TICKERS = [t.strip().upper() for t in os.getenv(
    "ML_EVAL_TICKERS",
    "AAPL,MSFT,GOOGL,TSLA,SPY,QQQ,NVDA"
).split(",") if t.strip()]
REPORT_PATH = Path(os.getenv("ML_EVAL_REPORT_PATH", os.path.join(current_dir, "reports", "ml_accuracy_report.json")))
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


def _build_url(base_url: str, path: str) -> str:
    if not base_url.endswith("/"):
        base_url = base_url + "/"
    return urljoin(base_url, path.lstrip("/"))


def _is_local(base_url: str) -> bool:
    parsed = urlparse(base_url)
    return parsed.hostname in {"localhost", "127.0.0.1"} or base_url.startswith("http://0.0.0.0")

def test_backend_availability(base_url: str = DEFAULT_BASE_URL):
    """Test if backend services are running"""
    print("Testing Backend Availability...")
    proxy_status = False
    mobile_status = True

    health_url = _build_url(base_url, "api/stats")
    try:
        response = requests.get(health_url, timeout=10)
        if response.status_code == 200:
            print(f"[SUCCESS] Backend reachable at {health_url}")
            proxy_status = True
        else:
            print(f"[ERROR] Backend responded with status {response.status_code} at {health_url}")
    except Exception as exc:
        print(f"[ERROR] Unable to reach backend at {health_url}: {exc}")

    if _is_local(base_url):
        try:
            response = requests.get("http://localhost:8001/api/stats", timeout=5)
            if response.status_code == 200:
                print("[SUCCESS] Mobile API (port 8001): Running")
                mobile_status = True
            else:
                print("[ERROR] Mobile API (port 8001): Not responding")
                mobile_status = False
        except Exception:
            print("[ERROR] Mobile API (port 8001): Not running")
            mobile_status = False

    return proxy_status, mobile_status

def test_ml_prediction_accuracy(base_url: str = DEFAULT_BASE_URL, tickers=None):
    """Test ML prediction accuracy with multiple tickers"""
    print("\nTesting ML Prediction Accuracy...")
    
    # Test tickers with different characteristics
    test_tickers = tickers or DEFAULT_TICKERS
    results = {}
    
    for ticker in test_tickers:
        print(f"\nTesting {ticker}...")
        
        try:
            # Test ML predictions endpoint
            url = _build_url(base_url, f"api/ml/predictions/{ticker}")
            response = requests.get(url, params={'prediction_days': 5}, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('status') == 'success':
                    predictions = data.get('predictions', {})
                    model_metrics = data.get('model_metrics', {})
                    
                    # Extract key metrics
                    r2_score = model_metrics.get('r2_score', 0)
                    rmse = model_metrics.get('rmse', 0)
                    mae = model_metrics.get('mae', 0)
                    confidence = data.get('confidence_score') or model_metrics.get('confidence', 0)
                    
                    # Get price forecast
                    price_forecast = predictions.get('price_forecast', [])
                    current_price = data.get('current_price', 0)
                    
                    # Calculate prediction accuracy metrics
                    if len(price_forecast) > 0 and current_price > 0:
                        next_day_pred = price_forecast[0]
                        price_change_pct = ((next_day_pred - current_price) / current_price) * 100
                        
                        results[ticker] = {
                            'status': 'success',
                            'r2_score': r2_score,
                            'rmse': rmse,
                            'mae': mae,
                            'confidence': confidence,
                            'current_price': current_price,
                            'next_day_prediction': next_day_pred,
                            'predicted_change_pct': price_change_pct,
                            'model_version': data.get('model_version', 'unknown'),
                            'last_training_date': data.get('last_training_date')
                        }
                        
                        print(f"  [SUCCESS] R2 Score: {r2_score:.3f}")
                        print(f"  [SUCCESS] RMSE: ${rmse:.2f}")
                        print(f"  [SUCCESS] MAE: ${mae:.2f}")
                        print(f"  [SUCCESS] Confidence: {confidence:.1%}")
                        print(f"  [SUCCESS] Current: ${current_price:.2f}")
                        print(f"  [SUCCESS] Next Day: ${next_day_pred:.2f} ({price_change_pct:+.2f}%)")
                    else:
                        results[ticker] = {
                            'status': 'partial',
                            'error': 'No price forecast available'
                        }
                        print(f"  [WARNING] Partial success - no price forecast")
                else:
                    error_msg = data.get('error', 'Unknown error')
                    results[ticker] = {
                        'status': 'error',
                        'error': error_msg
                    }
                    print(f"  [ERROR] Error: {error_msg}")
            else:
                results[ticker] = {
                    'status': 'error',
                    'error': f'HTTP {response.status_code}'
                }
                print(f"  [ERROR] HTTP Error: {response.status_code}")
                
        except Exception as e:
            results[ticker] = {
                'status': 'error',
                'error': str(e)
            }
            print(f"  [ERROR] Exception: {str(e)}")
        
        # Small delay to avoid rate limiting
        time.sleep(1)
    
    return results

def test_data_sources():
    """Test all available data sources"""
    print("\nTesting Data Sources...")
    
    data_sources = {
        'Yahoo Finance': test_yahoo_finance,
        'Tiingo API': test_tiingo_api,
        'NewsAPI': test_news_api,
        'FRED API': test_fred_api
    }
    
    results = {}
    
    for source_name, test_func in data_sources.items():
        print(f"\nTesting {source_name}...")
        try:
            result = test_func()
            results[source_name] = result
            if result.get('status') == 'success':
                print(f"  [SUCCESS] {source_name}: Working")
            else:
                print(f"  [ERROR] {source_name}: {result.get('error', 'Unknown error')}")
        except Exception as e:
            results[source_name] = {'status': 'error', 'error': str(e)}
            print(f"  [ERROR] {source_name}: {str(e)}")
    
    return results

def test_yahoo_finance():
    """Test Yahoo Finance data availability"""
    try:
        import yfinance as yf
        stock = yf.Ticker('AAPL')
        hist = stock.history(period="5d")
        
        if len(hist) > 0:
            return {'status': 'success', 'data_points': len(hist)}
        else:
            return {'status': 'error', 'error': 'No data returned'}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

def test_tiingo_api():
    """Test Tiingo API"""
    try:
        from tiingo_service import get_company_info, get_stock_price
        company_info = get_company_info('AAPL')
        price_data = get_stock_price('AAPL')
        
        if company_info and price_data:
            return {'status': 'success', 'company_info': bool(company_info), 'price_data': len(price_data)}
        else:
            return {'status': 'error', 'error': 'No data returned'}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

def test_news_api():
    """Test NewsAPI"""
    try:
        from news_service import get_news_for_ticker
        news_data = get_news_for_ticker('AAPL')
        
        if news_data and len(news_data) > 0:
            return {'status': 'success', 'articles': len(news_data)}
        else:
            return {'status': 'error', 'error': 'No news data returned'}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

def test_fred_api():
    """Test FRED API"""
    try:
        from fred_service import get_interest_rates, get_treasury_rates
        interest_rates = get_interest_rates()
        treasury_rates = get_treasury_rates()
        
        if interest_rates and treasury_rates:
            return {'status': 'success', 'interest_rates': bool(interest_rates), 'treasury_rates': len(treasury_rates)}
        else:
            return {'status': 'error', 'error': 'No economic data returned'}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

def calculate_overall_accuracy(ml_results):
    """Calculate overall ML accuracy metrics"""
    print("\nCalculating Overall ML Accuracy...")
    
    successful_tests = [r for r in ml_results.values() if r.get('status') == 'success']
    
    if not successful_tests:
        print("[ERROR] No successful ML tests to analyze")
        return None
    
    # Calculate average metrics
    avg_r2 = np.mean([r.get('r2_score', 0) for r in successful_tests])
    avg_rmse = np.mean([r.get('rmse', 0) for r in successful_tests])
    avg_mae = np.mean([r.get('mae', 0) for r in successful_tests])
    avg_confidence = np.mean([r.get('confidence', 0) for r in successful_tests])
    
    # Calculate accuracy categories
    high_accuracy = len([r for r in successful_tests if r.get('r2_score', 0) > 0.7])
    medium_accuracy = len([r for r in successful_tests if 0.4 <= r.get('r2_score', 0) <= 0.7])
    low_accuracy = len([r for r in successful_tests if r.get('r2_score', 0) < 0.4])
    
    total_tests = len(ml_results)
    success_rate = len(successful_tests) / total_tests
    
    accuracy_summary = {
        'total_tests': total_tests,
        'successful_tests': len(successful_tests),
        'success_rate': success_rate,
        'avg_r2_score': avg_r2,
        'avg_rmse': avg_rmse,
        'avg_mae': avg_mae,
        'avg_confidence': avg_confidence,
        'high_accuracy_count': high_accuracy,
        'medium_accuracy_count': medium_accuracy,
        'low_accuracy_count': low_accuracy,
        'accuracy_distribution': {
            'high': high_accuracy,
            'medium': medium_accuracy,
            'low': low_accuracy
        }
    }
    
    print(f"Overall ML Accuracy Summary:")
    print(f"  [SUCCESS] Success Rate: {success_rate:.1%} ({len(successful_tests)}/{total_tests})")
    print(f"  [METRIC] Average R2 Score: {avg_r2:.3f}")
    print(f"  [METRIC] Average RMSE: ${avg_rmse:.2f}")
    print(f"  [METRIC] Average MAE: ${avg_mae:.2f}")
    print(f"  [METRIC] Average Confidence: {avg_confidence:.1%}")
    print(f"  [HIGH] High Accuracy (R2 > 0.7): {high_accuracy}")
    print(f"  [MEDIUM] Medium Accuracy (R2 0.4-0.7): {medium_accuracy}")
    print(f"  [LOW] Low Accuracy (R2 < 0.4): {low_accuracy}")
    
    return accuracy_summary

def generate_recommendations(ml_results, data_source_results, accuracy_summary):
    """Generate recommendations for improving ML accuracy"""
    print("\nML Accuracy Improvement Recommendations:")
    
    recommendations = []
    
    # Data quality recommendations
    if accuracy_summary and accuracy_summary['avg_r2_score'] < 0.6:
        recommendations.append({
            'category': 'Data Quality',
            'priority': 'High',
            'recommendation': 'Increase historical data period to 2+ years for better model training',
            'impact': 'High'
        })
    
    # Feature engineering recommendations
    if accuracy_summary and accuracy_summary['avg_confidence'] < 0.7:
        recommendations.append({
            'category': 'Feature Engineering',
            'priority': 'High',
            'recommendation': 'Add more technical indicators and macroeconomic features',
            'impact': 'Medium'
        })
    
    # Model improvements
    if accuracy_summary and accuracy_summary['success_rate'] < 0.8:
        recommendations.append({
            'category': 'Model Architecture',
            'priority': 'Medium',
            'recommendation': 'Implement LSTM neural networks for time series prediction',
            'impact': 'High'
        })
    
    # Data source improvements
    failed_sources = [name for name, result in data_source_results.items() if result.get('status') != 'success']
    if failed_sources:
        recommendations.append({
            'category': 'Data Sources',
            'priority': 'High',
            'recommendation': f'Fix or replace failed data sources: {", ".join(failed_sources)}',
            'impact': 'High'
        })
    
    # API upgrades
    recommendations.append({
        'category': 'API Upgrades',
        'priority': 'Medium',
        'recommendation': 'Upgrade FMP to Starter plan for enhanced financial data',
        'impact': 'Medium'
    })
    
    recommendations.append({
        'category': 'API Upgrades',
        'priority': 'Low',
        'recommendation': 'Consider upgrading Tiingo to Premium for higher rate limits',
        'impact': 'Low'
    })
    
    # Display recommendations
    for i, rec in enumerate(recommendations, 1):
        priority_text = {'High': '[HIGH]', 'Medium': '[MEDIUM]', 'Low': '[LOW]'}[rec['priority']]
        impact_text = {'High': '[HIGH IMPACT]', 'Medium': '[MEDIUM IMPACT]', 'Low': '[LOW IMPACT]'}[rec['impact']]
        
        print(f"  {i}. {priority_text} {rec['category']}: {rec['recommendation']}")
        print(f"     Impact: {impact_text}")
    
    return recommendations

def write_report(report_payload: dict):
    try:
        REPORT_PATH.write_text(json.dumps(report_payload, indent=2, default=str))
        print(f"\n[INFO] Detailed report written to {REPORT_PATH}")
    except Exception as exc:
        print(f"[WARNING] Failed to write report: {exc}")


def main(base_url: str = DEFAULT_BASE_URL, tickers=None):
    """Main evaluation function"""
    print("Financial Analyzer ML Accuracy Evaluation")
    print("=" * 60)
    evaluation_time = datetime.now()
    print(f"Evaluation Time: {evaluation_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print(f"Target Base URL: {base_url}")
    
    # Test backend availability
    proxy_status, mobile_status = test_backend_availability(base_url)
    
    if not proxy_status:
        print("\n[ERROR] Backend services not running. Please start them first:")
        if _is_local(base_url):
            print("   python proxy.py")
        else:
            print(f"   Verify accessibility of {base_url}")
        return
    
    # Test data sources
    data_source_results = test_data_sources()
    
    # Test ML predictions
    ml_results = test_ml_prediction_accuracy(base_url, tickers=tickers)
    
    # Calculate overall accuracy
    accuracy_summary = calculate_overall_accuracy(ml_results)
    
    # Generate recommendations
    recommendations = generate_recommendations(ml_results, data_source_results, accuracy_summary)
    
    # Final summary
    print("\n" + "=" * 60)
    print("ML ACCURACY EVALUATION SUMMARY")
    print("=" * 60)
    
    if accuracy_summary:
        print(f"[SUCCESS] Overall Success Rate: {accuracy_summary['success_rate']:.1%}")
        print(f"[METRIC] Average R2 Score: {accuracy_summary['avg_r2_score']:.3f}")
        print(f"[METRIC] Average Confidence: {accuracy_summary['avg_confidence']:.1%}")
        
        if accuracy_summary['avg_r2_score'] > 0.7:
            print("[EXCELLENT] ML Accuracy: EXCELLENT")
        elif accuracy_summary['avg_r2_score'] > 0.5:
            print("[GOOD] ML Accuracy: GOOD")
        elif accuracy_summary['avg_r2_score'] > 0.3:
            print("[FAIR] ML Accuracy: FAIR")
        else:
            print("[NEEDS IMPROVEMENT] ML Accuracy: NEEDS IMPROVEMENT")
    else:
        print("[ERROR] Unable to calculate accuracy metrics")
    
    print(f"\nData Sources Status:")
    for source, result in data_source_results.items():
        status_text = "[SUCCESS]" if result.get('status') == 'success' else "[ERROR]"
        print(f"  {status_text} {source}: {result.get('status', 'unknown')}")
    
    print(f"\nRecommendations Generated: {len(recommendations)}")
    
    print("\nEvaluation Complete!")
    
    results_payload = {
        'ml_results': ml_results,
        'data_source_results': data_source_results,
        'accuracy_summary': accuracy_summary,
        'recommendations': recommendations
    }
    
    report_payload = {
        "evaluated_at": evaluation_time.isoformat(),
        "base_url": base_url,
        "tickers": tickers or DEFAULT_TICKERS,
        "backend_available": proxy_status,
        "mobile_api_available": mobile_status,
        "ml_results": ml_results,
        "accuracy_summary": accuracy_summary,
        "data_sources": data_source_results,
        "recommendations": recommendations,
    }
    write_report(report_payload)
    
    return results_payload

if __name__ == "__main__":
    results = main()