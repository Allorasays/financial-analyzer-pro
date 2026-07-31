"""
Daily Predictions Job
Automatically makes 10 predictions per day for tracking accuracy
"""
import sys
import requests
import logging
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Popular stocks to predict (rotates daily) — Skill: larger personal universe
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "TSLA", "META", "NFLX", "AMD", "INTC",
    "JPM", "BAC", "WMT", "JNJ", "PG",
    "SPY", "QQQ", "DIA", "IWM", "XLF",
    "BABA", "NIO", "PLTR", "RIVN", "SOFI",
    "CRM", "ORCL", "ADBE", "AVGO", "COST",
    "XOM", "CVX", "UNH", "V", "MA",
    "DIS", "KO", "PEP", "T", "VZ",
]

def make_daily_predictions(base_url: str = "http://localhost:8000", tickers: list = None, count: int = 30):
    """
    Make predictions for specified number of stocks
    
    Args:
        base_url: API base URL
        tickers: List of tickers to choose from (defaults to DEFAULT_TICKERS)
        count: Number of predictions to make (default: 30)
    
    Returns:
        Dictionary with results
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS
    
    # Select tickers (use date to rotate selection for variety)
    import hashlib
    date_str = datetime.now().strftime('%Y-%m-%d')
    date_hash = int(hashlib.md5(date_str.encode()).hexdigest(), 16)
    
    # Rotate selection based on date
    selected_tickers = []
    for i in range(count):
        idx = (date_hash + i) % len(tickers)
        selected_tickers.append(tickers[idx])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_tickers = []
    for ticker in selected_tickers:
        if ticker not in seen:
            seen.add(ticker)
            unique_tickers.append(ticker)
    
    # If we need more, fill from the list
    while len(unique_tickers) < count:
        for ticker in tickers:
            if ticker not in seen:
                seen.add(ticker)
                unique_tickers.append(ticker)
                if len(unique_tickers) >= count:
                    break
    
    unique_tickers = unique_tickers[:count]
    
    logger.info(f"Making predictions for {len(unique_tickers)} tickers: {', '.join(unique_tickers)}")
    
    results = {
        'date': date_str,
        'total_predictions': 0,
        'successful': 0,
        'failed': 0,
        'tickers': []
    }
    
    for ticker in unique_tickers:
        try:
            # Make prediction request with retry logic (for cold starts)
            url = f"{base_url}/api/ml/predictions/{ticker}"
            max_retries = 3
            response = None
            
            for attempt in range(max_retries):
                try:
                    response = requests.get(url, params={'prediction_days': 30}, timeout=60)
                    if response.status_code == 200:
                        break
                except requests.exceptions.RequestException as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt + 1} failed for {ticker}, retrying...")
                        import time
                        time.sleep(5)  # Wait 5 seconds before retry
                    else:
                        raise
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('status') == 'success':
                    results['successful'] += 1
                    results['total_predictions'] += 1
                    results['tickers'].append({
                        'ticker': ticker,
                        'status': 'success',
                        'current_price': data.get('current_price'),
                        'next_day_prediction': data.get('next_day'),
                        'confidence': data.get('confidence_score'),
                        'model_version': data.get('model_metadata', {}).get('model_version')
                    })
                    logger.info(f"✓ Prediction made for {ticker}: ${data.get('current_price'):.2f} → ${data.get('next_day'):.2f}")
                else:
                    results['failed'] += 1
                    results['tickers'].append({
                        'ticker': ticker,
                        'status': 'error',
                        'error': data.get('error', 'Unknown error')
                    })
                    logger.warning(f"✗ Prediction failed for {ticker}: {data.get('error', 'Unknown error')}")
            else:
                results['failed'] += 1
                results['tickers'].append({
                    'ticker': ticker,
                    'status': 'error',
                    'error': f"HTTP {response.status_code}"
                })
                logger.error(f"✗ HTTP error for {ticker}: {response.status_code}")
                
        except Exception as e:
            results['failed'] += 1
            results['tickers'].append({
                'ticker': ticker,
                'status': 'error',
                'error': str(e)
            })
            logger.error(f"✗ Exception for {ticker}: {e}")
    
    logger.info(f"Daily predictions complete: {results['successful']} successful, {results['failed']} failed")
    return results


def main():
    """Main function - makes 10 daily predictions"""
    import os
    
    # Get base URL from environment or use default
    base_url = os.getenv('API_BASE_URL', 'http://localhost:8000')
    
    # If running on Render, construct the service URL
    if 'RENDER' in os.environ:
        # Render provides service URL via environment
        render_service_url = os.getenv('RENDER_SERVICE_URL')
        if render_service_url:
            base_url = render_service_url
        else:
            # Fallback: construct from known service name
            base_url = 'https://moneta-backend-api.onrender.com'
    
    logger.info(f"Starting daily predictions job (target: 30 predictions)")
    logger.info(f"API Base URL: {base_url}")
    
    try:
        results = make_daily_predictions(base_url=base_url, count=30)
        
        logger.info("=" * 60)
        logger.info("DAILY PREDICTIONS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Date: {results['date']}")
        logger.info(f"Total Attempted: {results['total_predictions']}")
        logger.info(f"Successful: {results['successful']}")
        logger.info(f"Failed: {results['failed']}")
        logger.info("=" * 60)
        
        return results
        
    except Exception as e:
        logger.error(f"Daily predictions job failed: {e}", exc_info=True)
        return {'status': 'error', 'error': str(e)}


if __name__ == "__main__":
    main()

