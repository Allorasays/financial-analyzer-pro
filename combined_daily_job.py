"""
Combined Daily Job
Makes predictions AND validates pending ones
Run this daily (recommended: after market close, ~6 PM ET / 11 PM UTC)
"""
import sys
from pathlib import Path
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from daily_predictions_job import make_daily_predictions
import os
from prediction_validator import prediction_validator
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main combined job - makes predictions and validates pending ones"""
    logger.info("=" * 80)
    logger.info("COMBINED DAILY PREDICTION JOB")
    logger.info("=" * 80)
    logger.info(f"Date: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    # Get base URL
    base_url = os.getenv('API_BASE_URL', 'http://localhost:8000')
    if 'RENDER' in os.environ:
        render_service_url = os.getenv('RENDER_SERVICE_URL')
        if render_service_url:
            base_url = render_service_url
        else:
            base_url = 'https://moneta-backend-api.onrender.com'
    
    results = {
        'predictions': None,
        'validations': None,
        'screener': None,
    }
    
    # Step 1: Make new predictions
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: MAKING NEW PREDICTIONS")
    logger.info("=" * 80)
    try:
        results['predictions'] = make_daily_predictions(base_url=base_url, count=10)
        logger.info(f"✓ Made {results['predictions']['successful']} predictions")
    except Exception as e:
        logger.error(f"✗ Prediction step failed: {e}", exc_info=True)
        results['predictions'] = {'status': 'error', 'error': str(e)}
    
    # Step 2: Validate pending predictions
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: VALIDATING PENDING PREDICTIONS")
    logger.info("=" * 80)
    try:
        results['validations'] = prediction_validator.validate_pending_predictions(max_days_past=7)
        logger.info(f"✓ Validated {results['validations']['validated_count']} predictions")
    except Exception as e:
        logger.error(f"✗ Validation step failed: {e}", exc_info=True)
        results['validations'] = {'status': 'error', 'error': str(e)}

    # Step 3: Refresh investability screener rankings
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: SCREENER REFRESH")
    logger.info("=" * 80)
    try:
        from screener_daily_job import run_screener_refresh
        results['screener'] = run_screener_refresh(
            universe=os.getenv('SCREENER_UNIVERSE', 'core'),
            limit=int(os.getenv('SCREENER_LIMIT', '25')),
            top_n=int(os.getenv('SCREENER_TOP_N', '10')),
            mode=os.getenv('SCREENER_MODE', 'lite'),
        )
        logger.info(f"✓ Screener scored {results['screener'].get('scored', 0)} tickers")
    except Exception as e:
        logger.error(f"✗ Screener step failed: {e}", exc_info=True)
        results['screener'] = {'status': 'error', 'error': str(e)}
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("DAILY JOB SUMMARY")
    logger.info("=" * 80)
    
    if results['predictions']:
        pred = results['predictions']
        logger.info(f"Predictions: {pred.get('successful', 0)} successful, {pred.get('failed', 0)} failed")
    
    if results['validations']:
        val = results['validations']
        logger.info(f"Validations: {val.get('validated_count', 0)} validated, {val.get('failed_count', 0)} failed")
        
        if val.get('validations'):
            direction_correct = sum(1 for v in val['validations'] if v.get('direction_correct', False))
            direction_accuracy = (direction_correct / len(val['validations'])) * 100 if val['validations'] else 0
            logger.info(f"Direction Accuracy: {direction_accuracy:.1f}%")

    if results.get('screener'):
        scr = results['screener']
        logger.info(f"Screener: scored={scr.get('scored', 0)} elapsed={scr.get('elapsed_seconds')}s")
    
    logger.info("=" * 80)
    logger.info("Daily job complete!")
    
    return results


if __name__ == "__main__":
    main()

