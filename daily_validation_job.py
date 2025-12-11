"""
Daily Validation Job
Validates predictions that have reached their target dates
Runs after market close to validate previous day's predictions
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from prediction_validator import prediction_validator
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main validation job - validates all pending predictions"""
    logger.info("=" * 60)
    logger.info("DAILY PREDICTION VALIDATION JOB")
    logger.info("=" * 60)
    logger.info("Starting validation of pending predictions...")
    
    try:
        result = prediction_validator.validate_pending_predictions(max_days_past=7)
        
        logger.info("=" * 60)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Validated: {result['validated_count']}")
        logger.info(f"Failed: {result['failed_count']}")
        
        if result['validations']:
            # Calculate quick stats
            direction_correct = sum(1 for v in result['validations'] if v.get('direction_correct', False))
            direction_accuracy = (direction_correct / len(result['validations'])) * 100 if result['validations'] else 0
            
            price_errors = [abs(v.get('price_error_pct', 0)) for v in result['validations']]
            avg_error_pct = sum(price_errors) / len(price_errors) if price_errors else 0
            
            logger.info(f"Direction Accuracy: {direction_accuracy:.1f}% ({direction_correct}/{len(result['validations'])})")
            logger.info(f"Average Error: {avg_error_pct:.2f}%")
            
            # Show breakdown by ticker
            by_ticker = {}
            for v in result['validations']:
                ticker = v.get('ticker', 'UNKNOWN')
                if ticker not in by_ticker:
                    by_ticker[ticker] = {'correct': 0, 'total': 0}
                by_ticker[ticker]['total'] += 1
                if v.get('direction_correct', False):
                    by_ticker[ticker]['correct'] += 1
            
            logger.info("\nAccuracy by Ticker:")
            for ticker, stats in sorted(by_ticker.items()):
                acc = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
                logger.info(f"  {ticker}: {acc:.1f}% ({stats['correct']}/{stats['total']})")
        
        logger.info("=" * 60)
        
        return result
        
    except Exception as e:
        logger.error(f"Validation job failed: {e}", exc_info=True)
        return {'status': 'error', 'error': str(e)}


if __name__ == "__main__":
    main()

