"""
Prediction Validation Job
Can be run as a scheduled job (cron, etc.) to automatically validate predictions
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
    logger.info("Starting prediction validation job...")
    
    try:
        result = prediction_validator.validate_pending_predictions(max_days_past=7)
        
        logger.info(f"Validation complete:")
        logger.info(f"  - Validated: {result['validated_count']}")
        logger.info(f"  - Failed: {result['failed_count']}")
        
        if result['validations']:
            # Calculate quick stats
            direction_correct = sum(1 for v in result['validations'] if v.get('direction_correct', False))
            direction_accuracy = (direction_correct / len(result['validations'])) * 100 if result['validations'] else 0
            
            avg_error_pct = sum(abs(v.get('price_error_pct', 0)) for v in result['validations']) / len(result['validations']) if result['validations'] else 0
            
            logger.info(f"  - Direction Accuracy: {direction_accuracy:.1f}%")
            logger.info(f"  - Average Error: {avg_error_pct:.2f}%")
        
        return result
        
    except Exception as e:
        logger.error(f"Validation job failed: {e}", exc_info=True)
        return {'status': 'error', 'error': str(e)}

if __name__ == "__main__":
    main()

