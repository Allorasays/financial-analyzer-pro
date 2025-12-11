"""
Prediction Validation Job
Automatically validates stored predictions against actual prices
"""
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging
from prediction_tracker import prediction_tracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionValidator:
    """Validates predictions by fetching actual prices"""
    
    def __init__(self, tracker=None):
        self.tracker = tracker or prediction_tracker
    
    def validate_pending_predictions(self, max_days_past: int = 7) -> Dict[str, Any]:
        """
        Validate all pending predictions
        
        Args:
            max_days_past: Maximum days past target date to still validate
        
        Returns:
            Dictionary with validation results
        """
        pending = self.tracker.get_pending_validations(max_days_past=max_days_past)
        
        if not pending:
            logger.info("No pending predictions to validate")
            return {
                'status': 'success',
                'validated_count': 0,
                'failed_count': 0,
                'validations': []
            }
        
        logger.info(f"Validating {len(pending)} pending predictions...")
        
        validated_count = 0
        failed_count = 0
        validations = []
        
        # Group by ticker to batch fetch prices
        by_ticker = {}
        for pred in pending:
            ticker = pred['ticker']
            if ticker not in by_ticker:
                by_ticker[ticker] = []
            by_ticker[ticker].append(pred)
        
        # Validate each ticker's predictions
        for ticker, predictions in by_ticker.items():
            try:
                # Fetch historical data for this ticker
                stock = yf.Ticker(ticker)
                
                # Get dates we need
                target_dates = [p['target_date'] for p in predictions]
                min_date = min(target_dates)
                max_date = max(target_dates)
                
                # Fetch historical data
                hist = stock.history(start=min_date, end=(datetime.strptime(max_date, '%Y-%m-%d') + timedelta(days=2)).strftime('%Y-%m-%d'))
                
                if hist.empty:
                    logger.warning(f"No historical data for {ticker}")
                    failed_count += len(predictions)
                    continue
                
                # Validate each prediction
                for pred in predictions:
                    try:
                        target_date = pred['target_date']
                        target_datetime = datetime.strptime(target_date, '%Y-%m-%d')
                        
                        # Try to get exact date, if market was closed, use next trading day
                        if target_date in hist.index.strftime('%Y-%m-%d').tolist():
                            date_idx = hist.index.strftime('%Y-%m-%d').tolist().index(target_date)
                            actual_price = float(hist.iloc[date_idx]['Close'])
                        else:
                            # Market was closed, use closest trading day
                            # Find closest date after target
                            future_dates = hist[hist.index >= target_datetime]
                            if not future_dates.empty:
                                actual_price = float(future_dates.iloc[0]['Close'])
                            else:
                                logger.warning(f"Could not find price for {ticker} on or after {target_date}")
                                failed_count += 1
                                continue
                        
                        # Validate the prediction
                        validation = self.tracker.validate_prediction(
                            prediction_id=pred['id'],
                            actual_price=actual_price,
                            actual_date=target_date
                        )
                        
                        validations.append(validation)
                        validated_count += 1
                        
                    except Exception as e:
                        logger.error(f"Error validating prediction {pred['id']}: {e}")
                        failed_count += 1
                
            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {e}")
                failed_count += len(predictions)
        
        logger.info(f"Validation complete: {validated_count} validated, {failed_count} failed")
        
        return {
            'status': 'success',
            'validated_count': validated_count,
            'failed_count': failed_count,
            'validations': validations
        }
    
    def validate_single_prediction(self, prediction_id: str) -> Dict[str, Any]:
        """
        Validate a single prediction by ID
        
        Args:
            prediction_id: ID of prediction to validate
        
        Returns:
            Validation result
        """
        # Get prediction details
        conn = self.tracker.db_path
        import sqlite3
        conn_db = sqlite3.connect(str(conn))
        cursor = conn_db.cursor()
        
        cursor.execute('''
            SELECT ticker, target_date, predicted_price, current_price
            FROM predictions
            WHERE id = ?
        ''', (prediction_id,))
        
        result = cursor.fetchone()
        conn_db.close()
        
        if not result:
            return {'status': 'error', 'message': 'Prediction not found'}
        
        ticker, target_date, predicted_price, current_price = result
        
        try:
            # Fetch actual price
            stock = yf.Ticker(ticker)
            target_datetime = datetime.strptime(target_date, '%Y-%m-%d')
            
            # Fetch data around target date
            hist = stock.history(
                start=(target_datetime - timedelta(days=5)).strftime('%Y-%m-%d'),
                end=(target_datetime + timedelta(days=5)).strftime('%Y-%m-%d')
            )
            
            if hist.empty:
                return {'status': 'error', 'message': 'No historical data available'}
            
            # Get actual price (use closest trading day)
            if target_date in hist.index.strftime('%Y-%m-%d').tolist():
                date_idx = hist.index.strftime('%Y-%m-%d').tolist().index(target_date)
                actual_price = float(hist.iloc[date_idx]['Close'])
            else:
                future_dates = hist[hist.index >= target_datetime]
                if not future_dates.empty:
                    actual_price = float(future_dates.iloc[0]['Close'])
                else:
                    return {'status': 'error', 'message': 'Could not find price for target date'}
            
            # Validate
            validation = self.tracker.validate_prediction(
                prediction_id=prediction_id,
                actual_price=actual_price,
                actual_date=target_date
            )
            
            return {'status': 'success', 'validation': validation}
            
        except Exception as e:
            logger.error(f"Error validating prediction {prediction_id}: {e}")
            return {'status': 'error', 'message': str(e)}


# Global instance
prediction_validator = PredictionValidator()

