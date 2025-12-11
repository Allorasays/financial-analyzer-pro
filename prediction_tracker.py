"""
Prediction Tracking System
Stores predictions and validates them against actual future prices
"""
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = Path("data/prediction_tracker.db")

class PredictionTracker:
    """Tracks ML predictions and validates them against actual outcomes"""
    
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for prediction tracking"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Predictions table - stores predictions when made
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY,
                ticker TEXT NOT NULL,
                prediction_timestamp TEXT NOT NULL,
                current_price REAL NOT NULL,
                predicted_price REAL NOT NULL,
                target_date TEXT NOT NULL,
                horizon_days INTEGER NOT NULL,
                model_version TEXT,
                confidence_score REAL,
                r2_score REAL,
                features_used INTEGER,
                prediction_type TEXT,
                created_at TEXT NOT NULL
            )
        ''')
        
        # Validation table - stores actual results when validated
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS validations (
                id TEXT PRIMARY KEY,
                prediction_id TEXT NOT NULL,
                ticker TEXT NOT NULL,
                actual_price REAL,
                actual_date TEXT,
                price_error REAL,
                price_error_pct REAL,
                direction_correct INTEGER,
                validated_at TEXT NOT NULL,
                FOREIGN KEY (prediction_id) REFERENCES predictions(id)
            )
        ''')
        
        # Create indexes for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ticker ON predictions(ticker)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_target_date ON predictions(target_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_prediction_id ON validations(prediction_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_validation_date ON validations(validated_at)')
        
        conn.commit()
        conn.close()
    
    def store_prediction(
        self,
        ticker: str,
        predicted_price: float,
        current_price: float,
        horizon_days: int,
        model_version: str = None,
        confidence_score: float = None,
        r2_score: float = None,
        features_used: int = None,
        prediction_type: str = "next_day"
    ) -> str:
        """
        Store a prediction for later validation
        
        Args:
            ticker: Stock ticker
            predicted_price: Predicted price
            current_price: Current price when prediction was made
            horizon_days: Number of days ahead (1 = next day, 7 = next week, etc.)
            model_version: Model version string
            confidence_score: Model confidence (0-1)
            r2_score: R² score
            features_used: Number of features used
            prediction_type: Type of prediction (next_day, next_week, etc.)
        
        Returns:
            Prediction ID
        """
        prediction_id = f"{ticker}_{prediction_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        prediction_timestamp = datetime.utcnow().isoformat()
        target_date = (datetime.utcnow() + timedelta(days=horizon_days)).strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions 
            (id, ticker, prediction_timestamp, current_price, predicted_price, 
             target_date, horizon_days, model_version, confidence_score, 
             r2_score, features_used, prediction_type, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            prediction_id,
            ticker.upper(),
            prediction_timestamp,
            current_price,
            predicted_price,
            target_date,
            horizon_days,
            model_version,
            confidence_score,
            r2_score,
            features_used,
            prediction_type,
            prediction_timestamp
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Stored prediction {prediction_id} for {ticker}: ${predicted_price:.2f} on {target_date}")
        return prediction_id
    
    def validate_prediction(
        self,
        prediction_id: str,
        actual_price: float,
        actual_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate a prediction against actual price
        
        Args:
            prediction_id: ID of the prediction to validate
            actual_price: Actual price on target date
            actual_date: Actual date (defaults to today)
        
        Returns:
            Validation results dictionary
        """
        if actual_date is None:
            actual_date = datetime.utcnow().strftime('%Y-%m-%d')
        
        # Get the original prediction
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('SELECT predicted_price, current_price, ticker FROM predictions WHERE id = ?', (prediction_id,))
        result = cursor.fetchone()
        
        if not result:
            conn.close()
            raise ValueError(f"Prediction {prediction_id} not found")
        
        predicted_price, current_price, ticker = result
        
        # Calculate errors
        price_error = actual_price - predicted_price
        price_error_pct = (price_error / predicted_price) * 100 if predicted_price > 0 else 0
        
        # Check direction accuracy (did price move in predicted direction?)
        predicted_direction = 1 if predicted_price > current_price else -1 if predicted_price < current_price else 0
        actual_direction = 1 if actual_price > current_price else -1 if actual_price < current_price else 0
        direction_correct = 1 if predicted_direction == actual_direction else 0
        
        # Store validation
        cursor.execute('''
            INSERT OR REPLACE INTO validations
            (id, prediction_id, ticker, actual_price, actual_date, 
             price_error, price_error_pct, direction_correct, validated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            f"{prediction_id}_validation",
            prediction_id,
            ticker,
            actual_price,
            actual_date,
            price_error,
            price_error_pct,
            direction_correct,
            datetime.utcnow().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        validation_result = {
            'prediction_id': prediction_id,
            'ticker': ticker,
            'predicted_price': predicted_price,
            'actual_price': actual_price,
            'current_price': current_price,
            'price_error': price_error,
            'price_error_pct': price_error_pct,
            'direction_correct': bool(direction_correct),
            'predicted_direction': 'up' if predicted_direction > 0 else 'down' if predicted_direction < 0 else 'flat',
            'actual_direction': 'up' if actual_direction > 0 else 'down' if actual_direction < 0 else 'flat'
        }
        
        logger.info(f"Validated {prediction_id}: Error ${price_error:.2f} ({price_error_pct:.2f}%), Direction: {'✓' if direction_correct else '✗'}")
        return validation_result
    
    def get_pending_validations(self, max_days_past: int = 7) -> List[Dict[str, Any]]:
        """
        Get predictions that are ready for validation (target date has passed)
        
        Args:
            max_days_past: Maximum days past target date to still validate
        
        Returns:
            List of predictions ready for validation
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cutoff_date = (datetime.utcnow() - timedelta(days=max_days_past)).strftime('%Y-%m-%d')
        
        # Get predictions that haven't been validated yet and target date has passed
        cursor.execute('''
            SELECT p.id, p.ticker, p.predicted_price, p.current_price, 
                   p.target_date, p.horizon_days, p.model_version
            FROM predictions p
            LEFT JOIN validations v ON p.id = v.prediction_id
            WHERE p.target_date <= date('now')
              AND p.target_date >= ?
              AND v.id IS NULL
            ORDER BY p.target_date ASC
        ''', (cutoff_date,))
        
        results = cursor.fetchall()
        conn.close()
        
        pending = []
        for row in results:
            pending.append({
                'id': row[0],
                'ticker': row[1],
                'predicted_price': row[2],
                'current_price': row[3],
                'target_date': row[4],
                'horizon_days': row[5],
                'model_version': row[6]
            })
        
        return pending
    
    def calculate_accuracy_metrics(
        self,
        ticker: Optional[str] = None,
        model_version: Optional[str] = None,
        horizon_days: Optional[int] = None,
        min_validations: int = 10
    ) -> Dict[str, Any]:
        """
        Calculate accuracy metrics from validated predictions
        
        Args:
            ticker: Filter by ticker (None = all tickers)
            model_version: Filter by model version
            horizon_days: Filter by prediction horizon
            min_validations: Minimum number of validations required
        
        Returns:
            Dictionary with accuracy metrics
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Build query
        query = '''
            SELECT v.price_error, v.price_error_pct, v.direction_correct,
                   p.ticker, p.model_version, p.horizon_days, p.confidence_score
            FROM validations v
            JOIN predictions p ON v.prediction_id = p.id
            WHERE 1=1
        '''
        params = []
        
        if ticker:
            query += ' AND p.ticker = ?'
            params.append(ticker.upper())
        
        if model_version:
            query += ' AND p.model_version = ?'
            params.append(model_version)
        
        if horizon_days is not None:
            query += ' AND p.horizon_days = ?'
            params.append(horizon_days)
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        
        if len(results) < min_validations:
            return {
                'total_validations': len(results),
                'status': 'insufficient_data',
                'message': f'Need at least {min_validations} validations, got {len(results)}'
            }
        
        # Calculate metrics
        price_errors = [r[0] for r in results]
        price_error_pcts = [r[1] for r in results]
        direction_correct = [r[2] for r in results]
        
        mean_error = sum(price_errors) / len(price_errors) if price_errors else 0
        mean_abs_error = sum(abs(e) for e in price_errors) / len(price_errors) if price_errors else 0
        rmse = (sum(e**2 for e in price_errors) / len(price_errors))**0.5 if price_errors else 0
        mean_abs_pct_error = sum(abs(p) for p in price_error_pcts) / len(price_error_pcts) if price_error_pcts else 0
        
        direction_accuracy = sum(direction_correct) / len(direction_correct) if direction_correct else 0
        
        return {
            'total_validations': len(results),
            'mean_error': mean_error,
            'mean_absolute_error': mean_abs_error,
            'rmse': rmse,
            'mean_absolute_percent_error': mean_abs_pct_error,
            'direction_accuracy': direction_accuracy,
            'direction_accuracy_pct': direction_accuracy * 100,
            'correct_predictions': sum(direction_correct),
            'total_predictions': len(direction_correct)
        }
    
    def get_recent_accuracy(self, days: int = 30) -> Dict[str, Any]:
        """Get accuracy metrics for recent validations"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cutoff_date = (datetime.utcnow() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        cursor.execute('''
            SELECT v.price_error, v.price_error_pct, v.direction_correct
            FROM validations v
            WHERE v.validated_at >= ?
        ''', (cutoff_date,))
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return {'total_validations': 0, 'status': 'no_data'}
        
        return self.calculate_accuracy_metrics()

# Global instance
prediction_tracker = PredictionTracker()

