"""
Enhanced Machine Learning Service for Financial Analyzer Pro
Advanced models, ensemble methods, and improved prediction accuracy
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Advanced ML imports with graceful fallbacks
try:
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    from scipy import stats
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

class EnhancedMLService:
    """Enhanced machine learning service with advanced models and techniques"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_performance = {}
        self.ensemble_weights = {}
        
    def create_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create advanced technical features for ML models"""
        df = data.copy()
        
        # Price-based features
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
        df['Price_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        df['Gap'] = df['Open'] - df['Close'].shift(1)
        df['Gap_Percent'] = df['Gap'] / df['Close'].shift(1)
        
        # Volume features
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        df['Volume_Price_Trend'] = df['Volume'] * df['Price_Change']
        
        # Moving averages with different periods
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()
            df[f'Price_vs_SMA_{period}'] = (df['Close'] - df[f'SMA_{period}']) / df[f'SMA_{period}']
            df[f'Price_vs_EMA_{period}'] = (df['Close'] - df[f'EMA_{period}']) / df[f'EMA_{period}']
        
        # Technical indicators
        df = self._add_technical_indicators(df)
        
        # Volatility features
        df = self._add_volatility_features(df)
        
        # Momentum features
        df = self._add_momentum_features(df)
        
        # Trend features
        df = self._add_trend_features(df)
        
        # Market microstructure features
        df = self._add_microstructure_features(df)
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators"""
        # RSI with multiple periods
        for period in [14, 21, 30]:
            df[f'RSI_{period}'] = self._calculate_rsi(df['Close'], period)
        
        # MACD variations
        df['MACD_12_26'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
        df['MACD_Signal_12_26'] = df['MACD_12_26'].ewm(span=9).mean()
        df['MACD_Histogram_12_26'] = df['MACD_12_26'] - df['MACD_Signal_12_26']
        
        # Bollinger Bands with multiple periods
        for period in [20, 50]:
            df[f'BB_Middle_{period}'] = df['Close'].rolling(window=period).mean()
            bb_std = df['Close'].rolling(window=period).std()
            df[f'BB_Upper_{period}'] = df[f'BB_Middle_{period}'] + (bb_std * 2)
            df[f'BB_Lower_{period}'] = df[f'BB_Middle_{period}'] - (bb_std * 2)
            df[f'BB_Width_{period}'] = (df[f'BB_Upper_{period}'] - df[f'BB_Lower_{period}']) / df[f'BB_Middle_{period}']
            df[f'BB_Position_{period}'] = (df['Close'] - df[f'BB_Lower_{period}']) / (df[f'BB_Upper_{period}'] - df[f'BB_Lower_{period}'])
        
        # Stochastic Oscillator
        for period in [14, 21]:
            df[f'Stoch_K_{period}'] = self._calculate_stochastic_k(df, period)
            df[f'Stoch_D_{period}'] = df[f'Stoch_K_{period}'].rolling(window=3).mean()
        
        # Williams %R
        for period in [14, 21]:
            df[f'Williams_R_{period}'] = self._calculate_williams_r(df, period)
        
        # Commodity Channel Index
        for period in [20, 50]:
            df[f'CCI_{period}'] = self._calculate_cci(df, period)
        
        # Average True Range
        df['ATR_14'] = self._calculate_atr(df, 14)
        df['ATR_21'] = self._calculate_atr(df, 21)
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features"""
        # Historical volatility
        for period in [5, 10, 20, 30]:
            df[f'Volatility_{period}'] = df['Price_Change'].rolling(window=period).std() * np.sqrt(252)
        
        # Parkinson volatility
        df['Parkinson_Vol'] = np.sqrt(1/(4*np.log(2)) * np.log(df['High']/df['Low'])**2).rolling(window=20).mean() * np.sqrt(252)
        
        # Garman-Klass volatility
        df['GK_Vol'] = np.sqrt(0.5 * np.log(df['High']/df['Low'])**2 - (2*np.log(2)-1) * np.log(df['Close']/df['Open'])**2).rolling(window=20).mean() * np.sqrt(252)
        
        # Volatility of volatility
        df['Vol_of_Vol'] = df['Volatility_20'].rolling(window=20).std()
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based features"""
        # Rate of Change
        for period in [5, 10, 20, 50]:
            df[f'ROC_{period}'] = ((df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)) * 100
        
        # Momentum
        for period in [5, 10, 20]:
            df[f'Momentum_{period}'] = df['Close'] - df['Close'].shift(period)
        
        # Price Rate of Change
        for period in [5, 10, 20]:
            df[f'PROC_{period}'] = df['Close'].pct_change(period)
        
        # Relative Strength Index momentum
        for period in [14, 21]:
            df[f'RSI_Momentum_{period}'] = df[f'RSI_{period}'].diff()
        
        return df
    
    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend-based features"""
        # Trend strength
        for period in [20, 50]:
            df[f'Trend_Strength_{period}'] = (df['Close'] - df[f'SMA_{period}']) / df[f'SMA_{period}']
        
        # Trend direction
        for period in [5, 10, 20]:
            df[f'Trend_Direction_{period}'] = np.where(df['Close'] > df['Close'].shift(period), 1, -1)
        
        # Moving average convergence
        df['MA_Convergence'] = df['SMA_20'] - df['SMA_50']
        df['MA_Convergence_Percent'] = df['MA_Convergence'] / df['SMA_50']
        
        # Price position in range
        for period in [20, 50]:
            rolling_high = df['High'].rolling(window=period).max()
            rolling_low = df['Low'].rolling(window=period).min()
            df[f'Price_Position_{period}'] = (df['Close'] - rolling_low) / (rolling_high - rolling_low)
        
        return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features"""
        # Bid-ask spread proxy
        df['Spread_Proxy'] = (df['High'] - df['Low']) / df['Close']
        
        # Price impact
        df['Price_Impact'] = df['Price_Change'] / df['Volume_Change'].replace(0, np.nan)
        
        # Volume-weighted average price approximation
        df['VWAP_20'] = (df['Close'] * df['Volume']).rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()
        df['Price_vs_VWAP'] = (df['Close'] - df['VWAP_20']) / df['VWAP_20']
        
        # Order flow imbalance proxy
        df['OFI_Proxy'] = (df['Close'] - df['Open']) / (df['High'] - df['Low']).replace(0, np.nan)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_stochastic_k(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Stochastic %K"""
        lowest_low = df['Low'].rolling(window=period).min()
        highest_high = df['High'].rolling(window=period).max()
        return 100 * (df['Close'] - lowest_low) / (highest_high - lowest_low)
    
    def _calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        highest_high = df['High'].rolling(window=period).max()
        lowest_low = df['Low'].rolling(window=period).min()
        return -100 * (highest_high - df['Close']) / (highest_high - lowest_low)
    
    def _calculate_cci(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index"""
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        return (tp - sma_tp) / (0.015 * mad)
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift(1))
        low_close = np.abs(df['Low'] - df['Close'].shift(1))
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(window=period).mean()
    
    def prepare_ml_data(self, data: pd.DataFrame, target_col: str = 'Close', 
                       lookback: int = 60, forecast_horizon: int = 1) -> tuple:
        """Prepare data for ML models with time series structure"""
        # Create advanced features
        df_features = self.create_advanced_features(data)
        
        # Select numeric features
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col != target_col and not col.startswith('Target')]
        
        # Remove columns with too many NaN values
        valid_cols = []
        for col in feature_cols:
            if df_features[col].notna().sum() > len(df_features) * 0.7:  # At least 70% non-NaN
                valid_cols.append(col)
        
        if len(valid_cols) < 5:
            st.warning(f"Only {len(valid_cols)} valid features found. Consider using more data.")
            return None, None, None, None
        
        # Prepare features and target
        X = df_features[valid_cols].fillna(method='ffill').fillna(method='bfill')
        y = df_features[target_col].shift(-forecast_horizon)  # Predict future price
        
        # Remove rows with NaN target
        valid_idx = y.notna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        # Create time series sequences for LSTM/GRU
        X_sequences, y_sequences = self._create_sequences(X.values, y.values, lookback)
        
        return X, y, X_sequences, y_sequences
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray, lookback: int) -> tuple:
        """Create sequences for time series models"""
        X_seq, y_seq = [], []
        
        for i in range(lookback, len(X)):
            X_seq.append(X[i-lookback:i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    def train_ensemble_models(self, X: pd.DataFrame, y: pd.Series, 
                            X_sequences: np.ndarray = None, y_sequences: np.ndarray = None) -> dict:
        """Train ensemble of advanced ML models"""
        if not SKLEARN_AVAILABLE:
            return {"error": "Scikit-learn not available"}
        
        results = {}
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train traditional ML models
        traditional_models = self._train_traditional_models(X_train_scaled, y_train, X_test_scaled, y_test)
        results.update(traditional_models)
        
        # Train gradient boosting models
        if XGBOOST_AVAILABLE or LIGHTGBM_AVAILABLE:
            gb_models = self._train_gradient_boosting_models(X_train, y_train, X_test, y_test)
            results.update(gb_models)
        
        # Train neural networks
        if TENSORFLOW_AVAILABLE and X_sequences is not None:
            nn_models = self._train_neural_networks(X_sequences, y_sequences)
            results.update(nn_models)
        
        # Create ensemble
        ensemble_result = self._create_ensemble(results, X_test_scaled, y_test)
        results['ensemble'] = ensemble_result
        
        return results
    
    def _train_traditional_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                                X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Train traditional ML models"""
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'MLP Regressor': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                mape = mean_absolute_percentage_error(y_test, y_pred)
                
                results[name] = {
                    'model': model,
                    'predictions': y_pred,
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'mape': mape,
                    'feature_importance': getattr(model, 'feature_importances_', None)
                }
                
            except Exception as e:
                st.warning(f"Error training {name}: {str(e)}")
                continue
        
        return results
    
    def _train_gradient_boosting_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                                      X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Train gradient boosting models"""
        results = {}
        
        if XGBOOST_AVAILABLE:
            try:
                # XGBoost
                xgb_model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1
                )
                xgb_model.fit(X_train, y_train)
                y_pred = xgb_model.predict(X_test)
                
                results['XGBoost'] = {
                    'model': xgb_model,
                    'predictions': y_pred,
                    'mse': mean_squared_error(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'mae': mean_absolute_error(y_test, y_pred),
                    'r2': r2_score(y_test, y_pred),
                    'mape': mean_absolute_percentage_error(y_test, y_pred),
                    'feature_importance': xgb_model.feature_importances_
                }
            except Exception as e:
                st.warning(f"Error training XGBoost: {str(e)}")
        
        if LIGHTGBM_AVAILABLE:
            try:
                # LightGBM
                lgb_model = lgb.LGBMRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )
                lgb_model.fit(X_train, y_train)
                y_pred = lgb_model.predict(X_test)
                
                results['LightGBM'] = {
                    'model': lgb_model,
                    'predictions': y_pred,
                    'mse': mean_squared_error(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'mae': mean_absolute_error(y_test, y_pred),
                    'r2': r2_score(y_test, y_pred),
                    'mape': mean_absolute_percentage_error(y_test, y_pred),
                    'feature_importance': lgb_model.feature_importances_
                }
            except Exception as e:
                st.warning(f"Error training LightGBM: {str(e)}")
        
        return results
    
    def _train_neural_networks(self, X_sequences: np.ndarray, y_sequences: np.ndarray) -> dict:
        """Train neural network models"""
        results = {}
        
        if not TENSORFLOW_AVAILABLE or X_sequences is None:
            return results
        
        # Split sequences
        split_idx = int(len(X_sequences) * 0.8)
        X_train_seq, X_test_seq = X_sequences[:split_idx], X_sequences[split_idx:]
        y_train_seq, y_test_seq = y_sequences[:split_idx], y_sequences[split_idx:]
        
        # LSTM Model
        try:
            lstm_model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X_sequences.shape[1], X_sequences.shape[2])),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            # Callbacks
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
            
            # Train
            history = lstm_model.fit(
                X_train_seq, y_train_seq,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            # Predictions
            y_pred = lstm_model.predict(X_test_seq).flatten()
            
            results['LSTM'] = {
                'model': lstm_model,
                'predictions': y_pred,
                'mse': mean_squared_error(y_test_seq, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test_seq, y_pred)),
                'mae': mean_absolute_error(y_test_seq, y_pred),
                'r2': r2_score(y_test_seq, y_pred),
                'mape': mean_absolute_percentage_error(y_test_seq, y_pred),
                'history': history.history
            }
            
        except Exception as e:
            st.warning(f"Error training LSTM: {str(e)}")
        
        # GRU Model
        try:
            gru_model = Sequential([
                GRU(50, return_sequences=True, input_shape=(X_sequences.shape[1], X_sequences.shape[2])),
                Dropout(0.2),
                GRU(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            gru_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            # Train
            history = gru_model.fit(
                X_train_seq, y_train_seq,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            # Predictions
            y_pred = gru_model.predict(X_test_seq).flatten()
            
            results['GRU'] = {
                'model': gru_model,
                'predictions': y_pred,
                'mse': mean_squared_error(y_test_seq, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test_seq, y_pred)),
                'mae': mean_absolute_error(y_test_seq, y_pred),
                'r2': r2_score(y_test_seq, y_pred),
                'mape': mean_absolute_percentage_error(y_test_seq, y_pred),
                'history': history.history
            }
            
        except Exception as e:
            st.warning(f"Error training GRU: {str(e)}")
        
        return results
    
    def _create_ensemble(self, model_results: dict, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Create ensemble of all trained models"""
        # Get predictions from all models
        predictions = {}
        weights = {}
        
        for name, result in model_results.items():
            if name != 'ensemble' and 'predictions' in result:
                predictions[name] = result['predictions']
                # Weight by R² score (higher is better)
                weights[name] = max(0, result['r2'])
        
        if not predictions:
            return {"error": "No models available for ensemble"}
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        else:
            # Equal weights if all R² are negative
            weights = {k: 1/len(weights) for k in weights.keys()}
        
        # Create weighted ensemble prediction
        ensemble_pred = np.zeros_like(y_test)
        for name, pred in predictions.items():
            ensemble_pred += weights[name] * pred
        
        # Calculate ensemble metrics
        mse = mean_squared_error(y_test, ensemble_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, ensemble_pred)
        r2 = r2_score(y_test, ensemble_pred)
        mape = mean_absolute_percentage_error(y_test, ensemble_pred)
        
        return {
            'predictions': ensemble_pred,
            'weights': weights,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
    
    def get_feature_importance(self, model_results: dict, feature_names: list) -> pd.DataFrame:
        """Get feature importance from all models"""
        importance_data = []
        
        for model_name, result in model_results.items():
            if 'feature_importance' in result and result['feature_importance'] is not None:
                importance = result['feature_importance']
                for i, (feature, imp) in enumerate(zip(feature_names, importance)):
                    importance_data.append({
                        'Model': model_name,
                        'Feature': feature,
                        'Importance': imp
                    })
        
        if not importance_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(importance_data)
        return df.groupby('Feature')['Importance'].mean().sort_values(ascending=False).reset_index()
    
    def display_model_comparison(self, model_results: dict):
        """Display model comparison dashboard"""
        if not model_results or 'error' in model_results:
            st.error("No model results to display")
            return
        
        # Create comparison DataFrame
        comparison_data = []
        for name, result in model_results.items():
            if name != 'ensemble' and 'r2' in result:
                comparison_data.append({
                    'Model': name,
                    'R² Score': f"{result['r2']:.4f}",
                    'RMSE': f"{result['rmse']:.2f}",
                    'MAE': f"{result['mae']:.2f}",
                    'MAPE': f"{result['mape']:.2f}%"
                })
        
        if not comparison_data:
            st.warning("No valid model results found")
            return
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Sort by R² score
        df_comparison['R² Score'] = df_comparison['R² Score'].astype(float)
        df_comparison = df_comparison.sort_values('R² Score', ascending=False)
        
        # Display comparison table
        st.subheader("📊 Model Performance Comparison")
        st.dataframe(df_comparison, use_container_width=True)
        
        # Best model
        best_model = df_comparison.iloc[0]
        st.success(f"🏆 Best Model: {best_model['Model']} (R² = {best_model['R² Score']})")
        
        # Performance visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # R² Score comparison
            fig_r2 = px.bar(
                df_comparison, 
                x='Model', 
                y='R² Score',
                title="R² Score Comparison",
                color='R² Score',
                color_continuous_scale='Viridis'
            )
            fig_r2.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_r2, use_container_width=True)
        
        with col2:
            # RMSE comparison
            fig_rmse = px.bar(
                df_comparison, 
                x='Model', 
                y='RMSE',
                title="RMSE Comparison (Lower is Better)",
                color='RMSE',
                color_continuous_scale='Reds'
            )
            fig_rmse.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_rmse, use_container_width=True)

# Global instance
enhanced_ml_service = EnhancedMLService()
