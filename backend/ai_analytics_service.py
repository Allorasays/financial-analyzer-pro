#!/usr/bin/env python3
"""
AI Analytics Service for advanced financial analysis
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from cache_service import CacheService
from config import settings

logger = logging.getLogger(__name__)

class AIAnalyticsService:
    def __init__(self):
        self.cache_service = CacheService()
        self.models = {}
        self.scalers = {}
        self.background_task = None
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models for different prediction tasks"""
        try:
            # Price prediction models
            self.models['price_prediction'] = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=10
            )
            
            # Trend analysis models
            self.models['trend_classification'] = RandomForestRegressor(
                n_estimators=50,
                random_state=42,
                max_depth=5
            )
            
            # Risk assessment models
            self.models['risk_assessment'] = RandomForestRegressor(
                n_estimators=75,
                random_state=42,
                max_depth=8
            )
            
            # Initialize scalers
            self.scalers['price_prediction'] = StandardScaler()
            self.scalers['trend_classification'] = StandardScaler()
            self.scalers['risk_assessment'] = StandardScaler()
            
            logger.info("AI Analytics models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize AI models: {e}")
    
    async def start_background_tasks(self):
        """Start background AI analysis tasks"""
        if self.background_task is None:
            self.background_task = asyncio.create_task(self._background_analysis())
            logger.info("AI Analytics background tasks started")
    
    async def stop_background_tasks(self):
        """Stop background AI analysis tasks"""
        if self.background_task:
            self.background_task.cancel()
            try:
                await self.background_task
            except asyncio.CancelledError:
                pass
            self.background_task = None
            logger.info("AI Analytics background tasks stopped")
    
    async def predict_price(self, symbol: str, days_ahead: int = 5) -> Dict:
        """Predict future price for a symbol"""
        try:
            cache_key = f"price_prediction_{symbol}_{days_ahead}"
            cached_prediction = await self.cache_service.get(cache_key)
            if cached_prediction:
                return cached_prediction
            
            # Get historical data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1y")
            
            if hist is None or hist.empty or len(hist) < 30:
                return {"error": "Insufficient data for prediction"}
            
            # Prepare features
            features = self._extract_price_features(hist)
            if features is None:
                return {"error": "Failed to extract features"}
            
            # Train model
            X = features[:-days_ahead]
            y = hist['Close'].iloc[days_ahead:].values
            
            if len(X) < 10:
                return {"error": "Insufficient data for training"}
            
            # Scale features
            X_scaled = self.scalers['price_prediction'].fit_transform(X)
            
            # Train model
            self.models['price_prediction'].fit(X_scaled, y)
            
            # Make prediction
            last_features = features[-1:].reshape(1, -1)
            last_features_scaled = self.scalers['price_prediction'].transform(last_features)
            
            prediction = self.models['price_prediction'].predict(last_features_scaled)[0]
            current_price = hist['Close'].iloc[-1]
            
            # Calculate confidence
            confidence = self._calculate_prediction_confidence(X_scaled, y)
            
            result = {
                "symbol": symbol,
                "current_price": float(current_price),
                "predicted_price": float(prediction),
                "price_change": float(prediction - current_price),
                "price_change_percent": float((prediction - current_price) / current_price * 100),
                "confidence": float(confidence),
                "days_ahead": days_ahead,
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache for 1 hour
            await self.cache_service.set(cache_key, result, 3600)
            
            return result
            
        except Exception as e:
            logger.error(f"Price prediction failed for {symbol}: {e}")
            return {"error": str(e)}
    
    async def analyze_trend(self, symbol: str) -> Dict:
        """Analyze market trend for a symbol"""
        try:
            cache_key = f"trend_analysis_{symbol}"
            cached_analysis = await self.cache_service.get(cache_key)
            if cached_analysis:
                return cached_analysis
            
            # Get historical data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="6mo")
            
            if hist is None or hist.empty or len(hist) < 30:
                return {"error": "Insufficient data for trend analysis"}
            
            # Calculate technical indicators
            indicators = self._calculate_technical_indicators(hist)
            
            # Analyze trend
            trend_analysis = self._analyze_trend_patterns(hist, indicators)
            
            # Generate trend signals
            signals = self._generate_trend_signals(hist, indicators)
            
            result = {
                "symbol": symbol,
                "trend_direction": trend_analysis["direction"],
                "trend_strength": trend_analysis["strength"],
                "trend_confidence": trend_analysis["confidence"],
                "support_level": trend_analysis["support"],
                "resistance_level": trend_analysis["resistance"],
                "signals": signals,
                "indicators": indicators,
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache for 30 minutes
            await self.cache_service.set(cache_key, result, 1800)
            
            return result
            
        except Exception as e:
            logger.error(f"Trend analysis failed for {symbol}: {e}")
            return {"error": str(e)}
    
    async def assess_risk(self, symbol: str) -> Dict:
        """Assess risk level for a symbol"""
        try:
            cache_key = f"risk_assessment_{symbol}"
            cached_assessment = await self.cache_service.get(cache_key)
            if cached_assessment:
                return cached_assessment
            
            # Get historical data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1y")
            
            if hist is None or hist.empty or len(hist) < 30:
                return {"error": "Insufficient data for risk assessment"}
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(hist)
            
            # Assess risk level
            risk_level = self._assess_risk_level(risk_metrics)
            
            # Generate risk recommendations
            recommendations = self._generate_risk_recommendations(risk_level, risk_metrics)
            
            result = {
                "symbol": symbol,
                "risk_level": risk_level["level"],
                "risk_score": risk_level["score"],
                "volatility": risk_metrics["volatility"],
                "beta": risk_metrics["beta"],
                "sharpe_ratio": risk_metrics["sharpe_ratio"],
                "max_drawdown": risk_metrics["max_drawdown"],
                "var_95": risk_metrics["var_95"],
                "recommendations": recommendations,
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache for 1 hour
            await self.cache_service.set(cache_key, result, 3600)
            
            return result
            
        except Exception as e:
            logger.error(f"Risk assessment failed for {symbol}: {e}")
            return {"error": str(e)}
    
    async def analyze_portfolio_risk(self, portfolio_data: Dict) -> Dict:
        """Analyze portfolio risk and diversification"""
        try:
            cache_key = f"portfolio_risk_{hash(str(portfolio_data))}"
            cached_analysis = await self.cache_service.get(cache_key)
            if cached_analysis:
                return cached_analysis
            
            positions = portfolio_data.get("positions", [])
            if not positions:
                return {"error": "No positions in portfolio"}
            
            # Calculate portfolio metrics
            portfolio_metrics = self._calculate_portfolio_metrics(positions)
            
            # Assess diversification
            diversification = self._assess_diversification(positions)
            
            # Calculate portfolio risk
            portfolio_risk = self._calculate_portfolio_risk(positions)
            
            # Generate recommendations
            recommendations = self._generate_portfolio_recommendations(
                portfolio_metrics, diversification, portfolio_risk
            )
            
            result = {
                "portfolio_risk_score": portfolio_risk["score"],
                "diversification_score": diversification["score"],
                "concentration_risk": diversification["concentration_risk"],
                "sector_diversification": diversification["sector_diversification"],
                "geographic_diversification": diversification["geographic_diversification"],
                "portfolio_volatility": portfolio_metrics["volatility"],
                "portfolio_beta": portfolio_metrics["beta"],
                "portfolio_sharpe": portfolio_metrics["sharpe_ratio"],
                "recommendations": recommendations,
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache for 1 hour
            await self.cache_service.set(cache_key, result, 3600)
            
            return result
            
        except Exception as e:
            logger.error(f"Portfolio risk analysis failed: {e}")
            return {"error": str(e)}
    
    def _extract_price_features(self, hist: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract features for price prediction"""
        try:
            features = []
            
            # Price features
            features.append(hist['Close'].iloc[-1])
            features.append(hist['High'].iloc[-1])
            features.append(hist['Low'].iloc[-1])
            features.append(hist['Volume'].iloc[-1])
            
            # Moving averages
            features.append(hist['Close'].rolling(5).mean().iloc[-1])
            features.append(hist['Close'].rolling(10).mean().iloc[-1])
            features.append(hist['Close'].rolling(20).mean().iloc[-1])
            
            # Price changes
            features.append(hist['Close'].pct_change().iloc[-1])
            features.append(hist['Close'].pct_change(5).iloc[-1])
            features.append(hist['Close'].pct_change(10).iloc[-1])
            
            # Volatility
            features.append(hist['Close'].rolling(10).std().iloc[-1])
            features.append(hist['Close'].rolling(20).std().iloc[-1])
            
            # RSI
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            features.append(rsi.iloc[-1])
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None
    
    def _calculate_technical_indicators(self, hist: pd.DataFrame) -> Dict:
        """Calculate technical indicators"""
        try:
            indicators = {}
            
            # Moving averages
            indicators['sma_5'] = hist['Close'].rolling(5).mean().iloc[-1]
            indicators['sma_10'] = hist['Close'].rolling(10).mean().iloc[-1]
            indicators['sma_20'] = hist['Close'].rolling(20).mean().iloc[-1]
            indicators['sma_50'] = hist['Close'].rolling(50).mean().iloc[-1]
            
            # RSI
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi'] = (100 - (100 / (1 + rs))).iloc[-1]
            
            # MACD
            ema_12 = hist['Close'].ewm(span=12).mean()
            ema_26 = hist['Close'].ewm(span=26).mean()
            indicators['macd'] = (ema_12 - ema_26).iloc[-1]
            indicators['macd_signal'] = (ema_12 - ema_26).ewm(span=9).mean().iloc[-1]
            
            # Bollinger Bands
            sma_20 = hist['Close'].rolling(20).mean()
            std_20 = hist['Close'].rolling(20).std()
            indicators['bb_upper'] = (sma_20 + (std_20 * 2)).iloc[-1]
            indicators['bb_lower'] = (sma_20 - (std_20 * 2)).iloc[-1]
            indicators['bb_middle'] = sma_20.iloc[-1]
            
            return indicators
            
        except Exception as e:
            logger.error(f"Technical indicators calculation failed: {e}")
            return {}
    
    def _analyze_trend_patterns(self, hist: pd.DataFrame, indicators: Dict) -> Dict:
        """Analyze trend patterns"""
        try:
            current_price = hist['Close'].iloc[-1]
            sma_20 = indicators.get('sma_20', current_price)
            sma_50 = indicators.get('sma_50', current_price)
            rsi = indicators.get('rsi', 50)
            
            # Determine trend direction
            if current_price > sma_20 > sma_50:
                direction = "Bullish"
                strength = min(100, (current_price - sma_50) / sma_50 * 100)
            elif current_price < sma_20 < sma_50:
                direction = "Bearish"
                strength = min(100, (sma_50 - current_price) / sma_50 * 100)
            else:
                direction = "Sideways"
                strength = 0
            
            # Calculate confidence
            confidence = 0
            if direction == "Bullish":
                if rsi < 70:  # Not overbought
                    confidence += 30
                if current_price > sma_20:
                    confidence += 20
                if sma_20 > sma_50:
                    confidence += 20
            elif direction == "Bearish":
                if rsi > 30:  # Not oversold
                    confidence += 30
                if current_price < sma_20:
                    confidence += 20
                if sma_20 < sma_50:
                    confidence += 20
            
            # Calculate support and resistance
            support = hist['Low'].rolling(20).min().iloc[-1]
            resistance = hist['High'].rolling(20).max().iloc[-1]
            
            return {
                "direction": direction,
                "strength": strength,
                "confidence": confidence,
                "support": support,
                "resistance": resistance
            }
            
        except Exception as e:
            logger.error(f"Trend pattern analysis failed: {e}")
            return {"direction": "Unknown", "strength": 0, "confidence": 0}
    
    def _generate_trend_signals(self, hist: pd.DataFrame, indicators: Dict) -> List[Dict]:
        """Generate trend signals"""
        try:
            signals = []
            current_price = hist['Close'].iloc[-1]
            rsi = indicators.get('rsi', 50)
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            
            # RSI signals
            if rsi > 70:
                signals.append({
                    "type": "RSI Overbought",
                    "signal": "SELL",
                    "strength": "Strong",
                    "description": "RSI indicates overbought conditions"
                })
            elif rsi < 30:
                signals.append({
                    "type": "RSI Oversold",
                    "signal": "BUY",
                    "strength": "Strong",
                    "description": "RSI indicates oversold conditions"
                })
            
            # MACD signals
            if macd > macd_signal:
                signals.append({
                    "type": "MACD Bullish",
                    "signal": "BUY",
                    "strength": "Medium",
                    "description": "MACD line above signal line"
                })
            elif macd < macd_signal:
                signals.append({
                    "type": "MACD Bearish",
                    "signal": "SELL",
                    "strength": "Medium",
                    "description": "MACD line below signal line"
                })
            
            # Bollinger Bands signals
            bb_upper = indicators.get('bb_upper', current_price)
            bb_lower = indicators.get('bb_lower', current_price)
            
            if current_price > bb_upper:
                signals.append({
                    "type": "Bollinger Upper",
                    "signal": "SELL",
                    "strength": "Medium",
                    "description": "Price above upper Bollinger Band"
                })
            elif current_price < bb_lower:
                signals.append({
                    "type": "Bollinger Lower",
                    "signal": "BUY",
                    "strength": "Medium",
                    "description": "Price below lower Bollinger Band"
                })
            
            return signals
            
        except Exception as e:
            logger.error(f"Trend signal generation failed: {e}")
            return []
    
    def _calculate_risk_metrics(self, hist: pd.DataFrame) -> Dict:
        """Calculate risk metrics"""
        try:
            returns = hist['Close'].pct_change().dropna()
            
            # Volatility (annualized)
            volatility = returns.std() * np.sqrt(252)
            
            # Beta (simplified - would need market data for accurate calculation)
            beta = 1.0  # Placeholder
            
            # Sharpe ratio (simplified)
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            
            # Maximum drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Value at Risk (95%)
            var_95 = np.percentile(returns, 5)
            
            return {
                "volatility": volatility,
                "beta": beta,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "var_95": var_95
            }
            
        except Exception as e:
            logger.error(f"Risk metrics calculation failed: {e}")
            return {}
    
    def _assess_risk_level(self, risk_metrics: Dict) -> Dict:
        """Assess risk level based on metrics"""
        try:
            volatility = risk_metrics.get("volatility", 0)
            max_drawdown = abs(risk_metrics.get("max_drawdown", 0))
            var_95 = abs(risk_metrics.get("var_95", 0))
            
            # Calculate risk score (0-100)
            risk_score = 0
            
            # Volatility component (40% weight)
            if volatility > 0.3:
                risk_score += 40
            elif volatility > 0.2:
                risk_score += 30
            elif volatility > 0.1:
                risk_score += 20
            else:
                risk_score += 10
            
            # Drawdown component (30% weight)
            if max_drawdown > 0.5:
                risk_score += 30
            elif max_drawdown > 0.3:
                risk_score += 20
            elif max_drawdown > 0.1:
                risk_score += 10
            
            # VaR component (30% weight)
            if var_95 > 0.05:
                risk_score += 30
            elif var_95 > 0.03:
                risk_score += 20
            elif var_95 > 0.01:
                risk_score += 10
            
            # Determine risk level
            if risk_score >= 80:
                level = "Very High"
            elif risk_score >= 60:
                level = "High"
            elif risk_score >= 40:
                level = "Medium"
            elif risk_score >= 20:
                level = "Low"
            else:
                level = "Very Low"
            
            return {
                "level": level,
                "score": risk_score
            }
            
        except Exception as e:
            logger.error(f"Risk level assessment failed: {e}")
            return {"level": "Unknown", "score": 0}
    
    def _generate_risk_recommendations(self, risk_level: Dict, risk_metrics: Dict) -> List[str]:
        """Generate risk recommendations"""
        try:
            recommendations = []
            level = risk_level.get("level", "Unknown")
            
            if level in ["High", "Very High"]:
                recommendations.append("Consider reducing position size")
                recommendations.append("Implement stop-loss orders")
                recommendations.append("Diversify across different sectors")
            elif level == "Medium":
                recommendations.append("Monitor position closely")
                recommendations.append("Consider partial profit-taking")
            elif level in ["Low", "Very Low"]:
                recommendations.append("Suitable for conservative investors")
                recommendations.append("Consider increasing position size")
            
            # Specific recommendations based on metrics
            volatility = risk_metrics.get("volatility", 0)
            if volatility > 0.3:
                recommendations.append("High volatility detected - use caution")
            
            max_drawdown = abs(risk_metrics.get("max_drawdown", 0))
            if max_drawdown > 0.3:
                recommendations.append("Significant drawdown risk - consider hedging")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Risk recommendation generation failed: {e}")
            return []
    
    def _calculate_portfolio_metrics(self, positions: List[Dict]) -> Dict:
        """Calculate portfolio-level metrics"""
        try:
            if not positions:
                return {}
            
            # Calculate weighted metrics
            total_value = sum(pos.get("current_value", 0) for pos in positions)
            
            # Portfolio volatility (simplified)
            portfolio_volatility = 0.15  # Placeholder
            
            # Portfolio beta (simplified)
            portfolio_beta = 1.0  # Placeholder
            
            # Portfolio Sharpe ratio (simplified)
            portfolio_sharpe = 0.5  # Placeholder
            
            return {
                "volatility": portfolio_volatility,
                "beta": portfolio_beta,
                "sharpe_ratio": portfolio_sharpe
            }
            
        except Exception as e:
            logger.error(f"Portfolio metrics calculation failed: {e}")
            return {}
    
    def _assess_diversification(self, positions: List[Dict]) -> Dict:
        """Assess portfolio diversification"""
        try:
            if not positions:
                return {"score": 0, "concentration_risk": "High"}
            
            total_value = sum(pos.get("current_value", 0) for pos in positions)
            
            # Calculate concentration risk
            weights = [pos.get("current_value", 0) / total_value for pos in positions]
            max_weight = max(weights) if weights else 0
            
            if max_weight > 0.4:
                concentration_risk = "High"
                score = 20
            elif max_weight > 0.25:
                concentration_risk = "Medium"
                score = 50
            else:
                concentration_risk = "Low"
                score = 80
            
            # Sector diversification (simplified)
            sector_diversification = "Good" if len(positions) > 5 else "Limited"
            
            # Geographic diversification (simplified)
            geographic_diversification = "Good" if len(positions) > 3 else "Limited"
            
            return {
                "score": score,
                "concentration_risk": concentration_risk,
                "sector_diversification": sector_diversification,
                "geographic_diversification": geographic_diversification
            }
            
        except Exception as e:
            logger.error(f"Diversification assessment failed: {e}")
            return {"score": 0, "concentration_risk": "High"}
    
    def _calculate_portfolio_risk(self, positions: List[Dict]) -> Dict:
        """Calculate portfolio risk score"""
        try:
            if not positions:
                return {"score": 0}
            
            # Simplified portfolio risk calculation
            diversification = self._assess_diversification(positions)
            portfolio_metrics = self._calculate_portfolio_metrics(positions)
            
            # Calculate risk score
            risk_score = 0
            
            # Diversification component
            risk_score += (100 - diversification["score"]) * 0.4
            
            # Volatility component
            volatility = portfolio_metrics.get("volatility", 0.15)
            risk_score += volatility * 100 * 0.3
            
            # Concentration component
            concentration_risk = diversification["concentration_risk"]
            if concentration_risk == "High":
                risk_score += 30
            elif concentration_risk == "Medium":
                risk_score += 15
            
            return {"score": min(100, risk_score)}
            
        except Exception as e:
            logger.error(f"Portfolio risk calculation failed: {e}")
            return {"score": 0}
    
    def _generate_portfolio_recommendations(self, portfolio_metrics: Dict, 
                                          diversification: Dict, portfolio_risk: Dict) -> List[str]:
        """Generate portfolio recommendations"""
        try:
            recommendations = []
            
            # Diversification recommendations
            if diversification["score"] < 50:
                recommendations.append("Increase portfolio diversification")
                recommendations.append("Consider adding positions in different sectors")
            
            # Concentration recommendations
            if diversification["concentration_risk"] == "High":
                recommendations.append("Reduce concentration in largest positions")
                recommendations.append("Consider rebalancing portfolio")
            
            # Risk recommendations
            risk_score = portfolio_risk.get("score", 0)
            if risk_score > 70:
                recommendations.append("Portfolio risk is high - consider reducing exposure")
                recommendations.append("Implement risk management strategies")
            elif risk_score < 30:
                recommendations.append("Portfolio risk is low - consider increasing exposure")
                recommendations.append("Opportunity for higher returns")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Portfolio recommendation generation failed: {e}")
            return []
    
    def _calculate_prediction_confidence(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate prediction confidence"""
        try:
            if len(X) < 10:
                return 0.5
            
            # Use R² score as confidence measure
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            
            # Convert R² to confidence (0-1)
            confidence = max(0, min(1, r2))
            return confidence
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.5
    
    async def _background_analysis(self):
        """Background task for continuous analysis"""
        while True:
            try:
                # This would perform continuous analysis on popular symbols
                # For now, we'll just log that the task is running
                logger.debug("AI Analytics background task running")
                await asyncio.sleep(3600)  # Run every hour
                
            except Exception as e:
                logger.error(f"Background analysis error: {e}")
                await asyncio.sleep(3600)
    
    async def get_ai_stats(self) -> Dict:
        """Get AI analytics service statistics"""
        try:
            return {
                "models_initialized": len(self.models),
                "scalers_initialized": len(self.scalers),
                "background_task_running": self.background_task is not None,
                "cache_hits": 0,  # Would track actual cache hits
                "predictions_made": 0,  # Would track actual predictions
                "analyses_completed": 0  # Would track actual analyses
            }
            
        except Exception as e:
            logger.error(f"Failed to get AI stats: {e}")
            return {"error": str(e)}
