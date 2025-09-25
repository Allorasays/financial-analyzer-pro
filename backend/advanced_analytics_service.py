#!/usr/bin/env python3
"""
Advanced Analytics Service for comprehensive financial analysis
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import yfinance as yf
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from cache_service import CacheService
from config import settings

logger = logging.getLogger(__name__)

class AdvancedAnalyticsService:
    def __init__(self):
        self.cache_service = CacheService()
        self.background_task = None
        
        # Market benchmarks
        self.benchmarks = {
            'SPY': 'S&P 500',
            'QQQ': 'NASDAQ',
            'IWM': 'Russell 2000',
            'EFA': 'EAFE',
            'VTI': 'Total Stock Market'
        }
    
    async def start_background_tasks(self):
        """Start background analytics tasks"""
        if self.background_task is None:
            self.background_task = asyncio.create_task(self._background_analytics())
            logger.info("Advanced Analytics background tasks started")
    
    async def stop_background_tasks(self):
        """Stop background analytics tasks"""
        if self.background_task:
            self.background_task.cancel()
            try:
                await self.background_task
            except asyncio.CancelledError:
                pass
            self.background_task = None
            logger.info("Advanced Analytics background tasks stopped")
    
    async def analyze_portfolio_performance(self, portfolio_data: Dict) -> Dict:
        """Comprehensive portfolio performance analysis"""
        try:
            cache_key = f"portfolio_performance_{hash(str(portfolio_data))}"
            cached_analysis = await self.cache_service.get(cache_key)
            if cached_analysis:
                return cached_analysis
            
            positions = portfolio_data.get("positions", [])
            if not positions:
                return {"error": "No positions in portfolio"}
            
            # Calculate performance metrics
            performance_metrics = await self._calculate_performance_metrics(positions)
            
            # Benchmark comparison
            benchmark_comparison = await self._compare_to_benchmarks(positions)
            
            # Risk analysis
            risk_analysis = await self._analyze_portfolio_risk(positions)
            
            # Attribution analysis
            attribution_analysis = await self._analyze_attribution(positions)
            
            # Generate recommendations
            recommendations = await self._generate_performance_recommendations(
                performance_metrics, benchmark_comparison, risk_analysis
            )
            
            result = {
                "performance_metrics": performance_metrics,
                "benchmark_comparison": benchmark_comparison,
                "risk_analysis": risk_analysis,
                "attribution_analysis": attribution_analysis,
                "recommendations": recommendations,
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache for 1 hour
            await self.cache_service.set(cache_key, result, 3600)
            
            return result
            
        except Exception as e:
            logger.error(f"Portfolio performance analysis failed: {e}")
            return {"error": str(e)}
    
    async def analyze_market_correlation(self, symbols: List[str]) -> Dict:
        """Analyze correlation between symbols"""
        try:
            cache_key = f"correlation_{'_'.join(sorted(symbols))}"
            cached_analysis = await self.cache_service.get(cache_key)
            if cached_analysis:
                return cached_analysis
            
            # Get historical data for all symbols
            price_data = {}
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1y")
                if hist is not None and not hist.empty:
                    price_data[symbol] = hist['Close']
            
            if len(price_data) < 2:
                return {"error": "Insufficient data for correlation analysis"}
            
            # Calculate correlation matrix
            correlation_matrix = self._calculate_correlation_matrix(price_data)
            
            # Find highly correlated pairs
            high_correlations = self._find_high_correlations(correlation_matrix, symbols)
            
            # Analyze correlation clusters
            correlation_clusters = self._analyze_correlation_clusters(correlation_matrix, symbols)
            
            # Generate insights
            insights = self._generate_correlation_insights(high_correlations, correlation_clusters)
            
            result = {
                "correlation_matrix": correlation_matrix,
                "high_correlations": high_correlations,
                "correlation_clusters": correlation_clusters,
                "insights": insights,
                "symbols": symbols,
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache for 2 hours
            await self.cache_service.set(cache_key, result, 7200)
            
            return result
            
        except Exception as e:
            logger.error(f"Market correlation analysis failed: {e}")
            return {"error": str(e)}
    
    async def analyze_sector_rotation(self) -> Dict:
        """Analyze sector rotation patterns"""
        try:
            cache_key = "sector_rotation"
            cached_analysis = await self.cache_service.get(cache_key)
            if cached_analysis:
                return cached_analysis
            
            # Sector ETFs
            sector_etfs = {
                'XLK': 'Technology',
                'XLF': 'Financials',
                'XLV': 'Healthcare',
                'XLE': 'Energy',
                'XLI': 'Industrials',
                'XLY': 'Consumer Discretionary',
                'XLP': 'Consumer Staples',
                'XLU': 'Utilities',
                'XLB': 'Materials',
                'XLRE': 'Real Estate',
                'XLC': 'Communication Services'
            }
            
            # Get sector performance data
            sector_data = {}
            for etf, sector_name in sector_etfs.items():
                ticker = yf.Ticker(etf)
                hist = ticker.history(period="3mo")
                if hist is not None and not hist.empty:
                    sector_data[sector_name] = hist['Close']
            
            # Calculate sector performance
            sector_performance = self._calculate_sector_performance(sector_data)
            
            # Analyze rotation patterns
            rotation_patterns = self._analyze_rotation_patterns(sector_performance)
            
            # Identify leading sectors
            leading_sectors = self._identify_leading_sectors(sector_performance)
            
            # Generate rotation insights
            rotation_insights = self._generate_rotation_insights(
                rotation_patterns, leading_sectors
            )
            
            result = {
                "sector_performance": sector_performance,
                "rotation_patterns": rotation_patterns,
                "leading_sectors": leading_sectors,
                "rotation_insights": rotation_insights,
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache for 1 hour
            await self.cache_service.set(cache_key, result, 3600)
            
            return result
            
        except Exception as e:
            logger.error(f"Sector rotation analysis failed: {e}")
            return {"error": str(e)}
    
    async def analyze_volatility_patterns(self, symbol: str) -> Dict:
        """Analyze volatility patterns for a symbol"""
        try:
            cache_key = f"volatility_patterns_{symbol}"
            cached_analysis = await self.cache_service.get(cache_key)
            if cached_analysis:
                return cached_analysis
            
            # Get historical data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1y")
            
            if hist is None or hist.empty:
                return {"error": "Insufficient data for volatility analysis"}
            
            # Calculate volatility metrics
            volatility_metrics = self._calculate_volatility_metrics(hist)
            
            # Analyze volatility patterns
            volatility_patterns = self._analyze_volatility_patterns(hist)
            
            # Identify volatility regimes
            volatility_regimes = self._identify_volatility_regimes(hist)
            
            # Generate volatility insights
            volatility_insights = self._generate_volatility_insights(
                volatility_metrics, volatility_patterns, volatility_regimes
            )
            
            result = {
                "symbol": symbol,
                "volatility_metrics": volatility_metrics,
                "volatility_patterns": volatility_patterns,
                "volatility_regimes": volatility_regimes,
                "volatility_insights": volatility_insights,
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache for 1 hour
            await self.cache_service.set(cache_key, result, 3600)
            
            return result
            
        except Exception as e:
            logger.error(f"Volatility patterns analysis failed for {symbol}: {e}")
            return {"error": str(e)}
    
    async def _calculate_performance_metrics(self, positions: List[Dict]) -> Dict:
        """Calculate comprehensive performance metrics"""
        try:
            if not positions:
                return {}
            
            # Calculate portfolio metrics
            total_value = sum(pos.get("current_value", 0) for pos in positions)
            total_cost = sum(pos.get("cost_basis", 0) for pos in positions)
            total_gain_loss = total_value - total_cost
            total_gain_loss_pct = (total_gain_loss / total_cost * 100) if total_cost > 0 else 0
            
            # Calculate weighted metrics
            weighted_returns = []
            weighted_volatilities = []
            
            for position in positions:
                weight = position.get("current_value", 0) / total_value if total_value > 0 else 0
                pnl_pct = position.get("pnl_percent", 0)
                weighted_returns.append(weight * pnl_pct)
                weighted_volatilities.append(weight * 0.2)  # Simplified volatility
            
            portfolio_return = sum(weighted_returns)
            portfolio_volatility = sum(weighted_volatilities)
            
            # Calculate Sharpe ratio (simplified)
            risk_free_rate = 0.02  # 2% annual risk-free rate
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
            
            # Calculate maximum drawdown (simplified)
            max_drawdown = -0.1  # Placeholder
            
            return {
                "total_value": total_value,
                "total_cost": total_cost,
                "total_gain_loss": total_gain_loss,
                "total_gain_loss_pct": total_gain_loss_pct,
                "portfolio_return": portfolio_return,
                "portfolio_volatility": portfolio_volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "positions_count": len(positions)
            }
            
        except Exception as e:
            logger.error(f"Performance metrics calculation failed: {e}")
            return {}
    
    async def _compare_to_benchmarks(self, positions: List[Dict]) -> Dict:
        """Compare portfolio performance to benchmarks"""
        try:
            if not positions:
                return {}
            
            # Calculate portfolio performance
            portfolio_metrics = await self._calculate_performance_metrics(positions)
            portfolio_return = portfolio_metrics.get("portfolio_return", 0)
            
            # Get benchmark performance
            benchmark_performance = {}
            for benchmark_symbol, benchmark_name in self.benchmarks.items():
                ticker = yf.Ticker(benchmark_symbol)
                hist = ticker.history(period="1y")
                if hist is not None and not hist.empty:
                    benchmark_return = (hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0] * 100
                    benchmark_performance[benchmark_name] = {
                        "symbol": benchmark_symbol,
                        "return": benchmark_return,
                        "outperformance": portfolio_return - benchmark_return
                    }
            
            # Find best performing benchmark
            best_benchmark = max(benchmark_performance.items(), 
                               key=lambda x: x[1]["return"]) if benchmark_performance else None
            
            return {
                "benchmark_performance": benchmark_performance,
                "best_benchmark": best_benchmark,
                "portfolio_vs_benchmarks": benchmark_performance
            }
            
        except Exception as e:
            logger.error(f"Benchmark comparison failed: {e}")
            return {}
    
    async def _analyze_portfolio_risk(self, positions: List[Dict]) -> Dict:
        """Analyze portfolio risk metrics"""
        try:
            if not positions:
                return {}
            
            # Calculate concentration risk
            total_value = sum(pos.get("current_value", 0) for pos in positions)
            weights = [pos.get("current_value", 0) / total_value for pos in positions]
            
            # Herfindahl-Hirschman Index (concentration measure)
            hhi = sum(w**2 for w in weights)
            
            # Concentration risk level
            if hhi > 0.25:
                concentration_risk = "High"
            elif hhi > 0.15:
                concentration_risk = "Medium"
            else:
                concentration_risk = "Low"
            
            # Calculate portfolio beta (simplified)
            portfolio_beta = 1.0  # Placeholder
            
            # Calculate Value at Risk (simplified)
            portfolio_var = -0.05  # 5% daily VaR placeholder
            
            return {
                "concentration_risk": concentration_risk,
                "hhi_index": hhi,
                "portfolio_beta": portfolio_beta,
                "portfolio_var": portfolio_var,
                "risk_level": "Medium"  # Placeholder
            }
            
        except Exception as e:
            logger.error(f"Portfolio risk analysis failed: {e}")
            return {}
    
    async def _analyze_attribution(self, positions: List[Dict]) -> Dict:
        """Analyze performance attribution"""
        try:
            if not positions:
                return {}
            
            # Calculate contribution by position
            total_gain_loss = sum(pos.get("pnl", 0) for pos in positions)
            attribution = []
            
            for position in positions:
                contribution = position.get("pnl", 0)
                contribution_pct = (contribution / total_gain_loss * 100) if total_gain_loss != 0 else 0
                
                attribution.append({
                    "symbol": position.get("symbol", ""),
                    "contribution": contribution,
                    "contribution_pct": contribution_pct,
                    "weight": position.get("weight", 0)
                })
            
            # Sort by contribution
            attribution.sort(key=lambda x: x["contribution"], reverse=True)
            
            # Top contributors
            top_contributors = attribution[:3]
            bottom_contributors = attribution[-3:]
            
            return {
                "attribution": attribution,
                "top_contributors": top_contributors,
                "bottom_contributors": bottom_contributors,
                "total_attribution": total_gain_loss
            }
            
        except Exception as e:
            logger.error(f"Attribution analysis failed: {e}")
            return {}
    
    async def _generate_performance_recommendations(self, performance_metrics: Dict, 
                                                  benchmark_comparison: Dict, 
                                                  risk_analysis: Dict) -> List[str]:
        """Generate performance recommendations"""
        try:
            recommendations = []
            
            # Performance-based recommendations
            portfolio_return = performance_metrics.get("portfolio_return", 0)
            if portfolio_return < 0:
                recommendations.append("Portfolio is underperforming - consider rebalancing")
            elif portfolio_return > 10:
                recommendations.append("Strong performance - consider taking some profits")
            
            # Risk-based recommendations
            concentration_risk = risk_analysis.get("concentration_risk", "Medium")
            if concentration_risk == "High":
                recommendations.append("High concentration risk - diversify portfolio")
            
            # Benchmark-based recommendations
            benchmark_performance = benchmark_comparison.get("benchmark_performance", {})
            if benchmark_performance:
                avg_outperformance = np.mean([b["outperformance"] for b in benchmark_performance.values()])
                if avg_outperformance < -5:
                    recommendations.append("Underperforming benchmarks - review strategy")
                elif avg_outperformance > 5:
                    recommendations.append("Outperforming benchmarks - consider scaling up")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Performance recommendations generation failed: {e}")
            return []
    
    def _calculate_correlation_matrix(self, price_data: Dict[str, pd.Series]) -> Dict:
        """Calculate correlation matrix"""
        try:
            # Create DataFrame from price data
            df = pd.DataFrame(price_data)
            
            # Calculate returns
            returns = df.pct_change().dropna()
            
            # Calculate correlation matrix
            correlation_matrix = returns.corr()
            
            # Convert to dictionary
            return correlation_matrix.to_dict()
            
        except Exception as e:
            logger.error(f"Correlation matrix calculation failed: {e}")
            return {}
    
    def _find_high_correlations(self, correlation_matrix: Dict, symbols: List[str]) -> List[Dict]:
        """Find highly correlated pairs"""
        try:
            high_correlations = []
            
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols[i+1:], i+1):
                    if symbol1 in correlation_matrix and symbol2 in correlation_matrix[symbol1]:
                        correlation = correlation_matrix[symbol1][symbol2]
                        
                        if abs(correlation) > 0.7:  # High correlation threshold
                            high_correlations.append({
                                "symbol1": symbol1,
                                "symbol2": symbol2,
                                "correlation": correlation,
                                "strength": "Strong" if abs(correlation) > 0.8 else "Moderate"
                            })
            
            # Sort by absolute correlation
            high_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
            
            return high_correlations
            
        except Exception as e:
            logger.error(f"High correlations finding failed: {e}")
            return []
    
    def _analyze_correlation_clusters(self, correlation_matrix: Dict, symbols: List[str]) -> Dict:
        """Analyze correlation clusters"""
        try:
            # Create correlation matrix for clustering
            corr_data = []
            for symbol1 in symbols:
                row = []
                for symbol2 in symbols:
                    if symbol1 in correlation_matrix and symbol2 in correlation_matrix[symbol1]:
                        row.append(correlation_matrix[symbol1][symbol2])
                    else:
                        row.append(0)
                corr_data.append(row)
            
            # Perform K-means clustering
            if len(corr_data) > 1:
                kmeans = KMeans(n_clusters=min(3, len(symbols)), random_state=42)
                clusters = kmeans.fit_predict(corr_data)
                
                # Group symbols by cluster
                cluster_groups = {}
                for i, symbol in enumerate(symbols):
                    cluster = clusters[i]
                    if cluster not in cluster_groups:
                        cluster_groups[cluster] = []
                    cluster_groups[cluster].append(symbol)
                
                return {
                    "clusters": cluster_groups,
                    "cluster_count": len(cluster_groups)
                }
            else:
                return {"clusters": {}, "cluster_count": 0}
            
        except Exception as e:
            logger.error(f"Correlation clustering failed: {e}")
            return {"clusters": {}, "cluster_count": 0}
    
    def _generate_correlation_insights(self, high_correlations: List[Dict], 
                                     correlation_clusters: Dict) -> List[str]:
        """Generate correlation insights"""
        try:
            insights = []
            
            # High correlation insights
            if high_correlations:
                insights.append(f"Found {len(high_correlations)} highly correlated pairs")
                
                # Strongest correlation
                strongest = high_correlations[0]
                insights.append(
                    f"Strongest correlation: {strongest['symbol1']} and {strongest['symbol2']} "
                    f"({strongest['correlation']:.2f})"
                )
            
            # Cluster insights
            clusters = correlation_clusters.get("clusters", {})
            if len(clusters) > 1:
                insights.append(f"Symbols grouped into {len(clusters)} correlation clusters")
                
                for cluster_id, symbols in clusters.items():
                    if len(symbols) > 1:
                        insights.append(f"Cluster {cluster_id}: {', '.join(symbols)}")
            
            return insights
            
        except Exception as e:
            logger.error(f"Correlation insights generation failed: {e}")
            return []
    
    def _calculate_sector_performance(self, sector_data: Dict[str, pd.Series]) -> Dict:
        """Calculate sector performance metrics"""
        try:
            sector_performance = {}
            
            for sector_name, prices in sector_data.items():
                if len(prices) > 1:
                    # Calculate returns
                    total_return = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0] * 100
                    
                    # Calculate volatility
                    returns = prices.pct_change().dropna()
                    volatility = returns.std() * np.sqrt(252) * 100
                    
                    # Calculate Sharpe ratio
                    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
                    
                    sector_performance[sector_name] = {
                        "total_return": total_return,
                        "volatility": volatility,
                        "sharpe_ratio": sharpe_ratio,
                        "performance_rank": 0  # Will be set later
                    }
            
            # Rank sectors by performance
            sorted_sectors = sorted(sector_performance.items(), 
                                  key=lambda x: x[1]["total_return"], reverse=True)
            
            for rank, (sector_name, _) in enumerate(sorted_sectors, 1):
                sector_performance[sector_name]["performance_rank"] = rank
            
            return sector_performance
            
        except Exception as e:
            logger.error(f"Sector performance calculation failed: {e}")
            return {}
    
    def _analyze_rotation_patterns(self, sector_performance: Dict) -> Dict:
        """Analyze sector rotation patterns"""
        try:
            if not sector_performance:
                return {}
            
            # Identify cyclical vs defensive sectors
            cyclical_sectors = ['Technology', 'Financials', 'Industrials', 'Consumer Discretionary']
            defensive_sectors = ['Healthcare', 'Consumer Staples', 'Utilities']
            
            cyclical_performance = []
            defensive_performance = []
            
            for sector_name, performance in sector_performance.items():
                if sector_name in cyclical_sectors:
                    cyclical_performance.append(performance["total_return"])
                elif sector_name in defensive_sectors:
                    defensive_performance.append(performance["total_return"])
            
            # Calculate average performance
            avg_cyclical = np.mean(cyclical_performance) if cyclical_performance else 0
            avg_defensive = np.mean(defensive_performance) if defensive_performance else 0
            
            # Determine rotation pattern
            if avg_cyclical > avg_defensive + 2:
                rotation_pattern = "Risk-On"
            elif avg_defensive > avg_cyclical + 2:
                rotation_pattern = "Risk-Off"
            else:
                rotation_pattern = "Neutral"
            
            return {
                "rotation_pattern": rotation_pattern,
                "cyclical_performance": avg_cyclical,
                "defensive_performance": avg_defensive,
                "rotation_strength": abs(avg_cyclical - avg_defensive)
            }
            
        except Exception as e:
            logger.error(f"Rotation patterns analysis failed: {e}")
            return {}
    
    def _identify_leading_sectors(self, sector_performance: Dict) -> Dict:
        """Identify leading sectors"""
        try:
            if not sector_performance:
                return {}
            
            # Sort sectors by performance
            sorted_sectors = sorted(sector_performance.items(), 
                                  key=lambda x: x[1]["total_return"], reverse=True)
            
            # Top 3 performers
            top_performers = sorted_sectors[:3]
            
            # Bottom 3 performers
            bottom_performers = sorted_sectors[-3:]
            
            return {
                "top_performers": top_performers,
                "bottom_performers": bottom_performers,
                "leading_sector": top_performers[0][0] if top_performers else None,
                "lagging_sector": bottom_performers[0][0] if bottom_performers else None
            }
            
        except Exception as e:
            logger.error(f"Leading sectors identification failed: {e}")
            return {}
    
    def _generate_rotation_insights(self, rotation_patterns: Dict, leading_sectors: Dict) -> List[str]:
        """Generate rotation insights"""
        try:
            insights = []
            
            # Rotation pattern insights
            pattern = rotation_patterns.get("rotation_pattern", "Neutral")
            if pattern == "Risk-On":
                insights.append("Market showing risk-on sentiment - cyclical sectors leading")
                insights.append("Consider growth-oriented strategies")
            elif pattern == "Risk-Off":
                insights.append("Market showing risk-off sentiment - defensive sectors leading")
                insights.append("Consider defensive strategies")
            else:
                insights.append("Market showing neutral sentiment - mixed sector performance")
            
            # Leading sector insights
            leading_sector = leading_sectors.get("leading_sector")
            if leading_sector:
                insights.append(f"Leading sector: {leading_sector}")
            
            lagging_sector = leading_sectors.get("lagging_sector")
            if lagging_sector:
                insights.append(f"Lagging sector: {lagging_sector}")
            
            return insights
            
        except Exception as e:
            logger.error(f"Rotation insights generation failed: {e}")
            return []
    
    def _calculate_volatility_metrics(self, hist: pd.DataFrame) -> Dict:
        """Calculate volatility metrics"""
        try:
            returns = hist['Close'].pct_change().dropna()
            
            # Historical volatility
            historical_vol = returns.std() * np.sqrt(252)
            
            # Rolling volatility (20-day)
            rolling_vol = returns.rolling(20).std() * np.sqrt(252)
            current_vol = rolling_vol.iloc[-1]
            avg_vol = rolling_vol.mean()
            
            # Volatility of volatility
            vol_of_vol = rolling_vol.std()
            
            return {
                "historical_volatility": historical_vol,
                "current_volatility": current_vol,
                "average_volatility": avg_vol,
                "volatility_of_volatility": vol_of_vol,
                "volatility_percentile": stats.percentileofscore(rolling_vol.dropna(), current_vol)
            }
            
        except Exception as e:
            logger.error(f"Volatility metrics calculation failed: {e}")
            return {}
    
    def _analyze_volatility_patterns(self, hist: pd.DataFrame) -> Dict:
        """Analyze volatility patterns"""
        try:
            returns = hist['Close'].pct_change().dropna()
            rolling_vol = returns.rolling(20).std() * np.sqrt(252)
            
            # Volatility clustering
            high_vol_periods = rolling_vol > rolling_vol.quantile(0.8)
            low_vol_periods = rolling_vol < rolling_vol.quantile(0.2)
            
            # Volatility trends
            vol_trend = "Increasing" if rolling_vol.iloc[-1] > rolling_vol.iloc[-20] else "Decreasing"
            
            return {
                "volatility_clustering": high_vol_periods.sum(),
                "low_volatility_periods": low_vol_periods.sum(),
                "volatility_trend": vol_trend,
                "volatility_regime": "High" if rolling_vol.iloc[-1] > rolling_vol.mean() else "Low"
            }
            
        except Exception as e:
            logger.error(f"Volatility patterns analysis failed: {e}")
            return {}
    
    def _identify_volatility_regimes(self, hist: pd.DataFrame) -> Dict:
        """Identify volatility regimes"""
        try:
            returns = hist['Close'].pct_change().dropna()
            rolling_vol = returns.rolling(20).std() * np.sqrt(252)
            
            # Define regimes based on volatility percentiles
            vol_75 = rolling_vol.quantile(0.75)
            vol_25 = rolling_vol.quantile(0.25)
            
            current_vol = rolling_vol.iloc[-1]
            
            if current_vol > vol_75:
                regime = "High Volatility"
            elif current_vol < vol_25:
                regime = "Low Volatility"
            else:
                regime = "Normal Volatility"
            
            return {
                "current_regime": regime,
                "regime_threshold_high": vol_75,
                "regime_threshold_low": vol_25,
                "regime_duration": 0  # Would calculate actual duration
            }
            
        except Exception as e:
            logger.error(f"Volatility regimes identification failed: {e}")
            return {}
    
    def _generate_volatility_insights(self, volatility_metrics: Dict, 
                                    volatility_patterns: Dict, 
                                    volatility_regimes: Dict) -> List[str]:
        """Generate volatility insights"""
        try:
            insights = []
            
            # Current volatility insights
            current_vol = volatility_metrics.get("current_volatility", 0)
            avg_vol = volatility_metrics.get("average_volatility", 0)
            
            if current_vol > avg_vol * 1.2:
                insights.append("Current volatility is elevated - consider risk management")
            elif current_vol < avg_vol * 0.8:
                insights.append("Current volatility is low - potential for increased activity")
            
            # Regime insights
            current_regime = volatility_regimes.get("current_regime", "Normal Volatility")
            insights.append(f"Current volatility regime: {current_regime}")
            
            # Trend insights
            vol_trend = volatility_patterns.get("volatility_trend", "Stable")
            insights.append(f"Volatility trend: {vol_trend}")
            
            return insights
            
        except Exception as e:
            logger.error(f"Volatility insights generation failed: {e}")
            return []
    
    async def _background_analytics(self):
        """Background task for continuous analytics"""
        while True:
            try:
                # This would perform continuous analytics
                logger.debug("Advanced Analytics background task running")
                await asyncio.sleep(3600)  # Run every hour
                
            except Exception as e:
                logger.error(f"Background analytics error: {e}")
                await asyncio.sleep(3600)
    
    async def get_analytics_stats(self) -> Dict:
        """Get advanced analytics service statistics"""
        try:
            return {
                "background_task_running": self.background_task is not None,
                "benchmarks_tracked": len(self.benchmarks),
                "analyses_completed": 0,  # Would track actual analyses
                "cache_hits": 0  # Would track actual cache hits
            }
            
        except Exception as e:
            logger.error(f"Failed to get analytics stats: {e}")
            return {"error": str(e)}
