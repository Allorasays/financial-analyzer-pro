#!/usr/bin/env python3
"""
Sentiment Analysis Service for financial news and market sentiment
"""

import asyncio
import logging
import requests
import json
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import yfinance as yf
from textblob import TextBlob
import re

from cache_service import CacheService
from config import settings

logger = logging.getLogger(__name__)

class SentimentAnalysisService:
    def __init__(self):
        self.cache_service = CacheService()
        self.background_task = None
        
        # News sources and APIs
        self.news_sources = [
            "financial-news",
            "market-news",
            "earnings-news"
        ]
        
        # Sentiment keywords
        self.positive_keywords = [
            "bullish", "surge", "rally", "gain", "profit", "growth", "positive",
            "strong", "outperform", "buy", "upgrade", "beat", "exceed", "rise",
            "increase", "improve", "optimistic", "confident", "success"
        ]
        
        self.negative_keywords = [
            "bearish", "decline", "fall", "loss", "drop", "negative", "weak",
            "underperform", "sell", "downgrade", "miss", "disappoint", "fall",
            "decrease", "worsen", "pessimistic", "concern", "risk", "crash"
        ]
    
    async def start_background_tasks(self):
        """Start background sentiment analysis tasks"""
        if self.background_task is None:
            self.background_task = asyncio.create_task(self._background_sentiment_analysis())
            logger.info("Sentiment Analysis background tasks started")
    
    async def stop_background_tasks(self):
        """Stop background sentiment analysis tasks"""
        if self.background_task:
            self.background_task.cancel()
            try:
                await self.background_task
            except asyncio.CancelledError:
                pass
            self.background_task = None
            logger.info("Sentiment Analysis background tasks stopped")
    
    async def analyze_symbol_sentiment(self, symbol: str) -> Dict:
        """Analyze sentiment for a specific symbol"""
        try:
            cache_key = f"sentiment_{symbol}"
            cached_sentiment = await self.cache_service.get(cache_key)
            if cached_sentiment:
                return cached_sentiment
            
            # Get news data
            news_data = await self._fetch_symbol_news(symbol)
            
            # Analyze sentiment
            sentiment_analysis = self._analyze_news_sentiment(news_data)
            
            # Get social sentiment (simplified)
            social_sentiment = await self._analyze_social_sentiment(symbol)
            
            # Calculate overall sentiment
            overall_sentiment = self._calculate_overall_sentiment(
                sentiment_analysis, social_sentiment
            )
            
            result = {
                "symbol": symbol,
                "overall_sentiment": overall_sentiment["sentiment"],
                "sentiment_score": overall_sentiment["score"],
                "confidence": overall_sentiment["confidence"],
                "news_sentiment": sentiment_analysis,
                "social_sentiment": social_sentiment,
                "sentiment_trend": overall_sentiment["trend"],
                "key_insights": overall_sentiment["insights"],
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache for 30 minutes
            await self.cache_service.set(cache_key, result, 1800)
            
            return result
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed for {symbol}: {e}")
            return {"error": str(e)}
    
    async def analyze_market_sentiment(self) -> Dict:
        """Analyze overall market sentiment"""
        try:
            cache_key = "market_sentiment"
            cached_sentiment = await self.cache_service.get(cache_key)
            if cached_sentiment:
                return cached_sentiment
            
            # Get market news
            market_news = await self._fetch_market_news()
            
            # Analyze sentiment
            sentiment_analysis = self._analyze_news_sentiment(market_news)
            
            # Get VIX data (fear gauge)
            vix_sentiment = await self._analyze_vix_sentiment()
            
            # Calculate market sentiment
            market_sentiment = self._calculate_market_sentiment(
                sentiment_analysis, vix_sentiment
            )
            
            result = {
                "market_sentiment": market_sentiment["sentiment"],
                "sentiment_score": market_sentiment["score"],
                "confidence": market_sentiment["confidence"],
                "news_sentiment": sentiment_analysis,
                "vix_sentiment": vix_sentiment,
                "sentiment_trend": market_sentiment["trend"],
                "key_insights": market_sentiment["insights"],
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache for 15 minutes
            await self.cache_service.set(cache_key, result, 900)
            
            return result
            
        except Exception as e:
            logger.error(f"Market sentiment analysis failed: {e}")
            return {"error": str(e)}
    
    async def analyze_news_sentiment(self, news_text: str) -> Dict:
        """Analyze sentiment of news text"""
        try:
            # Clean text
            cleaned_text = self._clean_text(news_text)
            
            # TextBlob sentiment
            blob = TextBlob(cleaned_text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Keyword-based sentiment
            keyword_sentiment = self._analyze_keyword_sentiment(cleaned_text)
            
            # Combine analyses
            combined_sentiment = self._combine_sentiment_analyses(
                polarity, subjectivity, keyword_sentiment
            )
            
            return {
                "textblob_polarity": polarity,
                "textblob_subjectivity": subjectivity,
                "keyword_sentiment": keyword_sentiment,
                "combined_sentiment": combined_sentiment["sentiment"],
                "combined_score": combined_sentiment["score"],
                "confidence": combined_sentiment["confidence"]
            }
            
        except Exception as e:
            logger.error(f"News sentiment analysis failed: {e}")
            return {"error": str(e)}
    
    async def _fetch_symbol_news(self, symbol: str) -> List[Dict]:
        """Fetch news data for a symbol"""
        try:
            # Use yfinance to get news
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            # Process news data
            processed_news = []
            for article in news[:10]:  # Limit to 10 articles
                processed_news.append({
                    "title": article.get("title", ""),
                    "summary": article.get("summary", ""),
                    "publisher": article.get("publisher", ""),
                    "published": article.get("providerPublishTime", 0),
                    "url": article.get("link", "")
                })
            
            return processed_news
            
        except Exception as e:
            logger.error(f"Failed to fetch news for {symbol}: {e}")
            return []
    
    async def _fetch_market_news(self) -> List[Dict]:
        """Fetch general market news"""
        try:
            # This would integrate with news APIs like NewsAPI, Alpha Vantage, etc.
            # For now, we'll use a simplified approach
            
            market_news = [
                {
                    "title": "Market Update: Stocks Show Mixed Signals",
                    "summary": "The market is showing mixed signals with some sectors performing well while others face headwinds.",
                    "publisher": "Financial News",
                    "published": int(datetime.now().timestamp()),
                    "url": ""
                }
            ]
            
            return market_news
            
        except Exception as e:
            logger.error(f"Failed to fetch market news: {e}")
            return []
    
    def _analyze_news_sentiment(self, news_data: List[Dict]) -> Dict:
        """Analyze sentiment from news data"""
        try:
            if not news_data:
                return {"sentiment": "Neutral", "score": 0, "confidence": 0}
            
            total_sentiment = 0
            total_confidence = 0
            article_count = 0
            
            for article in news_data:
                # Combine title and summary
                text = f"{article.get('title', '')} {article.get('summary', '')}"
                
                # Analyze sentiment
                sentiment_result = self._analyze_text_sentiment(text)
                
                if sentiment_result:
                    total_sentiment += sentiment_result["score"]
                    total_confidence += sentiment_result["confidence"]
                    article_count += 1
            
            if article_count == 0:
                return {"sentiment": "Neutral", "score": 0, "confidence": 0}
            
            avg_sentiment = total_sentiment / article_count
            avg_confidence = total_confidence / article_count
            
            # Determine sentiment category
            if avg_sentiment > 0.2:
                sentiment = "Positive"
            elif avg_sentiment < -0.2:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"
            
            return {
                "sentiment": sentiment,
                "score": avg_sentiment,
                "confidence": avg_confidence,
                "article_count": article_count
            }
            
        except Exception as e:
            logger.error(f"News sentiment analysis failed: {e}")
            return {"sentiment": "Neutral", "score": 0, "confidence": 0}
    
    def _analyze_text_sentiment(self, text: str) -> Optional[Dict]:
        """Analyze sentiment of text"""
        try:
            if not text:
                return None
            
            # Clean text
            cleaned_text = self._clean_text(text)
            
            # TextBlob sentiment
            blob = TextBlob(cleaned_text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Keyword-based sentiment
            keyword_sentiment = self._analyze_keyword_sentiment(cleaned_text)
            
            # Combine analyses
            combined_score = (polarity + keyword_sentiment["score"]) / 2
            combined_confidence = (subjectivity + keyword_sentiment["confidence"]) / 2
            
            return {
                "score": combined_score,
                "confidence": combined_confidence
            }
            
        except Exception as e:
            logger.error(f"Text sentiment analysis failed: {e}")
            return None
    
    def _analyze_keyword_sentiment(self, text: str) -> Dict:
        """Analyze sentiment based on keywords"""
        try:
            text_lower = text.lower()
            
            positive_count = sum(1 for keyword in self.positive_keywords if keyword in text_lower)
            negative_count = sum(1 for keyword in self.negative_keywords if keyword in text_lower)
            
            total_keywords = positive_count + negative_count
            
            if total_keywords == 0:
                return {"score": 0, "confidence": 0}
            
            # Calculate score (-1 to 1)
            score = (positive_count - negative_count) / total_keywords
            
            # Calculate confidence (0 to 1)
            confidence = min(1, total_keywords / 10)  # More keywords = higher confidence
            
            return {"score": score, "confidence": confidence}
            
        except Exception as e:
            logger.error(f"Keyword sentiment analysis failed: {e}")
            return {"score": 0, "confidence": 0}
    
    async def _analyze_social_sentiment(self, symbol: str) -> Dict:
        """Analyze social media sentiment (simplified)"""
        try:
            # This would integrate with social media APIs
            # For now, we'll return a mock analysis
            
            return {
                "sentiment": "Neutral",
                "score": 0,
                "confidence": 0.5,
                "source": "Social Media",
                "note": "Social sentiment analysis not implemented"
            }
            
        except Exception as e:
            logger.error(f"Social sentiment analysis failed: {e}")
            return {"sentiment": "Neutral", "score": 0, "confidence": 0}
    
    async def _analyze_vix_sentiment(self) -> Dict:
        """Analyze VIX (fear gauge) sentiment"""
        try:
            # Get VIX data
            vix = yf.Ticker("^VIX")
            hist = vix.history(period="5d")
            
            if hist is None or hist.empty:
                return {"sentiment": "Neutral", "score": 0, "confidence": 0}
            
            current_vix = hist['Close'].iloc[-1]
            avg_vix = hist['Close'].mean()
            
            # VIX interpretation
            if current_vix > avg_vix * 1.2:
                sentiment = "Fearful"
                score = -0.8
            elif current_vix > avg_vix * 1.1:
                sentiment = "Cautious"
                score = -0.4
            elif current_vix < avg_vix * 0.8:
                sentiment = "Complacent"
                score = 0.4
            else:
                sentiment = "Neutral"
                score = 0
            
            return {
                "sentiment": sentiment,
                "score": score,
                "confidence": 0.8,
                "vix_level": current_vix,
                "avg_vix": avg_vix
            }
            
        except Exception as e:
            logger.error(f"VIX sentiment analysis failed: {e}")
            return {"sentiment": "Neutral", "score": 0, "confidence": 0}
    
    def _calculate_overall_sentiment(self, news_sentiment: Dict, social_sentiment: Dict) -> Dict:
        """Calculate overall sentiment from multiple sources"""
        try:
            # Weighted average of sentiment sources
            news_weight = 0.7
            social_weight = 0.3
            
            overall_score = (
                news_sentiment.get("score", 0) * news_weight +
                social_sentiment.get("score", 0) * social_weight
            )
            
            overall_confidence = (
                news_sentiment.get("confidence", 0) * news_weight +
                social_sentiment.get("confidence", 0) * social_weight
            )
            
            # Determine sentiment
            if overall_score > 0.2:
                sentiment = "Positive"
                trend = "Bullish"
            elif overall_score < -0.2:
                sentiment = "Negative"
                trend = "Bearish"
            else:
                sentiment = "Neutral"
                trend = "Sideways"
            
            # Generate insights
            insights = self._generate_sentiment_insights(
                sentiment, overall_score, news_sentiment, social_sentiment
            )
            
            return {
                "sentiment": sentiment,
                "score": overall_score,
                "confidence": overall_confidence,
                "trend": trend,
                "insights": insights
            }
            
        except Exception as e:
            logger.error(f"Overall sentiment calculation failed: {e}")
            return {"sentiment": "Neutral", "score": 0, "confidence": 0}
    
    def _calculate_market_sentiment(self, news_sentiment: Dict, vix_sentiment: Dict) -> Dict:
        """Calculate overall market sentiment"""
        try:
            # Weighted average
            news_weight = 0.6
            vix_weight = 0.4
            
            overall_score = (
                news_sentiment.get("score", 0) * news_weight +
                vix_sentiment.get("score", 0) * vix_weight
            )
            
            overall_confidence = (
                news_sentiment.get("confidence", 0) * news_weight +
                vix_sentiment.get("confidence", 0) * vix_weight
            )
            
            # Determine sentiment
            if overall_score > 0.2:
                sentiment = "Bullish"
                trend = "Positive"
            elif overall_score < -0.2:
                sentiment = "Bearish"
                trend = "Negative"
            else:
                sentiment = "Neutral"
                trend = "Sideways"
            
            # Generate insights
            insights = self._generate_market_insights(
                sentiment, overall_score, news_sentiment, vix_sentiment
            )
            
            return {
                "sentiment": sentiment,
                "score": overall_score,
                "confidence": overall_confidence,
                "trend": trend,
                "insights": insights
            }
            
        except Exception as e:
            logger.error(f"Market sentiment calculation failed: {e}")
            return {"sentiment": "Neutral", "score": 0, "confidence": 0}
    
    def _generate_sentiment_insights(self, sentiment: str, score: float, 
                                    news_sentiment: Dict, social_sentiment: Dict) -> List[str]:
        """Generate sentiment insights"""
        try:
            insights = []
            
            if sentiment == "Positive":
                insights.append("Overall sentiment is positive")
                insights.append("Consider bullish strategies")
            elif sentiment == "Negative":
                insights.append("Overall sentiment is negative")
                insights.append("Consider defensive strategies")
            else:
                insights.append("Overall sentiment is neutral")
                insights.append("Market is in consolidation phase")
            
            # News-specific insights
            news_score = news_sentiment.get("score", 0)
            if abs(news_score) > 0.5:
                insights.append("Strong news sentiment detected")
            
            # Social-specific insights
            social_score = social_sentiment.get("score", 0)
            if abs(social_score) > 0.3:
                insights.append("Social media sentiment is significant")
            
            return insights
            
        except Exception as e:
            logger.error(f"Sentiment insights generation failed: {e}")
            return []
    
    def _generate_market_insights(self, sentiment: str, score: float, 
                                news_sentiment: Dict, vix_sentiment: Dict) -> List[str]:
        """Generate market insights"""
        try:
            insights = []
            
            if sentiment == "Bullish":
                insights.append("Market sentiment is bullish")
                insights.append("Consider growth-oriented strategies")
            elif sentiment == "Bearish":
                insights.append("Market sentiment is bearish")
                insights.append("Consider defensive strategies")
            else:
                insights.append("Market sentiment is neutral")
                insights.append("Market is in consolidation phase")
            
            # VIX-specific insights
            vix_sentiment_level = vix_sentiment.get("sentiment", "Neutral")
            if vix_sentiment_level == "Fearful":
                insights.append("High fear levels detected (VIX elevated)")
                insights.append("Potential buying opportunity")
            elif vix_sentiment_level == "Complacent":
                insights.append("Low fear levels detected (VIX low)")
                insights.append("Market may be overconfident")
            
            return insights
            
        except Exception as e:
            logger.error(f"Market insights generation failed: {e}")
            return []
    
    def _clean_text(self, text: str) -> str:
        """Clean text for sentiment analysis"""
        try:
            # Remove special characters and normalize
            cleaned = re.sub(r'[^\w\s]', ' ', text)
            cleaned = re.sub(r'\s+', ' ', cleaned)
            return cleaned.strip().lower()
            
        except Exception as e:
            logger.error(f"Text cleaning failed: {e}")
            return text
    
    def _combine_sentiment_analyses(self, polarity: float, subjectivity: float, 
                                  keyword_sentiment: Dict) -> Dict:
        """Combine different sentiment analyses"""
        try:
            # Weighted combination
            textblob_weight = 0.6
            keyword_weight = 0.4
            
            combined_score = (
                polarity * textblob_weight +
                keyword_sentiment.get("score", 0) * keyword_weight
            )
            
            combined_confidence = (
                subjectivity * textblob_weight +
                keyword_sentiment.get("confidence", 0) * keyword_weight
            )
            
            # Determine sentiment
            if combined_score > 0.1:
                sentiment = "Positive"
            elif combined_score < -0.1:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"
            
            return {
                "sentiment": sentiment,
                "score": combined_score,
                "confidence": combined_confidence
            }
            
        except Exception as e:
            logger.error(f"Sentiment combination failed: {e}")
            return {"sentiment": "Neutral", "score": 0, "confidence": 0}
    
    async def _background_sentiment_analysis(self):
        """Background task for continuous sentiment analysis"""
        while True:
            try:
                # This would perform continuous sentiment analysis
                # For now, we'll just log that the task is running
                logger.debug("Sentiment Analysis background task running")
                await asyncio.sleep(1800)  # Run every 30 minutes
                
            except Exception as e:
                logger.error(f"Background sentiment analysis error: {e}")
                await asyncio.sleep(1800)
    
    async def get_sentiment_stats(self) -> Dict:
        """Get sentiment analysis service statistics"""
        try:
            return {
                "background_task_running": self.background_task is not None,
                "positive_keywords": len(self.positive_keywords),
                "negative_keywords": len(self.negative_keywords),
                "news_sources": len(self.news_sources),
                "analyses_completed": 0,  # Would track actual analyses
                "cache_hits": 0  # Would track actual cache hits
            }
            
        except Exception as e:
            logger.error(f"Failed to get sentiment stats: {e}")
            return {"error": str(e)}

