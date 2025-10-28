"""
News Service for Financial Analyzer Pro
Integrates NewsAPI for real-time news tracking and market event detection
"""

import requests
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
from config import NEWSAPI_CONFIG
import re
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsService:
    """Service for fetching and analyzing financial news"""
    
    def __init__(self):
        self.api_key = NEWSAPI_CONFIG['api_key']
        self.base_url = NEWSAPI_CONFIG['base_url']
        self.timeout = NEWSAPI_CONFIG['timeout']
        self.cache_ttl = NEWSAPI_CONFIG['cache_ttl']
        self.cache = {}
        self.cache_timestamps = {}
        
        # Market-moving keywords
        self.market_keywords = {
            'earnings': ['earnings', 'quarterly results', 'q1', 'q2', 'q3', 'q4', 'profit', 'revenue'],
            'mergers': ['merger', 'acquisition', 'buyout', 'takeover', 'deal'],
            'ipo': ['ipo', 'initial public offering', 'public offering', 'listing'],
            'dividend': ['dividend', 'payout', 'yield'],
            'analyst': ['upgrade', 'downgrade', 'rating', 'target price', 'analyst'],
            'regulatory': ['sec', 'fda', 'regulation', 'investigation', 'lawsuit'],
            'economic': ['fed', 'interest rate', 'inflation', 'gdp', 'unemployment'],
            'crypto': ['bitcoin', 'cryptocurrency', 'crypto', 'blockchain'],
            'oil': ['oil', 'crude', 'energy', 'opec'],
            'tech': ['apple', 'microsoft', 'google', 'amazon', 'meta', 'tesla']
        }
    
    def _get_cached_data(self, key: str) -> Optional[Any]:
        """Get cached data if not expired"""
        if key in self.cache and key in self.cache_timestamps:
            if (time.time() - self.cache_timestamps[key]) < self.cache_ttl:
                return self.cache[key]
            else:
                # Remove expired cache
                self.cache.pop(key, None)
                self.cache_timestamps.pop(key, None)
        return None
    
    def _set_cached_data(self, key: str, data: Any):
        """Set cached data with timestamp"""
        self.cache[key] = data
        self.cache_timestamps[key] = time.time()
    
    def get_news_for_ticker(self, ticker: str, hours_back: int = 24) -> Dict[str, Any]:
        """Get news articles for a specific ticker"""
        cache_key = f"news_{ticker}_{hours_back}"
        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            return cached_data
        
        try:
            # Calculate date range
            to_date = datetime.now()
            from_date = to_date - timedelta(hours=hours_back)
            
            # Search for ticker-specific news
            params = {
                'apiKey': self.api_key,
                'q': ticker,
                'from': from_date.strftime('%Y-%m-%d'),
                'to': to_date.strftime('%Y-%m-%d'),
                'sortBy': 'publishedAt',
                'language': 'en',
                'pageSize': 50
            }
            
            response = requests.get(
                f"{self.base_url}/everything",
                params=params,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                
                # Process and categorize articles
                processed_articles = self._process_articles(articles, ticker)
                
                result = {
                    'ticker': ticker.upper(),
                    'total_articles': len(articles),
                    'processed_articles': len(processed_articles),
                    'articles': processed_articles,
                    'market_impact': self._assess_market_impact(processed_articles),
                    'timestamp': datetime.now().isoformat(),
                    'status': 'success'
                }
                
                self._set_cached_data(cache_key, result)
                return result
                
            else:
                logger.error(f"NewsAPI error: {response.status_code} - {response.text}")
                return self._get_fallback_response(ticker)
                
        except Exception as e:
            logger.error(f"Error fetching news for {ticker}: {str(e)}")
            return self._get_fallback_response(ticker)
    
    def get_market_news(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get general market news"""
        cache_key = f"market_news_{hours_back}"
        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            return cached_data
        
        try:
            # Calculate date range
            to_date = datetime.now()
            from_date = to_date - timedelta(hours=hours_back)
            
            params = {
                'apiKey': self.api_key,
                'category': 'business',
                'country': 'us',
                'from': from_date.strftime('%Y-%m-%d'),
                'to': to_date.strftime('%Y-%m-%d'),
                'sortBy': 'publishedAt',
                'pageSize': 100
            }
            
            response = requests.get(
                f"{self.base_url}/top-headlines",
                params=params,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                
                # Process and categorize articles
                processed_articles = self._process_market_articles(articles)
                
                result = {
                    'total_articles': len(articles),
                    'processed_articles': len(processed_articles),
                    'articles': processed_articles,
                    'market_sentiment': self._calculate_market_sentiment(processed_articles),
                    'key_events': self._extract_key_events(processed_articles),
                    'timestamp': datetime.now().isoformat(),
                    'status': 'success'
                }
                
                self._set_cached_data(cache_key, result)
                return result
                
            else:
                logger.error(f"NewsAPI error: {response.status_code} - {response.text}")
                return self._get_fallback_market_response()
                
        except Exception as e:
            logger.error(f"Error fetching market news: {str(e)}")
            return self._get_fallback_market_response()
    
    def _process_articles(self, articles: List[Dict], ticker: str) -> List[Dict]:
        """Process and categorize articles for a specific ticker"""
        processed = []
        
        for article in articles:
            if not article.get('title') or not article.get('description'):
                continue
                
            # Extract relevant information
            processed_article = {
                'title': article.get('title', ''),
                'description': article.get('description', ''),
                'url': article.get('url', ''),
                'source': article.get('source', {}).get('name', 'Unknown'),
                'published_at': article.get('publishedAt', ''),
                'relevance_score': self._calculate_relevance_score(article, ticker),
                'market_impact': self._assess_article_impact(article),
                'categories': self._categorize_article(article),
                'sentiment': self._analyze_article_sentiment(article)
            }
            
            # Only include relevant articles
            if processed_article['relevance_score'] > 0.3:
                processed.append(processed_article)
        
        # Sort by relevance and impact
        processed.sort(key=lambda x: (x['relevance_score'], x['market_impact']), reverse=True)
        return processed[:20]  # Return top 20 most relevant
    
    def _process_market_articles(self, articles: List[Dict]) -> List[Dict]:
        """Process general market articles"""
        processed = []
        
        for article in articles:
            if not article.get('title') or not article.get('description'):
                continue
                
            processed_article = {
                'title': article.get('title', ''),
                'description': article.get('description', ''),
                'url': article.get('url', ''),
                'source': article.get('source', {}).get('name', 'Unknown'),
                'published_at': article.get('publishedAt', ''),
                'market_impact': self._assess_article_impact(article),
                'categories': self._categorize_article(article),
                'sentiment': self._analyze_article_sentiment(article),
                'tickers_mentioned': self._extract_tickers(article)
            }
            
            processed.append(processed_article)
        
        # Sort by market impact
        processed.sort(key=lambda x: x['market_impact'], reverse=True)
        return processed[:50]  # Return top 50 most impactful
    
    def _calculate_relevance_score(self, article: Dict, ticker: str) -> float:
        """Calculate how relevant an article is to a specific ticker"""
        text = f"{article.get('title', '')} {article.get('description', '')}".lower()
        ticker_lower = ticker.lower()
        
        score = 0.0
        
        # Direct ticker mention
        if ticker_lower in text:
            score += 0.8
        
        # Company name variations (simplified)
        company_names = {
            'AAPL': ['apple'],
            'MSFT': ['microsoft'],
            'GOOGL': ['google', 'alphabet'],
            'AMZN': ['amazon'],
            'TSLA': ['tesla'],
            'META': ['facebook', 'meta'],
            'NVDA': ['nvidia']
        }
        
        if ticker in company_names:
            for name in company_names[ticker]:
                if name in text:
                    score += 0.6
                    break
        
        return min(score, 1.0)
    
    def _assess_article_impact(self, article: Dict) -> float:
        """Assess the potential market impact of an article"""
        text = f"{article.get('title', '')} {article.get('description', '')}".lower()
        
        impact_score = 0.0
        
        # High impact keywords
        high_impact = ['earnings', 'merger', 'acquisition', 'ipo', 'lawsuit', 'investigation', 'sec', 'fda']
        medium_impact = ['upgrade', 'downgrade', 'dividend', 'partnership', 'contract']
        low_impact = ['analysis', 'forecast', 'opinion', 'review']
        
        for keyword in high_impact:
            if keyword in text:
                impact_score += 0.3
        
        for keyword in medium_impact:
            if keyword in text:
                impact_score += 0.2
        
        for keyword in low_impact:
            if keyword in text:
                impact_score += 0.1
        
        return min(impact_score, 1.0)
    
    def _categorize_article(self, article: Dict) -> List[str]:
        """Categorize article based on content"""
        text = f"{article.get('title', '')} {article.get('description', '')}".lower()
        categories = []
        
        for category, keywords in self.market_keywords.items():
            if any(keyword in text for keyword in keywords):
                categories.append(category)
        
        return categories
    
    def _analyze_article_sentiment(self, article: Dict) -> Dict[str, Any]:
        """Analyze sentiment of an article"""
        text = f"{article.get('title', '')} {article.get('description', '')}"
        
        # Simple sentiment analysis based on keywords
        positive_words = ['growth', 'profit', 'gain', 'up', 'rise', 'increase', 'positive', 'strong', 'beat', 'exceed']
        negative_words = ['loss', 'decline', 'down', 'fall', 'decrease', 'negative', 'weak', 'miss', 'disappoint']
        
        positive_count = sum(1 for word in positive_words if word in text.lower())
        negative_count = sum(1 for word in negative_words if word in text.lower())
        
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            sentiment_score = 0.0
            sentiment_label = 'Neutral'
        else:
            sentiment_score = (positive_count - negative_count) / total_sentiment_words
            if sentiment_score > 0.2:
                sentiment_label = 'Positive'
            elif sentiment_score < -0.2:
                sentiment_label = 'Negative'
            else:
                sentiment_label = 'Neutral'
        
        return {
            'score': sentiment_score,
            'label': sentiment_label,
            'positive_words': positive_count,
            'negative_words': negative_count
        }
    
    def _extract_tickers(self, article: Dict) -> List[str]:
        """Extract ticker symbols mentioned in article"""
        text = f"{article.get('title', '')} {article.get('description', '')}"
        
        # Simple ticker extraction (look for common patterns)
        ticker_pattern = r'\b[A-Z]{1,5}\b'
        potential_tickers = re.findall(ticker_pattern, text)
        
        # Filter for likely tickers (common financial tickers)
        common_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'SPY', 'QQQ', 'DIA']
        mentioned_tickers = [ticker for ticker in potential_tickers if ticker in common_tickers]
        
        return list(set(mentioned_tickers))  # Remove duplicates
    
    def _assess_market_impact(self, articles: List[Dict]) -> Dict[str, Any]:
        """Assess overall market impact from articles"""
        if not articles:
            return {'overall_impact': 'Low', 'sentiment': 'Neutral', 'key_themes': []}
        
        # Calculate average sentiment
        sentiments = [article['sentiment']['score'] for article in articles]
        avg_sentiment = sum(sentiments) / len(sentiments)
        
        # Determine sentiment label
        if avg_sentiment > 0.2:
            sentiment_label = 'Positive'
        elif avg_sentiment < -0.2:
            sentiment_label = 'Negative'
        else:
            sentiment_label = 'Neutral'
        
        # Calculate average impact
        impacts = [article['market_impact'] for article in articles]
        avg_impact = sum(impacts) / len(impacts)
        
        # Determine impact level
        if avg_impact > 0.6:
            impact_level = 'High'
        elif avg_impact > 0.3:
            impact_level = 'Medium'
        else:
            impact_level = 'Low'
        
        # Extract key themes
        all_categories = []
        for article in articles:
            all_categories.extend(article['categories'])
        
        category_counts = defaultdict(int)
        for category in all_categories:
            category_counts[category] += 1
        
        key_themes = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'overall_impact': impact_level,
            'sentiment': sentiment_label,
            'sentiment_score': avg_sentiment,
            'impact_score': avg_impact,
            'key_themes': [theme[0] for theme in key_themes],
            'article_count': len(articles)
        }
    
    def _calculate_market_sentiment(self, articles: List[Dict]) -> Dict[str, Any]:
        """Calculate overall market sentiment from articles"""
        if not articles:
            return {'sentiment': 'Neutral', 'score': 0.0, 'confidence': 0.0}
        
        sentiments = [article['sentiment']['score'] for article in articles]
        avg_sentiment = sum(sentiments) / len(sentiments)
        
        # Calculate confidence based on article count and sentiment consistency
        sentiment_variance = sum((s - avg_sentiment) ** 2 for s in sentiments) / len(sentiments)
        confidence = max(0.0, 1.0 - sentiment_variance)
        
        if avg_sentiment > 0.2:
            sentiment_label = 'Bullish'
        elif avg_sentiment < -0.2:
            sentiment_label = 'Bearish'
        else:
            sentiment_label = 'Neutral'
        
        return {
            'sentiment': sentiment_label,
            'score': avg_sentiment,
            'confidence': confidence,
            'article_count': len(articles)
        }
    
    def _extract_key_events(self, articles: List[Dict]) -> List[Dict]:
        """Extract key market events from articles"""
        key_events = []
        
        for article in articles:
            if article['market_impact'] > 0.5:  # High impact articles
                event = {
                    'title': article['title'],
                    'description': article['description'],
                    'source': article['source'],
                    'published_at': article['published_at'],
                    'impact': article['market_impact'],
                    'categories': article['categories'],
                    'tickers': article.get('tickers_mentioned', []),
                    'url': article['url']
                }
                key_events.append(event)
        
        return key_events[:10]  # Return top 10 key events
    
    def _get_fallback_response(self, ticker: str) -> Dict[str, Any]:
        """Fallback response when NewsAPI is unavailable"""
        return {
            'ticker': ticker.upper(),
            'total_articles': 0,
            'processed_articles': 0,
            'articles': [],
            'market_impact': {
                'overall_impact': 'Unknown',
                'sentiment': 'Neutral',
                'key_themes': []
            },
            'timestamp': datetime.now().isoformat(),
            'status': 'error',
            'error': 'NewsAPI temporarily unavailable'
        }
    
    def _get_fallback_market_response(self) -> Dict[str, Any]:
        """Fallback response for market news when NewsAPI is unavailable"""
        return {
            'total_articles': 0,
            'processed_articles': 0,
            'articles': [],
            'market_sentiment': {
                'sentiment': 'Neutral',
                'score': 0.0,
                'confidence': 0.0
            },
            'key_events': [],
            'timestamp': datetime.now().isoformat(),
            'status': 'error',
            'error': 'NewsAPI temporarily unavailable'
        }

# Initialize global news service
news_service = NewsService()

def get_news_for_ticker(ticker: str, hours_back: int = 24) -> Dict[str, Any]:
    """Get news for a specific ticker"""
    return news_service.get_news_for_ticker(ticker, hours_back)

def get_market_news(hours_back: int = 24) -> Dict[str, Any]:
    """Get general market news"""
    return news_service.get_market_news(hours_back)

# Test the service
if __name__ == "__main__":
    # Test with AAPL
    print("Testing NewsAPI integration...")
    result = get_news_for_ticker("AAPL", 24)
    print(f"Found {result['total_articles']} articles for AAPL")
    print(f"Market impact: {result['market_impact']['overall_impact']}")
    print(f"Sentiment: {result['market_impact']['sentiment']}")
    
    # Test market news
    market_news = get_market_news(24)
    print(f"Found {market_news['total_articles']} market articles")
    print(f"Market sentiment: {market_news['market_sentiment']['sentiment']}")




