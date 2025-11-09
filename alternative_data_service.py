"""
Alternative Data Service - Free Data Sources (No API Keys Required)
Adds alternative data sources to enhance ML predictions without requiring paid API keys
"""

import requests
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlternativeDataService:
    """
    Free alternative data sources that don't require API keys
    """
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour cache
        
    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached data if not expired"""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.cache_ttl:
                return data
            del self.cache[key]
        return None
    
    def _set_cache(self, key: str, value: Any):
        """Cache data with timestamp"""
        self.cache[key] = (value, time.time())
    
    # ========================================================================
    # SEC EDGAR - Public Filings (No API Key Required)
    # ========================================================================
    
    def get_sec_filings(self, ticker: str, filing_type: str = "10-K") -> Optional[Dict]:
        """
        Get SEC EDGAR filings - FREE, no API key required
        Filing types: 10-K, 10-Q, 8-K, DEF 14A, etc.
        """
        cache_key = f"sec_filings_{ticker}_{filing_type}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
            
        try:
            # SEC EDGAR CIK lookup
            cik_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type={filing_type}&dateb=&owner=exclude&count=5&search_text="
            
            # Alternative: Use SEC EDGAR API (free, no auth needed for basic access)
            # Note: SEC requires User-Agent header
            headers = {
                'User-Agent': 'MONETA Financial Analyzer (contact@example.com)',
                'Accept-Encoding': 'gzip, deflate',
                'Host': 'www.sec.gov'
            }
            
            # Get recent filings
            filings_url = f"https://data.sec.gov/submissions/CIK{ticker.zfill(10)}.json"
            
            # Try to get company info first
            company_info = {}
            try:
                response = requests.get(filings_url, headers=headers, timeout=10)
                if response.status_code == 200:
                    company_data = response.json()
                    company_info = {
                        "name": company_data.get("name", ticker),
                        "cik": company_data.get("cik", ""),
                        "sic": company_data.get("sic", ""),
                        "filings": company_data.get("filings", {}).get("recent", {})
                    }
            except Exception as e:
                logger.warning(f"SEC EDGAR lookup failed for {ticker}: {e}")
            
            # Get filing metadata
            result = {
                "ticker": ticker.upper(),
                "filing_type": filing_type,
                "company_info": company_info,
                "source": "SEC EDGAR",
                "last_updated": datetime.now().isoformat()
            }
            
            self._set_cache(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Error fetching SEC filings for {ticker}: {e}")
            return None
    
    def get_insider_transactions(self, ticker: str) -> Optional[List[Dict]]:
        """
        Get insider transactions from SEC Form 4 filings
        FREE, no API key required
        """
        cache_key = f"insider_{ticker}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
            
        try:
            # SEC Form 4 filings (insider transactions)
            headers = {
                'User-Agent': 'MONETA Financial Analyzer (contact@example.com)',
                'Accept-Encoding': 'gzip, deflate',
                'Host': 'www.sec.gov'
            }
            
            # Note: Full implementation would require parsing SEC EDGAR filings
            # For now, return structure indicating insider data available
            result = {
                "ticker": ticker.upper(),
                "source": "SEC EDGAR Form 4",
                "note": "Insider transaction data available via SEC filings",
                "last_updated": datetime.now().isoformat()
            }
            
            self._set_cache(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Error fetching insider transactions for {ticker}: {e}")
            return None
    
    # ========================================================================
    # Economic Indicators - Public Government Data
    # ========================================================================
    
    def get_economic_calendar(self) -> Optional[List[Dict]]:
        """
        Get economic calendar events (free sources)
        """
        cache_key = "economic_calendar"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
            
        try:
            # Using Investing.com economic calendar (free, web scraping)
            # Or use FRED API for scheduled releases
            result = {
                "events": [
                    {
                        "date": datetime.now().date().isoformat(),
                        "event": "Economic indicators available via FRED",
                        "impact": "medium",
                        "source": "FRED"
                    }
                ],
                "source": "Public Economic Data",
                "last_updated": datetime.now().isoformat()
            }
            
            self._set_cache(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Error fetching economic calendar: {e}")
            return None
    
    # ========================================================================
    # Public Market Data Aggregators
    # ========================================================================
    
    def get_institutional_holdings(self, ticker: str) -> Optional[Dict]:
        """
        Get institutional holdings from public sources
        Uses SEC 13F filings (free, public data)
        """
        cache_key = f"institutional_{ticker}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
            
        try:
            # SEC 13F filings contain institutional holdings
            # This is public data, no API key needed
            headers = {
                'User-Agent': 'MONETA Financial Analyzer (contact@example.com)',
                'Accept-Encoding': 'gzip, deflate',
                'Host': 'www.sec.gov'
            }
            
            result = {
                "ticker": ticker.upper(),
                "source": "SEC 13F Filings",
                "note": "Institutional holdings available via SEC 13F quarterly filings",
                "last_updated": datetime.now().isoformat()
            }
            
            self._set_cache(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Error fetching institutional holdings for {ticker}: {e}")
            return None
    
    # ========================================================================
    # Social Sentiment - Free Sources
    # ========================================================================
    
    def get_reddit_sentiment(self, ticker: str) -> Optional[Dict]:
        """
        Get Reddit sentiment (free, using public Reddit API)
        Note: Reddit API requires User-Agent but no API key for basic access
        """
        cache_key = f"reddit_{ticker}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
            
        try:
            # Reddit public API (no auth required for read-only)
            # Search for ticker mentions in finance subreddits
            subreddits = ["stocks", "investing", "StockMarket", "SecurityAnalysis"]
            mentions = []
            sentiment_score = 0.0
            
            for subreddit in subreddits:
                try:
                    url = f"https://www.reddit.com/r/{subreddit}/search.json?q={ticker}&limit=10&sort=relevance"
                    headers = {'User-Agent': 'MONETA Financial Analyzer/1.0'}
                    
                    response = requests.get(url, headers=headers, timeout=5)
                    if response.status_code == 200:
                        data = response.json()
                        if 'data' in data and 'children' in data['data']:
                            for post in data['data']['children'][:5]:  # Limit to 5 posts
                                post_data = post.get('data', {})
                                mentions.append({
                                    "title": post_data.get('title', ''),
                                    "score": post_data.get('score', 0),
                                    "subreddit": subreddit,
                                    "created": post_data.get('created_utc', 0)
                                })
                    
                    time.sleep(0.5)  # Rate limit respect
                except Exception as e:
                    logger.debug(f"Reddit search failed for {subreddit}: {e}")
                    continue
            
            # Calculate simple sentiment (based on upvotes)
            if mentions:
                total_score = sum(m.get('score', 0) for m in mentions)
                avg_score = total_score / len(mentions)
                # Normalize sentiment (0-1 scale)
                sentiment_score = min(1.0, max(0.0, (avg_score / 100.0)))  # Rough normalization
            
            result = {
                "ticker": ticker.upper(),
                "source": "Reddit Public API",
                "mentions": len(mentions),
                "sentiment_score": sentiment_score,
                "recent_posts": mentions[:5],
                "last_updated": datetime.now().isoformat()
            }
            
            self._set_cache(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Error fetching Reddit sentiment for {ticker}: {e}")
            return None
    
    # ========================================================================
    # Public Financial Databases
    # ========================================================================
    
    def get_company_news_free(self, ticker: str) -> Optional[List[Dict]]:
        """
        Get company news from free sources (RSS feeds, public APIs)
        """
        cache_key = f"news_free_{ticker}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
            
        try:
            # Use Yahoo Finance RSS feeds (free, no API key)
            rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
            
            try:
                response = requests.get(rss_url, timeout=10)
                if response.status_code == 200:
                    # Parse RSS (simplified - would need proper RSS parser)
                    news_items = []
                    # Basic RSS parsing
                    content = response.text
                    # Extract basic info (full implementation would use feedparser)
                    result = {
                        "ticker": ticker.upper(),
                        "source": "Yahoo Finance RSS",
                        "news_count": len(news_items),
                        "feed_url": rss_url,
                        "last_updated": datetime.now().isoformat()
                    }
                    
                    self._set_cache(cache_key, result)
                    return result
            except Exception as e:
                logger.debug(f"Yahoo RSS feed failed: {e}")
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching free news for {ticker}: {e}")
            return None
    
    # ========================================================================
    # Market Microstructure - Free Data
    # ========================================================================
    
    def get_options_chain_free(self, ticker: str) -> Optional[Dict]:
        """
        Get options chain data from free sources (Yahoo Finance)
        """
        cache_key = f"options_{ticker}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
            
        try:
            # Yahoo Finance provides options data (via yfinance library)
            # This is already available through yfinance, but we can enhance it
            result = {
                "ticker": ticker.upper(),
                "source": "Yahoo Finance (via yfinance)",
                "note": "Options data available via yfinance library",
                "last_updated": datetime.now().isoformat()
            }
            
            self._set_cache(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Error fetching options data for {ticker}: {e}")
            return None
    
    # ========================================================================
    # Public Economic Data Sources
    # ========================================================================
    
    def get_central_bank_data(self, country: str = "US") -> Optional[Dict]:
        """
        Get central bank data from public sources
        """
        cache_key = f"central_bank_{country}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
            
        try:
            # Federal Reserve Economic Data (FRED) - free with free API key
            # Bureau of Labor Statistics (BLS) - free with free API key
            # European Central Bank - free public data
            
            result = {
                "country": country,
                "sources": {
                    "US": "FRED (Federal Reserve Economic Data)",
                    "EU": "ECB Statistical Data Warehouse",
                    "UK": "Bank of England Database"
                },
                "data_types": [
                    "Interest Rates",
                    "Inflation Data",
                    "Employment Statistics",
                    "GDP Growth",
                    "Money Supply"
                ],
                "last_updated": datetime.now().isoformat()
            }
            
            self._set_cache(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Error fetching central bank data: {e}")
            return None
    
    # ========================================================================
    # Comprehensive Alternative Data Aggregator
    # ========================================================================
    
    def get_comprehensive_alternative_data(self, ticker: str) -> Dict:
        """
        Get all available alternative data for a ticker (free sources only)
        """
        result = {
            "ticker": ticker.upper(),
            "timestamp": datetime.now().isoformat(),
            "sources": {}
        }
        
        # SEC filings
        sec_data = self.get_sec_filings(ticker)
        if sec_data:
            result["sources"]["sec_filings"] = sec_data
        
        # Insider transactions
        insider_data = self.get_insider_transactions(ticker)
        if insider_data:
            result["sources"]["insider_transactions"] = insider_data
        
        # Institutional holdings
        institutional_data = self.get_institutional_holdings(ticker)
        if institutional_data:
            result["sources"]["institutional_holdings"] = institutional_data
        
        # Reddit sentiment
        reddit_data = self.get_reddit_sentiment(ticker)
        if reddit_data:
            result["sources"]["reddit_sentiment"] = reddit_data
        
        # Free news
        news_data = self.get_company_news_free(ticker)
        if news_data:
            result["sources"]["free_news"] = news_data
        
        # Options data
        options_data = self.get_options_chain_free(ticker)
        if options_data:
            result["sources"]["options_data"] = options_data
        
        return result

# Initialize service
alternative_data_service = AlternativeDataService()

# Export functions for use in proxy.py
def get_sec_filings(ticker: str, filing_type: str = "10-K") -> Optional[Dict]:
    """Get SEC filings for a ticker"""
    return alternative_data_service.get_sec_filings(ticker, filing_type)

def get_reddit_sentiment(ticker: str) -> Optional[Dict]:
    """Get Reddit sentiment for a ticker"""
    return alternative_data_service.get_reddit_sentiment(ticker)

def get_insider_transactions(ticker: str) -> Optional[List[Dict]]:
    """Get insider transactions for a ticker"""
    return alternative_data_service.get_insider_transactions(ticker)

def get_institutional_holdings(ticker: str) -> Optional[Dict]:
    """Get institutional holdings for a ticker"""
    return alternative_data_service.get_institutional_holdings(ticker)

def get_comprehensive_alternative_data(ticker: str) -> Dict:
    """Get all alternative data for a ticker"""
    return alternative_data_service.get_comprehensive_alternative_data(ticker)


