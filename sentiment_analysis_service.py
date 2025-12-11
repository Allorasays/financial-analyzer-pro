"""
Sentiment Analysis Service for Financial Analyzer Pro
News and social media sentiment analysis for enhanced ML predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import re
import warnings
warnings.filterwarnings('ignore')

# Sentiment analysis imports with graceful fallbacks
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class SentimentAnalysisService:
    """Sentiment analysis service for financial news and social media"""
    
    def __init__(self):
        self.news_sentiment_cache = {}
        self.social_sentiment_cache = {}
        self.sentiment_models = {}
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize sentiment analysis models"""
        # TextBlob (always available if installed)
        if TEXTBLOB_AVAILABLE:
            self.sentiment_models['textblob'] = TextBlob
        
        # NLTK VADER
        if NLTK_AVAILABLE:
            try:
                # Download required NLTK data
                import nltk
                nltk.download('vader_lexicon', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('punkt', quiet=True)
                
                self.sentiment_models['vader'] = SentimentIntensityAnalyzer()
                self.sentiment_models['stopwords'] = set(stopwords.words('english'))
            except Exception as e:
                st.warning(f"NLTK initialization failed: {str(e)}")
        
        # Transformers (Hugging Face)
        if TRANSFORMERS_AVAILABLE:
            try:
                # Financial sentiment model
                self.sentiment_models['financial_bert'] = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    tokenizer="ProsusAI/finbert"
                )
            except Exception as e:
                st.warning(f"Financial BERT initialization failed: {str(e)}")
                try:
                    # Fallback to general sentiment model
                    self.sentiment_models['general_bert'] = pipeline(
                        "sentiment-analysis",
                        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
                    )
                except Exception as e2:
                    st.warning(f"General BERT initialization failed: {str(e2)}")
    
    def analyze_news_sentiment(self, symbol: str, days_back: int = 7) -> Dict[str, Any]:
        """Analyze news sentiment for a symbol"""
        cache_key = f"news_{symbol}_{days_back}"
        
        if cache_key in self.news_sentiment_cache:
            return self.news_sentiment_cache[cache_key]
        
        # Simulate news data (in real implementation, use news API)
        news_data = self._simulate_news_data(symbol, days_back)
        
        if not news_data:
            return {"error": "No news data available"}
        
        # Analyze sentiment for each news item
        sentiment_scores = []
        for article in news_data:
            sentiment = self._analyze_text_sentiment(article['title'] + " " + article['content'])
            sentiment_scores.append({
                'date': article['date'],
                'title': article['title'],
                'sentiment': sentiment,
                'source': article['source']
            })
        
        # Calculate aggregate sentiment
        aggregate_sentiment = self._calculate_aggregate_sentiment(sentiment_scores)
        
        result = {
            'symbol': symbol,
            'period_days': days_back,
            'total_articles': len(news_data),
            'sentiment_scores': sentiment_scores,
            'aggregate_sentiment': aggregate_sentiment,
            'last_updated': datetime.now().isoformat()
        }
        
        self.news_sentiment_cache[cache_key] = result
        return result
    
    def analyze_social_sentiment(self, symbol: str, platform: str = "twitter", 
                               days_back: int = 7) -> Dict[str, Any]:
        """Analyze social media sentiment for a symbol"""
        cache_key = f"social_{symbol}_{platform}_{days_back}"
        
        if cache_key in self.social_sentiment_cache:
            return self.social_sentiment_cache[cache_key]
        
        # Simulate social media data (in real implementation, use social media API)
        social_data = self._simulate_social_data(symbol, platform, days_back)
        
        if not social_data:
            return {"error": "No social media data available"}
        
        # Analyze sentiment for each post
        sentiment_scores = []
        for post in social_data:
            sentiment = self._analyze_text_sentiment(post['text'])
            sentiment_scores.append({
                'date': post['date'],
                'text': post['text'][:100] + "...",  # Truncate for display
                'sentiment': sentiment,
                'platform': post['platform'],
                'engagement': post.get('engagement', 0)
            })
        
        # Calculate aggregate sentiment
        aggregate_sentiment = self._calculate_aggregate_sentiment(sentiment_scores)
        
        result = {
            'symbol': symbol,
            'platform': platform,
            'period_days': days_back,
            'total_posts': len(social_data),
            'sentiment_scores': sentiment_scores,
            'aggregate_sentiment': aggregate_sentiment,
            'last_updated': datetime.now().isoformat()
        }
        
        self.social_sentiment_cache[cache_key] = result
        return result
    
    def _simulate_news_data(self, symbol: str, days_back: int) -> List[Dict[str, Any]]:
        """Simulate news data (replace with real news API)"""
        # This is a simulation - in production, use real news APIs like:
        # - NewsAPI
        # - Alpha Vantage News
        # - Financial Modeling Prep News
        # - Benzinga News
        
        news_items = []
        base_date = datetime.now() - timedelta(days=days_back)
        
        # Simulate different types of news with varying sentiment
        news_templates = [
            f"{symbol} reports strong quarterly earnings, beating analyst expectations",
            f"{symbol} announces major partnership deal, boosting investor confidence",
            f"{symbol} faces regulatory challenges, causing market uncertainty",
            f"{symbol} stock price surges on positive analyst upgrades",
            f"{symbol} experiences technical difficulties, affecting user experience",
            f"{symbol} launches innovative product, gaining market traction",
            f"{symbol} faces increased competition, impacting market share",
            f"{symbol} receives positive analyst coverage, driving investor interest",
            f"{symbol} announces cost-cutting measures, improving profitability",
            f"{symbol} stock volatility increases due to market conditions"
        ]
        
        for i in range(min(20, days_back * 3)):  # Simulate 3 articles per day
            news_items.append({
                'date': (base_date + timedelta(days=i//3, hours=i%24)).isoformat(),
                'title': news_templates[i % len(news_templates)],
                'content': f"Detailed analysis of {symbol} performance and market outlook. " + 
                          f"Analysts are closely watching the company's strategic moves and " +
                          f"market positioning in the current economic environment.",
                'source': f"Financial News {i%5 + 1}",
                'url': f"https://example.com/news/{symbol}_{i}"
            })
        
        return news_items
    
    def _simulate_social_data(self, symbol: str, platform: str, days_back: int) -> List[Dict[str, Any]]:
        """Simulate social media data (replace with real social media API)"""
        # This is a simulation - in production, use real social media APIs like:
        # - Twitter API v2
        # - Reddit API
        # - StockTwits API
        
        social_posts = []
        base_date = datetime.now() - timedelta(days=days_back)
        
        # Simulate different types of social media posts
        post_templates = [
            f"$SYMBOL looking bullish today! Great earnings report 📈",
            f"$SYMBOL stock is overvalued, time to sell 💸",
            f"$SYMBOL breaking resistance levels, moon incoming 🚀",
            f"$SYMBOL disappointing guidance, bearish outlook 📉",
            f"$SYMBOL solid fundamentals, long-term hold 💎",
            f"$SYMBOL technical analysis shows strong support levels",
            f"$SYMBOL management team making smart strategic moves",
            f"$SYMBOL facing headwinds in current market conditions",
            f"$SYMBOL innovation pipeline looks promising for growth",
            f"$SYMBOL regulatory concerns weighing on stock price"
        ]
        
        for i in range(min(50, days_back * 10)):  # Simulate 10 posts per day
            post_text = post_templates[i % len(post_templates)].replace("SYMBOL", symbol)
            social_posts.append({
                'date': (base_date + timedelta(days=i//10, hours=i%24)).isoformat(),
                'text': post_text,
                'platform': platform,
                'engagement': np.random.randint(1, 1000),
                'author': f"user_{i%20}"
            })
        
        return social_posts
    
    def _analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text using available models"""
        sentiment_scores = {}
        
        # Clean text
        cleaned_text = self._clean_text(text)
        
        # TextBlob sentiment
        if 'textblob' in self.sentiment_models:
            try:
                blob = self.sentiment_models['textblob'](cleaned_text)
                sentiment_scores['textblob'] = {
                    'polarity': blob.sentiment.polarity,  # -1 to 1
                    'subjectivity': blob.sentiment.subjectivity,  # 0 to 1
                    'label': 'positive' if blob.sentiment.polarity > 0.1 else 
                            'negative' if blob.sentiment.polarity < -0.1 else 'neutral'
                }
            except Exception as e:
                st.warning(f"TextBlob analysis failed: {str(e)}")
        
        # VADER sentiment
        if 'vader' in self.sentiment_models:
            try:
                vader_scores = self.sentiment_models['vader'].polarity_scores(cleaned_text)
                sentiment_scores['vader'] = {
                    'compound': vader_scores['compound'],
                    'positive': vader_scores['pos'],
                    'negative': vader_scores['neg'],
                    'neutral': vader_scores['neu'],
                    'label': 'positive' if vader_scores['compound'] > 0.05 else
                            'negative' if vader_scores['compound'] < -0.05 else 'neutral'
                }
            except Exception as e:
                st.warning(f"VADER analysis failed: {str(e)}")
        
        # BERT sentiment
        if 'financial_bert' in self.sentiment_models:
            try:
                bert_result = self.sentiment_models['financial_bert'](cleaned_text[:512])  # Truncate for BERT
                sentiment_scores['financial_bert'] = {
                    'label': bert_result[0]['label'],
                    'score': bert_result[0]['score']
                }
            except Exception as e:
                st.warning(f"Financial BERT analysis failed: {str(e)}")
        elif 'general_bert' in self.sentiment_models:
            try:
                bert_result = self.sentiment_models['general_bert'](cleaned_text[:512])
                sentiment_scores['general_bert'] = {
                    'label': bert_result[0]['label'],
                    'score': bert_result[0]['score']
                }
            except Exception as e:
                st.warning(f"General BERT analysis failed: {str(e)}")
        
        return sentiment_scores
    
    def _clean_text(self, text: str) -> str:
        """Clean text for sentiment analysis"""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove stopwords if available
        if 'stopwords' in self.sentiment_models:
            words = text.lower().split()
            words = [word for word in words if word not in self.sentiment_models['stopwords']]
            text = ' '.join(words)
        
        return text
    
    def _calculate_aggregate_sentiment(self, sentiment_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate sentiment from individual scores"""
        if not sentiment_scores:
            return {"error": "No sentiment scores available"}
        
        # Collect all sentiment scores
        all_scores = []
        for item in sentiment_scores:
            if 'sentiment' in item:
                all_scores.append(item['sentiment'])
        
        if not all_scores:
            return {"error": "No valid sentiment scores"}
        
        # Calculate aggregate metrics
        aggregate = {
            'total_items': len(sentiment_scores),
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0,
            'average_polarity': 0,
            'average_compound': 0,
            'sentiment_distribution': {},
            'confidence_score': 0
        }
        
        # Aggregate TextBlob scores
        textblob_scores = [s.get('textblob', {}) for s in all_scores if 'textblob' in s]
        if textblob_scores:
            polarities = [s.get('polarity', 0) for s in textblob_scores]
            aggregate['average_polarity'] = np.mean(polarities)
            
            for s in textblob_scores:
                label = s.get('label', 'neutral')
                if label == 'positive':
                    aggregate['positive_count'] += 1
                elif label == 'negative':
                    aggregate['negative_count'] += 1
                else:
                    aggregate['neutral_count'] += 1
        
        # Aggregate VADER scores
        vader_scores = [s.get('vader', {}) for s in all_scores if 'vader' in s]
        if vader_scores:
            compounds = [s.get('compound', 0) for s in vader_scores]
            aggregate['average_compound'] = np.mean(compounds)
        
        # Calculate sentiment distribution
        total_items = aggregate['total_items']
        if total_items > 0:
            aggregate['sentiment_distribution'] = {
                'positive': aggregate['positive_count'] / total_items,
                'negative': aggregate['negative_count'] / total_items,
                'neutral': aggregate['neutral_count'] / total_items
            }
        
        # Calculate confidence score (based on consistency)
        if len(textblob_scores) > 1:
            polarities = [s.get('polarity', 0) for s in textblob_scores]
            aggregate['confidence_score'] = 1 - np.std(polarities)  # Lower std = higher confidence
        
        # Overall sentiment label
        if aggregate['average_polarity'] > 0.1:
            aggregate['overall_sentiment'] = 'positive'
        elif aggregate['average_polarity'] < -0.1:
            aggregate['overall_sentiment'] = 'negative'
        else:
            aggregate['overall_sentiment'] = 'neutral'
        
        return aggregate
    
    def get_sentiment_features(self, symbol: str, days_back: int = 7) -> Dict[str, float]:
        """Get sentiment features for ML models"""
        # Get news sentiment
        news_sentiment = self.analyze_news_sentiment(symbol, days_back)
        
        # Get social sentiment
        social_sentiment = self.analyze_social_sentiment(symbol, "twitter", days_back)
        
        features = {}
        
        # News sentiment features
        if 'aggregate_sentiment' in news_sentiment:
            news_agg = news_sentiment['aggregate_sentiment']
            features['news_polarity'] = news_agg.get('average_polarity', 0)
            features['news_compound'] = news_agg.get('average_compound', 0)
            features['news_confidence'] = news_agg.get('confidence_score', 0)
            features['news_positive_ratio'] = news_agg.get('sentiment_distribution', {}).get('positive', 0)
            features['news_negative_ratio'] = news_agg.get('sentiment_distribution', {}).get('negative', 0)
            features['news_article_count'] = news_agg.get('total_items', 0)
        
        # Social sentiment features
        if 'aggregate_sentiment' in social_sentiment:
            social_agg = social_sentiment['aggregate_sentiment']
            features['social_polarity'] = social_agg.get('average_polarity', 0)
            features['social_compound'] = social_agg.get('average_compound', 0)
            features['social_confidence'] = social_agg.get('confidence_score', 0)
            features['social_positive_ratio'] = social_agg.get('sentiment_distribution', {}).get('positive', 0)
            features['social_negative_ratio'] = social_agg.get('sentiment_distribution', {}).get('negative', 0)
            features['social_post_count'] = social_agg.get('total_items', 0)
        
        # Combined sentiment features
        features['combined_sentiment'] = (features.get('news_polarity', 0) + features.get('social_polarity', 0)) / 2
        features['sentiment_volatility'] = np.std([features.get('news_polarity', 0), features.get('social_polarity', 0)])
        
        return features
    
    def display_sentiment_analysis(self, symbol: str, days_back: int = 7):
        """Display sentiment analysis dashboard"""
        st.subheader(f"📰 Sentiment Analysis for {symbol}")
        
        # News sentiment
        with st.expander("📰 News Sentiment", expanded=True):
            news_sentiment = self.analyze_news_sentiment(symbol, days_back)
            
            if 'error' in news_sentiment:
                st.error(f"News analysis error: {news_sentiment['error']}")
            else:
                agg = news_sentiment['aggregate_sentiment']
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Articles", agg.get('total_items', 0))
                with col2:
                    st.metric("Polarity", f"{agg.get('average_polarity', 0):.3f}")
                with col3:
                    st.metric("Confidence", f"{agg.get('confidence_score', 0):.3f}")
                with col4:
                    sentiment_label = agg.get('overall_sentiment', 'neutral')
                    color = "🟢" if sentiment_label == 'positive' else "🔴" if sentiment_label == 'negative' else "⚪"
                    st.metric("Overall", f"{color} {sentiment_label.title()}")
                
                # Sentiment distribution
                if 'sentiment_distribution' in agg:
                    dist = agg['sentiment_distribution']
                    fig = px.pie(
                        values=[dist.get('positive', 0), dist.get('negative', 0), dist.get('neutral', 0)],
                        names=['Positive', 'Negative', 'Neutral'],
                        title="News Sentiment Distribution",
                        color_discrete_map={'Positive': 'green', 'Negative': 'red', 'Neutral': 'gray'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Social sentiment
        with st.expander("🐦 Social Media Sentiment", expanded=True):
            social_sentiment = self.analyze_social_sentiment(symbol, "twitter", days_back)
            
            if 'error' in social_sentiment:
                st.error(f"Social analysis error: {social_sentiment['error']}")
            else:
                agg = social_sentiment['aggregate_sentiment']
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Posts", agg.get('total_items', 0))
                with col2:
                    st.metric("Polarity", f"{agg.get('average_polarity', 0):.3f}")
                with col3:
                    st.metric("Confidence", f"{agg.get('confidence_score', 0):.3f}")
                with col4:
                    sentiment_label = agg.get('overall_sentiment', 'neutral')
                    color = "🟢" if sentiment_label == 'positive' else "🔴" if sentiment_label == 'negative' else "⚪"
                    st.metric("Overall", f"{color} {sentiment_label.title()}")
                
                # Sentiment distribution
                if 'sentiment_distribution' in agg:
                    dist = agg['sentiment_distribution']
                    fig = px.pie(
                        values=[dist.get('positive', 0), dist.get('negative', 0), dist.get('neutral', 0)],
                        names=['Positive', 'Negative', 'Neutral'],
                        title="Social Media Sentiment Distribution",
                        color_discrete_map={'Positive': 'green', 'Negative': 'red', 'Neutral': 'gray'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

# Global instance
sentiment_service = SentimentAnalysisService()
