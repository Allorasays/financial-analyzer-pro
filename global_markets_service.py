"""
Global Markets, Forex, and Cryptocurrency Service
Comprehensive service for international markets, forex pairs, and crypto assets
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Try to import crypto and forex libraries with fallbacks
try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False

try:
    from forex_python.converter import CurrencyRates
    FOREX_PYTHON_AVAILABLE = True
except ImportError:
    FOREX_PYTHON_AVAILABLE = False

class GlobalMarketsService:
    """Service for global markets, forex, and cryptocurrency data"""
    
    def __init__(self):
        self.cache = {}
        self.cache_timestamps = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Initialize crypto exchange if available
        if CCXT_AVAILABLE:
            try:
                self.binance = ccxt.binance()
                self.coinbase = ccxt.coinbasepro()
            except:
                self.binance = None
                self.coinbase = None
        
        # Initialize forex converter if available
        if FOREX_PYTHON_AVAILABLE:
            try:
                self.currency_rates = CurrencyRates()
            except:
                self.currency_rates = None
    
    def _get_cached_data(self, key: str) -> Optional[any]:
        """Get cached data if not expired"""
        if key in self.cache and key in self.cache_timestamps:
            if (datetime.now() - self.cache_timestamps[key]).seconds < self.cache_ttl:
                return self.cache[key]
            else:
                # Remove expired cache
                self.cache.pop(key, None)
                self.cache_timestamps.pop(key, None)
        return None
    
    def _set_cached_data(self, key: str, data: any):
        """Set cached data with timestamp"""
        self.cache[key] = data
        self.cache_timestamps[key] = datetime.now()
    
    def get_global_markets_overview(self) -> Dict:
        """Get overview of global markets"""
        cache_key = "global_markets_overview"
        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            return cached_data
        
        try:
            # Major global indices
            indices = {
                'US': ['^GSPC', '^IXIC', '^DJI', '^RUT'],  # S&P 500, NASDAQ, Dow, Russell 2000
                'Europe': ['^FTSE', '^GDAXI', '^FCHI', '^STOXX50E'],  # FTSE, DAX, CAC, Euro Stoxx
                'Asia': ['^N225', '^HSI', '000001.SS', '^AXJO'],  # Nikkei, Hang Seng, Shanghai, ASX
                'Emerging': ['^BVSP', '^MXX', '^BSESN', '^JKSE']  # Bovespa, IPC, Sensex, Jakarta
            }
            
            markets_data = {}
            
            for region, symbols in indices.items():
                region_data = []
                for symbol in symbols:
                    try:
                        ticker = yf.Ticker(symbol)
                        info = ticker.info
                        hist = ticker.history(period="1d")
                        
                        if not hist.empty:
                            current_price = hist['Close'].iloc[-1]
                            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                            change = current_price - prev_close
                            change_pct = (change / prev_close) * 100 if prev_close != 0 else 0
                            
                            region_data.append({
                                'symbol': symbol,
                                'name': info.get('longName', symbol),
                                'price': round(current_price, 2),
                                'change': round(change, 2),
                                'change_pct': round(change_pct, 2),
                                'volume': hist['Volume'].iloc[-1] if 'Volume' in hist.columns else 0
                            })
                    except Exception as e:
                        print(f"Error fetching {symbol}: {e}")
                        continue
                
                markets_data[region] = region_data
            
            result = {
                'timestamp': datetime.now().isoformat(),
                'markets': markets_data,
                'status': 'success'
            }
            
            self._set_cached_data(cache_key, result)
            return result
            
        except Exception as e:
            return {
                'timestamp': datetime.now().isoformat(),
                'markets': {},
                'status': 'error',
                'error': str(e)
            }
    
    def get_forex_rates(self, base_currency: str = 'USD') -> Dict:
        """Get forex exchange rates"""
        cache_key = f"forex_rates_{base_currency}"
        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            return cached_data
        
        try:
            # Major currency pairs
            major_pairs = [
                'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCHF=X',
                'AUDUSD=X', 'USDCAD=X', 'NZDUSD=X', 'USDSEK=X',
                'USDNOK=X', 'USDDKK=X', 'USDPLN=X', 'USDCZK=X',
                'USDHUF=X', 'USDTRY=X', 'USDZAR=X', 'USDMXN=X'
            ]
            
            forex_data = []
            
            for pair in major_pairs:
                try:
                    ticker = yf.Ticker(pair)
                    hist = ticker.history(period="1d")
                    
                    if not hist.empty:
                        current_rate = hist['Close'].iloc[-1]
                        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_rate
                        change = current_rate - prev_close
                        change_pct = (change / prev_close) * 100 if prev_close != 0 else 0
                        
                        # Extract currencies from pair symbol
                        currencies = pair.replace('=X', '').replace('USD', 'USD')
                        if len(currencies) == 6:
                            from_curr = currencies[:3]
                            to_curr = currencies[3:]
                        else:
                            from_curr = 'USD'
                            to_curr = currencies.replace('USD', '')
                        
                        forex_data.append({
                            'pair': pair,
                            'from_currency': from_curr,
                            'to_currency': to_curr,
                            'rate': round(current_rate, 4),
                            'change': round(change, 4),
                            'change_pct': round(change_pct, 2)
                        })
                except Exception as e:
                    print(f"Error fetching {pair}: {e}")
                    continue
            
            result = {
                'timestamp': datetime.now().isoformat(),
                'base_currency': base_currency,
                'rates': forex_data,
                'status': 'success'
            }
            
            self._set_cached_data(cache_key, result)
            return result
            
        except Exception as e:
            return {
                'timestamp': datetime.now().isoformat(),
                'base_currency': base_currency,
                'rates': [],
                'status': 'error',
                'error': str(e)
            }
    
    def get_cryptocurrency_data(self) -> Dict:
        """Get cryptocurrency market data"""
        cache_key = "cryptocurrency_data"
        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            return cached_data
        
        try:
            # Major cryptocurrencies
            crypto_symbols = [
                'BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'SOL-USD',
                'XRP-USD', 'DOT-USD', 'DOGE-USD', 'AVAX-USD', 'MATIC-USD',
                'LINK-USD', 'UNI-USD', 'LTC-USD', 'BCH-USD', 'ATOM-USD'
            ]
            
            crypto_data = []
            
            for symbol in crypto_symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="1d")
                    info = ticker.info
                    
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                        change = current_price - prev_close
                        change_pct = (change / prev_close) * 100 if prev_close != 0 else 0
                        
                        # Get additional info
                        market_cap = info.get('marketCap', 0)
                        volume_24h = hist['Volume'].iloc[-1] if 'Volume' in hist.columns else 0
                        
                        crypto_data.append({
                            'symbol': symbol,
                            'name': info.get('longName', symbol),
                            'price': round(current_price, 2),
                            'change': round(change, 2),
                            'change_pct': round(change_pct, 2),
                            'market_cap': market_cap,
                            'volume_24h': volume_24h,
                            'rank': len(crypto_data) + 1
                        })
                except Exception as e:
                    print(f"Error fetching {symbol}: {e}")
                    continue
            
            # Sort by market cap
            crypto_data.sort(key=lambda x: x['market_cap'], reverse=True)
            
            result = {
                'timestamp': datetime.now().isoformat(),
                'cryptocurrencies': crypto_data,
                'status': 'success'
            }
            
            self._set_cached_data(cache_key, result)
            return result
            
        except Exception as e:
            return {
                'timestamp': datetime.now().isoformat(),
                'cryptocurrencies': [],
                'status': 'error',
                'error': str(e)
            }
    
    def get_commodity_prices(self) -> Dict:
        """Get commodity prices"""
        cache_key = "commodity_prices"
        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            return cached_data
        
        try:
            # Major commodities
            commodities = {
                'Gold': 'GC=F',
                'Silver': 'SI=F',
                'Crude Oil': 'CL=F',
                'Natural Gas': 'NG=F',
                'Copper': 'HG=F',
                'Corn': 'ZC=F',
                'Wheat': 'ZW=F',
                'Soybeans': 'ZS=F'
            }
            
            commodity_data = []
            
            for name, symbol in commodities.items():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="1d")
                    
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                        change = current_price - prev_close
                        change_pct = (change / prev_close) * 100 if prev_close != 0 else 0
                        
                        commodity_data.append({
                            'name': name,
                            'symbol': symbol,
                            'price': round(current_price, 2),
                            'change': round(change, 2),
                            'change_pct': round(change_pct, 2),
                            'unit': 'USD'  # Most commodities are priced in USD
                        })
                except Exception as e:
                    print(f"Error fetching {symbol}: {e}")
                    continue
            
            result = {
                'timestamp': datetime.now().isoformat(),
                'commodities': commodity_data,
                'status': 'success'
            }
            
            self._set_cached_data(cache_key, result)
            return result
            
        except Exception as e:
            return {
                'timestamp': datetime.now().isoformat(),
                'commodities': [],
                'status': 'error',
                'error': str(e)
            }
    
    def get_economic_indicators(self) -> Dict:
        """Get economic indicators"""
        cache_key = "economic_indicators"
        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            return cached_data
        
        try:
            # Economic indicators
            indicators = {
                '10-Year Treasury': '^TNX',
                '30-Year Treasury': '^TYX',
                'Dollar Index': 'DX-Y.NYB',
                'VIX': '^VIX',
                'Fed Funds Rate': '^IRX'
            }
            
            indicator_data = []
            
            for name, symbol in indicators.items():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="1d")
                    
                    if not hist.empty:
                        current_value = hist['Close'].iloc[-1]
                        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_value
                        change = current_value - prev_close
                        change_pct = (change / prev_close) * 100 if prev_close != 0 else 0
                        
                        indicator_data.append({
                            'name': name,
                            'symbol': symbol,
                            'value': round(current_value, 2),
                            'change': round(change, 2),
                            'change_pct': round(change_pct, 2),
                            'unit': '%' if 'Treasury' in name or 'Rate' in name else 'Index'
                        })
                except Exception as e:
                    print(f"Error fetching {symbol}: {e}")
                    continue
            
            result = {
                'timestamp': datetime.now().isoformat(),
                'indicators': indicator_data,
                'status': 'success'
            }
            
            self._set_cached_data(cache_key, result)
            return result
            
        except Exception as e:
            return {
                'timestamp': datetime.now().isoformat(),
                'indicators': [],
                'status': 'error',
                'error': str(e)
            }
    
    def convert_currency(self, amount: float, from_currency: str, to_currency: str) -> Dict:
        """Convert currency amount"""
        try:
            if FOREX_PYTHON_AVAILABLE and self.currency_rates:
                # Use forex-python for conversion
                converted_amount = self.currency_rates.convert(from_currency, to_currency, amount)
                return {
                    'amount': amount,
                    'from_currency': from_currency,
                    'to_currency': to_currency,
                    'converted_amount': round(converted_amount, 2),
                    'status': 'success'
                }
            else:
                # Fallback to yfinance
                pair = f"{from_currency}{to_currency}=X"
                ticker = yf.Ticker(pair)
                hist = ticker.history(period="1d")
                
                if not hist.empty:
                    rate = hist['Close'].iloc[-1]
                    converted_amount = amount * rate
                    
                    return {
                        'amount': amount,
                        'from_currency': from_currency,
                        'to_currency': to_currency,
                        'rate': round(rate, 4),
                        'converted_amount': round(converted_amount, 2),
                        'status': 'success'
                    }
                else:
                    return {
                        'amount': amount,
                        'from_currency': from_currency,
                        'to_currency': to_currency,
                        'converted_amount': 0,
                        'status': 'error',
                        'error': 'Unable to fetch exchange rate'
                    }
        except Exception as e:
            return {
                'amount': amount,
                'from_currency': from_currency,
                'to_currency': to_currency,
                'converted_amount': 0,
                'status': 'error',
                'error': str(e)
            }
    
    def get_market_sentiment(self) -> Dict:
        """Get overall market sentiment"""
        try:
            # Get VIX (Fear gauge)
            vix_ticker = yf.Ticker('^VIX')
            vix_hist = vix_ticker.history(period="5d")
            
            if not vix_hist.empty:
                current_vix = vix_hist['Close'].iloc[-1]
                avg_vix = vix_hist['Close'].mean()
                
                # VIX interpretation
                if current_vix < 20:
                    fear_level = "Low Fear"
                    sentiment = "Bullish"
                elif current_vix < 30:
                    fear_level = "Moderate Fear"
                    sentiment = "Neutral"
                else:
                    fear_level = "High Fear"
                    sentiment = "Bearish"
                
                return {
                    'vix': round(current_vix, 2),
                    'avg_vix': round(avg_vix, 2),
                    'fear_level': fear_level,
                    'sentiment': sentiment,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'success'
                }
            else:
                return {
                    'vix': 0,
                    'avg_vix': 0,
                    'fear_level': "Unknown",
                    'sentiment': "Unknown",
                    'timestamp': datetime.now().isoformat(),
                    'status': 'error',
                    'error': 'Unable to fetch VIX data'
                }
        except Exception as e:
            return {
                'vix': 0,
                'avg_vix': 0,
                'fear_level': "Unknown",
                'sentiment': "Unknown",
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'error': str(e)
            }

# Initialize global service
global_markets_service = GlobalMarketsService()



