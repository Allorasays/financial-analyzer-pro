"""
FRED Economic Indicators Integration
Fetches key economic data from FRED API (Federal Reserve Economic Data)
"""
import requests
import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime, timedelta
import logging
from config import FRED_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FREDIndicators:
    """Service for fetching FRED economic indicators"""
    
    def __init__(self):
        self.api_key = FRED_CONFIG['api_key']
        self.base_url = FRED_CONFIG['base_url']
        self.timeout = FRED_CONFIG['timeout']
        self.cache = {}
        self.cache_timestamps = {}
    
    def _get_series(self, series_id: str, lookback_days: int = 365) -> Optional[pd.Series]:
        """
        Fetch FRED series data
        
        Args:
            series_id: FRED series ID (e.g., 'FEDFUNDS' for federal funds rate)
            lookback_days: Number of days to look back
        
        Returns:
            Series with economic indicator values
        """
        cache_key = f"fred_{series_id}"
        if cache_key in self.cache:
            cache_time = self.cache_timestamps.get(cache_key, datetime.min)
            if (datetime.now() - cache_time).total_seconds() < 86400:  # Cache for 24 hours
                return self.cache[cache_key]
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            url = f"{self.base_url}{FRED_CONFIG['endpoints']['series_observations']}"
            params = {
                'series_id': series_id,
                'api_key': self.api_key,
                'file_type': 'json',
                'observation_start': start_date.strftime('%Y-%m-%d'),
                'observation_end': end_date.strftime('%Y-%m-%d'),
                'sort_order': 'desc',
                'limit': 1000
            }
            
            response = requests.get(url, params=params, timeout=self.timeout)
            
            if response.status_code == 200:
                data = response.json()
                observations = data.get('observations', [])
                
                if not observations:
                    return None
                
                # Convert to pandas Series
                dates = [pd.to_datetime(obs['date']) for obs in observations if obs.get('value') != '.']
                values = [float(obs['value']) for obs in observations if obs.get('value') != '.']
                
                series = pd.Series(values, index=dates, name=series_id)
                series = series.sort_index()
                
                self.cache[cache_key] = series
                self.cache_timestamps[cache_key] = datetime.now()
                
                return series
            else:
                logger.warning(f"FRED API returned {response.status_code} for {series_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching FRED series {series_id}: {e}")
            return None
    
    def get_economic_indicators(self) -> Dict:
        """
        Get key economic indicators
        
        Returns:
            Dictionary with current economic indicator values
        """
        indicators = {}
        
        # Federal Funds Rate (Interest Rate)
        try:
            fed_funds = self._get_series('FEDFUNDS', lookback_days=90)
            if fed_funds is not None and len(fed_funds) > 0:
                indicators['fed_funds_rate'] = fed_funds.iloc[-1]
                indicators['fed_funds_rate_change'] = (fed_funds.iloc[-1] - fed_funds.iloc[-2]) if len(fed_funds) > 1 else 0
        except Exception as e:
            logger.error(f"Error fetching Fed Funds Rate: {e}")
            indicators['fed_funds_rate'] = np.nan
            indicators['fed_funds_rate_change'] = np.nan
        
        # Inflation Rate (CPI - Consumer Price Index)
        try:
            cpi = self._get_series('CPIAUCSL', lookback_days=365)
            if cpi is not None and len(cpi) > 1:
                # Calculate year-over-year inflation
                current_cpi = cpi.iloc[-1]
                year_ago_cpi = cpi.iloc[-13] if len(cpi) >= 13 else cpi.iloc[0]
                inflation_rate = ((current_cpi - year_ago_cpi) / year_ago_cpi) * 100
                indicators['inflation_rate'] = inflation_rate
        except Exception as e:
            logger.error(f"Error fetching inflation rate: {e}")
            indicators['inflation_rate'] = np.nan
        
        # Unemployment Rate
        try:
            unemployment = self._get_series('UNRATE', lookback_days=90)
            if unemployment is not None and len(unemployment) > 0:
                indicators['unemployment_rate'] = unemployment.iloc[-1]
                indicators['unemployment_change'] = (unemployment.iloc[-1] - unemployment.iloc[-2]) if len(unemployment) > 1 else 0
        except Exception as e:
            logger.error(f"Error fetching unemployment rate: {e}")
            indicators['unemployment_rate'] = np.nan
            indicators['unemployment_change'] = np.nan
        
        # VIX (Volatility Index) - Fetch from yfinance since not in FRED
        try:
            import yfinance as yf
            vix = yf.Ticker("^VIX")
            vix_hist = vix.history(period="30d")
            if not vix_hist.empty:
                indicators['vix'] = vix_hist['Close'].iloc[-1]
                indicators['vix_change'] = (vix_hist['Close'].iloc[-1] - vix_hist['Close'].iloc[-2]) if len(vix_hist) > 1 else 0
            else:
                indicators['vix'] = np.nan
                indicators['vix_change'] = np.nan
        except Exception as e:
            logger.error(f"Error fetching VIX: {e}")
            indicators['vix'] = np.nan
            indicators['vix_change'] = np.nan
        
        # GDP Growth (Quarterly, most recent)
        try:
            gdp = self._get_series('GDP', lookback_days=730)  # 2 years for quarterly data
            if gdp is not None and len(gdp) > 1:
                # Calculate quarter-over-quarter growth
                current_gdp = gdp.iloc[-1]
                previous_gdp = gdp.iloc[-2] if len(gdp) > 1 else gdp.iloc[0]
                gdp_growth = ((current_gdp - previous_gdp) / previous_gdp) * 100
                indicators['gdp_growth'] = gdp_growth
                indicators['gdp'] = current_gdp
        except Exception as e:
            logger.error(f"Error fetching GDP: {e}")
            indicators['gdp_growth'] = np.nan
            indicators['gdp'] = np.nan
        
        return indicators


def get_fred_indicators() -> Dict:
    """Get current FRED economic indicators"""
    service = FREDIndicators()
    return service.get_economic_indicators()

# Global instance
fred_service = FREDIndicators()

