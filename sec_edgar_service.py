"""
SEC EDGAR Data Service
Fetches fundamental data from SEC EDGAR database (free, official)
"""
import requests
import pandas as pd
import json
from typing import Dict, Optional, Any
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SEC EDGAR API configuration
SEC_EDGAR_BASE_URL = "https://data.sec.gov"
SEC_EDGAR_COMPANY_FACTS = "/api/xbrl/companyfacts/CIK{}.json"
SEC_EDGAR_SUBMISSIONS = "/cgi-bin/browse-edgar?action=getcompany&CIK={}&type=10-K&dateb=&owner=exclude&count=10"

# User agent required by SEC (must identify your application)
USER_AGENT = "Moneta Financial Analyzer contact@moneta.financial"

class SECEdgarService:
    """Service for fetching SEC EDGAR fundamental data"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': USER_AGENT,
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'data.sec.gov'
        })
        self.cache = {}
        self.cache_timestamps = {}
    
    def _get_cik_from_ticker(self, ticker: str) -> Optional[str]:
        """Convert ticker symbol to CIK (Central Index Key)"""
        try:
            # SEC provides a ticker to CIK mapping
            ticker_url = f"https://www.sec.gov/files/company_tickers.json"
            response = requests.get(ticker_url, headers={'User-Agent': USER_AGENT}, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                for entry in data.values():
                    if entry.get('ticker', '').upper() == ticker.upper():
                        cik = str(entry.get('cik_str', ''))
                        # Pad CIK with zeros to 10 digits
                        return cik.zfill(10)
            return None
        except Exception as e:
            logger.error(f"Error fetching CIK for {ticker}: {e}")
            return None
    
    def get_company_facts(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get company facts from SEC EDGAR"""
        cache_key = f"company_facts_{ticker}"
        if cache_key in self.cache:
            cache_time = self.cache_timestamps.get(cache_key, datetime.min)
            if (datetime.now() - cache_time).days < 1:  # Cache for 1 day
                return self.cache[cache_key]
        
        try:
            cik = self._get_cik_from_ticker(ticker)
            if not cik:
                logger.warning(f"Could not find CIK for ticker {ticker}")
                return None
            
            url = f"{SEC_EDGAR_BASE_URL}{SEC_EDGAR_COMPANY_FACTS.format(cik)}"
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                self.cache[cache_key] = data
                self.cache_timestamps[cache_key] = datetime.now()
                return data
            else:
                logger.warning(f"SEC EDGAR API returned {response.status_code} for {ticker}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching SEC EDGAR data for {ticker}: {e}")
            return None
    
    def extract_financial_metrics(self, ticker: str) -> Dict[str, Any]:
        """Extract key financial metrics from SEC EDGAR data"""
        facts = self.get_company_facts(ticker)
        if not facts:
            return {}
        
        metrics = {}
        
        try:
            # Extract from US-GAAP facts
            us_gaap = facts.get('facts', {}).get('us-gaap', {})
            
            # Revenue
            if 'Revenues' in us_gaap:
                revenue_data = us_gaap['Revenues']['units']['USD']
                if revenue_data:
                    # Get most recent annual revenue
                    annual_revenues = [item for item in revenue_data if item.get('form') == '10-K']
                    if annual_revenues:
                        latest = sorted(annual_revenues, key=lambda x: x.get('end', ''), reverse=True)[0]
                        metrics['revenue'] = latest.get('val')
                        metrics['revenue_date'] = latest.get('end')
            
            # Net Income
            if 'NetIncomeLoss' in us_gaap:
                ni_data = us_gaap['NetIncomeLoss']['units']['USD']
                if ni_data:
                    annual_ni = [item for item in ni_data if item.get('form') == '10-K']
                    if annual_ni:
                        latest = sorted(annual_ni, key=lambda x: x.get('end', ''), reverse=True)[0]
                        metrics['net_income'] = latest.get('val')
                        metrics['net_income_date'] = latest.get('end')
            
            # Total Assets
            if 'Assets' in us_gaap:
                assets_data = us_gaap['Assets']['units']['USD']
                if assets_data:
                    annual_assets = [item for item in assets_data if item.get('form') == '10-K']
                    if annual_assets:
                        latest = sorted(annual_assets, key=lambda x: x.get('end', ''), reverse=True)[0]
                        metrics['total_assets'] = latest.get('val')
            
            # Total Liabilities
            if 'Liabilities' in us_gaap:
                liab_data = us_gaap['Liabilities']['units']['USD']
                if liab_data:
                    annual_liab = [item for item in liab_data if item.get('form') == '10-K']
                    if annual_liab:
                        latest = sorted(annual_liab, key=lambda x: x.get('end', ''), reverse=True)[0]
                        metrics['total_liabilities'] = latest.get('val')
            
            # Calculate ratios if we have the data
            if metrics.get('revenue') and metrics.get('net_income'):
                metrics['profit_margin'] = (metrics['net_income'] / metrics['revenue']) * 100
            
            if metrics.get('total_assets') and metrics.get('total_liabilities'):
                metrics['debt_to_assets'] = (metrics['total_liabilities'] / metrics['total_assets']) * 100
            
            # Get revenue growth (compare last 2 years)
            if 'Revenues' in us_gaap:
                revenue_data = us_gaap['Revenues']['units']['USD']
                annual_revenues = sorted(
                    [item for item in revenue_data if item.get('form') == '10-K'],
                    key=lambda x: x.get('end', ''), 
                    reverse=True
                )[:2]
                
                if len(annual_revenues) >= 2:
                    current_rev = annual_revenues[0].get('val')
                    previous_rev = annual_revenues[1].get('val')
                    if previous_rev and previous_rev > 0:
                        metrics['revenue_growth'] = ((current_rev - previous_rev) / previous_rev) * 100
        
        except Exception as e:
            logger.error(f"Error extracting financial metrics for {ticker}: {e}")
        
        return metrics

# Global instance
sec_edgar_service = SECEdgarService()

def get_financial_metrics(ticker: str) -> Dict[str, Any]:
    """Get financial metrics from SEC EDGAR"""
    return sec_edgar_service.extract_financial_metrics(ticker)

