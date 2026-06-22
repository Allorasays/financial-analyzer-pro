"""
Financial Modeling Prep (FMP) API Service
Provides comprehensive financial data including income statements, balance sheets, cash flow, ratios, and key metrics
"""

import requests
import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class FMPService:
    """Service for fetching data from Financial Modeling Prep API"""
    
    def __init__(self):
        # Get API key from environment variable or use default
        self.api_key = os.getenv('FMP_API_KEY', 'R9F8nfYK9yGdmiq7I5ETw7e6EhTuG8ve')
        self.base_url = "https://financialmodelingprep.com/stable"
        self.enabled = bool(self.api_key)
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make API request to FMP stable API (symbol via query param, not path)."""
        if not self.enabled:
            logger.warning(f"FMP service not enabled (no API key)")
            return None
        
        try:
            url = f"{self.base_url}/{endpoint}"
            if params is None:
                params = {}
            params['apikey'] = self.api_key
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors in response
            if isinstance(data, dict) and 'Error Message' in data:
                logger.error(f"FMP API error for {endpoint}: {data.get('Error Message')}")
                return None
            
            # Check for rate limiting or invalid API key
            if isinstance(data, dict) and ('Note' in data or 'message' in data):
                error_msg = data.get('Note') or data.get('message', '')
                if 'limit' in error_msg.lower() or 'subscription' in error_msg.lower():
                    logger.warning(f"FMP API limit/subscription issue for {endpoint}: {error_msg}")
                else:
                    logger.error(f"FMP API message for {endpoint}: {error_msg}")
                return None
            
            return data if data else None
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                logger.error(f"FMP API authentication failed - check API key for {endpoint}")
            elif e.response.status_code == 403:
                logger.error(f"FMP API access forbidden - check subscription for {endpoint}")
            elif e.response.status_code == 429:
                logger.warning(f"FMP API rate limit exceeded for {endpoint}")
            else:
                logger.error(f"FMP API HTTP error for {endpoint}: {e.response.status_code} - {e}")
            return None
        except Exception as e:
            logger.error(f"FMP API request failed for {endpoint}: {e}")
            return None
    
    def get_key_metrics(self, ticker: str) -> Optional[Dict]:
        """Get key financial metrics"""
        data = self._make_request("key-metrics", {'symbol': ticker, 'limit': 1})
        return data[0] if data and len(data) > 0 else None
    
    def get_financial_ratios(self, ticker: str) -> Optional[Dict]:
        """Get financial ratios"""
        data = self._make_request("ratios", {'symbol': ticker, 'limit': 1})
        return data[0] if data and len(data) > 0 else None
    
    def get_income_statement(self, ticker: str) -> Optional[Dict]:
        """Get income statement"""
        data = self._make_request("income-statement", {'symbol': ticker, 'limit': 1})
        return data[0] if data and len(data) > 0 else None
    
    def get_balance_sheet(self, ticker: str) -> Optional[Dict]:
        """Get balance sheet"""
        data = self._make_request("balance-sheet-statement", {'symbol': ticker, 'limit': 1})
        return data[0] if data and len(data) > 0 else None
    
    def get_cash_flow(self, ticker: str) -> Optional[Dict]:
        """Get cash flow statement"""
        data = self._make_request("cash-flow-statement", {'symbol': ticker, 'limit': 1})
        return data[0] if data and len(data) > 0 else None
    
    def get_company_profile(self, ticker: str) -> Optional[Dict]:
        """Get company profile"""
        data = self._make_request("profile", {'symbol': ticker})
        return data[0] if data and len(data) > 0 else None
    
    def get_quote(self, ticker: str) -> Optional[Dict]:
        """Get real-time quote"""
        data = self._make_request("quote", {'symbol': ticker})
        return data[0] if data and len(data) > 0 else None

    def _normalize_field_names(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Map FMP field names to backend/Android FinancialDataResponse keys."""
        aliases = {
            'net_margin': 'profit_margin',
            'cash_and_equivalents': 'total_cash',
            'year_low': '52_week_low',
            'year_high': '52_week_high',
            'avg_volume': 'average_volume',
            'price_change_percent': 'change_percent',
        }
        for src, dest in aliases.items():
            if financial_data.get(src) is not None and financial_data.get(dest) is None:
                financial_data[dest] = financial_data[src]
        return financial_data
    
    def get_comprehensive_financial_data(self, ticker: str) -> Dict[str, Any]:
        """Get comprehensive financial data from multiple FMP endpoints"""
        if not self.enabled:
            return {}
        
        try:
            # Fetch all data in parallel (or sequentially with error handling)
            metrics = self.get_key_metrics(ticker)
            ratios = self.get_financial_ratios(ticker)
            income = self.get_income_statement(ticker)
            balance = self.get_balance_sheet(ticker)
            cash_flow = self.get_cash_flow(ticker)
            profile = self.get_company_profile(ticker)
            quote = self.get_quote(ticker)
            
            # Combine all data into comprehensive structure
            financial_data = {}
            
            # From Income Statement
            if income:
                financial_data.update({
                    'revenue': income.get('revenue'),
                    'gross_profit': income.get('grossProfit'),
                    'operating_income': income.get('operatingIncome'),
                    'net_income': income.get('netIncome'),
                    'ebitda': income.get('ebitda'),
                    'operating_expenses': income.get('operatingExpenses'),
                    'depreciation_amortization': income.get('depreciationAndAmortization'),
                    'interest_expense': income.get('interestExpense'),
                    'income_tax_expense': income.get('incomeTaxExpense'),
                })
            
            # From Balance Sheet
            if balance:
                financial_data.update({
                    'total_assets': balance.get('totalAssets'),
                    'total_current_assets': balance.get('totalCurrentAssets'),
                    'cash_and_equivalents': balance.get('cashAndCashEquivalents'),
                    'total_liabilities': balance.get('totalLiabilities'),
                    'total_current_liabilities': balance.get('totalCurrentLiabilities'),
                    'total_debt': balance.get('totalDebt'),
                    'total_equity': balance.get('totalStockholdersEquity'),
                    'retained_earnings': balance.get('retainedEarnings'),
                })
            
            # From Cash Flow
            if cash_flow:
                financial_data.update({
                    'operating_cash_flow': cash_flow.get('operatingCashFlow'),
                    'free_cash_flow': cash_flow.get('freeCashFlow'),
                    'capital_expenditure': cash_flow.get('capitalExpenditure'),
                    'dividends_paid': cash_flow.get('dividendsPaid'),
                })
            
            # From Key Metrics
            if metrics:
                financial_data.update({
                    'pe_ratio': metrics.get('peRatio'),
                    'price_to_book': metrics.get('pbRatio'),
                    'price_to_sales': metrics.get('priceToSalesRatio'),
                    'enterprise_value': metrics.get('enterpriseValue'),
                    'ev_to_revenue': metrics.get('evToSales'),
                    'ev_to_ebitda': metrics.get('evToEbitda'),
                    'market_cap': metrics.get('marketCap'),
                    'shares_outstanding': metrics.get('sharesOutstanding'),
                })
            
            # From Ratios
            if ratios:
                financial_data.update({
                    'pe_ratio': ratios.get('priceToEarningsRatio') or financial_data.get('pe_ratio'),
                    'current_ratio': ratios.get('currentRatio'),
                    'quick_ratio': ratios.get('quickRatio'),
                    'debt_to_equity': ratios.get('debtEquityRatio'),
                    'debt_to_assets': ratios.get('debtRatio'),
                    'return_on_equity': ratios.get('returnOnEquity'),
                    'return_on_assets': ratios.get('returnOnAssets'),
                    'return_on_invested_capital': ratios.get('returnOnCapitalEmployed'),
                    'gross_margin': ratios.get('grossProfitMargin'),
                    'operating_margin': ratios.get('operatingProfitMargin'),
                    'net_margin': ratios.get('netProfitMargin'),
                    'asset_turnover': ratios.get('assetTurnover'),
                    'inventory_turnover': ratios.get('inventoryTurnover'),
                })
            
            # From Profile
            if profile:
                financial_data.update({
                    'company_name': profile.get('companyName'),
                    'industry': profile.get('industry'),
                    'sector': profile.get('sector'),
                    'website': profile.get('website'),
                    'description': profile.get('description'),
                    'ceo': profile.get('ceo'),
                    'employees': profile.get('fullTimeEmployees'),
                })
            
            # From Quote
            if quote:
                financial_data.update({
                    'current_price': quote.get('price'),
                    'price_change': quote.get('change'),
                    'price_change_percent': quote.get('changePercentage') or quote.get('changesPercentage'),
                    'day_low': quote.get('dayLow'),
                    'day_high': quote.get('dayHigh'),
                    'year_low': quote.get('yearLow'),
                    'year_high': quote.get('yearHigh'),
                    'volume': quote.get('volume'),
                    'avg_volume': quote.get('avgVolume') or quote.get('averageVolume'),
                    'market_cap': quote.get('marketCap') or financial_data.get('market_cap'),
                    'previous_close': quote.get('previousClose'),
                    'open': quote.get('open'),
                })

            if profile:
                financial_data.setdefault('average_volume', profile.get('averageVolume'))
                financial_data.setdefault('beta', profile.get('beta'))
            
            financial_data = self._normalize_field_names(financial_data)
            financial_data['data_source'] = 'FMP'
            financial_data['timestamp'] = datetime.now().isoformat()
            
            # Log what we got
            data_count = len([v for v in financial_data.values() if v is not None and v != ''])
            logger.info(f"FMP comprehensive data for {ticker}: {data_count} non-null fields")
            
            # If we have very little data, log a warning
            if data_count < 10:
                logger.warning(f"FMP returned limited data for {ticker}: only {data_count} fields")
            
            return financial_data
            
        except Exception as e:
            logger.error(f"Error fetching comprehensive FMP data for {ticker}: {e}", exc_info=True)
            return {}

# Global instance
fmp_service = FMPService()


