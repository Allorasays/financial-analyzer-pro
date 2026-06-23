"""
Financial Data Aggregator
Combines data from multiple APIs (FMP, Alpha Vantage, yfinance, SEC EDGAR) to maximize data coverage
"""

import requests
import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import yfinance as yf

logger = logging.getLogger(__name__)


class FinancialDataAggregator:
    """Aggregates financial data from multiple sources to maximize coverage"""
    
    def __init__(self):
        self.fmp_api_key = os.getenv('FMP_API_KEY', 'R9F8nfYK9yGdmiq7I5ETw7e6EhTuG8ve')
        self.alpha_vantage_key = os.getenv('ALPHAVANTAGE_API_KEY', 'C04TV0QS7GVJF0RU')
        self.polygon_key = os.getenv('POLYGON_API_KEY', 'gqvp07BQCfnH7Xq5p7GbbfAXLpvv7HTm')
    
    def _merge_data(self, base_data: Dict, new_data: Dict) -> Dict:
        """Merge new_data into base_data, only adding non-None values that don't exist in base_data"""
        merged = base_data.copy()
        for key, value in new_data.items():
            # Skip metadata fields
            if key in ['data_source', 'timestamp', 'ticker', 'data_sources', 'data_coverage']:
                continue
            # Only add if value is not None and (field doesn't exist in base or base value is None)
            if value is not None:
                if key not in merged or merged[key] is None:
                    merged[key] = value
        return merged
    
    def _get_fmp_data(self, ticker: str) -> Dict[str, Any]:
        """Get data from Financial Modeling Prep"""
        if not self.fmp_api_key:
            return {}
        
        try:
            from fmp_service import fmp_service
            return fmp_service.get_comprehensive_financial_data(ticker)
        except Exception as e:
            logger.debug(f"FMP data fetch failed for {ticker}: {e}")
            return {}
    
    def _get_alpha_vantage_data(self, ticker: str) -> Dict[str, Any]:
        """Get data from Alpha Vantage"""
        if not self.alpha_vantage_key:
            return {}
        
        try:
            # Overview
            overview_url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={self.alpha_vantage_key}"
            overview_response = requests.get(overview_url, timeout=10)
            
            if overview_response.status_code == 200:
                overview_data = overview_response.json()
                
                # Check for API errors
                if 'Error Message' in overview_data or 'Note' in overview_data:
                    return {}
                
                # Income Statement
                income_url = f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={ticker}&apikey={self.alpha_vantage_key}"
                income_response = requests.get(income_url, timeout=10)
                income_data = income_response.json() if income_response.status_code == 200 else {}
                
                # Balance Sheet
                balance_url = f"https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={ticker}&apikey={self.alpha_vantage_key}"
                balance_response = requests.get(balance_url, timeout=10)
                balance_data = balance_response.json() if balance_response.status_code == 200 else {}
                
                # Cash Flow
                cashflow_url = f"https://www.alphavantage.co/query?function=CASH_FLOW&symbol={ticker}&apikey={self.alpha_vantage_key}"
                cashflow_response = requests.get(cashflow_url, timeout=10)
                cashflow_data = cashflow_response.json() if cashflow_response.status_code == 200 else {}
                
                # Extract and combine data
                financial_data = {}
                
                # From Overview
                if overview_data and isinstance(overview_data, dict):
                    financial_data.update({
                        'company_name': overview_data.get('Name'),
                        'sector': overview_data.get('Sector'),
                        'industry': overview_data.get('Industry'),
                        'market_cap': self._safe_float(overview_data.get('MarketCapitalization')),
                        'ebitda': self._safe_float(overview_data.get('EBITDA')),
                        'pe_ratio': self._safe_float(overview_data.get('PERatio')),
                        'peg_ratio': self._safe_float(overview_data.get('PEGRatio')),
                        'book_value': self._safe_float(overview_data.get('BookValue')),
                        'dividend_per_share': self._safe_float(overview_data.get('DividendPerShare')),
                        'dividend_yield': self._safe_float(overview_data.get('DividendYield')),
                        'eps': self._safe_float(overview_data.get('EPS')),
                        'revenue_per_share': self._safe_float(overview_data.get('RevenuePerShareTTM')),
                        'profit_margin': self._safe_float(overview_data.get('ProfitMargin')),
                        'operating_margin': self._safe_float(overview_data.get('OperatingMarginTTM')),
                        'return_on_assets': self._safe_float(overview_data.get('ReturnOnAssetsTTM')),
                        'return_on_equity': self._safe_float(overview_data.get('ReturnOnEquityTTM')),
                        'revenue': self._safe_float(overview_data.get('RevenueTTM')),
                        'gross_profit': self._safe_float(overview_data.get('GrossProfitTTM')),
                        'diluted_eps': self._safe_float(overview_data.get('DilutedEPSTTM')),
                        'quarterly_earnings_growth': self._safe_float(overview_data.get('QuarterlyEarningsGrowthYOY')),
                        'quarterly_revenue_growth': self._safe_float(overview_data.get('QuarterlyRevenueGrowthYOY')),
                        'analyst_target_price': self._safe_float(overview_data.get('AnalystTargetPrice')),
                        'trailing_pe': self._safe_float(overview_data.get('TrailingPE')),
                        'forward_pe': self._safe_float(overview_data.get('ForwardPE')),
                        'price_to_sales': self._safe_float(overview_data.get('PriceToSalesRatioTTM')),
                        'price_to_book': self._safe_float(overview_data.get('PriceToBookRatio')),
                        'ev_to_revenue': self._safe_float(overview_data.get('EVToRevenue')),
                        'ev_to_ebitda': self._safe_float(overview_data.get('EVToEBITDA')),
                        'beta': self._safe_float(overview_data.get('Beta')),
                        '52_week_high': self._safe_float(overview_data.get('52WeekHigh')),
                        '52_week_low': self._safe_float(overview_data.get('52WeekLow')),
                        'shares_outstanding': self._safe_float(overview_data.get('SharesOutstanding')),
                    })
                
                # From Income Statement (most recent annual)
                if income_data and isinstance(income_data, dict) and 'annualReports' in income_data:
                    annual_reports = income_data['annualReports']
                    if annual_reports and len(annual_reports) > 0:
                        latest = annual_reports[0]
                        financial_data.update({
                            'revenue': financial_data.get('revenue') or self._safe_float(latest.get('totalRevenue')),
                            'gross_profit': financial_data.get('gross_profit') or self._safe_float(latest.get('grossProfit')),
                            'operating_income': self._safe_float(latest.get('operatingIncome')),
                            'net_income': self._safe_float(latest.get('netIncome')),
                            'total_revenue': self._safe_float(latest.get('totalRevenue')),
                            'cost_of_revenue': self._safe_float(latest.get('costOfRevenue')),
                            'total_operating_expenses': self._safe_float(latest.get('totalOperatingExpenses')),
                            'income_before_tax': self._safe_float(latest.get('incomeBeforeTax')),
                            'income_tax_expense': self._safe_float(latest.get('incomeTaxExpense')),
                        })
                
                # From Balance Sheet (most recent annual)
                if balance_data and isinstance(balance_data, dict) and 'annualReports' in balance_data:
                    annual_reports = balance_data['annualReports']
                    if annual_reports and len(annual_reports) > 0:
                        latest = annual_reports[0]
                        financial_data.update({
                            'total_assets': self._safe_float(latest.get('totalAssets')),
                            'total_current_assets': self._safe_float(latest.get('totalCurrentAssets')),
                            'cash_and_equivalents': self._safe_float(latest.get('cashAndCashEquivalentsAtCarryingValue')),
                            'total_liabilities': self._safe_float(latest.get('totalLiabilities')),
                            'total_current_liabilities': self._safe_float(latest.get('totalCurrentLiabilities')),
                            'total_debt': self._safe_float(latest.get('totalDebt')),
                            'total_equity': self._safe_float(latest.get('totalShareholderEquity')),
                            'retained_earnings': self._safe_float(latest.get('retainedEarnings')),
                            'common_stock': self._safe_float(latest.get('commonStock')),
                        })
                
                # From Cash Flow (most recent annual)
                if cashflow_data and isinstance(cashflow_data, dict) and 'annualReports' in cashflow_data:
                    annual_reports = cashflow_data['annualReports']
                    if annual_reports and len(annual_reports) > 0:
                        latest = annual_reports[0]
                        financial_data.update({
                            'operating_cash_flow': self._safe_float(latest.get('operatingCashflow')),
                            'capital_expenditure': self._safe_float(latest.get('capitalExpenditures')),
                            'free_cash_flow': self._safe_float(latest.get('operatingCashflow')) - abs(self._safe_float(latest.get('capitalExpenditures', 0)) or 0),
                            'cashflow_from_investing': self._safe_float(latest.get('cashflowFromInvestment')),
                            'cashflow_from_financing': self._safe_float(latest.get('cashflowFromFinancing')),
                            'dividends_paid': self._safe_float(latest.get('dividendsPaid')),
                            'net_change_in_cash': self._safe_float(latest.get('netChangeInCash')),
                        })
                
                if financial_data:
                    financial_data['data_source'] = 'Alpha Vantage'
                    return financial_data
        except Exception as e:
            logger.debug(f"Alpha Vantage data fetch failed for {ticker}: {e}")
        
        return {}
    
    def _get_yfinance_data(self, ticker: str) -> Dict[str, Any]:
        """Get data from yfinance"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            def safe_get(key, default=None, allow_zero=True):
                """Get value from info dict, preserving 0 and False as valid values"""
                value = info.get(key, default)
                # Only return None for truly missing/invalid values
                if value is None or value == '':
                    return None
                # Handle NaN and infinity values
                try:
                    if isinstance(value, float):
                        import math
                        if math.isnan(value) or math.isinf(value):
                            return None
                except:
                    pass
                # For most financial metrics, 0 is a valid value
                if not allow_zero and value == 0:
                    return None
                return value
            
            # Comprehensive financial data from yfinance (matches original proxy.py mapping)
            financial_data = {
                # Company Information
                'company_name': info.get('longName') or info.get('shortName'),
                'industry': info.get('industry'),
                'sector': info.get('sector'),
                'website': info.get('website'),
                'description': info.get('longBusinessSummary'),
                
                # Market Data
                'current_price': safe_get('currentPrice') or safe_get('regularMarketPrice'),
                'previous_close': safe_get('previousClose'),
                'market_cap': safe_get('marketCap'),
                'enterprise_value': safe_get('enterpriseValue'),
                'shares_outstanding': safe_get('sharesOutstanding'),
                'float_shares': safe_get('floatShares'),
                'shares_short': safe_get('sharesShort'),
                'short_ratio': safe_get('shortRatio'),
                '52_week_high': safe_get('fiftyTwoWeekHigh'),
                '52_week_low': safe_get('fiftyTwoWeekLow'),
                
                # Valuation Ratios
                'pe_ratio': safe_get('trailingPE'),
                'forward_pe': safe_get('forwardPE'),
                'peg_ratio': safe_get('pegRatio'),
                'price_to_book': safe_get('priceToBook'),
                'price_to_sales': safe_get('priceToSalesTrailing12Months'),
                'enterprise_value_to_revenue': safe_get('enterpriseToRevenue'),
                'enterprise_value_to_ebitda': safe_get('enterpriseToEbitda'),
                'ev_to_revenue': safe_get('enterpriseToRevenue'),
                'ev_to_ebitda': safe_get('enterpriseToEbitda'),
                
                # Profitability Metrics
                'revenue': safe_get('totalRevenue'),
                'revenue_per_share': safe_get('revenuePerShare'),
                'revenue_growth': safe_get('revenueGrowth'),
                'net_income': safe_get('netIncomeToCommon') or safe_get('netIncome'),
                'net_income_common': safe_get('netIncomeToCommon'),
                'earnings_per_share': safe_get('trailingEps') or safe_get('epsTrailingTwelveMonths'),
                'forward_eps': safe_get('forwardEps'),
                'earnings_growth': safe_get('earningsGrowth'),
                'earnings_quarterly_growth': safe_get('earningsQuarterlyGrowth'),
                
                # Margins
                'gross_margin': safe_get('grossMargins'),
                'operating_margin': safe_get('operatingMargins'),
                'profit_margin': safe_get('profitMargins'),
                'ebitda_margin': safe_get('ebitdaMargins'),
                
                # Cash Flow
                'ebitda': safe_get('ebitda'),
                'free_cash_flow': safe_get('freeCashflow'),
                'operating_cash_flow': safe_get('operatingCashflow'),
                'cash_per_share': safe_get('totalCashPerShare'),
                
                # Returns
                'return_on_equity': safe_get('returnOnEquity'),
                'return_on_assets': safe_get('returnOnAssets'),
                'return_on_invested_capital': safe_get('returnOnInvestedCapital'),
                
                # Debt & Liquidity
                'debt_to_equity': safe_get('debtToEquity'),
                'debt_to_assets': safe_get('debtToAssets'),
                'current_ratio': safe_get('currentRatio'),
                'quick_ratio': safe_get('quickRatio'),
                'cash_ratio': safe_get('cashRatio'),
                'total_debt': safe_get('totalDebt'),
                'total_cash': safe_get('totalCash'),
                'total_cash_per_share': safe_get('totalCashPerShare'),
                
                # Dividends
                'dividend_yield': safe_get('dividendYield'),
                'dividend_rate': safe_get('dividendRate'),
                'dividend_per_share': safe_get('trailingAnnualDividendRate'),
                'payout_ratio': safe_get('payoutRatio'),
                'ex_dividend_date': info.get('exDividendDate'),
                'dividend_date': info.get('dividendDate'),
                
                # Growth Metrics
                'revenue_growth': safe_get('revenueGrowth'),
                'earnings_growth': safe_get('earningsGrowth'),
                'earnings_quarterly_growth': safe_get('earningsQuarterlyGrowth'),
                
                # Trading Metrics
                'beta': safe_get('beta'),
                'volume': safe_get('volume', allow_zero=True),
                'average_volume': safe_get('averageVolume', allow_zero=True),
                'average_volume_10days': safe_get('averageVolume10days', allow_zero=True),
                'bid': safe_get('bid'),
                'ask': safe_get('ask'),
                'bid_size': safe_get('bidSize'),
                'ask_size': safe_get('askSize'),
                'day_low': safe_get('dayLow'),
                'day_high': safe_get('dayHigh'),
                'open': safe_get('open'),
                
                # Analyst Data
                'target_high_price': safe_get('targetHighPrice'),
                'target_low_price': safe_get('targetLowPrice'),
                'target_mean_price': safe_get('targetMeanPrice'),
                'target_median_price': safe_get('targetMedianPrice'),
                'recommendation_mean': info.get('recommendationMean'),
                'recommendation_key': info.get('recommendationKey'),
                'number_of_analyst_opinions': safe_get('numberOfAnalystOpinions'),
                
                # Additional Metrics
                'book_value': safe_get('bookValue'),
                'price_to_book': safe_get('priceToBook'),
                'price_to_sales_trailing_12months': safe_get('priceToSalesTrailing12Months'),
                'enterprise_value': safe_get('enterpriseValue'),
                'enterprise_value_to_revenue': safe_get('enterpriseToRevenue'),
                'enterprise_value_to_ebitda': safe_get('enterpriseToEbitda'),
                'held_percent_insiders': safe_get('heldPercentInsiders'),
                'held_percent_institutions': safe_get('heldPercentInstitutions'),
            }
            
            # Keep all fields (including None) - merge logic will handle filling gaps
            financial_data['data_source'] = 'yfinance'
            return financial_data
        except Exception as e:
            logger.debug(f"yfinance data fetch failed for {ticker}: {e}")
            return {}
    
    def _safe_float(self, value) -> Optional[float]:
        """Safely convert value to float"""
        if value is None or value == '' or value == 'None':
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def get_comprehensive_financial_data(self, ticker: str) -> Dict[str, Any]:
        """
        Get comprehensive financial data by aggregating from multiple sources
        Priority: FMP > Alpha Vantage > yfinance
        """
        ticker = ticker.upper()
        financial_data = {
            'ticker': ticker,
            'timestamp': datetime.now().isoformat(),
            'data_sources': []
        }
        
        # Priority 1: Start with yfinance to get comprehensive field structure (even if some are None)
        # This ensures we have all possible fields defined
        yf_data = self._get_yfinance_data(ticker)
        if yf_data:
            financial_data = self._merge_data(financial_data, yf_data)
            financial_data['data_sources'].append('yfinance')
            yf_non_null = len([v for k, v in yf_data.items() if k not in ['data_source', 'timestamp', 'ticker'] and v is not None])
            logger.info(f"[Aggregator] yfinance data added for {ticker}: {yf_non_null} non-null fields")
        
        # Priority 2: Add FMP data to fill gaps (most comprehensive financial statements)
        fmp_data = self._get_fmp_data(ticker)
        if fmp_data:
            financial_data = self._merge_data(financial_data, fmp_data)
            if 'FMP' not in financial_data['data_sources']:
                financial_data['data_sources'].append('FMP')
            fmp_non_null = len([v for k, v in fmp_data.items() if k not in ['data_source', 'timestamp', 'ticker'] and v is not None])
            logger.info(f"[Aggregator] FMP data added for {ticker}: {fmp_non_null} non-null fields")
        
        # Priority 3: Add Alpha Vantage data to fill remaining gaps
        av_data = self._get_alpha_vantage_data(ticker)
        if av_data:
            financial_data = self._merge_data(financial_data, av_data)
            if 'Alpha Vantage' not in financial_data['data_sources']:
                financial_data['data_sources'].append('Alpha Vantage')
            av_non_null = len([v for k, v in av_data.items() if k not in ['data_source', 'timestamp', 'ticker'] and v is not None])
            logger.info(f"[Aggregator] Alpha Vantage data added for {ticker}: {av_non_null} non-null fields")
        
        # Calculate data coverage (non-null fields)
        non_null_count = len([v for k, v in financial_data.items() if k not in ['data_source', 'data_sources', 'timestamp', 'ticker', 'data_coverage'] and v is not None])
        financial_data['data_coverage'] = non_null_count
        financial_data['data_source'] = '+'.join(financial_data['data_sources']) if financial_data['data_sources'] else 'none'
        
        logger.info(f"[Aggregator] Total data coverage for {ticker}: {non_null_count} non-null fields from {len(financial_data['data_sources'])} sources")
        
        return financial_data


# Global instance
financial_data_aggregator = FinancialDataAggregator()

