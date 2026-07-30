"""
Comprehensive Financial Data Aggregator (canonical)
Uses ALL available APIs: FMP, Alpha Vantage, Polygon.io, SEC EDGAR, yfinance.
This is the single source of truth for /api/financials — env keys only, no defaults.
"""

import requests
import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import yfinance as yf
import time

logger = logging.getLogger(__name__)


class ComprehensiveFinancialAggregator:
    """Aggregates financial data from ALL available sources to maximize coverage"""
    
    def __init__(self):
        # Personal use: supply your own keys via environment variables (.env).
        self.fmp_api_key = os.getenv('FMP_API_KEY', '').strip()
        self.alpha_vantage_key = os.getenv('ALPHAVANTAGE_API_KEY', '').strip()
        self.polygon_key = os.getenv('POLYGON_API_KEY', '').strip()
        self.tiingo_key = os.getenv('TIINGO_API_KEY', '').strip()
        
        # Rate limiting
        self.last_alpha_vantage_call = 0
        self.alpha_vantage_min_interval = 12  # 5 requests/minute = 12 seconds between calls
    
    def _merge_data(self, base_data: Dict, new_data: Dict) -> Dict:
        """Merge new_data into base_data, filling gaps with non-None values"""
        merged = base_data.copy()
        for key, value in new_data.items():
            # Skip metadata fields
            if key in ['data_source', 'timestamp', 'ticker', 'data_sources', 'data_coverage']:
                continue
            # Only add if value is not None and (field doesn't exist or base value is None)
            if value is not None:
                if key not in merged or merged[key] is None:
                    merged[key] = value
        return merged
    
    def _safe_float(self, value) -> Optional[float]:
        """Safely convert value to float"""
        if value is None or value == '' or value == 'None':
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
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
        """Get data from Alpha Vantage (with rate limiting)"""
        if not self.alpha_vantage_key:
            return {}
        
        # Rate limiting: 5 requests/minute
        current_time = time.time()
        time_since_last = current_time - self.last_alpha_vantage_call
        if time_since_last < self.alpha_vantage_min_interval:
            wait_time = self.alpha_vantage_min_interval - time_since_last
            logger.debug(f"Alpha Vantage rate limit: waiting {wait_time:.1f}s")
            time.sleep(wait_time)
        
        try:
            self.last_alpha_vantage_call = time.time()
            
            # Overview
            overview_url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={self.alpha_vantage_key}"
            overview_response = requests.get(overview_url, timeout=10)
            
            if overview_response.status_code == 200:
                overview_data = overview_response.json()
                
                # Check for API errors
                if 'Error Message' in overview_data or 'Note' in overview_data:
                    return {}
                
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
                
                if financial_data:
                    financial_data['data_source'] = 'Alpha Vantage'
                    return financial_data
        except Exception as e:
            logger.debug(f"Alpha Vantage data fetch failed for {ticker}: {e}")
        
        return {}
    
    def _get_polygon_data(self, ticker: str) -> Dict[str, Any]:
        """Get data from Polygon.io"""
        if not self.polygon_key:
            return {}
        
        try:
            # Get ticker details
            url = f"https://api.polygon.io/v3/reference/tickers/{ticker}"
            params = {'apiKey': self.polygon_key}
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'results' in data and data['results']:
                    ticker_info = data['results']
                    financial_data = {
                        'company_name': ticker_info.get('name'),
                        'market': ticker_info.get('market'),
                        'locale': ticker_info.get('locale'),
                        'primary_exchange': ticker_info.get('primary_exchange'),
                        'type': ticker_info.get('type'),
                        'active': ticker_info.get('active'),
                        'currency': ticker_info.get('currency_name'),
                        'cik': ticker_info.get('cik'),
                        'composite_figi': ticker_info.get('composite_figi'),
                        'share_class_figi': ticker_info.get('share_class_figi'),
                    }
                    
                    # Get financials if available
                    try:
                        financials_url = f"https://api.polygon.io/vX/reference/financials"
                        financials_params = {'ticker': ticker, 'apiKey': self.polygon_key, 'limit': 1}
                        financials_response = requests.get(financials_url, params=financials_params, timeout=10)
                        
                        if financials_response.status_code == 200:
                            financials_data = financials_response.json()
                            if 'results' in financials_data and financials_data['results']:
                                latest_financials = financials_data['results'][0]
                                if 'financials' in latest_financials:
                                    fin = latest_financials['financials']
                                    financial_data.update({
                                        'revenue': self._safe_float(fin.get('revenues')),
                                        'net_income': self._safe_float(fin.get('net_income_loss')),
                                        'total_assets': self._safe_float(fin.get('assets')),
                                        'total_equity': self._safe_float(fin.get('equity')),
                                    })
                    except:
                        pass
                    
                    if financial_data:
                        financial_data['data_source'] = 'Polygon.io'
                        return financial_data
        except Exception as e:
            logger.debug(f"Polygon.io data fetch failed for {ticker}: {e}")
        
        return {}
    
    def _get_sec_edgar_data(self, ticker: str) -> Dict[str, Any]:
        """Get data from SEC EDGAR (free, official)"""
        try:
            # Try new structured SEC EDGAR client first
            try:
                from sec_edgar import SecEdgarClient, CIKResolver, CompanyFactsService
                client = SecEdgarClient()
                cik_resolver = CIKResolver(client)
                cik = cik_resolver.cik_from_ticker(ticker)
                facts_service = CompanyFactsService(client)
                financial_data = facts_service.extract_financial_metrics(ticker, cik)
                if financial_data:
                    financial_data['data_source'] = 'SEC EDGAR'
                    return financial_data
            except ImportError:
                # Fallback to old service
                pass
            except Exception as e:
                logger.debug(f"New SEC EDGAR client failed: {e}, trying fallback")
            
            # Fallback to old SEC EDGAR service
            from sec_edgar_service import SECEdgarService
            sec_service = SECEdgarService()
            company_facts = sec_service.get_company_facts(ticker)
            
            if company_facts:
                financial_data = {}
                
                # Extract key financial metrics from SEC data
                if 'facts' in company_facts and 'us-gaap' in company_facts['facts']:
                    gaap = company_facts['facts']['us-gaap']
                    
                    # Revenue - try different keys
                    for rev_key in ['Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax']:
                        if rev_key in gaap:
                            revenues = gaap[rev_key]
                            if 'units' in revenues and 'USD' in revenues['units']:
                                usd_units = revenues['units']['USD']
                                if usd_units:
                                    # Get most recent annual (10-K) or quarterly
                                    annual = [x for x in usd_units if x.get('form') == '10-K']
                                    if annual:
                                        latest = sorted(annual, key=lambda x: x.get('end', ''), reverse=True)[0]
                                        val = self._safe_float(latest.get('val'))
                                        if val:
                                            financial_data['revenue'] = val
                                            break
                    
                    # Net Income
                    for ni_key in ['NetIncomeLoss', 'ProfitLoss']:
                        if ni_key in gaap:
                            net_income = gaap[ni_key]
                            if 'units' in net_income and 'USD' in net_income['units']:
                                usd_units = net_income['units']['USD']
                                if usd_units:
                                    annual = [x for x in usd_units if x.get('form') == '10-K']
                                    if annual:
                                        latest = sorted(annual, key=lambda x: x.get('end', ''), reverse=True)[0]
                                        val = self._safe_float(latest.get('val'))
                                        if val:
                                            financial_data['net_income'] = val
                                            break
                    
                    # Total Assets
                    if 'Assets' in gaap:
                        assets = gaap['Assets']
                        if 'units' in assets and 'USD' in assets['units']:
                            usd_units = assets['units']['USD']
                            if usd_units:
                                annual = [x for x in usd_units if x.get('form') == '10-K']
                                if annual:
                                    latest = sorted(annual, key=lambda x: x.get('end', ''), reverse=True)[0]
                                    val = self._safe_float(latest.get('val'))
                                    if val:
                                        financial_data['total_assets'] = val
                    
                    # Total Equity
                    for eq_key in ['StockholdersEquity', 'Equity']:
                        if eq_key in gaap:
                            equity = gaap[eq_key]
                            if 'units' in equity and 'USD' in equity['units']:
                                usd_units = equity['units']['USD']
                                if usd_units:
                                    annual = [x for x in usd_units if x.get('form') == '10-K']
                                    if annual:
                                        latest = sorted(annual, key=lambda x: x.get('end', ''), reverse=True)[0]
                                        val = self._safe_float(latest.get('val'))
                                        if val:
                                            financial_data['total_equity'] = val
                                            break
                
                if financial_data:
                    financial_data['data_source'] = 'SEC EDGAR'
                    return financial_data
        except Exception as e:
            logger.debug(f"SEC EDGAR data fetch failed for {ticker}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        return {}
    
    def _get_yfinance_data(self, ticker: str) -> Dict[str, Any]:
        """Get comprehensive data from yfinance"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            def safe_get(key, default=None, allow_zero=True):
                """Get value from info dict, preserving 0 and False as valid values"""
                value = info.get(key, default)
                if value is None or value == '':
                    return None
                try:
                    if isinstance(value, float):
                        import math
                        if math.isnan(value) or math.isinf(value):
                            return None
                except:
                    pass
                if not allow_zero and value == 0:
                    return None
                return value
            
            # Comprehensive financial data from yfinance
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
            
            financial_data['data_source'] = 'yfinance'
            return financial_data
        except Exception as e:
            err = str(e).lower()
            if 'rate limit' in err or 'too many requests' in err:
                logger.warning(f"yfinance rate limited for {ticker} — using other sources")
            else:
                logger.debug(f"yfinance data fetch failed for {ticker}: {e}")
            return {}
    
    def _normalize_android_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Align field names with Android FinancialDataResponse."""
        aliases = {
            'eps': 'earnings_per_share',
            'diluted_eps': 'earnings_per_share',
            'net_margin': 'profit_margin',
            'cash_and_equivalents': 'total_cash',
            'avg_volume': 'average_volume',
            'year_high': '52_week_high',
            'year_low': '52_week_low',
            'price_change_percent': 'change_percent',
            'trailing_pe': 'pe_ratio',
            'dividend_per_share': 'dividend_rate',
        }
        for src, dest in aliases.items():
            if data.get(dest) is None and data.get(src) is not None:
                data[dest] = data[src]
        if data.get('earnings_per_share') is None and data.get('forward_eps') is not None:
            data['earnings_per_share'] = data['forward_eps']
        return data

    def get_comprehensive_financial_data(self, ticker: str) -> Dict[str, Any]:
        """
        Get comprehensive financial data by aggregating from ALL available sources.
        Priority: FMP -> yfinance -> Alpha Vantage -> Polygon -> SEC EDGAR
        (FMP first so yfinance rate limits do not block the response.)
        """
        ticker = ticker.upper()
        financial_data = {
            'ticker': ticker,
            'timestamp': datetime.now().isoformat(),
            'data_sources': []
        }
        
        # Step 1: FMP when user has configured a personal API key
        fmp_data = self._get_fmp_data(ticker)
        if fmp_data:
            financial_data = self._merge_data(financial_data, fmp_data)
            financial_data['data_sources'].append('FMP')
            fmp_non_null = len([v for k, v in fmp_data.items() if k not in ['data_source', 'timestamp', 'ticker'] and v is not None])
            logger.info(f"[Comprehensive Aggregator] FMP: {fmp_non_null} fields for {ticker}")
        
        # Step 2: yfinance only if FMP did not supply enough statement-level fields
        fmp_field_count = len([
            v for k, v in financial_data.items()
            if k in ('revenue', 'net_income', 'gross_margin', 'operating_cash_flow', 'total_debt')
            and v is not None
        ])
        if fmp_field_count < 3:
            yf_data = self._get_yfinance_data(ticker)
            if yf_data:
                financial_data = self._merge_data(financial_data, yf_data)
                if 'yfinance' not in financial_data['data_sources']:
                    financial_data['data_sources'].append('yfinance')
                yf_non_null = len([v for k, v in yf_data.items() if k not in ['data_source', 'timestamp', 'ticker'] and v is not None])
                logger.info(f"[Comprehensive Aggregator] yfinance: {yf_non_null} fields for {ticker}")
        else:
            logger.info(f"[Comprehensive Aggregator] Skipping yfinance for {ticker} — FMP has statement data")
        
        # Step 3: Add Alpha Vantage data (good financial statements)
        av_data = self._get_alpha_vantage_data(ticker)
        if av_data:
            financial_data = self._merge_data(financial_data, av_data)
            if 'Alpha Vantage' not in financial_data['data_sources']:
                financial_data['data_sources'].append('Alpha Vantage')
            av_non_null = len([v for k, v in av_data.items() if k not in ['data_source', 'timestamp', 'ticker'] and v is not None])
            logger.info(f"[Comprehensive Aggregator] Alpha Vantage: {av_non_null} fields for {ticker}")
        
        # Step 4: Add Polygon.io data (market data, company info)
        polygon_data = self._get_polygon_data(ticker)
        if polygon_data:
            financial_data = self._merge_data(financial_data, polygon_data)
            if 'Polygon.io' not in financial_data['data_sources']:
                financial_data['data_sources'].append('Polygon.io')
            polygon_non_null = len([v for k, v in polygon_data.items() if k not in ['data_source', 'timestamp', 'ticker'] and v is not None])
            logger.info(f"[Comprehensive Aggregator] Polygon.io: {polygon_non_null} fields for {ticker}")
        
        # Step 5: Add SEC EDGAR data (official, free, authoritative)
        sec_data = self._get_sec_edgar_data(ticker)
        if sec_data:
            financial_data = self._merge_data(financial_data, sec_data)
            if 'SEC EDGAR' not in financial_data['data_sources']:
                financial_data['data_sources'].append('SEC EDGAR')
            sec_non_null = len([v for k, v in sec_data.items() if k not in ['data_source', 'timestamp', 'ticker'] and v is not None])
            logger.info(f"[Comprehensive Aggregator] SEC EDGAR: {sec_non_null} fields for {ticker}")
        
        # Calculate total data coverage
        non_null_count = len([v for k, v in financial_data.items() if k not in ['data_source', 'data_sources', 'timestamp', 'ticker', 'data_coverage'] and v is not None])
        financial_data['data_coverage'] = non_null_count
        financial_data['data_source'] = '+'.join(financial_data['data_sources']) if financial_data['data_sources'] else 'none'
        
        logger.info(f"[Comprehensive Aggregator] ✅ Total for {ticker}: {non_null_count} non-null fields from {len(financial_data['data_sources'])} sources: {financial_data['data_source']}")
        
        # Debug: Log sample key fields to verify data
        key_fields = ['revenue', 'net_income', 'ebitda', 'operating_cash_flow', 'total_assets', 'total_debt', 'market_cap', 'pe_ratio']
        sample_data = {k: financial_data.get(k) for k in key_fields if financial_data.get(k) is not None}
        if sample_data:
            logger.info(f"[Comprehensive Aggregator] Sample fields for {ticker}: {list(sample_data.keys())}")

        # Gson-safe: Android expects a string, not a JSON array
        if isinstance(financial_data.get('data_sources'), list):
            financial_data['data_sources'] = '+'.join(str(s) for s in financial_data['data_sources'] if s)

        return self._normalize_android_fields(financial_data)


# Global instance
comprehensive_financial_aggregator = ComprehensiveFinancialAggregator()

