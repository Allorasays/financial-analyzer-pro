"""
Financial Modeling Prep (FMP) API Service
Provides comprehensive financial data including income statements, balance sheets, cash flow, ratios, and key metrics
"""

import requests
import os
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

class FMPService:
    """Service for fetching data from Financial Modeling Prep API"""
    
    def __init__(self):
        # Personal use: set FMP_API_KEY in your local .env only (no bundled keys).
        self.api_key = os.getenv('FMP_API_KEY', '').strip()
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

    def get_stock_peers(self, ticker: str) -> List[str]:
        """Return peer ticker symbols from FMP (best-effort across endpoint shapes)."""
        for endpoint, params in (
            ("stock-peers", {"symbol": ticker}),
            ("peers", {"symbol": ticker}),
        ):
            data = self._make_request(endpoint, params)
            if not data:
                continue
            # Shapes: [{"symbol":"AAPL","peersList":[...]}] or ["MSFT","GOOGL"] or {"peersList":[...]}
            if isinstance(data, list):
                if data and isinstance(data[0], str):
                    return [str(x).upper() for x in data if x]
                if data and isinstance(data[0], dict):
                    peers = data[0].get("peersList") or data[0].get("peers") or []
                    if isinstance(peers, str):
                        peers = [p.strip() for p in peers.split(",") if p.strip()]
                    if peers:
                        return [str(x).upper() for x in peers if x]
                    # list of peer objects
                    syms = [d.get("symbol") for d in data if isinstance(d, dict) and d.get("symbol")]
                    if syms and ticker.upper() not in syms:
                        return [str(s).upper() for s in syms]
                    return [str(s).upper() for s in syms if str(s).upper() != ticker.upper()]
            if isinstance(data, dict):
                peers = data.get("peersList") or data.get("peers") or []
                if isinstance(peers, str):
                    peers = [p.strip() for p in peers.split(",") if p.strip()]
                if peers:
                    return [str(x).upper() for x in peers if x]
        return []

    def get_peer_snapshots(self, ticker: str, max_peers: int = 6) -> Dict[str, Any]:
        """Peer tickers plus lightweight quote snapshots for relative valuation."""
        symbols = [s for s in self.get_stock_peers(ticker) if s != ticker.upper()][:max_peers]
        peers = []
        for sym in symbols:
            q = self.get_quote(sym) or {}
            peers.append({
                "ticker": sym,
                "price": q.get("price"),
                "pe_ratio": q.get("pe"),
                "market_cap": q.get("marketCap"),
                "change_percent": q.get("changesPercentage") or q.get("changePercentage"),
                "company_name": q.get("name"),
            })
        return {"ticker": ticker.upper(), "peers": peers, "peer_symbols": symbols}

    def get_financial_growth(self, ticker: str) -> Optional[Dict]:
        """YoY growth metrics."""
        data = self._make_request("financial-growth", {"symbol": ticker, "limit": 1})
        return data[0] if data and len(data) > 0 else None

    def get_ratios_ttm(self, ticker: str) -> Optional[Dict]:
        data = self._make_request("ratios-ttm", {"symbol": ticker})
        return data[0] if data and len(data) > 0 else None

    def get_key_metrics_ttm(self, ticker: str) -> Optional[Dict]:
        data = self._make_request("key-metrics-ttm", {"symbol": ticker})
        return data[0] if data and len(data) > 0 else None

    def get_price_target_consensus(self, ticker: str) -> Optional[Dict]:
        data = self._make_request("price-target-consensus", {"symbol": ticker})
        return data[0] if data and len(data) > 0 else None

    @staticmethod
    def _first(d: Optional[Dict], *keys):
        """Return first non-None value for candidate keys (stable API renamed many fields)."""
        if not d:
            return None
        for key in keys:
            val = d.get(key)
            if val is not None and val != "":
                return val
        return None

    @staticmethod
    def _put(target: Dict[str, Any], updates: Dict[str, Any], overwrite: bool = False) -> None:
        """Merge only non-None values so missing keys stay open for later sources/TTM."""
        for key, value in updates.items():
            if value is None or value == "":
                continue
            if overwrite or target.get(key) is None:
                target[key] = value

    def _normalize_field_names(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Map FMP field names to backend/Android FinancialDataResponse keys."""
        aliases = {
            "net_margin": "profit_margin",
            "cash_and_equivalents": "total_cash",
            "year_low": "52_week_low",
            "year_high": "52_week_high",
            "avg_volume": "average_volume",
            "price_change_percent": "change_percent",
            "eps": "earnings_per_share",
            "dividend_per_share": "dividend_rate",
        }
        for src, dest in aliases.items():
            if financial_data.get(src) is not None and financial_data.get(dest) is None:
                financial_data[dest] = financial_data[src]

        # Derived margins when statements exist but ratio endpoints omit them
        rev = financial_data.get("revenue")
        if rev and isinstance(rev, (int, float)) and rev != 0:
            if financial_data.get("ebitda_margin") is None and financial_data.get("ebitda") is not None:
                try:
                    financial_data["ebitda_margin"] = float(financial_data["ebitda"]) / float(rev)
                except (TypeError, ValueError, ZeroDivisionError):
                    pass
            if financial_data.get("gross_margin") is None and financial_data.get("gross_profit") is not None:
                try:
                    financial_data["gross_margin"] = float(financial_data["gross_profit"]) / float(rev)
                except (TypeError, ValueError, ZeroDivisionError):
                    pass
            if financial_data.get("profit_margin") is None and financial_data.get("net_income") is not None:
                try:
                    financial_data["profit_margin"] = float(financial_data["net_income"]) / float(rev)
                except (TypeError, ValueError, ZeroDivisionError):
                    pass
            if financial_data.get("operating_margin") is None and financial_data.get("operating_income") is not None:
                try:
                    financial_data["operating_margin"] = float(financial_data["operating_income"]) / float(rev)
                except (TypeError, ValueError, ZeroDivisionError):
                    pass

        shares = financial_data.get("shares_outstanding")
        if shares and isinstance(shares, (int, float)) and shares != 0:
            if financial_data.get("cash_per_share") is None and financial_data.get("total_cash") is not None:
                try:
                    financial_data["cash_per_share"] = float(financial_data["total_cash"]) / float(shares)
                except (TypeError, ValueError, ZeroDivisionError):
                    pass

        return financial_data

    def get_comprehensive_financial_data(self, ticker: str) -> Dict[str, Any]:
        """Get comprehensive financial data from multiple FMP stable endpoints."""
        if not self.enabled:
            return {}

        try:
            metrics = self.get_key_metrics(ticker)
            metrics_ttm = self.get_key_metrics_ttm(ticker)
            ratios = self.get_financial_ratios(ticker)
            ratios_ttm = self.get_ratios_ttm(ticker)
            income = self.get_income_statement(ticker)
            balance = self.get_balance_sheet(ticker)
            cash_flow = self.get_cash_flow(ticker)
            profile = self.get_company_profile(ticker)
            quote = self.get_quote(ticker)
            growth = self.get_financial_growth(ticker)
            target = self.get_price_target_consensus(ticker)

            first = self._first
            put = self._put
            financial_data: Dict[str, Any] = {}

            if income:
                put(financial_data, {
                    "revenue": income.get("revenue"),
                    "gross_profit": income.get("grossProfit"),
                    "operating_income": income.get("operatingIncome"),
                    "net_income": income.get("netIncome"),
                    "ebitda": income.get("ebitda"),
                    "operating_expenses": income.get("operatingExpenses"),
                    "depreciation_amortization": income.get("depreciationAndAmortization"),
                    "interest_expense": income.get("interestExpense"),
                    "income_tax_expense": income.get("incomeTaxExpense"),
                    "earnings_per_share": first(income, "eps", "epsdiluted", "epsDiluted"),
                    "forward_eps": first(income, "epsDiluted", "eps"),
                })

            if balance:
                put(financial_data, {
                    "total_assets": balance.get("totalAssets"),
                    "total_current_assets": balance.get("totalCurrentAssets"),
                    "cash_and_equivalents": balance.get("cashAndCashEquivalents"),
                    "total_cash": balance.get("cashAndCashEquivalents"),
                    "total_liabilities": balance.get("totalLiabilities"),
                    "total_current_liabilities": balance.get("totalCurrentLiabilities"),
                    "total_debt": balance.get("totalDebt"),
                    "total_equity": balance.get("totalStockholdersEquity"),
                    "retained_earnings": balance.get("retainedEarnings"),
                    "book_value": first(balance, "totalStockholdersEquity"),
                })

            if cash_flow:
                put(financial_data, {
                    "operating_cash_flow": cash_flow.get("operatingCashFlow"),
                    "free_cash_flow": cash_flow.get("freeCashFlow"),
                    "capital_expenditure": cash_flow.get("capitalExpenditure"),
                    "dividends_paid": cash_flow.get("dividendsPaid"),
                })

            # Annual ratios (stable field names + legacy fallbacks)
            if ratios:
                put(financial_data, {
                    "pe_ratio": first(ratios, "priceToEarningsRatio", "peRatio"),
                    "forward_pe": first(ratios, "forwardPriceToEarningsRatio", "forwardPE"),
                    "peg_ratio": first(
                        ratios,
                        "priceToEarningsGrowthRatio",
                        "forwardPriceToEarningsGrowthRatio",
                        "pegRatio",
                    ),
                    "price_to_book": first(ratios, "priceToBookRatio", "pbRatio", "priceToBook"),
                    "price_to_sales": first(ratios, "priceToSalesRatio", "priceToSales"),
                    "earnings_per_share": first(ratios, "netIncomePerShare", "eps"),
                    "revenue_per_share": first(ratios, "revenuePerShare"),
                    "cash_per_share": first(ratios, "cashPerShare"),
                    "current_ratio": first(ratios, "currentRatio"),
                    "quick_ratio": first(ratios, "quickRatio"),
                    "debt_to_equity": first(ratios, "debtToEquityRatio", "debtEquityRatio", "debtToEquity"),
                    "debt_to_assets": first(ratios, "debtToAssetsRatio", "debtRatio", "debtToAssets"),
                    "gross_margin": first(ratios, "grossProfitMargin", "grossMargin"),
                    "operating_margin": first(ratios, "operatingProfitMargin", "operatingMargin"),
                    "profit_margin": first(ratios, "netProfitMargin", "netMargin", "bottomLineProfitMargin"),
                    "ebitda_margin": first(ratios, "ebitdaMargin"),
                    "dividend_yield": first(ratios, "dividendYield", "dividendYieldPercentage"),
                    "dividend_rate": first(ratios, "dividendPerShare"),
                    "dividend_per_share": first(ratios, "dividendPerShare"),
                    "asset_turnover": first(ratios, "assetTurnover"),
                    "inventory_turnover": first(ratios, "inventoryTurnover"),
                })

            if ratios_ttm:
                put(financial_data, {
                    "pe_ratio": first(ratios_ttm, "priceToEarningsRatioTTM"),
                    "peg_ratio": first(
                        ratios_ttm, "priceToEarningsGrowthRatioTTM", "forwardPriceToEarningsGrowthRatioTTM"
                    ),
                    "price_to_book": first(ratios_ttm, "priceToBookRatioTTM"),
                    "price_to_sales": first(ratios_ttm, "priceToSalesRatioTTM"),
                    "revenue_per_share": first(ratios_ttm, "revenuePerShareTTM"),
                    "cash_per_share": first(ratios_ttm, "cashPerShareTTM"),
                    "current_ratio": first(ratios_ttm, "currentRatioTTM"),
                    "quick_ratio": first(ratios_ttm, "quickRatioTTM"),
                    "debt_to_equity": first(ratios_ttm, "debtToEquityRatioTTM"),
                    "gross_margin": first(ratios_ttm, "grossProfitMarginTTM"),
                    "operating_margin": first(ratios_ttm, "operatingProfitMarginTTM"),
                    "profit_margin": first(ratios_ttm, "netProfitMarginTTM"),
                    "ebitda_margin": first(ratios_ttm, "ebitdaMarginTTM"),
                    "dividend_yield": first(ratios_ttm, "dividendYieldTTM"),
                })

            # Key metrics / TTM — ROE/ROA/ROIC live here on stable API
            for src in (metrics, metrics_ttm):
                if not src:
                    continue
                put(financial_data, {
                    "pe_ratio": first(src, "peRatio", "peRatioTTM"),
                    "price_to_book": first(src, "pbRatio", "pbRatioTTM", "priceToBookRatio"),
                    "price_to_sales": first(src, "priceToSalesRatio", "priceToSalesRatioTTM"),
                    "enterprise_value": first(src, "enterpriseValue", "enterpriseValueTTM"),
                    "ev_to_revenue": first(src, "evToSales", "evToSalesTTM"),
                    "ev_to_ebitda": first(src, "evToEBITDA", "evToEbitda", "evToEBITDATTM"),
                    "market_cap": first(src, "marketCap", "marketCapTTM"),
                    "shares_outstanding": first(src, "sharesOutstanding", "weightedAverageShsOut"),
                    "return_on_equity": first(src, "returnOnEquity", "returnOnEquityTTM"),
                    "return_on_assets": first(src, "returnOnAssets", "returnOnAssetsTTM"),
                    "return_on_invested_capital": first(
                        src,
                        "returnOnInvestedCapital",
                        "returnOnInvestedCapitalTTM",
                        "returnOnCapitalEmployed",
                        "returnOnCapitalEmployedTTM",
                    ),
                    "current_ratio": first(src, "currentRatio", "currentRatioTTM"),
                    "free_cash_flow": first(src, "freeCashFlowToFirm"),
                })

            if growth:
                put(financial_data, {
                    "revenue_growth": first(growth, "revenueGrowth"),
                    "earnings_growth": first(growth, "netIncomeGrowth", "epsgrowth", "epsdilutedGrowth"),
                    "eps_growth": first(growth, "epsgrowth", "epsdilutedGrowth"),
                    "operating_income_growth": first(growth, "operatingIncomeGrowth"),
                    "free_cash_flow_growth": first(growth, "freeCashFlowGrowth"),
                })

            if profile:
                put(financial_data, {
                    "company_name": profile.get("companyName"),
                    "industry": profile.get("industry"),
                    "sector": profile.get("sector"),
                    "website": profile.get("website"),
                    "description": profile.get("description"),
                    "ceo": profile.get("ceo"),
                    "employees": profile.get("fullTimeEmployees"),
                    "beta": first(profile, "beta"),
                    "average_volume": first(profile, "averageVolume"),
                    "dividend_rate": first(profile, "lastDividend"),
                })

            if quote:
                put(financial_data, {
                    "current_price": quote.get("price"),
                    "price_change": quote.get("change"),
                    "price_change_percent": first(quote, "changePercentage", "changesPercentage"),
                    "day_low": quote.get("dayLow"),
                    "day_high": quote.get("dayHigh"),
                    "year_low": quote.get("yearLow"),
                    "year_high": quote.get("yearHigh"),
                    "52_week_low": quote.get("yearLow"),
                    "52_week_high": quote.get("yearHigh"),
                    "volume": quote.get("volume"),
                    "avg_volume": first(quote, "avgVolume", "averageVolume"),
                    "market_cap": quote.get("marketCap"),
                    "previous_close": quote.get("previousClose"),
                    "open": quote.get("open"),
                    "pe_ratio": first(quote, "pe"),
                    "earnings_per_share": first(quote, "eps"),
                })

            if target:
                put(financial_data, {
                    "target_mean_price": first(target, "targetConsensus", "targetMedian"),
                    "target_high_price": first(target, "targetHigh"),
                    "target_low_price": first(target, "targetLow"),
                    "recommendation_key": "consensus_target",
                })

            # Shares for per-share calcs if missing
            if financial_data.get("shares_outstanding") is None and income:
                put(financial_data, {
                    "shares_outstanding": first(
                        income, "weightedAverageShsOut", "weightedAverageShsOutDil"
                    )
                })

            financial_data = self._normalize_field_names(financial_data)
            financial_data["data_source"] = "FMP"
            financial_data["timestamp"] = datetime.now().isoformat()

            data_count = len([v for v in financial_data.values() if v is not None and v != ""])
            logger.info(f"FMP comprehensive data for {ticker}: {data_count} non-null fields")
            if data_count < 10:
                logger.warning(f"FMP returned limited data for {ticker}: only {data_count} fields")

            return financial_data

        except Exception as e:
            logger.error(f"Error fetching comprehensive FMP data for {ticker}: {e}", exc_info=True)
            return {}

# Global instance
fmp_service = FMPService()


