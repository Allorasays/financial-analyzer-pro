import streamlit as st
import requests
import pandas as pd
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import yfinance as yf
import asyncio
import time
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Performance monitoring
import time
from functools import wraps

def performance_monitor(func):
    """Decorator to monitor function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            if execution_time > 1.0:  # Log slow functions
                st.warning(f"⚠️ {func.__name__} took {execution_time:.2f}s to execute")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            st.error(f"❌ {func.__name__} failed after {execution_time:.2f}s: {e}")
            raise
    return wrapper

# Enhanced data fetching with caching and performance monitoring
@st.cache_data(ttl=300)  # Cache for 5 minutes
@performance_monitor
def fetch_real_time_data(ticker: str) -> Dict:
    """Fetch real-time market data for a ticker"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get real-time price data
        hist = stock.history(period="5d")
        current_price = hist['Close'].iloc[-1] if not hist.empty else None
        price_change = hist['Close'].iloc[-1] - hist['Close'].iloc[-2] if len(hist) > 1 else 0
        price_change_pct = (price_change / hist['Close'].iloc[-2] * 100) if len(hist) > 1 and hist['Close'].iloc[-2] != 0 else 0
        
        return {
            'current_price': current_price,
            'price_change': price_change,
            'price_change_pct': price_change_pct,
            'volume': info.get('volume', 0),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'dividend_yield': info.get('dividendYield', 0),
            'beta': info.get('beta', 0),
            'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0),
            'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0),
            'avg_volume': info.get('averageVolume', 0),
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    except Exception as e:
        st.error(f"Error fetching real-time data: {e}")
        return {}

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_enhanced_financials(ticker: str) -> Dict:
    """Fetch enhanced financial data with multiple sources"""
    try:
        # Primary API call
        import os
        api_base_url = os.getenv('API_BASE_URL', 'http://localhost:8000')
        response = requests.get(f"{api_base_url}/api/financials/{ticker}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Enhance with yfinance data if available
            try:
                stock = yf.Ticker(ticker)
                yf_info = stock.info
                
                # Add additional metrics
                enhanced_data = {
                    'financials': data,
                    'market_data': {
                        'sector': yf_info.get('sector', 'N/A'),
                        'industry': yf_info.get('industry', 'N/A'),
                        'country': yf_info.get('country', 'N/A'),
                        'employees': yf_info.get('fullTimeEmployees', 0),
                        'website': yf_info.get('website', 'N/A'),
                        'business_summary': yf_info.get('longBusinessSummary', 'N/A')
                    }
                }
                return enhanced_data
            except:
                return {'financials': data, 'market_data': {}}
        else:
            return {}
            
    except Exception as e:
        st.error(f"Error fetching financial data: {e}")
        return {}

# Health check for Render
import os
if os.getenv('RENDER'):
    st.set_page_config(
        page_title="Financial Analyzer Pro",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
else:
    st.set_page_config(
        page_title="Financial Analyzer Pro",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
    }
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>📊 Financial Analyzer Pro</h1>
    <p>Advanced financial analysis and market insights</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state for custom criteria and portfolio
if 'custom_criteria' not in st.session_state:
    st.session_state.custom_criteria = []
if 'use_custom_criteria' not in st.session_state:
    st.session_state.use_custom_criteria = False
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = []
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'
if 'export_format' not in st.session_state:
    st.session_state.export_format = 'csv'

# Predefined financial analysis criteria
PREDEFINED_CRITERIA = {
    'Liquidity Ratios': {
        'Current Ratio': {'formula': 'TotalCurrentAssets / TotalCurrentLiabilities', 'threshold': 1.5, 'description': 'Measures ability to pay short-term obligations'},
        'Quick Ratio': {'formula': '(TotalCurrentAssets - Inventory) / TotalCurrentLiabilities', 'threshold': 1.0, 'description': 'Acid test ratio excluding inventory'},
        'Cash Ratio': {'formula': 'CashAndCashEquivalents / TotalCurrentLiabilities', 'threshold': 0.5, 'description': 'Most conservative liquidity measure'}
    },
    'Profitability Ratios': {
        'Gross Margin': {'formula': '(Revenue - CostOfGoodsSold) / Revenue * 100', 'threshold': 30.0, 'description': 'Profit after direct costs'},
        'Operating Margin': {'formula': 'OperatingIncome / Revenue * 100', 'threshold': 15.0, 'description': 'Profit after operating expenses'},
        'Net Profit Margin': {'formula': 'NetIncome / Revenue * 100', 'threshold': 10.0, 'description': 'Final profit percentage'},
        'ROE': {'formula': 'NetIncome / TotalStockholdersEquity * 100', 'threshold': 15.0, 'description': 'Return on equity'},
        'ROA': {'formula': 'NetIncome / TotalAssets * 100', 'threshold': 8.0, 'description': 'Return on assets'}
    },
    'Efficiency Ratios': {
        'Asset Turnover': {'formula': 'Revenue / TotalAssets', 'threshold': 1.0, 'description': 'Asset utilization efficiency'},
        'Inventory Turnover': {'formula': 'CostOfGoodsSold / Inventory', 'threshold': 5.0, 'description': 'Inventory management efficiency'},
        'Receivables Turnover': {'formula': 'Revenue / AccountsReceivable', 'threshold': 10.0, 'description': 'Collection efficiency'}
    },
    'Leverage Ratios': {
        'Debt to Equity': {'formula': 'TotalLiabilities / TotalStockholdersEquity', 'threshold': 0.5, 'description': 'Financial leverage level'},
        'Debt to Assets': {'formula': 'TotalLiabilities / TotalAssets', 'threshold': 0.4, 'description': 'Asset financing through debt'},
        'Interest Coverage': {'formula': 'OperatingIncome / InterestExpense', 'threshold': 3.0, 'description': 'Ability to pay interest'}
    },
    'Growth Metrics': {
        'Revenue Growth': {'formula': '(Revenue - Revenue_prev) / Revenue_prev * 100', 'threshold': 10.0, 'description': 'Annual revenue growth'},
        'EPS Growth': {'formula': '(EarningsPerShare - EPS_prev) / EPS_prev * 100', 'threshold': 8.0, 'description': 'Earnings per share growth'},
        'Asset Growth': {'formula': '(TotalAssets - TotalAssets_prev) / TotalAssets_prev * 100', 'threshold': 5.0, 'description': 'Asset base expansion'}
    }
}

def calculate_financial_metric(formula, data, year_index=0):
    """Calculate financial metric based on formula and data"""
    try:
        # Input validation
        if not isinstance(formula, str) or not formula.strip():
            st.warning("⚠️ Invalid formula provided")
            return None
        
        if data is None or data.empty:
            st.warning("⚠️ No data provided for calculation")
            return None
        
        if year_index < 0 or year_index >= len(data):
            st.warning(f"⚠️ Invalid year index: {year_index}")
            return None
        
        # Create a safe evaluation environment
        local_vars = {}
        
        # Add available data columns to local variables
        for col in data.columns:
            if col in data.columns and not data[col].isna().iloc[year_index]:
                local_vars[col] = data[col].iloc[year_index]
        
        # Handle previous year data for growth calculations
        if year_index < len(data) - 1:
            for col in data.columns:
                if col in data.columns and not data[col].isna().iloc[year_index + 1]:
                    local_vars[f"{col}_prev"] = data[col].iloc[year_index + 1]
        
        # Replace common financial terms with available data
        formula_safe = formula
        replacements = {
            'TotalCurrentAssets': 'CurrentAssets' if 'CurrentAssets' in local_vars else 'TotalAssets',
            'TotalCurrentLiabilities': 'CurrentLiabilities' if 'CurrentLiabilities' in local_vars else 'TotalLiabilities',
            'CashAndCashEquivalents': 'Cash' if 'Cash' in local_vars else 'CashAndCashEquivalents',
            'CostOfGoodsSold': 'CostOfRevenue' if 'CostOfRevenue' in local_vars else 'CostOfGoodsSold',
            'OperatingIncome': 'OperatingIncome' if 'OperatingIncome' in local_vars else 'EBIT',
            'TotalStockholdersEquity': 'StockholdersEquity' if 'StockholdersEquity' in local_vars else 'TotalEquity',
            'AccountsReceivable': 'Receivables' if 'Receivables' in local_vars else 'AccountsReceivable',
            'EarningsPerShare': 'EPS' if 'EPS' in local_vars else 'EarningsPerShare'
        }
        
        for old, new in replacements.items():
            if old in formula_safe and new in local_vars:
                formula_safe = formula_safe.replace(old, new)
        
        # Validate formula contains only safe operations
        safe_chars = set('0123456789.+-*/()_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ')
        if not all(c in safe_chars for c in formula_safe):
            st.error("❌ Formula contains unsafe characters")
            return None
        
        # Evaluate the formula safely
        result = eval(formula_safe, {"__builtins__": {}}, local_vars)
        
        # Validate result
        if result is None or pd.isna(result):
            return None
        
        # Check for infinite or extremely large values
        if abs(result) > 1e15:
            st.warning("⚠️ Calculated value is extremely large")
            return None
        
        return result
        
    except ZeroDivisionError:
        st.warning("⚠️ Division by zero in formula")
        return None
    except Exception as e:
        st.error(f"❌ Error calculating metric: {str(e)}")
        return None

def get_metric_status(value, threshold, metric_type='ratio'):
    """Determine if metric meets threshold criteria"""
    if value is None or pd.isna(value):
        return "⚠️ Insufficient Data"
    
    if metric_type == 'ratio':
        if value >= threshold:
            return "✅ Good"
        elif value >= threshold * 0.8:
            return "🟡 Acceptable"
        else:
            return "❌ Poor"
    elif metric_type == 'percentage':
        if value >= threshold:
            return "✅ Good"
        elif value >= threshold * 0.8:
            return "🟡 Acceptable"
        else:
            return "❌ Poor"
    else:
        return "ℹ️ N/A"

def calculate_dcf_valuation(financial_data: pd.DataFrame, discount_rate: float = 0.1, growth_rate: float = 0.03) -> Dict:
    """Calculate DCF (Discounted Cash Flow) valuation"""
    try:
        if len(financial_data) < 2:
            return {'dcf_value': None, 'error': 'Insufficient data for DCF calculation'}
        
        # Get latest free cash flow (simplified calculation)
        latest_revenue = financial_data['Revenue'].iloc[0] if 'Revenue' in financial_data.columns else 0
        latest_net_income = financial_data['NetIncome'].iloc[0] if 'NetIncome' in financial_data.columns else 0
        
        # Estimate free cash flow (simplified)
        fcf = latest_net_income * 0.8  # Assume 80% of net income becomes FCF
        
        # Calculate terminal value
        terminal_value = fcf * (1 + growth_rate) / (discount_rate - growth_rate)
        
        # Calculate present value of FCF for next 5 years
        pv_fcf = 0
        for year in range(1, 6):
            pv_fcf += fcf * (1 + growth_rate) ** year / (1 + discount_rate) ** year
        
        # Add terminal value
        pv_terminal = terminal_value / (1 + discount_rate) ** 5
        
        dcf_value = pv_fcf + pv_terminal
        
        return {
            'dcf_value': dcf_value,
            'fcf': fcf,
            'terminal_value': terminal_value,
            'pv_fcf': pv_fcf,
            'pv_terminal': pv_terminal,
            'assumptions': {
                'discount_rate': discount_rate,
                'growth_rate': growth_rate
            }
        }
    except Exception as e:
        return {'dcf_value': None, 'error': str(e)}

def calculate_risk_metrics(financial_data: pd.DataFrame) -> Dict:
    """Calculate various risk metrics"""
    try:
        if len(financial_data) < 2:
            return {}
        
        # Volatility calculation
        if 'Revenue' in financial_data.columns:
            revenue_volatility = financial_data['Revenue'].pct_change().std() * np.sqrt(12)  # Annualized
        else:
            revenue_volatility = None
        
        # Beta calculation (simplified)
        if 'NetIncome' in financial_data.columns:
            income_volatility = financial_data['NetIncome'].pct_change().std() * np.sqrt(12)
        else:
            income_volatility = None
        
        # Debt coverage ratio
        if 'TotalLiabilities' in financial_data.columns and 'OperatingIncome' in financial_data.columns:
            debt_coverage = financial_data['OperatingIncome'].iloc[0] / financial_data['TotalLiabilities'].iloc[0] if financial_data['TotalLiabilities'].iloc[0] != 0 else None
        else:
            debt_coverage = None
        
        return {
            'revenue_volatility': revenue_volatility,
            'income_volatility': income_volatility,
            'debt_coverage_ratio': debt_coverage,
            'risk_level': 'High' if (revenue_volatility and revenue_volatility > 0.3) else 'Medium' if (revenue_volatility and revenue_volatility > 0.15) else 'Low'
        }
    except Exception as e:
        return {'error': str(e)}

def perform_sensitivity_analysis(base_value: float, variables: Dict[str, float], ranges: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    """Perform sensitivity analysis on key variables"""
    try:
        results = []
        
        for var_name, base_var_value in variables.items():
            if var_name in ranges:
                min_val, max_val = ranges[var_name]
                steps = 5
                
                for i in range(steps + 1):
                    current_val = min_val + (max_val - min_val) * i / steps
                    
                    # Calculate impact (simplified)
                    impact = base_value * (current_val / base_var_value)
                    
                    results.append({
                        'Variable': var_name,
                        'Value': current_val,
                        'Impact': impact,
                        'Change_Percent': ((current_val - base_var_value) / base_var_value) * 100
                    })
        
        return pd.DataFrame(results)
    except Exception as e:
        return pd.DataFrame({'error': [str(e)]})

def predict_financial_metrics(financial_data: pd.DataFrame, forecast_periods: int = 3) -> Dict:
    """Predict future financial metrics using simple ML models"""
    try:
        if len(financial_data) < 3:
            return {'error': 'Insufficient data for prediction (need at least 3 years)'}
        
        predictions = {}
        
        # Simple linear regression for each metric
        for column in financial_data.columns:
            if column != 'Year' and financial_data[column].dtype in ['int64', 'float64']:
                try:
                    # Prepare data
                    X = np.arange(len(financial_data)).reshape(-1, 1)
                    y = financial_data[column].values
                    
                    # Remove NaN values
                    mask = ~np.isnan(y)
                    if np.sum(mask) > 1:
                        X_clean = X[mask]
                        y_clean = y[mask]
                        
                        # Simple linear regression
                        if len(X_clean) > 1:
                            # Calculate trend
                            slope = np.polyfit(X_clean.flatten(), y_clean, 1)[0]
                            
                            # Predict future values
                            future_X = np.arange(len(financial_data), len(financial_data) + forecast_periods).reshape(-1, 1)
                            future_values = []
                            
                            for i in range(forecast_periods):
                                pred_value = y_clean[-1] + slope * (i + 1)
                                future_values.append(max(0, pred_value))  # Ensure non-negative
                            
                            predictions[column] = {
                                'trend_slope': slope,
                                'future_values': future_values,
                                'confidence': 'Medium' if abs(slope) < 0.1 else 'High' if abs(slope) > 0.2 else 'Medium'
                            }
                except:
                    continue
        
        return predictions
    except Exception as e:
        return {'error': str(e)}

def detect_anomalies(financial_data: pd.DataFrame, threshold: float = 2.0) -> Dict:
    """Detect anomalies in financial data using statistical methods"""
    try:
        anomalies = {}
        
        for column in financial_data.columns:
            if column != 'Year' and financial_data[column].dtype in ['int64', 'float64']:
                try:
                    # Calculate z-scores
                    values = financial_data[column].dropna()
                    if len(values) > 1:
                        mean_val = values.mean()
                        std_val = values.std()
                        
                        if std_val > 0:
                            z_scores = np.abs((values - mean_val) / std_val)
                            anomaly_indices = np.where(z_scores > threshold)[0]
                            
                            if len(anomaly_indices) > 0:
                                anomalies[column] = {
                                    'anomaly_years': financial_data.iloc[anomaly_indices]['Year'].tolist(),
                                    'anomaly_values': values.iloc[anomaly_indices].tolist(),
                                    'z_scores': z_scores[anomaly_indices].tolist(),
                                    'severity': 'High' if np.max(z_scores) > 3 else 'Medium'
                                }
                except:
                    continue
        
        return anomalies
    except Exception as e:
        return {'error': str(e)}

def calculate_risk_score(financial_data: pd.DataFrame) -> Dict:
    """Calculate comprehensive risk score based on multiple factors"""
    try:
        if len(financial_data) < 2:
            return {'error': 'Insufficient data for risk assessment'}
        
        risk_factors = {}
        total_risk_score = 0
        max_possible_score = 100
        
        # Revenue volatility risk (25 points)
        if 'Revenue' in financial_data.columns:
            revenue_volatility = financial_data['Revenue'].pct_change().std()
            revenue_risk = min(25, revenue_volatility * 100)
            risk_factors['Revenue Volatility'] = revenue_risk
            total_risk_score += revenue_risk
        
        # Profit margin stability (25 points)
        if 'Revenue' in financial_data.columns and 'NetIncome' in financial_data.columns:
            margins = (financial_data['NetIncome'] / financial_data['Revenue'] * 100).dropna()
            if len(margins) > 1:
                margin_volatility = margins.std()
                margin_risk = min(25, margin_volatility * 2)
                risk_factors['Margin Stability'] = margin_risk
                total_risk_score += margin_risk
        
        # Growth consistency (20 points)
        if 'Revenue' in financial_data.columns and len(financial_data) > 2:
            growth_rates = financial_data['Revenue'].pct_change().dropna()
            growth_volatility = growth_rates.std()
            growth_risk = min(20, growth_volatility * 50)
            risk_factors['Growth Consistency'] = growth_risk
            total_risk_score += growth_risk
        
        # Financial leverage risk (15 points)
        if 'TotalLiabilities' in financial_data.columns and 'TotalAssets' in financial_data.columns:
            leverage_ratio = financial_data['TotalLiabilities'].iloc[-1] / financial_data['TotalAssets'].iloc[-1]
            leverage_risk = min(15, leverage_ratio * 30)
            risk_factors['Financial Leverage'] = leverage_risk
            total_risk_score += leverage_risk
        
        # Liquidity risk (15 points)
        if 'TotalCurrentAssets' in financial_data.columns and 'TotalCurrentLiabilities' in financial_data.columns:
            current_ratio = financial_data['TotalCurrentAssets'].iloc[-1] / financial_data['TotalCurrentLiabilities'].iloc[-1]
            liquidity_risk = 15 if current_ratio < 1 else max(0, 15 - (current_ratio - 1) * 5)
            risk_factors['Liquidity'] = liquidity_risk
            total_risk_score += liquidity_risk
        
        # Calculate overall risk level
        risk_percentage = (total_risk_score / max_possible_score) * 100
        
        if risk_percentage < 30:
            risk_level = "Low"
            risk_color = "green"
        elif risk_percentage < 60:
            risk_level = "Medium"
            risk_color = "orange"
        else:
            risk_level = "High"
            risk_color = "red"
        
        return {
            'total_risk_score': total_risk_score,
            'risk_percentage': risk_percentage,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'risk_factors': risk_factors,
            'max_possible_score': max_possible_score
        }
    except Exception as e:
        return {'error': str(e)}

# Sidebar
with st.sidebar:
    st.header("🔑 Authentication")
    st.info("Auth0 integration coming soon")
    
    st.header("⚙️ Settings")
    chart_theme = st.selectbox("Chart Theme", ["plotly", "plotly_white", "plotly_dark"])
    default_ticker = st.text_input("Default Ticker", value="AAPL")
    
    st.header("📊 Analysis Criteria")
    use_custom = st.checkbox("Use Custom Criteria", value=st.session_state.use_custom_criteria)
    st.session_state.use_custom_criteria = use_custom
    
    # Show predefined criteria preview
    if not use_custom:
        st.subheader("📚 Available Predefined Criteria")
        with st.expander("View Predefined Metrics"):
            for category, metrics in PREDEFINED_CRITERIA.items():
                st.write(f"**{category}:**")
                for metric_name, metric_data in metrics.items():
                    st.write(f"• {metric_name}: {metric_data['description']}")
                st.write("---")
    
    if use_custom:
        st.subheader("➕ Add Custom Metric")
        with st.form("add_criteria"):
            metric_name = st.text_input("Metric Name", placeholder="e.g., Custom Ratio")
            formula = st.text_input("Formula", placeholder="e.g., Revenue / TotalAssets")
            description = st.text_input("Description", placeholder="Brief description of the metric")
            threshold = st.number_input("Target Threshold", value=1.0, step=0.1)
            
            if st.form_submit_button("Add Metric"):
                if metric_name and formula:
                    new_criteria = {
                        'name': metric_name,
                        'formula': formula,
                        'description': description,
                        'threshold': threshold
                    }
                    st.session_state.custom_criteria.append(new_criteria)
                    st.success(f"Added: {metric_name}")
                    st.rerun()
        
        # Display custom criteria
        if st.session_state.custom_criteria:
            st.subheader("📋 Custom Metrics")
            for i, criteria in enumerate(st.session_state.custom_criteria):
                with st.expander(f"{criteria['name']} - {criteria['description']}"):
                    st.write(f"**Formula:** {criteria['formula']}")
                    st.write(f"**Target:** {criteria['threshold']}")
                    if st.button(f"Delete {criteria['name']}", key=f"del_{i}"):
                        st.session_state.custom_criteria.pop(i)
                        st.rerun()
            
            # Criteria management
            st.subheader("💾 Save/Load Criteria")
            
            # Save criteria
            if st.button("💾 Save Current Criteria"):
                criteria_json = {
                    'name': f"Criteria_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    'criteria': st.session_state.custom_criteria,
                    'timestamp': datetime.now().isoformat()
                }
                st.download_button(
                    label="📥 Download Criteria File",
                    data=str(criteria_json),
                    file_name=f"financial_criteria_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            # Load criteria
            uploaded_file = st.file_uploader("📤 Load Criteria from File", type=['json'])
            if uploaded_file is not None:
                try:
                    import json
                    loaded_criteria = json.load(uploaded_file)
                    if 'criteria' in loaded_criteria:
                        st.session_state.custom_criteria = loaded_criteria['criteria']
                        st.success(f"Loaded {len(loaded_criteria['criteria'])} criteria from {loaded_criteria.get('name', 'file')}")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error loading criteria: {e}")
    
    st.header("📈 Quick Actions")
    if st.button("📊 Market Overview"):
        st.session_state.show_market_overview = True
    
    if st.button("🏭 Industry Analysis"):
        st.session_state.show_industry_analysis = True
    
    st.header("💼 Portfolio Management")
    
    # Add to portfolio
    with st.expander("➕ Add to Portfolio"):
        portfolio_ticker = st.text_input("Ticker", key="portfolio_ticker")
        portfolio_shares = st.number_input("Shares", min_value=0.0, value=100.0, key="portfolio_shares")
        portfolio_price = st.number_input("Purchase Price", min_value=0.0, value=0.0, key="portfolio_price")
        
        if st.button("Add to Portfolio", key="add_portfolio"):
            if portfolio_ticker and portfolio_shares > 0:
                # Get current price if not provided
                if portfolio_price == 0:
                    real_time_data = fetch_real_time_data(portfolio_ticker.upper())
                    portfolio_price = real_time_data.get('current_price', 0)
                
                portfolio_item = {
                    'ticker': portfolio_ticker.upper(),
                    'shares': portfolio_shares,
                    'purchase_price': portfolio_price,
                    'date_added': datetime.now().strftime('%Y-%m-%d')
                }
                st.session_state.portfolio.append(portfolio_item)
                st.success(f"Added {portfolio_ticker.upper()} to portfolio")
                st.rerun()
    
    # Portfolio overview
    if st.session_state.portfolio:
        st.subheader("📊 Portfolio Overview")
        total_value = 0
        total_cost = 0
        
        for item in st.session_state.portfolio:
            real_time_data = fetch_real_time_data(item['ticker'])
            current_price = real_time_data.get('current_price', item['purchase_price'])
            current_value = current_price * item['shares']
            cost_basis = item['purchase_price'] * item['shares']
            gain_loss = current_value - cost_basis
            gain_loss_pct = (gain_loss / cost_basis * 100) if cost_basis > 0 else 0
            
            total_value += current_value
            total_cost += cost_basis
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(f"**{item['ticker']}** ({item['shares']} shares)")
            with col2:
                st.write(f"${current_value:,.0f}")
                color = "green" if gain_loss >= 0 else "red"
                st.markdown(f"<span style='color: {color}'>{gain_loss_pct:+.1f}%</span>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.metric("Total Portfolio Value", f"${total_value:,.0f}")
        st.metric("Total Cost Basis", f"${total_cost:,.0f}")
        overall_gain_loss = total_value - total_cost
        overall_gain_loss_pct = (overall_gain_loss / total_cost * 100) if total_cost > 0 else 0
        color = "green" if overall_gain_loss >= 0 else "red"
        st.markdown(f"**Overall P&L:** <span style='color: {color}'>${overall_gain_loss:+,.0f} ({overall_gain_loss_pct:+.1f}%)</span>", unsafe_allow_html=True)
    
    # Watchlist management
    st.subheader("👀 Watchlist")
    watchlist_ticker = st.text_input("Add to Watchlist", key="watchlist_ticker")
    if st.button("Add", key="add_watchlist"):
        if watchlist_ticker and watchlist_ticker.upper() not in [w['ticker'] for w in st.session_state.watchlist]:
            st.session_state.watchlist.append({
                'ticker': watchlist_ticker.upper(),
                'date_added': datetime.now().strftime('%Y-%m-%d')
            })
            st.success(f"Added {watchlist_ticker.upper()} to watchlist")
            st.rerun()
    
    if st.session_state.watchlist:
        for i, item in enumerate(st.session_state.watchlist):
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"**{item['ticker']}**")
            with col2:
                real_time_data = fetch_real_time_data(item['ticker'])
                current_price = real_time_data.get('current_price', 'N/A')
                if current_price != 'N/A':
                    st.write(f"${current_price:.2f}")
                else:
                    st.write("N/A")
            with col3:
                if st.button("Remove", key=f"remove_watchlist_{i}"):
                    st.session_state.watchlist.pop(i)
                    st.rerun()

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    ticker = st.text_input("Enter Stock Ticker:", value=default_ticker, placeholder="e.g., TSLA, AAPL, MSFT")
    
with col2:
    analysis_type = st.selectbox("Analysis Type", ["Financial Metrics", "Technical Analysis", "Valuation", "Risk Assessment"])

if ticker:
    st.markdown(f"## 📈 Analysis for {ticker.upper()}")
    
    # Fetch enhanced data
    try:
        enhanced_data = fetch_enhanced_financials(ticker)
        real_time_data = fetch_real_time_data(ticker)
        
        if enhanced_data and 'financials' in enhanced_data:
            data = enhanced_data['financials']
            df = pd.DataFrame(data)
            market_data = enhanced_data.get('market_data', {})
            
            # Real-time market data display
            if real_time_data:
                st.subheader("📊 Real-time Market Data")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    current_price = real_time_data.get('current_price', 'N/A')
                    price_change = real_time_data.get('price_change', 0)
                    price_change_pct = real_time_data.get('price_change_pct', 0)
                    
                    if current_price != 'N/A':
                        st.metric(
                            "Current Price", 
                            f"${current_price:.2f}",
                            delta=f"{price_change:+.2f} ({price_change_pct:+.1f}%)"
                        )
                    else:
                        st.metric("Current Price", "N/A")
                
                with col2:
                    market_cap = real_time_data.get('market_cap', 0)
                    st.metric("Market Cap", f"${market_cap:,.0f}M" if market_cap > 0 else "N/A")
                
                with col3:
                    pe_ratio = real_time_data.get('pe_ratio', 0)
                    st.metric("P/E Ratio", f"{pe_ratio:.2f}" if pe_ratio > 0 else "N/A")
                
                with col4:
                    volume = real_time_data.get('volume', 0)
                    st.metric("Volume", f"{volume:,.0f}" if volume > 0 else "N/A")
                
                # Additional market info
                if market_data:
                    st.subheader("🏢 Company Information")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**Sector:** {market_data.get('sector', 'N/A')}")
                        st.write(f"**Industry:** {market_data.get('industry', 'N/A')}")
                    
                    with col2:
                        st.write(f"**Country:** {market_data.get('country', 'N/A')}")
                        st.write(f"**Employees:** {market_data.get('employees', 'N/A'):,}" if market_data.get('employees') else "**Employees:** N/A")
                    
                    with col3:
                        st.write(f"**Website:** {market_data.get('website', 'N/A')}")
                    
                    if market_data.get('business_summary'):
                        with st.expander("📝 Business Summary"):
                            st.write(market_data['business_summary'])
            
            # Key Financial Metrics
            st.subheader("💰 Key Financial Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                latest_revenue = df['Revenue'].iloc[-1] if 'Revenue' in df.columns else 0
                st.metric("Latest Revenue", f"${latest_revenue:,.0f}M")
            
            with col2:
                latest_net_income = df['NetIncome'].iloc[-1] if 'NetIncome' in df.columns else 0
                st.metric("Latest Net Income", f"${latest_net_income:,.0f}M")
            
            with col3:
                if len(df) > 1:
                    revenue_growth = ((df['Revenue'].iloc[-1] - df['Revenue'].iloc[-2]) / df['Revenue'].iloc[-2]) * 100
                    st.metric("Revenue Growth YoY", f"{revenue_growth:.1f}%")
                else:
                    st.metric("Revenue Growth YoY", "N/A")
            
            with col4:
                if 'Revenue' in df.columns and 'NetIncome' in df.columns:
                    profit_margin = (df['NetIncome'].iloc[-1] / df['Revenue'].iloc[-1]) * 100
                    st.metric("Profit Margin", f"{profit_margin:.1f}%")
                else:
                    st.metric("Profit Margin", "N/A")
            
            # Charts
            tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
                "📊 Financial Trends", "📈 Performance Metrics", "🏭 Peer Comparison", "📋 Data Table", 
                "🎯 Criteria Analysis", "💰 DCF Valuation", "⚠️ Risk Analysis", "📊 Sensitivity Analysis", "🤖 ML Analysis"
            ])
            
            with tab1:
                if "Revenue" in df.columns and "NetIncome" in df.columns:
                    # Enhanced interactive chart
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=df['Year'],
                        y=df['Revenue'],
                        mode='lines+markers',
                        name='Revenue',
                        line=dict(color='#667eea', width=3),
                        marker=dict(size=8),
                        hovertemplate='<b>Year:</b> %{x}<br><b>Revenue:</b> $%{y:,.0f}M<extra></extra>'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=df['Year'],
                        y=df['NetIncome'],
                        mode='lines+markers',
                        name='Net Income',
                        line=dict(color='#764ba2', width=3),
                        marker=dict(size=8),
                        yaxis='y2',
                        hovertemplate='<b>Year:</b> %{x}<br><b>Net Income:</b> $%{y:,.0f}M<extra></extra>'
                    ))
                    
                    # Add trend line for revenue
                    if len(df) > 2:
                        z = np.polyfit(range(len(df)), df['Revenue'], 1)
                        p = np.poly1d(z)
                        fig.add_trace(go.Scatter(
                            x=df['Year'],
                            y=p(range(len(df))),
                            mode='lines',
                            name='Revenue Trend',
                            line=dict(color='#667eea', width=2, dash='dash'),
                            showlegend=True
                        ))
                    
                    fig.update_layout(
                        title=f"{ticker.upper()} Financial Performance Analysis",
                        xaxis_title="Year",
                        yaxis_title="Revenue ($M)",
                        yaxis2=dict(title="Net Income ($M)", overlaying="y", side="right"),
                        template=chart_theme,
                        height=500,
                        hovermode='x unified',
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                                # Add financial health indicators
            if len(df) > 1:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Revenue stability
                    revenue_std = df['Revenue'].std()
                    revenue_mean = df['Revenue'].mean()
                    revenue_cv = revenue_std / revenue_mean if revenue_mean > 0 else 0
                    stability_color = "green" if revenue_cv < 0.2 else "orange" if revenue_cv < 0.4 else "red"
                    st.markdown(f"**Revenue Stability:** <span style='color: {stability_color}'>{revenue_cv:.1%}</span>", unsafe_allow_html=True)
                
                with col2:
                    # Profit margin trend
                    if 'NetIncome' in df.columns:
                        margins = (df['NetIncome'] / df['Revenue'] * 100).dropna()
                        if len(margins) > 1:
                            margin_trend = margins.iloc[-1] - margins.iloc[0]
                            trend_color = "green" if margin_trend > 0 else "red"
                            st.markdown(f"**Margin Trend:** <span style='color: {trend_color}'>{margin_trend:+.1f}%</span>", unsafe_allow_html=True)
                
                with col3:
                    # Growth consistency
                    if len(df) > 2:
                        growth_rates = df['Revenue'].pct_change().dropna()
                        growth_consistency = 1 - growth_rates.std()
                        consistency_color = "green" if growth_consistency > 0.7 else "orange" if growth_consistency > 0.5 else "red"
                        st.markdown(f"**Growth Consistency:** <span style='color: {consistency_color}'>{growth_consistency:.1%}</span>", unsafe_allow_html=True)
                
                # Advanced visualization options
                st.subheader("📊 Advanced Visualizations")
                
                viz_type = st.selectbox(
                    "Choose Visualization Type:",
                    ["Candlestick Chart", "Area Chart", "Scatter Plot", "Box Plot", "Correlation Heatmap"]
                )
                
                if viz_type == "Candlestick Chart" and len(df) > 1:
                    # Create candlestick-style chart for financial metrics
                    fig_candlestick = go.Figure()
                    
                    # Revenue as candlestick body
                    fig_candlestick.add_trace(go.Bar(
                        x=df['Year'],
                        y=df['Revenue'],
                        name='Revenue',
                        marker_color='#667eea',
                        opacity=0.7
                    ))
                    
                    # Net Income as line overlay
                    if 'NetIncome' in df.columns:
                        fig_candlestick.add_trace(go.Scatter(
                            x=df['Year'],
                            y=df['NetIncome'],
                            name='Net Income',
                            line=dict(color='#764ba2', width=3),
                            yaxis='y2'
                        ))
                    
                    fig_candlestick.update_layout(
                        title=f"{ticker.upper()} Financial Candlestick View",
                        yaxis2=dict(overlaying="y", side="right"),
                        template=chart_theme,
                        height=500
                    )
                    
                    st.plotly_chart(fig_candlestick, use_container_width=True)
                
                elif viz_type == "Area Chart":
                    # Stacked area chart
                    fig_area = go.Figure()
                    
                    if 'Revenue' in df.columns:
                        fig_area.add_trace(go.Scatter(
                            x=df['Year'],
                            y=df['Revenue'],
                            fill='tonexty',
                            name='Revenue',
                            line=dict(color='#667eea')
                        ))
                    
                    if 'NetIncome' in df.columns:
                        fig_area.add_trace(go.Scatter(
                            x=df['Year'],
                            y=df['NetIncome'],
                            fill='tonexty',
                            name='Net Income',
                            line=dict(color='#764ba2')
                        ))
                    
                    fig_area.update_layout(
                        title=f"{ticker.upper()} Financial Metrics Area Chart",
                        template=chart_theme,
                        height=500
                    )
                    
                    st.plotly_chart(fig_area, use_container_width=True)
                
                elif viz_type == "Scatter Plot":
                    # Scatter plot of Revenue vs Net Income
                    if 'Revenue' in df.columns and 'NetIncome' in df.columns:
                        fig_scatter = px.scatter(
                            df,
                            x='Revenue',
                            y='NetIncome',
                            size='Revenue',
                            color='Year',
                            title=f"{ticker.upper()} Revenue vs Net Income Scatter",
                            template=chart_theme,
                            size_max=20
                        )
                        
                        fig_scatter.update_layout(height=500)
                        st.plotly_chart(fig_scatter, use_container_width=True)
                        
                        # Add trend line
                        if len(df) > 2:
                            z = np.polyfit(df['Revenue'], df['NetIncome'], 1)
                            p = np.poly1d(z)
                            
                            fig_scatter.add_trace(go.Scatter(
                                x=df['Revenue'],
                                y=p(df['Revenue']),
                                mode='lines',
                                name='Trend Line',
                                line=dict(color='red', dash='dash')
                            ))
                            st.plotly_chart(fig_scatter, use_container_width=True)
                
                elif viz_type == "Box Plot":
                    # Box plot for distribution analysis
                    if len(df) > 3:
                        fig_box = go.Figure()
                        
                        for col in ['Revenue', 'NetIncome']:
                            if col in df.columns:
                                fig_box.add_trace(go.Box(
                                    y=df[col],
                                    name=col,
                                    boxpoints='outliers'
                                ))
                        
                        fig_box.update_layout(
                            title=f"{ticker.upper()} Financial Metrics Distribution",
                            template=chart_theme,
                            height=500
                        )
                        
                        st.plotly_chart(fig_box, use_container_width=True)
                
                elif viz_type == "Correlation Heatmap":
                    # Correlation matrix heatmap
                    if len(df.columns) > 2:
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 1:
                            correlation_matrix = df[numeric_cols].corr()
                            
                            fig_heatmap = px.imshow(
                                correlation_matrix,
                                title=f"{ticker.upper()} Financial Metrics Correlation",
                                template=chart_theme,
                                color_continuous_scale='RdBu',
                                aspect="auto"
                            )
                            
                            fig_heatmap.update_layout(height=500)
                            st.plotly_chart(fig_heatmap, use_container_width=True)
                            
                            # Correlation insights
                            st.subheader("🔍 Correlation Insights")
                            high_corr = []
                            for i in range(len(correlation_matrix.columns)):
                                for j in range(i+1, len(correlation_matrix.columns)):
                                    corr_val = correlation_matrix.iloc[i, j]
                                    if abs(corr_val) > 0.7:
                                        high_corr.append({
                                            'Metric 1': correlation_matrix.columns[i],
                                            'Metric 2': correlation_matrix.columns[j],
                                            'Correlation': corr_val
                                        })
                            
                            if high_corr:
                                st.write("**Strong Correlations (>0.7):**")
                                for corr in high_corr:
                                    st.write(f"• {corr['Metric 1']} ↔ {corr['Metric 2']}: {corr['Correlation']:.3f}")
                            else:
                                st.info("No strong correlations found between metrics")
            
            with tab2:
                if len(df) > 1:
                    # Calculate comprehensive growth rates
                    revenue_growth_rates = []
                    net_income_growth_rates = []
                    years = []
                    
                    for i in range(1, len(df)):
                        rev_growth = ((df['Revenue'].iloc[i] - df['Revenue'].iloc[i-1]) / df['Revenue'].iloc[i-1]) * 100
                        ni_growth = ((df['NetIncome'].iloc[i] - df['NetIncome'].iloc[i-1]) / df['NetIncome'].iloc[i-1]) * 100
                        
                        revenue_growth_rates.append(rev_growth)
                        net_income_growth_rates.append(ni_growth)
                        years.append(df['Year'].iloc[i])
                    
                    growth_df = pd.DataFrame({
                        'Year': years,
                        'Revenue Growth (%)': revenue_growth_rates,
                        'Net Income Growth (%)': net_income_growth_rates
                    })
                    
                    # Enhanced growth chart
                    fig = px.bar(growth_df, x='Year', y=['Revenue Growth (%)', 'Net Income Growth (%)'],
                                title=f"{ticker.upper()} Growth Rates Analysis",
                                template=chart_theme,
                                barmode='group',
                                color_discrete_map={'Revenue Growth (%)': '#667eea', 'Net Income Growth (%)': '#764ba2'})
                    
                    fig.update_layout(
                        height=500,
                        hovermode='x unified',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Performance heatmap
                    st.subheader("🔥 Performance Heatmap")
                    
                    # Create performance matrix
                    performance_metrics = []
                    for i, year in enumerate(years):
                        year_data = df[df['Year'] == year].iloc[0]
                        prev_year_data = df[df['Year'] == years[i-1]].iloc[0] if i > 0 else None
                        
                        metrics = {
                            'Year': year,
                            'Revenue Growth': revenue_growth_rates[i] if i < len(revenue_growth_rates) else 0,
                            'Net Income Growth': net_income_growth_rates[i] if i < len(net_income_growth_rates) else 0,
                            'Profit Margin': (year_data['NetIncome'] / year_data['Revenue'] * 100) if 'NetIncome' in year_data and 'Revenue' in year_data else 0
                        }
                        
                        if prev_year_data is not None:
                            metrics['Efficiency'] = (year_data['Revenue'] / year_data['TotalAssets']) if 'TotalAssets' in year_data else 0
                        else:
                            metrics['Efficiency'] = 0
                        
                        performance_metrics.append(metrics)
                    
                    performance_df = pd.DataFrame(performance_metrics)
                    
                    # Create heatmap
                    if len(performance_df) > 1:
                        heatmap_data = performance_df.set_index('Year').T
                        
                        fig_heatmap = px.imshow(
                            heatmap_data,
                            title=f"{ticker.upper()} Performance Heatmap",
                            template=chart_theme,
                            aspect="auto",
                            color_continuous_scale="RdYlGn"
                        )
                        
                        fig_heatmap.update_layout(
                            height=400,
                            xaxis_title="Year",
                            yaxis_title="Metrics"
                        )
                        
                        st.plotly_chart(fig_heatmap, use_container_width=True)
                        
                        # Performance summary
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            avg_revenue_growth = np.mean(revenue_growth_rates)
                            st.metric("Avg Revenue Growth", f"{avg_revenue_growth:.1f}%")
                        
                        with col2:
                            avg_ni_growth = np.mean(net_income_growth_rates)
                            st.metric("Avg Net Income Growth", f"{avg_ni_growth:.1f}%")
                        
                        with col3:
                            growth_volatility = np.std(revenue_growth_rates)
                            st.metric("Growth Volatility", f"{growth_volatility:.1f}%")
                    
                    # Trend analysis
                    st.subheader("📈 Trend Analysis")
                    
                    if len(growth_df) > 2:
                        # Linear trend for revenue growth
                        x_trend = np.arange(len(growth_df))
                        z_rev = np.polyfit(x_trend, growth_df['Revenue Growth (%)'], 1)
                        p_rev = np.poly1d(z_rev)
                        
                        z_ni = np.polyfit(x_trend, growth_df['Net Income Growth (%)'], 1)
                        p_ni = np.poly1d(z_ni)
                        
                        trend_fig = go.Figure()
                        
                        trend_fig.add_trace(go.Scatter(
                            x=growth_df['Year'],
                            y=growth_df['Revenue Growth (%)'],
                            mode='markers',
                            name='Revenue Growth',
                            marker=dict(color='#667eea', size=10)
                        ))
                        
                        trend_fig.add_trace(go.Scatter(
                            x=growth_df['Year'],
                            y=p_rev(x_trend),
                            mode='lines',
                            name='Revenue Trend',
                            line=dict(color='#667eea', width=2, dash='dash')
                        ))
                        
                        trend_fig.add_trace(go.Scatter(
                            x=growth_df['Year'],
                            y=growth_df['Net Income Growth (%)'],
                            mode='markers',
                            name='Net Income Growth',
                            marker=dict(color='#764ba2', size=10)
                        ))
                        
                        trend_fig.add_trace(go.Scatter(
                            x=growth_df['Year'],
                            y=p_ni(x_trend),
                            mode='lines',
                            name='Net Income Trend',
                            line=dict(color='#764ba2', width=2, dash='dash')
                        ))
                        
                        trend_fig.update_layout(
                            title=f"{ticker.upper()} Growth Trend Analysis",
                            template=chart_theme,
                            height=400,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(trend_fig, use_container_width=True)
                        
                        # Trend interpretation
                        rev_trend_slope = z_rev[0]
                        ni_trend_slope = z_ni[0]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if rev_trend_slope > 0:
                                st.success(f"📈 Revenue growth is trending upward ({rev_trend_slope:.1f}% per year)")
                            else:
                                st.warning(f"📉 Revenue growth is trending downward ({rev_trend_slope:.1f}% per year)")
                        
                        with col2:
                            if ni_trend_slope > 0:
                                st.success(f"📈 Net income growth is trending upward ({ni_trend_slope:.1f}% per year)")
                            else:
                                st.warning(f"📉 Net income growth is trending downward ({ni_trend_slope:.1f}% per year)")
            
            with tab3:
                st.subheader("🏭 Peer Comparison & Industry Analysis")
                
                # Get industry information
                industry = market_data.get('industry', 'Unknown') if market_data else 'Unknown'
                sector = market_data.get('sector', 'Unknown') if market_data else 'Unknown'
                
                if industry != 'Unknown':
                    st.info(f"**Industry:** {industry} | **Sector:** {sector}")
                    
                    # Industry benchmarks from config
                    industry_benchmarks = {
                        'Technology': {'avg_pe': 28.5, 'avg_growth': 15.2, 'avg_margin': 18.5},
                        'Healthcare': {'avg_pe': 22.1, 'avg_growth': 8.5, 'avg_margin': 12.3},
                        'Finance': {'avg_pe': 15.8, 'avg_growth': 6.2, 'avg_margin': 25.1},
                        'Energy': {'avg_pe': 12.3, 'avg_growth': 4.1, 'avg_margin': 8.7},
                        'Consumer Goods': {'avg_pe': 18.9, 'avg_growth': 9.8, 'avg_margin': 15.2}
                    }
                    
                    # Find closest industry match
                    closest_industry = None
                    for ind, _ in industry_benchmarks.items():
                        if ind.lower() in industry.lower() or industry.lower() in ind.lower():
                            closest_industry = ind
                            break
                    
                    if closest_industry:
                        benchmarks = industry_benchmarks[closest_industry]
                        
                        # Company vs Industry comparison
                        st.subheader(f"📊 {ticker.upper()} vs {closest_industry} Industry")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            company_pe = real_time_data.get('pe_ratio', 0)
                            if company_pe > 0:
                                pe_status = "Above" if company_pe > benchmarks['avg_pe'] else "Below"
                                pe_color = "red" if company_pe > benchmarks['avg_pe'] * 1.5 else "orange" if company_pe > benchmarks['avg_pe'] else "green"
                                st.metric("P/E Ratio", f"{company_pe:.1f}", f"{pe_status} Industry Avg")
                                st.markdown(f"Industry Avg: **{benchmarks['avg_pe']:.1f}**")
                            else:
                                st.metric("P/E Ratio", "N/A")
                        
                        with col2:
                            if len(df) > 1 and 'Revenue' in df.columns:
                                company_growth = ((df['Revenue'].iloc[0] - df['Revenue'].iloc[1]) / df['Revenue'].iloc[1]) * 100
                                growth_status = "Above" if company_growth > benchmarks['avg_growth'] else "Below"
                                growth_color = "green" if company_growth > benchmarks['avg_growth'] else "orange"
                                st.metric("Revenue Growth", f"{company_growth:.1f}%", f"{growth_status} Industry Avg")
                                st.markdown(f"Industry Avg: **{benchmarks['avg_growth']:.1f}%**")
                            else:
                                st.metric("Revenue Growth", "N/A")
                        
                        with col3:
                            if 'Revenue' in df.columns and 'NetIncome' in df.columns:
                                company_margin = (df['NetIncome'].iloc[0] / df['Revenue'].iloc[0]) * 100
                                margin_status = "Above" if company_margin > benchmarks['avg_margin'] else "Below"
                                margin_color = "green" if company_margin > benchmarks['avg_margin'] else "orange"
                                st.metric("Profit Margin", f"{company_margin:.1f}%", f"{margin_status} Industry Avg")
                                st.markdown(f"Industry Avg: **{benchmarks['avg_margin']:.1f}%**")
                            else:
                                st.metric("Profit Margin", "N/A")
                        
                        # Industry comparison chart
                        comparison_data = {
                            'Metric': ['P/E Ratio', 'Revenue Growth (%)', 'Profit Margin (%)'],
                            ticker.upper(): [
                                company_pe if company_pe > 0 else 0,
                                company_growth if 'company_growth' in locals() else 0,
                                company_margin if 'company_margin' in locals() else 0
                            ],
                            f'{closest_industry} Avg': [
                                benchmarks['avg_pe'],
                                benchmarks['avg_growth'],
                                benchmarks['avg_margin']
                            ]
                        }
                        
                        comparison_df = pd.DataFrame(comparison_data)
                        
                        fig_comparison = go.Figure()
                        
                        fig_comparison.add_trace(go.Bar(
                            name=ticker.upper(),
                            x=comparison_df['Metric'],
                            y=comparison_df[ticker.upper()],
                            marker_color='#667eea'
                        ))
                        
                        fig_comparison.add_trace(go.Bar(
                            name=f'{closest_industry} Avg',
                            x=comparison_df['Metric'],
                            y=comparison_df[f'{closest_industry} Avg'],
                            marker_color='#764ba2'
                        ))
                        
                        fig_comparison.update_layout(
                            title=f"{ticker.upper()} vs Industry Benchmarks",
                            template=chart_theme,
                            barmode='group',
                            height=400
                        )
                        
                        st.plotly_chart(fig_comparison, use_container_width=True)
                        
                        # Industry insights
                        st.subheader("💡 Industry Insights")
                        
                        insights = []
                        if company_pe > 0:
                            if company_pe > benchmarks['avg_pe'] * 1.5:
                                insights.append("⚠️ **High Valuation:** P/E ratio significantly above industry average")
                            elif company_pe < benchmarks['avg_pe'] * 0.7:
                                insights.append("💡 **Value Opportunity:** P/E ratio below industry average")
                        
                        if 'company_growth' in locals():
                            if company_growth > benchmarks['avg_growth'] * 1.5:
                                insights.append("🚀 **Growth Leader:** Revenue growth significantly above industry average")
                            elif company_growth < benchmarks['avg_growth'] * 0.7:
                                insights.append("📉 **Growth Lagging:** Revenue growth below industry average")
                        
                        if 'company_margin' in locals():
                            if company_margin > benchmarks['avg_margin'] * 1.3:
                                insights.append("💰 **High Profitability:** Profit margin above industry average")
                            elif company_margin < benchmarks['avg_margin'] * 0.7:
                                insights.append("🔍 **Margin Pressure:** Profit margin below industry average")
                        
                        for insight in insights:
                            st.write(insight)
                        
                        if not insights:
                            st.success("🎯 Company performance is in line with industry averages")
                    
                    else:
                        st.warning("Industry benchmarks not available for this sector")
                else:
                    st.info("Industry information not available for comparison")
                
                # Peer comparison placeholder
                st.subheader("👥 Peer Company Comparison")
                st.info("Peer comparison data will be loaded from financial APIs")
                
                # Mock peer data with enhanced metrics
                peers_data = {
                    'Company': [ticker.upper(), 'PEER1', 'PEER2', 'PEER3'],
                    'Market Cap ($M)': [1000, 800, 1200, 900],
                    'P/E Ratio': [25.5, 22.1, 28.3, 24.7],
                    'Revenue Growth (%)': [15.2, 12.8, 18.5, 14.1],
                    'Profit Margin (%)': [18.5, 15.2, 22.1, 16.8],
                    'Debt/Equity': [0.45, 0.52, 0.38, 0.61]
                }
                
                peers_df = pd.DataFrame(peers_data)
                
                # Color code the peer comparison
                def color_peer_metrics(val, col_name):
                    if col_name == 'P/E Ratio':
                        return 'background-color: #ffebee' if val > 30 else 'background-color: #e8f5e8' if val < 20 else ''
                    elif col_name == 'Revenue Growth (%)':
                        return 'background-color: #e8f5e8' if val > 15 else 'background-color: #fff3e0' if val > 10 else 'background-color: #ffebee'
                    elif col_name == 'Profit Margin (%)':
                        return 'background-color: #e8f5e8' if val > 20 else 'background-color: #fff3e0' if val > 15 else 'background-color: #ffebee'
                    elif col_name == 'Debt/Equity':
                        return 'background-color: #ffebee' if val > 0.6 else 'background-color: #e8f5e8' if val < 0.4 else 'background-color: #fff3e0'
                    return ''
                
                styled_peers = peers_df.style.apply(lambda x: [color_peer_metrics(v, x.name) for v in x], axis=0)
                st.dataframe(styled_peers, use_container_width=True)
                
                # Peer analysis summary
                if ticker in peers_df['Company'].values:
                    company_row = peers_df[peers_df['Company'] == ticker].iloc[0]
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        pe_rank = peers_df['P/E Ratio'].rank(ascending=False).iloc[0]
                        st.metric("P/E Ranking", f"{pe_rank:.0f}/{len(peers_df)}")
                    
                    with col2:
                        growth_rank = peers_df['Revenue Growth (%)'].rank(ascending=False).iloc[0]
                        st.metric("Growth Ranking", f"{growth_rank:.0f}/{len(peers_df)}")
                    
                    with col3:
                        margin_rank = peers_df['Profit Margin (%)'].rank(ascending=False).iloc[0]
                        st.metric("Margin Ranking", f"{margin_rank:.0f}/{len(peers_df)}")
            
            with tab4:
                st.subheader("📋 Financial Data Table")
                
                # Data filtering and controls
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Year filter
                    if 'Year' in df.columns:
                        selected_years = st.multiselect(
                            "Filter by Year:",
                            options=sorted(df['Year'].unique(), reverse=True),
                            default=sorted(df['Year'].unique(), reverse=True)[:3] if len(df) > 3 else sorted(df['Year'].unique(), reverse=True)
                        )
                    else:
                        selected_years = []
                
                with col2:
                    # Metric filter
                    available_metrics = [col for col in df.columns if col != 'Year']
                    selected_metrics = st.multiselect(
                        "Select Metrics:",
                        options=available_metrics,
                        default=available_metrics[:5] if len(available_metrics) > 5 else available_metrics
                    )
                
                with col3:
                    # Sort options
                    sort_by = st.selectbox(
                        "Sort by:",
                        options=['Year'] + selected_metrics if selected_metrics else ['Year'],
                        index=0
                    )
                    sort_order = st.selectbox("Order:", ["Descending", "Ascending"])
                
                # Apply filters and sorting
                if selected_years and 'Year' in df.columns:
                    filtered_df = df[df['Year'].isin(selected_years)]
                else:
                    filtered_df = df.copy()
                
                if selected_metrics:
                    display_columns = ['Year'] + selected_metrics
                    filtered_df = filtered_df[display_columns]
                
                # Apply sorting
                if sort_by in filtered_df.columns:
                    ascending = sort_order == "Ascending"
                    if sort_by == 'Year':
                        filtered_df = filtered_df.sort_values(by=sort_by, ascending=ascending)
                    else:
                        filtered_df = filtered_df.sort_values(by=sort_by, ascending=ascending)
                
                # Display filtered data
                st.write(f"**Showing {len(filtered_df)} records**")
                
                # Enhanced dataframe display
                if not filtered_df.empty:
                    # Format numeric columns
                    numeric_columns = filtered_df.select_dtypes(include=[np.number]).columns
                    formatted_df = filtered_df.copy()
                    
                    for col in numeric_columns:
                        if col != 'Year':
                            # Format large numbers with M/B suffixes
                            if filtered_df[col].max() > 1000000:
                                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x/1000000:.1f}M" if pd.notna(x) else "N/A")
                            elif filtered_df[col].max() > 1000:
                                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x/1000:.1f}K" if pd.notna(x) else "N/A")
                            else:
                                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
                    
                    # Color code the data
                    def color_dataframe(val, col_name):
                        if col_name == 'Year':
                            return 'background-color: #f0f0f0'
                        elif col_name in numeric_columns and col_name != 'Year':
                            try:
                                # Convert back to numeric for comparison
                                if 'M' in str(val):
                                    num_val = float(str(val).replace('M', '')) * 1000000
                                elif 'K' in str(val):
                                    num_val = float(str(val).replace('K', '')) * 1000
                                else:
                                    num_val = float(val)
                                
                                # Color based on value ranges
                                if 'Revenue' in col_name or 'NetIncome' in col_name:
                                    if num_val > 0:
                                        return 'background-color: #e8f5e8'
                                    else:
                                        return 'background-color: #ffebee'
                                elif 'Growth' in col_name:
                                    if num_val > 10:
                                        return 'background-color: #e8f5e8'
                                    elif num_val > 0:
                                        return 'background-color: #fff3e0'
                                    else:
                                        return 'background-color: #ffebee'
                            except:
                                pass
                        return ''
                    
                    styled_df = formatted_df.style.apply(lambda x: [color_dataframe(v, x.name) for v in x], axis=0)
                    st.dataframe(styled_df, use_container_width=True)
                    
                    # Data summary statistics
                    st.subheader("📊 Data Summary")
                    
                    if numeric_columns:
                        summary_stats = filtered_df[numeric_columns].describe()
                        st.write("**Statistical Summary:**")
                        st.dataframe(summary_stats, use_container_width=True)
                        
                        # Data quality indicators
                        st.write("**Data Quality:**")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            total_cells = len(filtered_df) * len(filtered_df.columns)
                            missing_cells = filtered_df.isnull().sum().sum()
                            completeness = (total_cells - missing_cells) / total_cells * 100
                            st.metric("Data Completeness", f"{completeness:.1f}%")
                        
                        with col2:
                            if len(filtered_df) > 1:
                                data_consistency = 1 - (filtered_df.std() / filtered_df.mean()).mean()
                                st.metric("Data Consistency", f"{data_consistency:.1f}%")
                            else:
                                st.metric("Data Consistency", "N/A")
                        
                        with col3:
                            if len(filtered_df) > 1:
                                data_range = (filtered_df.max() - filtered_df.min()).mean()
                                st.metric("Data Range", f"{data_range:.0f}")
                            else:
                                st.metric("Data Range", "N/A")
                else:
                    st.warning("No data available with current filters")
                
                # Export options for filtered data
                if not filtered_df.empty:
                    st.subheader("📥 Export Filtered Data")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        csv_data = filtered_df.to_csv(index=False)
                        st.download_button(
                            label="📊 Export as CSV",
                            data=csv_data,
                            file_name=f"{ticker}_filtered_data_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        # JSON export
                        json_data = filtered_df.to_json(orient='records', indent=2)
                        st.download_button(
                            label="📄 Export as JSON",
                            data=json_data,
                            file_name=f"{ticker}_filtered_data_{datetime.now().strftime('%Y%m%d')}.json",
                            mime="application/json"
                        )
            
            with tab5:
                st.subheader("🎯 Financial Criteria Analysis")
                
                if st.session_state.use_custom_criteria and st.session_state.custom_criteria:
                    st.info("📊 Using Custom Criteria")
                    criteria_to_analyze = st.session_state.custom_criteria
                else:
                    st.info("📊 Using Predefined Criteria")
                    criteria_to_analyze = []
                    for category, metrics in PREDEFINED_CRITERIA.items():
                        for metric_name, metric_data in metrics.items():
                            criteria_to_analyze.append({
                                'name': metric_name,
                                'formula': metric_data['formula'],
                                'description': metric_data['description'],
                                'threshold': metric_data['threshold'],
                                'category': category
                            })
                
                if criteria_to_analyze:
                    # Create analysis results
                    analysis_results = []
                    
                    for criteria in criteria_to_analyze:
                        value = calculate_financial_metric(criteria['formula'], df)
                        status = get_metric_status(value, criteria['threshold'])
                        
                        analysis_results.append({
                            'Metric': criteria['name'],
                            'Category': criteria.get('category', 'Custom'),
                            'Current Value': f"{value:.2f}" if value is not None else "N/A",
                            'Target Threshold': f"{criteria['threshold']:.2f}",
                            'Status': status,
                            'Description': criteria['description']
                        })
                    
                    # Display results in a nice table
                    results_df = pd.DataFrame(analysis_results)
                    
                    # Color code the status column
                    def color_status(val):
                        if '✅' in str(val):
                            return 'background-color: #d4edda; color: #155724;'
                        elif '🟡' in str(val):
                            return 'background-color: #fff3cd; color: #856404;'
                        elif '❌' in str(val):
                            return 'background-color: #f8d7da; color: #721c24;'
                        else:
                            return 'background-color: #e2e3e5; color: #383d41;'
                    
                    styled_df = results_df.style.applymap(color_status, subset=['Status'])
                    st.dataframe(styled_df, use_container_width=True)
                    
                    # Summary statistics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        good_count = len([r for r in analysis_results if '✅' in r['Status']])
                        st.metric("✅ Good", good_count)
                    
                    with col2:
                        acceptable_count = len([r for r in analysis_results if '🟡' in r['Status']])
                        st.metric("🟡 Acceptable", acceptable_count)
                    
                    with col3:
                        poor_count = len([r for r in analysis_results if '❌' in r['Status']])
                        st.metric("❌ Poor", poor_count)
                    
                    with col4:
                        total_metrics = len(analysis_results)
                        overall_score = (good_count + acceptable_count * 0.5) / total_metrics * 100
                        st.metric("Overall Score", f"{overall_score:.1f}%")
                    
                    # Recommendations
                    st.subheader("💡 Recommendations")
                    
                    poor_metrics = [r for r in analysis_results if '❌' in r['Status']]
                    if poor_metrics:
                        st.warning("**Areas for Improvement:**")
                        for metric in poor_metrics[:3]:  # Show top 3
                            st.write(f"• **{metric['Metric']}**: {metric['Description']}")
                            st.write(f"  Current: {metric['Current Value']}, Target: {metric['Target Threshold']}")
                    else:
                        st.success("🎉 All metrics are performing well!")
                        
                else:
                    st.info("No criteria available for analysis. Add custom criteria in the sidebar or use predefined criteria.")
            
            # DCF Valuation Tab
            with tab6:
                st.subheader("💰 DCF (Discounted Cash Flow) Valuation")
                
                if len(df) > 0:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        discount_rate = st.slider("Discount Rate (%)", 5.0, 20.0, 10.0, 0.5) / 100
                        growth_rate = st.slider("Growth Rate (%)", 0.0, 10.0, 3.0, 0.5) / 100
                    
                    with col2:
                        if st.button("Calculate DCF"):
                            dcf_results = calculate_dcf_valuation(df, discount_rate, growth_rate)
                            
                            if dcf_results.get('dcf_value'):
                                st.success(f"**DCF Value: ${dcf_results['dcf_value']:,.0f}M**")
                                
                                # Display DCF breakdown
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Free Cash Flow", f"${dcf_results['fcf']:,.0f}M")
                                    st.metric("Terminal Value", f"${dcf_results['terminal_value']:,.0f}M")
                                
                                with col2:
                                    st.metric("PV of FCF (5 years)", f"${dcf_results['pv_fcf']:,.0f}M")
                                    st.metric("PV of Terminal Value", f"${dcf_results['pv_terminal']:,.0f}M")
                                
                                # Assumptions
                                st.subheader("📋 Assumptions")
                                st.write(f"**Discount Rate:** {discount_rate*100:.1f}%")
                                st.write(f"**Growth Rate:** {growth_rate*100:.1f}%")
                                st.write("**Forecast Period:** 5 years")
                                st.write("**Terminal Value:** Perpetuity growth model")
                            else:
                                st.error(f"DCF calculation failed: {dcf_results.get('error', 'Unknown error')}")
                else:
                    st.info("No financial data available for DCF calculation.")
            
            # Risk Analysis Tab
            with tab7:
                st.subheader("⚠️ Risk Analysis")
                
                if len(df) > 1:
                    risk_metrics = calculate_risk_metrics(df)
                    
                    if risk_metrics and 'error' not in risk_metrics:
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            volatility = risk_metrics.get('revenue_volatility')
                            if volatility:
                                st.metric("Revenue Volatility", f"{volatility:.2%}")
                            else:
                                st.metric("Revenue Volatility", "N/A")
                        
                        with col2:
                            income_vol = risk_metrics.get('income_volatility')
                            if income_vol:
                                st.metric("Income Volatility", f"{income_vol:.2%}")
                            else:
                                st.metric("Income Volatility", "N/A")
                        
                        with col3:
                            debt_coverage = risk_metrics.get('debt_coverage_ratio')
                            if debt_coverage:
                                st.metric("Debt Coverage", f"{debt_coverage:.2f}")
                            else:
                                st.metric("Debt Coverage", "N/A")
                        
                        with col4:
                            risk_level = risk_metrics.get('risk_level', 'Unknown')
                            color_map = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
                            color = color_map.get(risk_level, 'gray')
                            st.markdown(f"**Risk Level:** <span style='color: {color}'>{risk_level}</span>", unsafe_allow_html=True)
                        
                        # Risk visualization
                        if risk_metrics.get('revenue_volatility') and risk_metrics.get('income_volatility'):
                            risk_data = {
                                'Metric': ['Revenue Volatility', 'Income Volatility'],
                                'Value': [risk_metrics['revenue_volatility'], risk_metrics['income_volatility']]
                            }
                            
                            fig = px.bar(
                                pd.DataFrame(risk_data), 
                                x='Metric', 
                                y='Value',
                                title='Risk Metrics Comparison',
                                template=chart_theme
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(f"Risk analysis failed: {risk_metrics.get('error', 'Unknown error')}")
                else:
                    st.info("Insufficient data for risk analysis (need at least 2 years of data).")
            
            # Sensitivity Analysis Tab
            with tab8:
                st.subheader("📊 Sensitivity Analysis")
                
                if len(df) > 0:
                    # Get base values for sensitivity analysis
                    base_revenue = df['Revenue'].iloc[0] if 'Revenue' in df.columns else 1000
                    base_net_income = df['NetIncome'].iloc[0] if 'NetIncome' in df.columns else 100
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Revenue Sensitivity")
                        revenue_range = st.slider("Revenue Range (%)", -50, 100, (-20, 20))
                        
                        if st.button("Analyze Revenue Sensitivity"):
                            variables = {'Revenue': base_revenue}
                            ranges = {'Revenue': (base_revenue * (1 + revenue_range[0]/100), base_revenue * (1 + revenue_range[1]/100))}
                            
                            sensitivity_results = perform_sensitivity_analysis(base_revenue, variables, ranges)
                            
                            if not sensitivity_results.empty and 'error' not in sensitivity_results.columns:
                                fig = px.line(
                                    sensitivity_results, 
                                    x='Value', 
                                    y='Impact',
                                    title='Revenue Sensitivity Analysis',
                                    template=chart_theme
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                st.dataframe(sensitivity_results, use_container_width=True)
                            else:
                                st.error("Sensitivity analysis failed.")
                    
                    with col2:
                        st.subheader("Net Income Sensitivity")
                        ni_range = st.slider("Net Income Range (%)", -50, 100, (-20, 20))
                        
                        if st.button("Analyze Net Income Sensitivity"):
                            variables = {'NetIncome': base_net_income}
                            ranges = {'NetIncome': (base_net_income * (1 + ni_range[0]/100), base_net_income * (1 + ni_range[1]/100))}
                            
                            sensitivity_results = perform_sensitivity_analysis(base_net_income, variables, ranges)
                            
                            if not sensitivity_results.empty and 'error' not in sensitivity_results.columns:
                                fig = px.line(
                                    sensitivity_results, 
                                    x='Value', 
                                    y='Impact',
                                    title='Net Income Sensitivity Analysis',
                                    template=chart_theme
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                st.dataframe(sensitivity_results, use_container_width=True)
                            else:
                                st.error("Sensitivity analysis failed.")
                else:
                    st.info("No financial data available for sensitivity analysis.")
            
            # ML Analysis Tab
            with tab9:
                st.subheader("🤖 Machine Learning Analysis")
                
                if len(df) > 2:
                    # ML Analysis Controls
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        forecast_periods = st.slider("Forecast Periods", 1, 5, 3)
                        anomaly_threshold = st.slider("Anomaly Detection Threshold", 1.5, 3.0, 2.0, 0.1)
                    
                    with col2:
                        if st.button("🔮 Generate ML Predictions"):
                            # Financial predictions
                            predictions = predict_financial_metrics(df, forecast_periods)
                            
                            if predictions and 'error' not in predictions:
                                st.success(f"✅ Generated predictions for {len(predictions)} metrics")
                                
                                # Display predictions
                                st.subheader("📈 Financial Predictions")
                                
                                for metric, pred_data in predictions.items():
                                    with st.expander(f"🔮 {metric} Predictions"):
                                        col1, col2, col3 = st.columns(3)
                                        
                                        with col1:
                                            st.metric("Trend Slope", f"{pred_data['trend_slope']:.3f}")
                                        
                                        with col2:
                                            st.metric("Confidence", pred_data['confidence'])
                                        
                                        with col3:
                                            st.metric("Latest Value", f"${df[metric].iloc[0]:,.0f}M" if metric in df.columns else "N/A")
                                        
                                        # Prediction chart
                                        if len(pred_data['future_values']) > 0:
                                            years = list(df['Year']) + [f"Year {i+1}" for i in range(1, forecast_periods + 1)]
                                            values = list(df[metric].iloc[:3]) + pred_data['future_values']
                                            
                                            fig_pred = go.Figure()
                                            
                                            # Historical data
                                            fig_pred.add_trace(go.Scatter(
                                                x=years[:len(df)],
                                                y=values[:len(df)],
                                                mode='lines+markers',
                                                name=f'Historical {metric}',
                                                line=dict(color='#667eea', width=3)
                                            ))
                                            
                                            # Predictions
                                            fig_pred.add_trace(go.Scatter(
                                                x=years[len(df)-1:],
                                                y=values[len(df)-1:],
                                                mode='lines+markers',
                                                name=f'Predicted {metric}',
                                                line=dict(color='#764ba2', width=3, dash='dash')
                                            ))
                                            
                                            fig_pred.update_layout(
                                                title=f"{metric} - Historical vs Predicted",
                                                template=chart_theme,
                                                height=400
                                            )
                                            
                                            st.plotly_chart(fig_pred, use_container_width=True)
                                            
                                            # Prediction insights
                                            if pred_data['trend_slope'] > 0:
                                                st.success(f"📈 {metric} is predicted to grow at {pred_data['trend_slope']:.1%} per period")
                                            else:
                                                st.warning(f"📉 {metric} is predicted to decline at {pred_data['trend_slope']:.1%} per period")
                            else:
                                st.error(f"Prediction failed: {predictions.get('error', 'Unknown error')}")
                        
                        if st.button("🔍 Detect Anomalies"):
                            # Anomaly detection
                            anomalies = detect_anomalies(df, anomaly_threshold)
                            
                            if anomalies and 'error' not in anomalies:
                                st.success(f"🔍 Detected anomalies in {len(anomalies)} metrics")
                                
                                # Display anomalies
                                st.subheader("⚠️ Detected Anomalies")
                                
                                for metric, anomaly_data in anomalies.items():
                                    with st.expander(f"⚠️ {metric} Anomalies"):
                                        col1, col2, col3 = st.columns(3)
                                        
                                        with col1:
                                            st.metric("Anomaly Count", len(anomaly_data['anomaly_years']))
                                        
                                        with col2:
                                            st.metric("Severity", anomaly_data['severity'])
                                        
                                        with col3:
                                            max_z_score = max(anomaly_data['z_scores'])
                                            st.metric("Max Z-Score", f"{max_z_score:.2f}")
                                        
                                        # Anomaly details
                                        anomaly_df = pd.DataFrame({
                                            'Year': anomaly_data['anomaly_years'],
                                            'Value': anomaly_data['anomaly_values'],
                                            'Z-Score': anomaly_data['z_scores']
                                        })
                                        
                                        st.dataframe(anomaly_df, use_container_width=True)
                                        
                                        # Anomaly insights
                                        if anomaly_data['severity'] == 'High':
                                            st.warning(f"🚨 High-severity anomalies detected in {metric}")
                                        else:
                                            st.info(f"⚠️ Medium-severity anomalies detected in {metric}")
                            else:
                                st.info("No anomalies detected or anomaly detection failed")
                        
                        if st.button("🎯 Calculate Risk Score"):
                            # Risk assessment
                            risk_assessment = calculate_risk_score(df)
                            
                            if risk_assessment and 'error' not in risk_assessment:
                                st.success("🎯 Risk assessment completed")
                                
                                # Display risk score
                                st.subheader("🎯 Risk Assessment")
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    risk_color = risk_assessment['risk_color']
                                    st.markdown(f"**Risk Level:** <span style='color: {risk_color}'>{risk_assessment['risk_level']}</span>", unsafe_allow_html=True)
                                
                                with col2:
                                    st.metric("Risk Score", f"{risk_assessment['total_risk_score']:.1f}/{risk_assessment['max_possible_score']}")
                                
                                with col3:
                                    st.metric("Risk Percentage", f"{risk_assessment['risk_percentage']:.1f}%")
                                
                                # Risk breakdown
                                st.subheader("📊 Risk Factor Breakdown")
                                
                                risk_factors_df = pd.DataFrame([
                                    {'Risk Factor': factor, 'Score': score}
                                    for factor, score in risk_assessment['risk_factors'].items()
                                ])
                                
                                # Color code risk factors
                                def color_risk_factors(val, col_name):
                                    if col_name == 'Score':
                                        if val < 10:
                                            return 'background-color: #e8f5e8'
                                        elif val < 20:
                                            return 'background-color: #fff3e0'
                                        else:
                                            return 'background-color: #ffebee'
                                    return ''
                                
                                styled_risk = risk_factors_df.style.apply(lambda x: [color_risk_factors(v, x.name) for v in x], axis=0)
                                st.dataframe(styled_risk, use_container_width=True)
                                
                                # Risk insights
                                st.subheader("💡 Risk Insights")
                                
                                high_risk_factors = [factor for factor, score in risk_assessment['risk_factors'].items() if score > 20]
                                if high_risk_factors:
                                    st.warning(f"🚨 High-risk factors: {', '.join(high_risk_factors)}")
                                
                                low_risk_factors = [factor for factor, score in risk_assessment['risk_factors'].items() if score < 10]
                                if low_risk_factors:
                                    st.success(f"✅ Low-risk factors: {', '.join(low_risk_factors)}")
                                
                                # Risk recommendations
                                if risk_assessment['risk_level'] == 'High':
                                    st.error("⚠️ **High Risk Company** - Consider detailed due diligence before investment")
                                elif risk_assessment['risk_level'] == 'Medium':
                                    st.warning("🟡 **Medium Risk Company** - Monitor key risk factors closely")
                                else:
                                    st.success("🟢 **Low Risk Company** - Generally stable financial profile")
                            else:
                                st.error(f"Risk assessment failed: {risk_assessment.get('error', 'Unknown error')}")
                    
                    # ML Summary Dashboard
                    if 'predictions' in locals() and 'anomalies' in locals() and 'risk_assessment' in locals():
                        st.subheader("📊 ML Analysis Summary")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if 'predictions' in locals() and predictions and 'error' not in predictions:
                                st.metric("Metrics Predicted", len(predictions))
                            else:
                                st.metric("Metrics Predicted", 0)
                        
                        with col2:
                            if 'anomalies' in locals() and anomalies and 'error' not in anomalies:
                                total_anomalies = sum(len(anom['anomaly_years']) for anom in anomalies.values())
                                st.metric("Total Anomalies", total_anomalies)
                            else:
                                st.metric("Total Anomalies", 0)
                        
                        with col3:
                            if 'risk_assessment' in locals() and risk_assessment and 'error' not in risk_assessment:
                                st.metric("Risk Level", risk_assessment['risk_level'])
                            else:
                                st.metric("Risk Level", "N/A")
                        
                        # Export ML results
                        if st.button("📥 Export ML Analysis"):
                            ml_results = {
                                'predictions': predictions if 'predictions' in locals() else {},
                                'anomalies': anomalies if 'anomalies' in locals() else {},
                                'risk_assessment': risk_assessment if 'risk_assessment' in locals() else {},
                                'timestamp': datetime.now().isoformat(),
                                'ticker': ticker
                            }
                            
                            json_data = str(ml_results)
                            st.download_button(
                                label="📄 Download ML Analysis (JSON)",
                                data=json_data,
                                file_name=f"{ticker}_ml_analysis_{datetime.now().strftime('%Y%m%d')}.json",
                                mime="application/json"
                            )
                else:
                    st.info("Insufficient data for ML analysis (need at least 3 years of data)")
                
        else:
            st.error("No financial data found for this ticker.")
            
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.info("Make sure the FastAPI backend is running and API_BASE_URL is configured")

# Market Overview Section
if st.session_state.get('show_market_overview', False):
    st.markdown("## 🌍 Market Overview")
    
    # Real-time market data
    try:
        # Fetch major indices (using yfinance)
        major_indices = ['^GSPC', '^IXIC', '^DJI', '^RUT']  # S&P 500, NASDAQ, DOW, Russell 2000
        indices_data = []
        
        for index_symbol in major_indices:
            try:
                index_data = yf.Ticker(index_symbol)
                hist = index_data.history(period="2d")
                if not hist.empty:
                    current_value = hist['Close'].iloc[-1]
                    prev_value = hist['Close'].iloc[-2]
                    change = current_value - prev_value
                    change_pct = (change / prev_value) * 100
                    
                    indices_data.append({
                        'Index': index_symbol.replace('^', ''),
                        'Value': current_value,
                        'Change': change,
                        'Change %': change_pct
                    })
            except:
                continue
        
        if indices_data:
            indices_df = pd.DataFrame(indices_data)
            
            # Market sentiment
            positive_changes = len([x for x in indices_data if x['Change'] > 0])
            negative_changes = len([x for x in indices_data if x['Change'] < 0])
            
            if positive_changes > negative_changes:
                market_sentiment = "🟢 Bullish"
                sentiment_color = "green"
            elif negative_changes > positive_changes:
                market_sentiment = "🔴 Bearish"
                sentiment_color = "red"
            else:
                market_sentiment = "🟡 Neutral"
                sentiment_color = "orange"
            
            st.subheader(f"📊 Market Sentiment: {market_sentiment}")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📈 Major Indices")
                
                # Color code the indices
                def color_indices(val, col_name):
                    if col_name == 'Change %':
                        return 'background-color: #e8f5e8' if val > 0 else 'background-color: #ffebee' if val < 0 else ''
                    elif col_name == 'Change':
                        return 'background-color: #e8f5e8' if val > 0 else 'background-color: #ffebee' if val < 0 else ''
                    return ''
                
                styled_indices = indices_df.style.apply(lambda x: [color_indices(v, x.name) for v in x], axis=0)
                st.dataframe(styled_indices, use_container_width=True)
                
                # Market performance chart
                if len(indices_data) > 1:
                    fig_indices = go.Figure()
                    
                    for _, row in indices_df.iterrows():
                        fig_indices.add_trace(go.Bar(
                            name=row['Index'],
                            x=['Value', 'Change %'],
                            y=[row['Value'], row['Change %']],
                            text=[f"${row['Value']:,.2f}", f"{row['Change %']:+.2f}%"],
                            textposition='auto'
                        ))
                    
                    fig_indices.update_layout(
                        title="Market Indices Performance",
                        template=chart_theme,
                        barmode='group',
                        height=400
                    )
                    
                    st.plotly_chart(fig_indices, use_container_width=True)
            
            with col2:
                st.subheader("📊 Market Summary")
                
                # Overall market metrics
                total_change = sum([abs(x['Change %']) for x in indices_data])
                avg_change = total_change / len(indices_data)
                
                st.metric("Average Change", f"{avg_change:.2f}%")
                st.metric("Positive Indices", f"{positive_changes}/{len(indices_data)}")
                st.metric("Negative Indices", f"{negative_changes}/{len(indices_data)}")
                
                # Market volatility indicator
                volatility = np.std([x['Change %'] for x in indices_data])
                if volatility < 1:
                    vol_level = "Low"
                    vol_color = "green"
                elif volatility < 3:
                    vol_level = "Medium"
                    vol_color = "orange"
                else:
                    vol_level = "High"
                    vol_color = "red"
                
                st.markdown(f"**Volatility Level:** <span style='color: {vol_color}'>{vol_level}</span>", unsafe_allow_html=True)
                st.markdown(f"**Volatility:** {volatility:.2f}%")
        
        # Trending stocks section
        st.subheader("🔥 Trending Stocks")
        
        trending_tickers = ['TSLA', 'NVDA', 'META', 'GOOGL', 'AAPL', 'MSFT']
        trending_data = []
        
        for ticker in trending_tickers:
            try:
                stock_data = fetch_real_time_data(ticker)
                if stock_data and stock_data.get('current_price'):
                    trending_data.append({
                        'Ticker': ticker,
                        'Price': stock_data['current_price'],
                        'Change %': stock_data.get('price_change_pct', 0),
                        'Volume': stock_data.get('volume', 0),
                        'Market Cap': stock_data.get('market_cap', 0)
                    })
            except:
                continue
        
        if trending_data:
            trending_df = pd.DataFrame(trending_data)
            
            # Sort by absolute change percentage
            trending_df['Abs Change %'] = trending_df['Change %'].abs()
            trending_df = trending_df.sort_values('Abs Change %', ascending=False)
            
            # Color code trending stocks
            def color_trending(val, col_name):
                if col_name == 'Change %':
                    return 'background-color: #e8f5e8' if val > 0 else 'background-color: #ffebee' if val < 0 else ''
                return ''
            
            styled_trending = trending_df.style.apply(lambda x: [color_trending(v, x.name) for v in x], axis=0)
            st.dataframe(styled_trending[['Ticker', 'Price', 'Change %', 'Volume', 'Market Cap']], use_container_width=True)
            
            # Trending stocks chart
            fig_trending = px.bar(
                trending_df, 
                x='Ticker', 
                y='Change %',
                color='Change %',
                color_continuous_scale='RdYlGn',
                title="Stock Performance Today",
                template=chart_theme
            )
            
            fig_trending.update_layout(height=400)
            st.plotly_chart(fig_trending, use_container_width=True)
        
        # Market insights
        st.subheader("💡 Market Insights")
        
        insights = []
        
        # Add market insights based on data
        if 'indices_data' in locals() and indices_data:
            if positive_changes > negative_changes:
                insights.append("📈 **Market Momentum:** Major indices showing positive momentum")
            else:
                insights.append("📉 **Market Pressure:** Major indices under selling pressure")
            
            if volatility > 2:
                insights.append("⚠️ **High Volatility:** Market experiencing increased volatility")
            elif volatility < 0.5:
                insights.append("🟢 **Low Volatility:** Market showing stability")
        
        if 'trending_data' in locals() and trending_data:
            tech_performance = [x for x in trending_data if x['Ticker'] in ['TSLA', 'NVDA', 'META', 'GOOGL', 'AAPL', 'MSFT']]
            if tech_performance:
                tech_avg = np.mean([x['Change %'] for x in tech_performance])
                if tech_avg > 2:
                    insights.append("🚀 **Tech Rally:** Technology stocks leading market gains")
                elif tech_avg < -2:
                    insights.append("💻 **Tech Selloff:** Technology stocks under pressure")
        
        for insight in insights:
            st.write(insight)
        
        if not insights:
            st.info("Market conditions appear stable with mixed performance across sectors")
            
    except Exception as e:
        st.error(f"Error fetching market data: {e}")
        st.info("Showing sample market data")
        
        # Fallback to sample data
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Major Indices (Sample)")
            indices_data = {
                'Index': ['S&P 500', 'NASDAQ', 'DOW', 'Russell 2000'],
                'Value': [4500.25, 14250.75, 35250.50, 1850.25],
                'Change': [0.85, 1.25, -0.45, 0.65]
            }
            
            indices_df = pd.DataFrame(indices_data)
            st.dataframe(indices_df, use_container_width=True)
        
        with col2:
            st.subheader("🔥 Trending Stocks (Sample)")
            trending_data = {
                'Ticker': ['TSLA', 'NVDA', 'META', 'GOOGL'],
                'Price': [250.75, 450.25, 320.50, 140.25],
                'Change %': [5.25, 3.75, -1.25, 2.50]
            }
            
            trending_df = pd.DataFrame(trending_data)
            st.dataframe(trending_df, use_container_width=True)

# Industry Analysis Section
if st.session_state.get('show_industry_analysis', False):
    st.markdown("## 🏭 Industry Analysis")
    
    # Mock industry data
    industries = ['Technology', 'Healthcare', 'Finance', 'Energy', 'Consumer Goods']
    avg_pe = [28.5, 22.1, 15.8, 12.3, 18.9]
    avg_growth = [15.2, 8.5, 6.2, 4.1, 9.8]
    
    industry_df = pd.DataFrame({
        'Industry': industries,
        'Average P/E': avg_pe,
        'Average Growth (%)': avg_growth
    })
    
    fig = px.scatter(industry_df, x='Average P/E', y='Average Growth (%)', 
                     text='Industry', title='Industry P/E vs Growth Analysis',
                     template=chart_theme)
    
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)

# Export and Download Section
st.markdown("---")
st.subheader("📥 Export & Download")

col1, col2, col3 = st.columns(3)

with col1:
    if ticker and 'df' in locals():
        # Export financial data
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="📊 Export Financial Data (CSV)",
            data=csv_data,
            file_name=f"{ticker}_financial_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

with col2:
    if ticker and 'analysis_results' in locals():
        # Export analysis results
        analysis_csv = pd.DataFrame(analysis_results).to_csv(index=False)
        st.download_button(
            label="📈 Export Analysis Results (CSV)",
            data=analysis_csv,
            file_name=f"{ticker}_analysis_results_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

with col3:
    if ticker and 'real_time_data' in locals():
        # Export real-time data
        rt_data = pd.DataFrame([real_time_data])
        rt_csv = rt_data.to_csv(index=False)
        st.download_button(
            label="⏰ Export Real-time Data (CSV)",
            data=rt_csv,
            file_name=f"{ticker}_realtime_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Financial Analyzer Pro - Built with Streamlit and FastAPI</p>
    <p>Data for educational purposes only. Not financial advice.</p>
    <p>Last updated: {}</p>
</div>
""".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)
