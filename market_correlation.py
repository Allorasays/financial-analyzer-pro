"""
Market Correlation and Beta Calculations
Calculates correlation with S&P 500 and beta
"""
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_sp500_data(period: str = "1y") -> Optional[pd.DataFrame]:
    """
    Fetch S&P 500 historical data
    
    Args:
        period: Time period (e.g., "1y", "2y")
    
    Returns:
        DataFrame with S&P 500 OHLCV data
    """
    try:
        sp500 = yf.Ticker("^GSPC")
        hist = sp500.history(period=period)
        if not hist.empty:
            return hist
        return None
    except Exception as e:
        logger.error(f"Error fetching S&P 500 data: {e}")
        return None


def calculate_beta(stock_returns: pd.Series, market_returns: pd.Series) -> float:
    """
    Calculate Beta (sensitivity to market movements)
    
    Beta = Covariance(stock_returns, market_returns) / Variance(market_returns)
    
    Beta > 1: Stock moves more than market (volatile)
    Beta = 1: Stock moves with market
    Beta < 1: Stock moves less than market (stable)
    Beta < 0: Stock moves opposite to market (rare)
    
    Args:
        stock_returns: Series of stock daily returns
        market_returns: Series of market (S&P 500) daily returns
    
    Returns:
        Beta value
    """
    # Align dates
    aligned = pd.concat([stock_returns, market_returns], axis=1).dropna()
    if len(aligned) < 20:
        return np.nan
    
    stock_ret = aligned.iloc[:, 0]
    market_ret = aligned.iloc[:, 1]
    
    # Calculate covariance and variance
    covariance = np.cov(stock_ret, market_ret)[0][1]
    market_variance = np.var(market_ret)
    
    if market_variance == 0:
        return np.nan
    
    beta = covariance / market_variance
    return beta


def calculate_correlation(stock_returns: pd.Series, market_returns: pd.Series) -> float:
    """
    Calculate correlation coefficient with S&P 500
    
    Correlation ranges from -1 to 1:
    -1: Perfect negative correlation
    0: No correlation
    1: Perfect positive correlation
    
    Args:
        stock_returns: Series of stock daily returns
        market_returns: Series of market (S&P 500) daily returns
    
    Returns:
        Correlation coefficient (-1 to 1)
    """
    # Align dates
    aligned = pd.concat([stock_returns, market_returns], axis=1).dropna()
    if len(aligned) < 20:
        return np.nan
    
    stock_ret = aligned.iloc[:, 0]
    market_ret = aligned.iloc[:, 1]
    
    correlation = stock_ret.corr(market_ret)
    return correlation


def calculate_market_metrics(ticker: str, hist: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate market correlation metrics (Beta, Correlation with S&P 500)
    
    Args:
        ticker: Stock ticker symbol
        hist: Historical stock data DataFrame
    
    Returns:
        Dictionary with beta, correlation, and other market metrics
    """
    try:
        # Get S&P 500 data for same period
        sp500_data = get_sp500_data("1y")
        
        if sp500_data is None or len(sp500_data) < 20:
            return {
                "beta": None,
                "correlation": None,
                "market_data_available": False
            }
        
        # Calculate daily returns
        stock_returns = hist['Close'].pct_change().dropna()
        market_returns = sp500_data['Close'].pct_change().dropna()
        
        # Calculate beta
        beta = calculate_beta(stock_returns, market_returns)
        
        # Calculate correlation
        correlation = calculate_correlation(stock_returns, market_returns)
        
        # Additional metrics
        stock_volatility = stock_returns.std() * np.sqrt(252)  # Annualized
        market_volatility = market_returns.std() * np.sqrt(252)  # Annualized
        
        # Relative volatility
        relative_volatility = stock_volatility / market_volatility if market_volatility > 0 else None
        
        return {
            "beta": round(beta, 4) if not np.isnan(beta) else None,
            "correlation": round(correlation, 4) if not np.isnan(correlation) else None,
            "stock_volatility": round(stock_volatility, 4),
            "market_volatility": round(market_volatility, 4),
            "relative_volatility": round(relative_volatility, 4) if relative_volatility else None,
            "market_data_available": True,
            "data_points": len(pd.concat([stock_returns, market_returns], axis=1).dropna())
        }
        
    except Exception as e:
        logger.error(f"Error calculating market metrics for {ticker}: {e}")
        return {
            "beta": None,
            "correlation": None,
            "market_data_available": False,
            "error": str(e)
        }


def classify_by_beta(beta: Optional[float]) -> str:
    """
    Classify stock based on beta value
    
    Args:
        beta: Beta value
    
    Returns:
        Classification string
    """
    if beta is None or np.isnan(beta):
        return "Unknown"
    
    if beta > 1.5:
        return "Highly Volatile"
    elif beta > 1.0:
        return "Volatile"
    elif beta > 0.5:
        return "Moderate"
    elif beta > 0:
        return "Stable"
    else:
        return "Inverse"  # Negative beta (rare)

