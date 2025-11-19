"""
Drawdown and Risk Metrics
Calculates maximum drawdown, recovery metrics, and risk-adjusted returns
"""
import pandas as pd
import numpy as np
from typing import Dict

def calculate_drawdown(returns: pd.Series) -> pd.Series:
    """
    Calculate running drawdown from peak
    
    Args:
        returns: Series of returns
    
    Returns:
        Series of drawdown values (negative = below peak)
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown


def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    Calculate maximum drawdown (largest peak-to-trough decline)
    
    Args:
        returns: Series of returns
    
    Returns:
        Maximum drawdown as negative percentage
    """
    drawdown = calculate_drawdown(returns)
    return drawdown.min()


def calculate_drawdown_duration(returns: pd.Series) -> Dict:
    """
    Calculate drawdown duration metrics
    
    Args:
        returns: Series of returns
    
    Returns:
        Dictionary with drawdown duration metrics
    """
    drawdown = calculate_drawdown(returns)
    
    # Find periods in drawdown
    in_drawdown = drawdown < 0
    
    if not in_drawdown.any():
        return {
            'current_drawdown_days': 0,
            'max_drawdown_duration': 0,
            'avg_drawdown_duration': 0
        }
    
    # Current drawdown duration
    current_drawdown_days = 0
    for i in range(len(in_drawdown) - 1, -1, -1):
        if in_drawdown.iloc[i]:
            current_drawdown_days += 1
        else:
            break
    
    # Find all drawdown periods
    drawdown_periods = []
    in_period = False
    period_start = None
    
    for i, is_dd in enumerate(in_drawdown):
        if is_dd and not in_period:
            in_period = True
            period_start = i
        elif not is_dd and in_period:
            in_period = False
            drawdown_periods.append(i - period_start)
    
    # If still in drawdown at end
    if in_period:
        drawdown_periods.append(len(in_drawdown) - period_start)
    
    max_duration = max(drawdown_periods) if drawdown_periods else 0
    avg_duration = np.mean(drawdown_periods) if drawdown_periods else 0
    
    return {
        'current_drawdown_days': current_drawdown_days,
        'max_drawdown_duration': max_duration,
        'avg_drawdown_duration': avg_duration
    }


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sharpe Ratio (risk-adjusted returns)
    
    Sharpe = (Mean Return - Risk Free Rate) / Std Dev of Returns
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate (default 2%)
    
    Returns:
        Sharpe ratio
    """
    if len(returns) < 2:
        return np.nan
    
    mean_return = returns.mean()
    std_return = returns.std()
    
    if std_return == 0:
        return np.nan
    
    # Annualize (assuming daily returns)
    annualized_return = mean_return * 252
    annualized_std = std_return * np.sqrt(252)
    
    sharpe = (annualized_return - risk_free_rate) / annualized_std
    return sharpe


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sortino Ratio (downside risk-adjusted returns)
    
    Sortino = (Mean Return - Risk Free Rate) / Downside Deviation
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate (default 2%)
    
    Returns:
        Sortino ratio
    """
    if len(returns) < 2:
        return np.nan
    
    mean_return = returns.mean()
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) == 0:
        return np.nan
    
    downside_std = downside_returns.std()
    
    if downside_std == 0:
        return np.nan
    
    # Annualize
    annualized_return = mean_return * 252
    annualized_downside_std = downside_std * np.sqrt(252)
    
    sortino = (annualized_return - risk_free_rate) / annualized_downside_std
    return sortino


def calculate_recovery_metrics(returns: pd.Series) -> Dict:
    """
    Calculate recovery metrics after drawdowns
    
    Args:
        returns: Series of returns
    
    Returns:
        Dictionary with recovery metrics
    """
    drawdown = calculate_drawdown(returns)
    
    # Find recovery rate (how fast stock recovers from drawdowns)
    recovery_periods = []
    
    for i in range(1, len(drawdown)):
        if drawdown.iloc[i-1] < 0 and drawdown.iloc[i] >= 0:
            # Recovery happened, count days to recovery
            days_to_recover = 0
            for j in range(i-1, -1, -1):
                if drawdown.iloc[j] >= 0:
                    break
                days_to_recover += 1
            if days_to_recover > 0:
                recovery_periods.append(days_to_recover)
    
    avg_recovery_days = np.mean(recovery_periods) if recovery_periods else np.nan
    max_recovery_days = max(recovery_periods) if recovery_periods else np.nan
    
    return {
        'avg_recovery_days': avg_recovery_days,
        'max_recovery_days': max_recovery_days,
        'recovery_count': len(recovery_periods)
    }


def add_drawdown_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add drawdown and risk metrics to DataFrame
    
    Args:
        df: DataFrame with 'Close' column
    
    Returns:
        DataFrame with drawdown/risk features added
    """
    df = df.copy()
    
    try:
        # Calculate returns if not present
        if 'Returns' not in df.columns:
            df['Returns'] = df['Close'].pct_change()
        
        returns = df['Returns'].dropna()
        
        if len(returns) < 20:
            # Not enough data, add NaN placeholders
            df['Max_Drawdown'] = np.nan
            df['Current_Drawdown'] = np.nan
            df['Drawdown_Duration'] = np.nan
            df['Sharpe_Ratio'] = np.nan
            df['Sortino_Ratio'] = np.nan
            df['Avg_Recovery_Days'] = np.nan
            return df
        
        # Calculate drawdown
        drawdown = calculate_drawdown(returns)
        df.loc[returns.index, 'Drawdown'] = drawdown.values
        
        # Maximum drawdown (rolling 252-day window for recent max DD)
        rolling_returns = returns.tail(252) if len(returns) > 252 else returns
        max_drawdown = calculate_max_drawdown(rolling_returns)
        df['Max_Drawdown'] = max_drawdown
        
        # Current drawdown (latest value)
        df['Current_Drawdown'] = drawdown.iloc[-1] if len(drawdown) > 0 else np.nan
        
        # Drawdown duration metrics
        dd_duration = calculate_drawdown_duration(returns)
        df['Drawdown_Duration'] = dd_duration['current_drawdown_days']
        df['Max_Drawdown_Duration'] = dd_duration['max_drawdown_duration']
        df['Avg_Drawdown_Duration'] = dd_duration['avg_drawdown_duration']
        
        # Risk-adjusted returns
        sharpe = calculate_sharpe_ratio(returns)
        sortino = calculate_sortino_ratio(returns)
        df['Sharpe_Ratio'] = sharpe
        df['Sortino_Ratio'] = sortino
        
        # Recovery metrics
        recovery = calculate_recovery_metrics(returns)
        df['Avg_Recovery_Days'] = recovery['avg_recovery_days']
        df['Max_Recovery_Days'] = recovery['max_recovery_days']
        
        # Drawdown magnitude (current vs max)
        df['Drawdown_Magnitude'] = abs(df['Current_Drawdown']) / abs(max_drawdown) if max_drawdown < 0 else np.nan
        
    except Exception as e:
        print(f"Error calculating drawdown metrics: {e}")
        # Add NaN placeholders
        df['Max_Drawdown'] = np.nan
        df['Current_Drawdown'] = np.nan
        df['Drawdown_Duration'] = np.nan
        df['Max_Drawdown_Duration'] = np.nan
        df['Avg_Drawdown_Duration'] = np.nan
        df['Sharpe_Ratio'] = np.nan
        df['Sortino_Ratio'] = np.nan
        df['Avg_Recovery_Days'] = np.nan
        df['Max_Recovery_Days'] = np.nan
        df['Drawdown_Magnitude'] = np.nan
    
    return df

