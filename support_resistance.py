"""
Support and Resistance Level Detection
Identifies key price levels where reversals are likely
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.signal import argrelextrema

def find_support_resistance_levels(df: pd.DataFrame, window: int = 10, lookback: int = 50) -> Dict:
    """
    Find support (local minima) and resistance (local maxima) levels
    
    Args:
        df: DataFrame with 'High', 'Low', 'Close' columns
        window: Window size for local extrema detection
        lookback: Number of periods to look back for levels
    
    Returns:
        Dictionary with support and resistance levels and metrics
    """
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    
    # Find local minima (support levels)
    support_indices = argrelextrema(low, np.less, order=window)[0]
    support_levels = low[support_indices]
    
    # Find local maxima (resistance levels)
    resistance_indices = argrelextrema(high, np.greater, order=window)[0]
    resistance_levels = high[resistance_indices]
    
    # Get recent levels (within lookback period)
    recent_support = support_levels[support_indices >= len(close) - lookback] if len(support_indices) > 0 else np.array([])
    recent_resistance = resistance_levels[resistance_indices >= len(close) - lookback] if len(resistance_indices) > 0 else np.array([])
    
    # Current price
    current_price = close[-1]
    
    # Find nearest support and resistance
    if len(support_levels) > 0:
        valid_support = support_levels[support_levels < current_price]
        nearest_support = valid_support[-1] if len(valid_support) > 0 else np.nan
        distance_to_support = ((current_price - nearest_support) / current_price * 100) if not np.isnan(nearest_support) else np.nan
    else:
        nearest_support = np.nan
        distance_to_support = np.nan
    
    if len(resistance_levels) > 0:
        valid_resistance = resistance_levels[resistance_levels > current_price]
        nearest_resistance = valid_resistance[0] if len(valid_resistance) > 0 else np.nan
        distance_to_resistance = ((nearest_resistance - current_price) / current_price * 100) if not np.isnan(nearest_resistance) else np.nan
    else:
        nearest_resistance = np.nan
        distance_to_resistance = np.nan
    
    # Count touches of nearest levels (strength indicator)
    support_touches = np.sum(np.abs(low - nearest_support) < (nearest_support * 0.01)) if not np.isnan(nearest_support) else 0
    resistance_touches = np.sum(np.abs(high - nearest_resistance) < (nearest_resistance * 0.01)) if not np.isnan(nearest_resistance) else 0
    
    # Price position between support and resistance
    if not np.isnan(nearest_support) and not np.isnan(nearest_resistance):
        price_position = (current_price - nearest_support) / (nearest_resistance - nearest_support)
    else:
        price_position = np.nan
    
    return {
        'nearest_support': nearest_support,
        'nearest_resistance': nearest_resistance,
        'distance_to_support_pct': distance_to_support,
        'distance_to_resistance_pct': distance_to_resistance,
        'support_touches': support_touches,
        'resistance_touches': resistance_touches,
        'price_position': price_position,  # 0 = at support, 1 = at resistance
        'support_strength': min(support_touches / 5, 1.0) if support_touches > 0 else 0,  # Normalize to 0-1
        'resistance_strength': min(resistance_touches / 5, 1.0) if resistance_touches > 0 else 0
    }


def calculate_pivot_points(df: pd.DataFrame) -> pd.Series:
    """
    Calculate classic pivot points (Support/Resistance)
    
    Pivot Point = (High + Low + Close) / 3
    Resistance 1 = 2 * Pivot - Low
    Resistance 2 = Pivot + (High - Low)
    Support 1 = 2 * Pivot - High
    Support 2 = Pivot - (High - Low)
    
    Args:
        df: DataFrame with 'High', 'Low', 'Close' columns
    
    Returns:
        Series with pivot point values
    """
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    pivot = (high + low + close) / 3
    return pivot


def add_support_resistance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add support and resistance features to DataFrame
    
    Args:
        df: DataFrame with OHLC data
    
    Returns:
        DataFrame with support/resistance features added
    """
    df = df.copy()
    
    try:
        # Calculate pivot points
        df['Pivot_Point'] = calculate_pivot_points(df)
        
        # Find support/resistance levels
        sr_levels = find_support_resistance_levels(df, window=10, lookback=50)
        
        # Add as constant features (same for all rows)
        df['Nearest_Support'] = sr_levels.get('nearest_support', np.nan)
        df['Nearest_Resistance'] = sr_levels.get('nearest_resistance', np.nan)
        df['Distance_to_Support_Pct'] = sr_levels.get('distance_to_support_pct', np.nan)
        df['Distance_to_Resistance_Pct'] = sr_levels.get('distance_to_resistance_pct', np.nan)
        df['Support_Touches'] = sr_levels.get('support_touches', 0)
        df['Resistance_Touches'] = sr_levels.get('resistance_touches', 0)
        df['Price_Position_SR'] = sr_levels.get('price_position', np.nan)
        df['Support_Strength'] = sr_levels.get('support_strength', 0)
        df['Resistance_Strength'] = sr_levels.get('resistance_strength', 0)
        
        # Relative position to pivot
        df['Distance_from_Pivot_Pct'] = ((df['Close'] - df['Pivot_Point']) / df['Pivot_Point']) * 100
        
    except Exception as e:
        print(f"Error calculating support/resistance: {e}")
        # Add NaN placeholders if calculation fails
        df['Nearest_Support'] = np.nan
        df['Nearest_Resistance'] = np.nan
        df['Distance_to_Support_Pct'] = np.nan
        df['Distance_to_Resistance_Pct'] = np.nan
        df['Support_Touches'] = 0
        df['Resistance_Touches'] = 0
        df['Price_Position_SR'] = np.nan
        df['Support_Strength'] = 0
        df['Resistance_Strength'] = 0
        df['Pivot_Point'] = np.nan
        df['Distance_from_Pivot_Pct'] = np.nan
    
    return df

