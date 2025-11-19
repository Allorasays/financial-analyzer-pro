"""
Volume-based Technical Indicators
VWAP, OBV, Accumulation/Distribution
"""
import pandas as pd
import numpy as np
from typing import Optional

def calculate_vwap(df: pd.DataFrame, period: int = None) -> pd.Series:
    """
    Calculate Volume Weighted Average Price (VWAP)
    
    VWAP = Sum(Price * Volume) / Sum(Volume)
    
    Args:
        df: DataFrame with 'High', 'Low', 'Close', 'Volume' columns
        period: Rolling period (None = cumulative from start)
    
    Returns:
        Series with VWAP values
    """
    # Typical Price = (High + Low + Close) / 3
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    
    if period is None:
        # Cumulative VWAP from start
        vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    else:
        # Rolling VWAP
        vwap = (typical_price * df['Volume']).rolling(window=period).sum() / \
               df['Volume'].rolling(window=period).sum()
    
    return vwap


def calculate_obv(df: pd.DataFrame) -> pd.Series:
    """
    Calculate On-Balance Volume (OBV)
    
    OBV = Previous OBV + Volume if Close > Previous Close
          Previous OBV - Volume if Close < Previous Close
          Previous OBV if Close == Previous Close
    
    Args:
        df: DataFrame with 'Close' and 'Volume' columns
    
    Returns:
        Series with OBV values
    """
    obv = pd.Series(index=df.index, dtype=float)
    obv.iloc[0] = df['Volume'].iloc[0]
    
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + df['Volume'].iloc[i]
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - df['Volume'].iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv


def calculate_accumulation_distribution(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Accumulation/Distribution Line
    
    A/D = Previous A/D + Money Flow Volume
    Money Flow Volume = Money Flow Multiplier * Volume
    Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
    
    Args:
        df: DataFrame with 'High', 'Low', 'Close', 'Volume' columns
    
    Returns:
        Series with A/D values
    """
    # Calculate Money Flow Multiplier
    high_low_range = df['High'] - df['Low']
    
    # Avoid division by zero
    high_low_range = high_low_range.replace(0, np.nan)
    
    money_flow_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / high_low_range
    
    # Fill NaN values (when High == Low) with 0
    money_flow_multiplier = money_flow_multiplier.fillna(0)
    
    # Calculate Money Flow Volume
    money_flow_volume = money_flow_multiplier * df['Volume']
    
    # Calculate Accumulation/Distribution Line (cumulative)
    ad_line = money_flow_volume.cumsum()
    
    return ad_line


def calculate_volume_roc(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """
    Calculate Volume Rate of Change
    
    Volume ROC = ((Current Volume - Volume N periods ago) / Volume N periods ago) * 100
    
    Args:
        df: DataFrame with 'Volume' column
        period: Number of periods to look back
    
    Returns:
        Series with Volume ROC values
    """
    volume_roc = ((df['Volume'] - df['Volume'].shift(period)) / df['Volume'].shift(period)) * 100
    return volume_roc


def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all volume indicators to DataFrame
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        DataFrame with volume indicators added
    """
    df = df.copy()
    
    try:
        # VWAP (daily)
        df['VWAP'] = calculate_vwap(df)
        df['VWAP_20'] = calculate_vwap(df, period=20)
        
        # OBV
        df['OBV'] = calculate_obv(df)
        # OBV rate of change
        df['OBV_ROC'] = df['OBV'].pct_change() * 100
        
        # Accumulation/Distribution
        df['AD_Line'] = calculate_accumulation_distribution(df)
        # A/D rate of change
        df['AD_ROC'] = df['AD_Line'].pct_change() * 100
        
        # Volume ROC
        df['Volume_ROC'] = calculate_volume_roc(df, period=10)
        
    except Exception as e:
        print(f"Error calculating volume indicators: {e}")
    
    return df

