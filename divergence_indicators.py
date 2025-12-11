"""
Divergence Indicators
Detects divergences between price, volume, and momentum indicators
"""
import pandas as pd
import numpy as np
from typing import Dict

def calculate_price_volume_divergence(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Calculate price vs volume divergence
    
    Bullish divergence: Price making lower lows, volume decreasing (weak selling)
    Bearish divergence: Price making higher highs, volume decreasing (weak buying)
    
    Args:
        df: DataFrame with 'Close' and 'Volume' columns
        period: Lookback period for divergence detection
    
    Returns:
        Series with divergence scores (-1 to 1)
    """
    close = df['Close']
    volume = df['Volume']
    
    # Calculate price and volume trends
    price_trend = close.rolling(window=period).apply(
        lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1
    )
    volume_trend = volume.rolling(window=period).apply(
        lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1
    )
    
    # Divergence: opposite trends = divergence
    divergence = price_trend - volume_trend
    # Normalize to -1 to 1
    divergence_normalized = divergence / 2.0
    
    return divergence_normalized


def calculate_price_rsi_divergence(df: pd.DataFrame, rsi_period: int = 14, lookback: int = 20) -> pd.Series:
    """
    Calculate price vs RSI divergence
    
    Bullish divergence: Price lower low, RSI higher low (momentum improving)
    Bearish divergence: Price higher high, RSI lower high (momentum weakening)
    
    Args:
        df: DataFrame with 'Close' and 'RSI' columns
        rsi_period: RSI period (if RSI not in df)
        lookback: Lookback period for divergence
    
    Returns:
        Series with divergence scores
    """
    close = df['Close']
    
    # Calculate RSI if not present
    if 'RSI' not in df.columns:
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = df['RSI']
    
    # Find local minima and maxima
    price_lows = close.rolling(window=lookback).min()
    price_highs = close.rolling(window=lookback).max()
    rsi_lows = rsi.rolling(window=lookback).min()
    rsi_highs = rsi.rolling(window=lookback).max()
    
    # Calculate divergence
    # Bullish: price making lower lows but RSI making higher lows
    bullish_divergence = ((close < price_lows.shift(lookback)) & 
                          (rsi > rsi_lows.shift(lookback))).astype(float)
    
    # Bearish: price making higher highs but RSI making lower highs
    bearish_divergence = ((close > price_highs.shift(lookback)) & 
                          (rsi < rsi_highs.shift(lookback))).astype(float)
    
    divergence = bullish_divergence - bearish_divergence
    return divergence


def calculate_price_macd_divergence(df: pd.DataFrame) -> pd.Series:
    """
    Calculate price vs MACD divergence
    
    Args:
        df: DataFrame with 'Close' and 'MACD' columns
    
    Returns:
        Series with divergence scores
    """
    if 'MACD' not in df.columns:
        return pd.Series(index=df.index, data=0)
    
    close = df['Close']
    macd = df['MACD']
    
    # Price trend
    price_trend = close.diff().rolling(window=5).mean()
    
    # MACD trend
    macd_trend = macd.diff().rolling(window=5).mean()
    
    # Divergence when trends are opposite
    divergence = np.where(
        (price_trend > 0) & (macd_trend < 0), -1,  # Bearish divergence
        np.where(
            (price_trend < 0) & (macd_trend > 0), 1,  # Bullish divergence
            0  # No divergence
        )
    )
    
    return pd.Series(index=df.index, data=divergence)


def calculate_volume_divergence(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """
    Calculate volume divergence (volume decreasing on moves)
    
    Low volume on up moves = weak buying (bearish)
    Low volume on down moves = weak selling (bullish)
    
    Args:
        df: DataFrame with 'Close' and 'Volume' columns
        period: Lookback period
    
    Returns:
        Series with volume divergence scores
    """
    close = df['Close']
    volume = df['Volume']
    
    # Price change
    price_change = close.pct_change()
    
    # Volume relative to average
    volume_ratio = volume / volume.rolling(window=period).mean()
    
    # Divergence: strong price move with low volume
    divergence = np.where(
        (price_change > 0.01) & (volume_ratio < 0.8), -1,  # Bearish: up move, low volume
        np.where(
            (price_change < -0.01) & (volume_ratio < 0.8), 1,  # Bullish: down move, low volume
            0
        )
    )
    
    return pd.Series(index=df.index, data=divergence)


def add_divergence_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add divergence indicator features to DataFrame
    
    Args:
        df: DataFrame with OHLCV and indicator data
    
    Returns:
        DataFrame with divergence features added
    """
    df = df.copy()
    
    try:
        # Price-Volume Divergence
        df['Price_Volume_Divergence'] = calculate_price_volume_divergence(df, period=20)
        
        # Price-RSI Divergence
        df['Price_RSI_Divergence'] = calculate_price_rsi_divergence(df, lookback=20)
        
        # Price-MACD Divergence
        df['Price_MACD_Divergence'] = calculate_price_macd_divergence(df)
        
        # Volume Divergence
        df['Volume_Divergence'] = calculate_volume_divergence(df, period=10)
        
        # Combined divergence score
        df['Divergence_Score'] = (
            df['Price_Volume_Divergence'].fillna(0) * 0.3 +
            df['Price_RSI_Divergence'].fillna(0) * 0.3 +
            df['Price_MACD_Divergence'].fillna(0) * 0.2 +
            df['Volume_Divergence'].fillna(0) * 0.2
        )
        
        # Divergence strength (absolute value)
        df['Divergence_Strength'] = abs(df['Divergence_Score'])
        
    except Exception as e:
        print(f"Error calculating divergence indicators: {e}")
        # Add placeholder columns
        df['Price_Volume_Divergence'] = 0
        df['Price_RSI_Divergence'] = 0
        df['Price_MACD_Divergence'] = 0
        df['Volume_Divergence'] = 0
        df['Divergence_Score'] = 0
        df['Divergence_Strength'] = 0
    
    return df

