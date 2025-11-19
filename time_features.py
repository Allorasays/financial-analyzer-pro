"""
Time-Based Feature Engineering
Extracts temporal patterns (day of week, month, earnings proximity, etc.)
"""
import pandas as pd
import numpy as np
from typing import Dict
from datetime import datetime

def add_time_features(df: pd.DataFrame, ticker: str = None) -> pd.DataFrame:
    """
    Add time-based features to DataFrame
    
    Args:
        df: DataFrame with datetime index
        ticker: Stock ticker (for earnings date lookup)
    
    Returns:
        DataFrame with time features added
    """
    df = df.copy()
    
    try:
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'Date' in df.columns:
                df.set_index('Date', inplace=True)
            else:
                df.index = pd.to_datetime(df.index)
        
        # Day of week (0 = Monday, 6 = Sunday)
        df['Day_of_Week'] = df.index.dayofweek
        
        # Monday effect (1 if Monday, 0 otherwise)
        df['Is_Monday'] = (df.index.dayofweek == 0).astype(int)
        
        # Friday effect (1 if Friday, 0 otherwise)
        df['Is_Friday'] = (df.index.dayofweek == 4).astype(int)
        
        # Month (1-12)
        df['Month'] = df.index.month
        
        # January effect (1 if January, 0 otherwise)
        df['Is_January'] = (df.index.month == 1).astype(int)
        
        # December effect (tax-loss selling)
        df['Is_December'] = (df.index.month == 12).astype(int)
        
        # Quarter (1-4)
        df['Quarter'] = df.index.quarter
        
        # Q1 effect (often strong)
        df['Is_Q1'] = (df.index.quarter == 1).astype(int)
        
        # Q4 effect (year-end effects)
        df['Is_Q4'] = (df.index.quarter == 4).astype(int)
        
        # Day of month (1-31)
        df['Day_of_Month'] = df.index.day
        
        # Month-end effect (last 3 days of month)
        df['Is_Month_End'] = (df.index.day >= 28).astype(int)
        
        # Week of year (1-52)
        try:
            df['Week_of_Year'] = df.index.isocalendar().week
        except:
            # Fallback for older pandas
            df['Week_of_Year'] = df.index.strftime('%U').astype(int)
        
        # Days since start of period (trend over time)
        df['Days_Since_Start'] = (df.index - df.index.min()).days
        
        # Cyclical encoding for day of week (better for ML)
        df['Day_of_Week_Sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['Day_of_Week_Cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        
        # Cyclical encoding for month
        df['Month_Sin'] = np.sin(2 * np.pi * df.index.month / 12)
        df['Month_Cos'] = np.cos(2 * np.pi * df.index.month / 12)
        
        # Earnings proximity (approximate - would need actual earnings dates for accuracy)
        # For now, we'll use quarterly patterns
        try:
            quarter_starts = pd.to_datetime([f"{y}-{max(1, q*3-2)}-01" for y, q in zip(df.index.year, df.index.quarter)])
            days_since_quarter_start = []
            for idx, q_start in zip(df.index, quarter_starts):
                if idx >= q_start:
                    days_since_quarter_start.append((idx - q_start).days)
                else:
                    days_since_quarter_start.append(0)
            df['Days_Since_Quarter_Start'] = days_since_quarter_start
        except:
            # Simplified calculation
            df['Days_Since_Quarter_Start'] = df.index.day
        
        # Earnings season indicator (companies typically report 1-2 months after quarter end)
        # Approximate: if within 45 days of quarter start, likely earnings season
        df['Near_Earnings_Season'] = (df['Days_Since_Quarter_Start'] <= 45).astype(int)
        
        # Holiday proximity (weekend before/after major holidays)
        # Approximate with month-end and year-end effects
        df['Holiday_Proximity'] = (
            ((df.index.month == 12) & (df.index.day >= 20)) |  # Near Christmas/New Year
            ((df.index.month == 1) & (df.index.day <= 5)) |    # After New Year
            ((df.index.month == 7) & (df.index.day >= 1) & (df.index.day <= 5))  # Near July 4
        ).astype(int)
        
        # Market regime (expansion/recession) - would need economic data, placeholder for now
        df['Market_Regime'] = 1  # Default to expansion (1), would update with FRED data
        
    except Exception as e:
        print(f"Error adding time features: {e}")
        # Add basic placeholders
        df['Day_of_Week'] = 0
        df['Month'] = 1
        df['Quarter'] = 1
    
    return df

