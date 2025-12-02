import pandas as pd

def calculate_moving_averages(df, periods=[20, 50, 200]):
    """
    Calculates Simple Moving Averages (SMA) for specified periods.
    Adds new columns to the DataFrame for each SMA.
    """
    for period in periods:
        df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
    return df

def calculate_volume_spikes(df, period=20, threshold=1.5):
    """
    Detects volume spikes by comparing current volume against a recent average.
    Adds a new column 'Volume_Spike' (boolean) to the DataFrame.
    """
    df['Volume_SMA'] = df['Volume'].rolling(window=period).mean()
    df['Volume_Spike'] = df['Volume'] > (df['Volume_SMA'] * threshold)
    return df

def detect_breakouts_breakdowns(df):
    """
    Detects breakouts (price closing above resistance) and breakdowns (price closing below support).
    Adds 'Breakout' and 'Breakdown' boolean columns to the DataFrame.
    """
    df['Breakout'] = (df['Close'] > df['Resistance'].shift(1)) & (df['Close'].shift(1) <= df['Resistance'].shift(1))
    df['Breakdown'] = (df['Close'] < df['Support'].shift(1)) & (df['Close'].shift(1) >= df['Support'].shift(1))
    return df

def find_swing_points(df, window=10):
    """
    Identifies swing highs and swing lows as potential support and resistance levels.
    Adds 'Support' and 'Resistance' columns to the DataFrame.
    """
    df['Support'] = df['Low'].rolling(window=window, center=True).min()
    df['Resistance'] = df['High'].rolling(window=window, center=True).max()
    return df

def calculate_atr(df, period=14):
    """
    Calculates Average True Range (ATR).
    Adds a new column 'ATR' to the DataFrame.
    """
    df['High-Low'] = df['High'] - df['Low']
    df['High-PrevClose'] = abs(df['High'] - df['Close'].shift(1))
    df['Low-PrevClose'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=period).mean()
    df = df.drop(columns=['High-Low', 'High-PrevClose', 'Low-PrevClose', 'TR'])
    return df