import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_stock_data():
    """
    Creates a sample dataframe mimicking stock data for testing indicators.
    """
    dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
    df = pd.DataFrame({
        'Date': dates,
        'Open': np.linspace(100, 150, 50),
        'High': np.linspace(105, 155, 50),
        'Low': np.linspace(95, 145, 50), # Corrected length to 50
        'Close': np.linspace(102, 152, 50),
        'Volume': np.random.randint(100000, 1000000, 50)
    })
    df.set_index('Date', inplace=True)
    
    # Add dummy indicator columns that are expected by the generator
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean() # Will be NaN mostly but okay
    df['ATR'] = 5.0
    df['Volume_Spike'] = 0
    df['Support'] = 90.0
    df['Resistance'] = 160.0
    df['Breakout'] = 0
    df['Breakdown'] = 0
    
    return {'TEST.NS': df}

@pytest.fixture
def sample_news_data():
    """
    Creates a sample news dataframe.
    """
    return pd.DataFrame([
        {'symbol': 'TEST.NS', 'sentiment': 'positive', 'confidence': 0.9},
        {'symbol': 'TEST.NS', 'sentiment': 'negative', 'confidence': 0.1}, # Weak negative
    ])
