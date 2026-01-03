import pytest
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.trade_signal_generator import TradeSignalGenerator

def test_initialization():
    generator = TradeSignalGenerator()
    assert generator.sentiment_threshold == 0.5
    assert generator.signals == []

def test_bullish_score_computation(sample_stock_data):
    generator = TradeSignalGenerator()
    df = sample_stock_data['TEST.NS']
    
    # Manually tweak data to trigger bullish score
    # Last row close > SMA_20 (which is true in linear space)
    
    score, reasons = generator._compute_bullish_score(df)
    
    # Since we set up linear growth, Close should be > SMA_20
    # SMA_20 at index 49 is mean of 30..49
    # Close at 49 is 152. Mean of 20 points ending at 152 is < 152.
    # So Price > SMA_20.
    
    assert score > 0
    assert "Price above SMA_20." in reasons

def test_signal_generation(sample_stock_data, sample_news_data):
    generator = TradeSignalGenerator()
    
    # The linear data is bullish
    # News is positive
    
    signals = generator.generate_btst_signals(sample_stock_data, sample_news_data, side="CALL")
    
    assert len(signals) >= 1
    assert signals[0]['symbol'] == 'TEST.NS'
    assert signals[0]['action'] == 'BUY'
