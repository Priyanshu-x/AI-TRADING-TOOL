import pytest
import pandas as pd
import numpy as np
import sys
import os

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.backtester import BacktestEngine

@pytest.fixture
def sample_data():
    dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
    data = {
        'Open': [100, 102, 104, 106, 108, 110, 112, 114, 116, 118],
        'High': [102, 105, 107, 109, 111, 113, 115, 117, 119, 121],
        'Low':  [99, 101, 103, 105, 107, 109, 111, 113, 115, 117],
        'Close': [101, 104, 106, 108, 110, 112, 114, 116, 118, 120],
        'Volume': [1000] * 10
    }
    df = pd.DataFrame(data, index=dates)
    # Add dummy indicators to skip calculation overhead
    df['SMA_20'] = [90] * 10 # Always below price (Uptrend)
    df['SMA_50'] = [80] * 10
    df['Breakout'] = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] # Breakout on day 2
    df['Support'] = [90] * 10
    df['Resistance'] = [105] * 10
    df['ATR'] = 2.0
    return df

def test_initialization():
    engine = BacktestEngine(initial_capital=10000)
    assert engine.initial_capital == 10000
    assert engine.trade_log == []

def test_simple_buy_and_target_exit(sample_data):
    # Strategy: Buy on breakout (Day 2, Close 104). Target 5%.
    # Target Price = 104 * 1.05 = 109.2
    # Day 3 High is 107 (No exit). Day 4 High is 109 (No). Day 5 High is 111 (Exit triggered).
    
    engine = BacktestEngine(initial_capital=100000, commission_pct=0.0)
    metrics = engine.run_backtest(sample_data, {'stop_loss_pct': 0.10, 'target_pct': 0.05})
    
    trade_log = engine.trade_log
    assert len(trade_log) == 2 # Buy and Sell
    
    buy_trade = trade_log[0]
    assert buy_trade['Action'] == 'BUY'
    assert buy_trade['Price'] == 104
    
    sell_trade = trade_log[1]
    assert 'SELL' in sell_trade['Action']
    assert sell_trade['Price'] == 104 * 1.05
    assert sell_trade['PnL'] > 0 # Profit check

def test_stop_loss_exit(sample_data):
    # Strategy: Buy Day 2 (104). SL 1%.
    # SL Price = 104 * 0.99 = 102.96
    # Day 3 Low is 103 (Safe). Day 4 Low is 105. 
    # Let's modify data to trigger SL
    sample_data.loc[sample_data.index[3], 'Low'] = 100 # Drop on Day 4
    
    engine = BacktestEngine(initial_capital=100000, commission_pct=0.0)
    metrics = engine.run_backtest(sample_data, {'stop_loss_pct': 0.01, 'target_pct': 0.50})
    
    sell_trade = engine.trade_log[1]
    assert 'SL' in sell_trade['Action']
    assert sell_trade['PnL'] < 0


