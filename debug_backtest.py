import yfinance as yf
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'src'))

# Mock logger to avoid errors
from unittest.mock import MagicMock
sys.modules['src.logger'] = MagicMock()

from backtester import BacktestEngine
from indicator_calculator import calculate_moving_averages, find_swing_points, detect_breakouts_breakdowns

def debug_run():
    symbol = "RELIANCE.NS"
    print(f"Fetching data for {symbol}...")
    df = yf.download(symbol, period="1y", interval="1d", progress=False)
    
    if df.empty:
        print("Error: No data fetched.")
        return

    print(f"Data fetched: {len(df)} rows.")

    # Manually run indicator steps to inspect
    df = calculate_moving_averages(df)
    df = find_swing_points(df)
    df = detect_breakouts_breakdowns(df)
    
    breakouts = df['Breakout'].sum()
    print(f"Total Breakouts Detected: {breakouts}")
    
    if breakouts > 0:
        print("Sample Breakout Rows:")
        print(df[df['Breakout'] == 1][['Close', 'Resistance', 'SMA_20', 'SMA_50']].tail())
    else:
        print("No breakouts found. Checking Resistance values...")
        print(df[['High', 'Resistance']].tail(20))

    # Run Engine
    engine = BacktestEngine(initial_capital=100000)
    metrics = engine.run_backtest(df)
    print("\nBacktest Metrics:")
    print(metrics)
    print(f"Total Trades in Log: {len(engine.trade_log)}")

if __name__ == "__main__":
    debug_run()
