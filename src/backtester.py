import pandas as pd
import numpy as np
import plotly.graph_objects as go
from .logger import setup_logger
from .indicator_calculator import (
    calculate_moving_averages,
    calculate_atr,
    calculate_volume_spikes,
    find_swing_points,
    detect_breakouts_breakdowns
)

logger = setup_logger(__name__)

class BacktestEngine:
    def __init__(self, initial_capital=100000, commission_pct=0.001):
        """
        Initializes the BacktestEngine.
        
        Args:
            initial_capital (float): Starting portfolio value.
            commission_pct (float): Commission per trade (e.g., 0.001 for 0.1%).
        """
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.trade_log = []
        self.equity_curve = []

    def run_backtest(self, df, strategy_params={}):
        """
        Runs a backtest on the provided DataFrame.
        
        Args:
            df (pd.DataFrame): Historical data with OHLCV.
            strategy_params (dict): Parameters like stop_loss_pct, target_pct.
            
        Returns:
            dict: Performance metrics.
        """
        if df.empty:
            logger.warning("Backtest received empty DataFrame.")
            return {}

        logger.info(f"Starting backtest on {len(df)} records. Params: {strategy_params}")
        
        # Ensure indicators are present
        df = self._ensure_indicators(df)
        
        # Reset state
        self.trade_log = []
        self.equity_curve = []
        capital = self.initial_capital
        position = 0 # 0: Flat, 1: Long, -1: Short (not implemented yet)
        entry_price = 0
        shares = 0
        
        stop_loss_pct = strategy_params.get('stop_loss_pct', 0.02)
        target_pct = strategy_params.get('target_pct', 0.04)
        
        # Iterate through data
        # Note: Iterating rows is slower than vectorization but allows easier state management for complex logic
        for i in range(1, len(df)):
            curr_date = df.index[i]
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            # --- Equity Curve Tracking ---
            # Mark-to-market value
            current_value = capital
            if position == 1:
                current_value = capital + (shares * (row['Close'] - entry_price))
            
            self.equity_curve.append({'Date': curr_date, 'Equity': current_value})
            
            # --- Exit Logic ---
            if position == 1:
                # Check Stop Loss
                if row['Low'] <= entry_price * (1 - stop_loss_pct):
                    exit_price = entry_price * (1 - stop_loss_pct) # Assume filled at SL
                    # Slippage simulation could be added here
                    pnl = (exit_price - entry_price) * shares
                    cost = (exit_price * shares) * self.commission_pct
                    capital += pnl - cost
                    
                    self.trade_log.append({
                        'Date': curr_date,
                        'Action': 'SELL (SL)',
                        'Price': exit_price,
                        'Shares': shares,
                        'PnL': pnl - cost,
                        'Balance': capital
                    })
                    position = 0
                    shares = 0
                    continue # Trade closed, move to next candle
                
                # Check Target
                elif row['High'] >= entry_price * (1 + target_pct):
                    exit_price = entry_price * (1 + target_pct)
                    pnl = (exit_price - entry_price) * shares
                    cost = (exit_price * shares) * self.commission_pct
                    capital += pnl - cost
                    
                    self.trade_log.append({
                        'Date': curr_date,
                        'Action': 'SELL (Target)',
                        'Price': exit_price,
                        'Shares': shares,
                        'PnL': pnl - cost,
                        'Balance': capital
                    })
                    position = 0
                    shares = 0
                    continue

                # EOD Exit (if BTST logic implies default hold duration, otherwise skipped for swing)
                # For this basic implementation, we hold until SL or Target.
            
            # --- Entry Logic ---
            if position == 0:
                # Simple Breakout Strategy Entry (matches Signal Generator logic roughly)
                # Condition: Close > SMA_20 and SMA_20 > SMA_50 (Uptrend) and Breakout detected
                if row['Close'] > row['SMA_20'] and row['SMA_20'] > row['SMA_50'] and row.get('Breakout', 0) == 1:
                    entry_price = row['Close']
                    # Position Sizing: Risk 2% of capital per trade? Or fixed fraction?
                    # Let's use max allocation 20% for now
                    allocation = capital * 0.2
                    shares = int(allocation / entry_price)
                    
                    if shares > 0:
                        cost = (entry_price * shares) * self.commission_pct
                        capital -= cost # Deduct entry commission
                        
                        position = 1
                        self.trade_log.append({
                            'Date': curr_date,
                            'Action': 'BUY',
                            'Price': entry_price,
                            'Shares': shares,
                            'PnL': -cost, # Realized loss is just commission for now
                            'Balance': capital
                        })

        metrics = self.calculate_metrics()
        return metrics

    def _ensure_indicators(self, df):
        """Calculates indicators if not present."""
        if 'SMA_20' not in df.columns: df = calculate_moving_averages(df)
        if 'ATR' not in df.columns: df = calculate_atr(df)
        if 'Volume_Spike' not in df.columns: df = calculate_volume_spikes(df)
        if 'Support' not in df.columns: df = find_swing_points(df)
        if 'Breakout' not in df.columns: df = detect_breakouts_breakdowns(df)
        return df

    def calculate_metrics(self):
        """Calculates Sharpe, Drawdown, Win Rate, etc."""
        if not self.equity_curve:
            return {}
            
        equity_df = pd.DataFrame(self.equity_curve).set_index('Date')
        equity_df['Returns'] = equity_df['Equity'].pct_change().dropna()
        
        total_return = (equity_df['Equity'].iloc[-1] - self.initial_capital) / self.initial_capital
        
        # Annualized Statistics (assuming daily data)
        days = (equity_df.index[-1] - equity_df.index[0]).days
        years = days / 365.25 if days > 0 else 0
        cagr = ((equity_df['Equity'].iloc[-1] / self.initial_capital) ** (1/years)) - 1 if years > 0 else 0
        
        mean_daily_return = equity_df['Returns'].mean()
        std_daily_return = equity_df['Returns'].std()
        
        sharpe_ratio = 0
        if std_daily_return != 0:
            sharpe_ratio = (mean_daily_return / std_daily_return) * np.sqrt(252)
            
        # Drawdown
        equity_df['Peak'] = equity_df['Equity'].cummax()
        equity_df['Drawdown'] = (equity_df['Equity'] - equity_df['Peak']) / equity_df['Peak']
        max_drawdown = equity_df['Drawdown'].min()
        
        # Trade Stats
        trades = pd.DataFrame(self.trade_log)
        win_rate = 0
        if not trades.empty:
            closed_trades = trades[trades['Action'].str.contains('SELL')]
            if not closed_trades.empty:
                winning_trades = closed_trades[closed_trades['PnL'] > 0]
                win_rate = len(winning_trades) / len(closed_trades)

        return {
            "Total Return": f"{total_return:.2%}",
            "CAGR": f"{cagr:.2%}",
            "Sharpe Ratio": f"{sharpe_ratio:.2f}",
            "Max Drawdown": f"{max_drawdown:.2%}",
            "Win Rate": f"{win_rate:.2%}",
            "Final Capital": f"{equity_df['Equity'].iloc[-1]:.2f}",
            "Trades Executed": len(trades) // 2 # approx
        }

    def plot_equity_curve(self):
        """Returns a Plotly figure for the equity curve."""
        if not self.equity_curve:
            return None
            
        equity_df = pd.DataFrame(self.equity_curve)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=equity_df['Date'], y=equity_df['Equity'], mode='lines', name='Portfolio Value'))
        fig.update_layout(title="Backtest Equity Curve", xaxis_title="Date", yaxis_title="Equity (INR)")
        return fig
