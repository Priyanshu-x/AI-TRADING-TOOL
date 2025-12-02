import pandas as pd
from datetime import datetime, timedelta
import logging
from .data_fetcher import fetch_stock_data # Assuming data_fetcher can fetch specific day's data
from .signal_database_manager import SignalDatabaseManager
from .market_timing_manager import MarketTimingManager # Import MarketTimingManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SignalValidator:
    def __init__(self, db_manager: SignalDatabaseManager, market_manager: MarketTimingManager):
        self.db_manager = db_manager
        self.market_manager = market_manager

    def _fetch_actual_market_data(self, symbol, trade_date):
        """
        Fetches actual market data (Open, High, Low, Close) for a given symbol
        on the next valid trading day after the trade_date.
        """
        # Determine the actual validation date (next trading day)
        signal_trade_date = datetime.strptime(trade_date, "%Y-%m-%d").date()
        validation_date = self.market_manager.get_next_trading_day(signal_trade_date)
        
        logging.info(f"Fetching validation data for {symbol} on {validation_date.isoformat()} (original trade date: {trade_date})")
        
        download_summary = fetch_stock_data([symbol], output_dir="data", start_date=validation_date, end_date=validation_date)
        if download_summary.get(symbol, {}).get("status") == "success":
            file_path = download_summary[symbol]["file"]
            try:
                df = pd.read_csv(file_path, index_col='Date', parse_dates=True) # Read 'Date' as index and parse dates
                
                # Convert validation_date to Timestamp for exact matching
                validation_timestamp = pd.Timestamp(validation_date)
                
                # Filter for the exact validation date
                if validation_timestamp in df.index:
                    return df.loc[validation_timestamp]
                else:
                    logging.warning(f"No data found for {symbol} on validation date {validation_date}.")
                    return None
            except Exception as e:
                logging.error(f"Error reading or processing data for {symbol} from {file_path}: {e}")
                return None
        else:
            logging.warning(f"Failed to fetch data for {symbol} for validation on {validation_date}.")
            return None


    def validate_signals(self, trade_date):
        """
        Fetches unvalidated signals for a given trade_date,
        fetches actual market data for the next trading day,
        and validates each signal.
        """
        signals_to_validate = self.db_manager.get_signals_for_validation(trade_date)
        if not signals_to_validate:
            logging.info(f"No signals to validate for trade date {trade_date}.")
            return []

        validation_results = []
        for signal in signals_to_validate:
            symbol = signal['symbol']
            signal_id = signal['db_id']
            action = signal['action']
            entry_price = signal.get('entry_price')
            target_price = signal.get('target_price')
            strike_price = signal.get('strike_price')
            trade_type = signal['trade_type']

            actual_data = self._fetch_actual_market_data(symbol, trade_date)
            
            if actual_data is None:
                logging.warning(f"Skipping validation for signal ID {signal_id} ({symbol}): No actual market data available.")
                continue

            actual_open = actual_data['Open']
            actual_high = actual_data['High']
            actual_low = actual_data['Low']
            actual_close = actual_data['Close']
            validation_date = actual_data.name.strftime("%Y-%m-%d") # The date of the actual data

            predicted_move_occurred = False
            target_hit = False
            pnl = 0.0

            if trade_type == "BTST":
                if action == "BUY":
                    # Predicted move: Price goes up from entry
                    predicted_move_occurred = actual_close > entry_price
                    # Target hit: High reaches or exceeds target price
                    target_hit = actual_high >= target_price
                    # P&L: (Close - Entry)
                    pnl = actual_close - entry_price
                elif action == "SELL": # Short sell
                    # Predicted move: Price goes down from entry
                    predicted_move_occurred = actual_close < entry_price
                    # Target hit: Low reaches or falls below target price
                    target_hit = actual_low <= target_price
                    # P&L: (Entry - Close)
                    pnl = entry_price - actual_close
            
            elif trade_type == "OPTIONS":
                # Options validation is more complex and depends on option type (CALL/PUT)
                # and strike price relative to actual prices.
                # For simplicity, let's assume a basic directional check and a dummy P&L.
                if action == "BUY CALL":
                    # Predicted move: Price goes up
                    predicted_move_occurred = actual_close > entry_price # Using entry_price from signal as a proxy for underlying
                    # Target hit: If actual_high exceeds strike price significantly, or a specific target for the option
                    # For now, let's say if actual_high is above strike, it's a "hit"
                    target_hit = actual_high >= strike_price
                    # P&L: This would require option pricing models, simplifying for now
                    pnl = (actual_close - strike_price) * 100 # Dummy P&L, assuming 100 shares per lot
                elif action == "BUY PUT":
                    # Predicted move: Price goes down
                    predicted_move_occurred = actual_close < entry_price
                    # Target hit: If actual_low is below strike price significantly
                    target_hit = actual_low <= strike_price
                    # P&L: Dummy P&L
                    pnl = (strike_price - actual_close) * 100 # Dummy P&L

            self.db_manager.save_validation_result(
                signal_id, validation_date, actual_open, actual_high, actual_low, actual_close,
                predicted_move_occurred, target_hit, pnl
            )
            
            validation_results.append({
                "signal_id": signal_id,
                "symbol": symbol,
                "trade_type": trade_type,
                "action": action,
                "validation_date": validation_date,
                "predicted_move_occurred": predicted_move_occurred,
                "target_hit": target_hit,
                "pnl": pnl
            })
            logging.info(f"Validated signal ID {signal_id} ({symbol}): Predicted move occurred: {predicted_move_occurred}, Target hit: {target_hit}, P&L: {pnl:.2f}")

        return validation_results

    def calculate_overall_metrics(self):
        """
        Calculates overall accuracy, P&L, and hit rate from all validated signals.
        """
        all_validated_signals = self.db_manager.get_all_validated_signals()
        if not all_validated_signals:
            logging.info("No validated signals to calculate metrics.")
            return {"accuracy": 0.0, "total_pnl": 0.0, "hit_rate": 0.0, "total_signals": 0}

        total_signals = len(all_validated_signals)
        correct_predictions = sum(1 for s in all_validated_signals if s['predicted_move_occurred'])
        target_hits = sum(1 for s in all_validated_signals if s['target_hit'])
        total_pnl = sum(s['pnl'] for s in all_validated_signals)

        accuracy = (correct_predictions / total_signals) * 100 if total_signals > 0 else 0.0
        hit_rate = (target_hits / total_signals) * 100 if total_signals > 0 else 0.0

        metrics = {
            "accuracy": accuracy,
            "total_pnl": total_pnl,
            "hit_rate": hit_rate,
            "total_signals": total_signals
        }
        logging.info(f"Overall Metrics: Accuracy={accuracy:.2f}%, Total P&L={total_pnl:.2f}, Hit Rate={hit_rate:.2f}%")
        return metrics

if __name__ == "__main__":
    # This block is for testing the SignalValidator in isolation.
    # In a real scenario, main.py would orchestrate this.

    # Ensure a test database exists and has some signals
    test_db_manager = SignalDatabaseManager(db_path="test_signals.db")
    
    # Add some dummy signals for validation if the DB is empty
    today_date_str = datetime.now().strftime("%Y-%m-%d")
    yesterday_date_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    # Check if there are signals from yesterday to validate
    signals_from_yesterday = test_db_manager.get_signals_for_validation(yesterday_date_str)
    if not signals_from_yesterday:
        print(f"No signals from {yesterday_date_str} to validate. Adding dummy signals for testing.")
        dummy_btst_signal_yesterday = {
            "timestamp": (datetime.now() - timedelta(days=1)).isoformat(),
            "symbol": "TCS.NS",
            "trade_type": "BTST",
            "action": "BUY",
            "entry_price": 3500.00,
            "target_price": 3535.00,
            "rationale": "Positive news and bullish momentum",
            "confidence_score": 0.85
        }
        dummy_options_signal_yesterday = {
            "timestamp": (datetime.now() - timedelta(days=1)).isoformat(),
            "symbol": "RELIANCE.NS",
            "trade_type": "OPTIONS",
            "action": "BUY CALL",
            "strike_price": 2500.00,
            "expiry_date": (datetime.now().date() + timedelta(days=30)).isoformat(),
            "rationale": "High volatility and positive sentiment",
            "confidence_score": 0.78
        }
        test_db_manager.save_signals([dummy_btst_signal_yesterday, dummy_options_signal_yesterday], yesterday_date_str)
        signals_from_yesterday = test_db_manager.get_signals_for_validation(yesterday_date_str)


    validator = SignalValidator(test_db_manager)

    print(f"\n--- Validating signals for {yesterday_date_str} ---")
    # To properly test, you'd need actual market data for the day after yesterday.
    # For this test, we'll assume data_fetcher can get it.
    validation_results = validator.validate_signals(yesterday_date_str)
    for res in validation_results:
        print(res)

    print("\n--- Overall Metrics ---")
    metrics = validator.calculate_overall_metrics()
    print(metrics)