import sqlite3
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SignalDatabaseManager:
    def __init__(self, db_path="signals.db"):
        self.db_path = db_path
        self._create_tables()

    def _create_tables(self):
        """Creates the necessary tables in the SQLite database."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Table for storing daily signals
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    trade_date TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    trade_type TEXT NOT NULL,
                    action TEXT NOT NULL,
                    entry_price REAL,
                    target_price REAL,
                    strike_price REAL,
                    expiry_date TEXT,
                    confidence_score REAL,
                    rationale TEXT,
                    signal_json TEXT NOT NULL
                )
            """)
            
            # Table for storing signal validation results
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signal_validation (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id INTEGER NOT NULL,
                    validation_date TEXT NOT NULL,
                    actual_open REAL,
                    actual_high REAL,
                    actual_low REAL,
                    actual_close REAL,
                    predicted_move_occurred BOOLEAN,
                    target_hit BOOLEAN,
                    pnl REAL,
                    FOREIGN KEY (signal_id) REFERENCES daily_signals(id)
                )
            """)
            conn.commit()
            logging.info("Database tables created or already exist.")
        except sqlite3.Error as e:
            logging.error(f"Error creating database tables: {e}")
        finally:
            if conn:
                conn.close()

    def _convert_numpy_types(self, obj):
        """Recursively converts numpy types in a dictionary to native Python types."""
        if isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(elem) for elem in obj]
        elif hasattr(obj, 'item') and callable(obj.item): # For numpy.int64, numpy.float64 etc.
            return obj.item()
        return obj

    def save_signals(self, signals, trade_date):
        """
        Saves a list of signals to the daily_signals table.
        Each signal dictionary is stored as a JSON string after converting numpy types.
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for signal in signals:
                # Convert numpy types to native Python types before JSON serialization
                converted_signal = self._convert_numpy_types(signal)

                timestamp = converted_signal.get("timestamp", datetime.now().isoformat())
                symbol = converted_signal.get("symbol")
                trade_type = converted_signal.get("trade_type")
                action = converted_signal.get("action")
                entry_price = converted_signal.get("entry_price")
                target_price = converted_signal.get("target_price")
                strike_price = converted_signal.get("strike_price")
                expiry_date = converted_signal.get("expiry_date")
                confidence_score = converted_signal.get("confidence_score")
                rationale = converted_signal.get("rationale")
                signal_json = json.dumps(converted_signal)

                cursor.execute("""
                    INSERT INTO daily_signals (
                        timestamp, trade_date, symbol, trade_type, action,
                        entry_price, target_price, strike_price, expiry_date,
                        confidence_score, rationale, signal_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    timestamp, trade_date, symbol, trade_type, action,
                    entry_price, target_price, strike_price, expiry_date,
                    confidence_score, rationale, signal_json
                ))
            conn.commit()
            logging.info(f"Successfully saved {len(signals)} signals for {trade_date}.")
        except sqlite3.Error as e:
            logging.error(f"Error saving signals: {e}")
        finally:
            if conn:
                conn.close()

    def get_signals_for_validation(self, trade_date):
        """
        Fetches signals from a specific trade_date that have not yet been validated.
        Returns a list of dictionaries, each containing signal details and its DB 'id'.
        """
        conn = None
        signals_to_validate = []
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT ds.id, ds.signal_json
                FROM daily_signals ds
                LEFT JOIN signal_validation sv ON ds.id = sv.signal_id
                WHERE ds.trade_date = ? AND sv.signal_id IS NULL
            """, (trade_date,))
            
            rows = cursor.fetchall()
            for row in rows:
                signal_id = row[0]
                signal_data = json.loads(row[1])
                signal_data['db_id'] = signal_id # Add the database ID for later reference
                signals_to_validate.append(signal_data)
            
            logging.info(f"Fetched {len(signals_to_validate)} signals for validation for {trade_date}.")
        except sqlite3.Error as e:
            logging.error(f"Error fetching signals for validation: {e}")
        finally:
            if conn:
                conn.close()
        return signals_to_validate

    def save_validation_result(self, signal_id, validation_date, actual_open, actual_high, actual_low, actual_close, predicted_move_occurred, target_hit, pnl):
        """
        Saves the validation result for a given signal.
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO signal_validation (
                    signal_id, validation_date, actual_open, actual_high, actual_low, 
                    actual_close, predicted_move_occurred, target_hit, pnl
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal_id, validation_date, actual_open, actual_high, actual_low, 
                actual_close, predicted_move_occurred, target_hit, pnl
            ))
            conn.commit()
            logging.info(f"Saved validation result for signal ID: {signal_id} on {validation_date}.")
        except sqlite3.Error as e:
            logging.error(f"Error saving validation result for signal ID {signal_id}: {e}")
        finally:
            if conn:
                conn.close()

    def get_all_validated_signals(self):
        """
        Fetches all validated signals with their validation results.
        Useful for the self-learning mechanism.
        """
        conn = None
        validated_signals = []
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    ds.symbol, ds.trade_type, ds.action, ds.entry_price, ds.target_price, 
                    ds.strike_price, ds.expiry_date, ds.confidence_score, ds.rationale,
                    sv.actual_open, sv.actual_high, sv.actual_low, sv.actual_close,
                    sv.predicted_move_occurred, sv.target_hit, sv.pnl
                FROM daily_signals ds
                JOIN signal_validation sv ON ds.id = sv.signal_id
                ORDER BY ds.trade_date DESC, ds.timestamp DESC
            """)
            
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            for row in rows:
                validated_signals.append(dict(zip(columns, row)))
            
            logging.info(f"Fetched {len(validated_signals)} validated signals.")
        except sqlite3.Error as e:
            logging.error(f"Error fetching all validated signals: {e}")
        finally:
            if conn:
                conn.close()
        return validated_signals

if __name__ == "__main__":
    # Example Usage:
    db_manager = SignalDatabaseManager(db_path="test_signals.db")

    # Dummy signals
    dummy_btst_signal = {
        "timestamp": datetime.now().isoformat(),
        "symbol": "TCS.NS",
        "trade_type": "BTST",
        "action": "BUY",
        "entry_price": 3500.00,
        "target_price": 3535.00,
        "rationale": "Positive news and bullish momentum",
        "confidence_score": 0.85
    }

    dummy_options_signal = {
        "timestamp": datetime.now().isoformat(),
        "symbol": "RELIANCE.NS",
        "trade_type": "OPTIONS",
        "action": "BUY CALL",
        "strike_price": 2500.00,
        "expiry_date": (datetime.now().date() + timedelta(days=30)).isoformat(),
        "rationale": "High volatility and positive sentiment",
        "confidence_score": 0.78
    }

    # Save signals
    today_date = datetime.now().strftime("%Y-%m-%d")
    db_manager.save_signals([dummy_btst_signal, dummy_options_signal], today_date)

    # Fetch signals for validation (assuming they were generated yesterday)
    yesterday_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    # For testing, let's fetch today's signals for validation
    signals_to_validate = db_manager.get_signals_for_validation(today_date)
    print(f"\nSignals to validate for {today_date}:")
    for signal in signals_to_validate:
        print(signal)
        
        # Simulate validation
        if signal['symbol'] == "TCS.NS":
            actual_open = 3510.00
            actual_high = 3550.00
            actual_low = 3490.00
            actual_close = 3540.00
            predicted_move_occurred = (actual_close > signal['entry_price']) if signal['action'] == 'BUY' else (actual_close < signal['entry_price'])
            target_hit = (actual_high >= signal['target_price']) if signal['action'] == 'BUY' else (actual_low <= signal['target_price'])
            pnl = (actual_close - signal['entry_price']) if signal['action'] == 'BUY' else (signal['entry_price'] - actual_close)
            db_manager.save_validation_result(
                signal['db_id'], today_date, actual_open, actual_high, actual_low, actual_close,
                predicted_move_occurred, target_hit, pnl
            )
        elif signal['symbol'] == "RELIANCE.NS":
            actual_open = 2490.00
            actual_high = 2520.00
            actual_low = 2480.00
            actual_close = 2515.00
            # For options, validation logic is more complex, simplifying for example
            predicted_move_occurred = (actual_close > signal['entry_price']) if signal['action'] == 'BUY CALL' else (actual_close < signal['entry_price'])
            target_hit = (actual_high >= signal['strike_price']) if signal['action'] == 'BUY CALL' else (actual_low <= signal['strike_price'])
            pnl = 50.0 # Dummy PnL for options
            db_manager.save_validation_result(
                signal['db_id'], today_date, actual_open, actual_high, actual_low, actual_close,
                predicted_move_occurred, target_hit, pnl
            )

    # Fetch all validated signals
    all_validated_signals = db_manager.get_all_validated_signals()
    print("\nAll validated signals:")
    for signal in all_validated_signals:
        print(signal)