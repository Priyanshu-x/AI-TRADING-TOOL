import unittest
import os
import sqlite3
from datetime import datetime, timedelta
import pandas as pd
from unittest.mock import patch, MagicMock

# Import the new modules
from src.signal_database_manager import SignalDatabaseManager
from src.market_timing_manager import MarketTimingManager
from src.signal_validator import SignalValidator
from src.self_learning_module import SelfLearningModule
from src.trade_signal_generator import TradeSignalGenerator

# Mock data_fetcher.fetch_stock_data for SignalValidator tests
def mock_fetch_stock_data(tickers, output_dir="data", start_date=None, end_date=None):
    summary = {}
    for ticker in tickers:
        # Ensure the requested date is always present in the mock data
        if start_date is None:
            start_date = datetime.now().date() - timedelta(days=5)
        if end_date is None:
            end_date = datetime.now().date()

        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate dummy OHLCV data for each requested date.
        # The SignalValidator is responsible for ensuring it requests data only for actual trading days.
        trading_dates_only_dates = [d.date() for d in dates]
        
        if not trading_dates_only_dates:
            summary[ticker] = {"status": "failed", "records": 0, "reason": "No dates in range for mock data generation"}
            continue

        # Generate dummy OHLCV data for each trading date
        data = {
            'Date': trading_dates_only_dates,
            'Open': [100.0 + i for i in range(len(trading_dates_only_dates))],
            'High': [102.0 + i for i in range(len(trading_dates_only_dates))],
            'Low': [99.0 + i for i in range(len(trading_dates_only_dates))],
            'Close': [101.0 + i for i in range(len(trading_dates_only_dates))],
            'Volume': [100000 + i * 1000 for i in range(len(trading_dates_only_dates))]
        }
        dummy_df = pd.DataFrame(data)
        dummy_df['Date'] = pd.to_datetime(dummy_df['Date'])
        dummy_df.set_index('Date', inplace=True) # Set Date as index before saving
        
        file_path = os.path.join(output_dir, f"{ticker}_data.csv")
        os.makedirs(output_dir, exist_ok=True)
        dummy_df.to_csv(file_path, index=True) # Save with index
        summary[ticker] = {"status": "success", "records": len(dummy_df), "file": file_path}
    return summary

class TestSignalDatabaseManager(unittest.TestCase):
    def setUp(self):
        self.db_path = "test_signals.db"
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.db_manager = SignalDatabaseManager(self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_create_tables(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        self.assertIn(('daily_signals',), tables)
        self.assertIn(('signal_validation',), tables)
        conn.close()

    def test_save_and_get_signals(self):
        today_date = datetime.now().strftime("%Y-%m-%d")
        signal1 = {
            "timestamp": datetime.now().isoformat(),
            "symbol": "TCS.NS",
            "trade_type": "BTST",
            "action": "BUY",
            "entry_price": 3500.00,
            "target_price": 3535.00,
            "confidence_score": 0.85,
            "rationale": "Test rationale 1"
        }
        signal2 = {
            "timestamp": datetime.now().isoformat(),
            "symbol": "RELIANCE.NS",
            "trade_type": "OPTIONS",
            "action": "BUY CALL",
            "strike_price": 2500.00,
            "expiry_date": (datetime.now().date() + timedelta(days=30)).isoformat(),
            "confidence_score": 0.78,
            "rationale": "Test rationale 2"
        }
        self.db_manager.save_signals([signal1, signal2], today_date)
        
        signals = self.db_manager.get_signals_for_validation(today_date)
        self.assertEqual(len(signals), 2)
        self.assertEqual(signals[0]['symbol'], "TCS.NS")
        self.assertIn('db_id', signals[0])

    def test_save_validation_result(self):
        today_date = datetime.now().strftime("%Y-%m-%d")
        signal1 = {
            "timestamp": datetime.now().isoformat(),
            "symbol": "TCS.NS",
            "trade_type": "BTST",
            "action": "BUY",
            "entry_price": 3500.00,
            "target_price": 3535.00,
            "confidence_score": 0.85,
            "rationale": "Test rationale 1"
        }
        self.db_manager.save_signals([signal1], today_date)
        signals = self.db_manager.get_signals_for_validation(today_date)
        signal_id = signals[0]['db_id']

        self.db_manager.save_validation_result(
            signal_id, today_date, 3510.0, 3550.0, 3490.0, 3540.0, True, True, 40.0
        )
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM signal_validation WHERE signal_id = ?", (signal_id,))
        result = cursor.fetchone()
        conn.close()
        self.assertIsNotNone(result)
        self.assertEqual(result[1], signal_id) # signal_id
        self.assertEqual(result[7], 1) # predicted_move_occurred (True)

    def test_get_all_validated_signals(self):
        today_date = datetime.now().strftime("%Y-%m-%d")
        signal1 = {
            "timestamp": datetime.now().isoformat(),
            "symbol": "TCS.NS",
            "trade_type": "BTST",
            "action": "BUY",
            "entry_price": 3500.00,
            "target_price": 3535.00,
            "confidence_score": 0.85,
            "rationale": "Test rationale 1"
        }
        self.db_manager.save_signals([signal1], today_date)
        signals = self.db_manager.get_signals_for_validation(today_date)
        signal_id = signals[0]['db_id']
        self.db_manager.save_validation_result(
            signal_id, today_date, 3510.0, 3550.0, 3490.0, 3540.0, True, True, 40.0
        )

        validated_signals = self.db_manager.get_all_validated_signals()
        self.assertEqual(len(validated_signals), 1)
        self.assertEqual(validated_signals[0]['symbol'], "TCS.NS")
        self.assertEqual(validated_signals[0]['pnl'], 40.0)

class TestMarketTimingManager(unittest.TestCase):
    def setUp(self):
        self.market_manager = MarketTimingManager()

    @patch('src.market_timing_manager.datetime')
    def test_is_market_open_during_hours(self, mock_dt):
        # Mock current time to be within market hours on a weekday
        mock_dt.datetime.utcnow.return_value = datetime(2025, 11, 19, 10, 0, 0) # 3:30 PM IST
        mock_dt.datetime.now.return_value = datetime(2025, 11, 19, 15, 30, 0) # 3:30 PM IST
        mock_dt.time.return_value = datetime(2025, 11, 19, 15, 30, 0).time()
        mock_dt.date.return_value = datetime(2025, 11, 19).date()
        mock_dt.timedelta = timedelta
        mock_dt.timezone = MagicMock(return_value=MagicMock(zone='Asia/Calcutta'))
        mock_dt.datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
        mock_dt.date.side_effect = lambda *args, **kw: datetime(*args, **kw).date()

        self.assertTrue(self.market_manager.is_market_open())

    @patch('src.market_timing_manager.datetime')
    def test_is_market_open_after_hours(self, mock_dt):
        # Mock current time to be after market hours on a weekday
        mock_dt.datetime.utcnow.return_value = datetime(2025, 11, 19, 11, 0, 0) # 4:30 PM IST
        mock_dt.datetime.now.return_value = datetime(2025, 11, 19, 16, 30, 0) # 4:30 PM IST
        mock_dt.time.return_value = datetime(2025, 11, 19, 16, 30, 0).time()
        mock_dt.date.return_value = datetime(2025, 11, 19).date()
        mock_dt.timedelta = timedelta
        mock_dt.timezone = MagicMock(return_value=MagicMock(zone='Asia/Calcutta'))
        mock_dt.datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
        mock_dt.date.side_effect = lambda *args, **kw: datetime(*args, **kw).date()

        self.assertFalse(self.market_manager.is_market_open())

    @patch('src.market_timing_manager.datetime')
    def test_is_market_open_on_weekend(self, mock_dt):
        # Mock current time to be on a Saturday
        mock_dt.datetime.utcnow.return_value = datetime(2025, 11, 22, 6, 0, 0) # Saturday 11:30 AM IST
        mock_dt.datetime.now.return_value = datetime(2025, 11, 22, 11, 30, 0) # Saturday 11:30 AM IST
        mock_dt.time.return_value = datetime(2025, 11, 22, 11, 30, 0).time()
        mock_dt.date.return_value = datetime(2025, 11, 22).date()
        mock_dt.timedelta = timedelta
        mock_dt.timezone = MagicMock(return_value=MagicMock(zone='Asia/Calcutta'))
        mock_dt.datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
        mock_dt.date.side_effect = lambda *args, **kw: datetime(*args, **kw).date()

        self.assertFalse(self.market_manager.is_market_open())

    @patch('src.market_timing_manager.datetime')
    def test_is_holiday(self, mock_dt):
        # Mock a holiday date
        mock_dt.date.return_value = datetime(2025, 1, 26).date() # Republic Day
        self.assertTrue(self.market_manager.is_holiday(mock_dt.date.return_value))

        mock_dt.date.return_value = datetime(2025, 1, 27).date() # Not a holiday
        self.assertFalse(self.market_manager.is_holiday(mock_dt.date.return_value))

    @patch('src.market_timing_manager.datetime')
    def test_is_market_closed_for_day(self, mock_dt):
        # Mock time after market close on a weekday
        mock_dt.datetime.utcnow.return_value = datetime(2025, 11, 19, 11, 0, 0) # 4:30 PM IST
        mock_dt.datetime.now.return_value = datetime(2025, 11, 19, 16, 30, 0) # 4:30 PM IST
        mock_dt.time.return_value = datetime(2025, 11, 19, 16, 30, 0).time()
        mock_dt.date.return_value = datetime(2025, 11, 19).date()
        mock_dt.timedelta = timedelta
        mock_dt.timezone = MagicMock(return_value=MagicMock(zone='Asia/Calcutta'))
        mock_dt.datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
        mock_dt.date.side_effect = lambda *args, **kw: datetime(*args, **kw).date()

        self.assertTrue(self.market_manager.is_market_closed_for_day())

        # Mock time during market hours on a weekday
        mock_dt.datetime.utcnow.return_value = datetime(2025, 11, 19, 6, 0, 0) # 11:30 AM IST
        mock_dt.datetime.now.return_value = datetime(2025, 11, 19, 11, 30, 0) # 11:30 AM IST
        mock_dt.time.return_value = datetime(2025, 11, 19, 11, 30, 0).time()
        self.assertFalse(self.market_manager.is_market_closed_for_day())

        # Mock a holiday
        mock_dt.date.return_value = datetime(2025, 1, 26).date() # Republic Day
        self.assertFalse(self.market_manager.is_market_closed_for_day())

class TestSignalValidator(unittest.TestCase):
    def setUp(self):
        self.db_path = "test_signals_validator.db"
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.db_manager = SignalDatabaseManager(self.db_path)
        self.market_manager = MarketTimingManager() # Initialize MarketTimingManager
        self.validator = SignalValidator(self.db_manager, self.market_manager)
        self.output_dir = "data" # Ensure this matches mock_fetch_stock_data

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        # Clean up dummy CSVs
        if os.path.exists(self.output_dir):
            for f in os.listdir(self.output_dir):
                os.remove(os.path.join(self.output_dir, f))
            os.rmdir(self.output_dir)

    @patch('src.signal_validator.fetch_stock_data', side_effect=mock_fetch_stock_data)
    def test_validate_btst_buy_signal(self, mock_fetch):
        trade_date = (datetime.now().date() - timedelta(days=1)).strftime("%Y-%m-%d")
        signal = {
            "timestamp": (datetime.now() - timedelta(days=1)).isoformat(),
            "symbol": "TCS.NS",
            "trade_type": "BTST",
            "action": "BUY",
            "entry_price": 100.00,
            "target_price": 102.00,
            "confidence_score": 0.8,
            "rationale": "Test BTST BUY"
        }
        self.db_manager.save_signals([signal], trade_date)
        
        # Mock actual data for validation day (today)
        # The mock_fetch_stock_data will create a CSV with today's date as the last entry
        # Close: 103.0, High: 104.0
        
        results = self.validator.validate_signals(trade_date)
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]['predicted_move_occurred']) # 101 > 100
        self.assertTrue(results[0]['target_hit']) # 102 >= 102
        self.assertAlmostEqual(results[0]['pnl'], 1.0) # 101 - 100

    @patch('src.signal_validator.fetch_stock_data', side_effect=mock_fetch_stock_data)
    def test_validate_options_buy_call_signal(self, mock_fetch):
        trade_date = (datetime.now().date() - timedelta(days=1)).strftime("%Y-%m-%d")
        signal = {
            "timestamp": (datetime.now() - timedelta(days=1)).isoformat(),
            "symbol": "RELIANCE.NS",
            "trade_type": "OPTIONS",
            "action": "BUY CALL",
            "entry_price": 2000.00, # Entry price of underlying
            "strike_price": 2010.00,
            "expiry_date": (datetime.now().date() + timedelta(days=30)).isoformat(),
            "confidence_score": 0.7,
            "rationale": "Test OPTIONS BUY CALL"
        }
        self.db_manager.save_signals([signal], trade_date)

        # Mock actual data for validation day (today)
        # Close: 103.0, High: 104.0 (from mock_fetch_stock_data)
        # This mock data is not realistic for options, but tests the logic.
        
        results = self.validator.validate_signals(trade_date)
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0]['predicted_move_occurred']) # 101 > 2000 is False
        self.assertFalse(results[0]['target_hit']) # 102 >= 2010 is False
        self.assertAlmostEqual(results[0]['pnl'], (101.0 - 2010.0) * 100) # (Close - Strike) * 100

    @patch('src.signal_validator.fetch_stock_data', side_effect=mock_fetch_stock_data)
    def test_calculate_overall_metrics(self, mock_fetch):
        trade_date1 = (datetime.now().date() - timedelta(days=2)).strftime("%Y-%m-%d")
        trade_date2 = (datetime.now().date() - timedelta(days=1)).strftime("%Y-%m-%d")

        signal1 = {"timestamp": "...", "symbol": "A.NS", "trade_type": "BTST", "action": "BUY", "entry_price": 100, "target_price": 102, "confidence_score": 0.8, "rationale": ""}
        signal2 = {"timestamp": "...", "symbol": "B.NS", "trade_type": "BTST", "action": "SELL", "entry_price": 200, "target_price": 198, "confidence_score": 0.7, "rationale": ""}
        signal3 = {"timestamp": "...", "symbol": "C.NS", "trade_type": "OPTIONS", "action": "BUY CALL", "entry_price": 300, "strike_price": 305, "confidence_score": 0.9, "rationale": ""}

        self.db_manager.save_signals([signal1], trade_date1)
        self.db_manager.save_signals([signal2, signal3], trade_date2)

        # Validate signals (this will save results to DB)
        self.validator.validate_signals(trade_date1)
        self.validator.validate_signals(trade_date2)

        metrics = self.validator.calculate_overall_metrics()
        self.assertGreater(metrics['total_signals'], 0)
        self.assertIsInstance(metrics['accuracy'], float)
        self.assertIsInstance(metrics['total_pnl'], float)
        self.assertIsInstance(metrics['hit_rate'], float)

class TestSelfLearningModule(unittest.TestCase):
    def setUp(self):
        self.db_path = "test_signals_learning.db"
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.db_manager = SignalDatabaseManager(self.db_path)
        self.learning_module = SelfLearningModule(self.db_manager)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_initial_weights(self):
        weights = self.learning_module.get_current_weights()
        self.assertIn("BTST_BUY", weights)
        self.assertEqual(weights["BTST_BUY"]["accuracy_weight"], 0.5)

    def test_update_weights(self):
        trade_date = (datetime.now().date() - timedelta(days=1)).strftime("%Y-%m-%d")
        signal1 = {
            "timestamp": "...", "symbol": "TCS.NS", "trade_type": "BTST", "action": "BUY", 
            "entry_price": 100, "target_price": 102, "confidence_score": 0.8, "rationale": ""
        }
        signal2 = {
            "timestamp": "...", "symbol": "RELIANCE.NS", "trade_type": "BTST", "action": "SELL", 
            "entry_price": 200, "target_price": 198, "confidence_score": 0.7, "rationale": ""
        }
        self.db_manager.save_signals([signal1, signal2], trade_date)
        
        # Manually add validation results for these signals
        signals_in_db = self.db_manager.get_signals_for_validation(trade_date)
        for s in signals_in_db:
            if s['symbol'] == "TCS.NS":
                self.db_manager.save_validation_result(s['db_id'], trade_date, 100, 103, 99, 102, True, True, 2.0)
            elif s['symbol'] == "RELIANCE.NS":
                self.db_manager.save_validation_result(s['db_id'], trade_date, 200, 201, 197, 199, True, True, 1.0)

        initial_btst_buy_weight = self.learning_module.weights["BTST_BUY"]["accuracy_weight"]
        self.learning_module.update_weights()
        updated_btst_buy_weight = self.learning_module.weights["BTST_BUY"]["accuracy_weight"]
        
        # Since both signals were "correct", weights should increase
        self.assertGreater(updated_btst_buy_weight, initial_btst_buy_weight)

    def test_get_adjusted_confidence(self):
        # Simulate updated weights (e.g., BTST_BUY performed very well)
        self.learning_module.weights["BTST_BUY"]["accuracy_weight"] = 0.8
        self.learning_module.weights["BTST_BUY"]["pnl_weight"] = 0.8

        signal = {
            "symbol": "INFY.NS",
            "trade_type": "BTST",
            "action": "BUY",
            "confidence_score": 0.6 # Original confidence
        }
        adjusted_confidence = self.learning_module.get_adjusted_confidence(signal)
        self.assertGreater(adjusted_confidence, 0.6) # Should be boosted

        # Simulate poor performance
        self.learning_module.weights["BTST_BUY"]["accuracy_weight"] = 0.2
        self.learning_module.weights["BTST_BUY"]["pnl_weight"] = 0.2
        adjusted_confidence = self.learning_module.get_adjusted_confidence(signal)
        self.assertLess(adjusted_confidence, 0.6) # Should be penalized

class TestTradeSignalGeneratorIntegration(unittest.TestCase):
    def setUp(self):
        self.db_path = "test_signals_generator_integration.db"
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.db_manager = SignalDatabaseManager(self.db_path)
        self.learning_module = SelfLearningModule(self.db_manager)
        # Initialize TradeSignalGenerator with the learning module
        self.signal_generator = TradeSignalGenerator(learning_module=self.learning_module)

        # Dummy Data for testing
        dummy_df_data = {
            'Open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129],
            'High': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 132, 133],
            'Low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128],
            'Close': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131],
            'Volume': [100000, 110000, 120000, 130000, 140000, 150000, 160000, 170000, 180000, 190000, 200000, 210000, 220000, 230000, 240000, 250000, 260000, 270000, 500000, 290000, 300000, 310000, 320000, 330000, 340000, 350000, 360000, 370000, 380000, 390000]
        }
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D') # Define dates here
        dummy_df = pd.DataFrame(dummy_df_data, index=dates)
        dummy_df.index.name = 'Date' # Ensure index has a name for consistency

        dummy_df['SMA_20'] = dummy_df['Close'].rolling(window=20).mean()
        dummy_df['SMA_50'] = dummy_df['Close'].rolling(window=20).mean()
        dummy_df['SMA_200'] = dummy_df['Close'].rolling(window=20).mean()
        dummy_df['ATR'] = 2.5
        dummy_df['Volume_Spike'] = False
        dummy_df.loc[dummy_df.index[-1], 'Volume_Spike'] = True
        dummy_df['Support'] = dummy_df['Low'].rolling(window=10, center=True).min()
        dummy_df['Resistance'] = dummy_df['High'].rolling(window=10, center=True).max()
        dummy_df['Breakout'] = False
        dummy_df.loc[dummy_df.index[-1], 'Breakout'] = True
        dummy_df['Breakdown'] = False

        self.all_stocks_data_dummy = {'TESTSTOCK.NS': dummy_df}

        self.dummy_news_df = pd.DataFrame([
            {'symbol': 'TESTSTOCK.NS', 'source': 'Google News', 'headline': 'TESTSTOCK announces record profits, bullish outlook.', 'published_at': datetime.now(), 'sentiment': 'positive', 'confidence': 0.85},
        ])

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_signal_generation_with_learning_module(self):
        # Initially, weights are default (0.5, 0.5)
        # Generate a signal
        btst_signals = self.signal_generator.generate_btst_signals(self.all_stocks_data_dummy, self.dummy_news_df)
        self.assertEqual(len(btst_signals), 1)
        initial_confidence = btst_signals[0]['confidence_score']
        
        # Simulate a positive validation result for this signal type
        trade_date = (datetime.now().date() - timedelta(days=1)).strftime("%Y-%m-%d")
        self.db_manager.save_signals(btst_signals, trade_date)
        signals_in_db = self.db_manager.get_signals_for_validation(trade_date)
        signal_id = signals_in_db[0]['db_id']
        self.db_manager.save_validation_result(signal_id, trade_date, 100, 103, 99, 102, True, True, 2.0)
        
        # Update learning weights
        self.learning_module.update_weights()

        # Generate a new signal of the same type
        new_btst_signals = self.signal_generator.generate_btst_signals(self.all_stocks_data_dummy, self.dummy_news_df)
        self.assertEqual(len(new_btst_signals), 1)
        adjusted_confidence = new_btst_signals[0]['confidence_score']

        # The adjusted confidence should be higher due to positive learning feedback
        self.assertGreater(adjusted_confidence, initial_confidence)


if __name__ == '__main__':
    unittest.main()