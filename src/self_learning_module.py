import pandas as pd
import numpy as np
import logging
from .signal_database_manager import SignalDatabaseManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SelfLearningModule:
    def __init__(self, db_manager: SignalDatabaseManager, initial_weights=None):
        self.db_manager = db_manager
        # Initialize weights for different signal types/actions
        # These weights will be adjusted based on performance
        self.weights = initial_weights if initial_weights is not None else {
            "BTST_BUY": {"accuracy_weight": 0.5, "pnl_weight": 0.5},
            "BTST_SELL": {"accuracy_weight": 0.5, "pnl_weight": 0.5},
            "OPTIONS_BUY CALL": {"accuracy_weight": 0.5, "pnl_weight": 0.5},
            "OPTIONS_BUY PUT": {"accuracy_weight": 0.5, "pnl_weight": 0.5},
        }
        self.learning_rate = 0.1 # How much to adjust weights each time

    def update_weights(self):
        """
        Updates the learning weights based on the performance of past signals.
        This is a simplified reinforcement learning-like mechanism.
        """
        all_validated_signals = self.db_manager.get_all_validated_signals()
        if not all_validated_signals:
            logging.info("No validated signals available for learning.")
            return

        df = pd.DataFrame(all_validated_signals)
        
        # Group by signal type and action to update specific weights
        for (trade_type, action), group in df.groupby(['trade_type', 'action']):
            signal_key = f"{trade_type}_{action}"
            if signal_key not in self.weights:
                self.weights[signal_key] = {"accuracy_weight": 0.5, "pnl_weight": 0.5}

            # Calculate performance metrics for this group
            total_signals = len(group)
            if total_signals == 0:
                continue

            accuracy = group['predicted_move_occurred'].mean() # Mean of booleans is accuracy
            avg_pnl = group['pnl'].mean()

            # Simple reward/penalty system for weights
            # Reward accuracy and positive P&L, penalize low accuracy and negative P&L
            
            # Adjust accuracy weight
            if accuracy > 0.5: # If better than random
                self.weights[signal_key]["accuracy_weight"] += self.learning_rate * (accuracy - 0.5)
            else:
                self.weights[signal_key]["accuracy_weight"] -= self.learning_rate * (0.5 - accuracy)
            
            # Adjust P&L weight (normalize P&L for comparison, e.g., by average entry price or ATR)
            # For simplicity, let's assume avg_pnl directly reflects performance
            if avg_pnl > 0:
                self.weights[signal_key]["pnl_weight"] += self.learning_rate * (avg_pnl / (abs(avg_pnl) + 1)) # Small normalization
            else:
                self.weights[signal_key]["pnl_weight"] -= self.learning_rate * (abs(avg_pnl) / (abs(avg_pnl) + 1))

            # Ensure weights stay within a reasonable range (e.g., 0 to 1)
            self.weights[signal_key]["accuracy_weight"] = np.clip(self.weights[signal_key]["accuracy_weight"], 0.1, 0.9)
            self.weights[signal_key]["pnl_weight"] = np.clip(self.weights[signal_key]["pnl_weight"], 0.1, 0.9)
            
            logging.info(f"Updated weights for {signal_key}: {self.weights[signal_key]}")

    def get_adjusted_confidence(self, signal):
        """
        Calculates an adjusted confidence score for a new signal based on learned weights.
        This method would be called by the TradeSignalGenerator.
        """
        trade_type = signal.get("trade_type")
        action = signal.get("action")
        original_confidence = signal.get("confidence_score", 0.5)

        signal_key = f"{trade_type}_{action}"
        
        if signal_key in self.weights:
            # For simplicity, let's just use the accuracy weight to adjust the original confidence
            # A more sophisticated model would combine accuracy, P&L, and other factors.
            accuracy_influence = self.weights[signal_key]["accuracy_weight"]
            pnl_influence = self.weights[signal_key]["pnl_weight"]
            
            # Simple linear adjustment: if accuracy weight is high, boost confidence
            # if low, reduce it.
            adjusted_confidence = original_confidence * (1 + (accuracy_influence - 0.5) + (pnl_influence - 0.5))
            
            # Ensure confidence stays within 0-1 range
            adjusted_confidence = np.clip(adjusted_confidence, 0.0, 1.0)
            logging.debug(f"Adjusted confidence for {signal_key} from {original_confidence:.2f} to {adjusted_confidence:.2f}")
            return adjusted_confidence
        else:
            logging.warning(f"No learning weights found for {signal_key}. Returning original confidence.")
            return original_confidence

    def get_current_weights(self):
        """Returns the current learning weights."""
        return self.weights

if __name__ == "__main__":
    # Example Usage:
    test_db_manager = SignalDatabaseManager(db_path="test_signals.db")
    learning_module = SelfLearningModule(test_db_manager)

    print("Initial Weights:")
    print(learning_module.get_current_weights())

    # Simulate some validation results being saved to the DB
    # (This would typically happen via SignalValidator)
    # For testing, let's manually add some results to the test_signals.db
    # Assuming test_signals.db already has some signals from previous runs.
    
    # Example: Manually add a validation result for a BTST BUY signal
    # You would need to know a valid signal_id from your test_signals.db
    # For demonstration, let's assume signal_id 1 is a BTST BUY
    # and signal_id 2 is an OPTIONS BUY CALL
    
    # To make this test runnable, ensure test_signals.db has entries.
    # Run signal_database_manager.py's __main__ block first to populate.
    
    # Let's add some dummy validation results if they don't exist
    # This is a bit hacky for a __main__ block, but for quick testing.
    
    # Fetch existing signals to get their IDs
    signals_for_test = test_db_manager.get_signals_for_validation((datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"))
    if not signals_for_test:
        print("No signals to validate for testing learning module. Please run signal_database_manager.py __main__ first.")
    else:
        for signal in signals_for_test:
            if signal['symbol'] == "TCS.NS" and signal['trade_type'] == "BTST" and signal['action'] == "BUY":
                test_db_manager.save_validation_result(
                    signal['db_id'], 
                    datetime.now().strftime("%Y-%m-%d"), 
                    3510.0, 3550.0, 3490.0, 3540.0, # Actual OHLC
                    True, True, 40.0 # Predicted move, target hit, PnL
                )
            elif signal['symbol'] == "RELIANCE.NS" and signal['trade_type'] == "OPTIONS" and signal['action'] == "BUY CALL":
                test_db_manager.save_validation_result(
                    signal['db_id'], 
                    datetime.now().strftime("%Y-%m-%d"), 
                    2490.0, 2520.0, 2480.0, 2515.0, # Actual OHLC
                    True, True, 50.0 # Predicted move, target hit, PnL
                )

    print("\nUpdating weights based on validated signals...")
    learning_module.update_weights()

    print("\nUpdated Weights:")
    print(learning_module.get_current_weights())

    # Test adjusted confidence for a new hypothetical signal
    new_btst_signal = {
        "timestamp": datetime.now().isoformat(),
        "symbol": "INFY.NS",
        "trade_type": "BTST",
        "action": "BUY",
        "entry_price": 1500.00,
        "target_price": 1515.00,
        "rationale": "Fresh positive news",
        "confidence_score": 0.70 # Original confidence
    }
    adjusted_conf = learning_module.get_adjusted_confidence(new_btst_signal)
    print(f"\nAdjusted confidence for new INFY BTST BUY signal (original 0.70): {adjusted_conf:.2f}")