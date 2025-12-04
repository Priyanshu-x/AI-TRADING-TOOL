# ai_trading_signal_tool/trade_signal_generator.py

import pandas as pd
from datetime import datetime, timedelta
import json
import os
import logging
from .self_learning_module import SelfLearningModule

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TradeSignalGenerator:
    def __init__(self, sentiment_threshold=0.5, momentum_strength_threshold=0.3, btst_target_percentage=0.01, learning_module=None,
                 gap_threshold_percent=0.005, volume_spike_multiplier=1.5, bearish_score_threshold=-0.3,
                 put_itm_delta=50, put_otm_delta=50,
                 btst_bearish_min_drop_pct=0.015, btst_bearish_close_to_low_pct=0.3,
                 btst_bearish_min_vol_multiple=1.2, btst_bearish_rsi_threshold=50):
        """
        Initializes the TradeSignalGenerator with various thresholds and modules.
        
        Args:
            sentiment_threshold (float): Minimum confidence for news sentiment to be considered.
            momentum_strength_threshold (float): Minimum bullish momentum score for a CALL signal.
            btst_target_percentage (float): Target profit percentage for BTST trades.
            learning_module (SelfLearningModule, optional): Module for adjusting signal confidence.
            gap_threshold_percent (float): Minimum percentage for a gap-down to be considered bearish.
            volume_spike_multiplier (float): Multiplier for average volume to detect a spike.
            bearish_score_threshold (float): Minimum bearish score for a PUT signal.
            put_itm_delta (int): Points for In-The-Money (ITM) strike selection for PUTs.
            put_otm_delta (int): Points for Out-The-Money (OTM) strike selection for PUTs.
            btst_bearish_min_drop_pct (float): Minimum percentage drop for a bearish BTST signal.
            btst_bearish_close_to_low_pct (float): Max percentage from day low for close for bearish BTST.
            btst_bearish_min_vol_multiple (float): Min volume multiplier vs previous day for bearish BTST.
            btst_bearish_rsi_threshold (int): Max RSI for bearish BTST signal.
        """
        self.sentiment_threshold = sentiment_threshold
        self.momentum_strength_threshold = momentum_strength_threshold
        self.btst_target_percentage = btst_target_percentage
        # New bearish configuration parameters
        self.gap_threshold_percent = gap_threshold_percent
        self.volume_spike_multiplier = volume_spike_multiplier
        self.bearish_score_threshold = bearish_score_threshold
        self.put_itm_delta = put_itm_delta
        self.put_otm_delta = put_otm_delta
        
        # BTST Bearish specific thresholds (configurable)
        self.btst_bearish_min_drop_pct = btst_bearish_min_drop_pct
        self.btst_bearish_close_to_low_pct = btst_bearish_close_to_low_pct
        self.btst_bearish_min_vol_multiple = btst_bearish_min_vol_multiple
        self.btst_bearish_rsi_threshold = btst_bearish_rsi_threshold

        self.signals = []
        self.learning_module = learning_module # Inject learning module

    def _assess_sentiment_for_stock(self, symbol, news_df):
        """Aggregates sentiment for a given stock."""
        stock_news = news_df[news_df['symbol'] == symbol]
        if stock_news.empty:
            return {"overall_sentiment": "neutral", "confidence": 0.0}

        positive_news = stock_news[(stock_news['sentiment'] == 'positive') & (stock_news['confidence'] >= self.sentiment_threshold)]
        negative_news = stock_news[(stock_news['sentiment'] == 'negative') & (stock_news['confidence'] >= self.sentiment_threshold)]

        if len(positive_news) > len(negative_news) and len(positive_news) > 0:
            return {"overall_sentiment": "positive", "confidence": positive_news['confidence'].mean()}
        elif len(negative_news) > len(positive_news) and len(negative_news) > 0:
            return {"overall_sentiment": "negative", "confidence": negative_news['confidence'].mean()}
        else:
            return {"overall_sentiment": "neutral", "confidence": 0.0}

    def _compute_bullish_score(self, df):
        """
        Computes a bullish score based on various technical indicators.
        Returns a score between 0 and 1.
        """
        if df.empty:
            return 0.0

        latest_data = df.iloc[-1]
        bullish_score = 0
        reasons = []

        # Price vs SMAs
        if latest_data['Close'] > latest_data['SMA_20']:
            bullish_score += 0.2
            reasons.append("Price above SMA_20.")
        if latest_data['Close'] > latest_data['SMA_50']:
            bullish_score += 0.2
            reasons.append("Price above SMA_50.")
        if latest_data['SMA_20'] > latest_data['SMA_50']:
            bullish_score += 0.2 # Short term trend up
            reasons.append("SMA_20 above SMA_50 (uptrend).")

        # Breakouts
        if latest_data['Breakout']:
            bullish_score += 0.2
            reasons.append("Breakout detected.")

        # Volume Spike (only if on a green candle)
        if latest_data['Volume_Spike'] and latest_data['Close'] > latest_data['Open']: # Green candle with spike
            bullish_score += 0.1
            reasons.append("Volume spike on a green candle.")
        
        # Price relative to Resistance
        if latest_data['Close'] > latest_data['Resistance']: # Above resistance
            bullish_score += 0.1
            reasons.append("Price above Resistance.")

        # Normalize score to be between 0 and 1
        return min(1.0, bullish_score / 1.0), reasons # Max possible score is around 1.0


    def _compute_bearish_score(self, df):
        """
        Computes a bearish score based on various technical indicators and custom bearish conditions.
        Returns a score between 0 and 1.
        """
        if df.empty:
            return 0.0

        bearish_score = 0
        reasons = []

        # Check custom bearish conditions
        is_gap_down, gap_down_reason = self.is_bearish_gapdown(df)
        if is_gap_down:
            bearish_score += 0.3 # Higher weight for strong gap-down
            reasons.append(gap_down_reason)

        is_downtrend, downtrend_reason = self.is_downtrend_with_volume(df)
        if is_downtrend:
            bearish_score += 0.3
            reasons.append(downtrend_reason)

        is_breakdown, breakdown_reason = self.is_support_breakdown(df)
        if is_breakdown:
            bearish_score += 0.3
            reasons.append(breakdown_reason)

        # Additional technical indicators for bearishness
        latest_data = df.iloc[-1]

        # Price vs SMAs
        if latest_data['Close'] < latest_data['SMA_20']:
            bearish_score += 0.1
            reasons.append("Price below SMA_20.")
        if latest_data['Close'] < latest_data['SMA_50']:
            bearish_score += 0.1
            reasons.append("Price below SMA_50.")
        if latest_data['SMA_20'] < latest_data['SMA_50']:
            bearish_score += 0.1 # Short term trend down
            reasons.append("SMA_20 below SMA_50 (downtrend).")

        # Breakdowns
        if latest_data['Breakdown']:
            bearish_score += 0.2
            reasons.append("Breakdown detected.")

        # Volume Spike (only if on a red candle)
        if latest_data['Volume_Spike'] and latest_data['Close'] < latest_data['Open']: # Red candle with spike
            bearish_score += 0.1
            reasons.append("Volume spike on a red candle.")
        
        # Price relative to Support
        if latest_data['Close'] < latest_data['Support']: # Below support
            bearish_score += 0.1
            reasons.append("Price below Support.")


        # Normalize score to be between 0 and 1
        return min(1.0, bearish_score / 1.3), reasons # Max possible score is around 1.3 (0.3*3 + 0.1*5 + 0.2)

    def btst_bullish_conditions(self, row):
        """
        Existing bullish BTST conditions (placeholder for actual logic from requirements).
        This function will be adapted to work with `row` (a single row from a DataFrame).
        """
        # Example: Based on existing _compute_bullish_score logic, simplified for row application
        # Actual implementation would use conditions like:
        # row['Close'] > row['SMA_20'] and row['Close'] > row['SMA_50']
        # and row['SMA_20'] > row['SMA_50'] and row['Breakout'] etc.
        # For now, this is a placeholder to be refined if actual row-based bullish conditions are needed.
        # The current implementation in generate_btst_signals uses sentiment and _compute_bullish_score.
        # This function will be more relevant if explicit row-wise filtering is added for BTST.
        
        # Placeholder for demonstration, assuming 'Bullish_BTST_Candidate' is a pre-calculated column
        # In this task, we're asked to keep existing logic untouched, so we will adjust generate_btst_signals
        # to use the existing scoring mechanism for bullish side, and use this function for bearish.
        # If the original design intended row-based filtering for bullish, that logic would go here.
        return True # Default to true for now, as existing bullish logic relies on scores and sentiment

    def btst_bearish_conditions(self, row):
        """
        Checks for bearish BTST conditions based on price, volume, and indicator confirmations.
        Assumes 'Open', 'High', 'Low', 'Close', 'Volume', 'Prev_Close', 'Prev_Volume', 'SMA_20', 'SMA_50', 'RSI' are available in the row.
        """
        # Ensure required data exists
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Prev_Close', 'Prev_Volume', 'SMA_20', 'SMA_50', 'RSI']
        if not all(col in row.index for col in required_cols):
            logging.debug(f"Missing required columns for bearish BTST conditions in row: {set(required_cols) - set(row.index)}")
            return False

        conditions_met = []

        # 1. Price in short-term downtrend (below SMA_20 or SMA_50)
        if row['Close'] < row['SMA_20'] or row['Close'] < row['SMA_50']:
            conditions_met.append("Downtrend: Price below short-term MAs.")
        
        # 2. Strong negative day:
        # Today’s close is significantly lower than previous close
        price_drop_pct = (row['Prev_Close'] - row['Close']) / row['Prev_Close']
        if price_drop_pct >= self.btst_bearish_min_drop_pct:
            conditions_met.append(f"Strong Negative Day: Price dropped {price_drop_pct:.2%} (>{self.btst_bearish_min_drop_pct:.2%}).")

        # Close is in the lower part of the day’s range (e.g., close <= 30% from day low).
        day_range = row['High'] - row['Low']
        if day_range > 0 and (row['Close'] - row['Low']) / day_range <= self.btst_bearish_close_to_low_pct:
            conditions_met.append(f"Strong Negative Day: Close near day low (within {self.btst_bearish_close_to_low_pct:.0%} range).")

        # 3. Volume confirmation:
        # Today’s volume greater than previous day volume.
        # Optionally, today’s volume above a moving-average volume (e.g., 20‑day avg volume) if that metric already exists.
        # For simplicity, we'll use previous day's volume for now. Avg_Volume can be added if available in row.
        if row['Volume'] > (row['Prev_Volume'] * self.btst_bearish_min_vol_multiple):
            conditions_met.append(f"Volume Confirmation: Current volume ({row['Volume']}) > {self.btst_bearish_min_vol_multiple}x previous volume ({row['Prev_Volume']}).")
        
        # 4. Optional indicator confirmation (if my data already has them):
        # RSI below a neutral level (e.g., RSI < 50).
        if 'RSI' in row.index and row['RSI'] < self.btst_bearish_rsi_threshold:
            conditions_met.append(f"RSI Confirmation: RSI ({row['RSI']:.2f}) below {self.btst_bearish_rsi_threshold}.")
        
        # A strong bearish signal should meet multiple conditions, e.g., at least 3
        # The number 3 here is arbitrary and can be made configurable if needed.
        if len(conditions_met) >= 3:
            logging.debug(f"Bearish BTST conditions met for {row.name}: {'; '.join(conditions_met)}")
            return True
        
        logging.debug(f"Bearish BTST conditions NOT met for {row.name}. Met {len(conditions_met)} conditions: {'; '.join(conditions_met)}")
        return False

    def is_bearish_gapdown(self, df):
        """
        Checks for a bearish gap-down from yesterday's close with above-average volume.
        Assumes 'Open' and 'Previous_Close' (or equivalent) are available in the DataFrame.
        Also assumes 'Avg_Volume' is available or can be calculated.
        """
        if df.empty or len(df) < 2:
            return False, "Insufficient data for gap-down analysis."

        latest_data = df.iloc[-1]
        previous_close = df.iloc[-2]['Close'] # Assuming previous day's close is needed for gap

        # Calculate gap-down percentage
        gap_down_percent = (previous_close - latest_data['Open']) / previous_close if previous_close else 0

        # Check for gap-down
        if gap_down_percent > self.gap_threshold_percent:
            # Check for above-average volume (assuming 'Volume' and 'Avg_Volume' are present)
            # Placeholder: In a real scenario, you'd calculate average volume over a period (e.g., 20 days)
            avg_volume = df['Volume'].rolling(window=20).mean().iloc[-1] if len(df) >= 20 else None
            
            if pd.isna(avg_volume) or latest_data['Volume'] > (avg_volume * self.volume_spike_multiplier):
                return True, f"Bearish gap-down of {gap_down_percent:.2%} with strong volume."
            else:
                return False, f"Bearish gap-down of {gap_down_percent:.2%}, but volume not strong enough."
        return False, "No significant bearish gap-down."

    def is_downtrend_with_volume(self, df):
        """
        Checks for price trading below VWAP/EMAs with lower highs/lows and increasing sell volume.
        Assumes 'VWAP', 'EMA_20', 'EMA_50' are available, and 'High', 'Low', 'Close', 'Open', 'Volume'.
        """
        if df.empty or len(df) < 3: # Need at least 3 candles for lower highs/lows
            return False, "Insufficient data for downtrend analysis."

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        prev_prev = df.iloc[-3]

        reasons = []

        # Price below VWAP and/or EMAs
        if 'VWAP' in latest and latest['Close'] < latest['VWAP']:
            reasons.append("Price below VWAP.")
        if 'EMA_20' in latest and latest['Close'] < latest['EMA_20']:
            reasons.append("Price below EMA_20.")
        if 'EMA_50' in latest and latest['Close'] < latest['EMA_50']:
            reasons.append("Price below EMA_50.")

        # Lower highs and lower lows (over last few candles)
        # Simplified: check last 3 candles
        lower_highs = latest['High'] < prev['High'] < prev_prev['High']
        lower_lows = latest['Low'] < prev['Low'] < prev_prev['Low']

        if lower_highs and lower_lows:
            reasons.append("Forming lower highs and lower lows.")

        # Increasing sell volume (red candle with higher volume than previous red candle)
        # This is a simplification; a more robust check would involve comparing current volume with average selling volume
        if latest['Close'] < latest['Open'] and prev['Close'] < prev['Open'] and latest['Volume'] > prev['Volume']:
            reasons.append("Increasing sell volume on red candles.")

        if reasons and len(reasons) >= 2: # Require at least two bearish indicators
            return True, "Downtrend confirmed: " + " ".join(reasons)
        return False, "No clear downtrend with volume."

    def is_support_breakdown(self, df):
        """
        Checks for a breakdown below a recent support level or previous day low with strong volume.
        Assumes 'Support' and 'Previous_Low' are available.
        """
        if df.empty or len(df) < 2:
            return False, "Insufficient data for support breakdown analysis."

        latest_data = df.iloc[-1]
        previous_low = df.iloc[-2]['Low'] # Assuming previous day's low is needed for breakdown

        reasons = []

        # Breakdown below recent support
        if 'Support' in latest_data and latest_data['Close'] < latest_data['Support']:
            reasons.append(f"Price broke below recent support ({latest_data['Support']:.2f}).")
        
        # Breakdown below previous day's low
        if latest_data['Close'] < previous_low:
            reasons.append(f"Price broke below previous day's low ({previous_low:.2f}).")

        # Check for strong volume on breakdown
        if reasons:
            avg_volume = df['Volume'].rolling(window=20).mean().iloc[-1] if len(df) >= 20 else None
            if pd.isna(avg_volume) or latest_data['Volume'] > (avg_volume * self.volume_spike_multiplier):
                reasons.append("Breakdown confirmed with strong volume.")
                return True, " ".join(reasons)
            else:
                return False, "Breakdown detected, but volume not strong enough."
        return False, "No significant support breakdown."

    def decide_signal_type(self, bullish_score, bearish_score, sentiment_confidence):
        """
        Decides whether to generate a CALL, PUT, or no signal based on bullish/bearish scores and sentiment.
        """
        # Threshold for a signal to be considered strong enough
        SIGNAL_STRENGTH_THRESHOLD = 0.4 # Can be adjusted
        SCORE_DIFFERENCE_THRESHOLD = 0.1 # How much one score must exceed the other

        # If positive sentiment and bullish score is strong and outweighs bearish score
        if sentiment_confidence >= self.sentiment_threshold and bullish_score > SIGNAL_STRENGTH_THRESHOLD and \
           bullish_score > bearish_score + SCORE_DIFFERENCE_THRESHOLD:
            return "CALL"
        
        # If negative sentiment and bearish score is strong and outweighs bullish score
        elif sentiment_confidence >= self.sentiment_threshold and bearish_score > SIGNAL_STRENGTH_THRESHOLD and \
             bearish_score > bullish_score + SCORE_DIFFERENCE_THRESHOLD:
            return "PUT"
        
        return None

    def generate_btst_signals(self, all_stocks_data, news_df, side: str = "BOTH"):
        """
        Generates Buy Today, Sell Tomorrow (BTST) trade signals for a specified side (CALL/PUT/BOTH).
        Signals are generated if news sentiment is strong and technicals confirm momentum.
        """
        logging.info(f"Generating BTST signals for side: {side}...")
        btst_signals = []

        for symbol, df in all_stocks_data.items():
            if df.empty or len(df) < 20: # Ensure enough data for indicators
                logging.warning(f"Skipping BTST signal generation for {symbol}: Insufficient data.")
                continue

            latest_data = df.iloc[-1]
            previous_data = df.iloc[-2]

            # Add 'Prev_Close', 'Prev_Volume' and 'RSI' to latest_data for row-wise condition checks
            # Assuming these are calculated and available in the DataFrame.
            # If not, main.py's indicator calculation needs to be updated.
            latest_data['Prev_Close'] = previous_data['Close']
            latest_data['Prev_Volume'] = previous_data['Volume']
            # If RSI is not directly in latest_data, ensure it's calculated in indicator_calculator.py
            if 'RSI' not in latest_data:
                # Placeholder: In a real scenario, ensure RSI is calculated in indicator_calculator.py
                latest_data['RSI'] = 50.0 # Neutral default if not found (should be calculated)
            
            # 1. Assess Sentiment
            sentiment_result = self._assess_sentiment_for_stock(symbol, news_df)
            overall_sentiment = sentiment_result['overall_sentiment']
            sentiment_confidence = sentiment_result['confidence']

            # 2. Assess Technical Momentum (using new scoring functions)
            bullish_score, bullish_reasons = self._compute_bullish_score(df)
            bearish_score, bearish_reasons = self._compute_bearish_score(df)

            signal = None
            rationale = []

            # Decision logic for BTST signals
            if side == "CALL" or side == "BOTH":
                # If bullish factors significantly outweigh bearish factors AND bullish conditions are met
                if overall_sentiment == "positive" and sentiment_confidence >= self.sentiment_threshold and \
                   bullish_score > self.momentum_strength_threshold and bullish_score > bearish_score + 0.1: # Add a buffer
                    
                    entry_price = latest_data['Close']
                    target_price = entry_price * (1 + self.btst_target_percentage)
                    
                    rationale.append(f"Strong positive news sentiment (confidence: {sentiment_confidence:.2f}).")
                    rationale.append(f"Bullish technical score: {bullish_score:.2f}.")
                    rationale.extend(bullish_reasons)
                    rationale.append(f"Bearish technical score: {bearish_score:.2f}.")

                    # Determine option details for BTST CALL
                    expiry_date = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d') # Nearest weekly expiry (simplified)
                    strike_price = self.select_option_strike(latest_data['Close'], "CALL", risk_profile="OTM") # Default OTM for CALL BTST
                    option_symbol = self._build_option_symbol(symbol, expiry_date, strike_price, "CE")

                    signal = {
                        "timestamp": datetime.now().isoformat(),
                        "symbol": symbol,
                        "trade_type": "BTST",
                        "action": "BUY",
                        "signal_type": "BTST_CALL", # Explicitly set signal type
                        "entry_price": round(entry_price, 2),
                        "target_price": round(target_price, 2),
                        "option_type": "CE", # Option type for BTST CALL
                        "strike_price": strike_price,
                        "expiry_date": expiry_date,
                        "option_symbol": option_symbol,
                        "rationale": " ".join(rationale),
                        "confidence_score": (sentiment_confidence + bullish_score) / 2
                    }
                    # Adjust confidence score using the learning module if available
                    if self.learning_module:
                        signal['confidence_score'] = self.learning_module.get_adjusted_confidence(signal)
                    btst_signals.append(signal)
                    logging.info(f"Generated BTST BUY CALL signal for {symbol}.")
                else:
                    logging.info(f"No BTST CALL signal for {symbol}: Sentiment ({overall_sentiment}, {sentiment_confidence:.2f}), Bullish Score ({bullish_score:.2f}), Bearish Score ({bearish_score:.2f}) not strong or clear enough, or bullish conditions not met.")

            if side == "PUT" or side == "BOTH":
                # If bearish factors significantly outweigh bullish factors AND bearish conditions are met
                if self.btst_bearish_conditions(latest_data) and \
                   overall_sentiment == "negative" and sentiment_confidence >= self.sentiment_threshold and \
                   bearish_score > self.bearish_score_threshold and bearish_score > bullish_score + 0.1: # Add a buffer
                    
                    entry_price = latest_data['Close']
                    target_price = entry_price * (1 - self.btst_target_percentage) # Target lower for sell
                    
                    rationale = [] # Reset rationale for PUT signal
                    rationale.append(f"Strong negative news sentiment (confidence: {sentiment_confidence:.2f}).")
                    rationale.append(f"Bearish technical score: {bearish_score:.2f}.")
                    rationale.extend(bearish_reasons)
                    rationale.append(f"Bullish technical score: {bullish_score:.2f}.") # Include for transparency

                    # Determine option details for BTST PUT
                    expiry_date = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d') # Nearest weekly expiry (simplified)
                    strike_price = self.select_option_strike(latest_data['Close'], "PUT", risk_profile="ATM") # Default ATM for PUT BTST
                    option_symbol = self._build_option_symbol(symbol, expiry_date, strike_price, "PE")

                    signal = {
                        "timestamp": datetime.now().isoformat(),
                        "symbol": symbol,
                        "trade_type": "BTST",
                        "action": "SELL", # Or "BUY PUT" - keeping "SELL" for consistency with shorting
                        "signal_type": "BTST_PUT", # Explicitly set signal type
                        "entry_price": round(entry_price, 2),
                        "target_price": round(target_price, 2),
                        "option_type": "PE", # Option type for BTST PUT
                        "strike_price": strike_price,
                        "expiry_date": expiry_date,
                        "option_symbol": option_symbol,
                        "rationale": " ".join(rationale),
                        "confidence_score": (sentiment_confidence + bearish_score) / 2
                    }
                    # Adjust confidence score using the learning module if available
                    if self.learning_module:
                        signal['confidence_score'] = self.learning_module.get_adjusted_confidence(signal)
                    btst_signals.append(signal)
                    logging.info(f"Generated BTST BUY PUT signal for {symbol}.")
                else:
                    logging.info(f"No BTST PUT signal for {symbol}: Sentiment ({overall_sentiment}, {sentiment_confidence:.2f}), Bearish Score ({bearish_score:.2f}), Bullish Score ({bullish_score:.2f}) not strong or clear enough, or bearish conditions not met.")

        self.signals.extend(btst_signals)
        return btst_signals

    def select_option_strike(self, spot_price, option_type, risk_profile="ATM"):
        """
        Selects an appropriate strike price for CALL or PUT options.
        option_type: "CE" for Call, "PE" for Put
        risk_profile: "ATM", "ITM", "OTM"
        """
        # For simplicity, assuming strike prices are in multiples of 100 or 50
        # In a real scenario, you'd fetch available strikes for the instrument
        
        if option_type == "CALL":
            if risk_profile == "ITM":
                strike = spot_price - self.put_itm_delta # Example: 50 points ITM
            elif risk_profile == "OTM":
                strike = spot_price + self.put_otm_delta # Example: 50 points OTM
            else: # ATM
                strike = spot_price
            # Round to nearest sensible strike (e.g., nearest 50 or 100)
            return round(strike / 50) * 50
        
        elif option_type == "PUT":
            if risk_profile == "ITM":
                strike = spot_price + self.put_itm_delta # Example: 50 points ITM
            elif risk_profile == "OTM":
                strike = spot_price - self.put_otm_delta # Example: 50 points OTM
            else: # ATM
                strike = spot_price
            # Round to nearest sensible strike
            return round(strike / 50) * 50
        
        return None


    def generate_options_signals(self, all_stocks_data, news_df):
        """
        Generates CALL/PUT options signals for stocks with high predicted volatility at market open.
        """
        logging.info("Generating Options signals...")
        options_signals = []

        for symbol, df in all_stocks_data.items():
            if df.empty or len(df) < 14: # ATR needs at least 14 periods
                logging.warning(f"Skipping Options signal generation for {symbol}: Insufficient data.")
                continue

            latest_data = df.iloc[-1]
            
            # 1. Assess Volatility (using ATR)
            atr = latest_data['ATR']
            if pd.isna(atr):
                logging.warning(f"Skipping Options signal generation for {symbol}: ATR not available.")
                continue

            # Define "high predicted volatility" - e.g., ATR is high relative to price or a fixed threshold
            # For simplicity, let's say ATR > 2% of Close price indicates high volatility
            high_volatility_threshold_ratio = 0.01 # Lowered for testing
            is_high_volatility = atr > (latest_data['Close'] * high_volatility_threshold_ratio)

            if not is_high_volatility:
                logging.info(f"Skipping Options signal for {symbol}: Volatility (ATR: {atr:.2f}) not high enough (Close: {latest_data['Close']:.2f}).")
                continue

            # 2. Assess Sentiment for direction
            sentiment_result = self._assess_sentiment_for_stock(symbol, news_df)
            overall_sentiment = sentiment_result['overall_sentiment']
            sentiment_confidence = sentiment_result['confidence']

            # 3. Assess Technicals for direction (using new scoring functions)
            bullish_score, bullish_reasons = self._compute_bullish_score(df)
            bearish_score, bearish_reasons = self._compute_bearish_score(df)

            # 4. Decide signal type (CALL/PUT/None)
            signal_type = self.decide_signal_type(bullish_score, bearish_score, sentiment_confidence)

            signal = None
            rationale = []
            strike_price = None
            expiry_date = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d') # Nearest weekly expiry (simplified)

            if signal_type == "CALL":
                action = "BUY CALL"
                # Select strike price using the generalized function
                strike_price = self.select_option_strike(latest_data['Close'], "CALL", risk_profile="OTM") # Example: OTM for options
                
                rationale.append(f"High predicted volatility (ATR: {atr:.2f}).")
                rationale.append(f"Strong positive news sentiment (confidence: {sentiment_confidence:.2f}).")
                rationale.append(f"Bullish technical score: {bullish_score:.2f}.")
                rationale.extend(bullish_reasons)
                rationale.append(f"Bearish technical score: {bearish_score:.2f}.") # Include for transparency


                signal = {
                    "timestamp": datetime.now().isoformat(),
                    "symbol": symbol,
                    "trade_type": "OPTIONS",
                    "action": action,
                    "option_type": "CE", # Explicitly set option type
                    "strike_price": strike_price,
                    "expiry_date": expiry_date,
                    "rationale": " ".join(rationale),
                    "confidence_score": (sentiment_confidence + bullish_score + (atr / latest_data['Close'])) / 3
                }
                # Adjust confidence score using the learning module if available
                if self.learning_module:
                    signal['confidence_score'] = self.learning_module.get_adjusted_confidence(signal)
                options_signals.append(signal)
                logging.info(f"Generated OPTIONS BUY CALL signal for {symbol}.")

            elif signal_type == "PUT":
                action = "BUY PUT"
                # Select strike price using the generalized function
                strike_price = self.select_option_strike(latest_data['Close'], "PUT", risk_profile="ATM") # Example: ATM for options
                
                rationale.append(f"High predicted volatility (ATR: {atr:.2f}).")
                rationale.append(f"Strong negative news sentiment (confidence: {sentiment_confidence:.2f}).")
                rationale.append(f"Bearish technical score: {bearish_score:.2f}.")
                rationale.extend(bearish_reasons)
                rationale.append(f"Bullish technical score: {bullish_score:.2f}.") # Include for transparency


                signal = {
                    "timestamp": datetime.now().isoformat(),
                    "symbol": symbol,
                    "trade_type": "OPTIONS",
                    "action": action,
                    "option_type": "PE", # Explicitly set option type
                    "strike_price": strike_price,
                    "expiry_date": expiry_date,
                    "rationale": " ".join(rationale),
                    "confidence_score": (sentiment_confidence + bearish_score + (atr / latest_data['Close'])) / 3
                }
                # Adjust confidence score using the learning module if available
                if self.learning_module:
                    signal['confidence_score'] = self.learning_module.get_adjusted_confidence(signal)
                options_signals.append(signal)
                logging.info(f"Generated OPTIONS BUY PUT signal for {symbol}.")
            else:
                logging.info(f"No Options signal for {symbol}: Volatility high, but no clear CALL or PUT signal based on scores.")

        self.signals.extend(options_signals)
        return options_signals

    def _build_option_symbol(self, symbol, expiry_date_str, strike_price, option_type):
        """
        Builds the option symbol string (e.g., NIFTY24DEC19400CE).
        Assumes expiry_date_str is in 'YYYY-MM-DD' format.
        """
        # Convert expiry_date_str to datetime object
        expiry_date = datetime.strptime(expiry_date_str, '%Y-%m-%d')
        
        # Format month (e.g., JAN, FEB)
        month_abbr = expiry_date.strftime('%b').upper()
        
        # Format year (last two digits)
        year_abbr = expiry_date.strftime('%y')
        
        # Combine parts to form the symbol
        # Example: NIFTY24DEC19400CE
        return f"{symbol}{year_abbr}{month_abbr}{int(strike_price)}{option_type}"

    def save_signals_to_json(self, filename="../outputs/trade_signals.json"):
        """Saves all generated signals to a JSON file."""
        if self.signals:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w') as f:
                json.dump(self.signals, f, indent=4)
            logging.info(f"All generated signals saved to {filename}")
        else:
            logging.info("No signals to save.")

    def get_all_signals(self):
        """Returns all generated signals."""
        return self.signals

    def rank_signals(self, signals):
        """
        Ranks signals based on their confidence_score in descending order.
        """
        if not signals:
            return []
        # Sort signals by confidence_score in descending order
        return sorted(signals, key=lambda x: x.get('confidence_score', 0), reverse=True)

    def get_top_n_signals(self, signals, n):
        """
        Returns the top N signals after ranking.
        """
        if not signals:
            return []
        ranked_signals = self.rank_signals(signals)
        return ranked_signals[:n]

if __name__ == "__main__":
    # This block is for testing the TradeSignalGenerator in isolation.
    # In a real scenario, main.py would orchestrate data fetching, news scraping,
    # indicator calculation, and then call this class.

    # Dummy Data for testing
    # Create a dummy DataFrame for a stock
    dummy_df_data = {
        'Open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129],
        'High': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132],
        'Low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128],
        'Close': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131],
        'Volume': [100000, 110000, 120000, 130000, 140000, 150000, 160000, 170000, 180000, 190000, 200000, 210000, 220000, 230000, 240000, 250000, 260000, 270000, 500000, 290000, 300000, 310000, 320000, 330000, 340000, 350000, 360000, 370000, 380000, 390000]
    }
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    dummy_df = pd.DataFrame(dummy_df_data, index=dates)

    # Add indicators (simplified for testing)
    dummy_df['SMA_20'] = dummy_df['Close'].rolling(window=20).mean()
    dummy_df['SMA_50'] = dummy_df['Close'].rolling(window=20).mean() # Using 20 for simplicity in dummy data
    dummy_df['SMA_200'] = dummy_df['Close'].rolling(window=20).mean() # Using 20 for simplicity in dummy data
    dummy_df['ATR'] = 2.5 # Dummy ATR
    dummy_df['Volume_Spike'] = False
    dummy_df.loc[dummy_df.index[-1], 'Volume_Spike'] = True # Simulate a spike
    dummy_df['Support'] = dummy_df['Low'].rolling(window=10, center=True).min()
    dummy_df['Resistance'] = dummy_df['High'].rolling(window=10, center=True).max()
    dummy_df['Breakout'] = False
    dummy_df.loc[dummy_df.index[-1], 'Breakout'] = True # Simulate a breakout
    dummy_df['Breakdown'] = False

    all_stocks_data_dummy = {'TESTSTOCK.NS': dummy_df}

    # Dummy News Data
    dummy_news_data = [
        {'symbol': 'TESTSTOCK.NS', 'source': 'Google News', 'headline': 'TESTSTOCK announces record profits, bullish outlook.', 'published_at': datetime.now(), 'sentiment': 'positive', 'confidence': 0.85},
        {'symbol': 'TESTSTOCK.NS', 'source': 'Google News', 'headline': 'Analyst upgrades TESTSTOCK rating.', 'published_at': datetime.now(), 'sentiment': 'positive', 'confidence': 0.78},
        {'symbol': 'ANOTHER.NS', 'source': 'Google News', 'headline': 'Another stock faces headwinds.', 'published_at': datetime.now(), 'sentiment': 'negative', 'confidence': 0.90},
    ]
    dummy_news_df = pd.DataFrame(dummy_news_data)

    # Initialize and generate signals
    signal_generator = TradeSignalGenerator()
    btst_signals = signal_generator.generate_btst_signals(all_stocks_data_dummy, dummy_news_df)
    options_signals = signal_generator.generate_options_signals(all_stocks_data_dummy, dummy_news_df)

    print("\n--- Generated BTST Signals ---")
    for signal in btst_signals:
        print(json.dumps(signal, indent=4))

    print("\n--- Generated Options Signals ---")
    for signal in options_signals:
        print(json.dumps(signal, indent=4))

    signal_generator.save_signals_to_json("../outputs/test_trade_signals.json")