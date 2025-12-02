# ai_trading_signal_tool/trade_signal_generator.py

import pandas as pd
from datetime import datetime, timedelta
import json
import os
import logging
from .self_learning_module import SelfLearningModule

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TradeSignalGenerator:
    def __init__(self, sentiment_threshold=0.5, momentum_strength_threshold=0.3, btst_target_percentage=0.01, learning_module=None):
        self.sentiment_threshold = sentiment_threshold
        self.momentum_strength_threshold = momentum_strength_threshold
        self.btst_target_percentage = btst_target_percentage
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

    def _assess_technical_momentum(self, df):
        """
        Assesses technical momentum based on various indicators.
        Returns a score between -1 (strong bearish) and 1 (strong bullish).
        """
        if df.empty:
            return 0.0

        latest_data = df.iloc[-1]
        momentum_score = 0

        # Price vs SMAs
        if latest_data['Close'] > latest_data['SMA_20']:
            momentum_score += 0.2
        if latest_data['Close'] > latest_data['SMA_50']:
            momentum_score += 0.2
        if latest_data['SMA_20'] > latest_data['SMA_50']:
            momentum_score += 0.2 # Short term trend up

        # Breakouts/Breakdowns
        if latest_data['Breakout']:
            momentum_score += 0.2
        if latest_data['Breakdown']:
            momentum_score -= 0.2

        # Volume Spike (can be bullish or bearish depending on context, here we'll treat it as confirming existing trend)
        if latest_data['Volume_Spike']:
            if latest_data['Close'] > latest_data['Open']: # Green candle with spike
                momentum_score += 0.1
            else: # Red candle with spike
                momentum_score -= 0.1
        
        # Price relative to Support/Resistance
        if latest_data['Close'] > latest_data['Resistance']: # Above resistance
            momentum_score += 0.1
        elif latest_data['Close'] < latest_data['Support']: # Below support
            momentum_score -= 0.1

        # Normalize score to be between -1 and 1 (approx)
        return min(1.0, max(-1.0, momentum_score / 1.0)) # Max possible score is around 1.0, min around -0.4

    def generate_btst_signals(self, all_stocks_data, news_df):
        """
        Generates Buy Today, Sell Tomorrow (BTST) trade signals.
        Signals are generated if news sentiment is strong and technicals confirm momentum.
        """
        logging.info("Generating BTST signals...")
        btst_signals = []

        for symbol, df in all_stocks_data.items():
            if df.empty or len(df) < 20: # Ensure enough data for indicators
                logging.warning(f"Skipping BTST signal generation for {symbol}: Insufficient data.")
                continue

            latest_data = df.iloc[-1]
            
            # 1. Assess Sentiment
            sentiment_result = self._assess_sentiment_for_stock(symbol, news_df)
            overall_sentiment = sentiment_result['overall_sentiment']
            sentiment_confidence = sentiment_result['confidence']

            # 2. Assess Technical Momentum
            momentum_score = self._assess_technical_momentum(df)

            signal = None
            rationale = []

            # Check for BUY signal
            if overall_sentiment == "positive" and sentiment_confidence >= self.sentiment_threshold and momentum_score >= self.momentum_strength_threshold:
                entry_price = latest_data['Close']
                target_price = entry_price * (1 + self.btst_target_percentage)
                
                rationale.append(f"Strong positive news sentiment (confidence: {sentiment_confidence:.2f}).")
                rationale.append(f"Bullish technical momentum (score: {momentum_score:.2f}).")
                if latest_data['Close'] > latest_data['SMA_20']: rationale.append("Price above SMA_20.")
                if latest_data['SMA_20'] > latest_data['SMA_50']: rationale.append("SMA_20 above SMA_50 (uptrend).")
                if latest_data['Breakout']: rationale.append("Breakout detected.")
                if latest_data['Volume_Spike']: rationale.append("Volume spike detected.")

                signal = {
                    "timestamp": datetime.now().isoformat(),
                    "symbol": symbol,
                    "trade_type": "BTST",
                    "action": "BUY",
                    "entry_price": round(entry_price, 2),
                    "target_price": round(target_price, 2),
                    "rationale": " ".join(rationale),
                    "confidence_score": (sentiment_confidence + momentum_score) / 2 # Simple average for now
                }
                # Adjust confidence score using the learning module if available
                if self.learning_module:
                    signal['confidence_score'] = self.learning_module.get_adjusted_confidence(signal)
                btst_signals.append(signal)
                logging.info(f"Generated BTST BUY signal for {symbol}.")

            # Check for SELL signal (if sentiment is negative and technicals confirm breakdown/downtrend)
            elif overall_sentiment == "negative" and sentiment_confidence >= self.sentiment_threshold and momentum_score <= -self.momentum_strength_threshold:
                entry_price = latest_data['Close']
                target_price = entry_price * (1 - self.btst_target_percentage) # Target lower for sell
                
                rationale.append(f"Strong negative news sentiment (confidence: {sentiment_confidence:.2f}).")
                rationale.append(f"Bearish technical momentum (score: {momentum_score:.2f}).")
                if latest_data['Close'] < latest_data['SMA_20']: rationale.append("Price below SMA_20.")
                if latest_data['SMA_20'] < latest_data['SMA_50']: rationale.append("SMA_20 below SMA_50 (downtrend).")
                if latest_data['Breakdown']: rationale.append("Breakdown detected.")
                if latest_data['Volume_Spike']: rationale.append("Volume spike detected.")

                signal = {
                    "timestamp": datetime.now().isoformat(),
                    "symbol": symbol,
                    "trade_type": "BTST",
                    "action": "SELL",
                    "entry_price": round(entry_price, 2),
                    "target_price": round(target_price, 2),
                    "rationale": " ".join(rationale),
                    "confidence_score": (sentiment_confidence + abs(momentum_score)) / 2
                }
                # Adjust confidence score using the learning module if available
                if self.learning_module:
                    signal['confidence_score'] = self.learning_module.get_adjusted_confidence(signal)
                btst_signals.append(signal)
                logging.info(f"Generated BTST SELL signal for {symbol}.")
            else:
                logging.info(f"No BTST signal for {symbol}: Sentiment ({overall_sentiment}, {sentiment_confidence:.2f}) or Momentum ({momentum_score:.2f}) not strong enough.")

        self.signals.extend(btst_signals)
        return btst_signals

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

            # 3. Assess Technicals for direction (similar to BTST, but perhaps more focused on immediate trend)
            momentum_score = self._assess_technical_momentum(df)

            signal = None
            rationale = []
            strike_price = None
            expiry_date = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d') # Approx 1 month out

            # Generate CALL signal
            if overall_sentiment == "positive" and sentiment_confidence >= self.sentiment_threshold and momentum_score >= self.momentum_strength_threshold:
                action = "BUY CALL"
                # Suggest strike price slightly OTM or ATM
                strike_price = round(latest_data['Close'] + atr, 0) # Example: Close + 1 ATR
                
                rationale.append(f"High predicted volatility (ATR: {atr:.2f}).")
                rationale.append(f"Strong positive news sentiment (confidence: {sentiment_confidence:.2f}).")
                rationale.append(f"Bullish technical momentum (score: {momentum_score:.2f}).")
                if latest_data['Close'] > latest_data['SMA_20']: rationale.append("Price above SMA_20.")

                signal = {
                    "timestamp": datetime.now().isoformat(),
                    "symbol": symbol,
                    "trade_type": "OPTIONS",
                    "action": action,
                    "strike_price": strike_price,
                    "expiry_date": expiry_date,
                    "rationale": " ".join(rationale),
                    "confidence_score": (sentiment_confidence + momentum_score + (atr / latest_data['Close'])) / 3 # Incorporate volatility
                }
                # Adjust confidence score using the learning module if available
                if self.learning_module:
                    signal['confidence_score'] = self.learning_module.get_adjusted_confidence(signal)
                options_signals.append(signal)
                logging.info(f"Generated OPTIONS BUY CALL signal for {symbol}.")

            # Generate PUT signal
            elif overall_sentiment == "negative" and sentiment_confidence >= self.sentiment_threshold and momentum_score <= -self.momentum_strength_threshold:
                action = "BUY PUT"
                # Suggest strike price slightly OTM or ATM
                strike_price = round(latest_data['Close'] - atr, 0) # Example: Close - 1 ATR
                
                rationale.append(f"High predicted volatility (ATR: {atr:.2f}).")
                rationale.append(f"Strong negative news sentiment (confidence: {sentiment_confidence:.2f}).")
                rationale.append(f"Bearish technical momentum (score: {momentum_score:.2f}).")
                if latest_data['Close'] < latest_data['SMA_20']: rationale.append("Price below SMA_20.")

                signal = {
                    "timestamp": datetime.now().isoformat(),
                    "symbol": symbol,
                    "trade_type": "OPTIONS",
                    "action": action,
                    "strike_price": strike_price,
                    "expiry_date": expiry_date,
                    "rationale": " ".join(rationale),
                    "confidence_score": (sentiment_confidence + abs(momentum_score) + (atr / latest_data['Close'])) / 3
                }
                # Adjust confidence score using the learning module if available
                if self.learning_module:
                    signal['confidence_score'] = self.learning_module.get_adjusted_confidence(signal)
                options_signals.append(signal)
                logging.info(f"Generated OPTIONS BUY PUT signal for {symbol}.")
            else:
                logging.info(f"No Options signal for {symbol}: Volatility high, but Sentiment ({overall_sentiment}, {sentiment_confidence:.2f}) or Momentum ({momentum_score:.2f}) not strong enough for direction.")

        self.signals.extend(options_signals)
        return options_signals

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