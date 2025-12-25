import os
from dotenv import load_dotenv
import google.generativeai as genai
import pandas as pd
import json

# Load environment variables
load_dotenv()

class AIChatbot:
    def __init__(self, db_manager, market_manager, signal_validator, learning_module, watchlist_manager, news_scraper, data_fetcher_func, indicator_calculator_funcs, trade_signal_generator):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found. Please set the environment variable.")
        
        genai.configure(api_key=api_key)

        log_file_path = os.path.join(os.path.dirname(__file__), "chatbot_debug_log.txt")

        # Log available models to a file for debugging, as Streamlit might suppress console output
        with open(log_file_path, "w") as f:
            f.write("Available Gemini Models:\n")
            for m in genai.list_models():
                if "generateContent" in m.supported_generation_methods:
                    f.write(f"- {m.name}\n")

        # Try to use a more widely available model or fallback
        try:
            self.model = genai.GenerativeModel('gemini-2.5-flash') # Recommended fallback
            with open(log_file_path, "a") as f:
                f.write(f"Attempting to use Gemini Model: gemini-1.5-flash\n")
        except Exception as e:
            with open(log_file_path, "a") as f:
                f.write(f"Error with gemini-1.5-flash: {e}. Falling back to gemini-pro.\n")
            self.model = genai.GenerativeModel('gemini-pro') # Keep as fallback if 1.5-flash fails
            
        with open(log_file_path, "a") as f:
            f.write(f"Using Gemini Model: {self.model.model_name}\n")
        self.chat_session = self.model.start_chat(history=[
            {"role": "user", "parts": ["You are an AI trading assistant. Provide information about trading signals, market updates, and stock suggestions based on the provided tools and data. Explain your reasoning clearly and concisely."]},
            {"role": "model", "parts": ["Understood. I will act as an AI trading assistant, providing insights based on available data and tools, explaining my reasoning for signals, market interpretations, and stock suggestions."]}
        ])

        self.db_manager = db_manager
        self.market_manager = market_manager
        self.signal_validator = signal_validator
        self.learning_module = learning_module
        self.watchlist_manager = watchlist_manager
        self.news_scraper = news_scraper
        self.data_fetcher_func = data_fetcher_func
        self.indicator_calculator_funcs = indicator_calculator_funcs
        self.trade_signal_generator = trade_signal_generator

        # Store a reference to all_stocks_data and news_df from main.py's session state
        self.all_stocks_data = {}
        self.news_df = pd.DataFrame()

    def update_data_references(self, all_stocks_data, news_df):
        """Updates the chatbot's references to the latest stock data and news."""
        self.all_stocks_data = all_stocks_data
        self.news_df = news_df

    def _get_market_status(self):
        current_time = self.market_manager.get_current_market_time()
        if self.market_manager.is_holiday(current_time.date()):
            return f"Market is closed today ({current_time.strftime('%Y-%m-%d')}) due to a holiday."
        elif not self.market_manager.is_market_open():
            return f"Market is currently closed. Current time: {current_time.strftime('%H:%M IST')}."
        else:
            return f"Market is open. Current time: {current_time.strftime('%H:%M IST')}."

    def _get_latest_news_sentiment(self):
        if not self.news_df.empty:
            return self.news_df.to_json(orient="records")
        return "No news headlines collected yet."
    
    def _get_overall_market_sentiment(self):
        if self.news_df.empty:
            return "No news collected to determine overall market sentiment."
        
        positive_count = self.news_df[self.news_df['sentiment'] == 'positive'].shape[0]
        negative_count = self.news_df[self.news_df['sentiment'] == 'negative'].shape[0]
        neutral_count = self.news_df[self.news_df['sentiment'] == 'neutral'].shape[0]
        total_news = self.news_df.shape[0]

        if total_news == 0:
            return "No news collected to determine overall market sentiment."

        sentiment_summary = f"Overall Market Sentiment (from {total_news} articles):\n"
        sentiment_summary += f"  - Positive: {positive_count} articles\n"
        sentiment_summary += f"  - Negative: {negative_count} articles\n"
        sentiment_summary += f"  - Neutral: {neutral_count} articles\n"

        if positive_count > negative_count and positive_count > neutral_count:
            sentiment_summary += "Overall sentiment appears to be **positive**."
        elif negative_count > positive_count and negative_count > neutral_count:
            sentiment_summary += "Overall sentiment appears to be **negative**."
        else:
            sentiment_summary += "Overall sentiment appears to be **neutral** or mixed."
        
        return sentiment_summary

    def _get_all_watchlist_tickers(self):
        # Use the watchlist_manager instance to get all tickers
        return self.watchlist_manager.get_all_watchlist_tickers()


    def _get_stock_indicators(self, symbol):
        if symbol in self.all_stocks_data and not self.all_stocks_data[symbol].empty:
            latest_data = self.all_stocks_data[symbol].iloc[-1]
            # Convert to a dict for easier LLM consumption
            return latest_data[['Close', 'SMA_20', 'SMA_50', 'SMA_200', 'ATR', 'Volume_Spike', 'Support', 'Resistance', 'Breakout', 'Breakdown']].to_json()
        return f"No indicator data available for {symbol}."

    def _get_current_signals(self, signal_type=None, min_confidence=0.0, max_risk=1.0):
        # Fetch validated signals from the database, including rationale
        signals = self.db_manager.get_all_validated_signals()
        
        # Apply filters
        filtered_signals = []
        for s in signals:
            confidence = s.get('confidence_score', 0.0)
            risk = s.get('risk_level', 0.5) # Assuming a default risk_level if not present

            if confidence >= min_confidence and risk <= max_risk:
                filtered_signals.append(s)
        
        signals = filtered_signals

        if signal_type:
            signals = [s for s in signals if s.get('trade_type', '').upper() == signal_type.upper() or s.get('signal_type', '').upper().startswith(signal_type.upper())]
        
        
        if signals:
            formatted_signals = []
            for s in signals:
                if s['trade_type'] == 'BTST':
                    formatted_signals.append(f"BTST {s['action']} | {s['symbol']} | Entry: {s['entry_price']} | Target: {s['target_price']} | Confidence: {s['confidence_score']:.2f} | Rationale: {s['rationale']}")
                elif s['trade_type'] == 'OPTIONS':
                    formatted_signals.append(f"OPTIONS {s['action']} | {s['symbol']} | Strike: {s['strike_price']} | Expiry: {s['expiry_date']} | Confidence: {s['confidence_score']:.2f} | Rationale: {s['rationale']}")
                elif s['trade_type'] == 'INTRADAY':
                    formatted_signals.append(f"INTRADAY {s['action']} | {s['symbol']} | Entry: {s['entry_price']} | Target: {s['target_price']} | Confidence: {s['confidence_score']:.2f} | Rationale: {s['rationale']}")
            return "\n".join(formatted_signals)
        return "No current signals available."

    def _get_top_stock_suggestions(self, num_suggestions=5):
        # This will require generating new signals or accessing previously generated (ranked) signals
        # For this integration, we will assume main.py has already run and populated session state with ranked signals
        # We need to access these from the main application's session state via the chatbot instance
        
        # Since the chatbot itself doesn't trigger signal generation, but consumes what's available
        # from the main app's last run, we should try to get the ranked signals directly.
        # However, the current AIChatbot doesn't have direct access to Streamlit's session_state.
        # This means we'd need to pass the ranked signals from main.py's session_state to this method
        # or have the signal_generator generate them on demand.

        # For now, let's mock this by asking the trade_signal_generator to give its top signals
        # This would require trade_signal_generator to have access to all_stocks_data and news_df
        # which is already passed in the constructor. We can call its methods directly.

        # A more robust solution might involve passing st.session_state.ranked_btst_signals and 
        # st.session_state.ranked_options_signals directly to this method from main.py

        # To avoid re-running heavy computations, the main app should store these and the chatbot
        # should primarily retrieve from there. Since we don't have direct `st` access here in the class,
        # we need to simulate. Let's get them from the trade_signal_generator for now.
        # This is a simplification assuming the trade_signal_generator can produce ranked signals
        # without needing to re-run full analysis within the chatbot context.
        
        # Assuming main.py would call update_data_references before any suggestion query
        # and that trade_signal_generator can work with current all_stocks_data and news_df
        
        if not self.all_stocks_data or self.news_df.empty:
            return "Cannot provide stock suggestions: Market data or news not available. Please run analysis first."
        
        # Generate BTST and Options Signals if not already available or if a fresh run is implied
        # This is a critical decision point: should the chatbot re-generate signals or only access existing ones?
        # For 'suggestions', it implies fresh, but for 'explain signals', existing is fine.
        # Let's assume for 'suggestions', we call the generator with current data.
        btst_signals = self.trade_signal_generator.generate_btst_signals(self.all_stocks_data, self.news_df, side="BOTH")
        options_signals = self.trade_signal_generator.generate_options_signals(self.all_stocks_data, self.news_df)

        ranked_btst = self.trade_signal_generator.get_top_n_signals(btst_signals, num_suggestions)
        ranked_options = self.trade_signal_generator.get_top_n_signals(options_signals, num_suggestions)

        suggestions_text = []
        if ranked_btst:
            suggestions_text.append("Top BTST Suggestions:")
            for signal in ranked_btst:
                suggestions_text.append(f"  - {signal['action']} {signal['symbol']} at {signal['entry_price']:.2f} (Confidence: {signal['confidence_score']:.2f}). Rationale: {signal['rationale']}")
        
        if ranked_options:
            suggestions_text.append("\nTop Options Suggestions:")
            for signal in ranked_options:
                suggestions_text.append(f"  - {signal['action']} {signal['symbol']} Strike: {signal['strike_price']:.2f} Expiry: {signal['expiry_date']} (Confidence: {signal['confidence_score']:.2f}). Rationale: {signal['rationale']}")

        return "\n".join(suggestions_text) if suggestions_text else "No top stock suggestions available at this time."

    def chat(self, user_query: str) -> str:
        # Simple RAG-like approach: provide context based on keywords
        context = ""
        
        # Prioritize queries that need fresh data or specific actions
        if "run analysis" in user_query.lower() or "generate signals" in user_query.lower() or "latest data" in user_query.lower():
            # In a real Streamlit app, this would trigger the main run_analysis function.
            # Here, we can only instruct the user or provide a dummy response.
            return "Please click the 'Run Analysis Now' button in the sidebar to fetch the latest data and generate signals."

        if "market status" in user_query.lower():
            context += f"Market Status: {self._get_market_status()}\n"
        
        if "overall market sentiment" in user_query.lower():
            context += f"Overall Market Sentiment: {self._get_overall_market_sentiment()}\n"
        elif "latest news" in user_query.lower() or "news sentiment" in user_query.lower():
            context += f"Latest News Headlines and Sentiment: {self._get_latest_news_sentiment()}\n"
        
        if "watchlist" in user_query.lower() or "stocks i follow" in user_query.lower():
            context += f"Watchlist Tickers: {', '.join(self._get_all_watchlist_tickers())}\n"
        
        if "indicators for" in user_query.lower():
            parts = user_query.lower().split("indicators for")
            if len(parts) > 1:
                symbol = parts[1].strip().split(' ')[0].upper() # Get first word after "indicators for"
                context += f"Technical Indicators for {symbol}: {self._get_stock_indicators(symbol)}\n"
        
        if "explain signals for" in user_query.lower():
            parts = user_query.lower().split("explain signals for")
            if len(parts) > 1:
                symbol = parts[1].strip().split(' ')[0].upper() 
                # Fetch signals for that specific symbol and include rationale
                all_signals = self.db_manager.get_all_validated_signals()
                symbol_signals = [s for s in all_signals if s.get('symbol') == symbol]
                if symbol_signals:
                    explanation = f"Signals for {symbol}:\n"
                    for sig in symbol_signals:
                        explanation += f"  - Type: {sig.get('signal_type', sig.get('trade_type'))}, Action: {sig.get('action')}, Entry: {sig.get('entry_price')}, Target: {sig.get('target_price', sig.get('strike_price'))}, Confidence: {sig.get('confidence_score')}, Rationale: {sig.get('rationale')}.\n"
                    context += explanation
                else:
                    context += f"No signals found for {symbol}.\n"
        elif "current signals" in user_query.lower() or "latest signals" in user_query.lower() or "intraday signals" in user_query.lower() or "options signals" in user_query.lower():
            signal_type = None
            if "btst" in user_query.lower():
                signal_type = "BTST"
            elif "options" in user_query.lower():
                signal_type = "OPTIONS"
            elif "intraday" in user_query.lower():
                signal_type = "INTRADAY"
            min_confidence = 0.0
            max_risk = 1.0

            if "high conviction only" in user_query.lower():
                min_confidence = 0.8
            if "low risk only" in user_query.lower():
                max_risk = 0.3 # Assuming 0.3 as a threshold for low risk

            context += f"Current Signals ({signal_type if signal_type else 'All'}, Min Confidence: {min_confidence}, Max Risk: {max_risk}):\n{self._get_current_signals(signal_type, min_confidence, max_risk)}\n"
        
        if "stock suggestions" in user_query.lower() or "recommendations" in user_query.lower():
            context += f"Stock Suggestions:\n{self._get_top_stock_suggestions(num_suggestions=5)}\n"

        full_query = f"Context: {context}\nUser Query: {user_query}" if context else user_query

        try:
            response = self.chat_session.send_message(full_query)
            return response.text
        except Exception as e:
            return f"Error communicating with AI: {e}"

if __name__ == "__main__":
    # In the full application, these would be instantiated by main.py
    # Mock implementations for standalone testing
    class MockDBManager:
        def get_all_validated_signals(self):
            return [
                {"symbol": "MOCK.NS", "trade_type": "BTST", "signal_type": "BTST_CALL", "action": "BUY", "entry_price": 100.0, "target_price": 101.0, "confidence_score": 0.85, "risk_level": 0.2, "rationale": "Price above SMA_20 and positive news."},
                {"symbol": "TEST.NS", "trade_type": "OPTIONS", "action": "BUY PUT", "strike_price": 50.0, "expiry_date": "2025-01-01", "confidence_score": 0.70, "risk_level": 0.6, "rationale": "High volatility and negative sentiment."},
                {"symbol": "INTRADAY.NS", "trade_type": "INTRADAY", "signal_type": "INTRADAY_BUY", "action": "BUY", "entry_price": 200.0, "target_price": 202.0, "confidence_score": 0.92, "risk_level": 0.15, "rationale": "Strong intraday momentum."}            ]

    class MockMarketTimingManager:
        def get_current_market_time(self):
            from datetime import datetime
            return datetime.now()
        def is_holiday(self, date):
            return False
        def is_market_open(self):
            return True

    class MockWatchlistManager:
        def get_all_watchlist_tickers(self):
            return ["MOCK.NS", "TEST.NS"]
        @property
        def watchlist(self):
            return {"NIFTY50": [{"symbol": "MOCK.NS", "name": "Mock Stock"}]}

    class MockNewsScraper:
        def __init__(self):
            pass
        def get_news_as_dataframe(self):
            return pd.DataFrame([
                {'symbol': 'MOCK.NS', 'headline': 'Mock positive news', 'sentiment': 'positive', 'confidence': 0.9},
                {'symbol': 'TEST.NS', 'headline': 'Mock negative news', 'sentiment': 'negative', 'confidence': 0.8},
                {'symbol': 'GENERAL', 'headline': 'Market report', 'sentiment': 'neutral', 'confidence': 0.6}
            ])

    class MockTradeSignalGenerator:
        def __init__(self, learning_module=None):
            self.learning_module = learning_module
        def generate_btst_signals(self, all_stocks_data, news_df, side="BOTH"):
            return [
                {"symbol": "MOCK.NS", "trade_type": "BTST", "action": "BUY", "entry_price": 100.0, "target_price": 101.0, "confidence_score": 0.9, "rationale": "Strong bullish indicators."}
            ]
        def generate_options_signals(self, all_stocks_data, news_df):
            return [
                {"symbol": "TEST.NS", "trade_type": "OPTIONS", "action": "BUY CALL", "strike_price": 50.0, "expiry_date": "2025-01-01", "confidence_score": 0.8, "rationale": "High volatility and positive outlook."}
            ]
        def get_top_n_signals(self, signals, n):
            return sorted(signals, key=lambda x: x.get('confidence_score', 0), reverse=True)[:n]


    mock_db_manager = MockDBManager()
    mock_market_manager = MockMarketTimingManager()
    mock_watchlist_manager = MockWatchlistManager()
    mock_news_scraper = MockNewsScraper()
    mock_trade_signal_generator = MockTradeSignalGenerator()

    chatbot = AIChatbot(
        db_manager=mock_db_manager,
        market_manager=mock_market_manager,
        signal_validator=None, 
        learning_module=None,  
        watchlist_manager=mock_watchlist_manager,
        news_scraper=mock_news_scraper,
        data_fetcher_func=None,
        indicator_calculator_funcs=None,
        trade_signal_generator=mock_trade_signal_generator
    )

    # Manually update data references for testing purposes
    chatbot.update_data_references(
        {'MOCK.NS': pd.DataFrame({
            'Close': [100.0, 101.0],
            'SMA_20': [98.0, 99.0],
            'SMA_50': [95.0, 96.0],
            'SMA_200': [90.0, 91.0],
            'ATR': [2.0, 2.1],
            'Volume_Spike': [0, 1],
            'Support': [97.0, 98.0],
            'Resistance': [102.0, 103.0],
            'Breakout': [0, 0],
            'Breakdown': [0, 0]
        }, index=pd.to_datetime(['2023-01-01', '2023-01-02']))
    },
        pd.DataFrame([
            {'symbol': 'MOCK.NS', 'headline': 'Mock positive news', 'sentiment': 'positive', 'confidence': 0.9},
            {'symbol': 'TEST.NS', 'headline': 'Mock negative news', 'sentiment': 'negative', 'confidence': 0.8},
            {'symbol': 'GENERAL', 'headline': 'Market report', 'sentiment': 'neutral', 'confidence': 0.6}
        ])
    )

    print("\n--- Chatbot Test ---")
    
    print("Chatbot: Hello, how can I help you today?")
    
    user_q1 = "What is the current market status?"
    print(f"You: {user_q1}")
    response1 = chatbot.chat(user_q1)
    print(f"Chatbot: {response1}")

    user_q2 = "Show me the current signals."
    print(f"You: {user_q2}")
    response2 = chatbot.chat(user_q2)
    print(f"Chatbot: {response2}")

    user_q3 = "What are the watchlist tickers?"
    print(f"You: {user_q3}")
    response3 = chatbot.chat(user_q3)
    print(f"Chatbot: {response3}")

    user_q4 = "Can you tell me about the indicators for MOCK.NS?"
    print(f"You: {user_q4}")
    response4 = chatbot.chat(user_q4)
    print(f"Chatbot: {response4}")

    user_q5 = "What is the overall market sentiment?"
    print(f"You: {user_q5}")
    response5 = chatbot.chat(user_q5)
    print(f"Chatbot: {response5}")

    user_q6 = "Give me some stock suggestions."
    print(f"You: {user_q6}")
    response6 = chatbot.chat(user_q6)
    print(f"Chatbot: {response6}")

    user_q7 = "Explain signals for MOCK.NS."
    print(f"You: {user_q7}")
    response7 = chatbot.chat(user_q7)
    print(f"Chatbot: {response7}")
