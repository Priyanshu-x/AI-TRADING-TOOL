# AI Trading Tool Overview

The `ai_trading_tool` is a Streamlit-based application designed to assist with intraday trading by generating and validating trade signals, providing market insights, and learning from historical performance. It integrates various modules to fetch data, calculate technical indicators, scrape news, and manage watchlists.

## Key Components:

### 1. Watchlist Management
- **[`watchlist_manager.py`](src/watchlist_manager.py)**: Manages loading, validating, and displaying stock watchlists from `config/config.yaml`.

### 2. Data Acquisition
- **[`data_fetcher.py`](src/data_fetcher.py)**: Responsible for fetching historical stock data for the tickers in the watchlist.
- **[`nse_scraper.py`](src/nse_scraper.py)**: (Presumed) Handles scraping specific data from the National Stock Exchange (NSE), though not directly invoked in `main.py`'s core loop for historical data.
- **[`news_scraper.py`](src/news_scraper.py)**: Scrapes news headlines from various sources (e.g., Google News) for sentiment analysis.

### 3. Technical Analysis & Signal Generation
- **[`indicator_calculator.py`](src/indicator_calculator.py)**: Calculates a range of technical indicators such as Moving Averages (SMA_20, SMA_50, SMA_200), Average True Range (ATR), Volume Spikes, Swing Points, Breakouts, and Breakdowns.
- **[`trade_signal_generator.py`](src/trade_signal_generator.py)**: Generates BTST (Buy Today Sell Tomorrow) and Options trade signals based on calculated indicators and news sentiment. It also ranks these signals.

### 4. Market & Signal Management
- **[`market_timing_manager.py`](src/market_timing_manager.py)**: Determines if the market is open, closed, or if it's a holiday, which impacts when signals are generated and validated.
- **[`signal_database_manager.py`](src/signal_database_manager.py)**: Manages the storage and retrieval of generated trade signals for historical tracking and validation.
- **[`signal_validator.py`](src/signal_validator.py)**: Validates the performance of past trade signals against actual market movements.

### 5. AI & Learning
- **[`sentiment_analyzer.py`](src/sentiment_analyzer.py)**: (Presumed) Analyzes the sentiment of scraped news headlines to incorporate into signal generation, although its direct invocation for sentiment analysis isn't explicit in the provided `main.py` snippet.
- **[`self_learning_module.py`](src/self_learning_module.py)**: Updates the internal model weights based on the performance and validation results of past signals, enabling the tool to improve its signal generation over time.

## Application Flow (as seen in `main.py`):
1.  **Initialization**: Streamlit page configuration, title, and session state variables are set.
2.  **Core Component Initialization**: Instances of `SignalDatabaseManager`, `MarketTimingManager`, `SignalValidator`, and `SelfLearningModule` are created.
3.  **Watchlist Loading**: `WatchlistManager` loads and validates the watchlists from `config/config.yaml`.
4.  **Data & Indicator Calculation**: Historical stock data is fetched via `data_fetcher.py`, and technical indicators are calculated using `indicator_calculator.py`.
5.  **News Scraping**: `NewsScraper` collects news for the watchlist stocks.
6.  **Signal Generation**: `TradeSignalGenerator` creates BTST and Options signals, which are then ranked.
7.  **Reporting & Storage**: Top signals are saved to a daily report and stored in the `SignalDatabaseManager`.
8.  **Signal Validation & Learning**: (Post-market close) `SignalValidator` assesses past signals, and `SelfLearningModule` updates its weights.
9.  **Streamlit UI**: Displays watchlists, news sentiment, technical indicators, and generated signals with filtering options. A chatbot interface is also integrated.