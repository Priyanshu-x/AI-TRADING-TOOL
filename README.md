# AI-Powered Trading Signal Tool

## Project Goals
This tool aims to provide actionable trading signals for Indian stock options (Nifty 50 and BankNifty stocks) by leveraging AI.

Key features include:
- Recommending BTST (Buy Today, Sell Tomorrow) trades just before market close.
- Suggesting options calls/puts at market open, targeting high-volatility stocks.
- Utilizing both news sentiment analysis and technical analysis for signal generation.
- Outputting signals in a dashboard or console for easy consumption.

## Setup Steps
1. Clone this repository.
2. Navigate to the project directory: `cd ai_trading_signal_tool`
3. Install dependencies: `pip install -r requirements.txt`
4. Configure your watchlist and other settings in `config/config.yaml`.

## Updating Watchlist
The tool can automatically fetch and update the Nifty 50 and Bank Nifty stock lists from the NSE India website.

To update the watchlist:
1. Run the `watchlist_manager.py` script directly:
   `python watchlist_manager.py`
   This will fetch the latest constituents, merge them with existing ones (avoiding duplicates), and save the updated list to `config/config.yaml`.

**Manual Update:**
If automatic scraping fails due to connectivity issues, rate-limiting, or website changes, you can manually update the `config/config.yaml` file.
1. Visit the official NSE India website (e.g., `https://www.nseindia.com/market-data/live-equity-market` and navigate to the Nifty 50 and Bank Nifty constituent pages).
2. Manually identify the stock symbols (e.g., RELIANCE.NS) and their full names.
3. Edit the `config/config.yaml` file under the `watchlist` section to add or remove stocks. Ensure the format `symbol: SYMBOL.NS` and `name: Company Name` is followed.

**Last Watchlist Update:**
The `config/config.yaml` file will contain a `last_updated` timestamp indicating when the watchlist was last automatically updated.

## Intended Features
- **Data Collection:** Automated fetching of historical stock data, news articles, and market data.
- **Sentiment Analysis:** AI-driven analysis of news sentiment to gauge market mood.
- **Technical Analysis:** Calculation of various technical indicators (e.g., RSI, MACD, Bollinger Bands).
- **Signal Generation:** Combining sentiment and technical analysis to generate trading recommendations.
- **Backtesting:** (Future) Ability to backtest strategies against historical data.
- **Real-time Dashboard:** (Future) A Streamlit-based dashboard for real-time signal display.