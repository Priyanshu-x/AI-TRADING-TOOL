import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
import logging
from .watchlist_manager import WatchlistManager

from concurrent.futures import ThreadPoolExecutor, as_completed
from .logger import setup_logger

logger = setup_logger(__name__)

def fetch_single_ticker(ticker, start_date, end_date, output_dir):
    """
    Helper function to fetch data for a single ticker.
    """
    import yfinance as yf # Import locally to avoid potential threading issues with some libs
    try:
        logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}")
        data = yf.download(ticker, start=start_date, end=end_date, progress=False, threads=False) # threads=False since we handle threading

        if data.empty:
            logger.warning(f"No data found for {ticker}")
            return ticker, {"status": "failed", "records": 0, "reason": "No data found"}

        file_path = os.path.join(output_dir, f"{ticker}_data.csv")
        data = data.reset_index()
        data.rename(columns={'index': 'Date'}, inplace=True)
        data.to_csv(file_path, index=False)
        logger.info(f"Downloaded {len(data)} records for {ticker}")
        return ticker, {"status": "success", "records": len(data), "file": file_path}

    except Exception as e:
        logger.error(f"Failed to download {ticker}: {e}")
        return ticker, {"status": "failed", "records": 0, "reason": str(e)}

def fetch_stock_data(tickers, output_dir="data", start_date=None, end_date=None):
    """
    Fetches live and historical price/volume data for a list of stock tickers
    concurrently using ThreadPoolExecutor.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")

    # Defaults
    if end_date is None:
        end_date = datetime.now()
    if start_date is None:
        start_date = end_date - timedelta(days=365)
    
    # Ensure date objects
    if isinstance(start_date, datetime): start_date = start_date.date()
    if isinstance(end_date, datetime): end_date = end_date.date()

    summary = {}
    max_workers = min(10, len(tickers)) # Limit threads to 10 or number of tickers

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(fetch_single_ticker, ticker, start_date, end_date, output_dir): ticker
            for ticker in tickers
        }
        
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                t, result = future.result()
                summary[t] = result
            except Exception as e:
                logger.error(f"Exception for {ticker}: {e}")
                summary[ticker] = {"status": "failed", "records": 0, "reason": str(e)}
    
    return summary

def get_all_watchlist_tickers(config_path=None):
    """
    Loads the watchlist using WatchlistManager and returns a flattened list of all ticker symbols.
    """
    manager = WatchlistManager(config_path=config_path)
    if manager.load_watchlist():
        all_tickers = []
        for index_name, stocks in manager.watchlist.items():
            for stock in stocks:
                if 'symbol' in stock:
                    all_tickers.append(stock['symbol'])
        return all_tickers
    else:
        logging.error("Failed to load watchlist from WatchlistManager.")
        return []

if __name__ == "__main__":
    # Integrate with watchlist_manager
    all_tickers = get_all_watchlist_tickers()
    if all_tickers:
        download_summary = fetch_stock_data(all_tickers)
        print("\n--- Download Summary ---")
        for ticker, info in download_summary.items():
            print(f"{ticker}: Status - {info['status']}, Records - {info['records']}")
    else:
        print("No tickers found in the watchlist to download data for.")