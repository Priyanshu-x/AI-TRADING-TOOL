import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
import logging
import random
from .watchlist_manager import WatchlistManager

from concurrent.futures import ThreadPoolExecutor, as_completed
from .logger import setup_logger

logger = setup_logger(__name__)

def fetch_single_ticker(ticker, start_date, end_date, output_dir):
    """
    Helper function to fetch data for a single ticker.
    """
    import yfinance as yf
    import time
    
    # Session removed as it was valid proof of failure in test_yf.py

    data = pd.DataFrame() # Initialize
    retries = 3
    for i in range(retries):
        try:
            # Add random delay to be polite and avoid rate limits
            delay = random.uniform(1.0, 3.0) 
            logger.info(f"Fetching data for {ticker} (Attempt {i+1}/{retries}) - Waiting {delay:.2f}s")
            time.sleep(delay)
            
            # Use Ticker.history which proved reliable in testing
            dat = yf.Ticker(ticker)
            data = dat.history(start=start_date, end=end_date)

            if not data.empty:
                break
            else:
                logger.warning(f"Empty data for {ticker} on attempt {i+1}")
                time.sleep(2)
        except Exception as e:
            logger.warning(f"Error fetching {ticker} on attempt {i+1}: {e}")
            time.sleep(2)
    
    if data.empty:
        logger.warning(f"No data found for {ticker} after {retries} attempts.")
        return ticker, {"status": "failed", "records": 0, "reason": "No data found"}
    
    try:
        file_path = os.path.join(output_dir, f"{ticker}_data.csv")
        data = data.reset_index()
        data.rename(columns={'index': 'Date'}, inplace=True)
        data.to_csv(file_path, index=False)
        logger.info(f"Downloaded {len(data)} records for {ticker}")
        return ticker, {"status": "success", "records": len(data), "file": file_path}

    except Exception as e:
        logger.error(f"Failed to process/save data for {ticker}: {e}")
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
    max_workers = 2 # Significantly reduced threads to avoid IP blocking on shared cloud

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
    # all_tickers = ["ADANIPORTS.NS", "RELIANCE.NS"]
    print(f"Testing fetch for: {all_tickers}")
    if all_tickers:
        download_summary = fetch_stock_data(all_tickers)
        print("\n--- Download Summary ---")
        for ticker, info in download_summary.items():
            print(f"{ticker}: Status - {info['status']}, Records - {info['records']}")
    else:
        print("No tickers found in the watchlist to download data for.")