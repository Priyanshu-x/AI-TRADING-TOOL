import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
import logging
from watchlist_manager import WatchlistManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_stock_data(tickers, output_dir="data", start_date=None, end_date=None):
    """
    Fetches live and historical price/volume data for a list of stock tickers
    using yfinance and stores it in CSV files.

    Args:
        tickers (list): A list of stock ticker symbols (e.g., ["AAPL", "MSFT"]).
        output_dir (str): The directory to save the CSV files. Defaults to "data".

    Returns:
        dict: A summary of downloaded records for each stock, including success/failure status.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")

    summary = {}
    # Default date range if not provided
    if end_date is None:
        end_date = datetime.now()
    if start_date is None:
        start_date = end_date - timedelta(days=365) # Fetch 1 year of data by default
    
    # Ensure dates are in datetime.date format if they are datetimes
    if isinstance(start_date, datetime):
        start_date = start_date.date()
    if isinstance(end_date, datetime):
        end_date = end_date.date()

    for ticker in tickers:
        try:
            logging.info(f"Fetching data for {ticker} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            # Download data
            data = yf.download(ticker, start=start_date, end=end_date)

            if data.empty:
                logging.warning(f"No data found for {ticker} for the requested date range. Skipping.")
                summary[ticker] = {"status": "failed", "records": 0, "reason": "No data found"}
                continue

            file_path = os.path.join(output_dir, f"{ticker}_data.csv")
            data = data.reset_index() # Converts the index (Dates) into a column
            data.rename(columns={'index': 'Date'}, inplace=True) # Renames the new column to 'Date'
            data.to_csv(file_path, index=False) # Do NOT write the DataFrame index again, as 'Date' is now a regular column
            logging.info(f"Successfully downloaded {len(data)} records for {ticker} to {file_path}")
            summary[ticker] = {"status": "success", "records": len(data), "file": file_path}

        except Exception as e:
            logging.error(f"Failed to download data for {ticker}: {e}")
            summary[ticker] = {"status": "failed", "records": 0, "reason": str(e)}
    
    return summary

def get_all_watchlist_tickers(config_path='config/config.yaml'):
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