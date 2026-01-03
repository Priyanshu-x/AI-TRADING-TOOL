import yaml
import os
from .nse_scraper import NSEScraper
import datetime
from .logger import setup_logger

logger = setup_logger(__name__)

class WatchlistManager:
    def __init__(self, config_path=None):
        if config_path is None:
            # Calculate absolute path to config.yaml relative to this file
            # This file is in src/, so we go up one level to project root, then into config/
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.config_path = os.path.join(base_dir, 'config', 'config.yaml')
        else:
            self.config_path = config_path
        self.watchlist = {}

    def load_watchlist(self):
        """
        Loads the Nifty 50 and BankNifty watchlists from the config file.
        """
        if not os.path.exists(self.config_path):
            logger.error(f"Config file not found at {self.config_path}")
            return False
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.watchlist['NIFTY50'] = config.get('watchlist', {}).get('NIFTY50', [])
                self.watchlist['BANKNIFTY'] = config.get('watchlist', {}).get('BANKNIFTY', [])
            logger.debug(f"Loaded config watchlist: {list(self.watchlist.keys())}")
            logger.info("Watchlist loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Error loading watchlist from config: {e}")
            return False

    def validate_watchlist(self):
        """
        Validates the loaded watchlist to ensure each item has 'symbol' and 'name'.
        """
        is_valid = True
        for index_name, stocks in self.watchlist.items():
            if not isinstance(stocks, list):
                logger.error(f"Validation Error: Watchlist for {index_name} is not a list.")
                is_valid = False
                continue
            for i, stock in enumerate(stocks):
                if not isinstance(stock, dict):
                    logger.error(f"Validation Error: Item {i} in {index_name} is not a dictionary: {stock}")
                    is_valid = False
                elif 'symbol' not in stock or 'name' not in stock:
                    logger.error(f"Validation Error: Item {i} in {index_name} missing 'symbol' or 'name': {stock}")
                    is_valid = False
        if is_valid:
            logger.info("Watchlist validated successfully.")
        return is_valid

    def print_watchlist(self):
        """
        Prints the loaded watchlist to the console (via logger).
        """
        logger.info("--- Current Watchlist ---")
        if not self.watchlist:
            logger.info("Watchlist is empty.")
            return

        for index_name, stocks in self.watchlist.items():
            logger.info(f"{index_name} ({len(stocks)} stocks):")
            if stocks:
                for stock in stocks:
                    logger.info(f"  - Symbol: {stock.get('symbol', 'N/A')}, Name: {stock.get('name', 'N/A')}")
            else:
                logger.info("  No stocks in this list.")
        logger.info("-------------------------")

    def fetch_and_update_watchlist(self):
        """
        Fetches the latest Nifty 50 and Bank Nifty constituents and updates the watchlist.
        Handles errors and ensures no duplicates.
        """
        logger.info("Fetching and updating watchlist from NSE India...")
        scraper = NSEScraper()
        
        nifty50_constituents = scraper.get_constituents_with_retry("NIFTY 50")
        banknifty_constituents = scraper.get_constituents_with_retry("NIFTY BANK")

        if not nifty50_constituents and not banknifty_constituents:
            logger.warning("Failed to fetch any constituents. Watchlist not updated.")
            return False

        updated_nifty50 = self._merge_constituents(self.watchlist.get('NIFTY50', []), nifty50_constituents)
        updated_banknifty = self._merge_constituents(self.watchlist.get('BANKNIFTY', []), banknifty_constituents)

        self.watchlist['NIFTY50'] = updated_nifty50
        self.watchlist['BANKNIFTY'] = updated_banknifty
        
        if self.save_watchlist():
            logger.info("Watchlist updated and saved successfully.")
            return True
        else:
            logger.error("Failed to save updated watchlist.")
            return False

    def _merge_constituents(self, existing_list, new_list):
        """Merges new constituents into an existing list, avoiding duplicates."""
        merged = {stock['symbol']: stock for stock in existing_list}
        for stock in new_list:
            merged[stock['symbol']] = stock
        return list(merged.values())

    def get_all_watchlist_tickers(self):
        """
        Returns a list of all unique stock symbols in the watchlist.
        """
        all_tickers = set()
        for index_name, stocks in self.watchlist.items():
            for stock in stocks:
                if 'symbol' in stock:
                    all_tickers.add(stock['symbol'])
        return sorted(list(all_tickers))

if __name__ == "__main__":
    manager = WatchlistManager()
    
    logger.info("Initial Watchlist:")
    if manager.load_watchlist():
        manager.print_watchlist()
    
    manager.fetch_and_update_watchlist()
    
    logger.info("Watchlist after update:")
    if manager.load_watchlist(): # Reload to reflect saved changes
        manager.print_watchlist()
        manager.validate_watchlist()
