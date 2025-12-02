import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging
import time # Import time for delays

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class NSEScraper:
    def __init__(self):
        self.base_url = "https://www.nseindia.com"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9,hi;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
            'Connection': 'keep-alive',
            'DNT': '1', # Do Not Track Request Header
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.session.get(self.base_url) # Initialize session and get cookies

    def _fetch_url(self, url):
        """Fetches the content of a given URL."""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()  # Raise an exception for HTTP errors
            logging.debug(f"Successfully fetched URL {url}. Response status: {response.status_code}")
            return response
        except requests.exceptions.HTTPError as e:
            logging.error(f"HTTP Error fetching {url}: {e.response.status_code} {e.response.reason}. Response text: {e.response.text}")
            return None
        except requests.exceptions.ConnectionError as e:
            logging.error(f"Connection Error fetching {url}: {e}")
            return None
        except requests.exceptions.Timeout as e:
            logging.error(f"Timeout Error fetching {url}: {e}")
            return None
        except requests.exceptions.RequestException as e:
            logging.error(f"General Request Error fetching {url}: {e}")
            return None

    def get_nifty_constituents(self, index_name="NIFTY 50"):
        """
        Fetches the constituents for a given Nifty index (e.g., "NIFTY 50", "NIFTY BANK").
        """
        index_map = {
            "NIFTY 50": "/api/equity-master?index=NIFTY%2050",
            "NIFTY BANK": "/api/equity-master?index=NIFTY%20BANK",
        }
        
        api_path = index_map.get(index_name)
        if not api_path:
            logging.error(f"Unsupported index name: {index_name}")
            return []

        url = f"{self.base_url}{api_path}"
        
        # Add a small delay to mimic human behavior and avoid immediate rate-limiting
        time.sleep(1)

        # Add Referer header for the API call
        self.session.headers.update({
            'Referer': self.base_url + '/market-data/live-equity-market',
            'X-Requested-With': 'XMLHttpRequest', # Often required for API endpoints
        })
        
        data = self._fetch_url(url)
        if data:
            logging.debug(f"Raw data fetched for {index_name}: {data[:500]}...") # Log first 500 chars of data
            # Reset Referer and X-Requested-With headers
            self.session.headers.pop('Referer', None)
            self.session.headers.pop('X-Requested-With', None)
            try:
                json_data = pd.read_json(data)
                constituents = []
                for _, row in json_data.iterrows():
                    symbol = row['symbol'] + ".NS" # Append .NS for Yahoo Finance compatibility
                    name = row['companyName']
                    constituents.append({'symbol': symbol, 'name': name})
                logging.info(f"Successfully fetched {len(constituents)} constituents for {index_name}.")
                return constituents
            except ValueError as e:
                logging.error(f"Error parsing JSON data for {index_name}: {e}. Raw data: {data}") # Added raw data logging
                return []
        
                return constituents
            except ValueError as e:
                logging.error(f"Error parsing JSON data for {index_name}: {e}")
                return []
        return []

    def get_constituents_with_retry(self, index_name, retries=3, delay=5):
        """
        Attempts to fetch constituents with retries, handling potential temporary blocks.
        """
        for i in range(retries):
            logging.info(f"Attempt {i+1}/{retries} to fetch {index_name} constituents...")
            constituents = self.get_nifty_constituents(index_name)
            if constituents:
                return constituents
            logging.warning(f"Failed to fetch {index_name} constituents on attempt {i+1}. Retrying in {delay} seconds...")
            time.sleep(delay)
        logging.error(f"Failed to fetch {index_name} constituents after {retries} attempts. Please consider manual update.")
        return []

if __name__ == "__main__":
    scraper = NSEScraper()
    
    print("Fetching NIFTY 50 constituents...")
    nifty50_stocks = scraper.get_constituents_with_retry("NIFTY 50")
    if nifty50_stocks:
        print(f"NIFTY 50 Constituents ({len(nifty50_stocks)}):")
        for stock in nifty50_stocks[:5]: # Print first 5 for brevity
            print(f"  - {stock['symbol']}: {stock['name']}")
    else:
        print("Failed to fetch NIFTY 50 constituents. Please refer to README.md for manual update instructions.")

    print("\nFetching NIFTY BANK constituents...")
    banknifty_stocks = scraper.get_constituents_with_retry("NIFTY BANK")
    if banknifty_stocks:
        print(f"NIFTY BANK Constituents ({len(banknifty_stocks)}):")
        for stock in banknifty_stocks[:5]: # Print first 5 for brevity
            print(f"  - {stock['symbol']}: {stock['name']}")
    else:
        print("Failed to fetch NIFTY BANK constituents. Please refer to README.md for manual update instructions.")