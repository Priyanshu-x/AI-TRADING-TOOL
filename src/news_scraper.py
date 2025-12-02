# ai_trading_signal_tool/news_scraper.py

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import logging
import re
import os
from .sentiment_analyzer import SentimentAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class NewsScraper:
    def __init__(self):
        self.news_data = []
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.sentiment_analyzer = SentimentAnalyzer()

    def _fetch_page(self, url):
        """Fetches the content of a given URL."""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
            return response.text
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching {url}: {e}")
            return None

    # Note: Direct scraping of Moneycontrol and Reuters is currently facing 403/401 Forbidden errors.
    # This is likely due to anti-scraping measures. For robust scraping of these sites,
    # more advanced techniques (e.g., rotating proxies, headless browsers, or specific APIs)
    # would be required. For now, these functions are commented out to allow the script to run.
    # If these sources are critical, further investigation into their APIs or advanced scraping
    # methods is needed.

    # def scrape_moneycontrol(self, symbol, name):
    #     """Scrapes news from Moneycontrol for a given stock symbol."""
    #     logging.info(f"Scraping Moneycontrol for {symbol} ({name})...")
    #     search_query = name.replace(" ", "%20")
    #     url = f"https://www.moneycontrol.com/news/tags/{search_query}.html"
        
    #     page_content = self._fetch_page(url)
    #     if not page_content:
    #         return

    #     soup = BeautifulSoup(page_content, 'html.parser')
    #     news_items = soup.find_all('li', class_='clearfix')
        
    #     for item in news_items:
    #         headline_tag = item.find('h2')
    #         link_tag = item.find('a')
    #         time_tag = item.find('span', class_='time_ago')

    #         if headline_tag and link_tag:
    #             headline = headline_tag.text.strip()
    #             link = link_tag['href']
    #             published_time = datetime.utcnow() # Placeholder, actual parsing needed
                
    #             self.news_data.append({
    #                 'symbol': symbol,
    #                 'source': 'Moneycontrol',
    #                 'headline': headline,
    #                 'link': link,
    #                 'published_at': published_time
    #             })
    #     logging.info(f"Finished scraping Moneycontrol for {symbol}.")

    # def scrape_reuters(self, symbol, name):
    #     """Scrapes news from Reuters for a given stock symbol."""
    #     logging.info(f"Scraping Reuters for {symbol} ({name})...")
    #     search_query = name.replace(" ", "+")
    #     url = f"https://www.reuters.com/search/news?blob={search_query}"
        
    #     page_content = self._fetch_page(url)
    #     if not page_content:
    #         return

    #     soup = BeautifulSoup(page_content, 'html.parser')
    #     news_items = soup.find_all('div', class_='search-result-content')
        
    #     for item in news_items:
    #         headline_tag = item.find('h3', class_='story-title')
    #         link_tag = item.find('a')
    #         time_tag = item.find('span', class_='timestamp')

    #         if headline_tag and link_tag:
    #             headline = headline_tag.text.strip()
    #             link = "https://www.reuters.com" + link_tag['href'] if link_tag['href'].startswith('/') else link_tag['href']
    #             published_time = datetime.utcnow() # Placeholder, actual parsing needed
                
    #             self.news_data.append({
    #                 'symbol': symbol,
    #                 'source': 'Reuters',
    #                 'headline': headline,
    #                 'link': link,
    #                 'published_at': published_time
    #             })
    #     logging.info(f"Finished scraping Reuters for {symbol}.")

    def scrape_google_news(self, symbol, name):
        """Scrapes news from Google News for a given stock symbol."""
        logging.info(f"Scraping Google News for {symbol} ({name})...")
        search_query = f"{name} stock news"
        url = f"https://news.google.com/rss/search?q={search_query}&hl=en-IN&gl=IN&ceid=IN:en"
        
        page_content = self._fetch_page(url)
        if not page_content:
            return

        soup = BeautifulSoup(page_content, 'xml') # Google News RSS is XML
        
        for item in soup.find_all('item'):
            title = item.find('title').text.strip()
            link = item.find('link').text.strip()
            pub_date_str = item.find('pubDate').text.strip()
            
            try:
                published_at = datetime.strptime(pub_date_str, '%a, %d %b %Y %H:%M:%S %Z')
                if datetime.utcnow() - published_at < timedelta(hours=24):
                    self.news_data.append({
                        'symbol': symbol,
                        'source': 'Google News',
                        'headline': title,
                        'link': link,
                        'published_at': published_at
                    })
            except ValueError as e:
                logging.warning(f"Could not parse date '{pub_date_str}': {e}")
        logging.info(f"Finished scraping Google News for {symbol}.")

    def get_news_as_dataframe(self):
        """Returns the collected news as a pandas DataFrame with sentiment scores."""
        df = pd.DataFrame(self.news_data)
        if not df.empty:
            headlines = df['headline'].tolist()
            sentiments = self.sentiment_analyzer.analyze_sentiment(headlines)
            
            sentiment_df = pd.DataFrame(sentiments)
            df = pd.concat([df, sentiment_df], axis=1)
        return df

    def save_news_to_csv(self, filename="../outputs/news_headlines.csv"):
        """Saves the collected news to a CSV file."""
        df = self.get_news_as_dataframe()
        if not df.empty:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            df.to_csv(filename, index=False)
            logging.info(f"News headlines saved to {filename}")
        else:
            logging.info("No news headlines to save.")

if __name__ == "__main__":
    # Example usage for testing
    scraper = NewsScraper()
    
    # Simulate a watchlist
    test_watchlist = [
        {'symbol': 'RELIANCE', 'name': 'Reliance Industries'},
        {'symbol': 'TCS', 'name': 'Tata Consultancy Services'},
        {'symbol': 'HDFCBANK', 'name': 'HDFC Bank'}
    ]

    for stock in test_watchlist:
        scraper.scrape_moneycontrol(stock['symbol'], stock['name'])
        scraper.scrape_reuters(stock['symbol'], stock['name'])
        scraper.scrape_google_news(stock['symbol'], stock['name'])
    
    news_df = scraper.get_news_as_dataframe()
    print("\n--- Collected News Headlines with Sentiment ---")
    print(news_df[['symbol', 'headline', 'sentiment', 'confidence']].head())
    scraper.save_news_to_csv()