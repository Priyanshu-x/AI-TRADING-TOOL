# ai_trading_signal_tool/main.py

import streamlit as st
from watchlist_manager import WatchlistManager
from news_scraper import NewsScraper
from data_fetcher import fetch_stock_data, get_all_watchlist_tickers
from indicator_calculator import (
    calculate_moving_averages,
    calculate_atr,
    calculate_volume_spikes,
    find_swing_points,
    detect_breakouts_breakdowns
)
import pandas as pd
import os
from trade_signal_generator import TradeSignalGenerator
import schedule
import time
from datetime import datetime
import logging

# Configure logging to display INFO messages in the console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_daily_report(btst_signals, options_signals):
    report_dir = "outputs"
    os.makedirs(report_dir, exist_ok=True)
    today_str = datetime.now().strftime("%Y-%m-%d")
    report_filename = os.path.join(report_dir, f"daily_signals_report_{today_str}.txt")

    with open(report_filename, "w") as f:
        f.write(f"Daily Trading Signals Report - {today_str}\n")
        f.write("="*40 + "\n\n")

        f.write("BTST Signals:\n")
        if btst_signals:
            for signal in btst_signals:
                f.write(f"  BTST {signal['action']} | {signal['symbol']} | Entry: {signal['entry_price']} | Target: {signal['target_price']} | Confidence: {signal['confidence_score']:.2f}\n")
        else:
            f.write("  No BTST signals generated.\n")
        f.write("\n")

        f.write("Options Signals:\n")
        if options_signals:
            for signal in options_signals:
                f.write(f"  OPTIONS {signal['action']} | {signal['symbol']} | Strike: {signal['strike_price']} | Expiry: {signal['expiry_date']} | Confidence: {signal['confidence_score']:.2f}\n")
        else:
            f.write("  No Options signals generated.\n")
        f.write("\n")
    print(f"Daily report saved to {report_filename}")

def run_analysis():
    st.write("Starting AI Trading Signal Tool...")
    # Initialize WatchlistManager
    manager = WatchlistManager()

    # Load watchlist
    if manager.load_watchlist():
        # Validate watchlist
        if manager.validate_watchlist():
            # Print watchlist to console
            manager.print_watchlist()
        else:
            print("Watchlist validation failed. Please check config file.")
    else:
        print("Failed to load watchlist. Exiting.")
        return

    all_stocks_data = {}
    if 'all_stocks_data' not in st.session_state:
        st.session_state['all_stocks_data'] = {}

    # Initialize TradeSignalGenerator
    signal_generator = TradeSignalGenerator()

    # Fetch all tickers from watchlist
    all_tickers = get_all_watchlist_tickers()
    if not all_tickers:
        print("No tickers found in watchlist. Exiting.")
        return

    # Fetch and save historical data for all tickers
    download_summary = fetch_stock_data(all_tickers)
    print("\n--- Data Download Summary ---")
    for ticker, info in download_summary.items():
        print(f"{ticker}: Status - {info['status']}, Records - {info['records']}")

    print("\n--- Calculating Indicators for Stock Data ---")
    for symbol in all_tickers:
        if download_summary.get(symbol, {}).get("status") == "success":
            file_path = download_summary[symbol]["file"]
            try:
                df = pd.read_csv(file_path, parse_dates=['Date'])
                if df is not None and not df.empty:
                    df.set_index('Date', inplace=True) # Set 'Date' column as index
                    # Ensure numeric columns are of numeric type
                    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                    for col in numeric_cols:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    df.dropna(subset=numeric_cols, inplace=True) # Drop rows where essential numeric data is missing
                    
                    print(f"Calculating indicators for {symbol}...")
                    df = calculate_moving_averages(df)
                    df = calculate_atr(df)
                    df = calculate_volume_spikes(df)
                    df = find_swing_points(df)
                    df = detect_breakouts_breakdowns(df)
                    
                    # Convert boolean columns to int (0 or 1) to avoid pyarrow.lib.ArrowInvalid with Streamlit
                    for col in ['Volume_Spike', 'Breakout', 'Breakdown']:
                        if col in df.columns:
                            df[col] = df[col].astype(int)

                    all_stocks_data[symbol] = df
                    st.session_state['all_stocks_data'][symbol] = df # Store in session state
                    print(f"Indicators calculated for {symbol}.")
                else:
                    print(f"No data loaded from {file_path} for {symbol}. Skipping indicator calculation.")
            except Exception as e:
                print(f"Error loading or processing data for {symbol} from {file_path}: {e}")
        else:
            print(f"Skipping indicator calculation for {symbol}: Data download failed or not available.")
    
    print(f"\n--- all_stocks_data content after indicator calculation: {list(all_stocks_data.keys())} ---")

    # Initialize NewsScraper
    scraper = NewsScraper()

    # Scrape news for each stock in the watchlist
    print("\n--- Starting News Scraping ---")
    for index_name, stocks in manager.watchlist.items():
        print(f"Scraping news for {index_name}...")
        for stock in stocks:
            symbol = stock.get('symbol')
            name = stock.get('name')
            if symbol and name:
                # scraper.scrape_moneycontrol(symbol, name) # Temporarily commented out due to 403/401 errors
                # scraper.scrape_reuters(symbol, name) # Temporarily commented out due to 403/401 errors
                scraper.scrape_google_news(symbol, name)
            else:
                print(f"Skipping stock due to missing symbol or name: {stock}")

    # Get news as DataFrame and save to CSV
    news_df = scraper.get_news_as_dataframe()
    if not news_df.empty:
        print("\n--- Collected News Headlines (first 5) ---")
        print(news_df.head())
        scraper.save_news_to_csv()
        st.session_state['news_df'] = news_df # Store news_df in session state
    else:
        print("\nNo news headlines collected.")
        st.session_state['news_df'] = pd.DataFrame() # Ensure news_df is always a DataFrame

    # Generate BTST and Options Signals
    print("\n--- Generating Trade Signals ---")
    btst_signals = signal_generator.generate_btst_signals(all_stocks_data, news_df)
    options_signals = signal_generator.generate_options_signals(all_stocks_data, news_df)
    st.session_state['btst_signals'] = btst_signals
    st.session_state['options_signals'] = options_signals

    if btst_signals:
        print(f"Generated {len(btst_signals)} BTST signals.")
        for signal in btst_signals:
            print(f"  BTST {signal['action']} | {signal['symbol']} | Entry: {signal['entry_price']} | Target: {signal['target_price']} | Confidence: {signal['confidence_score']:.2f}")
    else:
        print("No BTST signals generated.")

    if options_signals:
        print(f"Generated {len(options_signals)} Options signals.")
        for signal in options_signals:
            print(f"  OPTIONS {signal['action']} | {signal['symbol']} | Strike: {signal['strike_price']} | Expiry: {signal['expiry_date']} | Confidence: {signal['confidence_score']:.2f}")
    else:
        print("No Options signals generated.")

    # Save all generated signals
    signal_generator.save_signals_to_json()
    
    # Save daily report
    save_daily_report(btst_signals, options_signals)

    print("\n--- Indicator Report ---")
    if all_stocks_data:
        for symbol, df in all_stocks_data.items():
            print(f"\nReport for {symbol}:")
            if not df.empty:
                latest_data = df.iloc[-1]
                print(f"  Latest Close: {latest_data['Close']:.2f}")
                print(f"  SMA_20: {latest_data['SMA_20']:.2f}")
                print(f"  SMA_50: {latest_data['SMA_50']:.2f}")
                print(f"  SMA_200: {latest_data['SMA_200']:.2f}")
                print(f"  ATR: {latest_data['ATR']:.2f}")
                print(f"  Volume Spike: {latest_data['Volume_Spike']}")
                print(f"  Support: {latest_data['Support']:.2f}")
                print(f"  Resistance: {latest_data['Resistance']:.2f}")
                print(f"  Breakout: {latest_data['Breakout']}")
                print(f"  Breakdown: {latest_data['Breakdown']}")
            else:
                print(f"  No data available for {symbol}.")
    else:
        print("No stock data processed for indicator reports.")

    print("\nAI Trading Signal Tool finished successfully.")


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.title("AI Trading Signal Dashboard")

    # Initialize session state variables if not already present
    if 'all_stocks_data' not in st.session_state:
        st.session_state['all_stocks_data'] = {}
    if 'news_df' not in st.session_state:
        st.session_state['news_df'] = pd.DataFrame()
    if 'btst_signals' not in st.session_state:
        st.session_state['btst_signals'] = []
    if 'options_signals' not in st.session_state:
        st.session_state['options_signals'] = []

    # Sidebar for navigation or controls
    st.sidebar.title("Controls")
    
    if st.sidebar.button("Run Analysis Now", key="run_analysis_button"):
        run_analysis()

    st.sidebar.subheader("Schedule Analysis")
    schedule_option = st.sidebar.radio(
        "Choose a schedule:",
        ("Manual", "Pre-Close (3:00 PM IST)", "Open (9:15 AM IST)")
    )

    if schedule_option == "Pre-Close (3:00 PM IST)":
        schedule.every().day.at("15:00").do(run_analysis)
        st.sidebar.info("Scheduled daily analysis for 3:00 PM IST.")
    elif schedule_option == "Open (9:15 AM IST)":
        schedule.every().day.at("09:15").do(run_analysis)
        st.sidebar.info("Scheduled daily analysis for 9:15 AM IST.")
    
    # To run scheduled jobs in a Streamlit app, you'd typically need a separate thread or process.
    # For simplicity in this example, we'll just show the scheduled status.
    # In a real-world deployment, consider using a background task runner (e.g., Celery, APScheduler).
    st.sidebar.write("Note: Scheduled tasks require a persistent background process to run.")
    
    # Main content area
    # Initialize WatchlistManager
    manager = WatchlistManager()
    if manager.load_watchlist() and manager.validate_watchlist():
        st.header("Current Watchlist")
        for index_name, stocks in manager.watchlist.items():
            st.subheader(f"{index_name} ({len(stocks)} stocks)")
            if stocks:
                watchlist_df = pd.DataFrame(stocks)
                st.dataframe(watchlist_df, hide_index=True)
            else:
                st.write(f"No stocks in {index_name} list.")
    else:
        st.error("Failed to load or validate watchlist. Please check config file.")

    st.header("Latest News Sentiment")
    if 'news_df' in st.session_state and not st.session_state['news_df'].empty:
        st.subheader("Collected News Headlines with Sentiment")
        st.dataframe(st.session_state['news_df'][['symbol', 'headline', 'sentiment', 'confidence']], hide_index=True)
    else:
        st.info("No news headlines collected yet. Run analysis to fetch news.")

    st.header("Technical Indicators")
    if 'all_stocks_data' in st.session_state and st.session_state['all_stocks_data']:
        selected_symbol = st.selectbox("Select a stock to view indicators:", list(st.session_state['all_stocks_data'].keys()))
        if selected_symbol:
            st.subheader(f"Technical Indicators for {selected_symbol}")
            latest_data = st.session_state['all_stocks_data'][selected_symbol].iloc[-1]
            st.write(latest_data[['Close', 'SMA_20', 'SMA_50', 'SMA_200', 'ATR', 'Volume_Spike', 'Support', 'Resistance', 'Breakout', 'Breakdown']])
    else:
        st.info("No stock data processed for indicator reports yet. Run analysis to fetch data and calculate indicators.")

    st.header("Generated Signals")
    if 'btst_signals' in st.session_state and st.session_state['btst_signals']:
        st.subheader("BTST Signals")
        btst_df = pd.DataFrame(st.session_state['btst_signals'])
        st.dataframe(btst_df, hide_index=True)
    else:
        st.info("No BTST signals generated yet. Run analysis to generate signals.")

    if 'options_signals' in st.session_state and st.session_state['options_signals']:
        st.subheader("Options Signals")
        options_df = pd.DataFrame(st.session_state['options_signals'])
        st.dataframe(options_df, hide_index=True)
    else:
        st.info("No Options signals generated yet. Run analysis to generate signals.")
