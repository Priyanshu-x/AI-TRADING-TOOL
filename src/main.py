# ai_trading_signal_tool/main.py

import sys
import os

# Add the parent directory (the project root) to the Python path
# This allows the 'src' package to be discoverable when main.py is run directly.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from src.watchlist_manager import WatchlistManager
from src.news_scraper import NewsScraper
from src.data_fetcher import fetch_stock_data, get_all_watchlist_tickers
from src.indicator_calculator import (
    calculate_moving_averages,
    calculate_atr,
    calculate_volume_spikes,
    find_swing_points,
    detect_breakouts_breakdowns
)
import pandas as pd
import os
from src.trade_signal_generator import TradeSignalGenerator
import schedule
import time
from datetime import datetime, timedelta
import logging
from src.signal_database_manager import SignalDatabaseManager
from src.market_timing_manager import MarketTimingManager
from src.signal_validator import SignalValidator
from src.self_learning_module import SelfLearningModule

# Configure logging to display INFO messages in the console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_daily_report(ranked_btst_signals, ranked_options_signals):
    report_dir = "outputs"
    os.makedirs(report_dir, exist_ok=True)
    today_str = datetime.now().strftime("%Y-%m-%d")
    report_filename = os.path.join(report_dir, f"daily_signals_report_{today_str}.txt")

    with open(report_filename, "w") as f:
        f.write(f"Daily Trading Signals Report - {today_str}\n")
        f.write("="*40 + "\n\n")

        f.write("Top BTST Signals:\n")
        if ranked_btst_signals:
            for signal in ranked_btst_signals:
                f.write(f"  BTST {signal['action']} | {signal['symbol']} | Entry: {signal['entry_price']} | Target: {signal['target_price']} | Confidence: {signal['confidence_score']:.2f}\n")
        else:
            f.write("  No top BTST signals generated.\n")
        f.write("\n")

        f.write("Top Options Signals:\n")
        if ranked_options_signals:
            for signal in ranked_options_signals:
                f.write(f"  OPTIONS {signal['action']} | {signal['symbol']} | Strike: {signal['strike_price']} | Expiry: {signal['expiry_date']} | Confidence: {signal['confidence_score']:.2f}\n")
        else:
            f.write("  No top Options signals generated.\n")
        f.write("\n")
    print(f"Daily report saved to {report_filename}")

def run_analysis(db_manager, market_manager, signal_validator, learning_module):
    st.write("Starting AI Trading Signal Tool...")
    
    current_date_str = market_manager.get_current_market_time().strftime("%Y-%m-%d")
    
    # Check market timing for signal generation
    if not market_manager.is_market_open():
        st.warning("Market is currently closed. Generating signals for storage, but validation will be deferred.")
        # If market is closed, we still generate signals but mark them for deferred validation
        # and skip immediate validation/learning.
        # We will still proceed with signal generation and saving to DB.
    
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

    # Initialize TradeSignalGenerator, injecting the learning module
    signal_generator = TradeSignalGenerator(learning_module=learning_module)

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
                    # Set 'Date' column as index after reading
                    df.set_index('Date', inplace=True)
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
    
    # Generate BTST CALL signals
    btst_call_signals = signal_generator.generate_btst_signals(all_stocks_data, news_df, side="CALL")
    # Generate BTST PUT signals
    btst_put_signals = signal_generator.generate_btst_signals(all_stocks_data, news_df, side="PUT")
    
    # Combine both CALL and PUT BTST signals
    btst_signals = btst_call_signals + btst_put_signals

    options_signals = signal_generator.generate_options_signals(all_stocks_data, news_df)
    
    # Rank and shortlist signals
    top_n = st.session_state.get('top_n_recommendations', 5) # Get N from session state, default to 5
    ranked_btst_signals = signal_generator.get_top_n_signals(btst_signals, top_n)
    ranked_options_signals = signal_generator.get_top_n_signals(options_signals, top_n)

    st.session_state['btst_signals'] = btst_signals # Keep all signals for potential future use
    st.session_state['options_signals'] = options_signals # Keep all signals for potential future use
    st.session_state['ranked_btst_signals'] = ranked_btst_signals
    st.session_state['ranked_options_signals'] = ranked_options_signals

    if ranked_btst_signals:
        print(f"Generated {len(btst_signals)} total BTST signals. Displaying top {len(ranked_btst_signals)}.")
        for signal in ranked_btst_signals:
            print(f"  BTST {signal.get('signal_type', signal['action'])} | {signal['symbol']} | Entry: {signal['entry_price']} | Target: {signal['target_price']} | Confidence: {signal['confidence_score']:.2f}")
    else:
        print("No BTST signals generated or none met the ranking criteria.")

    if ranked_options_signals:
        print(f"Generated {len(options_signals)} Options signals. Displaying top {len(ranked_options_signals)}.")
        for signal in ranked_options_signals:
            print(f"  OPTIONS {signal['action']} | {signal['symbol']} | Strike: {signal['strike_price']} | Expiry: {signal['expiry_date']} | Confidence: {signal['confidence_score']:.2f}")
    else:
        print("No Options signals generated or none met the ranking criteria.")

    # Save all generated signals (not just top N) to JSON (for historical reference)
    signal_generator.save_signals_to_json()
    
    # Save daily report with top N signals to a text file
    save_daily_report(ranked_btst_signals, ranked_options_signals)

    # Store the top N signals in the database for validation
    all_top_signals = ranked_btst_signals + ranked_options_signals
    if all_top_signals:
        db_manager.save_signals(all_top_signals, current_date_str)
        st.success(f"Top {len(all_top_signals)} signals saved to database for {current_date_str}.")
    else:
        st.info("No top signals generated to save to database.")

    # --- Signal Validation and Learning (only after market closes on a trading day) ---
    if market_manager.is_market_closed_for_day():
        st.write(f"Market closed for the day. Running signal validation for {current_date_str}...")
        validation_results = signal_validator.validate_signals(current_date_str)
        if validation_results:
            st.success(f"Validated {len(validation_results)} signals for {current_date_str}.")
            metrics = signal_validator.calculate_overall_metrics()
            st.write(f"Overall Accuracy: {metrics['accuracy']:.2f}%, Total P&L: {metrics['total_pnl']:.2f}, Hit Rate: {metrics['hit_rate']:.2f}%")
            
            # Update learning model weights
            learning_module.update_weights()
            st.success("Self-learning model weights updated.")
        else:
            st.info(f"No signals were validated for {current_date_str}.")
    elif market_manager.is_holiday(market_manager.get_current_market_time().date()):
        st.info(f"Today ({current_date_str}) is a market holiday. Signal validation deferred until the next trading day.")
    else:
        st.info(f"Market is still open or not yet closed for the day. Signal validation will run after {market_manager.market_close_time.strftime('%I:%M %p IST')}.")

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

    # Initialize core components
    db_manager = SignalDatabaseManager()
    market_manager = MarketTimingManager()
    signal_validator = SignalValidator(db_manager, market_manager)
    learning_module = SelfLearningModule(db_manager)

    # Update learning weights at startup based on historical data
    learning_module.update_weights()
    logging.info("Self-learning module initialized and weights updated from historical data.")

    # Initialize session state variables if not already present
    if 'all_stocks_data' not in st.session_state:
        st.session_state['all_stocks_data'] = {}
    if 'news_df' not in st.session_state:
        st.session_state['news_df'] = pd.DataFrame()
    if 'btst_signals' not in st.session_state:
        st.session_state['btst_signals'] = []
    if 'options_signals' not in st.session_state:
        st.session_state['options_signals'] = []
    if 'ranked_btst_signals' not in st.session_state:
        st.session_state['ranked_btst_signals'] = []
    if 'ranked_options_signals' not in st.session_state:
        st.session_state['ranked_options_signals'] = []
    if 'top_n_recommendations' not in st.session_state:
        st.session_state['top_n_recommendations'] = 5 # Default value
    if 'signal_type_filter' not in st.session_state:
        st.session_state['signal_type_filter'] = "Both" # Default to show both

    # Sidebar for navigation or controls
    st.sidebar.title("Controls")
    
    # Slider for number of top recommendations
    st.session_state['top_n_recommendations'] = st.sidebar.slider(
        "Number of Top Recommendations to Display (N)",
        min_value=1,
        max_value=20,
        value=st.session_state['top_n_recommendations'],
        step=1
    )

    # Selectbox for filtering signal types (CALL, PUT, or Both)
    st.session_state['signal_type_filter'] = st.sidebar.selectbox(
        "Filter Signal Type:",
        options=["Both", "CALL", "PUT"],
        index=0 # Default to "Both"
    )

    # Display market status banner
    current_market_time = market_manager.get_current_market_time()
    if market_manager.is_holiday(current_market_time.date()):
        st.warning(f"Market Closed: Today ({current_market_time.strftime('%Y-%m-%d')}) is a market holiday.")
    elif not market_manager.is_market_open():
        st.warning(f"Market Closed: Current time ({current_market_time.strftime('%H:%M IST')}) is outside trading hours.")
    else:
        st.success(f"Market Open: Current time ({current_market_time.strftime('%H:%M IST')}).")

    if st.sidebar.button("Run Analysis Now", key="run_analysis_button"):
        run_analysis(db_manager, market_manager, signal_validator, learning_module)

    st.sidebar.subheader("Schedule Analysis")
    schedule_option = st.sidebar.radio(
        "Choose a schedule:",
        ("Manual", "Pre-Close (3:00 PM IST)", "Open (9:15 AM IST)", "Post-Close Validation (3:45 PM IST)")
    )

    if schedule_option == "Pre-Close (3:00 PM IST)":
        schedule.every().day.at("15:00").do(run_analysis, db_manager, market_manager, signal_validator, learning_module)
        st.sidebar.info("Scheduled daily analysis for 3:00 PM IST.")
    elif schedule_option == "Open (9:15 AM IST)":
        schedule.every().day.at("09:15").do(run_analysis, db_manager, market_manager, signal_validator, learning_module)
        st.sidebar.info("Scheduled daily analysis for 9:15 AM IST.")
    elif schedule_option == "Post-Close Validation (3:45 PM IST)":
        schedule.every().day.at("15:45").do(run_analysis, db_manager, market_manager, signal_validator, learning_module)
        st.sidebar.info("Scheduled daily post-close validation for 3:45 PM IST.")
    
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

    st.header(f"Top {st.session_state['top_n_recommendations']} Generated Signals")
    # Filter options signals based on the selected type from the sidebar
    filtered_options_signals = []
    if 'ranked_options_signals' in st.session_state and st.session_state['ranked_options_signals']:
        if st.session_state['signal_type_filter'] == "Both":
            filtered_options_signals = st.session_state['ranked_options_signals']
        else:
            # Filter by option_type (CE/PE) based on user's selection
            filtered_options_signals = [
                s for s in st.session_state['ranked_options_signals']
                if s.get('option_type') == st.session_state['signal_type_filter'].upper()
            ]

    # Filter BTST signals based on the selected type from the sidebar
    filtered_btst_signals = []
    if 'ranked_btst_signals' in st.session_state and st.session_state['ranked_btst_signals']:
        if st.session_state['signal_type_filter'] == "Both":
            filtered_btst_signals = st.session_state['ranked_btst_signals']
        else:
            # Filter by signal_type (BTST_CALL/BTST_PUT) based on user's selection
            filtered_btst_signals = [
                s for s in st.session_state['ranked_btst_signals']
                if s.get('signal_type', '').replace('BTST_', '') == st.session_state['signal_type_filter'].upper()
            ]

    if filtered_btst_signals:
        st.subheader("BTST Signals")
        btst_df = pd.DataFrame(filtered_btst_signals)
        # Display relevant columns for BTST, including new option details
        display_cols_btst = ['symbol', 'signal_type', 'action', 'entry_price', 'target_price', 'strike_price', 'expiry_date', 'option_symbol', 'confidence_score', 'rationale']
        # Filter columns to only include those present in the DataFrame
        btst_df_display = btst_df[[col for col in display_cols_btst if col in btst_df.columns]]
        
        # Rename 'signal_type' to 'Type' for better UI if present
        if 'signal_type' in btst_df_display.columns:
            btst_df_display.rename(columns={'signal_type': 'Type'}, inplace=True)
        
        st.dataframe(btst_df_display, hide_index=True)
    else:
        # Update the message to be specific to BTST CALL/PUT
        if st.session_state['signal_type_filter'] == "Both":
            st.info("No BTST signals generated yet or none met the ranking criteria. Run analysis to generate signals.")
        else:
            st.info(f"No BTST {st.session_state['signal_type_filter']} signals generated yet or none met the ranking criteria. Run analysis to generate signals.")

    if filtered_options_signals:
        st.subheader("Options Signals")
        options_df = pd.DataFrame(filtered_options_signals)
        # Add a 'Type' column for display, using 'option_type' (CE/PE)
        if 'option_type' in options_df.columns:
            options_df.rename(columns={'option_type': 'Type'}, inplace=True)
        else:
            options_df['Type'] = 'N/A' # Fallback if option_type not set
        st.dataframe(options_df, hide_index=True)
    else:
        st.info(f"No {st.session_state['signal_type_filter']} Options signals generated yet or none met the ranking criteria. Run analysis to generate signals.")
