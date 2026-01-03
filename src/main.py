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
from src.chatbot import AIChatbot


def get_indicator_calculator_funcs():
    from src.indicator_calculator import (
        calculate_moving_averages,
        calculate_atr,
        calculate_volume_spikes,
        find_swing_points,
        detect_breakouts_breakdowns
    )
    return {
        'calculate_moving_averages': calculate_moving_averages,
        'calculate_atr': calculate_atr,
        'calculate_volume_spikes': calculate_volume_spikes,
        'find_swing_points': find_swing_points,
        'detect_breakouts_breakdowns': detect_breakouts_breakdowns,
    }
from src.signal_database_manager import SignalDatabaseManager
from src.market_timing_manager import MarketTimingManager
from src.signal_validator import SignalValidator
from src.self_learning_module import SelfLearningModule
from src.logger import setup_logger
from src.chart_manager import ChartManager
from src.risk_manager import RiskManager
import time
import pytz

# Configure structured logging
logger = setup_logger(__name__)

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
        f.write("\n")
    logger.info(f"Daily report saved to {report_filename}")

def run_analysis(db_manager, market_manager, signal_validator, learning_module, chart_manager=None, risk_manager=None):
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
            manager.print_watchlist() # Helper prints internally, can refactor later
        else:
            logger.error("Watchlist validation failed. Please check config file.")
    else:
        logger.error("Failed to load watchlist. Exiting.")
        return

    all_stocks_data = {}
    if 'all_stocks_data' not in st.session_state:
        st.session_state['all_stocks_data'] = {}

    # Initialize TradeSignalGenerator, injecting the learning module
    signal_generator = TradeSignalGenerator(learning_module=learning_module)

    # Fetch all tickers from watchlist
    all_tickers = get_all_watchlist_tickers()
    if not all_tickers:
        logger.error("No tickers found in watchlist. Exiting.")
        st.error("No tickers found in watchlist. Please check config/config.yaml.")
        return

    # Fetch and save historical data for all tickers
    with st.spinner(f"Fetching data for {len(all_tickers)} tickers..."):
        download_summary = fetch_stock_data(all_tickers)
    
    logger.info("--- Data Download Summary ---")
    
    # Check for total failure
    failed_count = sum(1 for info in download_summary.values() if info['status'] != 'success')
    if failed_count == len(all_tickers):
        st.error("Critical Error: Failed to fetch data for ALL tickers. Please check your internet connection or Yahoo Finance availability.")
    
    # Debug info for user
    with st.expander("Debug: Data Fetch Details", expanded=False):
        st.json(download_summary)

    for ticker, info in download_summary.items():
        logger.info(f"{ticker}: Status - {info['status']}, Records - {info['records']}")

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
                    
                    logger.info(f"Calculating indicators for {symbol}...")
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
                    logger.info(f"Indicators calculated for {symbol}.")
                else:
                    logger.warning(f"No data loaded from {file_path} for {symbol}. Skipping indicator calculation.")
            except Exception as e:
                logger.error(f"Error loading or processing data for {symbol} from {file_path}: {e}")
        else:
            logger.warning(f"Skipping indicator calculation for {symbol}: Data download failed or not available.")
    
    logger.debug(f"\n--- all_stocks_data content after indicator calculation: {list(all_stocks_data.keys())} ---")

    # Initialize NewsScraper
    scraper = NewsScraper()

    # Scrape news for each stock in the watchlist
    logger.info("--- Starting News Scraping ---")
    for index_name, stocks in manager.watchlist.items():
        logger.info(f"Scraping news for {index_name}...")
        for stock in stocks:
            symbol = stock.get('symbol')
            name = stock.get('name')
            if symbol and name:
                # scraper.scrape_moneycontrol(symbol, name) # Temporarily commented out due to 403/401 errors
                # scraper.scrape_reuters(symbol, name) # Temporarily commented out due to 403/401 errors
                scraper.scrape_google_news(symbol, name)
            else:
                logger.warning(f"Skipping stock due to missing symbol or name: {stock}")

    # Get news as DataFrame and save to CSV
    news_df = scraper.get_news_as_dataframe()
    if not news_df.empty:
        logger.info("--- Collected News Headlines (first 5) ---")
        # logger.info(news_df.head()) # Don't log entire dataframe head to JSON logs
        scraper.save_news_to_csv()
        st.session_state['news_df'] = news_df # Store news_df in session state
    else:
        logger.info("No news headlines collected.")
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
        logger.info(f"Generated {len(btst_signals)} total BTST signals. Displaying top {len(ranked_btst_signals)}.")
        st.subheader("Top BTST Signals")
        for signal in ranked_btst_signals:
            with st.expander(f"BTST {signal.get('signal_type', signal['action'])} | {signal['symbol']} | Conf: {signal['confidence_score']:.2f}", expanded=True):
                # Columns for details and chart
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.write(f"**Action:** {signal.get('signal_type', signal['action'])}")
                    st.write(f"**Entry:** {signal['entry_price']}")
                    st.write(f"**Target:** {signal['target_price']}")
                    
                    # Risk Calc
                    if risk_manager:
                        # Assuming stop loss at some % below/above entry for calculation if not in signal
                        # Basic assumption: 1:2 Risk Reward, so risk is half of reward distance
                        reward = abs(signal['target_price'] - signal['entry_price'])
                        risk_dist = reward / 2
                        stop_loss = signal['entry_price'] - risk_dist if "BUY" in signal['action'] else signal['entry_price'] + risk_dist
                        
                        size_info = risk_manager.calculate_position_size(signal['entry_price'], stop_loss)
                        if size_info:
                            st.write(f"**Risk Mgmt:** Buy {size_info['quantity']} qty (Risk: ₹{size_info['total_risk_amount']})")
                            st.caption(f"SL est: {stop_loss:.2f}")

                    st.write(f"**Rationale:** {signal.get('rationale', 'N/A')}")

                with col2:
                    if chart_manager and signal['symbol'] in all_stocks_data:
                         fig = chart_manager.create_candlestick_chart(all_stocks_data[signal['symbol']], signal['symbol'], indicators=['SMA_20', 'SMA_50'])
                         if fig:
                             st.plotly_chart(fig, use_container_width=True)

    else:
        logger.info("No BTST signals generated or none met the ranking criteria.")
        st.info("No BTST signals generated.")

    if ranked_options_signals:
        logger.info(f"Generated {len(options_signals)} Options signals. Displaying top {len(ranked_options_signals)}.")
        st.subheader("Top Options Signals")
        for signal in ranked_options_signals:
             with st.expander(f"OPTIONS {signal['action']} | {signal['symbol']} | Strike: {signal['strike_price']}", expanded=True):
                 st.write(f"**Contract:** {signal.get('option_symbol', 'N/A')} ({signal['expiry_date']})")
                 st.write(f"**Confidence:** {signal['confidence_score']:.2f}")
                 st.write(f"**Rationale:** {signal.get('rationale', 'N/A')}")
                 if chart_manager and signal['symbol'] in all_stocks_data:
                         fig = chart_manager.create_candlestick_chart(all_stocks_data[signal['symbol']], signal['symbol'], indicators=['SMA_20', 'ATR'])
                         if fig:
                             st.plotly_chart(fig, use_container_width=True)
    else:
        logger.info("No Options signals generated or none met the ranking criteria.")
        st.info("No Options signals generated.")

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

    logger.info("--- Indicator Report ---")
    if all_stocks_data:
        for symbol, df in all_stocks_data.items():
            # logger.info(f"Report for {symbol}:") # Reduce verbosity
            if not df.empty:
                latest_data = df.iloc[-1]
                # Log detailed report only for debugging if needed, or structured
                logger.debug(f"Indicator Report for {symbol}: Close={latest_data['Close']:.2f}, ATR={latest_data['ATR']:.2f}, Breakout={latest_data['Breakout']}")
            else:
                logger.warning(f"  No data available for {symbol}.")
    else:
        st.info("No stock data processed for indicator reports.")

    # Update chatbot's data references after analysis
    if 'chatbot' in st.session_state:
        st.session_state['chatbot'].update_data_references(st.session_state['all_stocks_data'], st.session_state['news_df'])

    logger.info("AI Trading Signal Tool finished successfully.")


if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="AI Trading Signal Tool")

    # Initialize Managers
    db_manager = SignalDatabaseManager()
    market_manager = MarketTimingManager()
    signal_validator = SignalValidator(db_manager, market_manager)
    learning_module = SelfLearningModule(db_manager)

    # Sidebar Controls
    st.sidebar.title("Configuration")
    
    # Live Monitor Toggle
    live_monitor = st.sidebar.checkbox("Enable Live Monitoring", value=False)
    refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 60, 300, 60)
    
    # Risk Management Controls
    st.sidebar.subheader("Risk Management")
    account_size = st.sidebar.number_input("Account Size (INR)", value=100000, step=10000)
    risk_pct = st.sidebar.number_input("Risk Per Trade (%)", value=1.0, step=0.1)
    
    # Initialize Managers
    chart_manager = ChartManager()
    risk_manager = RiskManager(default_account_size=account_size, default_risk_per_trade_pct=risk_pct)
    
    from src.backtester import BacktestEngine
    backtester = BacktestEngine(initial_capital=account_size)
    
    from src.paper_trader import PaperTradingEngine
    paper_trader = PaperTradingEngine(db_manager)

    # --- Mode Selection ---
    mode = st.sidebar.selectbox("App Mode", ["Live Analysis", "Backtesting", "Paper Trading"])

    if st.sidebar.button("Run Analysis Now", key="sidebar_run_btn") and mode == "Live Analysis":
        learning_module.update_weights()
        logger.info("Self-learning module initialized and weights updated from historical data.")
        run_analysis(db_manager, market_manager, signal_validator, learning_module, chart_manager, risk_manager)

    # Main content area
    st.title("AI Trading Signal Tool (Industry Level)")

    # Handle Live Monitoring Loop (Only in Live Analysis Mode)
    if mode == "Live Analysis":
        if live_monitor:
            if market_manager.is_market_open():
                st.info(f"Live Monitoring Active. Refreshing every {refresh_interval} seconds...")
                st.empty() # Placeholder for countdown or status
                
                # Main Analysis Logic (Wrapped for reuse)
                with st.spinner("Fetching latest market data..."):
                    run_analysis(db_manager, market_manager, signal_validator, learning_module, chart_manager, risk_manager)
                
                time.sleep(refresh_interval)
                st.rerun()
            else:
                st.warning("Market is closed. Real-time monitoring is paused.")
                if st.button("Run Offline Analysis (Backtest/Review)"):
                        run_analysis(db_manager, market_manager, signal_validator, learning_module, chart_manager, risk_manager)
        else:
                if st.button("Run Analysis"):
                # Update learning weights at startup based on historical data
                    learning_module.update_weights()
                    logger.info("Self-learning module initialized and weights updated from historical data.")
                    run_analysis(db_manager, market_manager, signal_validator, learning_module, chart_manager, risk_manager)
    
    elif mode == "Backtesting":
        st.header("Strategy Backtesting")
        
        # Backtest inputs
        col1, col2 = st.columns(2)
        with col1:
             bt_symbol = st.selectbox("Select Stock", get_all_watchlist_tickers())
        with col2:
             bt_days = st.number_input("Days to Backtest", min_value=30, max_value=2000, value=365)
        
        with st.expander("Strategy Parameters"):
            stop_loss = st.slider("Stop Loss %", 1.0, 10.0, 2.0) / 100
            target = st.slider("Target %", 1.0, 20.0, 4.0) / 100
        
        if st.button("Run Backtest"):
            with st.spinner(f"Running backtest for {bt_symbol}..."):
                # Fetch data
                summary = fetch_stock_data([bt_symbol], start_date=datetime.now()-timedelta(days=bt_days))
                if summary[bt_symbol]['status'] == 'success':
                    df = pd.read_csv(summary[bt_symbol]['file'], parse_dates=['Date'])
                    df.set_index('Date', inplace=True)
                    # Numeric cleanup
                    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Run Backtest
                    metrics = backtester.run_backtest(df, {'stop_loss_pct': stop_loss, 'target_pct': target})
                    
                    if metrics:
                         # Display Metrics
                         m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                         m_col1.metric("Total Return", metrics['Total Return'])
                         m_col2.metric("Sharpe Ratio", metrics['Sharpe Ratio'])
                         m_col3.metric("Max Drawdown", metrics['Max Drawdown'])
                         m_col4.metric("Win Rate", metrics['Win Rate'])
                         
                         # Plot Equity Curve
                         fig = backtester.plot_equity_curve()
                         st.plotly_chart(fig, use_container_width=True)
                         
                         # Show Trade Log
                         with st.expander("Trade Log"):
                             st.dataframe(pd.DataFrame(backtester.trade_log))
                    else:
                        st.error("Backtest failed to produce metrics (possibly no data or no trades).")
                else:
                    st.error(f"Failed to fetch data: {summary[bt_symbol].get('reason')}")
    
    elif mode == "Paper Trading":
        st.header("Simulated Paper Trading (Forward Testing)")
        
        # 1. Portfolio Summary
        # Fetch live prices for open positions to calculate real-time P&L
        open_orders = db_manager.get_open_orders()
        open_symbols = list(set([o['symbol'] for o in open_orders]))
        
        live_prices = {}
        if open_symbols:
            with st.spinner("Fetching live prices for portfolio..."):
                 summary = fetch_stock_data(open_symbols)
                 for sym, info in summary.items():
                     if info['status'] == 'success':
                         # Read last line of CSV for latest price
                         df = pd.read_csv(info['file'])
                         live_prices[sym] = df.iloc[-1]['Close']

        portfolio = paper_trader.get_portfolio_status(live_prices)
        
        # Display Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Virtual Cash", f"₹{portfolio['Cash']:,.2f}")
        col2.metric("Equity Value", f"₹{portfolio['Equity']:,.2f}")
        col3.metric("Unrealized P&L", f"₹{portfolio['Unrealized PnL']:,.2f}", 
                    delta_color="normal" if portfolio['Unrealized PnL'] >= 0 else "inverse")
        col4.metric("Open Positions", len(portfolio['Positions']))
        
        # 2. Open Positions Table
        st.subheader("Open Positions")
        if portfolio['Positions']:
            st.dataframe(pd.DataFrame(portfolio['Positions']))
        else:
            st.info("No open positions.")
            
        # 3. Trade Execution (Manual from Signals)
        st.subheader("Available Signals to Trade")
        # Fetch latest signals from DB to allow trading
        latest_date = datetime.now().strftime("%Y-%m-%d")
        # We need a way to get today's signals. For now, let's run a quick analysis check or fetch from DB if stored.
        # Ideally, we should fetch from DB. 
        # For this prototype, let's provide a button to "Scan & Trade"
        
        if st.button("Scan for New Signals"):
             run_analysis(db_manager, market_manager, signal_validator, learning_module, chart_manager, risk_manager)
        
        # Logic to display 'Execute' button next to signals is tricky in Streamlit's loop.
        # Easier approach: Display list of recent signals and a "Trade" button next to each?
        # Or: Rely on the "Live Analysis" tab to see signals, but validation/execution is here.
        # Let's add a simple "Quick Trade" box for now just to test the engine.
        
        with st.expander("Manual Trade Entry (Debug/Test)"):
            t_sym = st.selectbox("Symbol", get_all_watchlist_tickers(), key="pt_sym")
            t_qty = st.number_input("Quantity", 1, 1000, 10, key="pt_qty")
            t_price = st.number_input("Price", 0.0, 10000.0, 0.0, key="pt_price") # 0.0 = Market (fetch)
            if st.button("Execute Paper Order"):
                if t_price == 0.0:
                    # Fetch price
                    summ = fetch_stock_data([t_sym])
                    if summ[t_sym]['status'] == 'success':
                        df = pd.read_csv(summ[t_sym]['file'])
                        t_price = df.iloc[-1]['Close']
                
                signal = {"symbol": t_sym, "action": "BUY", "entry_price": t_price}
                res = paper_trader.place_order(signal, t_qty)
                if res['status'] == 'success':
                    st.success(f"Order Placed! ID: {res['order_id']}")
                    st.rerun()
                else:
                    st.error(f"Failed: {res.get('reason')}")

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

    # How this works info tooltip
    with st.sidebar.expander("How this works"):
        st.markdown(
            """
            This AI Trading Signal Tool provides insights and trade signals based on various market data and indicators.

            **Controls & Workflows:**

            1.  **Watchlist Management:** Define your stocks in `config/config.yaml`. The tool loads and validates this list.
            2.  **Data Fetching:** Historical stock data is fetched for all watchlist tickers.
            3.  **Indicator Calculation:** Technical indicators (e.g., Moving Averages, ATR, Volume Spikes, Support/Resistance, Breakouts) are calculated from the fetched data.
            4.  **News Scraping & Sentiment:** Latest news is scraped for watchlist stocks, and sentiment analysis is performed.
            5.  **Signal Generation:** Based on technical indicators and news sentiment, BTST (Buy Today Sell Tomorrow) and Options trade signals are generated.
            6.  **Top Recommendations:** Signals are ranked by confidence, and the top 'N' (configurable via slider) are displayed.
            7.  **Scheduling:** You can run the analysis manually or schedule it for specific market times (Pre-Open, Open, Post-Close for validation).
            8.  **Signal Validation & Learning:** After market close, generated signals are validated against actual market movements, and the AI's learning module updates its weights to improve future signal accuracy.
            9.  **AI Chatbot:** Interact with the AI to ask questions about market data, signals, or the tool's functionality. The chatbot uses the latest processed data.

            **Output:**

            *   A daily report (`outputs/daily_signals_report_YYYY-MM-DD.txt`) summarizing top signals.
            *   Signals are stored in a database for historical tracking and learning.
            """
        )

    # Initialize WatchlistManager (moved here to be available for chatbot)
    manager = WatchlistManager()
    if manager.load_watchlist() and manager.validate_watchlist():
        # Chatbot Initialization
        # We need an instance of TradeSignalGenerator to pass to the Chatbot
        # For initial chatbot setup, we can pass a dummy/None if signals aren't needed immediately in chat.
        # However, for full functionality, it needs the actual signal_generator.
        # Let's re-initialize signal_generator here for main to use, and pass it to chatbot.
        # The learning_module is already initialized.
        signal_generator = TradeSignalGenerator(learning_module=learning_module)
        indicator_calculator_funcs = get_indicator_calculator_funcs()

        try:
            chatbot = AIChatbot(
                db_manager=db_manager,
                market_manager=market_manager,
                signal_validator=signal_validator,
                learning_module=learning_module,
                watchlist_manager=manager,
                news_scraper=NewsScraper(), # Create a new instance for the chatbot
                data_fetcher_func=fetch_stock_data,
                indicator_calculator_funcs=indicator_calculator_funcs,
                trade_signal_generator=signal_generator
            )
            st.session_state['chatbot'] = chatbot
            st.session_state['watchlist_manager'] = manager # Store manager in session state
        except Exception as e:
            st.error(f"Error initializing chatbot: {e}")
            logger.error(f"Error initializing chatbot: {e}")
            st.stop() # Stop the script execution here if chatbot fails to initialize

        # Main content area
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
        # If watchlist fails, prevent further execution that relies on it
        st.stop() # Stop the script execution here if critical components fail


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

    # Chatbot Interface
    st.header("AI Chatbot")
    if 'chatbot_messages' not in st.session_state:
        st.session_state['chatbot_messages'] = []

    for message in st.session_state['chatbot_messages']:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask the AI about trading signals, market, or the tool:"):
        st.session_state['chatbot_messages'].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Ensure the chatbot is initialized and available
            if 'chatbot' in st.session_state:
                try:
                    response = st.session_state['chatbot'].chat(prompt)
                    st.markdown(response)
                    st.session_state['chatbot_messages'].append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error during AI chat: {e}")
                    print(f"Error during AI chat: {e}")
            else:
                st.error("Chatbot not initialized. Please refresh the page.")
