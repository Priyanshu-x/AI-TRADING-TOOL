# AI Trading Signal Tool (Advanced)

## Overview
This is an **Industry-Level AI Trading Application** designed for the Indian Stock Market (NSE).
It provides:
1.  **Live Signals**: Real-time analysis of stocks using Technicals (Breakouts, moving averages) and News Sentiment.
2.  **Interactive Dashboard**: A Streamlit UI with Live Monitoring, Charts, and Risk Management.
3.  **Backtesting**: Validate strategies on historical data before trading.
4.  **Paper Trading**: Simulate live trades with a virtual portfolio (Forward Testing).

## Features
-   **Multi-Strategy**: Supports **BTST** (Buy Today Sell Tomorrow) and **Options** (Call/Put) strategies.
-   **News Sentiment**: Scrapes Google News to "filter" signals (e.g., only Buy if sentiment is Positive).
-   **Risk Manager**: accurate Position Sizing calculator based on your account size and risk tolerance.
-   **Vectorized Backtester**: High-performance engine to test strategies over years of data.
-   **Virtual Portfolio**: Track cash, open positions, and P&L in a simulated environment.

## Installation

### Prerequisites
-   Python 3.10+
-   Git

### Setup
1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/ai-trading-tool.git
    cd ai-trading-tool
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App**:
    ```bash
    streamlit run src/main.py
    ```

## Usage Guide

The application has three modes, selectable from the Sidebar:

### 1. Live Analysis
-   **Purpose**: Day-to-day trading.
-   **How to use**: 
    -   Click "Run Analysis Now".
    -   View the "BTST Signals" and "Options Signals" tables.
    -   Expand "Charts" to see the setup.
    -   Enable "Live Monitor" to auto-refresh every 60 seconds.

### 2. Backtesting
-   **Purpose**: Research and Validation.
-   **How to use**:
    -   Select "Backtesting" from the Sidebar.
    -   Choose a stock (e.g., `RELIANCE.NS`).
    -   Set your Strategy Parameters (Stop Loss %, Target %).
    -   Click "Run Backtest".
    -   Analyze the **Equity Curve** and **Metrics** (Sharpe Ratio, Win Rate).

### 3. Paper Trading (Forward Testing)
-   **Purpose**: Practice without risk.
-   **How to use**:
    -   Select "Paper Trading".
    -   View your **Virtual Cash** (starts at ₹1,00,000).
    -   Use the "Manual Trade Entry" to place test trades (or automate it in future).
    -   Watch your P&L update in real-time as prices change.

## Configuration
-   **Watchlist**: Edit `config/config.yaml` to add/remove stocks.
-   **Risk Settings**: Adjust "Account Size" and "Risk %" in the Sidebar.

## Disclaimer
**This tool is for educational and research purposes only.**
Trading stocks and options involves significant risk of loss. The AI signals are generated based on algorithms and historical patterns, which do not guarantee future performance. Always do your own due diligence.