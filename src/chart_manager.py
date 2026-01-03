import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from .logger import setup_logger

logger = setup_logger(__name__)

class ChartManager:
    """
    Manages the creation of interactive charts using Plotly.
    """
    def __init__(self):
        pass

    def create_candlestick_chart(self, df, symbol, indicators=None):
        """
        Creates an interactive candlestick chart with indicators.
        
        Args:
            df (pd.DataFrame): Dataframe with Open, High, Low, Close, Volume, and Date index.
            symbol (str): Stock symbol.
            indicators (list, optional): List of column names to overlay (e.g., ['SMA_20', 'SMA_50']).
            
        Returns:
            plotly.graph_objects.Figure: The updated figure object.
        """
        if df is None or df.empty:
            logger.error(f"No data provided for chart: {symbol}")
            return None

        # Create subplots: Row 1 for Price, Row 2 for Volume
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.03, subplot_titles=(f'{symbol} Price Action', 'Volume'),
                            row_width=[0.2, 0.7])

        # Candlestick Trace
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='OHLC'
        ), row=1, col=1)

        # Overlays (Indicators)
        if indicators:
            colors = ['blue', 'orange', 'purple', 'green', 'black']
            for i, ind in enumerate(indicators):
                if ind in df.columns:
                    color = colors[i % len(colors)]
                    fig.add_trace(go.Scatter(
                        x=df.index, 
                        y=df[ind], 
                        line=dict(color=color, width=1.5), 
                        name=ind
                    ), row=1, col=1)
                else:
                    logger.warning(f"Indicator {ind} not found in dataframe for {symbol}")

        # Support/Resistance Lines (if available)
        if 'Support' in df.columns:
             fig.add_trace(go.Scatter(x=df.index, y=df['Support'], line=dict(color='green', dash='dot'), name='Support'), row=1, col=1)
        if 'Resistance' in df.columns:
             fig.add_trace(go.Scatter(x=df.index, y=df['Resistance'], line=dict(color='red', dash='dot'), name='Resistance'), row=1, col=1)

        # Volume Trace
        # Color volume bars based on price change (Green if Close > Open, else Red)
        colors = ['green' if row['Close'] > row['Open'] else 'red' for index, row in df.iterrows()]
        fig.add_trace(go.Bar(
            x=df.index, 
            y=df['Volume'], 
            marker_color=colors,
            name='Volume'
        ), row=2, col=1)

        # Layout updates
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            height=600,
            template="plotly_white",
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode="x unified"
        )
        
        # Specific Y-axis formatting
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)

        logger.info(f"Created chart for {symbol}")
        return fig
