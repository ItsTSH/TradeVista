import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from lightweight_charts import Chart

# App title and description
st.title("TradeVista Stock Analysis")
st.markdown("Enter a stock ticker to view chart and add technical indicators")

# Main page input controls
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    ticker = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL, MSFT)", value="AAPL")

with col2:
    period_options = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"]
    time_period = st.select_slider("Select Time Period", options=period_options, value="1y")

with col3:
    interval_options = ["1d", "1wk", "1mo"]
    time_interval = st.select_slider("Select Interval", options=interval_options, value="1d")

fetch_button = st.button("Fetch Stock Data", use_container_width=True)

# Function to calculate indicators
def calculate_indicators(df):
    # Simple Moving Averages
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    
    # Exponential Moving Averages
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    # Bollinger Bands (20-day, 2 standard deviations)
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    std_dev = df['Close'].rolling(window=20).std().squeeze()
    df['BB_upper'] = df['BB_middle'] + (std_dev * 2)
    df['BB_lower'] = df['BB_middle'] - (std_dev * 2)
    
    # RSI (14-day)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = -delta.where(delta < 0, 0).fillna(0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal']
    
    return df

# Function to create OHLC data format for lightweight_charts
def prepare_chart_data(df):
    chart_data = []
    for index, row in df.iterrows():
        chart_data.append({
            'time': index.timestamp(),
            'open': row['Open'],
            'high': row['High'],
            'low': row['Low'],
            'close': row['Close'],
            'volume': row['Volume']
        })
    return chart_data

# Function to create line data for indicators
def prepare_indicator_data(df, column):
    indicator_data = []
    for index, row in df.iterrows():
        if pd.notna(row[column]):  # Check if value is not NaN
            indicator_data.append({
                'time': index.timestamp(),
                'value': float(row[column])  # Ensure it's a float
            })
    return indicator_data

# Main content area
main_container = st.container()

# Global variable to store data
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None

# Fetch data when the button is clicked
if fetch_button:
    try:
        with st.spinner(f'Fetching data for {ticker.upper()}...'):
            # Get stock data from Yahoo Finance
            stock_data = yf.download(ticker.upper(), period=time_period, interval=time_interval)
            
            if stock_data.empty:
                st.error(f"No data found for ticker {ticker.upper()}. Please check the ticker symbol.")
            else:
                # Calculate indicators
                stock_data = calculate_indicators(stock_data)
                st.session_state.stock_data = stock_data
                st.success(f"Successfully fetched data for {ticker.upper()}")
    except Exception as e:
        st.error(f"Error fetching data: {e}")

# Display chart and indicators if data is available
if st.session_state.stock_data is not None:
    with main_container:
        df = st.session_state.stock_data
        
        # Indicator selection
        st.subheader(f"{ticker.upper()} Stock Chart")
        
        indicators = st.multiselect(
            "Select Indicators to Display",
            [
                "SMA20", "SMA50", "SMA200", 
                "EMA20", "EMA50",
                "Bollinger Bands",
                "RSI", 
                "MACD"
            ],
            default=[]
        )
        
        # Prepare chart data
        chart_data = prepare_chart_data(df)
        
        # Create chart with lightweight_charts
        chart = Chart(width=1000, height=600)
        
        # Add candlestick series
        candlestick = chart.create_candlestick_series()
        candlestick.set_data(chart_data)
        
        # Add volume series (at the bottom of the chart)
        volume = chart.create_histogram_series(color='rgba(76, 175, 80, 0.5)', price_source='volume', price_scale_id='volume')
        volume.set_data(chart_data)
        
        # Add indicators based on user selection
        indicator_colors = {
            "SMA20": "rgba(255, 140, 0, 1)",
            "SMA50": "rgba(76, 175, 80, 1)",
            "SMA200": "rgba(75, 0, 130, 1)",
            "EMA20": "rgba(255, 0, 0, 1)",
            "EMA50": "rgba(0, 128, 255, 1)",
            "BB_upper": "rgba(169, 169, 169, 0.7)",
            "BB_middle": "rgba(169, 169, 169, 1)",
            "BB_lower": "rgba(169, 169, 169, 0.7)",
            "RSI": "rgba(75, 192, 192, 1)",
            "MACD": "rgba(33, 150, 243, 1)",
            "Signal": "rgba(255, 82, 82, 1)",
            "MACD_Histogram": "rgba(128, 128, 128, 0.5)"
        }
        
        for indicator in indicators:
            if indicator == "Bollinger Bands":
                # Add Bollinger Bands as area and line
                bb_upper = chart.create_line_series(color=indicator_colors["BB_upper"], name="BB Upper")
                bb_upper.set_data(prepare_indicator_data(df, "BB_upper"))
                
                bb_middle = chart.create_line_series(color=indicator_colors["BB_middle"], name="BB Middle")
                bb_middle.set_data(prepare_indicator_data(df, "BB_middle"))
                
                bb_lower = chart.create_line_series(color=indicator_colors["BB_lower"], name="BB Lower")
                bb_lower.set_data(prepare_indicator_data(df, "BB_lower"))
            
            elif indicator == "MACD":
                # Create a separate price scale for MACD at the bottom
                macd_series = chart.create_line_series(
                    color=indicator_colors["MACD"],
                    name="MACD",
                    price_scale_id='macd',
                    price_scale_options={
                        "position": "bottom",
                        "scaleMargins": {
                            "top": 0.8,
                            "bottom": 0
                        }
                    }
                )
                macd_series.set_data(prepare_indicator_data(df, "MACD"))
                
                signal_series = chart.create_line_series(
                    color=indicator_colors["Signal"],
                    name="Signal",
                    price_scale_id='macd'
                )
                signal_series.set_data(prepare_indicator_data(df, "Signal"))
                
                # Add histogram for MACD
                histogram = chart.create_histogram_series(
                    color=indicator_colors["MACD_Histogram"],
                    name="MACD Histogram",
                    price_scale_id='macd'
                )
                histogram_data = []
                for index, row in df.iterrows():
                    if pd.notna(row['MACD_Histogram']):
                        color = 'rgba(0, 150, 136, 0.5)' if row['MACD_Histogram'] > 0 else 'rgba(255, 82, 82, 0.5)'
                        histogram_data.append({
                            'time': index.timestamp(),
                            'value': float(row['MACD_Histogram']),  # Ensure it's a float
                            'color': color
                        })
                histogram.set_data(histogram_data)
            
            elif indicator == "RSI":
                # Create a separate price scale for RSI at the bottom
                rsi_series = chart.create_line_series(
                    color=indicator_colors["RSI"],
                    name="RSI",
                    price_scale_id='rsi',
                    price_scale_options={
                        "position": "bottom",
                        "scaleMargins": {
                            "top": 0.8,
                            "bottom": 0
                        }
                    }
                )
                rsi_series.set_data(prepare_indicator_data(df, "RSI"))
                
                # Add reference lines for RSI at 30 and 70
                chart.create_horizontal_line(price=30, color='rgba(255, 82, 82, 0.5)', price_scale_id='rsi')
                chart.create_horizontal_line(price=70, color='rgba(255, 82, 82, 0.5)', price_scale_id='rsi')
            
            else:
                # Add other indicators as line series
                indicator_series = chart.create_line_series(color=indicator_colors[indicator], name=indicator)
                indicator_series.set_data(prepare_indicator_data(df, indicator))
        
        # Display chart
        chart.render("TradeVista")
        
        # Stock info
        current_price = df['Close'].iloc[-1]
        previous_close = df['Close'].iloc[-2] if len(df) > 1 else None
        change = current_price - previous_close if previous_close else 0
        change_percent = (change / previous_close * 100) if previous_close else 0
        
        st.metric(
            label=f"{ticker.upper()}",
            value=f"${current_price:.2f}",
            delta=f"{change:.2f} ({change_percent:.2f}%)"
        )
        
        # Display data table
        with st.expander("View Data Table"):
            st.dataframe(df)