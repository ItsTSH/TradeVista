import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

def calculate_rsi(data, window=14):
  # Calculate RSI directly
  delta = data['Close'].diff()

  # Separate gains and losses
  gain = delta.where(delta > 0, 0)
  loss = -delta.where(delta < 0, 0)

  # Calculate average gain and loss
  avg_gain = gain.rolling(window=window, min_periods=1).mean()
  avg_loss = loss.rolling(window=window, min_periods=1).mean()

  # Calculate RS
  rs = avg_gain / avg_loss

  # Calculate RSI
  rsi = 100 - (100 / (1 + rs))

  # Convert to float and handle NaN
  return rsi.astype(float)

def calculate_mfi(data, window=14):
   # Calculate typical price
    typical_price = (data['High'].values + data['Low'].values + data['Close'].values) / 3
    volume = data['Volume'].values
    money_flow = typical_price * volume

    # Initialize arrays for positive and negative money flow
    positive_flow = np.zeros_like(typical_price)
    negative_flow = np.zeros_like(typical_price)

    # Calculate price difference manually instead of using np.diff
    price_diff = np.zeros_like(typical_price)
    for i in range(1, len(typical_price)):
        price_diff[i] = typical_price[i] - typical_price[i-1]

    # Calculate positive and negative money flow
    positive_mask = price_diff > 0
    negative_mask = price_diff < 0
    positive_flow[positive_mask] = money_flow[positive_mask]
    negative_flow[negative_mask] = money_flow[negative_mask]

    # Calculate rolling sums
    positive_mf = np.zeros_like(typical_price)
    negative_mf = np.zeros_like(typical_price)

    for i in range(len(typical_price)):
        start_idx = max(0, i - window + 1)
        positive_mf[i] = np.sum(positive_flow[start_idx:i+1])
        negative_mf[i] = np.sum(negative_flow[start_idx:i+1])

    # Handle division by zero
    epsilon = np.finfo(float).eps
    negative_mf[negative_mf == 0] = epsilon

    # Calculate Money Flow Index
    money_flow_ratio = positive_mf / negative_mf
    mfi = 100 - (100 / (1 + money_flow_ratio))

    return mfi.astype(float)

def check_condition(df, indicator, operator, value):
    """
    Check a single condition for the given DataFrame and indicator
    
    Args:
    - df (pd.DataFrame): DataFrame containing the stock data
    - indicator (str): Name of the indicator to check
    - operator (str): Comparison operator ('>', '<', '==')
    - value (float): Value to compare against
    
    Returns:
    - bool: Whether the condition is met
    """
    try:
        # Get the last value of the specified indicator
        last_value = df[indicator].iloc[-1]
        
        # Handle cases where the value might be None
        if last_value is None:
            return False
        
        # Perform the comparison based on the operator
        if operator == '>':
            return last_value > value
        elif operator == '<':
            return last_value < value
        elif operator == '==':
            return last_value == value
        
        return False
    except Exception as e:
        print(f"Error checking condition for {indicator}: {e}")
        return False

def stock_screener(percentile=0.7, conditions=[]):
    nifty50_CompanyTickers = [
        "ADANIENT.NS", "ADANIPORTS.NS", "ASIANPAINT.NS", "AXISBANK.NS", "BAJAJ-AUTO.NS",
        "BAJFINANCE.NS", "BAJAJFINSV.NS", "BHARTIARTL.NS", "BPCL.NS", "BRITANNIA.NS",
        "CIPLA.NS", "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS", "EICHERMOT.NS",
        "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS",
        "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "INDUSINDBK.NS", "INFY.NS",
        "ITC.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS", "M&M.NS",
        "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS",
        "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS", "SUNPHARMA.NS", "TCS.NS",
        "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "TECHM.NS", "TITAN.NS",
        "ULTRACEMCO.NS", "UPL.NS", "WIPRO.NS"
    ]

    nifty50_df = yf.Ticker("^NSEI").history(period="1y")
    nifty50_df['Pct Change'] = nifty50_df['Close'].pct_change()
    nifty50_return = (nifty50_df['Pct Change'] + 1).cumprod().iloc[-1]
    return_list = []

    final_df = pd.DataFrame(columns=['Ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 'Latest_Price', 'Score', 'PE_Ratio', 'PEG_Ratio', 'SMA_150', 'SMA_200', '52_Week_High', '52_Week_Low', 'RSI', 'MFI'])
    
    for ticker in nifty50_CompanyTickers:
        df = yf.Ticker(ticker).history(period="1y")
        df['Pct Change'] = df['Close'].pct_change()
        stock_return = (df['Pct Change'] + 1).cumprod().iloc[-1]
        returns_compared = round((stock_return / nifty50_return), 2)
        return_list.append(returns_compared)
    
    best_performers = pd.DataFrame(list(zip(nifty50_CompanyTickers, return_list)), columns=['Ticker', 'Returns Compared'])
    best_performers['Score'] = best_performers['Returns Compared'].rank(pct=True) * 100
    best_performers = best_performers[best_performers['Score'] >= best_performers['Score'].quantile(percentile)]
    
    for ticker in best_performers['Ticker']:
        try:
            df = yf.Ticker(ticker).history(period="1y")
            df['RSI'] = calculate_rsi(df, window=14)
            df['MFI'] = calculate_mfi(df, window=14)
            stock_info = yf.Ticker(ticker).info
            df['SMA_150'] = round(df['Close'].rolling(window=150).mean(), 2)
            df['SMA_200'] = round(df['Close'].rolling(window=200).mean(), 2)
            latest_price = df['Close'].iloc[-1]
            pe_ratio = stock_info.get('trailingPE', None)
            peg_ratio = stock_info.get('pegRatio', None)
            low_52_week = round(min(df['Low'][-(52*5):]), 2)
            high_52_week = round(max(df['High'][-(52*5):]), 2)
            rsi = round(df['RSI'].iloc[-1], 2)
            mfi = round(df['MFI'].iloc[-1], 2)
            score = round(best_performers[best_performers['Ticker'] == ticker]['Score'].tolist()[0])

            conditions_met = all(
                check_condition(df, cond['indicator'], cond['operator'], cond['value']) 
                for cond in conditions
            )           
            if conditions_met:
                final_df = pd.concat([final_df, pd.DataFrame([{
                    'Ticker': ticker,
                    'Open': df['Open'].iloc[-1],
                    'High': df['High'].iloc[-1],
                    'Low': df['Low'].iloc[-1],
                    'Close': df['Close'].iloc[-1],
                    'Volume': df['Volume'].iloc[-1],
                    'Latest_Price': latest_price,
                    'Score': score,
                    'PE_Ratio': pe_ratio,
                    'PEG_Ratio': peg_ratio,
                    'SMA_150': df['SMA_150'].iloc[-1],
                    'SMA_200': df['SMA_200'].iloc[-1],
                    '52_Week_Low': low_52_week,
                    '52_Week_High': high_52_week,
                    'RSI': rsi,
                    'MFI': mfi
                }])])
        except Exception as e:
            print(f"{e} in {ticker}")

    final_df.sort_values(by='Score', ascending=False)
    return final_df

st.title("📊 Stock Screener")
percentile = st.slider("Select percentile threshold", 0.1, 1.0, 0.7, 0.05)

conditions = []
indicators = ["RSI", "MFI", "PE_Ratio", "PEG_Ratio"]
operators = [">", "<", "==", "<=", ">="]

if 'conditions' not in st.session_state:
    st.session_state.conditions = []

def add_condition():
    st.session_state.conditions.append({'indicator': "RSI", 'operator': ">", 'value': 50})

def remove_condition(i):
    st.session_state.conditions.pop(i)

st.button("Add Condition", on_click=add_condition)

for i, condition in enumerate(st.session_state.conditions):
    cols = st.columns(4)
    condition['indicator'] = cols[0].selectbox("Indicator", indicators, index=indicators.index(condition['indicator']), key=f"indicator_{i}")
    condition['operator'] = cols[1].selectbox("Operator", operators, index=operators.index(condition['operator']), key=f"operator_{i}")
    condition['value'] = cols[2].number_input("Value", value=condition['value'], key=f"value_{i}")
    cols[3].button("Remove", on_click=remove_condition, args=(i,))

if st.button("Run Screener"):
    result_df = stock_screener(percentile, st.session_state.conditions)
    st.dataframe(result_df)
    st.download_button("Download CSV", result_df.to_csv(index=False), "stocks.csv", "text/csv")
