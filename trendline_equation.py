import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from datetime import date
import streamlit as st

@st.cache_data(ttl=30, show_spinner=False)
def fit_trendline_from_lows(ticker):
    """
    Fits a linear trendline through the lowest local lows of a stock's closing prices.

    Args:
        ticker (str): Stock symbol (e.g. "AAPL")
        period (str): Time period for historical data (e.g. "1y", "6mo", "5y")
        num_points (int): How many of the lowest local lows to use (2 or 3 recommended)
        order (int): The number of neighboring points to consider for local extremum

    Returns:
        (m, b): Slope and intercept of the trendline equation: y = m * x + b
    """
    df = yf.download(ticker)['Close']
    if df.empty:
        raise ValueError("Failed to download or parse price data.")
                         
    a = - int((len(df)/2))
    b = int(a / 2)

    # Find the y values
    prices = df[ticker].values
    prices_2 = df[ticker].values[a:]
    prices_3 = df[ticker].values[b:]

    df_1 = df.reset_index()
    df_1['Date_Num'] = (df_1['Date'] - df_1['Date'].min()).dt.days

    # Finde the x values
    num = df_1['Date_Num'].values
    num_2 = df_1['Date_Num'].values[a:]
    num_3 = df_1['Date_Num'].values[b:]

    # Fit a linear trendline
    m, b = np.polyfit(num, prices, 1)
    m_2, b_2 = np.polyfit(num_2, prices_2, 1)
    m_3, b_3 = np.polyfit(num_3, prices_3, 1)       

    today = pd.Timestamp(date.today())
    current_x = (today - df_1['Date'].min()).days

    current_y = (m * current_x) + b
    current_y_2 = (m_2 * current_x) + b_2
    current_y_3 = (m_3 * current_x) + b_3

    trend_prices = {'trend_p1': current_y, 
                    'trend_p2': current_y_2,
                    'trend_p3': current_y_3}

    return trend_prices
