import yfinance as yf
from yahooquery import Ticker
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from tabulate import tabulate
import warnings
import streamlit as st
from datetime import datetime

warnings.filterwarnings("ignore")
mpl.use("Agg")
plt.style.use("ggplot")
sm, med, lg = 10, 15, 20
plt.rc("font", size=sm)  # controls default text sizes
plt.rc("axes", titlesize=med)  # fontsize of the axes title
plt.rc("axes", labelsize=med)  # fontsize of the x & y labels
plt.rc("xtick", labelsize=sm)  # fontsize of the tick labels
plt.rc("ytick", labelsize=sm)  # fontsize of the tick labels
plt.rc("legend", fontsize=sm)  # legend fontsize
plt.rc("figure", titlesize=lg)  # fontsize of the figure title
plt.rc("axes", linewidth=2)  # linewidth of plot lines
plt.rcParams["figure.figsize"] = [18, 10]
plt.rcParams["figure.dpi"] = 150


def MovingAverageCrossStrategy(symbol, short_window, long_window, end_date, moving_avg, data, print_out=True, cc=0.0, ccc=0.0, inter='1d'):
    """
        The function takes the stock symbol, time-duration of analysis,
        look-back periods and the moving-average type(SMA or EMA) as input
        and returns the respective MA Crossover chart along with the buy/sell signals for the given period.
            >> symbol - (str)stock ticker as on Yahoo finance. Eg: 'ULTRACEMCO.NS'
            >> start_date - (str)start analysis from this date (format: 'YYYY-MM-DD') Eg: '2018-01-01'
            >> end_date - (str)end analysis on this date (format: 'YYYY-MM-DD') Eg: '2021-01-01'
            >> short_window - (int)lookback period for short-term moving average. Eg: 5, 10, 20
            >> long_window - (int)lookback period for long-term moving average. Eg: 50, 100, 200
            >> moving_avg - (str)the type of moving average to use ('SMA' or 'EMA')
            >> display_table - (bool)whether to display the date and price table at buy/sell positions(True/False)
            >> import the closing price data of the stock for the aforementioned period of time in Pandas dataframe
    """
    # data = yf.download(symbol, start='2020-01-03', end=str(end_date)[:10], interval=inter)
    data = pd.DataFrame(data)
    data.index = pd.to_datetime(data.index)
    stock_df = pd.DataFrame(data['adjclose'])        
    stock_df.columns = {"Close Price"}  # assign new colun name
    stock_df = stock_df.fillna(0.0)

    # column names for long and short moving average columns
    short_window_col = str(short_window) + "_" + moving_avg
    long_window_col = str(long_window) + "_" + moving_avg

    # Create a short simple moving average (short_sma) & create a long simple moving average (long_sma) column
    if moving_avg == "SMA":
        stock_df[short_window_col] = (stock_df["Close Price"].rolling(window=short_window, min_periods=1).mean())        
        stock_df[long_window_col] = (stock_df["Close Price"].rolling(window=long_window, min_periods=1).mean())

    # Create a short simple exponential average (short_ema) & create a long exponential moving average (long_ema) column
    elif moving_avg == "EMA":
        stock_df[short_window_col] = (stock_df["Close Price"].ewm(span=short_window, adjust=False).mean())
        stock_df[long_window_col] = (stock_df["Close Price"].ewm(span=long_window, adjust=False).mean())

    """ 
        > create a new column 'Signal' such that if faster moving average is 
        > greater than slower moving average
        > then set Signal as 1 else 0.
        > create a new column 'Position' which is a day-to-day difference of the 'Signal' column.
        > Determine current BUY/SELL Status of Security
    """
    stock_df["Signal"] = 0.0
    stock_df["Signal"] = np.where(stock_df[short_window_col] > stock_df[long_window_col], 1.0, 0.0)
    stock_df["Position"] = stock_df["Signal"].diff()
    df_pos = stock_df[(stock_df["Position"] == 1) | (stock_df["Position"] == -1)]
    df_pos["Position"] = df_pos["Position"].apply(lambda x: "Buy" if x == 1 else "Sell")
    # st.text(tabulate(df_pos.loc["2021":], headers="keys", tablefmt="psql"))


    if print_out == True:
        try:
            fig, ax = plt.subplots()

            # plot close price, short-term and long-term moving averages
            plt.tick_params(axis="both", labelsize=15)
            stock_df.loc["2021":, "Close Price"].plot(color="k", lw=1, label="Close Price")
            stock_df.loc["2021":][short_window_col].plot(color="b", lw=1, label=short_window_col)
            stock_df.loc["2021":][long_window_col].plot(color="g", lw=1, label=long_window_col)
            
            # plot 'buy' signals
            plt.plot(
                stock_df.loc["2021":][stock_df["Position"] == 1].index,
                stock_df.loc["2021":][short_window_col][stock_df["Position"] == 1],
                "^", markersize=15, color="g", alpha=0.7, label="buy",
            )

            # plot 'sell' signals
            plt.plot(
                stock_df.loc["2021":][stock_df["Position"] == -1].index,
                stock_df.loc["2021":][short_window_col][stock_df["Position"] == -1],
                "v", markersize=15, color="r", alpha=0.7, label="sell",
            )

            plt.ylabel("Price in $", fontsize=20, fontweight="bold")
            plt.xlabel("Date", fontsize=20, fontweight="bold")
            plt.title(f"{symbol} - {str(moving_avg)} Crossover", fontsize=30, fontweight="bold")
            plt.grid(True, color="k", linestyle="-", linewidth=1, alpha=0.3)
            ax.legend(loc="best", prop={"size": 16})
            plt.tight_layout()
            plt.show()
            st.pyplot(fig)
        except:
            pass


    if df_pos['Position'][-1] == 'Buy':
        st.metric(f"No. {cc} / {ccc} In Portfolio", f"{symbol}", f"{df_pos['Position'][-1]}")
        return symbol

    elif df_pos['Position'][-1] == 'Sell':
        st.metric(f"No. {cc} / {ccc} In Portfolio", f"{symbol}", f"- {df_pos['Position'][-1]}")
        return