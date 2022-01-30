import warnings
from datetime import date
import pandas as pd
import matplotlib.pyplot as plt
from termcolor import colored as cl
import numpy as np
import yfinance as yf
from yahooquery import Ticker
from matplotlib.font_manager import FontProperties
import streamlit as st

warnings.filterwarnings("ignore")
pd.plotting.register_matplotlib_converters()
plt.style.use("seaborn-poster")
sm, med, lg = "20", "25", "30"
plt.rcParams["font.size"] = sm  # controls default text sizes
plt.rc("axes", titlesize=med)  # fontsize of the axes title
plt.rc("axes", labelsize=med)  # fontsize of the x & y labels
plt.rc("xtick", labelsize=sm)  # fontsize of the tick labels
plt.rc("ytick", labelsize=sm)  # fontsize of the tick labels
plt.rc("legend", fontsize=sm)  # legend fontsize
plt.rc("figure", titlesize=lg)  # fontsize of the figure title
plt.rc("axes", linewidth=2)  # linewidth of plot lines
plt.rcParams["figure.figsize"] = [13, 7]
plt.rcParams["figure.dpi"] = 100
plt.rcParams["axes.facecolor"] = "silver"


def get_ticker_data(ticker, hist_per):
    """
    grab market data
    """
    data = yf.download(ticker, period=hist_per)
    data.index = pd.to_datetime(data.index)
    data.columns = ["open", "high", "low", "close", "adj close", "volume"]
    del data["adj close"]
    return data.dropna(axis=1)


def calc_moving_average(data, symbol, windows):
    """
    calculate the moving averages for both strategies
    """
    for i in range(len(windows)):
        data[f"{windows[i]} day MA"] = data.close.rolling(window=windows[i]).mean()
    if len(windows) > 1:
        data = data[data[f"{windows[1]} day MA"] > 0]
    else:
        data = data[data[f"{windows[0]} day MA"] > 0]
    return data


def ma_backtest(data, window, strategy="single", sellShort=False, slippage=0.003):
    """
    backtest for both strategies
    data is a df that contains the closing price of the stock and moving averages
    window can be a single value for price crossover or a list for moving average crossover
    crossover equals price or ma to determine which strategy should be use
    """
    # catch the enabling of short selling at the beginning
    if sellShort:
        sellSignal = -1
    else:
        sellSignal = 0
    # vectorized backtests by strategy
    if strategy == "Single Moving Average":
        data["Side"] = data.apply(
            lambda row: 1 if row.close >= row[f"{window[0]} day MA"] else sellSignal,
            axis=1,
        )

    elif strategy == "Crossover Moving Average":
        data["Side"] = data.apply(
            lambda row: 1
            if row[f"{window[0]} day MA"] >= row[f"{window[1]} day MA"]
            else sellSignal,
            axis=1,
        )
    # metrics for calculating return
    data["LagPrice"] = data["close"].shift(1)
    data["PctChange"] = (data["close"] - data["LagPrice"]) / data["LagPrice"]
    # variables to capture the buy and sell prices
    buyPrice = []
    sellPrice = []
    # Logic for noting each buy and sell by strategy/short selling included
    for i in range(len(data.close)):
        if data["Side"][i] > data["Side"][i - 1]:
            buyPrice.append(data.close[i])
            sellPrice.append(np.nan)
        elif data["Side"][i] < data["Side"][i - 1]:
            sellPrice.append(data.close[i])
            buyPrice.append(np.nan)
        else:
            if i < 1:
                if data.Side[i] < 0:
                    sellPrice.append(data.close[i])
                    buyPrice.append(np.nan)
                elif data.Side[i] == 0:
                    buyPrice.append(np.nan)
                    sellPrice.append(np.nan)
                else:
                    buyPrice.append(data.close[i])
                    sellPrice.append(np.nan)
            else:
                buyPrice.append(np.nan)
                sellPrice.append(np.nan)
    data["buyPrice"] = buyPrice
    data["sellPrice"] = sellPrice
    data["Slippage"] = (
        (data.buyPrice.fillna(0) + data.sellPrice.fillna(0)) * slippage
    ) / data.close
    data["Return"] = data.Side * data.PctChange - data.Slippage
    data["Return"][0] = -data.Slippage[0]
    data["Cumulative"] = data.Return.cumsum()
    st.dataframe(data)
    return data


def plot(data, ticker, windows, strategy):
    """
    tell the story with some visuals
    """

    def company_longName(symbol):
        d = Ticker(symbol).quote_type
        return list(d.values())[0]["longName"]

    company = company_longName(ticker)

    fig, ax = plt.subplots()
    plt.plot(data["close"], label=ticker)

    if len(windows) > 1:
        plt.plot(data[f"{windows[0]} day MA"], label=f"{windows[0]} day MA", lw=2)
        plt.plot(data[f"{windows[1]} day MA"], label=f"{windows[1]} day MA", lw=2)
    else:
        plt.plot(data[f"{windows[0]} day MA"], label=f"{windows[0]} day MA")

    plt.scatter(
        data.index,
        data.buyPrice,
        marker="^",
        s=200,
        color="darkblue",
        label="BUY SIGNAL",
    )
    plt.scatter(
        data.index,
        data.sellPrice,
        marker="v",
        s=200,
        # color="crimson",
        label="SELL SIGNAL",
    )

    ax.set_xlabel("Time (years)", fontsize=20, fontweight="bold")
    ax.set_ylabel("Prices", fontsize=20, fontweight="bold")
    ax.set_title(
        f"{company} [{ticker}] - {strategy} Trading Signals",
        fontsize=25,
        fontweight="bold",
    )
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(15)
    fontP = FontProperties()
    fontP.set_size("large")

    ax.legend(bbox_to_anchor=(1.05, 1), loc="best", prop=fontP)
    ax.grid(True, color="k", linestyle="-", linewidth=1, alpha=0.3)
    # plt.xlim(date(2019, 12, 1))
    plt.tight_layout()
    st.pyplot(fig)
