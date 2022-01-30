from yahooquery import Ticker
import streamlit as st
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from mplfinance.original_flavor import candlestick_ohlc
from datetime import datetime
import seaborn as sns
import yfinance as yf
import numpy as np
from matplotlib import pyplot as plt
import os

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
sns.set()
plt.rcParams["figure.figsize"] = [18, 13]
plt.rcParams["figure.dpi"] = 100
os.environ["NUMEXPR_MAX_THREADS"] = "24"
os.environ["NUMEXPR_NUM_THREADS"] = "12"


def get_company_longName(symbol):
    d = Ticker(symbol).quote_type
    return list(d.values())[0]["longName"]


class The_OverBought_OverSold(object):
    def __init__(self, ticker):
        self.ticker = ticker
        self.start_date = "2021-01-05"
        self.end_dte = datetime.now()
        self.company = get_company_longName(self.ticker)

    def generate(self):
        df = yf.download(
            self.ticker, start=self.start_date, end=self.end_dte, parse_dates=True
        )
        df.reset_index(inplace=True)
        df.Date = df.Date.astype("str")
        date = [datetime.strptime(d, "%Y-%m-%d") for d in df["Date"]]
        candlesticks = list(
            zip(
                mdates.date2num(date),
                df["Open"],
                df["High"],
                df["Low"],
                df["Close"],
                df["Volume"],
            )
        )

        def removal(signal, repeat):
            copy_signal = np.copy(signal)
            for j in range(repeat):
                for i in range(3, len(signal)):
                    copy_signal[i - 1] = (copy_signal[i - 2] + copy_signal[i]) / 2
            return copy_signal

        def get(original_signal, removed_signal):
            buffer = []
            for i in range(len(removed_signal)):
                buffer.append(original_signal[i] - removed_signal[i])
            return np.array(buffer)

        signal = np.copy(df.Open.values)
        removed_signal = removal(signal, 30)
        noise_open = get(signal, removed_signal)

        signal = np.copy(df.High.values)
        removed_signal = removal(signal, 30)
        noise_high = get(signal, removed_signal)

        signal = np.copy(df.Low.values)
        removed_signal = removal(signal, 30)
        noise_low = get(signal, removed_signal)

        signal = np.copy(df.Close.values)
        removed_signal = removal(signal, 30)
        noise_close = get(signal, removed_signal)

        noise_candlesticks = list(
            zip(mdates.date2num(date), noise_open, noise_high, noise_low, noise_close)
        )

        fig = plt.figure()
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        plt.title(
            f"Analysis of OverBought vs OverSold For {self.company} ({self.ticker})",
            fontsize=20,
            fontweight="bold",
        )
        ax1.set_ylabel("Quote ($)", fontsize=15, fontweight="bold")
        dates = [x[0] for x in candlesticks]
        dates = np.asarray(dates)
        volume = [x[5] for x in candlesticks]
        volume = np.asarray(volume)
        candlestick_ohlc(ax1, candlesticks, width=1, colorup="g", colordown="r")
        # ax1.plot(df['Close'], df.index, color='k')
        pad = 0.25
        yl = ax1.get_ylim()
        ax1.set_ylim(yl[0] - (yl[1] - yl[0]) * pad, yl[1])
        for label in ax1.get_xticklabels() + ax1.get_yticklabels():
            label.set_fontsize(15)
        ax1.grid(True, color="k", linestyle="-", linewidth=1, alpha=0.3)
        # ax1.legend(loc="best", prop={"size": 13})
        # ax1.set_xticklabels(ax1.get_xticklabels(), rotation= 60)
        plt.tight_layout()

        ax2 = ax1.twinx()
        pos = df["Open"] - df["Close"] < 0
        neg = df["Open"] - df["Close"] > 0
        ax2.bar(dates[pos], volume[pos], color="green", width=1, align="center", alpha=0.25)
        ax2.bar(dates[neg], volume[neg], color="red", width=1, align="center", alpha=0.25)
        ax2.set_xlim(min(dates), max(dates))
        yticks = ax2.get_yticks()
        ax2.set_yticks(yticks[::3])
        ax2.yaxis.set_label_position("right")
        ax2.set_ylabel("Volume", fontsize=15, fontweight="bold")
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
        ax2 = plt.subplot2grid((3, 1), (2, 0))
        ax2.set_ylabel("Quote ($)", fontsize=15, fontweight="bold")
        candlestick_ohlc(ax2, noise_candlesticks, width=1, colorup="g", colordown="r")
        ax2.plot(
            dates,
            [np.percentile(noise_close, 95)] * len(noise_candlesticks),
            color=(1.0, 0.792156862745098, 0.8, 1.0),
            linewidth=5.0,
            label="overbought line",
            alpha=1.0,
        )
        ax2.plot(
            dates,
            [np.percentile(noise_close, 10)] * len(noise_candlesticks),
            color=(0.6627450980392157, 1.0, 0.6392156862745098, 1.0),
            linewidth=5.0,
            label="oversold line",
            alpha=1.0,
        )
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax2.xaxis.set_major_locator(mticker.MaxNLocator(10))
        for label in ax2.get_xticklabels() + ax2.get_yticklabels():
            label.set_fontsize(15)
        ax2.grid(True, color="k", linestyle="-", linewidth=1, alpha=0.3)
        ax2.legend(loc="best", prop={"size": 13})
        # ax2.set_xticklabels(ax2.get_xticklabels(), rotation= 60)
        plt.tight_layout()
        st.pyplot(fig)


if __name__ == "__main__":
    The_OverBought_OverSold("AAPL").generate()