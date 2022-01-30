import warnings
import pandas as pd
import numpy as np
import yfinance as yf
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpl_dates
import matplotlib
from matplotlib import pyplot as plt
import streamlit as st
from yahooquery import Ticker

from src.tools import functions as f0

warnings.filterwarnings("ignore")
matplotlib.use("Agg")
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


class The_Support_Resistance(object):

    def __init__(self, ticker):
        self.ticker = ticker
        self.period = "6mo"
        self.company = f0.company_longName(self.ticker)


    def setup(self):
        self.df = yf.download(self.ticker, period=self.period, parse_dates=True)
        self.df["Date"] = pd.to_datetime(self.df.index)
        self.df["Date"] = self.df["Date"].apply(mpl_dates.date2num)
        self.df = self.df.loc[:, ["Date", "Open", "High", "Low", "Close"]]
        return self.df


    def isSupport(self, df, i):
        support = (
            self.df["Low"][i] < self.df["Low"][i - 1]
            and self.df["Low"][i] < self.df["Low"][i + 1]
            and self.df["Low"][i + 1] < self.df["Low"][i + 2]
            and self.df["Low"][i - 1] < self.df["Low"][i - 2]
        )
        return support


    def isResistance(self, df, i):
        resistance = (
            self.df["High"][i] > self.df["High"][i - 1]
            and self.df["High"][i] > self.df["High"][i + 1]
            and self.df["High"][i + 1] > self.df["High"][i + 2]
            and self.df["High"][i - 1] > self.df["High"][i - 2]
        )
        return resistance


    def plot_all(self):
        fig, ax = plt.subplots()
        candlestick_ohlc(ax, self.df.values, width=0.6, colorup="green", colordown="red", alpha=0.8)
        date_format = mpl_dates.DateFormatter("%d %b %Y")
        ax.xaxis.set_major_formatter(date_format)
        fig.autofmt_xdate()
        self.df["Close"].plot(linestyle="--", color="k", linewidth=1)
        for level in self.levels:
            plt.hlines(
                level[1],
                xmin=self.df["Date"][level[0]],
                xmax=max(self.df["Date"]),
                colors="blue",
            )
            plt.title(
                f"{self.company} ({self.ticker}) - Support & Resistance Price Levels",
                fontsize=30,
                fontweight="bold",
            )
            plt.tight_layout()
            plt.grid(True, linestyle="--")
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(15)
        ax.grid(True, color="k", linestyle="-", linewidth=1, alpha=0.3)
        ax.legend(loc="best", prop={"size": 16})
        plt.tight_layout()
        fig.show()
        st.pyplot(fig)


    def isFarFromLevel(self, lot1):
        return np.sum([abs(lot1 - x) < self.s for x in self.levels]) == 0


    def level(self):
        self.setup()
        self.levels = []
        for i in range(2, self.df.shape[0] - 2):
            if self.isSupport(self.df, i):
                self.levels.append((i, self.df["Low"][i]))
            elif self.isResistance(self.df, i):
                self.levels.append((i, self.df["High"][i]))
        self.s = np.mean(self.df["High"] - self.df["Low"])
        self.levels = []
        for i in range(2, self.df.shape[0] - 2):
            if self.isSupport(self.df, i):
                lot2 = self.df["Low"][i]
                if self.isFarFromLevel(lot2):
                    self.levels.append((i, lot2))
            elif self.isResistance(self.df, i):
                lot3 = self.df["High"][i]
                if self.isFarFromLevel(lot3):
                    self.levels.append((i, lot3))
        fd = pd.DataFrame(self.levels)
        fd.columns = ["day", "price_level"]

        new_lst = []
        for i in fd["day"]:
            n = int(i)
            enddate = pd.to_datetime(self.df.index[0]) + pd.DateOffset(days=n)
            new_lst.append(enddate)
        fd["date"] = new_lst
        fd.set_index("date", inplace=True)
        st.text("\n")
        self.plot_all()
        plt.show()
        st.dataframe(fd.round(2))


if __name__ == "__main__":
    The_Support_Resistance(ticker="AAPL", period="6mo").level()
