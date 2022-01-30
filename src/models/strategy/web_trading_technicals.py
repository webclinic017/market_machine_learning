import pandas as pd
import yfinance as yf
from yfinance import Ticker
from yahooquery import Ticker as yq_Ticker
import requests
import streamlit as st
import warnings
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import cufflinks as cf


warnings.filterwarnings("ignore")
pd.options.display.max_rows
pd.options.display.max_columns = 50
pd.options.display.max_rows = 25
register_matplotlib_converters()
cf.go_offline(connected=True)
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
plt.rcParams["figure.dpi"] = 100
print("SYSTEM LOAD COMPLETE")



class MovingAverage(object):
    def __init__(self, closing_prices):
        self.data = pd.DataFrame(closing_prices)

    def EMA(self, averaging_length=50):
        ret = self.data.ewm(span=averaging_length, adjust=False).mean()
        return ret.rename(columns={"Close": "EMA"})

    def MACD(self, a=12, b=26, c=9):
        MACD_line = self.EMA(a) - self.EMA(b)
        signal_line = MACD_line.ewm(span=c, adjust=False).mean()
        histogram = MACD_line - signal_line
        return MACD_line, signal_line, histogram



class Helpers(object):
    def __init__(self, symbol):
        self.symbol = symbol

    def companyName(self):
        try:
            d = yq_Ticker(self.symbol).quote_type
            return str(list(d.values())[0]["longName"])
        except:
            return '---'




class Trading_Technicals(object):

    def __init__(self, ticker):
        self.stock = ticker

    
    def trading_technicals(self):
        ticker_stock = Ticker(self.stock)
        df = ticker_stock.history(period="1y", parsed_dates=True)
        df = df.asfreq("B")  # B for Business calendar (no weekends)
        df = df.fillna(method="ffill")  # fills any missing day's data with previous day's
        closing_prices = df.Close  # takes column labelled 'Close'

        buy = pd.DataFrame(index=closing_prices.index, columns=["Buy"])
        sell = pd.DataFrame(index=closing_prices.index, columns=["Sell"])
        company = Helpers(ticker_stock).companyName()

        # Executing class and variables
        MACD_indicator = MovingAverage(closing_prices)
        MACD_line, signal_line, histogram = MACD_indicator.MACD()
        COOLDOWN_PERIOD = 30
        cooldown = 0

        # Initial variables
        INITIAL_INVESTMENT = 5000
        net_portfolio = INITIAL_INVESTMENT
        MAX_STOCK_PURCHASE = 1000  # maximum spent on each buy signal
        BROKERAGE_FEE = 9.50  # Selfwealth brokerage fee
        PROFIT_THRESHOLD = 1.2
        COOLDOWN_PERIOD = 30  # number of elapsed days before another buy can be executed
        # print(net_portfolio, MAX_STOCK_PURCHASE, BROKERAGE_FEE, PROFIT_THRESHOLD)

        # Generating buy and sell signals
        for i in range(1, len(closing_prices)):
            cooldown -= 1
            if i == 1:
                if MACD_line["EMA"].iloc[i] > signal_line["EMA"].iloc[i]:
                    high = "MACD"
                else:
                    high = "SIGNAL"
            elif MACD_line["EMA"].iloc[i] > signal_line["EMA"].iloc[i]:
                if high == "SIGNAL":  # MACD crossed signal - bottom to top BUY
                    if MACD_line["EMA"].iloc[i] < 0 and cooldown <= 0:
                        buy.iloc[i] = closing_prices[i]  # BUY
                        cooldown = COOLDOWN_PERIOD
                    high = "MACD"
            elif MACD_line["EMA"].iloc[i] < signal_line["EMA"].iloc[i]:
                if high == "MACD":  # MACD crossed signal - top to bottom SELL
                    if MACD_line["EMA"].iloc[i] > 0:
                        sell.iloc[i] = closing_prices[i]  # SELL
                    high = "SIGNAL"

        # Plotting results
        plt.rcParams["figure.figsize"] = [20, 13]
        plt.rcParams["lines.linewidth"] = 0.75
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        plt.grid(True)
        ax1.xaxis.set_tick_params(rotation=45)
        ax1.plot(closing_prices, label="Closing Prices")
        ax1.plot(buy, "g^", markersize=13, label="Buy")
        ax1.plot(sell, "rv", markersize=13, label="Sell")
        ax1.set_title(f"{company} Buy/Sell", fontsize=30)
        ax1.set(xlabel="Date", ylabel="($)")
        for label in ax1.get_xticklabels() + ax1.get_yticklabels():
            label.set_fontsize(15)
        ax1.grid(True, color="k", linestyle="-", linewidth=1, alpha=0.3)
        ax1.legend(loc="best", prop={"size": 16})

        ax2.plot(MACD_line, label="MACD Line")
        ax2.plot(signal_line, label="Signal Line")
        histogram_y = [histogram["EMA"].iloc[i] for i in range(0, len(histogram))]
        ax2.bar(
            histogram.index,
            histogram_y,
            color=["g" if histogram_y[i] > 0 else "r" for i in range(0, len(histogram_y))],
            width=1,
            label="Histogram",
        )
        ax2.set_title(f"{company} MACD vs Signal Line (with histogram)", fontsize=30)
        ax2.xaxis.set_tick_params(rotation=45)
        ax2.set(xlabel="Date", ylabel="Stock Price ($USD)")
        for label in ax2.get_xticklabels() + ax2.get_yticklabels():
            label.set_fontsize(15)
        ax2.grid(True, color="k", linestyle="-", linewidth=1, alpha=0.3)
        ax2.legend(loc="best", prop={"size": 16})

        ax3.plot(closing_prices, label="Closing Prices")
        ax3.plot(MACD_indicator.EMA(12), label="12 day EMA")
        ax3.plot(MACD_indicator.EMA(26), label="26 day EMA")
        ax3.set_title(f"{company} MACD Indicators (12) vs (26)", fontsize=30)
        ax3.set(xlabel="Date", ylabel="($)")
        for label in ax3.get_xticklabels() + ax3.get_yticklabels():
            label.set_fontsize(15)
        ax3.grid(True, color="k", linestyle="-", linewidth=1, alpha=0.3)
        ax3.legend(loc="best", prop={"size": 16})
        plt.tight_layout()
        st.pyplot(fig)

        # spy_ticker = yf.Ticker("SPY")
        # spy = spy_ticker.history(period="max")
        fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, figsize=(16, 9))
        ax[0].set_title("PRICES", fontsize=30, fontweight="bold")
        ax[0].plot(df["Close"])
        ax[1].set_title("Volume", fontsize=30, fontweight="bold")
        ax[1].plot(df["Volume"])
        st.pyplot(fig)

        SMA10 = df["Close"].rolling(window=10).mean()
        SMA20 = df["Close"].rolling(window=20).mean()
        SMA50 = df["Close"].rolling(window=50).mean()
        SMA200 = df["Close"].rolling(window=200).mean()

        # taking last 300 trading days
        fig, ax = plt.subplots()
        plt.plot(df["Close"][-300:], label=self.stock)
        plt.plot(SMA10[-300:], label="SMA10")
        plt.plot(SMA20[-300:], label="SMA20")
        plt.plot(SMA50[-300:], label="SMA50")
        plt.plot(SMA200[-300:], label="SMA200")

        plt.title(f"Moving Averages vs {self.stock}:", fontsize=30, fontweight="bold")
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(15)
        ax.grid(True, color="k", linestyle="-", linewidth=1, alpha=0.3)
        ax.legend(loc="best", prop={"size": 16})
        plt.xlabel("Date", fontsize=20, fontweight="bold")
        plt.ylabel("Price", fontsize=20, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)

        def get_points_above(sma_low, sma_high):
            points_above = {}
            for pair in zip(sma_low, sma_high):
                if pair[0] >= pair[1]:
                    date = sma_low[sma_low == pair[0]].index[0]
                    points_above[date] = pair[0]
            points_above = pd.Series(points_above, name="Price_Points")
            points_above.index.name = "Date"
            return points_above

        points_above_SMA50 = get_points_above(SMA20, SMA50)
        # st.write(f"** Points above SMA50: {points_above_SMA50}**")

        SMA20 = SMA20.reset_index()
        SMA50 = SMA50.reset_index()

        crossovers = pd.DataFrame()
        crossovers["Dates"] = SMA20["Date"]
        crossovers["Price"] = [i for i in df["Close"]]
        crossovers["SMA20"] = SMA20["Close"]
        crossovers["SMA50"] = SMA50["Close"]
        crossovers["position"] = crossovers["SMA20"] >= crossovers["SMA50"]
        crossovers["pre-position"] = crossovers["position"].shift(1)
        crossovers["Crossover"] = np.where(crossovers["position"] == crossovers["pre-position"], False, True)
        crossovers["Crossover"][0] = False
        # st.dataframe(crossovers)

        crossovers = crossovers.loc[crossovers["Crossover"] == True]
        crossovers = crossovers.reset_index()
        crossovers = crossovers.drop(["position", "pre-position", "Crossover", "index"], axis=1)
        crossovers["Signal"] = np.nan
        crossovers["Binary_Signal"] = 0.0
        for i in range(len(crossovers["SMA20"])):
            if crossovers["SMA20"][i] > crossovers["SMA50"][i]:
                crossovers["Binary_Signal"][i] = 1.0
                crossovers["Signal"][i] = "Buy"
            else:
                crossovers["Signal"][i] = "Sell"
        st.dataframe(crossovers)

        # taking last 600 trading days
        fig, ax = plt.subplots()
        SMA20 = df["Close"].rolling(window=20).mean()
        SMA50 = df["Close"].rolling(window=50).mean()
        plt.plot(df["Close"][-600:], label=self.stock)
        plt.plot(SMA20[-600:], label="SMA20")
        plt.plot(SMA50[-600:], label="SMA50")
        plt.plot(
            crossovers.loc[crossovers.Signal == "Buy"]["Dates"][-5:],
            crossovers["SMA20"][crossovers.Signal == "Buy"][-5:],
            "^",
            markersize=15,
            color="g",
            label="Buy",
        )
        plt.plot(
            crossovers.loc[crossovers.Signal == "Sell"]["Dates"][-4:],
            crossovers["SMA20"][crossovers.Signal == "Sell"][-4:],
            "v",
            markersize=15,
            color="r",
            label="Sell",
        )
        plt.title("Buy/Sell Chart", fontsize=30, fontweight="bold")
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(15)
        ax.grid(True, color="k", linestyle="-", linewidth=1, alpha=0.3)
        ax.legend(loc="best", prop={"size": 16})
        plt.xlabel("Date", fontsize=20, fontweight="bold")
        plt.ylabel("Price", fontsize=20, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)


if __name__ == "__main__":
    Trading_Technicals().trading_technicals("AAPL")