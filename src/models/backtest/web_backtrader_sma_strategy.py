import warnings

warnings.filterwarnings("ignore")
from datetime import datetime, date, timedelta
from pathlib import Path

today = str(datetime.now())[:10]
import matplotlib
import matplotlib as mpl

# matplotlib.use('TkAgg')
from matplotlib import style
from matplotlib import pyplot as plt

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
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import itertools
import streamlit as st

st.set_option("deprecation.showPyplotGlobalUse", False)
import yfinance as yf
from datetime import datetime
import backtrader as bt


#  * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


class MyBuySell(bt.observers.BuySell):
    plotlines = dict(
        buy=dict(marker="^", markersize=8.0, color="blue", fillstyle="full"),
        sell=dict(marker="v", markersize=8.0, color="red", fillstyle="full"),
    )


#  * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


# Create a Strategy
class SmaStrategy(bt.Strategy):
    params = (("ma_period", 20),)

    def __init__(self):
        # keep track of close price in the series
        self.data_close = self.datas[0].close

        # keep track of pending orders/buy price/buy commission
        self.order = None
        self.price = None
        self.comm = None

        # add a simple moving average indicator
        self.sma = bt.ind.SMA(self.datas[0], period=self.params.ma_period)

    def log(self, txt):
        """Logging function"""
        dt = self.datas[0].datetime.date(0).isoformat()
        st.text(f"{dt}, {txt}")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # order already submitted/accepted - no action required
            return

        # report executed order
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f"BUY EXECUTED --- Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Commission: {order.executed.comm:.2f}"
                )
                self.price = order.executed.price
                self.comm = order.executed.comm
            else:
                self.log(
                    f"SELL EXECUTED --- Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Commission: {order.executed.comm:.2f}"
                )

        # report failed order
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order Failed")

        # set no pending order
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log(
            f"OPERATION RESULT --- Gross: {trade.pnl:.2f}, Net: {trade.pnlcomm:.2f}"
        )

    def next(self):
        # do nothing if an order is pending
        if self.order:
            return

        # check if there is already a position
        if not self.position:
            # buy condition
            if self.data_close[0] > self.sma[0]:
                self.log(f"BUY CREATED --- Price: {self.data_close[0]:.2f}")
                self.order = self.buy()
        else:
            # sell condition
            if self.data_close[0] < self.sma[0]:
                self.log(f"SELL CREATED --- Price: {self.data_close[0]:.2f}")
                self.order = self.sell()


#  * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


def backtrader_sma_strategy_run(tick):

    # download data
    data = bt.feeds.YahooFinanceData(
        dataname=tick, fromdate=datetime(2021, 1, 1), todate=datetime.now()
    )

    # create a Cerebro entity
    cerebro = bt.Cerebro(stdstats=False)

    # set up the backtest
    cerebro.adddata(data)
    cerebro.broker.setcash(1000.0)
    cerebro.addstrategy(SmaStrategy)
    cerebro.addobserver(MyBuySell)
    cerebro.addobserver(bt.observers.Value)

    # run backtest
    Starting = f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}"
    cerebro.run()
    st.header(f"Starting Portfolio Value: {Starting}")
    st.header(f"Final Portfolio Value: {cerebro.broker.getvalue():.2f}")

    # plot results
    # fig, ax = plt.subplots()
    # ax = cerebro.plot(iplot=False)
    # st.pyplot(plt.show())
    st.pyplot(cerebro.plot(iplot=False, use="TkAgg"))


#  * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


# backtrader_sma_strategy_run('/home/gordon/modern_millennial_market_mapping/src/models/backtest/z_test_env/AAPL.pkl')


#  * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
