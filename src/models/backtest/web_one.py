import warnings
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn import metrics  # for the check the error and accuracy of the model
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    r2_score,
    accuracy_score,
    r2_score,
)
from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV,
    cross_validate,
)
from sklearn.model_selection import (
    cross_val_score,
    KFold,
    cross_validate,
    train_test_split,
    TimeSeriesSplit,
)
from math import sqrt
import xgboost as xgb
from xgboost import XGBRegressor
import matplotlib
import matplotlib as mpl
from matplotlib import style
from matplotlib import pyplot as plt
import seaborn as sns
import pyforest
import pyfolio as pf
import backtrader as bt
from backtrader.feeds import PandasData
import streamlit as st
from tscv import GapKFold

warnings.filterwarnings("ignore")
warnings.simplefilter(action="ignore", category=FutureWarning)
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 150)
# matplotlib.use("Agg")
plt.style.use("seaborn-poster")
sm, med, lg = 10, 15, 20
plt.rc("font", size=sm)  # controls default text sizes
plt.rc("axes", titlesize=med)  # fontsize of the axes title
plt.rc("axes", labelsize=med)  # fontsize of the x & y labels
plt.rc("xtick", labelsize=sm)  # fontsize of the tick labels
plt.rc("ytick", labelsize=sm)  # fontsize of the tick labels
plt.rc("legend", fontsize=sm)  # legend fontsize
plt.rc("figure", titlesize=lg)  # fontsize of the figure title
plt.rc("axes", linewidth=2)  # linewidth of plot lines
plt.rcParams["figure.figsize"] = [20, 10]
plt.rcParams["figure.dpi"] = 100
sns.set()
st.set_option("deprecation.showPyplotGlobalUse", False)


# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


class Web_One(object):
    def __init__(self, ticker):
        self.ticker = ticker

        import requests

        def get_symbol(symbol):
            url = "http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={}&region=1&lang=en".format(
                symbol
            )
            result = requests.get(url).json()
            for x in result["ResultSet"]["Result"]:
                if x["symbol"] == symbol:
                    return x["name"]

        self.company = get_symbol(self.ticker)

        dataset = web.DataReader(self.ticker, data_source="yahoo", start="2000-01-01")
        dataset = dataset.sort_index(ascending=True)
        # display /
        st.text(dataset.head())
        st.text(dataset.shape)
        # Plot the closing prices
        fig, ax = plt.subplots()
        dataset["Adj Close"].plot(grid=True, figsize=(10, 6))
        plt.title(f"{self.company} ({self.ticker}) - close price")
        plt.ylabel("price ($)")
        st.pyplot(fig)

        """### ***Predictors***
        """

        lags = 5
        # Create the shifted lag series of prior trading period close values
        for i in range(0, lags):
            dataset["Lag%s" % str(i + 1)] = dataset["Adj Close"].shift(i + 1)
        dataset["HL"] = (dataset["High"] - dataset["Close"]) / dataset["Close"] * 100
        # creating more features
        dataset["returns"] = dataset["Adj Close"].pct_change() * 100
        dataset["vol_increment"] = dataset.Volume.diff() / dataset.Volume
        dataset.dropna(inplace=True)
        st.text(dataset.head())
        st.text(dataset.shape)

        """### ***Target Variable***
        This will be a classification variable, if the average price will go either up or down the next day.  The target will be forecasting the difference between today’s price and tomorrow’s price (which is unkonwn).
        """

        dataset["target"] = dataset.Lag1.shift(-1) - dataset.Lag1
        dataset["target"] = dataset["Close"] - dataset["Open"]
        dataset.dropna(inplace=True)
        print(dataset.target)
        print()
        st.text(
            "Total dataset has {} samples, and {} features.".format(
                dataset.shape[0], dataset.shape[1]
            )
        )

        corr = dataset.corr()
        ax = sns.heatmap(
            corr,
            vmin=-1,
            vmax=1,
            center=0,
            cmap=sns.diverging_palette(20, 220, n=200),
            square=True,
        )
        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=45, horizontalalignment="right"
        )

        # Checking Correlation
        sns.set(style="darkgrid", context="talk", palette="Dark2")
        fig, ax = plt.subplots()
        plt.title(f"{self.company} ({self.ticker}) - Targets")
        dataset.corr()["target"].sort_values(ascending=False).plot(kind="bar")
        st.pyplot(fig)

        X = dataset[["Lag1", "Lag2", "HL", "returns"]]
        y = dataset["target"]

        gkcv = GapKFold(n_splits=5, gap_before=2, gap_after=1)
        tscv = TimeSeriesSplit(max_train_size=None, n_splits=5)
        for train_samples, test_samples in gkcv.split(X, y):
            # st.text("TRAIN:", train_samples, "TEST:", test_samples)
            XTrain, XTest = X.values[train_samples], X.values[test_samples]
            yTrain, yTest = y.values[train_samples], y.values[test_samples]

        feature_names = ["Lag1", "Lag2", "HL", "returns"]
        XTrain = pd.DataFrame(data=XTrain, columns=feature_names)
        XTest = pd.DataFrame(data=XTest, columns=feature_names)

        st.write(XTrain.shape, yTrain.shape)
        st.write(XTest.shape, yTest.shape)

        yTrain = pd.DataFrame(yTrain)
        yTrain.rename(columns={0: "target"}, inplace=True)

        yTest = pd.DataFrame(yTest)
        yTest.rename(columns={0: "target"}, inplace=True)

        def getBinary(val):
            if val > 0:
                return 1
            else:
                return -1

        yTest_binary = pd.DataFrame(yTest["target"].apply(getBinary))
        st.text(yTest_binary)

        """The target variables will be transformed for binary classification. A positive change in the value of prices will be classified as 1 and a non-positive change as 0."""

        regressor = xgb.XGBRegressor(
            objective="reg:squarederror",
            gamma=0.0,
            n_estimators=200,
            base_score=0.7,
            colsample_bytree=1,
            learning_rate=0.01,
        ).fit(XTrain, yTrain)
        scores = cross_val_score(regressor, XTrain.values, yTrain.values, cv=gkcv)
        st.text(
            "Training Accuracy (cross validated): %0.2f (+/- %0.2f)"
            % (scores.mean(), scores.std() * 2)
        )
        scores = cross_val_score(regressor, XTest.values, yTest.values, cv=gkcv)
        st.text(
            "Test Accuracy (cross validated): %0.2f (+/- %0.2f)"
            % (scores.mean(), scores.std() * 2)
        )

        yPredicted = regressor.predict(XTest)
        yPredicted_binary = [1 if yp > 0 else -1 for yp in yPredicted]

        from sklearn.metrics import mean_squared_error

        st.write("RMSE: XGBoost", round(sqrt(mean_squared_error(yTest, yPredicted)), 4))

        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            gamma=0.0,
            n_estimators=200,
            base_score=0.7,
            colsample_bytree=1,
            learning_rate=0.01,
        ).fit(X, y)
        Predict = model.predict(X)
        Predict_Binary = [1 if yp > 0 else -1 for yp in Predict]
        BinaryPredict = pd.DataFrame(Predict_Binary)
        BinaryPredict.rename({0: "Prediction"}, axis=1, inplace=True)
        BinaryPredict.index = dataset.index
        st.write(BinaryPredict)
        st.write()
        # number of trades over time for highest and second highest return strategy
        st.write("Number of trades = ", (BinaryPredict.Prediction.diff() != 0).sum())

        dataset.columns

        db = dataset[["Open", "High", "Low", "Close", "Volume"]]
        prices = pd.concat([db, BinaryPredict], 1)
        prices.rename(
            columns={
                "Prediction": "predicted",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            },
            inplace=True,
        )
        prices

        # Buy/Sell signals plot
        sells = prices.loc[prices["predicted"] == 1]
        buys = prices.loc[prices["predicted"] == -1]

        # Plot
        fig = plt.figure(figsize=(20, 5))
        plt.plot(prices.index[-100:], prices["close"][-100:], lw=2.0, label="Price")
        # Plot buy and sell signals
        plt.plot(
            buys.index[-100:],
            prices.loc[buys.index]["close"][-100:],
            "v",
            markersize=10,
            color="red",
            lw=2.0,
            label="Buy",
        )
        plt.plot(
            sells.index[-100:],
            prices.loc[sells.index]["close"][-100:],
            "^",
            markersize=10,
            color="green",
            lw=2.0,
            label="Sell",
        )
        plt.ylabel("Price (USD)")
        plt.xlabel("Date")
        plt.title(f"Last 100 Buy and Sell signals - {self.company} ({self.ticker})")
        plt.legend(loc="best")
        st.pyplot(fig)

        OHLCV = ["open", "high", "low", "close", "volume"]

        # class to define the columns we will provide
        class SignalData(PandasData):
            """
            Define pandas DataFrame structure
            """

            cols = OHLCV + ["predicted"]

            # create lines
            lines = tuple(cols)

            # define parameters
            params = {c: -1 for c in cols}
            params.update({"Date": None})
            params = tuple(params.items())

        # define backtesting strategy class
        class MLStrategy(bt.Strategy):
            params = dict()

            def __init__(self):
                # keep track of open, close prices and predicted value in the series
                self.data_predicted = self.datas[0].predicted
                self.data_open = self.datas[0].open
                self.data_close = self.datas[0].close

                # keep track of pending orders/buy price/buy commission
                self.order = None
                self.price = None
                self.comm = None

            # logging function
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
                            f"BUY EXECUTED --- Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f},Commission: {order.executed.comm:.2f}"
                        )
                        self.price = order.executed.price
                        self.comm = order.executed.comm
                    else:
                        self.log(
                            f"SELL EXECUTED --- Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f},Commission: {order.executed.comm:.2f}"
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

            # We have set cheat_on_open = True.This means that we calculated the signals on day t's close price,
            # but calculated the number of shares we wanted to buy based on day t+1's open price.
            def next_open(self):
                if not self.position:
                    if self.data_predicted > 0:
                        # calculate the max number of shares ('all-in')
                        size = int(self.broker.getcash() / self.datas[0].open)
                        # buy order
                        self.log(
                            f"BUY CREATED --- Size: {size}, Cash: {self.broker.getcash():.2f}, Open: {self.data_open[0]}, Close: {self.data_close[0]}"
                        )
                        self.buy(size=size)
                else:
                    if self.data_predicted < 0:
                        # sell order
                        self.log(f"SELL CREATED --- Size: {self.position.size}")
                        self.sell(size=self.position.size)

        # instantiate SignalData class
        data = SignalData(dataname=prices)

        # instantiate Cerebro, add strategy, data, initial cash, commission and pyfolio for performance analysis
        cerebro = bt.Cerebro(stdstats=False, cheat_on_open=True)
        cerebro.addstrategy(MLStrategy)
        cerebro.adddata(data)
        cerebro.broker.setcash(10000.0)
        cerebro.broker.setcommission(commission=0.001)
        cerebro.addanalyzer(bt.analyzers.PyFolio, _name="pyfolio")

        # run the backtest
        st.text("Starting Portfolio Value: %.2f" % cerebro.broker.getvalue())
        backtest_result = cerebro.run()
        st.text("Final Portfolio Value: %.2f" % cerebro.broker.getvalue())

        # Extract inputs for pyfolio
        strat = backtest_result[0]
        pyfoliozer = strat.analyzers.getbyname("pyfolio")
        returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()
        returns.name = "Strategy"
        # returns.tail()

        # benchmark returns
        benchmark_rets = dataset["returns"]
        benchmark_rets.index = benchmark_rets.index.tz_localize("UTC")
        benchmark_rets = benchmark_rets.filter(returns.index)
        benchmark_rets.name = "Nasdaq Composite"
        # benchmark_rets.tail()

        # get performance statistics for strategy
        pf.show_perf_stats(returns)

        # get performance statistics for benchmark
        pf.show_perf_stats(benchmark_rets)

        # plot performance for strategy vs benchmark
        fig, ax = plt.subplots(
            nrows=2, ncols=2, figsize=(16, 9), constrained_layout=True
        )
        axes = ax.flatten()
        pf.plot_rolling_beta(returns=returns, factor_returns=benchmark_rets, ax=axes[0])
        axes[0].grid(True)
        pf.plot_rolling_volatility(
            returns=returns, factor_returns=benchmark_rets, ax=axes[1]
        )
        axes[1].grid(True)
        pf.plot_annual_returns(returns=returns, ax=axes[2])
        axes[2].grid(True)
        pf.plot_monthly_returns_heatmap(
            returns=returns,
            ax=axes[3],
        )
        plt.tight_layout()
        st.pyplot(fig)


# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


if __name__ == "__main__":
    Web_One("BA")


# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
