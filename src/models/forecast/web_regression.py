import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import math
import yfinance as yf
from matplotlib import pyplot as plt
from datetime import datetime, timedelta, date
import streamlit as st
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import tensorflow as tf
from yahooquery import Ticker

plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = [15, 10]
plt.rcParams["figure.dpi"] = 100

# from src.tools import functions as f0


def company_longName(symbol):
    d = Ticker(symbol).quote_type
    return list(d.values())[0]["longName"]


class Regression_Model(object):
    days = 10

    def __init__(self, ticker):
        self.ticker = ticker
        self.sName = company_longName(self.ticker)

    def preprocessing(self):
        st.subheader(f"** > {self.sName} · [{self.ticker}] **")
        start = datetime(2018, 1, 1)
        df = yf.download(self.ticker, period="5y", parse_dates=True)
        close_px = df["Adj Close"]
        mavg = close_px.rolling(window=100).mean()

        # fig, ax = plt.subplots(nrows=1, ncols=1)
        # mavg.plot(label="mavg")
        # close_px.plot(label="price")
        # plt.ylabel(f"{self.sName} Price", fontsize=20, fontweight="bold")
        # plt.xlabel("Dates", fontsize=20, fontweight="bold")
        # plt.title(
        #     f"Moving Average vs {self.sName} [{self.ticker}] - Price",
        #     fontsize=30,
        #     fontweight="bold",
        # )
        # plt.xlim(date(2019, 1, 1))
        # for label in ax.get_xticklabels() + ax.get_yticklabels():
        #     label.set_fontsize(15)
        # ax.grid(True, color="k", linestyle="-", linewidth=1, alpha=0.3)
        # ax.legend(loc="best", prop={"size": 16})
        # plt.tight_layout()
        # st.pyplot(fig)
        # plt.close(fig)

        dfreg = df.loc[:, ["Adj Close", "Volume"]]
        dfreg["HL_PCT"] = (df["High"] - df["Low"]) / df["Close"] * 100.0
        dfreg["PCT_change"] = (df["Close"] - df["Open"]) / df["Open"] * 100.0

        # Drop missing value
        dfreg.fillna(value=-99999, inplace=True)

        # We want to separate 1 percent of the data to forecast
        forecast_out = int(math.ceil(0.01 * len(dfreg)))

        # Separating the label here, we want to predict the AdjClose
        forecast_col = "Adj Close"
        dfreg["label"] = dfreg[forecast_col].shift(-forecast_out)
        X = np.array(dfreg.drop(["label"], 1))

        # Scale the X so that everyone can have the same distribution for linear regression
        X = sklearn.preprocessing.scale(X)

        # Finally We want to find Data Series of late X & early X (train) for model generation & eval
        X_lately = X[-forecast_out:]
        X = X[:-forecast_out]

        # Separate label and identify it as y
        y = np.array(dfreg["label"])
        y = y[:-forecast_out]
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        # Linear regression
        clfreg = LinearRegression(n_jobs=-1)
        clfreg.fit(X_train, y_train)

        # Quadratic Regression 2
        clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
        clfpoly2.fit(X_train, y_train)

        # Quadratic Regression 3
        clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
        clfpoly3.fit(X_train, y_train)

        # KNN Regression
        clfknn = KNeighborsRegressor(n_neighbors=2)
        clfknn.fit(X_train, y_train)

        # results
        confidencereg = clfreg.score(X_test, y_test)
        confidencepoly2 = clfpoly2.score(X_test, y_test)
        confidencepoly3 = clfpoly3.score(X_test, y_test)
        confidenceknn = clfknn.score(X_test, y_test)

        st.subheader(f"** > Regression Analysis · Model Results **")
        st.write(
            f" - Linear Regression Confidence =  [{round(confidencereg * 100, 2)}%]"
        )
        st.write(
            f" - Quadratic Regression 2 Confidence =  [{round(confidencepoly2 * 100, 2)}%]"
        )
        st.write(
            f" - Quadratic Regression 3 Confidence =  [{round(confidencepoly3 * 100, 2)}%]"
        )
        st.write(f" - KNN Regression Confidence =  [{round(confidenceknn * 100, 2)}%]")

        fd = pd.DataFrame()
        fd["---Regression_Model---"] = [
            "linear_regression",
            "quadratic_regression_2",
            "quadratic_regression_3",
            "knn",
        ]
        fd["Model_Results"] = [
            confidencereg,
            confidencepoly2,
            confidencepoly3,
            confidenceknn,
        ]
        fd.set_index("---Regression_Model---", inplace=True)
        fd.sort_values("Model_Results", ascending=False, inplace=True)
        res_lst = [dfreg, X_lately, clfreg, clfpoly2, clfpoly3, clfknn, fd.index[0]]
        return res_lst


    def linear_regression(
        self, dfreg, X_lately, clfreg, clfpoly2, clfpoly3, clfknn, days=252
    ):
        st.subheader(f"** > Best Model Fit : ** Linear Regression Forecast")
        st.write("__" * 25)
        forecast_set = clfreg.predict(X_lately)
        dfreg["Forecast"] = np.nan
        dfreg["Forecast"]
        last_date = dfreg.iloc[-1].name
        last_unix = last_date
        next_unix = last_unix + timedelta(1)
        for i in forecast_set:
            next_date = next_unix
            next_unix += timedelta(days)
            dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns) - 1)] + [i]
        fig, ax = plt.subplots(nrows=1, ncols=1)
        dfreg["Adj Close"].tail(500).plot()
        dfreg["Forecast"].tail(500).plot()
        plt.title(
            f"{self.sName} ({self.ticker}) - Linear Regression - Forecast",
            fontsize=30,
            fontweight="bold",
        )
        plt.xlabel("Date", fontsize=20, fontweight="bold")
        plt.ylabel("Price", fontsize=20, fontweight="bold")
        plt.xlim(date(2020, 1, 1))
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(15)
        ax.grid(True, color="k", linestyle="-", linewidth=1, alpha=0.3)
        ax.legend(loc="best", prop={"size": 16})
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        fd = yf.download(self.ticker, period="1d")
        x = round(float(fd["Adj Close"]), 2)
        st.write(f" - {self.sName} · [{self.ticker}] · Current Stock Price = **${x}**")
        x = dfreg["Forecast"].iloc[-1]
        st.write(f" - 1 year Forcasted Price = **${round(float(x),2)}**")
        return

    def quadratic_regression_2(
        self, dfreg, X_lately, clfreg, clfpoly2, clfpoly3, clfknn, days=252
    ):
        st.subheader(f"** > Best Model Fit : ** Quadratic-2 Regression Forecast")
        st.write("__" * 25)
        forecast_set = clfpoly2.predict(X_lately)
        dfreg["Forecast"] = np.nan
        dfreg["Forecast"]
        last_date = dfreg.iloc[-1].name
        last_unix = last_date
        next_unix = last_unix + timedelta(1)
        for i in forecast_set:
            next_date = next_unix
            next_unix += timedelta(days)
            dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns) - 1)] + [i]
        fig, ax = plt.subplots(nrows=1, ncols=1)
        dfreg["Adj Close"].tail(500).plot()
        dfreg["Forecast"].tail(500).plot()
        plt.title(
            f"{self.sName} ({self.ticker}) - Quadratic (2) Regression - Forecast",
            fontsize=30,
            fontweight="bold",
        )
        plt.xlabel("Date", fontsize=20, fontweight="bold")
        plt.ylabel("Price", fontsize=20, fontweight="bold")
        plt.xlim(date(2020, 1, 1))
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(15)
        ax.grid(True, color="k", linestyle="-", linewidth=1, alpha=0.3)
        ax.legend(loc="best", prop={"size": 16})
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        fd = yf.download(self.ticker, period="1d")
        x = round(float(fd["Adj Close"]), 2)
        st.write(f" - {self.sName} · [{self.ticker}] · Current Stock Price = **${x}**")
        x = dfreg["Forecast"].iloc[-1]
        st.write(f" - 1 year Forcasted Price = **${round(float(x),2)}**")
        return

    def quadratic_regression_3(
        self, dfreg, X_lately, clfreg, clfpoly2, clfpoly3, clfknn, days=252
    ):
        st.subheader(f"** > Best Model Fit : ** Quadratic-3 Regression Forecast")
        st.write("__" * 25)
        forecast_set = clfpoly3.predict(X_lately)
        dfreg["Forecast"] = np.nan
        dfreg["Forecast"]
        last_date = dfreg.iloc[-1].name
        last_unix = last_date
        next_unix = last_unix + timedelta(1)
        for i in forecast_set:
            next_date = next_unix
            next_unix += timedelta(days)
            dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns) - 1)] + [i]
        fig, ax = plt.subplots(nrows=1, ncols=1)
        dfreg["Adj Close"].tail(500).plot()
        dfreg["Forecast"].tail(500).plot()
        plt.title(
            f"{self.sName} ({self.ticker}) - Quadratic (3) Regression - Forecast",
            fontsize=30,
            fontweight="bold",
        )
        plt.xlabel("Date", fontsize=20, fontweight="bold")
        plt.ylabel("Price", fontsize=20, fontweight="bold")
        plt.xlim(date(2020, 1, 1))
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(15)
        ax.grid(True, color="k", linestyle="-", linewidth=1, alpha=0.3)
        ax.legend(loc="best", prop={"size": 16})
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        fd = yf.download(self.ticker, period="1d")
        x = round(float(fd["Adj Close"]), 2)
        st.write(f" - {self.sName} · [{self.ticker}] · Current ticker Price = **${x}**")
        x = dfreg["Forecast"].iloc[-1]
        st.write(f" - 1 year Forcasted Price = **${round(float(x),2)}**")
        return

    def knn(self, dfreg, X_lately, clfreg, clfpoly2, clfpoly3, clfknn, days=252):
        st.subheader(f"** > Best Model Fit : ** KNN Regression Forecast")
        st.write("__" * 25)
        forecast_set = clfknn.predict(X_lately)
        dfreg["Forecast"] = np.nan
        dfreg["Forecast"]
        last_date = dfreg.iloc[-1].name
        last_unix = last_date
        next_unix = last_unix + timedelta(1)
        for i in forecast_set:
            next_date = next_unix
            next_unix += timedelta(days)
            dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns) - 1)] + [i]
        fig, ax = plt.subplots(nrows=1, ncols=1)
        dfreg["Adj Close"].tail(500).plot()
        dfreg["Forecast"].tail(500).plot()
        plt.title(
            f"{self.sName} ({self.ticker}) - KNN Regression - Forecast",
            fontsize=30,
            fontweight="bold",
        )
        plt.xlabel("Date", fontsize=20, fontweight="bold")
        plt.ylabel("Price", fontsize=20, fontweight="bold")
        plt.xlim(date(2020, 1, 1))
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(15)
        ax.grid(True, color="k", linestyle="-", linewidth=1, alpha=0.3)
        ax.legend(loc="best", prop={"size": 16})
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        fd = yf.download(self.ticker, period="1d")
        x = round(float(fd["Adj Close"]), 2)
        st.write(f" - {self.sName} · [{self.ticker}] · Current Stock Price = **${x}**")
        x = dfreg["Forecast"].iloc[-1]
        st.write(f" - 1 year Forcasted Price = **${round(float(x),2)}**")
        return


if __name__ == "__main__":

    stock_ticker = "TSLA"
    no_days = 24
    res_lst = Regression_Model(stock_ticker).preprocessing()
    if res_lst[-1] == "linear_regression":
        Regression_Model(stock_ticker).linear_regression(
            res_lst[0],
            res_lst[1],
            res_lst[2],
            res_lst[3],
            res_lst[4],
            res_lst[5],
            no_days,
        )
    if res_lst[-1] == "quadratic_regression_2":
        Regression_Model(stock_ticker).quadratic_regression_2(
            res_lst[0],
            res_lst[1],
            res_lst[2],
            res_lst[3],
            res_lst[4],
            res_lst[5],
            no_days,
        )
    if res_lst[-1] == "quadratic_regression_3":
        Regression_Model(stock_ticker).quadratic_regression_3(
            res_lst[0],
            res_lst[1],
            res_lst[2],
            res_lst[3],
            res_lst[4],
            res_lst[5],
            no_days,
        )
    if res_lst[-1] == "knn":
        Regression_Model(stock_ticker).knn(
            res_lst[0],
            res_lst[1],
            res_lst[2],
            res_lst[3],
            res_lst[4],
            res_lst[5],
            no_days,
        )
