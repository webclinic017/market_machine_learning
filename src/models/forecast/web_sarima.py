import warnings
from datetime import datetime, date
import pandas as pd
import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import itertools
import streamlit as st
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from yahooquery import Ticker

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
plt.rcParams["figure.figsize"] = [20, 10]
plt.rcParams["figure.dpi"] = 100
plt.rcParams["axes.facecolor"] = "silver"


def company_longName(symbol):
    d = Ticker(symbol).quote_type
    return list(d.values())[0]["longName"]



class The_SARIMA_Model(object):


    def __init__(self, stock):
        self.sss = stock
        self.company = company_longName(self.sss)


    def dataHull(self):
        self.start = "2011-10-01"
        self.end = "2021-10-19"

        self.x_data = yf.download(self.sss, start=self.end)["Adj Close"]
        self.x_data.columns = [self.company]

        self.spData = yf.download(self.sss, period='max')
        self.spData = pd.DataFrame(self.spData.loc[:self.end])
        self.dataSP = pd.DataFrame(self.spData["Close"])
        self.dataSP.columns = [self.sss]
        self.dataSP.index = pd.to_datetime(self.dataSP.index)

        self.df_settle = self.spData["Close"].resample("BM").ffill().dropna()
        self.df_rolling = self.df_settle.rolling(12)
        self.df_mean = self.df_rolling.mean()
        self.df_std = self.df_rolling.std()


    def adf(self):
        self.dataHull()
        self.result = adfuller(self.df_settle)
        self.critical_values = self.result[4]
        self.df_log = np.log(self.df_settle)
        self.df_log_ma = self.df_log.rolling(2).mean()
        self.df_detrend = self.df_log - self.df_log_ma
        self.df_detrend.dropna(inplace=True)

        # Mean and standard deviation of detrended data
        self.df_detrend_rolling = self.df_detrend.rolling(12)
        self.df_detrend_ma = self.df_detrend_rolling.mean()
        self.df_detrend_std = self.df_detrend_rolling.std()

        self.result2 = adfuller(self.df_detrend)
        self.critical_values2 = self.result2[4]
        self.df_log_diff = self.df_log.diff(periods=3).dropna()

        # Mean and standard deviation of differenced data
        self.df_diff_rolling = self.df_log_diff.rolling(12)
        self.df_diff_ma = self.df_diff_rolling.mean()
        self.df_diff_std = self.df_diff_rolling.std()


    def seasonal_decomp(self):
        self.adf()
        self.decompose_result = seasonal_decompose(self.df_log.dropna(), period=12)
        self.df_trend = self.decompose_result.trend
        self.df_season = self.decompose_result.seasonal
        self.df_residual = self.decompose_result.resid
        self.df_log_diff = self.df_residual.diff().dropna()

        # Mean and standard deviation of differenced data
        self.df_diff_rolling = self.df_log_diff.rolling(12)
        self.df_diff_ma = self.df_diff_rolling.mean()
        self.df_diff_std = self.df_diff_rolling.std()
        self.result = adfuller(self.df_residual.dropna())
        self.critical_values = self.result[4]


    def arima_grid_search(self, s=12):
        self.seasonal_decomp()
        self.s = s
        self.p = self.d = self.q = range(2)
        self.param_combinations = list(itertools.product(self.p, self.d, self.q))
        self.lowest_aic, self.pdq, self.pdqs = None, None, None
        self.total_iterations = 0
        for order in self.param_combinations:
            for (self.p, self.q, self.d) in self.param_combinations:
                self.seasonal_order = (self.p, self.q, self.d, self.s)
                self.total_iterations += 1
                try:
                    self.model = SARIMAX(
                        self.df_settle,
                        order=order,
                        seasonal_order=self.seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                        disp=False,
                    )
                    self.model_result = self.model.fit(maxiter=200, disp=False)
                    if not self.lowest_aic or self.model_result.aic < self.lowest_aic:
                        self.lowest_aic = self.model_result.aic
                        self.pdq, self.pdqs = order, self.seasonal_order
                except Exception:
                    continue
        return self.lowest_aic, self.pdq, self.pdqs


    def fitModel_to_SARIMAX(self):
        self.arima_grid_search()
        self.model = SARIMAX(
            self.df_settle,
            order=self.pdq,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=True,
            enforce_invertibility=True,
            disp=False,
        )
        self.model_results = self.model.fit(maxiter=200, disp=False)
        return self.model_results


    def predict(self):
        self.fitModel_to_SARIMAX()
        self.n = len(self.df_settle.index)
        self.prediction = self.model_results.get_prediction(start=self.n - 12 * 5, end=self.n + 12)
        self.prediction_ci = self.prediction.conf_int()

        fig, ax = plt.subplots()
        ax = self.df_settle["2017":].plot(label="actual")
        self.prediction_ci.plot(ax=ax, style=["--", "--"], lw=1, label="Predict")
        self.ci_index = self.prediction_ci.index
        self.lower_ci = self.prediction_ci.iloc[:, 0]
        self.upper_ci = self.prediction_ci.iloc[:, 1]
        ax.fill_between(
            self.ci_index,
            self.lower_ci,
            self.upper_ci,
            color="r",
            alpha=0.13,
            label="Confidence_Interval_(95%)",
        )
        ax.vlines(
            ["2018-05-01", "2019-12-25"],
            0,
            1,
            transform=ax.get_xaxis_transform(),
            colors="k",
            ls="--",
            lw=1.0,
            label="Train Period",
        )
        ax.vlines(
            ["2020-01-01", "2021-09-27"],
            0,
            1,
            transform=ax.get_xaxis_transform(),
            colors="r",
            ls="--",
            lw=1.0,
            label="Test Period",
        )
        ax.vlines(
            ["2021-10-19", "2022-10-19"],
            0,
            1,
            transform=ax.get_xaxis_transform(),
            colors="g",
            ls="--",
            lw=1.0,
            label="Forecast Period",
        )
        self.x_data.plot(lw=2, label="Price Since Prediction", marker='.', ms=10, color="b", ls="-")
        ax.set_xlabel("Time (years)", fontsize=20, fontweight="bold")
        ax.set_ylabel("Prices", fontsize=20, fontweight="bold")
        ax.set_title(
            f"{self.company} ({self.sss}) - SARIMA MODEL",
            fontsize=30,
            fontweight="bold",
        )
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(15)
        fontP = FontProperties()
        fontP.set_size("large")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", prop=fontP)
        ax.grid(True, color="k", linestyle="-", linewidth=1, alpha=0.3)
        plt.xlim(date(2018, 1, 1))
        plt.tight_layout()
        st.pyplot(fig)


if __name__ == "__main__":
    stock_ticker = "BA"
    if stock_ticker:
        The_SARIMA_Model(stock_ticker).predict()
