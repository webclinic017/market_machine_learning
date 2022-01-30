import warnings
import pandas as pd
from datetime import date
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.tsa.arima.model as smt
import yfinance as yf
import streamlit as st
from yahooquery import Ticker

warnings.filterwarnings("ignore", "statsmodels.tsa.arima_model.ARMA", FutureWarning)
warnings.filterwarnings("ignore", "statsmodels.tsa.arima_model.ARIMA", FutureWarning)
warnings.warn("ARIMA_DEPRECATION_WARN", FutureWarning)
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
color_discrete_sequence = px.colors.qualitative.G10


def company_longName(symbol):
    d = Ticker(symbol).quote_type
    return list(d.values())[0]["longName"]


class Arima2(object):


    def __init__(self, ticker):
        self.stock = ticker
        self.df = yf.download(self.stock, period="10y", parse_dates=True)
        self.companyLongName = company_longName(self.stock)


    def runArima(self):
        st.header(f"{self.companyLongName} [{self.stock}] A.R.I.M.A. Model")
        train_data, test_data = (
            self.df[0 : int(len(self.df) * 0.7)],
            self.df[int(len(self.df) * 0.7) :],
        )
        training_data = train_data["Close"].values
        test_data = test_data["Close"].values
        history = [x for x in training_data]
        model_predictions = []
        N_test_observations = len(test_data)

        for time_point in range(N_test_observations):
            model = smt.ARIMA(history, order=(4, 1, 0))
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            model_predictions.append(yhat)
            true_test_value = test_data[time_point]
            history.append(true_test_value)
        test_set_range = self.df[int(len(self.df) * 0.7) :].index

        plt.rcParams["figure.figsize"] = (15, 5)
        fig, ax = plt.subplots()
        plt.plot(
            test_set_range,
            model_predictions,
            color="blue",
            marker="o",
            linestyle="dashed",
            label="Predicted Price",
        )
        plt.plot(test_set_range, test_data, color="red", label="Actual Price")
        plt.title(
            f"{self.companyLongName} ({self.stock}) - Prices Prediction",
            fontsize=30,
            fontweight="bold",
        )
        plt.xlabel("Date", fontsize=20, fontweight="bold")
        plt.ylabel("Prices", fontsize=20, fontweight="bold")
        plt.xlim(date(2020, 9, 1), date(2021, 6, 1))
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(15)
        ax.grid(True, color="k", linestyle="-", linewidth=1, alpha=0.3)
        ax.legend(loc="best", prop={"size": 16})
        plt.tight_layout()
        st.pyplot(fig)


if __name__ == "__main__":
    Arima2("BA").runArima()

# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *



class The_Arima_Model(object):


    def __init__(self, ticker, period="1y", interval="1d"):
        self.ticker = ticker
        self.period = period
        self.interval = interval
        self.company = company_longName(self.ticker)


    def arima_model(self):
        data = yf.download(self.ticker, period=self.period, interval=self.interval)
        df = pd.DataFrame(data["Close"])
        df.reset_index(inplace=True)
        train_data, test_data = df[0 : int(len(df) * 0.7)], df[int(len(df) * 0.7) :]
        training_data = train_data["Close"].values
        test_data = test_data["Close"].values
        history = [x for x in training_data]
        model_predictions = []
        N_test_observations = len(test_data)

        for time_point in range(N_test_observations):
            model_fit = smt.ARIMA(history, order=(4, 1, 0)).fit()
            output = model_fit.forecast()[0]
            yhat = output[0]
            model_predictions.append(yhat)
            true_test_value = test_data[time_point]
            history.append(true_test_value)
        df.set_index("Date", inplace=True)
        test_set_range = df[int(len(df) * 0.7) :].index

        fig, ax = plt.subplots()
        plt.plot(
            test_set_range,
            model_predictions,
            color="blue",
            marker="X",
            linestyle="--",
            label="Predicted Price",
        )
        plt.plot(test_set_range, test_data, color="red", label="Actual Price")
        plt.title(
            f"{self.company} ({self.ticker}) - Prices Prediction",
            fontsize=30,
            fontweight="bold",
        )
        plt.xlabel("Date", fontsize=20, fontweight="bold")
        plt.ylabel("Prices", fontsize=20, fontweight="bold")
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(15)
        ax.grid(True, color="k", linestyle="-", linewidth=1, alpha=0.3)
        ax.legend(loc="best", prop={"size": 16})
        plt.tight_layout()
        plt.grid(True)
        st.pyplot(fig)


if __name__ == "__main__":
    The_Arima_Model("BA").arima_model()
