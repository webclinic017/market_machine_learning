import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime, date, timedelta
from yahooquery import Ticker, ticker

from src.tools import functions as f0
from src.tools import lists as l0
from src.tools import scripts as s0
from src.tools import widgets as w0
import src.models.forecast as f1


casting_periods = [21, 63, 126, 252, 378, 504]

prophet_url = "https://facebook.github.io/prophet/docs/quick_start.html#python-api"
prophet_script_1 = "\n\
- Prophet is a procedure for forecasting time series data based on an additive model where \
    non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects."
prophet_script_2 = "\n\
- It works best with time series that have strong seasonal effects and several seasons of historical data. \
    Prophet is robust to missing data and shifts in the trend, and typically handles outliers well."


univariate_script = f"\
    * The term 'univariate' implies that forecasting is based on a sample of time series \
    observations of the exchange rate without taking into account the effect of the other \
    variables such as prices and interest rates. If this is the case, then there is no need \
    to take an explicit account of these variables."


class Forecast(object):
    def __init__(self, today_stamp):
        self.today_stamp = today_stamp
        self.saveMonth = str(datetime.now())[:7]
        self.saveDay = str(datetime.now())[8:10]

        self.saveRec = Path(f"data/recommenders/{str(today_stamp)[:4]}/{self.saveMonth}/{self.today_stamp}/")
        if not self.saveRec.exists():
            self.saveRec.mkdir(parents=True)

        self.saveRaw = Path(f"data/raw/{self.saveMonth}/{self.today_stamp}/")
        if not self.saveRaw.exists():
            self.saveRaw.mkdir(parents=True)

        self.saveScreeners = Path(f"data/screeners/{self.saveMonth}/{self.today_stamp}/")
        if not self.saveScreeners.exists():
            self.saveScreeners.mkdir(parents=True)

        self.saveTickers = Path(f"data/tickers/{self.saveMonth}/{self.today_stamp}/")
        if not self.saveTickers.exists():
            self.saveTickers.mkdir(parents=True)

        st.sidebar.caption("* __BE ADVISED - ALL MODELS IN THIS SECTION WILL TAKE TIME TO RUN__")


    def prophet(self, ticker_list):
        st.header("**Prophet · Model** - [Time Series Analysis & Forecast]")
        my_expander = st.expander("Expand", expanded=False)
        with my_expander:
            clicked = w0.widget_prophet(prophet_script_1, prophet_script_2, prophet_url)

        ender_date = str(st.sidebar.date_input("End Date", datetime(2021, 7, 19)))[:10]
        prophet_period_1 = st.sidebar.radio("Forcast Period (DAYS)", casting_periods, index=3)

        st.sidebar.write(" *" * 25)
        st.sidebar.markdown("Click to run forecasting model")
        if st.sidebar.button("RUN PROPHET FORECAST"):
            if type(self.one_er_many) == str:
                for r in ticker_list:
                    f1.prophet(r, ender_date, prophet_period_1, hist="2y").run_prophet()
                    st.write(" *" * 25)
            elif type(self.one_er_many) == int:
                f1.prophet(ticker_list, ender_date, prophet_period_1, hist="2y").run_prophet()


    def mc_forecast(self, ticker_list):
        st.header("__Monte Carlo Simulation Forecast__")
        forecast_days = st.sidebar.number_input(label="Enter Forecast Days", value=252, min_value=30, max_value=750)

        if st.sidebar.button("Run Monte Carlo Sim Forecast"):
            f1.MC_Forecast().monte_carlo(
                tickers=ticker_list,
                days_forecast=forecast_days,
                iterations=13000,
                start_date="2010-01-01",
                return_type="log",
                plotten=False,
            )


    def stocker(self, ticker_list):
        st.header("** > Stocker · Model **")
        stocker_forcast_period = st.sidebar.radio("stocker forecast period", casting_periods, index=3)
        st.sidebar.markdown("Click to run forecasting model")

        e_date = str(st.sidebar.date_input("End Date", datetime(2021, 10, 19)))[:10]

        if st.sidebar.button("RUN STOCKER FORECAST"):

            if type(self.one_er_many) == str:
                for r in ticker_list:
                    f1.web_stocker_run(r, stocker_forcast_period, e_date)
            elif type(self.one_er_many) == int:
                f1.web_stocker_run(ticker_list, stocker_forcast_period, e_date)


    def regression(self, ticker_list):
        st.header("** Regression · Analysis **")
        st.write(" *" * 25)
        st.sidebar.markdown("Click to run forecasting model")
        if st.sidebar.button("RUN REGRESSION FORECAST"):
            if type(ticker_list) == list:
                for stock_ticker in ticker_list:
                    days = 2
                    (
                        dfreg,
                        X_lately,
                        clfreg,
                        clfpoly2,
                        clfpoly3,
                        clfknn,
                        modName,
                    ) = f1.regression(stock_ticker).preprocessing()

                    if modName == "linear_regression":
                        f1.regression(stock_ticker).linear_regression(
                            dfreg, X_lately, clfreg, clfpoly2, clfpoly3, clfknn, days
                        )
                        st.write(" *" * 25)

                    if modName == "quadratic_regression_2":
                        f1.regression(stock_ticker).quadratic_regression_2(
                            dfreg, X_lately, clfreg, clfpoly2, clfpoly3, clfknn, days
                        )
                        st.write(" *" * 25)

                    if modName == "quadratic_regression_3":
                        f1.regression(stock_ticker).quadratic_regression_3(
                            dfreg, X_lately, clfreg, clfpoly2, clfpoly3, clfknn, days
                        )
                        st.write(" *" * 25)

                    if modName == "knn":
                        f1.regression(stock_ticker).knn(
                            dfreg, X_lately, clfreg, clfpoly2, clfpoly3, clfknn, days
                        )
                        st.write(" *" * 25)
            else:
                days = 20
                (
                    dfreg,
                    X_lately,
                    clfreg,
                    clfpoly2,
                    clfpoly3,
                    clfknn,
                    modName,
                ) = f1.regression(ticker_list).preprocessing()

                if modName == "linear_regression":
                    f1.regression(ticker_list).linear_regression(
                        dfreg, X_lately, clfreg, clfpoly2, clfpoly3, clfknn, days
                    )
                    st.write(" *" * 25)

                if modName == "quadratic_regression_2":
                    f1.regression(ticker_list).quadratic_regression_2(
                        dfreg, X_lately, clfreg, clfpoly2, clfpoly3, clfknn, days
                    )
                    st.write(" *" * 25)

                if modName == "quadratic_regression_3":
                    f1.regression(ticker_list).quadratic_regression_3(
                        dfreg, X_lately, clfreg, clfpoly2, clfpoly3, clfknn, days
                    )
                    st.write(" *" * 25)

                if modName == "knn":
                    f1.regression(ticker_list).knn(
                        dfreg, X_lately, clfreg, clfpoly2, clfpoly3, clfknn, days
                    )
                    st.write(" *" * 25)


    def arima(self, ticker_list):
        st.header("(A.R.I.M.A)")
        st.subheader("Auto Regression Integrated Moving Average")
        st.write(" *" * 25)
        st.sidebar.markdown("Click to run forecasting model")
        if st.sidebar.button("RUN ARIMA FORECAST"):
            if type(ticker_list) == list:
                # ticker.split()
                for stock_ticker in ticker_list:
                    # f1.arima1(stock_ticker).arima_model()
                    f1.arima2(stock_ticker).runArima()
                    st.write(" *" * 25)
            if type(ticker_list) == str:
                f1.arima1(ticker_list).arima_model()
                f1.arima2(stock_ticker).runArima()


    def sarima(self, ticker_list):
        st.header(" > S.A.R.I.M.A")
        st.subheader("Seasonal AutoRegressive Integrated Moving Average")
        st.subheader("SARIMA MODELING & FORECAST")
        st.write(
            f"\
            - A Seasonal-AutoRegressive-Integrated-MovingAverage (SARIMA) model is 1 step more than \
            an ARIMA model based on the concept of seasonal trends \n\
            {'__'*25}"
        )
        st.sidebar.markdown("Click to run forecasting model")
        if st.sidebar.button("RUN SARIMA FORECAST"):
            if type(ticker_list) == list:
                # ticker_list.split()
                for stock_ticker in ticker_list:
                    f1.sarima(stock_ticker).predict()
                    st.write(" *" * 25)
            if type(ticker_list) == str:
                f1.sarima(ticker_list).predict()
                st.write(" *" * 25)


    def monte_carlo(self, ticker_list):
        st.header(" > Monte Carlo Simulations")
        st.write(
            f"\
            * A Monte Carlo simulation is a useful tool for predicting future results calculating \
                a formula multiple times with different random inputs. \n\
            {'__'*25}"
        )
        st.sidebar.markdown("Click to run forecasting model")
        if st.sidebar.button("RUN MONTE CARLO FORECAST"):
            for stock_ticker in ticker_list:
                f1.monteCarlo(stock_ticker)
                st.write(" *" * 25)


    def univariate(self, ticker_list):
        st.header(" > Univariate · TimeSeries · RNN · Forecast ")
        with st.expander("Expand"):
            clicked = w0.widget_univariate("> Details:", univariate_script)

        st.sidebar.markdown("Click to run forecasting model")
        if st.sidebar.button("RUN UNIVARIATE FORECAST"):

            if type(self.one_er_many) == str:
                for stock_ticker in ticker_list:
                    f1.univariate_1(stock_ticker).runs()
                    # f1.univariate_2(stock_ticker).runs()
                    st.write(" *" * 25)
            elif type(self.one_er_many) == int:
                f1.univariate_1(ticker_list).runs()

    # ------------------------------------------------------------------------------------------ > stage: [FORECAST]

    def run_forecast(self):
        st.title("** [Forecasting] **")
        options_f = ["Personal Portfolio", "Recommended Stocks"]#, "Ticker Lists"]
        st.sidebar.subheader("**[2] Select Stocks To Run**")
        st.write("__" * 25)

        self.one_er_many = "List"
        st.sidebar.caption("* __CAREFUL RUNNING THESE MODELS WITH TOO MANY TICKERS - Each Model is time consuming__")
        personal_stocks = st.sidebar.text_input("Enter Stocks", value="AAPL")
        st.sidebar.write(" *" * 25)


        personal_stocks = personal_stocks.split()
        st.sidebar.subheader("[3 ] Select Model To Run")
        model = st.sidebar.radio("Model List:", l0.feature_forecast)
        st.sidebar.write(" *" * 25)

        if model == "Prophet Model":
            self.prophet(personal_stocks)

        if model == "Monte Carlo Sim-Cast":
            self.mc_forecast(personal_stocks)

        if model == "Stocker Analysis":
            self.stocker(personal_stocks)

        if model == "Regression":
            self.regression(personal_stocks)

        if model == "A.R.I.M.A":
            self.arima(personal_stocks)

        if model == "S.A.R.I.M.A":
            self.sarima(personal_stocks)

        if model == "Monte Carlo Simulation":
            self.monte_carlo(personal_stocks)

        if model == "Univariate Analysis":
            self.univariate(personal_stocks)
