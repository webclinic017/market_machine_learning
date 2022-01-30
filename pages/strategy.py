from re import S
import streamlit as st
from datetime import datetime, date
from pathlib import Path
import time
from yahooquery import Ticker
import pandas as pd
import yfinance as yf

import src.models.strategy as s1
from src.tools import functions as f0
from src.tools import lists as l0


class Strategy(object):

    def __init__(self, today_stamp):
        self.today_stamp = str(today_stamp)[:10]
        self.saveMonth = str(datetime.now())[:7]
        self.saveDay = str(datetime.now())[8:10]
        self.saveRec = Path(f"data/recommenders/{str(today_stamp)[:4]}/{self.saveMonth}/{self.today_stamp}/")


    def run_the_strats(self):
        st.sidebar.header("[3] Select Stock")
        st.sidebar.write(" *" * 25)
        st.sidebar.header("[4] Select Method [All or 1]")
        method_strat = st.sidebar.radio(
            "Pick Method", 
            (
                "Individual Strategy", 
                "Run All Strategies"
            )
        )

        self.stock_ticker = st.sidebar.text_input(
            label="Enter Stock In ALL CAPS [example: TSLA]", 
            value="TSLA"
        )

        if method_strat == "Run All Strategies":
            st.sidebar.write(" *" * 25)
            if st.sidebar.button("Run Strategies"):
                self.run_movAvg_sma_ema(self.stock_ticker, "SMA")
                self.run_optimal_sma(self.stock_ticker)
                self.run_overBought_overSold(self.stock_ticker)
                self.run_supportResistance(self.stock_ticker)
                self.run_strategyII(self.stock_ticker)


        if method_strat == "Individual Strategy":
            st.sidebar.write(" *" * 25)
            st.sidebar.header("[5] Select Model")
            model = st.sidebar.radio("Choose A Model", l0.feature_strategy)
            st.sidebar.write(" *" * 25)

            if model == "-Select-Model-":
                self.run_homePage()

            if model == "Moving-Average - SMA & EMA":
                sma_ema_choice = st.sidebar.radio("Choose Moving Average Method", ("SMA", "EMA"))
                inter = st.sidebar.radio('Interval',
                    ('1m','2m','5m','15m','30m','60m','90m','1h','1d','5d','1wk','1mo','3mo'),
                    index=8
                )
                st.sidebar.write("__" * 25)
                if st.sidebar.button("Run Strategy"):
                    data = Ticker(
                        self.stock_ticker,
                        asynchronous=True,
                        formatted=False,
                        backoff_factor=0.34,
                        validate=True,
                        verify=True,
                    )
                    data = data.history(period='max').reset_index().set_index('date')
                    data.index = pd.to_datetime(data.index)
                    data = pd.DataFrame(data[data['symbol'] == [self.stock_ticker][0]])
                    del data['symbol']                    
                    self.run_movAvg_sma_ema(self.stock_ticker, data, sma_ema_choice, True, inter)


            if model == "Optimal SMA":
                if st.sidebar.button("Run Strategy"):
                    data = Ticker(
                        self.stock_ticker,
                        asynchronous=True,
                        formatted=False,
                        backoff_factor=0.34,
                        validate=True,
                        verify=True,
                    )
                    data = data.history(period='max').reset_index().set_index('date')
                    data.index = pd.to_datetime(data.index)
                    data = pd.DataFrame(data[data['symbol'] == [self.stock_ticker][0]])
                    del data['symbol']                     
                    self.run_optimal_sma(self.stock_ticker, data)


            if model == "OverBought & OverSold":
                if st.sidebar.button("Run Strategy"):
                    self.run_overBought_overSold(self.stock_ticker)


            if model == "Support & Resistance Lines":
                if st.sidebar.button("Run Strategy"):
                    self.run_supportResistance(self.stock_ticker)


            if model == "Strategy II":
                if st.sidebar.button("Run Strategy"):
                    self.run_strategyII(self.stock_ticker)


    def run_homePage(self):
        st.header("[Strategy Home Page]")
        st.write(" *" * 25)

        st.subheader("** > General Analysis Components **")
        with st.expander("Expand For Details", expanded=False):
            st.subheader("Moving Averages")
            st.write(
                """
                * Double Moving Averages
                * Exponential Moving Average (EMA)
                * Simple Moving Average (SMA)
                * Bollinger Bands
                * MOM
                * MACD
                * RSI
                * APO
                """
            )
            st.subheader("Regression")
            st.write(
                """
                * Linear Regression
                * Quadratic Regression 2 & 3
                * KNN
                * Lasso
                * Ridge
                * Logistic Regression
                """
            )
            st.subheader("Speciality Trading")
            st.write(
                """
                * naive momentum
                * Pairs Correlation Trading
                * Support & Resistance
                * Turtle Trading
                * Mean Reversion & Trend Following
                * Volatility Mean Reversion & Trend Following
                * OverBought & OverSold
                """
            )
            st.subheader("Strategy Backtesting")
            st.write("* xgboost sim/backtesting")
            st.write("* backtrader backtesting")


    def run_movAvg_sma_ema(self, stock_ticker, data, sma_ema_opt, p_out=True, cc=0.0, ccc=0.0, inter='1d'):
        res, S, L = s1.optimal_2sma(stock_ticker).grab_data(self.today_stamp, data, inter)
        stock_symbol = s1.movAvg_sma_ema(stock_ticker, S, L, self.today_stamp, sma_ema_opt, data, p_out, cc, ccc, inter)        
        st.write('_'*25)
        if stock_symbol == stock_ticker:
            return True
        if stock_symbol != stock_ticker:
            return False
        else:
            return False
        

    def run_optimal_sma(self, stock_ticker, data, graphit=True, cc=0.0, ccc=0.0):
        stock_symbol = s1.optimal_sma(stock_ticker, self.today_stamp).build_optimal_sma(data, graphit, cc, ccc)
        st.write('_'*25)
        if stock_symbol == stock_ticker:
            return True
        if stock_symbol != stock_ticker:
            return False
        else:
            return False



    def run_overBought_overSold(self, stock_ticker):
        st.write('_'*25)
        st.subheader("** > Over Bought & Over Sold Analysis **")
        s1.overBought_overSold(stock_ticker).generate()


    def run_supportResistance(self, stock_ticker):
        st.write('_'*25)
        st.subheader("** > Support & Resistance Lines **")
        s1.support_resistance(stock_ticker).level()


    def run_strategyII(self, stock_ticker):
        st.write('_'*25)
        st.subheader("** > Strategy II **")
        s1.Trading_Technicals(stock_ticker).trading_technicals()
