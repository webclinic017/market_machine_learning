from yahooquery import Ticker
import streamlit as st
import warnings
import pandas as pd
import numpy as np
import yfinance as yf
from itertools import product
import matplotlib.pyplot as plt
import os


class Optimal_Double_SMA(object):


    def __init__(self, tic):
        self.tic = tic


    def grab_data(self, ender, data, inter='1d'):
        # self.raw = pd.DataFrame(yf.download(self.tic, start='2020-01-03', end=str(ender)[:10], interval=inter))
        self.raw = pd.DataFrame(data.copy())
                        
        sma1 = range(2, 51, 2)
        sma2 = range(5, 126, 5)
        results = pd.DataFrame()
        for SMA1, SMA2 in product(sma1, sma2):
            data1 = pd.DataFrame(self.raw["adjclose"].copy())
            data1.columns = [self.tic]
            data1.dropna(inplace=True)
            data1["Returns"] = np.log(data1[self.tic] / data1[self.tic].shift(1))
            data1["SMA1"] = data1[self.tic].rolling(SMA1).mean()
            data1["SMA2"] = data1[self.tic].rolling(SMA2).mean()
            data1.dropna(inplace=True)
            data1["Position"] = np.where(data1["SMA1"] > data1["SMA2"], 1, -1)
            data1["Strategy"] = data1["Position"].shift(1) * data1["Returns"]
            data1.dropna(inplace=True)
            perf = np.exp(data1[["Returns", "Strategy"]].sum())
            results = results.append(
                pd.DataFrame(
                    {
                        "SMA1": SMA1,
                        "SMA2": SMA2,
                        "MARKET(%)": perf["Returns"],
                        "STRATEGY(%)": perf["Strategy"],
                        "OUT": (perf["Strategy"] - perf["Returns"]),
                    },
                    index=[0],
                ),
                ignore_index=True,
            )
        results = results.loc[results["SMA1"] < results["SMA2"]]
        results = (results.sort_values("OUT", ascending=False).reset_index(drop=True).head(10))

        S = results["SMA1"][0]
        L = results["SMA2"][0]
        # mkt = results["MARKET(%)"][0]
        # strat = results["STRATEGY(%)"][0]
        # out = results["OUT"][0]
        
        return results, S, L #, mkt, strat, out
