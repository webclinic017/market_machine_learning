from fastquant import backtest, get_stock_data
from datetime import datetime
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


today = str(datetime.now())[:10]


def build(ticker):
    fast = [5, 10, 15, 20, 25, 30]
    slow = [10, 20, 30, 40, 50, 60]

    prints = []

    df = get_stock_data(ticker, "2018-01-01", str(datetime.now())[:10])

    res0 = []

    for r in range(len(fast)):
        fig, ax = plt.subplots()
        prints.append(
            (res0[["fast_period", "slow_period", "init_cash", "final_value"]].head())
        )

    return prints


if __name__ == "__main__":
    P = build("AAPL")
    st.markdown(P)
