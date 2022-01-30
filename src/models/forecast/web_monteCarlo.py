import warnings
from datetime import datetime
import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from yahooquery import Ticker

warnings.filterwarnings("ignore")
mpl.use("Agg")
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


class The_Monte_Carlo(object):
    def __init__(self, stock, hist="5y"):
        self.stock = stock

        def company_longName(symbol):
            d = Ticker(symbol).quote_type
            return list(d.values())[0]["longName"]

        self.company = company_longName(self.stock)

        if self.stock:
            stock_data = yf.Ticker(stock)
            df = stock_data.history(hist, auto_adjust=True)
            # - Calculate daily returns:
            adj_close = df["Close"]
            returns = adj_close.pct_change().dropna()
            # - Split data into the training and test sets:
            train = returns[:"2021-01-01"]
            # a = '2020-01-01'
            # b = '2020-12-31'
            test = returns["2021-01-01":]
            # - Specify the parameters of the simulation:
            dt = 1
            T = 150
            N = len(test)
            S_0 = adj_close[train.index[-1]]
            N_SIM = 1000
            mu = train.mean()
            sigma = train.std()

            #     *     *     *     *     *     *     *        *     *     *     *     *     *
            def simulate_gbm(s_0, mu, sigma, n_sims, dt, T, N, random_seed=42):
                """
                Function used for simulating stock returns using Geometric Brownian Motion.
                Parameters ------------
                s_0 : float - Initial stock price
                mu : float - Drift coefficient
                sigma : float - Diffusion coefficient
                n_sims : int - Number of simulations paths
                dt : float - Time increment, most commonly a day
                T : float - Length of the forecast horizon, same unit as dt
                N : int - Number of time increments in the forecast horizon
                random_seed : int - Random seed for reproducibility
                Returns ----------- S_t : np.ndarray
                    Matrix (size: n_sims x (T+1)) containing the simulation results.
                    Rows respresent sample paths, while columns point of time.
                """
                np.random.seed(random_seed)
                dt = T / N
                dW = np.random.normal(scale=np.sqrt(dt), size=(n_sims, N))
                W = np.cumsum(dW, axis=1)

                #     *     *     *     *     *     *     *        *     *     *     *     *     *
                time_step = np.linspace(dt, T, N)
                time_steps = np.broadcast_to(time_step, (n_sims, N))

                #     *     *     *     *     *     *     *        *     *     *     *     *     *
                S_t = s_0 * np.exp((mu - 0.5 * sigma ** 2) * time_steps + sigma * W)
                S_t = np.insert(S_t, 0, s_0, axis=1)
                return S_t

            #     *     *     *     *     *     *     *     *     *     *     *      >Run the sim
            gbm_simulations = simulate_gbm(S_0, mu, sigma, N_SIM, dt, T, N)
            # >create sim date results
            last_train_date = train.index[-1].date()
            # first_test_date = test.index[0].date()
            last_test_date = test.index[-1].date()
            selected_indices = adj_close[last_train_date:last_test_date].index
            index = [date.date() for date in selected_indices]
            gbm_simulations_df = pd.DataFrame(
                np.transpose(gbm_simulations), index=index
            )

            #     *     plotting    *     *     *     *     *     *
            fig, ax = plt.subplots()
            ax = gbm_simulations_df.plot(alpha=0.2, legend=False)
            (line_1,) = ax.plot(index, gbm_simulations_df.mean(axis=1), color="red")
            (line_2,) = ax.plot(
                index, adj_close[last_train_date:last_test_date], color="blue"
            )
            ax.set_title(
                f"{self.company} - ({self.stock}) - Monte Carlo Simulations",
                fontsize=30,
                fontweight="bold",
            )
            ax.legend((line_1, line_2), ("mean-price", "actual-price"))
            plt.xlabel("Test Date Range", fontsize=20, fontweight="bold")
            plt.ylabel("Stock Price", fontsize=20, fontweight="bold")
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(15)
            ax.grid(True, color="k", linestyle="-", linewidth=1, alpha=0.3)
            plt.tight_layout()
            st.pyplot(plt.show())


if __name__ == "__main__":
    stock_ticker = "TSLA"
    if stock_ticker:
        The_Monte_Carlo(stock_ticker)
