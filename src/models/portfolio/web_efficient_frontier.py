import warnings

warnings.filterwarnings("ignore")
import yfinance as yf
import numpy as np
import pandas as pd

import matplotlib
import matplotlib as mpl

matplotlib.use("Agg")
from matplotlib import style
from matplotlib import pyplot as plt

plt.style.use("ggplot")
import seaborn as sns

plt.style.use("seaborn")
sns.set_palette("cubehelix")
plt.rcParams["figure.figsize"] = [18, 10]
plt.rcParams["figure.dpi"] = 150

sm, med, lg = 10, 15, 20
plt.rc("font", size=sm)  # controls default text sizes
plt.rc("axes", titlesize=med)  # fontsize of the axes title
plt.rc("axes", labelsize=med)  # fontsize of the x & y labels
plt.rc("xtick", labelsize=sm)  # fontsize of the tick labels
plt.rc("ytick", labelsize=sm)  # fontsize of the tick labels
plt.rc("legend", fontsize=sm)  # legend fontsize
plt.rc("figure", titlesize=lg)  # fontsize of the figure title
plt.rc("axes", linewidth=2)  # linewidth of plot lines

import streamlit as st

from pathlib import Path

path = str(Path.cwd()) + "/"
from datetime import datetime

today = str(datetime.now())[:10]


# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


class The_Efficient_Frontier(object):
    def __init__(self, RISKY_ASSETS):
        self.RISKY_ASSETS = RISKY_ASSETS
        self.prices_df = yf.download(self.RISKY_ASSETS, start="2020-01-01")["Adj Close"]
        self.N_PORTFOLIOS = 10 ** 5
        self.N_DAYS = 252
        self.n_assets = len(self.RISKY_ASSETS)

        self.string = ""
        for r in self.RISKY_ASSETS:
            self.string += r + "_"

    def ef_setup(self):
        self.returns_df = self.prices_df.pct_change().dropna()
        self.avg_returns = self.returns_df.mean() * self.N_DAYS
        self.cov_mat = self.returns_df.cov() * self.N_DAYS
        # simulate random portfolio weights:
        np.random.seed(42)
        self.weights = np.random.random(size=(self.N_PORTFOLIOS, self.n_assets))
        self.weights /= np.sum(self.weights, axis=1)[:, np.newaxis]
        # calculate portfolio metrics:
        self.portf_rtns = np.dot(self.weights, self.avg_returns)
        self.portf_vol = []
        for i in range(0, len(self.weights)):
            self.portf_vol.append(
                np.sqrt(
                    np.dot(self.weights[i].T, np.dot(self.cov_mat, self.weights[i]))
                )
            )
        self.portf_vol = np.array(self.portf_vol)
        self.portf_sharpe_ratio = self.portf_rtns / self.portf_vol
        # create joint dataframe with all data:
        self.portf_results_df = pd.DataFrame(
            {
                "returns": self.portf_rtns,
                "volatility": self.portf_vol,
                "sharpe_ratio": self.portf_sharpe_ratio,
            }
        )
        # locate points creating efficient frontier:
        self.N_POINTS = 100
        self.portf_vol_ef = []
        self.indices_to_skip = []
        self.portf_rtns_ef = np.linspace(
            self.portf_results_df.returns.min(),
            self.portf_results_df.returns.max(),
            self.N_POINTS,
        )
        self.portf_rtns_ef = np.round(self.portf_rtns_ef, 2)
        self.portf_rtns = np.round(self.portf_rtns, 2)
        for point_index in range(self.N_POINTS):
            if self.portf_rtns_ef[point_index] not in self.portf_rtns:
                self.indices_to_skip.append(point_index)
                continue
            self.matched_ind = np.where(
                self.portf_rtns == self.portf_rtns_ef[point_index]
            )
            self.portf_vol_ef.append(np.min(self.portf_vol[self.matched_ind]))
        self.portf_rtns_ef = np.delete(self.portf_rtns_ef, self.indices_to_skip)

    def results_maxSharpeRatio(self):
        self.ef_setup()
        self.max_sharpe_ind = np.argmax(self.portf_results_df.sharpe_ratio)
        self.max_sharpe_portf = self.portf_results_df.loc[self.max_sharpe_ind]
        self.min_vol_ind = np.argmin(self.portf_results_df.volatility)
        self.min_vol_portf = self.portf_results_df.loc[self.min_vol_ind]
        st.header("- - - Maximum Sharpe Ratio portfolio - - -")
        st.subheader("Performance:")
        for index, value in self.max_sharpe_portf.items():
            st.write(f"{index}: {100 * value:.2f}% ", end="", flush=True)
        st.subheader("\nWeights")
        for x, y in zip(
            self.RISKY_ASSETS,
            self.weights[np.argmax(self.portf_results_df.sharpe_ratio)],
        ):
            st.write(f"{x}: {100*y:.2f}% ", end="", flush=True)

    def results_minVolatility(self):
        self.results_maxSharpeRatio()
        st.header("- - - Minimum Volatility portfolio - - -")
        st.subheader("Performance:")
        for index, value in self.min_vol_portf.items():
            st.write(f"{index}: {100 * value:.2f}% ", end="", flush=True)
        st.subheader("\nWeights")
        for x, y in zip(
            self.RISKY_ASSETS, self.weights[np.argmin(self.portf_results_df.volatility)]
        ):
            st.write(f"{x}: {100*y:.2f}% ", end="", flush=True)

    def final_plot(self):
        self.results_minVolatility()
        fig, ax = plt.subplots()
        self.portf_results_df.plot(
            kind="scatter",
            x="volatility",
            y="returns",
            c="sharpe_ratio",
            cmap="RdYlGn",
            edgecolors="black",
            ax=ax,
        )
        ax.scatter(
            x=self.max_sharpe_portf.volatility,
            y=self.max_sharpe_portf.returns,
            c="black",
            marker="X",
            s=175,
            label="Max Sharpe Ratio",
        )
        ax.scatter(
            x=self.min_vol_portf.volatility,
            y=self.min_vol_portf.returns,
            c="black",
            marker="P",
            s=175,
            label="Min Volatility",
        )
        self.portf_results_df.plot(
            kind="scatter",
            x="volatility",
            y="returns",
            c="sharpe_ratio",
            cmap="RdYlGn",
            edgecolors="black",
            ax=ax,
        )
        ax.set(
            xlabel="Volatility", ylabel="Expected Returns", title="Efficient Frontier"
        )
        ax.plot(self.portf_vol_ef, self.portf_rtns_ef, "b--")
        for asset_index in range(self.n_assets):
            ax.scatter(
                x=np.sqrt(self.cov_mat.iloc[asset_index, asset_index]),
                y=self.avg_returns[asset_index],
                # marker=self.MARKS[asset_index],
                s=100,
                color="black",
                label=self.RISKY_ASSETS[asset_index],
            )
        ax.set(
            xlabel="Volatility",
            ylabel="Expected Returns",
            title=f"Efficient Frontier",  # {self.string}",
        )
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(15)
        ax.grid(True, color="k", linestyle="-", linewidth=1, alpha=0.3)
        ax.legend(loc="best", prop={"size": 16})
        plt.tight_layout()
        st.pyplot(fig)


# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


if __name__ == "__main__":

    RISKY_ASSETS = []
    manys = [2, 4, 6, 8, 10, 12, 14]
    how_many = int(
        st.sidebar.selectbox("Select Number Of Securities For Portfolio:", manys)
    )
    # how_many = int(input('How Many Stocks In Your Portfolio? (up to 14): '))
    for i in range(1, how_many + 1):
        tic = input(f"Enter Stock {i}: ")
        RISKY_ASSETS.append(tic)
    RISKY_ASSETS.sort()

    marks0 = ["o", "^", "s", "p", "h", "8", "*", "d", ">", "v", "<", "1", "2", "3", "4"]
    mark = marks0[: len(RISKY_ASSETS) + 1]

    The_Efficient_Frontier(RISKY_ASSETS).final_plot()


# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
