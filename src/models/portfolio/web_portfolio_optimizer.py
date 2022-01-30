from pathlib import Path
from datetime import datetime
from datetime import datetime
import yfinance as yf
from yahooquery import Ticker
import scipy.optimize as sco
from pandas.io.pickle import read_pickle
import pickle
import matplotlib
import matplotlib as mpl
from matplotlib import style
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow import keras

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
plt.rcParams["figure.figsize"] = [13, 7]
plt.rcParams["figure.dpi"] = 100
pd.options.display.max_rows = 999
pd.get_option("display.max_rows")
np.random.seed(777)

today_date = str(datetime.now())[:10]


# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


class The_Portfolio_Optimizer(object):
    def __init__(self, port_tics):
        self.port_tics = port_tics
        self.port_count = len(self.port_tics)

    def optimize(self):
        # saveName = str(f"{self.min_composite_score}_{self.min_RSI}_{self.min_Analyst_Recom}_{self.min_Sentiment}_{self.max_allocations}")
        hammerTime = Ticker(
            self.port_tics, 
            # asynchronous=True, 
            formatted=False, 
            backoff_factor=0.34
        )
        hT = hammerTime.history(start="2021-01-04")
        hT.head()

        Table = pd.DataFrame()
        for p in self.port_tics:
            hist = hT.copy().T
            hist = hist[p].T["adjclose"]
            # hist.to_pickle(self.saveAdvisor / f"{saveName}.pkl")
            Table[p] = hist.copy()

        # Table = yf.download(self.port_tics, period="1y", parse_dates=True)["Adj Close"]
        PT = pd.DataFrame(Table.iloc[1:])
        tickers = list(PT.columns)
        returns = PT.pct_change()
        returns.fillna(0.0, inplace=True)
        returns.dropna(inplace=True)
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        num_portfolios = 5000
        risk_free_rate = 0.0178

        def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
            Returns = np.sum(mean_returns * weights) * 252
            std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
            return std, Returns

        def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
            results = np.zeros((3, num_portfolios))
            weights_record = []
            for i in range(num_portfolios):
                weights = np.random.random(len(tickers))
                weights /= np.sum(weights)
                weights_record.append(weights)
                portfolio_std_dev, portfolio_return = portfolio_annualised_performance(
                    weights, mean_returns, cov_matrix
                )
                results[0, i] = portfolio_std_dev
                results[1, i] = portfolio_return
                results[2, i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
            return results, weights_record

        def display_simulated_ef_with_random(
            mean_returns, cov_matrix, num_portfolios, risk_free_rate
        ):
            results, weights = random_portfolios(
                num_portfolios, mean_returns, cov_matrix, risk_free_rate
            )
            max_sharpe_idx = np.argmax(results[2])
            sdp, rp = results[0, max_sharpe_idx], results[1, max_sharpe_idx]
            max_sharpe_allocation = pd.DataFrame(
                weights[max_sharpe_idx], index=PT.columns, columns=["allocation"]
            )
            max_sharpe_allocation.allocation = [
                round(i * 100, 2) for i in max_sharpe_allocation.allocation
            ]
            max_sharpe_allocation = max_sharpe_allocation.T
            min_vol_idx = np.argmin(results[0])
            sdp_min, rp_min = results[0, min_vol_idx], results[1, min_vol_idx]
            min_vol_allocation = pd.DataFrame(
                weights[min_vol_idx], index=PT.columns, columns=["allocation"]
            )
            min_vol_allocation.allocation = [
                round(i * 100, 2) for i in min_vol_allocation.allocation
            ]
            min_vol_allocation = min_vol_allocation.T

            max_sharpe_allocation_df = pd.DataFrame(
                max_sharpe_allocation.T
            ).reset_index()
            max_sharpe_allocation_df.columns = ["symbol", "allocation"]
            max_sharpe_allocation_df = max_sharpe_allocation_df.sort_values(
                "allocation", ascending=False
            )
            rank = []
            [
                rank.append(x)
                for x in range(1, len(max_sharpe_allocation_df["symbol"]) + 1)
            ]
            max_sharpe_allocation_df["rank"] = rank
            max_sharpe_allocation_df = max_sharpe_allocation_df.set_index("rank")
            max_sharpe_allocation_df = max_sharpe_allocation_df[
                max_sharpe_allocation_df["allocation"] != 0
            ]

            min_vol_allocation_df = pd.DataFrame(min_vol_allocation.T).reset_index()
            min_vol_allocation_df.columns = ["symbol", "allocation"]
            min_vol_allocation_df = min_vol_allocation_df.sort_values(
                "allocation", ascending=False
            )
            rank = []
            [rank.append(x) for x in range(1, len(min_vol_allocation_df["symbol"]) + 1)]
            min_vol_allocation_df["rank"] = rank
            min_vol_allocation_df = min_vol_allocation_df.set_index("rank")
            min_vol_allocation_df = min_vol_allocation_df[
                min_vol_allocation_df["allocation"] != 0
            ]

            def storage_a():
                st.subheader("** > Maximum Sharpe Ratio Portfolio Allocation**")
                st.write(
                    f" - Total Stocks Allocated: [{len(max_sharpe_allocation_df.symbol)} / {self.port_count}]"
                )
                st.write(f" - Annualised Return: {round(rp,2)}")
                st.write(f" - Annualised Volatility: {round(sdp,2)}")
                st.dataframe(max_sharpe_allocation_df)

                st.subheader("** > Minimum Volatility Portfolio Allocation**")
                st.write(
                    f" - Total Stocks Allocated: [{len(min_vol_allocation_df.symbol)} / {self.port_count}]"
                )
                st.write(f" - Annualised Return: {round(rp_min,2)}")
                st.write(f" - Annualised Volatility: {round(sdp_min,2)}")
                st.dataframe(min_vol_allocation_df)

            storage_a()

            fig, ax = plt.subplots()
            plt.scatter(
                results[0, :],
                results[1, :],
                c=results[2, :],
                cmap="YlGnBu",
                marker="o",
                s=10,
                alpha=0.3,
            )
            plt.colorbar()
            plt.scatter(
                sdp, rp, marker="*", color="r", s=500, label="Maximum Sharpe ratio"
            )
            plt.scatter(
                sdp_min,
                rp_min,
                marker="*",
                color="g",
                s=500,
                label="Minimum volatility",
            )
            plt.title(
                "Simulated · Efficient · Frontier · Portfolio · Optimization",
                fontsize=20,
                fontweight="bold",
            )
            plt.xlabel("annualised volatility", fontsize=20, fontweight="bold")
            plt.ylabel("annualised returns", fontsize=20, fontweight="bold")
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(15)
            ax.grid(True, color="k", linestyle="-", linewidth=1, alpha=0.3)
            ax.legend(loc="best", prop={"size": 16})
            plt.tight_layout()
            st.pyplot(fig)
            st.write("-" * 80)
            return (
                rp,
                sdp,
                rp_min,
                sdp_min,
                max_sharpe_allocation_df,
                min_vol_allocation_df,
            )

        def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
            p_var, p_ret = portfolio_annualised_performance(
                weights, mean_returns, cov_matrix
            )
            return -(p_ret - risk_free_rate) / p_var

        def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
            num_assets = len(mean_returns)
            args = (mean_returns, cov_matrix, risk_free_rate)
            constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
            bound = (0.0, 1.0)
            bounds = tuple(bound for asset in range(num_assets))
            result = sco.minimize(
                neg_sharpe_ratio,
                num_assets
                * [
                    1.0 / num_assets,
                ],
                args=args,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )
            return result

        def portfolio_volatility(weights, mean_returns, cov_matrix):
            return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[
                0
            ]

        def min_variance(mean_returns, cov_matrix):
            num_assets = len(mean_returns)
            args = (mean_returns, cov_matrix)
            constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
            bound = (0.0, 1.0)
            bounds = tuple(bound for asset in range(num_assets))
            result = sco.minimize(
                portfolio_volatility,
                num_assets
                * [
                    1.0 / num_assets,
                ],
                args=args,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )
            return result

        def efficient_return(mean_returns, cov_matrix, target):
            num_assets = len(mean_returns)
            args = (mean_returns, cov_matrix)

            def portfolio_return(weights):
                return portfolio_annualised_performance(
                    weights, mean_returns, cov_matrix
                )[1]

            constraints = (
                {"type": "eq", "fun": lambda x: portfolio_return(x) - target},
                {"type": "eq", "fun": lambda x: np.sum(x) - 1},
            )
            bounds = tuple((0, 1) for asset in range(num_assets))
            result = sco.minimize(
                portfolio_volatility,
                num_assets
                * [
                    1.0 / num_assets,
                ],
                args=args,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )
            return result

        def efficient_frontier(mean_returns, cov_matrix, returns_range):
            efficients = []
            for ret in returns_range:
                efficients.append(efficient_return(mean_returns, cov_matrix, ret))
            return efficients

        def display_calculated_ef_with_random(
            mean_returns, cov_matrix, num_portfolios, risk_free_rate
        ):
            results, _ = random_portfolios(
                num_portfolios, mean_returns, cov_matrix, risk_free_rate
            )
            max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
            sdp, rp = portfolio_annualised_performance(
                max_sharpe["x"], mean_returns, cov_matrix
            )
            max_sharpe_allocation = pd.DataFrame(
                max_sharpe.x, index=PT.columns, columns=["allocation"]
            )
            max_sharpe_allocation["allocation"] = [
                round(i * 100, 2) for i in max_sharpe_allocation.allocation
            ]
            max_sharpe_allocation = max_sharpe_allocation.T
            max_sharpe_allocation
            min_vol = min_variance(mean_returns, cov_matrix)
            sdp_min, rp_min = portfolio_annualised_performance(
                min_vol["x"], mean_returns, cov_matrix
            )
            min_vol_allocation = pd.DataFrame(
                min_vol.x, index=PT.columns, columns=["allocation"]
            )
            min_vol_allocation["allocation"] = [
                round(i * 100, 2) for i in min_vol_allocation.allocation
            ]
            min_vol_allocation = min_vol_allocation.T

            max_sharpe_allocation_df = pd.DataFrame(
                max_sharpe_allocation.T
            ).reset_index()
            max_sharpe_allocation_df.columns = ["symbol", "allocation"]
            max_sharpe_allocation_df = max_sharpe_allocation_df.sort_values(
                "allocation", ascending=False
            )
            rank = []
            [
                rank.append(x)
                for x in range(1, len(max_sharpe_allocation_df["symbol"]) + 1)
            ]
            max_sharpe_allocation_df["rank"] = rank
            max_sharpe_allocation_df = max_sharpe_allocation_df.set_index("rank")
            max_sharpe_allocation_df = max_sharpe_allocation_df[
                max_sharpe_allocation_df["allocation"] != 0
            ]

            min_vol_allocation_df = pd.DataFrame(min_vol_allocation.T).reset_index()
            min_vol_allocation_df.columns = ["symbol", "allocation"]
            min_vol_allocation_df = min_vol_allocation_df.sort_values(
                "allocation", ascending=False
            )
            rank = []
            [rank.append(x) for x in range(1, len(min_vol_allocation_df["symbol"]) + 1)]
            min_vol_allocation_df["rank"] = rank
            min_vol_allocation_df = min_vol_allocation_df.set_index("rank")
            min_vol_allocation_df = min_vol_allocation_df[
                min_vol_allocation_df["allocation"] != 0
            ]

            def storage_b():
                st.subheader("** > Maximum Sharpe Ratio Portfolio Allocation**")
                st.write(
                    f" - Total Stocks Allocated: [{len(max_sharpe_allocation_df.symbol)} / {self.port_count}]"
                )
                st.write(f"* Annualised Return: {round(rp,2)}")
                st.write(f"* Annualised Volatility: {round(sdp,2)}")
                st.dataframe(max_sharpe_allocation_df)

                st.subheader("Minimum Volatility Portfolio Allocation\n")
                st.write(
                    f" - Total Stocks Allocated: [{len(min_vol_allocation_df.symbol)} / {self.port_count}]"
                )
                st.write(f"* Annualised Return: {round(rp_min,2)}")
                st.write(f"* Annualised Volatility: {round(sdp_min,2)}")
                st.dataframe(min_vol_allocation_df)

            storage_b()

            fig, ax = plt.subplots()
            plt.scatter(
                results[0, :],
                results[1, :],
                c=results[2, :],
                cmap="YlGnBu",
                marker="o",
                s=10,
                alpha=0.3,
            )
            plt.colorbar()
            plt.scatter(
                sdp, rp, marker="*", color="r", s=500, label="Maximum Sharpe ratio"
            )
            plt.scatter(
                sdp_min,
                rp_min,
                marker="*",
                color="g",
                s=500,
                label="Minimum volatility",
            )
            target = np.linspace(rp_min, 0.32, 50)
            efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, target)
            plt.plot(
                [p["fun"] for p in efficient_portfolios],
                target,
                linestyle="-.",
                color="black",
                label="efficient frontier",
            )
            plt.title(
                "Calculated Portfolio Optimization based on Efficient Frontier",
                fontsize=30,
                fontweight="bold",
            )
            plt.xlabel("annualised volatility", fontsize=20, fontweight="bold")
            plt.ylabel("annualised returns", fontsize=20, fontweight="bold")
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(15)
            ax.grid(True, color="k", linestyle="-", linewidth=1, alpha=0.3)
            ax.legend(loc="best", prop={"size": 16})
            plt.tight_layout()
            st.pyplot(fig)
            st.write("-" * 80)
            return (
                rp,
                sdp,
                rp_min,
                sdp_min,
                max_sharpe_allocation_df,
                min_vol_allocation_df,
            )

        def display_ef_with_selected(mean_returns, cov_matrix, risk_free_rate):
            max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
            sdp, rp = portfolio_annualised_performance(
                max_sharpe["x"], mean_returns, cov_matrix
            )
            max_sharpe_allocation = pd.DataFrame(
                max_sharpe.x, index=PT.columns, columns=["allocation"]
            )
            max_sharpe_allocation["allocation"] = [
                round(i * 100, 2) for i in max_sharpe_allocation.allocation
            ]
            max_sharpe_allocation = max_sharpe_allocation.T
            max_sharpe_allocation
            min_vol = min_variance(mean_returns, cov_matrix)
            sdp_min, rp_min = portfolio_annualised_performance(
                min_vol["x"], mean_returns, cov_matrix
            )
            min_vol_allocation = pd.DataFrame(
                min_vol.x, index=PT.columns, columns=["allocation"]
            )
            min_vol_allocation["allocation"] = [
                round(i * 100, 2) for i in min_vol_allocation.allocation
            ]
            min_vol_allocation = min_vol_allocation.T
            an_vol = np.std(PT.pct_change()) * np.sqrt(252)
            an_rt = mean_returns * 252

            max_sharpe_allocation_df = pd.DataFrame(
                max_sharpe_allocation.T
            ).reset_index()
            max_sharpe_allocation_df.columns = ["symbol", "allocation"]
            max_sharpe_allocation_df = max_sharpe_allocation_df.sort_values(
                "allocation", ascending=False
            )
            rank = []
            [
                rank.append(x)
                for x in range(1, len(max_sharpe_allocation_df["symbol"]) + 1)
            ]
            max_sharpe_allocation_df["rank"] = rank
            max_sharpe_allocation_df = max_sharpe_allocation_df.set_index("rank")
            max_sharpe_allocation_df = max_sharpe_allocation_df[
                max_sharpe_allocation_df["allocation"] != 0
            ]

            min_vol_allocation_df = pd.DataFrame(min_vol_allocation.T).reset_index()
            min_vol_allocation_df.columns = ["symbol", "allocation"]
            min_vol_allocation_df = min_vol_allocation_df.sort_values(
                "allocation", ascending=False
            )
            rank = []
            [rank.append(x) for x in range(1, len(min_vol_allocation_df["symbol"]) + 1)]
            min_vol_allocation_df["rank"] = rank
            min_vol_allocation_df = min_vol_allocation_df.set_index("rank")
            min_vol_allocation_df = min_vol_allocation_df[
                min_vol_allocation_df["allocation"] != 0
            ]

            def storage_c():
                st.subheader("Maximum Sharpe Ratio Portfolio Allocation\n")
                st.write(
                    f" - Total Stocks Allocated: [{len(max_sharpe_allocation_df.symbol)} / {self.port_count}]"
                )
                st.write(f"* Annualised Return: {round(rp,2)}")
                st.write(f"* Annualised Volatility: {round(sdp,2)}")
                st.dataframe(max_sharpe_allocation_df)

                st.subheader("Minimum Volatility Portfolio Allocation\n")
                st.write(
                    f" - Total Stocks Allocated: [{len(min_vol_allocation_df.symbol)} / {self.port_count}]"
                )
                st.write(f"* Annualised Return: {round(rp_min,2)}")
                st.write(f"* Annualised Volatility: {round(sdp_min,2)}")
                st.dataframe(min_vol_allocation_df)

            storage_c()

            st.header("Individual Stock Returns and Volatility\n")
            for i, txt in enumerate(PT.columns):
                st.text(
                    f"{txt}: annuaised return {round(an_rt[i],2)} - annualised volatility: {round(an_vol[i],2)}"
                )
            st.write("-" * 80)

            fig, ax = plt.subplots()
            ax.scatter(an_vol, an_rt, marker="o", s=200)
            for i, txt in enumerate(PT.columns):
                ax.annotate(
                    txt,
                    (an_vol[i], an_rt[i]),
                    xytext=(10, 0),
                    textcoords="offset points",
                )
            ax.scatter(
                sdp, rp, marker="*", color="r", s=500, label="Maximum Sharpe ratio"
            )
            ax.scatter(
                sdp_min,
                rp_min,
                marker="*",
                color="g",
                s=500,
                label="Minimum volatility",
            )
            target = np.linspace(rp_min, 0.34, 50)
            efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, target)
            ax.plot(
                [p["fun"] for p in efficient_portfolios],
                target,
                linestyle="-.",
                color="black",
                label="efficient frontier",
            )
            ax.set_title(
                "Portfolio Optimization with Individual Stocks",
                fontsize=30,
                fontweight="bold",
            )
            ax.set_xlabel("annualised volatility", fontsize=20, fontweight="bold")
            ax.set_ylabel("annualised returns", fontsize=20, fontweight="bold")
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(15)
            ax.grid(True, color="k", linestyle="-", linewidth=1, alpha=0.3)
            ax.legend(loc="best", prop={"size": 16})
            plt.tight_layout()
            st.pyplot(fig)
            return (
                rp,
                sdp,
                rp_min,
                sdp_min,
                max_sharpe_allocation_df,
                min_vol_allocation_df,
            )

        st.header("**[Method · 1]**")
        st.write(f"- Total Stocks In Ticker List: {len(self.port_tics)}")
        st.write(
            " - Simulated Optimal Efficient Frontier Using A Random Number Of Portfolios & Random Position Weights"
        )
        (
            rpA,
            sdpA,
            rp_minA,
            sdp_minA,
            max_sharpe_df_1,
            min_vol_df_1,
        ) = display_simulated_ef_with_random(
            mean_returns, cov_matrix, num_portfolios, risk_free_rate
        )

        st.header("**[Method · 2]**")
        st.write(
            "Calculated Optimal Efficient Frontier A Random Number Of Portfolios & Position Weights"
        )
        (
            rpB,
            sdpB,
            rp_minB,
            sdp_minB,
            max_sharpe_df_2,
            min_vol_df_2,
        ) = display_calculated_ef_with_random(
            mean_returns, cov_matrix, num_portfolios, risk_free_rate
        )

        st.header("**[Method · 3]**")
        st.write("Calculated Efficient Frontier With Selected Position Weights")
        (
            rp,
            sdp,
            rp_min,
            sdp_min,
            max_sharpe_df_3,
            min_vol_df_3,
        ) = display_ef_with_selected(mean_returns, cov_matrix, risk_free_rate)

        saver_lst = [
            max_sharpe_df_1,
            min_vol_df_1,
            max_sharpe_df_2,
            min_vol_df_2,
            max_sharpe_df_3,
            min_vol_df_3,
        ]
        namer_lst = [
            "max_sharpe_df_1",
            "min_vol_df_1",
            "max_sharpe_df_2",
            "min_vol_df_2",
            "max_sharpe_df_3",
            "min_vol_df_3",
        ]

        save_loc = Path(f"reports/portfolio/{today_date}/")
        if not save_loc.exists():
            save_loc.mkdir(parents=True)

        for r in range(len(saver_lst)):
            fd = pd.DataFrame(saver_lst[r])
            fd.to_pickle(save_loc / f"{namer_lst[r]}.pkl")

        lst_of_returns = [
            [rpA, sdpA, rp_minA, sdp_minA, max_sharpe_df_1, min_vol_df_1],
            [rpB, sdpB, rp_minB, sdp_minB, max_sharpe_df_2, min_vol_df_2],
            [rp, sdp, rp_min, sdp_min, max_sharpe_df_3, min_vol_df_3],
        ]

        return lst_of_returns


# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


if __name__ == "__main__":
    # my_positions = pd.read_pickle("/home/gordon/gdp/project_active/Forecasting_For_Friends/tickers/jayci_ticker_lst.pkl")

    my_positions = "INSE OPCH GDEN HCCI DAR RWT TSQ LQDT AGO ONTO GLDD III WOW TGLS ORMP UTHR DAC EPRT EVC OLN ASIX AJX INMD SBLK IIPR PMBC RBNC SSTK CLH CROX MT AMRC STLD PACK ATLC CTO OMF ASX PXLW SI CUBI NET AMN CBNK BLFS SKY BW ALLY COF DECK STXB CSV AX SCU ALGN DVAX AXON TGH ELF GENI NVEE ZNTL SHYF GTYH ESEA TPX TX NSA CAMT AVNW JLL RCII EVR LPLA FUNC SNX KE HBIO AMSWA THC DKS ULTA AVNT HLIT UFPT ACLS OMCL DCOM CZWI MODV BOOT TRNS BYD MORF SONO CDXS TLYS JYNT SE NTLA ASML STAA RMNI WCC FCNCA FC MCB CRAI DOMO BEAM BNTX DFS CPRI EBSB ATKR CCS RH KSS CIGI CRL ABCB TRGP AFG SNAP CIDM VRTS TTEC KLIC STLA AMKR CBAN GPI CODI AMP CRNT CLF CSTM CNOB PNFP NAVI CNTY BLDR CBRE SYF ON AMG VRT TTGT MIDD GCO SNV PGC UHAL PACW GIL JCI VMI SEAS EPAM SRRA COOP FCCY JBL PPBI CMTL CNHI SCVL HWC FBIZ RJF ACM MXL FBNC ECC HTGC LPX BWFG LOB BRBR SCHW YETI BXC TECH QCRH CARR KKR EWBC FISI KEYS MHK NXST TEX CIVB AGCO FND ISBC HCA INTU CVCY FBMS PVBC MYFW RDN RPD MAT LKQ UAA MS PAG PW MAN ACHC SIMO BSIG RBB WSC EXP CDW ATEN SPG MITK ARES FFWM HBNC VCRA DIOD FFIC HCKT MSI FFNW TPB LSI JHX TRMB INBX WIRE LH DHR XPO OCSL ZBRA CSTR ZI MCHX SXI AMTB TXT LCUT VCTR FIVE CG GIC WNEB IQV KIM BWB DHX POWI BSY PIPR DGII BAM ATCO BCO MET GFED GMS HEES MTG ACRE HTBK NVDA AVY NDLS NVO DE ORI CSL HBMD R INOV AMX CR HWKN ALTR NVT UMPQ MRVL EXR MTSI GRBK UNTY FCBC BLK KRNY ETN CATC FMBH PFC RMBS EL BY HPE BFIN HBAN MOS GEF ORRF TCPC ARD BFST BANC LLY CC GRMN NFBK SRC SQ ISTR RBCAA NXPI BANR CUBE CHMG STRL TEL BFS USB SAFE LFUS OZK DOV MANH AYI FLOW AUB BRKL KLAC TTD BKU NKSH EVBN RMR GBX PBH IDXX BDC EFX FFIV"

    The_Portfolio_Optimizer(my_positions.split()).optimize()
