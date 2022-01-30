import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

plt.style.use("seaborn-poster")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["figure.figsize"] = (15, 6.5)
plt.rcParams["axes.grid"] = True
from pathlib import Path
from datetime import datetime
import streamlit as st

st.set_option("deprecation.showPyplotGlobalUse", False)
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


class The_Vectorized_Backtest(object):
    def __init__(
        self,
        stocks,
        saver,
        bulk=" ~ S&P500 - DOW - NASDAQ - NYSE - RUSSELL ~ ",
        period="1y",
    ):
        self.yf_prices = yf.download(stocks, period=period)
        self.prices = self.yf_prices["Adj Close"]
        self.rs = self.prices.apply(np.log).diff(1).fillna(0)
        self.string = ""
        for s in stocks:
            self.string += s + "_"
        self.bulk_name = bulk
        self.saver = saver

    def one(self):
        p1 = self.prices.iloc[0]
        rs1 = p1.apply(np.log).diff(1).fillna(0)
        w1 = 2
        w2 = 20
        ma_x = p1.rolling(w1).mean() - p1.rolling(w2).mean()
        ma_x1 = p1.rolling(w1).mean()
        ma_x2 = p1.rolling(w2).mean()
        pos = ma_x.apply(np.sign)

        fig, ax = plt.subplots()
        plt.grid(
            b=True,
            which="both",
            axis="both",
            color="k",
            linestyle="dashed",
            linewidth=1,
            alpha=0.2,
        )
        p1.plot(title="Moving Average Indicators (1.1)", label="sp500_price")
        ma_x1.plot(label="sp500_mAvg_S")
        ma_x2.plot(label="sp500_mAvg_L")
        plt.ylabel("S&P500 Index Price")
        plt.tight_layout()
        plt.legend()
        st.pyplot(fig)

        fig, ax = plt.subplots(2, 1)
        ma_x.plot(
            ax=ax[0],
            title="S&P500 Index - Moving Average Cross-Over (1.2)",
            label="sp500_mAvg_cross",
        )
        ax[0].set_ylabel("Portfolio Value")
        pos.plot(ax=ax[1], title="Position Plot (Rise = Buy ~ Dip = Sell)")
        ax[1].set_ylabel("position/ lot size taken in action")
        plt.grid(
            b=True,
            which="both",
            axis="both",
            color="k",
            linestyle="dashdot",
            linewidth=1,
            alpha=0.3,
        )
        plt.tight_layout()
        plt.legend()
        st.pyplot(fig)

        my_rs = pos.shift(1) * rs1
        my_rs.cumsum().apply(np.exp).plot(
            title="S&P500 Index - Strategy Performance VS BENCH (1.3)"
        )
        rs1.cumsum().apply(np.exp).plot()
        plt.ylabel("Total Returns ")
        plt.grid(
            b=True,
            which="both",
            axis="both",
            color="k",
            linestyle="dashed",
            linewidth=1,
            alpha=0.2,
        )
        plt.legend(["my performance", "benchmark performance"])
        plt.tight_layout()
        st.pyplot(fig)


    def two(self):
        self.one()
        w1 = 5
        w2 = 22
        ma_x = self.prices.rolling(w1).mean() - self.prices.rolling(w2).mean()
        self.pos = ma_x.apply(np.sign)
        self.pos /= self.pos.abs().sum(1).values.reshape(-1, 1)

        fig, ax = plt.subplots(2, 1)
        ma_x.plot(
            ax=ax[0], title=f"{self.bulk_name} Moving Average Cross-Overs (2.1)", lw=1.5
        )
        ax[0].set_ylabel("Portfolio Value")
        ax[0].legend(bbox_to_anchor=(1.1, 1.05))
        plt.grid(
            b=True,
            which="both",
            axis="both",
            color="k",
            linestyle="dashed",
            linewidth=1,
            alpha=0.2,
        )
        self.pos.plot(ax=ax[1], title="Positions", lw=1.5)
        ax[1].set_ylabel("position/ lot size taken in action")
        ax[1].legend(bbox_to_anchor=(1.1, 1.05))
        plt.grid(
            b=True,
            which="both",
            axis="both",
            color="k",
            linestyle="dashed",
            linewidth=1,
            alpha=0.2,
        )
        plt.tight_layout()
        st.pyplot(fig)

        fig, ax = plt.subplots()
        my_rs = self.pos.shift(1) * self.rs
        my_rs.cumsum().apply(np.exp).plot()
        plt.title(f"{self.bulk_name} - MY Performance VS BENCH (2.2)")
        plt.ylabel("Returns")
        plt.grid(
            b=True,
            which="both",
            axis="both",
            color="k",
            linestyle="dashed",
            linewidth=1,
            alpha=0.2,
        )
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig)

        fig, ax = plt.subplots()
        my_rs = (self.pos.shift(1) * self.rs).sum(1)
        my_rs.cumsum().apply(np.exp).plot(
            title=f"{self.bulk_name} - Strategy Performance vs BENCH (2.3)"
        )
        self.rs.mean(1).cumsum().apply(np.exp).plot()
        plt.ylabel("Returns")
        plt.legend(["my performance", "benchmark performance"])
        plt.grid(
            b=True,
            which="both",
            axis="both",
            color="k",
            linestyle="dashed",
            linewidth=1,
            alpha=0.2,
        )
        plt.tight_layout()
        st.pyplot(fig)

        fig, ax = plt.subplots()
        my_rs1 = (self.pos * self.rs).sum(1)
        my_rs1.cumsum().apply(np.exp).plot(
            title=f"{self.bulk_name} - Look-Ahead Bias Performance (2.4)",
            label="With Look-Ahead Bias",
        )
        my_rs2 = (self.pos.shift(1) * self.rs).sum(1)
        my_rs2.cumsum().apply(np.exp).plot(label="Without Look-Ahead Bias")
        plt.ylabel("Returns")
        plt.legend(["With Look-Ahead Bias", "Without Look-Ahead Bias"])
        plt.grid(
            b=True,
            which="both",
            axis="both",
            color="k",
            linestyle="dashed",
            linewidth=1,
            alpha=0.2,
        )
        plt.tight_layout()
        st.pyplot(fig)


    def three(self):
        self.two()
        self.rs = self.prices.apply(np.log).diff(1).fillna(0)
        lags = range(1, 11)
        lagged_rs = pd.Series(dtype=float, index=lags)
        fig, ax = plt.subplots()
        for lag in lags:
            my_rs = (self.pos.shift(lag) * self.rs).sum(1)
            my_rs.cumsum().apply(np.exp).plot()
            lagged_rs[lag] = my_rs.sum()
        plt.ylabel("Returns")
        plt.title(f"{self.bulk_name} - Strategy Performance with Lags (3.1)")
        plt.legend(lags, bbox_to_anchor=(1.1, 0.95))
        plt.grid(
            b=True,
            which="both",
            axis="both",
            color="k",
            linestyle="dashed",
            linewidth=1,
            alpha=0.2,
        )
        plt.tight_layout()
        st.pyplot(fig)


    def four(self):
        self.three()
        tc_pct = 0.01
        fig, ax = plt.subplots()
        delta_pos = self.pos.diff(1).abs().sum(1)
        my_tcs = tc_pct * delta_pos
        my_rs1 = (self.pos.shift(1) * self.rs).sum(1)
        my_rs2 = (self.pos.shift(1) * self.rs).sum(1) - my_tcs
        my_rs1.cumsum().apply(np.exp).plot()
        my_rs2.cumsum().apply(np.exp).plot()
        plt.ylabel("Returns")
        plt.legend(["without transaction costs", "with transaction costs"])
        plt.title("4.1")
        plt.grid(
            b=True,
            which="both",
            axis="both",
            color="k",
            linestyle="dashed",
            linewidth=1,
            alpha=0.2,
        )
        plt.title("Strategy comparision  ~ With & Without Transaction Costs: (4.1)")
        plt.tight_layout()
        st.pyplot(fig)
        return


if __name__ == "__main__":
    # The_Vectorized_Backtest(
    #     stocks=['^GSPC', '^DJI', '^IXIC','^NYA','^RUT'],
    #     saver='indexGroup_',
    #     bulk=' ~ S&P500 - DOW - NASDAQ - NYSE - RUSSELL ~ ',
    #     period='1y'
    #     ).four()

    # from pathlib import Path
    # from datetime import datetime
    # today = str(datetime.now())[:10]
    # saveTickers = Path('files/tickers/')

    # my_positions = pd.read_pickle(saveTickers / f'chuck_merged_ticker_lst.pkl')
    # watch_lst = pd.read_pickle(saveTickers / f'watch_merged_ticker_lst.pkl')
    # abc = my_positions + watch_lst
    # my_positions_lst_long = [list(x) for x in zip(abc, abc)]
    # my_positions_lst_short = [list(x) for x in zip(my_positions, my_positions)]

    The_Vectorized_Backtest(
        stocks=["TSLA", "AAPL", "MSFT"],
        saver="chuck_merged_",
        bulk=" ~ my holdings - chuck ~ ",
        period="1y",
    ).four()
