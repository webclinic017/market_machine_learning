from os import close
from PIL.Image import ROTATE_90
import yfinance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from datetime import datetime, date
from yahooquery import Ticker
import yfinance as yf
import streamlit as st
from tabulate import tabulate

plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = [15, 10]
plt.rc("font", size=14)
np.random.seed(0)


def company_longName(symbol):
    d = Ticker(symbol).quote_type
    return list(d.values())[0]["longName"]



class Optimal_SMA(object):

    def __init__(self, ticker, end_date):
        self.name = ticker
        self.n_forward = 1
        self.end_date = str(end_date)[:10]
        self.company_longName = company_longName(self.name)


    def build_optimal_sma(self, data, graphit=True, cc=0.0, ccc=0.0):
        # data = Ticker(self.name).history(period='2y').reset_index()
        # del data['symbol']
        # data = pd.DataFrame(data.set_index('date').copy())
        # self.data = pd.DataFrame(data[pd.to_datetime(data.index) < pd.to_datetime(self.end_date)])

        self.data = pd.DataFrame(data)
        self.data["Forward Close"] = self.data["adjclose"].shift(-self.n_forward)
        self.data["Forward Return"] = (self.data["Forward Close"] - self.data["adjclose"]) / self.data["adjclose"]
        result = []
        train_size = 0.80

        for sma_length in range(2, 100):
            self.data["SMA"] = self.data["adjclose"].rolling(sma_length).mean()
            self.data["input"] = [int(x) for x in self.data["adjclose"] > self.data["SMA"]]
            df = pd.DataFrame(self.data.copy())

            training = df.head(int(train_size * df.shape[0]))
            test = df.tail(int((1 - train_size) * df.shape[0]))
            tr_returns = training[training["input"] == 1]["Forward Return"]
            test_returns = test[test["input"] == 1]["Forward Return"]

            mean_forward_return_training = tr_returns.mean()
            mean_forward_return_test = test_returns.mean()
            pvalue = ttest_ind(tr_returns, test_returns, equal_var=False)[1]
            result.append(
                {
                    "sma_length": sma_length,
                    "training_forward_return": mean_forward_return_training,
                    "test_forward_return": mean_forward_return_test,
                    "p-value": pvalue,
                }
            )
            
        result.sort(key=lambda x: -x["training_forward_return"])
        # fd = pd.DataFrame(result).set_index("sma_length")
        best_sma = SMA_window = result[0]["sma_length"]
        SMA_window_col = str(SMA_window)

        # Create a short simple moving average column
        self.data[SMA_window_col] = (self.data["adjclose"].rolling(window=SMA_window, min_periods=1).mean())
        self.data["Signal"] = 0.0
        self.data["Signal"] = np.where(self.data[SMA_window_col] <= self.data['adjclose'], 1.0, 0.0)

        # create a new column 'Position' which is a day-to-day difference of the 'Signal' column.
        self.data["Position"] = self.data["Signal"].diff()

        if graphit is True:
            # self.data = self.data[pd.to_datetime(self.data.index) < pd.to_datetime('2021')]
            fig, ax = plt.subplots()
            
            plt.plot(self.data["adjclose"], label=self.company_longName)
            plt.plot(self.data[SMA_window_col], label="SMA-{}".format(best_sma))
            # plot 'buy' signals
            plt.plot(
                self.data.loc[pd.Timestamp("2021"):][self.data["Position"] == 1].index,
                self.data.loc[pd.Timestamp("2021"):][SMA_window_col][self.data["Position"] == 1],
                "^",
                markersize=15,
                color="g",
                alpha=0.7,
                label="buy",
            )
            # plot 'sell' signals
            plt.plot(
                self.data.loc[pd.Timestamp("2021"):][self.data["Position"] == -1].index,
                self.data.loc[pd.Timestamp("2021"):][SMA_window_col][self.data["Position"] == -1],
                "v",
                markersize=15,
                color="r",
                alpha=0.7,
                label="sell",
            )
            plt.ylabel("Price in $", fontsize=20, fontweight="bold")
            plt.xlabel("Date", fontsize=20, fontweight="bold")
            plt.title(f"{self.name} - {str(SMA_window)} Crossover", fontsize=30, fontweight="bold", )        
            plt.xlabel("Date", fontsize=20, fontweight="bold")
            plt.ylabel("Price", fontsize=20, fontweight="bold")
            plt.title(f"{self.company_longName} ({self.name}) - SMA", fontsize=30, fontweight="bold", )
            plt.xlim((date(2021, 1, 1)), (date(2021, 12, 30)))
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(15)
            ax.grid(True, color="k", linestyle="-", linewidth=1, alpha=0.3)
            plt.xticks(rotation=45)
            plt.yticks(rotation=90)
            ax.legend(loc="best", prop={"size": 16})
            plt.tight_layout()
            st.pyplot(fig)


        df_pos = self.data[(self.data["Position"] == 1) | (self.data["Position"] == -1)]
        action_lst = []
        for x in df_pos['Position']:
            if x == 1:
                action_lst.append("Buy")
            else:
                action_lst.append('Sell')
        df_pos["Action"] = action_lst
        
        # try:
        #     del df_pos['open']
        #     del df_pos['high']
        #     del df_pos['low']
        #     del df_pos['close']
        #     del df_pos['splits']
        #     del df_pos['dividends']
        #     del df_pos['input']
        #     del df_pos['SMA']
        #     del df_pos['Signal']
        #     del df_pos['Position']            
        #     # st.text(tabulate(df_pos.loc["2021":], headers="keys", tablefmt="psql"))
        #     st.text(tabulate(df_pos.iloc[-5:], headers="keys", tablefmt="psql"))
        # except:
        #     pass


        if df_pos['Action'][-1] == 'Buy':
            st.metric(f"[{cc}/{ccc}]", f"{self.name}", f"{df_pos['Position'][-1]}")
            return self.name

        elif df_pos['Action'][-1] == 'Sell':
            st.metric(f"[{cc}/{ccc}]", f"{self.name}", f"- {df_pos['Position'][-1]}")


if __name__ == '__main__':
    Optimal_SMA('AAPL', '2021-10-25').build_optimal_sma()