import streamlit as st
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import yfinance as yf
from yahooquery import Ticker
from sklearn.decomposition import PCA
from pathlib import Path
from os.path import exists
import pickle5 as pickle
from datetime import datetime
path = Path.cwd()

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



class The_PCA_Analysis(object):
    
    
    def __init__(self, tickers, report_date, save_final=True, x_factor=0.13):
        self.tickers = tickers
        self.report_date = report_date
        self.save_final = save_final
        self.x_factor = int(round(float(len(self.tickers) * x_factor)))

        self.saveAdvisor = Path(f"data/advisor/pca/{str(self.report_date)[:7]}/{self.report_date}/")
        if not self.saveAdvisor.exists():
            self.saveAdvisor.mkdir(parents=True)
            
        self.savePCApic = Path(f"data/images/pca/{str(self.report_date)[:7]}/{str(self.report_date)[:10]}/")
        if not self.savePCApic.exists():
            self.savePCApic.mkdir(parents=True)


    def build_pca(self, data, graph_it=True):
        self.graph_it = graph_it
        self.prices = pd.DataFrame(data)
        self.rs = self.prices.apply(np.log).diff(1)     

        # if self.graph_it:
        #     fig, ax = plt.subplots()
        #     ax = self.rs.plot(legend=0,figsize=(10, 6),grid=True,title=f"Daily Returns",)
        #     plt.tight_layout()
        #     st.pyplot()

        if self.graph_it:
            fig, ax = plt.subplots()
            (self.rs.cumsum().apply(np.exp)).plot(legend=0,figsize=(10, 6),grid=True,title=f"Cumulative Returns",)
            plt.tight_layout()
            st.pyplot()

        pca = PCA(1).fit(self.rs.fillna(0.0))
        pc1 = pd.Series(index=self.rs.columns, data=pca.components_[0])

        # fig, ax = plt.subplots()
        # pc1.plot(
        #     figsize=(10, 6),
        #     xticks=[],
        #     grid=True,
        #     title=f"First Principal Component",
        # )
        # plt.tight_layout()
        # st.pyplot()

        weights = abs(pc1) / sum(abs(pc1))
        myrs = (weights * self.rs).sum(1)

        # if self.graph_it:
            # fig, ax = plt.subplots()
            # myrs.cumsum().apply(np.exp).plot(figsize=(10, 6),grid=True,title=f"Cumulative Daily Returns of 1st Principal Component Stock",)
            # st.pyplot(fig)

        prices = yf.download(["^GSPC"], start="2021-01-04")["Adj Close"]
        rs_df = pd.concat([myrs, prices.apply(np.log).diff(1)], 1)
        rs_df.columns = ["PCA Portfolio", "SP500_Index"]

        if self.graph_it:
            fig, ax = plt.subplots()
            rs_df.dropna().cumsum().apply(np.exp).plot(subplots=True, figsize=(10, 6), grid=True, linewidth=3)
            plt.tight_layout()
            st.pyplot()

        fig, ax = plt.subplots(2, 1, figsize=(10, 6))
        pc1.nlargest(10).plot.bar(ax=ax[0],color="blue",grid=True,title="Stocks with Highest PCA Score (-OR- Least Negative) PCA Weights",)
        pc1.nsmallest(10).plot.bar(ax=ax[1],color="green",grid=True,title="Stocks with Lowest PCA Score (-OR- Most Negative) PCA Weights",)
        plt.tight_layout()
        st.pyplot(fig)

        ws = [-1,] * 10 + [1,] * 10
        myrs = self.rs[pc1.nlargest(self.x_factor).index].mean(1)

        fig, ax = plt.subplots()
        myrs1 = myrs.cumsum().apply(np.exp)
        myrs1.plot(figsize=(15, 5),grid=True,linewidth=3,title=f"PCA Portfolio ({self.x_factor} Most Impactful) vs S&P500 Index",)
        prices["2020":].apply(np.log).diff(1).cumsum().apply(np.exp).plot(figsize=(10, 6), grid=True, linewidth=3)
        plt.legend(["PCA Selection", "SP500_Index"])
        plt.tight_layout()
        st.pyplot(fig)
        
        fig, ax = plt.subplots()
        ws = [-1,] * 10 + [1,] * 10
        myrs = self.rs[pc1.nsmallest(self.x_factor).index].mean(1)
        myrs2 = myrs.cumsum().apply(np.exp)
        myrs2.plot(figsize=(15, 5),grid=True,linewidth=3,title=f"PCA Portfolio ({self.x_factor} Least Impactful) vs S&P500 Index",)
        prices["2020":].apply(np.log).diff(1).cumsum().apply(np.exp).plot(figsize=(10, 6), grid=True, linewidth=3)
        plt.legend(["PCA Selection", "SP500_Index"])
        plt.tight_layout()
        st.pyplot(fig)

        largest_ret = myrs1.iloc[-1]
        smallest_ret = myrs2.iloc[-1]
        if len(self.tickers) > 10:
            ws = [-1,] * 5 + [1,] * 5
            myrs = (self.rs[list(pc1.nsmallest(5).index) + list(pc1.nlargest(5).index)] * ws).mean(1)

            # fig, ax = plt.subplots()
            # myrs.cumsum().apply(np.exp).plot(
            #     figsize=(15, 5),
            #     grid=True,
            #     linewidth=3,
            #     title=f"PCA Portfolio (5 Most & 5 Least Impactful) vs The Round 5 Stocks",
            # )
            # prices["2020":].apply(np.log).diff(1).cumsum().apply(np.exp).plot(
            #     figsize=(10, 6), grid=True, linewidth=3
            # )
            # plt.legend(["PCA Selection", "SP500_Index"])
            # plt.tight_layout()
            # st.pyplot(fig)

            st.subheader(f"Below Are The Principal Components From The Ticker List:")
            st.write(f"- LARGEST PCA VALUES == [{round(largest_ret,2)}]")
            st.write(f"- SMALLEST PCA VALUES == [{round(smallest_ret,2)}]")
            if largest_ret > smallest_ret:
                return self.rs[pc1.nlargest(self.x_factor).index]
            else:
                return self.rs[pc1.nsmallest(self.x_factor).index]
            
        else:
            st.subheader(f"Below Are The Principal Components From The Ticker List:")
            st.write(f"- LARGEST PCA VALUES == [{round(largest_ret,2)}]")
            st.write(f"- SMALLEST PCA VALUES == [{round(smallest_ret,2)}]")
            if largest_ret > smallest_ret:
                return self.rs[pc1.nlargest(self.x_factor).index]
            else:
                return self.rs[pc1.nsmallest(self.x_factor).index]
