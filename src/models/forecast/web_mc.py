import warnings
warnings.filterwarnings("ignore")
from yfinance import ticker
from yahoo_fin import stock_info as si
import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
from scipy.stats import norm, gmean, cauchy
import seaborn as sns
from datetime import datetime
import yfinance as yf
import streamlit as st
from finvizfinance.quote import finvizfinance

from src.tools.functions import company_longName


class MC_Forecast(object):


    def __init__(self):
        pass


    def import_stock_data(self, tickers, start="2010-1-1", end=datetime.today().strftime("%Y-%m-%d")):
        data = yf.download(tickers, start=start, end=end)["Adj Close"]
        return data


    def log_returns(self, data):
        return np.log(1 + data.pct_change())


    def simple_returns(self, data):
        return (data / data.shift(1)) - 1


    def market_data_combination(self, tickers, data, mark_ticker, start="2010-1-1"):
        market_data = self.import_stock_data(mark_ticker, start)
        market_rets = self.log_returns(market_data).dropna()
        ann_return = np.exp(market_rets.mean() * 252)
        data = pd.DataFrame(data.merge(market_data, left_index=True, right_index=True))

        temp_lst = []
        [temp_lst.append(i) for i in tickers]
        temp_lst.append("^GSPC")
        data.columns = temp_lst
        return data, ann_return


    def beta_sharpe(self, tickers, data, mark_ticker="^GSPC", start="2010-1-1", riskfree=0.025):
        """
        > Input:
        >> 1. data: dataframe of stock price data
        >> 2. mark_ticker: ticker of the market data you want to compute CAPM metrics with (default is ^GSPC)
        >> 3. start: data from which to download data (default Jan 1st 2010)
        >> 4. riskfree: the assumed risk free yield (US 10 Year Bond is assumed: 2.5%)

        > Output:
        >> 1. Dataframe with CAPM metrics computed against specified market procy
        """

        # Beta
        dd, mark_ret = self.market_data_combination(tickers, data, mark_ticker, start)
        log_ret = self.log_returns(dd)
        covar = log_ret.cov() * 252
        covar = pd.DataFrame(covar.iloc[:-1, -1])
        mrk_var = log_ret.iloc[:, -1].var() * 252
        beta = covar / mrk_var
        stdev_ret = pd.DataFrame(((log_ret.std() * 250 ** 0.5)[:-1]), columns=["STD"])
        beta = beta.merge(stdev_ret, left_index=True, right_index=True)
        
        # CAPM
        for i, row in beta.iterrows():
            beta.at[i, "CAPM"] = riskfree + (row[mark_ticker] * (mark_ret - riskfree))

        # Sharpe
        for i, row in beta.iterrows():
            beta.at[i, "Sharpe"] = (row["CAPM"] - riskfree) / (row["STD"])

        beta.rename(columns={"^GSPC": "Beta"}, inplace=True)
        return beta


    def drift_calc(self, data, return_type="log"):
        if return_type == "log":
            lr = self.log_returns(data)

        elif return_type == "simple":
            lr = self.simple_returns(data)

        u = lr.mean()
        var = lr.var()
        drift = u - (0.5 * var)

        try:
            return drift.values

        except:
            return drift


    def daily_returns(self, data, days, iterations, return_type="log"):
        """
        > Oftentimes, we find that the distribution of returns is a variation 
           of the normal distribution where it has a fat tail
        > This distribution is called cauchy distribution
        """

        ft = self.drift_calc(data, return_type)

        if return_type == "log":
            try:
                stv = self.log_returns(data).std().values
            except:
                stv = self.log_returns(data).std()

        elif return_type == "simple":
            try:
                stv = self.simple_returns(data).std().values
            except:
                stv = self.simple_returns(data).std()

        dr = np.exp(ft + stv * norm.ppf(np.random.rand(days, iterations)))
        return dr


    def probs_find(self, predicted, higherthan, on="value"):
        """
        > This function calculated the probability of a stock being above a certain threshhold, which can be defined as a value (final stock price) or return rate (percentage change)
        > Input:
        >> 1. predicted: dataframe with all the predicted prices (days and simulations)
        >> 2. higherthan: specified threshhold to which compute the probability (ex. 0 on return will compute the probability of at least breakeven)
        >> 3. on: 'return' or 'value', the return of the stock or the final value of stock for every simulation over the time specified
        """
        if on == "return":
            predicted0 = predicted.iloc[0, 0]
            predicted = predicted.iloc[-1]
            predList = list(predicted)
            over = [
                (i * 100) / predicted0
                for i in predList
                if ((i - predicted0) * 100) / predicted0 >= higherthan
            ]
            less = [
                (i * 100) / predicted0
                for i in predList
                if ((i - predicted0) * 100) / predicted0 < higherthan
            ]
        elif on == "value":
            predicted = predicted.iloc[-1]
            predList = list(predicted)
            over = [i for i in predList if i >= higherthan]
            less = [i for i in predList if i < higherthan]
        else:
            print("'on' must be either value or return")
        try:
            return len(over) / (len(over) + len(less))
        except:
            return 0


    def simulate_mc(self, data, days, iterations, return_type="log", plot=True):
        # Generate daily returns
        returns = self.daily_returns(data, days, iterations, return_type)

        # Create empty matrix
        price_list = np.zeros_like(returns)

        # Put the last actual price in the first row of matrix.
        price_list[0] = data.iloc[-1]

        # Calculate the price of each day
        for t in range(1, days):
            price_list[t] = price_list[t - 1] * returns[t]

        # Plot Option
        if plot == True:
            x = pd.DataFrame(price_list).iloc[-1]
            fig, ax = plt.subplots(1, 2, figsize=(14, 4))
            sns.distplot(x, ax=ax[0])
            sns.distplot(x, hist_kws={"cumulative": True}, kde_kws={"cumulative": True}, ax=ax[1])
            plt.xlabel("Stock Price")
            st.pyplot(fig)

        # CAPM and Sharpe Ratio

        # Printing information about stock
        try:
            [print(nam) for nam in data.columns]
        except:
            st.header(f"{company_longName(data.name)} [{data.name}]")

        df = yf.download(data.name, period="1d")
        x = round(float(df["Adj Close"]), 2)

        d = si.get_quote_table(data.name)
        y = d.get('1y Target Est')


        st.write(f"__Days:__ {days-1}")
        st.write(f"__Current Value:__ $ {x}")
        st.write(f"__Expected Value:__ ${round(pd.DataFrame(price_list).iloc[-1].mean(),2)}")
        st.write(f"__Analyst Average 1y Est:__ ${y}")
        st.write(f"__Return:__ {round(100*(pd.DataFrame(price_list).iloc[-1].mean()-price_list[0,1])/pd.DataFrame(price_list).iloc[-1].mean(),2)}%")
        st.write(f"__Probability of Breakeven:__ {self.probs_find(pd.DataFrame(price_list),0, on='return')}")
        return pd.DataFrame(price_list)


    def monte_carlo(self, tickers, days_forecast, iterations, start_date="2000-1-1", return_type="log", plotten=False):
        tickers.append('SPY')

        data = self.import_stock_data(tickers, start=start_date)
        inform = self.beta_sharpe(tickers, data, mark_ticker="^GSPC", start=start_date)
        simulatedDF = []

        for t in range(len(tickers)):
            y = self.simulate_mc(data.iloc[:, t], (days_forecast + 1), iterations, return_type)

            if plotten == True:
                forplot = y.iloc[:, 0:10]
                forplot.plot(figsize=(15, 4))

            st.write(f"Beta: {round(inform.iloc[t,inform.columns.get_loc('Beta')],2)}")
            st.write(f"Sharpe: {round(inform.iloc[t,inform.columns.get_loc('Sharpe')],2)}")
            st.write(f"CAPM Return: {round(100*inform.iloc[t,inform.columns.get_loc('CAPM')],2)}%")
            st.write(f"{'__'*25} \n {'__'*25}")

            y["ticker"] = tickers[t]
            cols = y.columns.tolist()
            cols = cols[-1:] + cols[:-1]
            y = y[cols]
            simulatedDF.append(y)

        simulatedDF = pd.concat(simulatedDF)
        return simulatedDF


if __name__ == "__main__":
    start = "2015-1-1"
    ret_sim_df = MC_Forecast().monte_carlo(
        tickers=["NVDA", "TSLA", "AAPL", "SNOW", "PLTR", "GBTC", "ETHE", "ASML"],
        days_forecast=252,
        iterations=1000,
        start_date=start,
        return_type="log",
        plotten=False,
    )
