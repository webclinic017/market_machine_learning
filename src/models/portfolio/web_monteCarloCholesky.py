from genericpath import exists
import warnings
warnings.filterwarnings("ignore")
from yahoo_fin import stock_info as si
import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
from scipy.stats import norm, gmean, cauchy
import seaborn as sns
from datetime import date, datetime, timedelta
import yfinance as yf
from yahooquery import Ticker
import streamlit as st
from finvizfinance.quote import finvizfinance
from pathlib import Path

from src.tools.functions import company_longName


class MonteCarloCholesky(object):

    def __init__(self, date_report):
        self.report_date = date_report
        self.saveAdvisor = Path(f"data/advisor/{str(self.report_date)[:7]}/{self.report_date}/")
        if not self.saveAdvisor.exists():
            self.saveAdvisor.mkdir(parents=True)


    def import_stock_data(self, tickers, saveName="xxx", start="2010-1-1", end=datetime.today().strftime("%Y-%m-%d"),):        
        y = Ticker(
            tickers, 
            formatted=False, 
            backoff_factor=0.34, 
            validate=True
        ).history(
            start=start,
            end=end,
            interval='1d'
        ).reset_index().set_index('date')

        df = pd.DataFrame()
        for i in tickers:
            z = pd.DataFrame(y[y['symbol'] == i]['adjclose'])
            df[i] = z
        return df.round(2)
        # return pd.DataFrame(yf.download(tickers, start=start, end=self.report_date)["Adj Close"])


    def log_returns(self, data):
        return np.log(1 + data.pct_change())


    def simple_returns(self, data):
        return (data / data.shift(1)) - 1


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


    def get_tickers(self, data):
        tickers = [i for i in data.columns]
        return tickers


    def probs_find(self, predicted, higherthan, ticker=None, on="value"):
        """
        This function calculated the probability of a stock being above a certain threshhold, which can be defined as a value (final stock price) or return rate (percentage change)
        Input:
        1. predicted: dataframe with all the predicted prices (days and simulations)
        2. higherthan: specified threshhold to which compute the probability (ex. 0 on return will compute the probability of at least breakeven)
        3. on: 'return' or 'value', the return of the stock or the final value of stock for every simulation over the time specified
        4. ticker: specific ticker to compute probability for
        """
        if ticker == None:
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
                st.write("'on' must be either value or return")

        else:

            if on == "return":
                predicted = predicted[predicted["ticker"] == ticker]
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
                st.write("'on' must be either value or return")
        return len(over) / (len(over) + len(less))


    def montecarlo_cholesky(self,tickers,days,iterations,start="2010-1-1",show_hist=True,show_line=True,show_stats=True,):

        # Get tickers and number of tickers involved in this portfolio
        data = self.import_stock_data(tickers, "monteCarloCholesky", start=start)
        ticks = self.get_tickers(data)
        numstocks = len(ticks)

        # Basic information and data
        log_return = self.log_returns(data)

        # Brownian motion component: drif
        drift = self.drift_calc(data)
        stdev = self.log_returns(data).std().values

        # Cholesky decomposition
        covari = log_return.cov()
        chol = np.linalg.cholesky(covari)

        # Generate uncorralated random variables and use cholesky decomposition to correlate them
        uncorr_x = norm.ppf(np.random.rand(numstocks, iterations * days))
        corr_x = np.dot(chol, uncorr_x)

        # Calculate daily return
        corr_2 = np.zeros_like(corr_x)
        for i in range(numstocks):
            corr_2[i] = np.exp(drift[i] + corr_x[i])
        simulatedDF = []

        # For every stock, calculate the returns
        for s in range(len(ticks)):
            ret_reshape = corr_2[s]
            ret_reshape = ret_reshape.reshape(days, iterations)
            price_list = np.zeros_like(ret_reshape)
            price_list[0] = data.iloc[-1, s]

            # Calculate returns for the s stock
            for t in range(1, days):
                price_list[t] = price_list[t - 1] * ret_reshape[t]

            # Create dataframe for this run, input the name of stock and load
            y = pd.DataFrame(price_list)
            y["ticker"] = tickers[s]
            cols = y.columns.tolist()
            cols = cols[-1:] + cols[:-1]
            y = y[cols]
            simulatedDF.append(y)

            # plotting
            if show_hist == True:
                x = pd.DataFrame(price_list).iloc[-1]
                fig, ax = plt.subplots(1, 2, figsize=(14, 4))
                sns.distplot(x, ax=ax[0], axlabel="Stock Price")
                sns.distplot(
                    x,
                    hist_kws={"cumulative": True},
                    kde_kws={"cumulative": True},
                    ax=ax[1],
                )
                plt.xlabel("Stock Price")
                plt.show()

            if show_line == True:
                y = pd.DataFrame(price_list[:, 0:10]).plot(figsize=(15, 6))

            if show_stats == True:
                # Printing basic stats
                df = yf.download(ticks[s], period="1d")
                x = round(float(df["Adj Close"]), 2)

                d = si.get_quote_table(ticks[s])
                y = d.get('1y Target Est')                

                st.subheader(f"__{company_longName(ticks[s])} [{ticks[s]}]__")
                st.write(f"__Forcast Days:__ {days}")
                st.write(f"__Current Value:__ $ {x}")
                st.write(f"__Analyst Average 1y Est:__ ${y}")
                st.write(
                    f"__Expected Value:__ ${round(pd.DataFrame(price_list).iloc[-1].mean(),2)}"
                )
                st.write(
                    f"__Return:__ {round(100*(pd.DataFrame(price_list).iloc[-1].mean()-price_list[0,1])/pd.DataFrame(price_list).iloc[-1].mean(),2)}%"
                )
                st.write(
                    f"__Probability of Breakeven:__ {self.probs_find(pd.DataFrame(price_list), 0, on='return')}"
                )
                st.write(f"{'__'*25}")

        simulatedDF = pd.concat(simulatedDF)
        return simulatedDF


    def market_data_combination(self, tickers, data, mark_ticker="^GSPC", start="2010-1-1"):
        ticks = [col for col in data.columns]

        if mark_ticker in ticks:
            ann_return = np.exp(self.log_returns(data).mean() * 252).values - 1

        else:
            market_data = self.import_stock_data(mark_ticker, "mkt_data_combo_1", start)
            market_rets = self.log_returns(market_data).dropna()
            ann_return = np.exp(market_rets.mean() * 252)
            data = data.merge(market_data, left_index=True, right_index=True)
            data.columns = [tickers[0], tickers[1], tickers[2], "^GSPC"]

        return data, ann_return

    def beta_sharpe(self,tickers,data,mark_ticker="^GSPC",start="2010-1-1",riskfree=0.025,riskpremium=0.05,):

        # Beta
        dd, mark_ret = self.market_data_combination(tickers, data, mark_ticker, start)

        # load data and annual returns
        log_ret = self.log_returns(dd)  # create the logarithmic returns of the data
        covar = log_ret.cov() * 252  # Annualized covariance matrix
        covar = pd.DataFrame(covar.iloc[:-1, -1])

        # Get the part of the covariance matrix that is related between the stocks and the market
        mrk_var = log_ret.iloc[:, -1].var() * 252  # Annualized variance of market
        beta = covar / mrk_var  # Now we have our betas!

        # Add the standard deviation to the beta dataframe
        stdev_ret = pd.DataFrame(((log_ret.std() * 250 ** 0.5)[:-1]), columns=["STD"])
        beta = beta.merge(stdev_ret, left_index=True, right_index=True)

        # Get tickers of all the stocks in the dataframe used
        tickers = self.get_tickers(dd)

        # Make dictionary for the annual return of each stock
        mark_ret = {tickers[i]: mark_ret[i] for i in range(len(tickers))}

        # CAPM
        for i, row in beta.iterrows():
            beta.at[i, "CAPM"] = riskfree + (
                row[mark_ticker] * (mark_ret[mark_ticker] - riskfree)
            )
        # Sharpe
        for i, row in beta.iterrows():
            beta.at[i, "Sharpe"] = (row["CAPM"] - riskfree) / (row["STD"])
        beta.rename(columns={"^GSPC": "Beta"}, inplace=True)
        return beta


    def market_data_combination(self, data, mark_ticker="^GSPC", start="2010-1-1"):
        market_data = self.import_stock_data(mark_ticker, "mkt_data_combo_2", start)
        market_rets = self.log_returns(market_data).dropna()
        ann_return = np.exp(market_rets.mean() * 252).values - 1
        data = data.merge(market_data, left_index=True, right_index=True)
        return data, ann_return


    def montecarlo_sharpe_optimal_portfolio(self, tickers, trials=13000, end_date=None, start_date="2020-1-1", riskfree=0.025, plot_eff=True,):

        if end_date == None:
            end_date = self.report_date
            # end_date = datetime.today().strftime("%Y-%m-%d")

        data = self.import_stock_data(tickers, "monteCarloSharpe")
        allWeights = np.zeros((trials, len(data.columns)))
        allReturns = np.zeros(trials)
        allVolatility = np.zeros(trials)
        allSharpeValues = np.zeros(trials)
        log_return = self.log_returns(data)

        for t in range(trials):
            weights = np.random.rand(len(data.columns))
            weights = weights / np.sum(weights)
            allWeights[t, :] = weights
            allReturns[t] = np.sum((log_return.mean() * weights) * 252)
            allVolatility[t] = np.sqrt(
                np.dot(weights.T, np.dot(log_return.cov() * 252, weights))
            )
            allSharpeValues[t] = (allReturns[t] - riskfree) / allVolatility[t]

        maxsharpe = allSharpeValues.max()
        pointsharpe = allSharpeValues.argmax()
        weightSharpe = allWeights[pointsharpe]
        x_sharpe = allVolatility[pointsharpe]
        y_sharpe = allReturns[pointsharpe]
        maxret = allReturns.max()
        pointret = allReturns.argmax()
        weightRet = allWeights[pointret]
        x_ret = allVolatility[pointret]
        y_ret = allReturns[pointret]

        if plot_eff == True:
            pass
            # fig, ax = plt.subplots(figsize=(14, 9))
            # plt.scatter(allVolatility, allReturns, c=allSharpeValues, cmap="plasma")
            # plt.colorbar(label="Sharpe Ratio")
            # plt.xlabel("Volatility")
            # plt.ylabel("Expected Return")
            # plt.scatter(x_sharpe, y_sharpe, c="black")
            # plt.scatter(x_ret, y_ret, c="black")
            # st.pyplot(fig)

        optim_dic = []
        for i in range(len(tickers)):
            optim_dic.append({"ticker": tickers[i], "Weight": weightSharpe[i] * 100})

        fin = pd.DataFrame(optim_dic)
        return fin



# if __name__ == "__main__":
#     t_lst = ["GOOG", "FB", "^GSPC"]
#     MonteCarloCholesky().montecarlo_cholesky(
#         t_lst, days=252, iterations=10000, start="2015-1-1", show_hist=True
#     )

#     ticker_lst = ["AAPL", "ASML", "NVDA", "SNOW", "PLTR", "TSLA"]
#     no_trials = 1000
#     fin = MonteCarloCholesky().montecarlo_sharpe_optimal_portfolio(
#         ticker_lst, no_trials
#     )
#     st.dataframe(fin)
