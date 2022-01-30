import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import random

num_portfolios = 5000
risk_free_rate = 0.0178
import streamlit as st

st.set_option("deprecation.showPyplotGlobalUse", False)
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


class Analize_Portfolio(object):
    def __init__(self, stocks):
        self.stocks = stocks

    def situation(self):
        num_portfolios = 1
        weights_record = []

        for i in range(num_portfolios):
            weights = np.random.random(len(self.stocks))
            weights /= np.sum(weights)
            weights_record.append(weights)

        d = zip(self.stocks, weights_record[0])
        portfolio_composition = list(d)

        returns = pd.DataFrame({})
        for t in portfolio_composition:
            name = t[0]
            ticker = yf.Ticker(name)
            data = ticker.history(interval="1d", start="2010-01-01", end="2020-12-31")
            # data = yf.download(name, start="2020-01-01")
            data["return_%s" % (name)] = data["Close"].pct_change(1)

            returns = returns.join(data[["return_%s" % (name)]], how="outer").dropna()
            returns
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        st.write(
            f"\n{'* * '*25}\n\n * Presenting The Covariance Matrix For This Portfolio:\n\n\n{'* * '*25}\n"
        )
        st.dataframe(cov_matrix)

        def simulate_returns(historical_returns, forecast_days):
            return historical_returns.sample(n=forecast_days, replace=True).reset_index(
                drop=True
            )

        simulate_returns(returns["return_" + self.stocks[0]], 1000)

        def simulate_portfolio(historical_returns, composition, forecast_days):
            result = 0
            for t in composition:
                name, weight = t[0], t[1]
                s = simulate_returns(
                    historical_returns["return_%s" % (name)], forecast_days
                )
                result = result + s * weight
            return result

        simulate_portfolio(returns, portfolio_composition, 10)

        def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
            Returns = np.sum(mean_returns * weights) * 252
            std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
            return std, Returns

        def simulate_modified_returns(
            historical_returns, forecast_days, correct_mean_by
        ):
            h = historical_returns.copy()
            new_series = h + correct_mean_by
            return new_series.sample(n=forecast_days, replace=True).reset_index(
                drop=True
            )

        def simulate_modified_portfolio(historical_returns, composition, forecast_days):
            result = 0
            for t in composition:
                name, weight, correction = t[0], t[1], t[2]
                s = simulate_modified_returns(
                    historical_returns["return_%s" % (name)], forecast_days, correction
                )
                result = result + s * weight
            return result

        def simulation(historical_returns, composition, forecast_days, n_iterations):
            simulated_portfolios = None
            for i in range(n_iterations):
                sim = simulate_modified_portfolio(
                    historical_returns, composition, forecast_days
                )
                sim_port = pd.DataFrame({"returns_%d" % (i): sim})
                if simulated_portfolios is None:
                    simulated_portfolios = sim_port
                else:
                    simulated_portfolios = simulated_portfolios.join(sim_port)
            return simulated_portfolios

        st.write(f" * Presenting Individual Stock Returns:\n\n{'* * '*25}\n")
        st.dataframe(returns.mean(axis=0))

        correct_mean_by_record = []
        for idx, num in enumerate(portfolio_composition):
            y = list(portfolio_composition[idx])
            rr = random.uniform(-0.0001, -0.001)
            correct_mean_by_record.append(rr)
            y.append(rr)
            portfolio_composition[idx] = tuple(y)
        forecast_days = 20
        n_iterations = 200
        simulated_portfolios = simulation(
            returns, portfolio_composition, forecast_days, n_iterations
        )

        st.dataframe(simulated_portfolios.head())
        percentile_5th = simulated_portfolios.cumsum().apply(
            lambda x: np.percentile(x, 5), axis=1
        )
        percentile_95th = simulated_portfolios.cumsum().apply(
            lambda x: np.percentile(x, 95), axis=1
        )
        average_port = simulated_portfolios.cumsum().apply(lambda x: np.mean(x), axis=1)

        st.write(f"\n * Presentating Percentile Metrics For Portfolio:\n\n")
        st.write(f"     > 5th Percentile == \n")
        st.dataframe(percentile_5th.tail(1))
        st.write(f"     > Portfolio Average == \n")
        st.dataframe(average_port.tail(1))
        st.write(f"     > 95th Percentile == \n\n{'* * '*25}")
        st.dataframe(percentile_95th.tail(1))
        x = range(forecast_days)

        fig, ax = plt.subplots(nrows=1, ncols=1)
        plt.rcParams["figure.figsize"] = [10, 10]
        plt.plot(x, average_port, label="Average portfolio")
        plt.xlabel("Day")
        plt.ylabel("Portfolio return")
        plt.fill_between(x, percentile_5th, percentile_95th, alpha=0.2)
        plt.grid()
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig)

        target_return = 0.02
        target_prob_port = simulated_portfolios.cumsum().apply(
            lambda x: np.mean(x > target_return), axis=1
        )
        st.write(
            f"\n   * Probability of Target Portfolio Outperforming Target Return of:"
        )
        st.dataframe(target_return * 100)
        st.dataframe(target_prob_port.tail(1))
        err_bars = np.sqrt(target_prob_port * (1 - target_prob_port) / n_iterations)
        st.write(
            f"\n   * Probability of Modeling Errors In Iterations == >\n\n{'* * '*25}"
        )
        st.dataframe(err_bars.tail(1))
        x = range(forecast_days)

        fig, ax = plt.subplots(nrows=1, ncols=1)
        plt.rcParams["figure.figsize"] = [10, 10]
        plt.bar(x, target_prob_port, yerr=err_bars)
        plt.xlabel("Day")
        plt.ylabel("Probability of return >= %.2f" % (target_return))
        plt.grid()
        plt.tight_layout()
        st.pyplot(fig)

        sharpe_indices = simulated_portfolios.apply(lambda x: np.mean(x) / np.std(x))

        fig, ax = plt.subplots(nrows=1, ncols=1)
        plt.hist(sharpe_indices, bins="rice")
        plt.xlabel("Sharpe ratio")
        plt.tight_layout()
        st.pyplot(fig)

        df = pd.DataFrame(portfolio_composition)
        df.columns = ["stock_tickers", "random_wt", "mean_return_var"]
        df.set_index("stock_tickers", inplace=True)
        st.write(f"  >Mean Simulated Portfolio Mean Returns == \n\n{'* * '*25}\n")
        st.dataframe(np.mean(sharpe_indices))
        st.write(f"\n  > Portfolio:\n\n\n{'* * '*25}/n")
        st.dataframe(df)


if __name__ == "__main__":
    my_positions = pd.read_pickle(
        "/home/gordon/gdp/code/portfolio/Market_Mapping/files/tickers/chuck_merged_ticker_lst.pkl"
    )
    Analize_Portfolio(my_positions, "chuck_merged").situation()
