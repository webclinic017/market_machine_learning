from tracemalloc import start
import warnings
import streamlit as st
from pathlib import Path
from fbprophet.plot import add_changepoints_to_plot
from fbprophet import Prophet
import yfinance as yf
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from finvizfinance.quote import finvizfinance
import seaborn as sns
from yahoo_fin import stock_info as si
from yahooquery import Ticker
import pickle5 as pickle

warnings.filterwarnings("ignore")
# mpl.use("Agg")
plt.style.use("seaborn-poster")
sm, med, lg = 10, 15, 20
plt.rc("font", size=sm)  # controls default text sizes
plt.rc("axes", titlesize=med)  # fontsize of the axes title
plt.rc("axes", labelsize=med)  # fontsize of the x & y labels
plt.rc("xtick", labelsize=sm)  # fontsize of the tick labels
plt.rc("ytick", labelsize=sm)  # fontsize of the tick labels
plt.rc("legend", fontsize=sm)  # legend fontsize
plt.rc("figure", titlesize=lg)  # fontsize of the figure title
plt.rc("axes", linewidth=2)  # linewidth of plot lines
plt.rcParams["figure.figsize"] = [20, 10]
plt.rcParams["figure.dpi"] = 100


score = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
rating = ["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell", "No Data"]
scale = dict(zip(rating, score))

def get_key(val):
    for key, value in scale.items():
        if val == value:
            return key

def company_longName(symbol):
    d = Ticker(symbol).quote_type
    return list(d.values())[0]["longName"]





class Web_prophet_kyle(object):


    def __init__(self, stock, ender_date, per=360, hist="2y"):
        """
        forecast the given ticker (stock) period days into the future (from today)
        ---------inputs----------
        > ticker ->> ticker of stock to forecast
        > periods->> number of days into the future to forecast (from today's date)
        > hist   ->> amount of historical data to use
            [default=max] -> options(1d,5d,1mo,3mo,6mo,1y,2y,5y,10y}
        """
        self.stock = stock
        self.per = per
        self.end = str(ender_date)[:10]
        self.hist = hist
        self.company = company_longName(self.stock)
        st.write(self.end)


    def run_prophet(self):
        file = Path(f"data/raw/2021-09-01/{self.stock}.pkl")
        if file.exists():
            with open(file, "rb") as fh:
                stock_data = pickle.load(fh)

        hammerTime = Ticker(
            self.stock,
            asynchronous=True,
            formatted=False,
            backoff_factor=0.34,
            validate=True,
            verify=True,
        )
        spy_hist = hammerTime.history(period=self.hist).reset_index().set_index('date')
        stock_data = pd.DataFrame(spy_hist[spy_hist['symbol'] == [self.stock][0]])
        del stock_data['symbol']

        stock_data.reset_index(inplace=True)
        stock_data.fillna(0.0, inplace=True)
        df = pd.DataFrame(stock_data[["date", "close"]])  # select Date and Price
        df = df.rename(columns={"date": "ds", "close": "y"})

        x = pd.DataFrame(df.copy()).set_index('ds')
        x.index = pd.to_datetime(x.index)
        df_train = x.loc[:self.end].reset_index().dropna()
        df_test = x.loc[self.end:].reset_index().dropna()        


        # create a Prophet model from that data
        m = Prophet(
            growth='linear',
            n_changepoints=13,
            daily_seasonality=True,
            yearly_seasonality=True,
            seasonality_prior_scale=10,
            changepoint_prior_scale=0.1,
            uncertainty_samples=1300,
            stan_backend=None,
        )

        m.fit(df_train)
        future = m.make_future_dataframe(self.per, freq="D", include_history=True)
        forecast = m.predict(future)
        forecast = forecast[["ds", "trend", "yhat_lower", "yhat_upper", "yhat"]]

        # create plot
        fig1 = m.plot(
            forecast,
            ax=None,
            uncertainty=True,
            plot_cap=True,
            xlabel="Date",
            ylabel="Stock Price",
        )

        add_changepoints_to_plot(fig1.gca(), m, forecast)
        plt.title(f"Prophet Model ChangePoints - {self.company} ({self.stock}) - {self.per} Day Forecast")
        plt.legend(["actual", "prediction", "changePoint_line"], loc="best")
        st.pyplot(fig1)


        model_prophet = Prophet(
            growth='linear',
            n_changepoints=13,
            daily_seasonality=True,
            yearly_seasonality=True,
            seasonality_mode="additive",
            seasonality_prior_scale=10,
            changepoint_prior_scale=0.1,
            uncertainty_samples=1300,
            stan_backend=None,
        )

        model_prophet.add_seasonality(name="monthly", period=1, fourier_order=5)
        model_prophet.fit(df_train)

        df_future = model_prophet.make_future_dataframe(periods=self.per, freq="D", include_history=True)
        df_pred = model_prophet.predict(df_future)
        fig2 = model_prophet.plot(
            df_pred,
            ax=None,
            uncertainty=True,
            plot_cap=True,
            xlabel="Date",
            ylabel="Stock Price",
        )
        plt.show()
        st.pyplot(fig2)

        # fig2 = model_prophet.plot_components(df_pred, uncertainty=True)
        # plt.show()
        # st.pyplot(fig2)        



        # create a Prophet model from that data
        mp = Prophet(
            growth='linear',
            n_changepoints=13,
            daily_seasonality=True,
            yearly_seasonality=True,
            seasonality_prior_scale=10,
            changepoint_prior_scale=0.1,
            uncertainty_samples=1300,
            stan_backend=None,
        )

        mp.fit(df)
        future1 = mp.make_future_dataframe(self.per, freq="D", include_history=True)
        forecast1 = mp.predict(future1)
        forecast1 = forecast1[["ds", "trend", "yhat_lower", "yhat_upper", "yhat"]]

        # create plot
        fig3 = mp.plot(
            forecast1,
            ax=None,
            uncertainty=True,
            plot_cap=True,
            xlabel="Date",
            ylabel="Stock Price",
        )

        add_changepoints_to_plot(fig1.gca(), mp, forecast1)
        plt.title(f"Prophet Model ChangePoints - {self.company} ({self.stock}) - {self.per} Day Forecast")
        plt.legend(["actual", "prediction", "changePoint_line"], loc="best")
        st.pyplot(fig3)        




        selected_columns = ["ds", "yhat_lower", "yhat_upper", "yhat"]
        df_pred = forecast1.loc[:, selected_columns].reset_index(drop=True)
        df_test = df_test.merge(df_pred, on=["ds"], how="left")
        df_test.ds = pd.to_datetime(df_test.ds)
        df_test.set_index("ds", inplace=True)

        df_test.columns = ['actual', 'lower_confidence_boundary (95%)', 'upper_confidence_boundary (95%)','prediction']

        fig, ax = plt.subplots(1, 1)
        ax = sns.lineplot(data=df_test[['actual', 'lower_confidence_boundary (95%)', 'upper_confidence_boundary (95%)','prediction']])
        ax.fill_between(df_test.index, df_test['lower_confidence_boundary (95%)'], df_test['upper_confidence_boundary (95%)'], alpha=0.3)
        ax.set(
            title=f"{self.company} [{self.stock}] - actual vs. predicted",
            xlabel="Date",
            ylabel="Price ($)",
        )
        st.pyplot(fig)


        try:
            fd = yf.download(self.stock, period="1d")
            x = round(float(fd["Adj Close"]), 2)
            st.write(f"__ > {self.company} · [{self.stock}] - Current Stock Price = **${x}** __")
        except Exception:
            pass

        try:
            st.write(f"__> {self.per} · Day Forcast · [{self.company} ({self.stock}) ] = ${round(float(forecast['yhat'].iloc[-1]),2)} __")
        except Exception:
            pass

        try:
            analyst_1yr = float(finvizfinance(self.stock).TickerFundament()["Target Price"])
            st.write(f"__ > {self.company} · [{self.stock}] - Current Analyst 1yr Price Estimate = **{analyst_1yr}** __")
        except Exception:
            pass

        return


if __name__ == "__main__":
    Web_prophet_kyle("NVDA").run_prophet()
