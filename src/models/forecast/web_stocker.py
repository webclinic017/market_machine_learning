import matplotlib.pyplot as plt
from .web_stocker_helper import Stocker
import streamlit as st
import matplotlib as mpl
import yfinance as yf
import datetime
import pandas as pd
from yahooquery import Ticker as yq_Ticker

from src.tools import functions as f1

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
st.set_option("deprecation.showPyplotGlobalUse", False)


def web_stocker_run(stock_ticker, forcast_period, e_date):

    company_long_name = f1.company_longName(stock_ticker)
    microsoft = Stocker(stock_ticker)

    df = yf.download(stock_ticker, period="max")["Adj Close"]
    df.index = pd.to_datetime(df.index)
    starter = datetime.date(2021, 1, 5)
    starter_date = pd.Timestamp(starter)
    variable = float(df[df.index == starter_date].values)
    nshares = round(1000.00 / variable, 2)

    f"""
     * Lets assume we purchased shares of {company_long_name} at it's IPO for  $1,000 giving us {nshares} shares
    """
    st.write(
        f"- Lets assume we purchased shares of {company_long_name} at the beginning of this year (01/05/2021)"
    )
    st.write(
        f"\
        - Purchase Price Per Share: ${round(variable,2)} \n\
        - Shares Purchased: {round(nshares,2)} \n\
        - Total Cost: $1,000"
    )
    st.pyplot(microsoft.buy_and_hold(start_date=starter, nshares=nshares))

    microsoft.weekly_seasonality = True
    model, model_data = microsoft.create_prophet_model(days=forcast_period)
    st.subheader(f"Forecast With Seasonality - {forcast_period} Days")
    st.pyplot(model.plot(model_data))
    microsoft.weekly_seasonality = False

    """
    [Important Concept for Time-Series]
    * Changepoints occur at the maximum value of the second derivative.
    * Prophet only identifies changepoints in the first 80% of the data
    """

    microsoft.create_model()
    st.pyplot(microsoft.changepoint_prior_analysis())

    microsoft.create_model()
    st.pyplot(microsoft.changepoint_prior_validation())

    microsoft.create_model()
    st.pyplot(microsoft.changepoint_date_analysis())

    microsoft.create_model()
    model, model_data = microsoft.create_prophet_model(days=forcast_period)
    st.subheader(f"{company_long_name} [{stock_ticker}] {forcast_period}-Day Forecast")
    plt.grid(True)
    st.pyplot(model.plot(model_data))

    st.pyplot(microsoft.evaluate_prediction(nshares=10))

    microsoft.create_model()
    st.pyplot(microsoft.predict_future(days=forcast_period))


if __name__ == "__main__":
    web_stocker_run("LAZR", 90)
