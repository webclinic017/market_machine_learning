import pandas as pd
import yfinance as yf
from yahooquery import Ticker
import streamlit as st
import webbrowser
from datetime import date, timedelta, datetime
from pathlib import Path




def recommended_stocks_0(name_lst, report_date=''):
    edate = datetime.now()
    if report_date:
        saveReport_port_results = Path(f"reports/port_results/{str(report_date)[:7]}/{str(report_date)[:10]}/")
        r_stocks = list(pd.read_csv(saveReport_port_results / f"{name_lst}.csv")["symbol"])
        return r_stocks, report_date        


    if not report_date:      
        report_date = st.sidebar.date_input(
        label="> recommender date:",
            value=date(2021, 7, 14),
            min_value=date(2021, 7, 14),
            max_value=edate,
            key="date to run proof",
            help="Select a date in the range between 2021.07.15 - 2021.08.26. \
                This date will be the date the recommender model was run and we \
                    will use the resulting tickers for our proof",
        )
        saveReport_port_results = Path(f"reports/port_results/{str(report_date)[:7]}/{str(report_date)[:10]}/")
        r_stocks = list(pd.read_csv(saveReport_port_results / f"{name_lst}.csv")["symbol"])
        return r_stocks, report_date





def recommended_stocks(name_lst, report_date='2021-07-13'):
    edate = datetime.now()
    
    report_date = st.sidebar.date_input(
    label="> recommender date:",
        value=date(2021, 7, 14),
        min_value=date(2021, 7, 14),
        max_value=edate,
        key="date to run proof",
        help="Select a date in the range between 2021.07.15 - 2021.08.26. \
            This date will be the date the recommender model was run and we \
                will use the resulting tickers for our proof",
    )
    saveReport_port_results = Path(f"reports/port_results/{str(report_date)[:7]}/{str(report_date)[:10]}/")
    r_stocks = list(pd.read_csv(saveReport_port_results / f"{name_lst}.csv")["symbol"])
    return r_stocks, report_date

    # else:

    


def recommended_stocks_2(name_lst, report_date):
    r_stocks = list(pd.read_csv(
        f"reports/port_results/{str(report_date)[:7]}/{str(report_date)[:10]}/{name_lst}.csv")["symbol"]
    )
    st.write(f"** - Below Are The Selected Stocks - total stocks = [{len(r_stocks)}]**")
    st.text(r_stocks)
    st.sidebar.write(" *" * 25)
    return r_stocks


def display_as_percent(val):
    return str(round(val * 100, 1)) + "%"


def company_longName(symbol):
    d = Ticker(symbol).quote_type
    return list(d.values())[0]["longName"]


def time_fixer(obj):
    x = ""
    y = list(str(obj))[:10]
    for i in y:
        x += i
    return x


def generate_household_watch_list(a_lst, b_lst, c_lst):
    one = a_lst + b_lst
    two = one + c_lst
    three = list(set(two))
    four = sorted(three)
    five = ""
    for i in four:
        five += i + " "
    return five


def stock_selection(ex_lst):
    st.sidebar.write("**Enter Your Stocks**")
    st.sidebar.markdown(
        f" \n\
    - Seperated each ticker with a space"
    )
    st.sidebar.write(f"** Example:** \n\n {ex_lst}")



def open_webpage(site):
    try:  # Open URL in a new tab, if a browser window is already open.
        webbrowser.open_new_tab(site)
    except Exception:  # Open URL in new window, raising the window if possible.
        webbrowser.open_new(site)


def stages(round_df, round_count, total_asset_count):
    new_asset_count = len(round_df["Symbol"])
    st.subheader(f"** > Round {round_count} Results**")
    st.write(f"** - Total Assets Pass Round {round_count} = [{new_asset_count:,d}] **")
    st.write(
        f"** - Success · Rate = [{round((new_asset_count / total_asset_count) * 100, 2)}%]**"
    )
    st.dataframe(round_df)
    st.write("__" * 25)
