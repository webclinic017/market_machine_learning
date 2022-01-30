import streamlit as st

st.set_page_config(
    page_title="Invest · 4m",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",    
    menu_items={
        "Get Help": "https://www.extremelycoolapp.com/help",
        "Report a bug": "https://www.extremelycoolapp.com/bug",
        "About": "# · Invest 4m · Created By: Gordon D. Pisciotta · ",
    },
)
st.markdown(
    f""" 
    <style>
    #.reportview-container .main .block-container{{
        padding-top: {1.3}rem;
        padding-right: {2.5}rem;
        padding-left: {3.4}rem;
        padding-bottom: {3.4}rem;
    }} 
    </style> 
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <style>
    MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    footer:after {
        content:" · Invest · 4M · "; 
        visibility: visible;
        display: block;
        position: 'fixed';
        #background-color: red;
        padding: 10px;
        top: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown("<div id='linkto_top'></div>", unsafe_allow_html=True)


import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
import pages as p1
from src.tools import lists as l0


today_stamp = str(datetime.now())[:10]






if __name__ == '__main__':

    st.sidebar.title(" · NAVIGATION · ")
    st.sidebar.caption("- Navigate using this side pannel")
    st.sidebar.markdown(f"{'__'*25}")
    st.sidebar.header("__[1] Select Investment Focus__")
    
    systemStage = st.sidebar.radio("", l0.general_pages, key="nunya")
    st.sidebar.markdown(f"{'__'*25}")

    if systemStage == "Home":
        p1.Home().run_home()

    if systemStage == "Snapshot":
        p1.Snapshot(today_stamp).run_mkt_snap()

    if systemStage == "Strategy":
        p1.Strategy(today_stamp).run_the_strats()

    if systemStage == "Backtesting":
        p1.Backtest().backtest_1()

    if systemStage == "Forecasting":
        p1.Forecast(today_stamp).run_forecast()

    if systemStage == "Portfolio":
        p1.Portfolio(today_stamp).run_portfolio()

    if systemStage == "Analysis":
        p1.Analysis(today_stamp).run_analysis()


st.markdown("<a href='#linkto_top'>Link to top</a>", unsafe_allow_html=True)
