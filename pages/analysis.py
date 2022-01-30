import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, date
from yahooquery import Ticker as yq_Ticker
import yfinance as yf
from yfinance import Ticker
from src.data import yahoo_fin_stock_info as si
import src.data.yahoo_fin_stock_info as SI
import yahoo_fin.news as news
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

import src.models.analysis as a1
from src.tools import functions as f0
from src.tools import lists as l0
from src.tools import scripts as s0
from src.tools import widgets as w0

color_discrete_sequence = px.colors.qualitative.Plotly


def widget_1(script):
    st.subheader("** > Details:**")
    st.write(f"{script}")
    st.write(f"{'__'*25}")


def get_company_longName(symbol):
    d = yq_Ticker(symbol).quote_type
    return list(d.values())[0]["longName"]


analysis_script_1 = "\
    - Fundamental analysis is a method of evaluating the intrinsic value of an asset and analyzing \
            the factors that could influence its price in the future. \n\
    * This form of analysis is based on \
            external events and influences, as well as financial statements and industry trends. \n\
    * Fundamental analysts are concerned with the difference between a stock's value, and the price at which it is trading."
analysis_script_2 = "\
    - Use the financial ratios for initial screening \n\
    - Understand the company \n\
    - Study the financial reports of the company \n\
    - Check the debt and red signs \n\
    - Find the company's competitors \n\
    - Analyse the future prospects."
analysis_script_3 = f"\
    * Fundamental analysis is a method of determining a stock's real or 'fair market' value. \n\
    * Fundamental analysts search for stocks that are currently trading at prices that are higher or lower than their real value. \n\
    * If the fair market value is higher than the market price, the stock is deemed to be undervalued and a buy recommendation is given. \n\
    * In contrast, technical analysts ignore the fundamentals in favor of studying the historical price trends of the stock."
analysis_script_4 = " \
    * Technical analysis is a trading discipline employed to evaluate investments and identify \
        trading opportunities by analyzing statistical trends gathered from trading activity, \
        such as price movement and volume... \n\
    * Technical analysis can be used on any security with historical trading data. \n\
    * In finance, technical analysis is an analysis methodology for forecasting \
        the direction of prices through the study of past market data, primarily price and volume. \n\
    * Technical analysis is concerned with price action, which gives clues as to the stock's supply and demand dynamics \
        – which is what ultimately determines the stock price."
analysis_script_5 = "\
    * All of the tools have the same purpose: to make understanding chart \
        movements and identifying trends easier for technical traders. \n\
    * moving averages \n\
    * support and resistance levels \n\
    * Bollinger bands \n\
    * and more \n\
    * https://www.investopedia.com/top-7-technical-analysis-tools-4773275"
analysis_script_6 = f"\
    * Technical analysis, or using charts to identify trading signals and price patterns, \
        may seem overwhelming or esoteric at first. \n\
    * Beginners should first understand why technical analysis works as a window \
        into market psychology to identify opportunities to profit. \n\
    * Focus on a particular trading approach and develop a disciplined strategy that you \
        can follow without letting emotions or second-guessing get in the way. \n\
    * Find a broker that can help you execute your plan affordably while also providing \
        a trading platform with the right suite of tools you'll need. "
instructions_script = "\
    * All Interaction will operate through the side-pannel \n\
    * The side-pannel can be accessed via the '>' icon in the upper-left corner \n\
    * Follow the Steps on the side pannel for each model. \n\
    * Each graph can be maximized to full screen by clicking the icon off to the upper-right of the graph \n\
    * Each model will have their own criteria either asking for a single ticker or ticker list\n \n\
    * When the model is ready to run a 'RUN' button will appear - click it to start the model"





class Analysis(object):


    def __init__(self, today_stamp):
        self.today_stamp = today_stamp
        self.saveMonth = str(self.today_stamp)[:7]
        self.saveTickers = Path(f"data/tickers/{self.saveMonth}/{self.today_stamp}/")
        if not self.saveTickers.exists():
            self.saveTickers.mkdir(parents=True)


    def analysis_homePage(self):
        st.header(
            f"** · [Analysis] · ** \n\
{'__'*25}"
        )

        st.header(" > Fundamental Analysis")
        my_expander = st.expander("Expand", expanded=False)
        with my_expander:
            clicked = w0.widget_online_resource("https://www.investopedia.com/terms/f/fundamentalanalysis.asp")
            clicked = w0.widget_analysis(" > Definition", analysis_script_1)
            clicked = w0.widget_analysis(" > The 6 Segments to perform fundamental analysis on stocks",analysis_script_2,)
            clicked = w0.widget_analysis(" > KEY TAKEAWAYS", analysis_script_3)

        st.header(" > Technical Analysis")
        my_expander = st.expander("Expand", expanded=False)
        with my_expander:
            clicked = w0.widget_online_resource("https://www.investopedia.com/terms/t/technicalanalysis.asp")
            clicked = w0.widget_analysis(" > Definition", analysis_script_4)
            clicked = w0.widget_analysis(" > Examples of technical analysis tools include:", analysis_script_5)
            clicked = w0.widget_analysis(" > KEY TAKEAWAYS", analysis_script_6)

        st.header("> Procedure")
        my_expander = st.expander("Expand", expanded=False)
        with my_expander:
            clicked = w0.widget_analysis("Instructions:", instructions_script)


    def run_signal_analysis(self, ticker):
        a1.The_Financial_Signal_Processing(ticker)
        st.header("**Analysis Done**")


    def capm_cagr(self, tickers):
        st.header("__CAPM & CAGR__")
        st.write(f"{'__'*25} \n {'__'*25}")
        a1.CAPM_CAGR().configure_mod(tickers)


    def run_fundamental_analysis(self, ticker):
        fundamental_company = get_company_longName(ticker)
        stock = yf.Ticker(ticker)
        info = stock.info
        st.title("Company Profile")
        st.subheader(info["longName"])
        st.markdown("** Sector **: " + info["sector"])
        st.markdown("** Industry **: " + info["industry"])
        st.markdown("** Phone **: " + info["phone"])
        st.markdown(
            "** Address **: "
            + info["address1"]
            + ", "
            + info["city"]
            + ", "
            + info["zip"]
            + ", "
            + info["country"]
        )
        st.markdown("** Website **: " + info["website"])
        st.markdown("** Business Summary **")
        st.info(info["longBusinessSummary"])
        fundInfo = {
            "Enterprise Value (USD)": info["enterpriseValue"],
            "Enterprise To Revenue Ratio": info["enterpriseToRevenue"],
            "Enterprise To Ebitda Ratio": info["enterpriseToEbitda"],
            "Net Income (USD)": info["netIncomeToCommon"],
            "Profit Margin Ratio": info["profitMargins"],
            "Forward PE Ratio": info["forwardPE"],
            "PEG Ratio": info["pegRatio"],
            "Price to Book Ratio": info["priceToBook"],
            "Forward EPS (USD)": info["forwardEps"],
            "Beta ": info["beta"],
            "Book Value (USD)": info["bookValue"],
            "Dividend Rate (%)": info["dividendRate"],
            "Dividend Yield (%)": info["dividendYield"],
            "Five year Avg Dividend Yield (%)": info["fiveYearAvgDividendYield"],
            "Payout Ratio": info["payoutRatio"],
        }
        fundDF = pd.DataFrame.from_dict(fundInfo, orient="index")
        fundDF = fundDF.rename(columns={0: "Value"})
        st.subheader("Fundamental Info")
        st.table(fundDF)

        st.subheader("General Stock Info")
        st.markdown("** Market **: " + info["market"])
        st.markdown("** Exchange **: " + info["exchange"])
        st.markdown("** Quote Type **: " + info["quoteType"])
        start = datetime.today() - timedelta(2 * 365)
        end = datetime.today()
        df = yf.download(ticker, start, end)
        df = df.reset_index()

        fig = go.Figure(data=go.Scatter(x=df["Date"], y=df["Adj Close"]))
        fig.update_layout(
            title={
                "text": "Stock Prices Over Past Two Years",
                "y": 0.9,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
            }
        )
        st.plotly_chart(fig, use_container_width=True)

        marketInfo = {
            "Volume": info["volume"],
            "Average Volume": info["averageVolume"],
            "Market Cap": info["marketCap"],
            "Float Shares": info["floatShares"],
            "Regular Market Price (USD)": info["regularMarketPrice"],
            "Bid Size": info["bidSize"],
            "Ask Size": info["askSize"],
            "Share Short": info["sharesShort"],
            "Short Ratio": info["shortRatio"],
            "Share Outstanding": info["sharesOutstanding"],
        }
        marketDF = pd.DataFrame(data=marketInfo, index=[0])
        st.table(marketDF)
        st.write(" *" * 25)

        for i in range(1):
            try:
                st.write(f"live_price = $ {si.get_live_price(ticker)}")
                st.write(f"postmarket_price = $ {si.get_postmarket_price(ticker)}")
                st.write(f"premarket_price = $ {si.get_premarket_price(ticker)}")
            except Exception:
                pass
        st.header("**Analysis Done**")


    def run_technical_analysis(self, ticker):

        def calcMovingAverage(data, size):
            df = data.copy()
            df["sma"] = df["Adj Close"].rolling(size).mean()
            df["ema"] = df["Adj Close"].ewm(span=size, min_periods=size).mean()
            df.dropna(inplace=True)
            return df

        def calc_macd(data):
            df = data.copy()
            df["ema12"] = df["Adj Close"].ewm(span=12, min_periods=12).mean()
            df["ema26"] = df["Adj Close"].ewm(span=26, min_periods=26).mean()
            df["macd"] = df["ema12"] - df["ema26"]
            df["signal"] = df["macd"].ewm(span=9, min_periods=9).mean()
            df.dropna(inplace=True)
            return df

        def calcBollinger(data, size):
            df = data.copy()
            df["sma"] = df["Adj Close"].rolling(size).mean()
            df["bolu"] = df["sma"] + 2 * df["Adj Close"].rolling(size).std(ddof=0)
            df["bold"] = df["sma"] - 2 * df["Adj Close"].rolling(size).std(ddof=0)
            df["width"] = df["bolu"] - df["bold"]
            df.dropna(inplace=True)
            return df

        technical_company = get_company_longName(ticker)
        st.subheader(f"** > {technical_company} [{ticker}]**")

        st.subheader("Moving Average")
        coMA1, coMA2 = st.columns(2)
        with coMA1:
            numYearMA = st.number_input(
                "Insert period (Year): ",
                min_value=1,
                max_value=10,
                value=2,
                key=f"{ticker}_1",
            )
        with coMA2:
            windowSizeMA = st.number_input(
                "Window Size (Day): ",
                min_value=5,
                max_value=500,
                value=20,
                key=f"{ticker}_2",
            )

        start = datetime.today() - timedelta(numYearMA * 365)
        end = datetime.today()
        dataMA = yf.download(ticker, start, end)
        df_ma = calcMovingAverage(dataMA, windowSizeMA)
        df_ma = df_ma.reset_index()

        figMA = go.Figure()
        figMA.add_trace(
            go.Scatter(
                x=df_ma["Date"],
                y=df_ma["Adj Close"],
                name="Prices Over Last " + str(numYearMA) + " Year(s)",
            )
        )
        figMA.add_trace(
            go.Scatter(
                x=df_ma["Date"],
                y=df_ma["sma"],
                name="SMA"
                + str(windowSizeMA)
                + " Over Last "
                + str(numYearMA)
                + " Year(s)",
            )
        )
        figMA.add_trace(
            go.Scatter(
                x=df_ma["Date"],
                y=df_ma["ema"],
                name="EMA"
                + str(windowSizeMA)
                + " Over Last "
                + str(numYearMA)
                + " Year(s)",
            )
        )
        figMA.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        figMA.update_layout(legend_title_text="Trend")
        figMA.update_yaxes(tickprefix="$")
        st.plotly_chart(figMA, use_container_width=True)

        st.subheader("Moving Average Convergence Divergence (MACD)")
        numYearMACD = st.number_input(
            "Insert period (Year): ",
            min_value=1,
            max_value=10,
            value=2,
            key=f"{ticker}_3",
        )
        startMACD = datetime.today() - timedelta(numYearMACD * 365)
        endMACD = datetime.today()
        dataMACD = yf.download(ticker, startMACD, endMACD)
        df_macd = calc_macd(dataMACD)
        df_macd = df_macd.reset_index()

        figMACD = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.01
        )
        figMACD.add_trace(
            go.Scatter(
                x=df_macd["Date"],
                y=df_macd["Adj Close"],
                name="Prices Over Last " + str(numYearMACD) + " Year(s)",
            ),
            row=1,
            col=1,
        )
        figMACD.add_trace(
            go.Scatter(
                x=df_macd["Date"],
                y=df_macd["ema12"],
                name="EMA 12 Over Last " + str(numYearMACD) + " Year(s)",
            ),
            row=1,
            col=1,
        )
        figMACD.add_trace(
            go.Scatter(
                x=df_macd["Date"],
                y=df_macd["ema26"],
                name="EMA 26 Over Last " + str(numYearMACD) + " Year(s)",
            ),
            row=1,
            col=1,
        )
        figMACD.add_trace(
            go.Scatter(x=df_macd["Date"], y=df_macd["macd"], name="MACD Line"),
            row=2,
            col=1,
        )
        figMACD.add_trace(
            go.Scatter(x=df_macd["Date"], y=df_macd["signal"], name="Signal Line"),
            row=2,
            col=1,
        )
        figMACD.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="left", x=0)
        )
        figMACD.update_yaxes(tickprefix="$")
        st.plotly_chart(figMACD, use_container_width=True)

        st.subheader("Bollinger Band")
        coBoll1, coBoll2 = st.columns(2)
        with coBoll1:
            numYearBoll = st.number_input(
                "Insert period (Year): ",
                min_value=1,
                max_value=10,
                value=2,
                key=f"{ticker}_4",
            )
        with coBoll2:
            windowSizeBoll = st.number_input(
                "Window Size (Day): ",
                min_value=5,
                max_value=500,
                value=20,
                key=f"{ticker}_5",
            )

        startBoll = datetime.today() - timedelta(numYearBoll * 365)
        endBoll = datetime.today()
        dataBoll = yf.download(ticker, startBoll, endBoll)
        df_boll = calcBollinger(dataBoll, windowSizeBoll)
        df_boll = df_boll.reset_index()

        figBoll = go.Figure()
        figBoll.add_trace(
            go.Scatter(x=df_boll["Date"], 
            y=df_boll["bolu"], 
            name="Upper Band"
            )
        )
        figBoll.add_trace(
            go.Scatter(
                x=df_boll["Date"],
                y=df_boll["sma"],
                name="SMA"
                + str(windowSizeBoll)
                + " Over Last "
                + str(numYearBoll)
                + " Year(s)",
            )
        )
        figBoll.add_trace(
            go.Scatter(x=df_boll["Date"], y=df_boll["bold"], name="Lower Band")
        )
        figBoll.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="left", x=0)
        )
        figBoll.update_yaxes(tickprefix="$")
        st.plotly_chart(figBoll, use_container_width=True)
        st.header("**Analysis Done**")

    # ------------------------------------------------------------------------------------------ > stage: [ANALYSIS]

    def run_analysis(self):
        st.sidebar.header("[Step # 2]")
        analysis_method = st.sidebar.radio(
            "Select Analysis Medthod:", 
            l0.feature_analysis
        )
        st.sidebar.write(" *" * 25)


        if analysis_method == "Analysis Home Page":
            self.analysis_homePage()

        
        if analysis_method == "Individual Asset Analysis":
            st.title("Single Asset Analysis")
            a1.Single_Asset_Analysis().run()



        if analysis_method == "CAPM & CAGR":
            st.title("Financial Calculations")
            st.sidebar.subheader("**[3] Select Ticker Category To Run**")
            options_f = ["Personal Portfolio", "Recommended Stocks", "Ticker Lists"]
            tickers = st.sidebar.radio("Model List:", options_f)
            st.sidebar.write("__" * 25)

            if tickers == "Personal Portfolio":
                st.sidebar.subheader("**[4] Select Stock List To Run**")
                personal_stocks = st.sidebar.text_input("Enter Stocks", value="AAPL NVDA TSLA ASML SNOW ALB")
                personal_stocks = personal_stocks.split()
                st.sidebar.write(" *" * 25)
                if st.sidebar.button("Run Analysis"):
                    self.capm_cagr(personal_stocks)

            if tickers == "Recommended Stocks":
                st.sidebar.subheader("**[4] Select Stock List Date To Run**")

                report_date = st.sidebar.date_input(
                label="> recommender date:",
                    value=date(2021, 7, 14),
                    min_value=date(2021, 7, 14),
                    max_value=date.today(),
                    key="date to run proof",
                    help="Select a date in the range between 2021.07.15 - 2021.08.26. \
                        This date will be the date the recommender model was run and we \
                            will use the resulting tickers for our proof",
                )
                name_lst = st.sidebar.radio(
                    label="", 
                    options=(
                        'markowitz_random',
                        'minimum_volatility_portfolio',    
                        'maximum_sharpe_ratio',
                        'minimum_volatility_random',
                        'maximum_sharpe_equalWT',
                        'monte_carlo_cholesky',
                    )
                )
                recommended_stocks = f0.recommended_stocks_2(name_lst, report_date)                
                st.sidebar.write(" *" * 25)
                if st.sidebar.button("Run Analysis"):
                    self.capm_cagr(recommended_stocks)

            if tickers == "Ticker Lists":
                st.sidebar.subheader("**[4] Select Stock List To Run**")

                screener = st.sidebar.radio("Choose A Set Of Tickers To Model", l0.stock_name_list)
                if st.sidebar.button(f"Run Analysis [{screener}]"):                 

                    if screener == "DOW_Symbols":
                        ticker_list = list(si.tickers_dow())
                        self.capm_cagr(ticker_list)

                    if screener == "S&P100_Symbols":
                        ticker_list = list(si.tickers_dow())
                        self.capm_cagr(ticker_list)    

                    if screener == "S&P500_Symbols":
                        ticker_list = list(si.tickers_dow())
                        self.capm_cagr(ticker_list)                                                                        


                    if screener == "Day_Gainer_Stocks":
                        try:
                            st.write("** Today's Gainers **")
                            self.dataFile = si.get_day_gainers().round(2)
                            st.dataframe(self.dataFile)
                            ticker_list = list(self.dataFile['Symbol'])
                            self.capm_cagr(ticker_list)
                        except Exception:
                            pass

                    elif screener == "Day_Loser_Stocks":
                        try:
                            st.write("** Today's Losers **")
                            self.dataFile = si.get_day_losers().round(2)
                            st.dataframe(self.dataFile)
                            ticker_list = list(self.dataFile['Symbol'])
                            self.capm_cagr(ticker_list)
                        except Exception:
                            pass

                    elif screener == "Most_Active_Stocks":
                        try:
                            st.write("** Today's Most Active **")
                            self.dataFile = si.get_day_most_active().round(2)
                            st.dataframe(self.dataFile)
                            ticker_list = list(self.dataFile['Symbol'])
                            self.capm_cagr(ticker_list)
                        except Exception:
                            pass

                    elif screener == "Trending_Tickers":
                        try:
                            st.write("** Today's Trending Tickers **")
                            self.dataFile = si.get_trending_tickers().round(2)
                            st.dataframe(self.dataFile)
                            ticker_list = list(self.dataFile['Symbol'])
                            self.capm_cagr(ticker_list)
                        except Exception:
                            pass

                    elif screener == "Most_Shorted_Stocks":
                        try:
                            st.write("** Today's Most Shorted Stocks **")
                            self.dataFile = si.get_most_shorted_stocks().round(2)
                            st.dataframe(self.dataFile)
                            ticker_list = list(self.dataFile['Symbol'])
                            self.capm_cagr(ticker_list)                            
                        except Exception:
                            pass

                    elif screener == "Undervalued_Large_Cap_Stocks":
                        try:
                            st.write("** Undervalued Large Cap Stocks **")
                            self.dataFile = si.get_undervalued_large_caps().round(2)
                            st.dataframe(self.dataFile)
                            ticker_list = list(self.dataFile['Symbol'])
                            self.capm_cagr(ticker_list)
                        except Exception:
                            pass

                    elif screener == "Undervalued_Growth_Stocks":
                        try:
                            st.write("** Undervalued Growth Stocks **")
                            self.dataFile = si.get_undervalued_growth_stocks().round(2)
                            st.dataframe(self.dataFile)
                            ticker_list = list(self.dataFile['Symbol'])
                            self.capm_cagr(ticker_list)
                        except Exception:
                            pass

                    elif screener == "Growth_Technology_Stocks":
                        try:
                            st.write("** Growth Technology Stocks **")
                            self.dataFile = si.get_growth_technology_stocks().round(2)
                            st.dataframe(self.dataFile)
                            ticker_list = list(self.dataFile['Symbol'])
                            self.capm_cagr(ticker_list)
                        except Exception:
                            pass

                    elif screener == "Aggressive_Small_Cap_Stocks":
                        try:
                            st.write("** Aggressive Small Cap Stocks **")
                            self.dataFile = si.get_aggressive_small_caps().round(2)
                            st.dataframe(self.dataFile)
                            ticker_list = list(self.dataFile['Symbol'])
                            self.capm_cagr(ticker_list)
                        except Exception:
                            pass

                    elif screener == "Small_Cap_Gainer_Stocks":
                        try:
                            st.write("** Small Cap Gainer Stocks **")
                            self.dataFile = si.get_small_cap_gainers().round(2)
                            st.dataframe(self.dataFile)
                            ticker_list = list(self.dataFile['Symbol'])
                            self.capm_cagr(ticker_list)
                        except Exception:
                            pass

                    elif screener == "Top_Crypto_Securities":
                        try:
                            st.write("** Top Crypto Assets **")
                            self.dataFile = SI.get_top_crypto()
                            st.dataframe(self.dataFile)
                            ticker_list = list(self.dataFile['Symbol'])
                            self.capm_cagr(ticker_list)
                        except Exception:
                            pass

                    elif screener == "Top_Mutual_Funds":
                        try:
                            st.write("** Top Mutual Funds **")
                            self.dataFile = si.get_top_mutual_funds().round(2)
                            st.dataframe(self.dataFile)
                            ticker_list = list(self.dataFile['Symbol'])
                            self.capm_cagr(ticker_list)
                        except Exception:
                            pass

                    elif screener == "Portfolio_Anchor_Securities":
                        try:
                            st.write("** Portfolio Anchors **")
                            self.dataFile = si.get_portfolio_anchors().round(2)
                            st.dataframe(self.dataFile)
                            ticker_list = list(self.dataFile['Symbol'])
                            self.capm_cagr(ticker_list)
                        except Exception:
                            pass

                    elif screener == "Solid_Large_Cap_Growth_Funds":
                        try:
                            st.write("** Solid Large-Cap Growth Funds **")
                            self.dataFile = si.get_solid_large_growth_funds().round(2)
                            st.dataframe(self.dataFile)
                            ticker_list = list(self.dataFile['Symbol'])
                            self.capm_cagr(ticker_list)
                        except Exception:
                            pass

                    elif screener == "Solid_Mid_Cap_Growth_Funds":
                        try:
                            st.write("** Solid Mid-Cap Growth Funds **")
                            self.dataFile = si.get_solid_midcap_growth_funds().round(2)
                            st.dataframe(self.dataFile)
                            ticker_list = list(self.dataFile['Symbol'])
                            self.capm_cagr(ticker_list)
                        except Exception:
                            pass

                    elif screener == "Conservative_Foreign_Funds":
                        try:
                            st.write("** Conservative Foreign Funds **")
                            self.dataFile = si.get_conservative_foreign_funds().round(2)
                            st.dataframe(self.dataFile)
                            ticker_list = list(self.dataFile['Symbol'])
                            self.capm_cagr(ticker_list)
                        except Exception:
                            pass

                    elif screener == "High_Yield_Bond_Funds":
                        try:
                            st.write("** High Yield Bond funds **")
                            self.dataFile = si.get_high_yield_bond().round(2)
                            st.dataframe(self.dataFile)
                            ticker_list = list(self.dataFile['Symbol'])
                            self.capm_cagr(ticker_list)                            
                        except Exception:
                            pass



        if analysis_method == "Financial Signal Processing":
            st.title("**Financial Signal Processing**")
            st.write("__" * 25)
            st.sidebar.header("[Step # 3]")
            ticker = st.sidebar.text_input(
                "Enter Stock Ticker For Analysis", 
                value="TSLA"
            )
            st.sidebar.write(" *" * 25)
            if st.sidebar.button("Run Analysis"):
                self.run_signal_analysis(ticker)

        if analysis_method == "Fundamental Analysis":
            st.title("**Fundamental Analysis**")
            st.write("__" * 25)
            st.sidebar.header("[Step # 3]")
            ticker = st.sidebar.text_input(
                "Enter Stock Ticker For Analysis", 
                value="TSLA"
            )
            st.sidebar.write(" *" * 25)
            if st.sidebar.button("Run Analysis"):
                self.run_fundamental_analysis(ticker)

        if analysis_method == "Technical Analysis":
            st.title("**Technical Analysis**")
            st.write("__" * 25)
            st.sidebar.header("[Step # 3]")
            ticker = st.sidebar.text_input(
                "Enter Stock Ticker For Analysis", 
                value="TSLA"
            )
            st.sidebar.write(" *" * 25)
            if st.sidebar.button("Run Analysis"):
                self.run_technical_analysis(ticker)
