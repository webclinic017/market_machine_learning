from yahooquery import Ticker
from os.path import exists
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
from pathlib import Path
import streamlit as st
import yfinance as yf
import pickle5 as pickle 
import time
import yfinance as yf


from src.tools.functions import company_longName

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
pd.options.display.float_format = "{:,}".format





class Proof_of_Concept_Viewer(object):


    def __init__(self, today_stamp, initial_cash=1000, save_output=True, graphit=True):
        self.save_output = save_output
        self.graphit = graphit
        self.initial_cash = initial_cash
        day = int(str(today_stamp)[8:10])
        month = int(str(today_stamp)[5:7])
        year = int(str(today_stamp)[:4])
        self.today_stamp = datetime(year, month, day) #+ timedelta(days=1)

        self.saveMonth = str(today_stamp)[:7]
        self.ender = str(datetime.now())[:10]
        self.saveRec = Path(f"data/recommenders/{str(self.today_stamp)[:4]}/{self.saveMonth}/{today_stamp}/")
        self.saveReport = Path(f"reports/portfolio/{self.saveMonth}/{today_stamp}/")

        self.final_loc = Path(f"reports/port_results/{self.saveMonth}/{today_stamp}/")
        if not self.final_loc.exists():
            self.final_loc.mkdir(parents=True)        

        self.saveAdvisor = Path(f"data/advisor/{self.saveMonth}/{today_stamp}/")
        if not self.saveAdvisor.exists():
            self.saveAdvisor.mkdir(parents=True)

        self.saveProof = Path(f"data/proof/{today_stamp}/{self.ender}/")
        if not self.saveProof.exists():
            self.saveProof.mkdir(parents=True)


    def performance(self, portfolio_file, namer):
        portfolio_file = pd.DataFrame(portfolio_file)
        divisor = len(portfolio_file["symbol"])
        total_allocation = portfolio_file["allocation"].sum() / 100
        port_tics = sorted(list(portfolio_file["symbol"]))

        y = Ticker(
            port_tics, 
            asynchronous=True,
            formatted=False, 
            backoff_factor=0.34, 
            validate=True
        ).history(
            start=self.today_stamp,
            interval='1d'
        ).reset_index().set_index('date')
        df = pd.DataFrame()
        for i in port_tics:
            z = pd.DataFrame(y[y['symbol'] == i]['adjclose'])
            df[i] = z
        df = df.round(2)

        proof = pd.DataFrame(portfolio_file[["symbol", "allocation"]])
        proof = proof.sort_values("symbol")
        b = []
        for i in proof["symbol"]:
            b.append(company_longName(i))
        proof["companyName"] = b
        proof["start_price"] = list(df.iloc[0])
        proof["Current_Price"] = list(df.iloc[-1])
        proof = proof.dropna()
        proof["initial_investment"] = round(self.initial_cash * (proof["allocation"] / 100), 2)
        proof["shares"] = round(proof["initial_investment"] / proof["start_price"], 2)
        proof["cash_now"] = round(proof["shares"] * proof["Current_Price"], 2)
        proof["return"] = round(((proof["cash_now"] - proof["initial_investment"]) / proof["initial_investment"]) * 100,2,)

        self.spy = yf.download("SPY", start=self.today_stamp)
        f = pd.DataFrame(self.spy.copy())
        f.to_pickle(self.saveAdvisor / "spy_hist.pkl")
        og_price = round(self.spy["Adj Close"][0], 2)
        new_price = round(self.spy["Adj Close"][-1], 2)
        proof_spy = pd.DataFrame(["SPY"], columns=["SPY"])
        proof_spy["start_price"] = og_price
        proof_spy["Current_Price"] = new_price
        proof_spy["initial_investment"] = round(self.initial_cash / len(proof_spy["SPY"]), 2)
        proof_spy["shares"] = round(proof_spy["initial_investment"] / proof_spy["start_price"], 2)
        proof_spy["cash_now"] = round(proof_spy["shares"] * proof_spy["Current_Price"], 2)
        proof_spy["return"] = round(((proof_spy["cash_now"] - proof_spy["initial_investment"]) / proof_spy["initial_investment"])* 100,2,)
        beat_num = proof_spy["return"][0]
        proof_2 = proof[proof["return"] > 0.0]
        proof_3 = proof_2[proof_2["return"] > beat_num]
        winning_percentage = round((len(proof_2["symbol"]) / divisor) * 100, 2)
        beat_spy_percentage = round((len(proof_3["symbol"]) / divisor), 2)

        one = pd.DataFrame(df.copy())
        shares = []
        allocation_lst = list(proof["allocation"])
        for k, v in enumerate(list(proof["symbol"])):
            shares.append(((allocation_lst[k] / 100) * self.initial_cash) / one[v].iloc[0])
        for k, v in enumerate(list(proof["symbol"])):
            one[v] = one[v] * shares[k]
        lst = list(proof["symbol"])
        one["portfolio"] = one[lst].sum(axis=1)
        eno = one.reset_index()
        
        start_cash = round(proof["initial_investment"].sum(), 2)
        avg_1 = round(one["portfolio"].mean(), 2)
        high_1 = round(one["portfolio"].max(), 2)
        low_1 = round(one["portfolio"].min(), 2)
        mean_watermark = round(((avg_1 - start_cash) / start_cash) * 100, 2)
        high_watermark = round(((high_1 - start_cash) / start_cash) * 100, 2)
        low_watermark = round(((low_1 - start_cash) / start_cash) * 100, 2)

        mean_watermark_spy = round(proof_spy["return"].mean(), 2)
        high_watermark_spy = round(proof_spy["return"].max(), 2)
        low_watermark_spy = round(proof_spy["return"].min(), 2)

        beat_num = proof_spy["return"][0]
        proof_2 = proof[proof["return"] > 0.0]
        proof_3 = proof_2[proof_2["return"] > beat_num]                  

        x1 = eno[eno["portfolio"] == eno["portfolio"].max()]["date"]
        y1 = one["portfolio"].max()
        x2 = eno[eno["portfolio"] == eno["portfolio"].min()]["date"]
        y2 = one["portfolio"].min()

        for i in list(one["portfolio"]):
            if float(i) > high_1:
                high_1 = float(i)
            else:
                pass  

        one["up_from_start"] = round(((one["portfolio"] - start_cash) / start_cash) * 100, 2)
        one["down_from_top"] = round(((one["portfolio"] - high_1) / high_1) * 100, 2)
        act_ror = round(((list(one["portfolio"])[-1] - list(one["portfolio"])[0])/ list(one["portfolio"])[0])* 100,2,)
        gdp = pd.DataFrame(["Recommended Stocks", "SPY Index"], columns=["strategy_vs_benchmark"])


        gdp["starting_money"] = [
            f"${round(list(one['portfolio'])[0],2)}",
            f"${round(proof_spy['initial_investment'].sum(),2)}",
        ]
        gdp["ending_money"] = [
            f"${round(list(one['portfolio'])[-1],2)}",
            f"${round(proof_spy['cash_now'].sum(), 2)}",
        ]
        gdp["return"] = [
            f"{round(act_ror,2)}%",
            f"{round(float(proof_spy['return']),2)}%",
        ]
        gdp["mean_mark"] = [
            f"{mean_watermark}%",
            f"{mean_watermark_spy}%",
        ]        
        gdp["high_mark"] = [
            f"{high_watermark}%",
            f"{high_watermark_spy}%",
        ]
        gdp["low_mark"] = [
            f"{low_watermark}%",
            f"{low_watermark_spy}%",
        ]                        
        gdp = gdp.set_index("strategy_vs_benchmark")


        st.caption(f"{'__'*25}\n{'__'*25}")
        st.header(f"> __[{namer} vs SPY]__")
        st.write(f" - Start Position [{self.today_stamp}] ")
        st.write(f" - Today's Position [{str(datetime.now())[:10]}] ")
        st.write(f"Total Allocation == {total_allocation}%")
        st.table(gdp)

        st.write(f" - __Proof returns__")
        st.write(f" - {namer}")
        st.write(f" - Winning Stock Picks [Positive Return] = {len(proof_2['symbol'])}/{divisor}, [{winning_percentage}%] ")
        st.write(f" - Stocks Outperforming The SPY  = {len(proof_3['symbol'])}/{divisor}, [{beat_spy_percentage}%   ]")
        st.write(f" - __Initial Portfolio Optimization Modeled On {self.today_stamp}__")

        proof = proof.sort_values("return", ascending=False)
        proof["rank"] = proof["return"].rank(ascending=False)
        st.table(proof.set_index(["rank", "companyName", "symbol"]))

        zero = pd.DataFrame(one[one['up_from_start'] >= 10.0])
        if self.graphit:
            fig, ax = plt.subplots()
            ax.plot(one["portfolio"], color='black', lw=1, marker='.', ms=10, label='Portfolio')
            ax.plot(zero['portfolio'], color='blue', lw=1, marker='.', ms=10, label='Up10')
            ax.axhline(start_cash, color='black', lw=1)
            ax.axhline(start_cash * 1.1, color='green', lw=1)
            ax.axhline(start_cash * 0.9, color='red', lw=1)
            try:
                ax.plot(x1, y1, color="green", marker="*", ms=25, label="High Watermark")
                ax.plot(x2, y2, color="red", marker="X", ms=20, label="low Watermark")
            except:
                print("")
            plt.legend()
            st.subheader("__Portfolio Balance History__")
            st.pyplot(fig)

        st.write(f" * __HIGH WATERMARK:__ ${high_1} __[{round(((high_1 - start_cash) / start_cash) * 100, 2)}%]__")
        st.write(f" * __LOW WATERMARK:__ ${low_1} __[{round(((low_1 - start_cash) / start_cash) * 100, 2)}%]__")
        st.caption(f"{'__'*25}\n{'__'*25}")


        if self.save_output == True:
            gdp = pd.DataFrame(gdp)
            proof = pd.DataFrame(proof)
            proof_spy = pd.DataFrame(proof_spy)

            gdp.to_csv(self.final_loc / f"spy_vs_{namer}.csv")
            proof.to_csv(self.final_loc / f"{namer}.csv")
            proof_spy.to_csv(self.final_loc / f"spy.csv")

            @st.cache
            def convert_df(df):
                return df.to_csv().encode('utf-8')

            csv = convert_df(proof)
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name=f"{str(self.final_loc)}/{str(namer)}.csv",
                mime='text/csv',
                key=str(namer),
            )
            return 


    def setup(self, portfolio_option):
        if 'maximum_sharpe' in portfolio_option:
            maximum_sharpe_ratio = pd.read_csv(self.final_loc / "maximum_sharpe_ratio.csv").set_index("rank")
            self.performance(maximum_sharpe_ratio[["symbol", "allocation"]], "maximum_sharpe_ratio")
            time.sleep(1.3)

        if 'minimum_volatility' in portfolio_option:
            minimum_volatility_portfolio = pd.read_csv(self.final_loc / "minimum_volatility_portfolio.csv").set_index("rank")
            self.performance(minimum_volatility_portfolio[["symbol", "allocation"]],"minimum_volatility_portfolio",)
            time.sleep(1.3)

        if 'equalWT' in portfolio_option:
            maximum_sharpe_equalWT = pd.read_csv(self.final_loc / "maximum_sharpe_equalWT.csv").set_index("rank")
            self.performance(maximum_sharpe_equalWT[["symbol", "allocation"]], "maximum_sharpe_equalWT")
            time.sleep(1.3)        

        if 'monteCarloC' in portfolio_option:
            monte_carlo_cholesky = pd.read_csv(self.final_loc / "monte_carlo_cholesky.csv").set_index("rank")
            self.performance(monte_carlo_cholesky[["symbol", "allocation"]], "monte_carlo_cholesky")
            time.sleep(1.3)

        print('\n', f">>> {self.today_stamp} <<<", '\n')
        return





class Proof_of_Concept(object):

    
    def __init__(self, today_stamp, ender_date=str(datetime.now())[:10], save_output=True, graphit=True):
        self.save_output = save_output
        self.graphit = graphit
        self.today_stamp = str(today_stamp)[:10]
        self.saveMonth = self.today_stamp[:7]

        self.final_loc = Path(f"reports/port_results/{self.saveMonth}/{self.today_stamp}/")
        if not self.final_loc.exists():
            self.final_loc.mkdir(parents=True)

        self.saveReport = Path(f"reports/portfolio/{self.saveMonth}/{self.today_stamp}/")
        if not self.saveReport.exists():
            self.saveReport.mkdir(parents=True)        

        self.saveAdvisor = Path(f"data/advisor/{self.saveMonth}/{self.today_stamp}/")
        if not self.saveAdvisor.exists():
            self.saveAdvisor.mkdir(parents=True)

        self.saveProof = Path(f"data/proof/{self.today_stamp}/{ender_date}/")
        if not self.saveProof.exists():
            self.saveProof.mkdir(parents=True)


    def setup(self, portfolio_file, namer, data, initial_cash=1000):
        portfolio_file = pd.DataFrame(portfolio_file).sort_values('symbol')
        self.namer = namer
        self.initial_cash = initial_cash
        divisor = len(portfolio_file["symbol"])
        total_allocation = portfolio_file["allocation"].sum() / 100
        port_tics = sorted(list(portfolio_file["symbol"]))
        df = pd.DataFrame(data).round(2)

        proof = pd.DataFrame(portfolio_file[["symbol", "allocation"]])
        proof = proof.sort_values("symbol")
        b = []
        for i in proof["symbol"]:
            b.append(company_longName(i))
        proof["companyName"] = b
        proof["start_price"] = list(df.iloc[0])
        proof["Current_Price"] = list(df.iloc[-1])
        proof = proof.dropna()
        proof["initial_investment"] = round(self.initial_cash * (proof["allocation"] / 100), 2)
        proof["shares"] = round(proof["initial_investment"] / proof["start_price"], 2)
        proof["cash_now"] = round(proof["shares"] * proof["Current_Price"], 2)
        proof["return"] = round(((proof["cash_now"] - proof["initial_investment"])/ proof["initial_investment"])* 100,2,)
        
        if exists(self.saveAdvisor / "spy_hist.pkl"):
            spy_hist = pd.read_pickle(self.saveAdvisor / "spy_hist.pkl")
        else:
            spy_hist = yf.download("SPY", start=self.today_stamp)
            f = pd.DataFrame(spy_hist.copy())
            f.to_pickle(self.saveAdvisor / "spy_hist.pkl")

        og_price = round(spy_hist["Adj Close"][0], 2)
        new_price = round(spy_hist["Adj Close"][-1], 2)            
        proof_spy = pd.DataFrame(["SPY"], columns=["SPY"])
        proof_spy["start_price"] = og_price
        proof_spy["Current_Price"] = new_price
        proof_spy["initial_investment"] = round(self.initial_cash / len(proof_spy["SPY"]), 2)
        proof_spy["shares"] = round(proof_spy["initial_investment"] / proof_spy["start_price"], 2)
        proof_spy["cash_now"] = round(proof_spy["shares"] * proof_spy["Current_Price"], 2)
        proof_spy["return"] = round(((proof_spy["cash_now"] - proof_spy["initial_investment"])/ proof_spy["initial_investment"])* 100,2,)

        high_watermark_spy = round(proof_spy["return"].max(), 2)
        low_watermark_spy = round(proof_spy["return"].min(), 2)

        beat_num = proof_spy["return"][0]
        proof_2 = proof[proof["return"] > 0.0]
        proof_3 = proof_2[proof_2["return"] > beat_num]
        winning_percentage = round((len(proof_2["symbol"]) / divisor) * 100, 2)
        beat_spy_percentage = round((len(proof_3["symbol"]) / divisor), 2)

        one = pd.DataFrame(df.copy())
        shares = []
        allocation_lst = list(proof["allocation"])
        for k, v in enumerate(list(proof["symbol"])):
            shares.append(((allocation_lst[k] / 100) * initial_cash) / one[v].iloc[0])
        for k, v in enumerate(list(proof["symbol"])):
            one[v] = one[v] * shares[k]
        lst = list(proof["symbol"])
        one["portfolio"] = one[lst].sum(axis=1)
        eno = one.reset_index()

        start_cash = round(proof["initial_investment"].sum(), 2)
        avg_1 = round(one["portfolio"].mean(), 2)
        high_1 = round(one["portfolio"].max(), 2)
        low_1 = round(one["portfolio"].min(), 2)
        mean_watermark = round(((avg_1 - start_cash) / start_cash) * 100, 2)
        high_watermark = round(((high_1 - start_cash) / start_cash) * 100, 2)
        low_watermark = round(((low_1 - start_cash) / start_cash) * 100, 2)

        mean_watermark_spy = round(proof_spy["return"].mean(), 2)
        high_watermark_spy = round(proof_spy["return"].max(), 2)
        low_watermark_spy = round(proof_spy["return"].min(), 2)

        beat_num = proof_spy["return"][0]
        proof_2 = proof[proof["return"] > 0.0]
        proof_3 = proof_2[proof_2["return"] > beat_num]                  

        x1 = eno[eno["portfolio"] == eno["portfolio"].max()]["date"]
        y1 = one["portfolio"].max()
        x2 = eno[eno["portfolio"] == eno["portfolio"].min()]["date"]
        y2 = one["portfolio"].min()

        for i in list(one["portfolio"]):
            if float(i) > high_1:
                high_1 = float(i)
            else:
                pass  

        one["up_from_start"] = round(((one["portfolio"] - start_cash) / start_cash) * 100, 2)
        one["down_from_top"] = round(((one["portfolio"] - high_1) / high_1) * 100, 2)
        act_ror = round(((list(one["portfolio"])[-1] - list(one["portfolio"])[0])/ list(one["portfolio"])[0])* 100,2,)
        gdp = pd.DataFrame(["Recommended Stocks", "SPY Index"], columns=["strategy_vs_benchmark"])
        
        gdp["starting_money"] = [
            f"${round(list(one['portfolio'])[0],2)}",
            f"${round(proof_spy['initial_investment'].sum(),2)}",
        ]
        gdp["ending_money"] = [
            f"${round(list(one['portfolio'])[-1],2)}",
            f"${round(proof_spy['cash_now'].sum(), 2)}",
        ]
        gdp["return"] = [
            f"{round(act_ror,2)}%",
            f"{round(float(proof_spy['return']),2)}%",
        ]
        gdp["mean_mark"] = [
            f"{mean_watermark}%",
            f"{mean_watermark_spy}%",
        ]        
        gdp["high_mark"] = [
            f"{high_watermark}%",
            f"{high_watermark_spy}%",
        ]
        gdp["low_mark"] = [
            f"{low_watermark}%",
            f"{low_watermark_spy}%",
        ]                        
        gdp = gdp.set_index("strategy_vs_benchmark")


        st.caption(f"{'__'*25}\n{'__'*25}")
        st.header(f"> __[{self.namer} vs SPY]__")
        st.write(f" - Start Position [{self.today_stamp}] ")
        st.write(f" - Today's Position [{str(datetime.now())[:10]}] ")
        st.write(f" - Total Allocation: {round(total_allocation*100,2)}%")
        st.table(gdp)

        st.write(f" - __Proof returns__")
        st.write(f" - {self.namer}")
        st.write(f" - Winning Stock Picks [Positive Return] = {len(proof_2['symbol'])}/{divisor}, [{winning_percentage}%] ")
        st.write(f" - Stocks Outperforming The SPY  = {len(proof_3['symbol'])}/{divisor}, [{beat_spy_percentage}%   ]")
        st.write(f" - __Initial Portfolio Optimization Modeled On {self.today_stamp}__")

        proof = proof.sort_values("return", ascending=False)
        proof["rank"] = proof["return"].rank(ascending=False)
        st.table(proof.set_index(["rank", "companyName", "symbol"]))

        zero = pd.DataFrame(one[one['up_from_start'] >= 10.0])

        if self.graphit:
            fig, ax = plt.subplots()
            ax.plot(one["portfolio"], color='black', lw=1, marker='.', ms=10, label='Portfolio')
            ax.plot(zero['portfolio'], color='blue', lw=1, marker='.', ms=10, label='Up10')
            ax.axhline(start_cash, color='black', lw=1)
            ax.axhline(start_cash * 1.1, color='green', lw=1)
            ax.axhline(start_cash * 0.9, color='red', lw=1)
            try:
                ax.plot(x1, y1, color="green", marker="*", ms=20, label="High Watermark")
                ax.plot(x2, y2, color="red", marker="X", ms=17, label="low Watermark")
            except:
                print("")
            plt.legend()
            st.subheader("__Portfolio Balance History__")
            st.pyplot(fig)

        st.write(f" * __HIGH WATERMARK:__ ${high_1} __[{round(((high_1 - start_cash) / start_cash) * 100, 2)}%]__")
        st.write(f" * __LOW WATERMARK:__ ${low_1} __[{round(((low_1 - start_cash) / start_cash) * 100, 2)}%]__")


        if self.save_output == True:
            gdp = pd.DataFrame(gdp)
            proof = pd.DataFrame(proof)
            proof_spy = pd.DataFrame(proof_spy)

            gdp.to_csv(self.final_loc / f"spy_vs_{self.namer}.csv")
            proof.to_csv(self.final_loc / f"{self.namer}.csv")
            proof_spy.to_csv(self.final_loc / f"spy.csv")

            @st.cache
            def convert_df(df):
                return df.to_csv().encode('utf-8')

            csv = convert_df(proof)
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name=f"{str(self.final_loc)}/{str(self.namer)}.csv",
                mime='text/csv',
                key=str(self.namer),
            )
            return 
            
        else:
            return 