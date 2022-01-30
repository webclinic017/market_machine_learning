import warnings
import os
from pathlib import Path
import pandas as pd
from datetime import datetime
import streamlit as st
from src.data import yahoo_fin_stock_info as si

# import pickle
import pickle5 as pickle


warnings.filterwarnings("ignore")
pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", -1)
pd.options.display.float_format = "{:,}".format
os.environ["NUMEXPR_MAX_THREADS"] = "24"
os.environ["NUMEXPR_NUM_THREADS"] = "12"


class Source_Data(object):
    def __init__(self, today_stamp=str(datetime.now())[:10]):
        self.list_of_screeners = []
        self.list_of_symbols = []
        self.names_of_screeners = [
            "Day_Gainer_Stocks",
            "Day_Loser_Stocks",
            "Most_Active_Stocks",
            "Trending_Tickers",
            "Most_Shorted_Stocks",
            "Undervalued_Large_Cap_Stocks",
            "Undervalued_Growth_Stocks",
            "Growth_Technology_Stocks",
            "Aggressive_Small_Cap_Stocks",
            "Small_Cap_Gainer_Stocks",
            "Top_Crypto_Securities",
            "Top_Mutual_Funds",
            "Portfolio_Anchor_Securities",
            "Solid_Large_Cap_Growth_Funds",
            "Solid_Mid_Cap_Growth_Funds",
            "Conservative_Foreign_Funds",
            "High_Yield_Bond_Funds",
        ]
        self.names_of_ticker_lists = [
            "DOW_Symbols",
            "S&P100_Symbols",
            "S&P400_Symbols",
            "S&P500_Symbols",
            "S&P600_Symbols",
            "NASDAQ_Symbols",
            "Finviz_Symbols",
            "Other_Symbols",
            "Fool_Symbols",
            "Oxford_Symbols",
            "Day_Gainer_Symbols",
            "Day_Losers_Symbols",
            "Day_Most_Active_Symbols",
            "Trending_Symbols",
            "MostShorted_Symbols",
            "Undervalued_Large_Cap_Symbols",
            "Undervalued_Growth_Symbols",
            "Growth_Technology_Symbols",
            "Aggressive_Small_Cap_Symbols",
            "Small_Cap_Gainers_Symbols",
            "Top_Crypto_Symbols",
            "Top_Mutual_Fund_Symbols",
            "Portfolio_Anchor_Symbols",
            "Solid_Growth_Funds_Symbols",
            "Solid_Mid_Cap_Growth_Funds_Symbols",
            "Conservative_Foreign_Funds_Symbols",
            "High_Yield_Bond_Symbols",
        ]

        self.today_stamp = today_stamp
        self.saveMonth = str(datetime.now())[:7]
        self.saveDay = str(datetime.now())[8:10]
        self.reportDate = str(datetime.now())[:10]

        self.saveScreeners = Path(
            f"data/screeners/{self.saveMonth}/{self.today_stamp}/"
        )
        if not self.saveScreeners.exists():
            self.saveScreeners.mkdir(parents=True)

        self.saveTickers = Path(f"data/tickers/{self.saveMonth}/{self.today_stamp}/")
        if not self.saveTickers.exists():
            self.saveTickers.mkdir(parents=True)

    def source_screeners(self):
        try:
            st.write("** Today's Gainers **")
            st.dataframe(si.get_day_gainers())
        except Exception:
            pass

        try:
            st.write("** Today's Losers **")
            st.dataframe(si.get_day_losers())
        except Exception:
            pass

        try:
            st.write("** Today's Most Active **")
            st.dataframe(si.get_day_most_active())
        except Exception:
            pass

        try:
            st.write("** Today's Trending Tickers **")
            st.dataframe(si.get_trending_tickers())
        except Exception:
            pass

        try:
            st.write("** Today's Most Shorted Stocks **")
            st.dataframe(si.get_most_shorted_stocks())
        except Exception:
            pass

        try:
            st.write("** Undervalued Large Cap Stocks **")
            st.dataframe(si.get_undervalued_large_caps())
        except Exception:
            pass

        try:
            st.write("** Undervalued Growth Stocks **")
            st.dataframe(si.get_undervalued_growth_stocks())
        except Exception:
            pass

        try:
            st.write("** Growth Technology Stocks **")
            st.dataframe(si.get_growth_technology_stocks())
        except Exception:
            pass

        try:
            st.write("** Aggressive Small Cap Stocks **")
            st.dataframe(si.get_aggressive_small_caps())
        except Exception:
            pass

        try:
            st.write("** Small Cap Gainer Stocks **")
            st.dataframe(si.get_small_cap_gainers())
        except Exception:
            pass

        try:
            st.write("** Top Crypto Assets **")
            st.dataframe(si.get_top_crypto())
        except Exception:
            pass

        try:
            st.write("** Top Mutual Funds **")
            st.dataframe(si.get_top_mutual_funds())
        except Exception:
            pass

        try:
            st.write("** Portfolio Anchors **")
            st.dataframe(si.get_portfolio_anchors())
        except Exception:
            pass

        try:
            st.write("** Solid Large-Cap Growth Funds **")
            st.dataframe(si.get_solid_large_growth_funds())
        except Exception:
            pass

        try:
            st.write("** Solid Mid-Cap Growth Funds **")
            st.dataframe(si.get_solid_midcap_growth_funds())
        except Exception:
            pass

        try:
            st.write("** Conservative Foreign Funds **")
            st.dataframe(si.get_conservative_foreign_funds())
        except Exception:
            pass

        try:
            st.write("** High Yield Bond funds **")
            st.dataframe(si.get_high_yield_bond())
        except Exception:
            pass

        return

    def save_screeners_and_tickers(self):
        try:
            self.tickers_dow = list(si.tickers_dow())
            self.tickers_sp100 = list(si.tickers_sp100())
            self.tickers_sp400 = list(si.tickers_sp400())
            self.tickers_sp500 = list(si.tickers_sp500())
            self.tickers_sp600 = list(si.tickers_sp600())
            self.tickers_nasdaq = list(si.tickers_nasdaq())
            # self.tickers_finviz = si.tickers_finviz()
            self.tickers_other = list(si.tickers_other())
            self.tickers_fool = list(si.tickers_fool())
            self.tickers_oxford = list(si.tickers_oxford())
        except Exception:
            pass

        self.list_of_ticker_lists = [
            self.tickers_dow,
            self.tickers_sp100,
            self.tickers_sp400,
            self.tickers_sp500,
            self.tickers_sp600,
            self.tickers_nasdaq,
            # self.tickers_finviz,
            self.tickers_other,
            self.tickers_fool,
            self.tickers_oxford,
        ]

        try:
            self.get_day_gainers = si.get_day_gainers()
            self.list_of_tickers.append(
                list(self.get_day_gainers["Symbol"]),
            )

            self.get_day_losers = si.get_day_losers()
            self.list_of_tickers.append(
                list(self.get_day_losers["Symbol"]),
            )

            self.get_day_most_active = si.get_day_most_active()
            self.list_of_tickers.append(
                list(self.get_day_most_active["Symbol"]),
            )

            self.get_trending_tickers = si.get_trending_tickers()
            self.list_of_tickers.append(
                list(self.get_trending_tickers["Symbol"]),
            )

            self.get_most_shorted_stocks = si.get_most_shorted_stocks()
            self.list_of_tickers.append(
                list(self.get_most_shorted_stocks["Symbol"]),
            )

            self.get_undervalued_large_caps = si.get_undervalued_large_caps()
            self.list_of_tickers.append(
                list(self.get_undervalued_large_caps["Symbol"]),
            )

            self.get_undervalued_growth_stocks = si.get_undervalued_growth_stocks()
            self.list_of_tickers.append(
                list(self.get_undervalued_growth_stocks["Symbol"]),
            )

            self.get_growth_technology_stocks = si.get_growth_technology_stocks()
            self.list_of_tickers.append(
                list(self.get_growth_technology_stocks["Symbol"]),
            )

            self.get_aggressive_small_caps = si.get_aggressive_small_caps()
            self.list_of_tickers.append(
                list(self.get_aggressive_small_caps["Symbol"]),
            )

            self.get_small_cap_gainers = si.get_small_cap_gainers()
            self.list_of_tickers.append(
                list(self.get_small_cap_gainers["Symbol"]),
            )

            self.get_top_crypto = si.get_top_crypto()
            self.list_of_tickers.append(
                list(self.get_top_crypto["Symbol"]),
            )

            self.get_top_mutual_funds = si.get_top_mutual_funds()
            self.list_of_tickers.append(
                list(self.get_top_mutual_funds["Symbol"]),
            )

            self.get_portfolio_anchors = si.get_portfolio_anchors()
            self.list_of_tickers.append(
                list(self.get_portfolio_anchors["Symbol"]),
            )

            self.get_solid_large_growth_funds = si.get_solid_large_growth_funds()
            self.list_of_tickers.append(
                list(self.get_solid_large_growth_funds["Symbol"]),
            )

            self.get_solid_midcap_growth_funds = si.get_solid_midcap_growth_funds()
            self.list_of_tickers.append(
                list(self.get_solid_midcap_growth_funds["Symbol"]),
            )

            self.get_conservative_foreign_funds = si.get_conservative_foreign_funds()
            self.list_of_tickers.append(
                list(self.get_conservative_foreign_funds["Symbol"]),
            )

            self.get_high_yield_bond = si.get_high_yield_bond()
            self.list_of_tickers.append(
                list(self.get_high_yield_bond["Symbol"]),
            )

            self.list_of_screeners.append(self.get_day_gainers)
            self.list_of_screeners.append(self.get_day_losers)
            self.list_of_screeners.append(self.get_day_most_active)
            self.list_of_screeners.append(self.get_trending_tickers)
            self.list_of_screeners.append(self.get_most_shorted_stocks)
            self.list_of_screeners.append(self.get_undervalued_large_caps)
            self.list_of_screeners.append(self.get_undervalued_growth_stocks)
            self.list_of_screeners.append(self.get_growth_technology_stocks)
            self.list_of_screeners.append(self.get_aggressive_small_caps)
            self.list_of_screeners.append(self.get_small_cap_gainers)
            self.list_of_screeners.append(self.get_top_crypto)
            self.list_of_screeners.append(self.get_top_mutual_funds)
            self.list_of_screeners.append(self.get_portfolio_anchors)
            self.list_of_screeners.append(self.get_solid_large_growth_funds)
            self.list_of_screeners.append(self.get_solid_midcap_growth_funds)
            self.list_of_screeners.append(self.get_conservative_foreign_funds)
            self.list_of_screeners.append(self.get_high_yield_bond)
        except Exception:
            pass

        for r in range(len(self.list_of_screeners)):
            df = pd.DataFrame(self.list_of_screeners[r])
            df.to_pickle(self.saveScreeners / f"{self.names_of_screeners[r]}.pkl")

        for lst in self.list_of_ticker_lists[:20]:
            [self.list_of_symbols.append(l) for l in lst]

        self.all_stock_list = list(set(self.list_of_symbols))
        fd = pd.DataFrame(self.all_stock_list, columns=["Symbol"]).sort_values("Symbol")
        fd.to_pickle(self.saveTickers / "list_of_all_symbols.pkl")

        for r in range(len(self.list_of_ticker_lists)):
            df = pd.DataFrame(self.list_of_ticker_lists[r], columns=["Symbol"])
            df.to_pickle(self.saveTickers / f"{self.names_of_ticker_lists[r]}.pkl")
            df.to_pickle(f"data/tickers/{self.names_of_ticker_lists[r]}.pkl")

        fd_01 = pd.DataFrame(self.names_of_screeners, columns=["Name"])
        fd_01.to_pickle(self.saveScreeners / "names_of_all_screeners.pkl")

        fd_02 = pd.DataFrame(self.names_of_ticker_lists, columns=["Name"])
        fd_02.to_pickle(self.saveTickers / "names_of_all_ticker_list.pkl")

        fd_03 = pd.DataFrame(self.names_of_ticker_lists[:20], columns=["Name"])
        fd_03.to_pickle(self.saveTickers / "names_of_ticker_list.pkl")


if __name__ == "__main__":
    today_stamp = str(datetime.now())[:10]
    Source_Data(today_stamp).save_screeners_and_tickers()
    print("DONE")
