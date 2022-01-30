import requests
import pandas as pd
import ftplib
import io
import re
import json
import datetime
from finvizfinance.screener.overview import Overview

try:
    from requests_html import HTMLSession
except Exception:
    print(
        """Warning - Certain functionality
                requires requests_html, which is not installed.
             Install using:
                pip install requests_html
             After installation, you may have to restart your Python session."""
    )
base_url = "https://query1.finance.yahoo.com/v8/finance/chart/"

#                               > TOOLS <   *


def build_url(ticker, start_date=None, end_date=None, interval="1d"):
    if end_date is None:
        end_seconds = int(pd.Timestamp("now").timestamp())
    else:
        end_seconds = int(pd.Timestamp(end_date).timestamp())
    if start_date is None:
        start_seconds = 7223400
    else:
        start_seconds = int(pd.Timestamp(start_date).timestamp())
    site = base_url + ticker
    params = {
        "period1": start_seconds,
        "period2": end_seconds,
        "interval": interval.lower(),
        "events": "div,splits",
    }
    return site, params


def force_float(elt):
    try:
        return float(elt)
    except Exception:
        return elt


def _convert_to_numeric(s):
    if "M" in s:
        s = s.strip("M")
        return force_float(s) * 1_000_000
    if "B" in s:
        s = s.strip("B")
        return force_float(s) * 1_000_000_000
    return force_float(s)


def get_data(ticker, start_date=None, end_date=None, index_as_date=True, interval="1d"):
    """Downloads historical stock price data into a pandas data frame.  Interval
    must be "1d", "1wk", "1mo", or "1m" for daily, weekly, monthly, or minute data.
    Intraday minute data is limited to 7 days.
    @param: ticker
    @param: start_date = None
    @param: end_date = None
    @param: index_as_date = True
    @param: interval = "1d"
    """
    if interval not in ("1d", "1wk", "1mo", "1m"):
        raise AssertionError("interval must be of of '1d', '1wk', '1mo', or '1m'")

    # build and connect to URL
    site, params = build_url(ticker, start_date, end_date, interval)
    resp = requests.get(site, params=params)
    if not resp.ok:
        raise AssertionError(resp.json())

    # get JSON response
    data = resp.json()

    # get open / high / low / close data
    frame = pd.DataFrame(data["chart"]["result"][0]["indicators"]["quote"][0])

    # get the date info
    temp_time = data["chart"]["result"][0]["timestamp"]
    if interval != "1m":
        # add in adjclose
        frame["adjclose"] = data["chart"]["result"][0]["indicators"]["adjclose"][0][
            "adjclose"
        ]
        frame.index = pd.to_datetime(temp_time, unit="s")
        frame.index = frame.index.map(lambda dt: dt.floor("d"))
        frame = frame[["open", "high", "low", "close", "adjclose", "volume"]]
    else:
        frame.index = pd.to_datetime(temp_time, unit="s")
        frame = frame[["open", "high", "low", "close", "volume"]]
    frame["ticker"] = ticker.upper()
    if not index_as_date:
        frame = frame.reset_index()
        frame.rename(columns={"index": "date"}, inplace=True)
    return frame


def get_quote_table(ticker, dict_result=True):
    """Scrapes data elements found on Yahoo Finance's quote page
    of input ticker
    @param: ticker
    @param: dict_result = True
    """
    site = "https://finance.yahoo.com/quote/" + ticker + "?p=" + ticker
    tables = pd.read_html(site)
    data = tables[0].append(tables[1])
    data.columns = ["attribute", "value"]
    quote_price = pd.DataFrame(["Quote Price", get_live_price(ticker)]).transpose()
    quote_price.columns = data.columns.copy()
    data = data.append(quote_price)
    data = data.sort_values("attribute")
    data = data.drop_duplicates().reset_index(drop=True)
    data["value"] = data.value.map(force_float)
    if dict_result:
        result = {key: val for key, val in zip(data.attribute, data.value)}
        return result
    return data


def get_stats(ticker):
    """Scrapes information from the statistics tab on Yahoo Finance
    for an input ticker
    @param: ticker
    """
    stats_site = (
        "https://finance.yahoo.com/quote/" + ticker + "/key-statistics?p=" + ticker
    )
    tables = pd.read_html(stats_site)
    tables = [table for table in tables[1:] if table.shape[1] == 2]
    table = tables[0]
    for elt in tables[1:]:
        table = table.append(elt)
    table.columns = ["Attribute", "Value"]
    table = table.reset_index(drop=True)
    return table


def get_stats_valuation(ticker):
    """Scrapes Valuation Measures table from the statistics tab on Yahoo Finance
    for an input ticker
    @param: ticker
    """
    stats_site = (
        "https://finance.yahoo.com/quote/" + ticker + "/key-statistics?p=" + ticker
    )
    tables = pd.read_html(stats_site)
    tables = [table for table in tables if "Trailing P/E" in table.iloc[:, 0].tolist()]
    table = tables[0].reset_index(drop=True)
    return table


def _parse_json(url):
    html = requests.get(url=url).text
    json_str = (
        html.split("root.App.main =")[1].split("(this)")[0].split(";\n}")[0].strip()
    )
    data = json.loads(json_str)["context"]["dispatcher"]["stores"]["QuoteSummaryStore"]
    new_data = json.dumps(data).replace("{}", "null")
    new_data = re.sub(r"\{[\'|\"]raw[\'|\"]:(.*?),(.*?)\}", r"\1", new_data)
    json_info = json.loads(new_data)
    return json_info


def _parse_table(json_info):
    df = pd.DataFrame(json_info)
    del df["maxAge"]
    df.set_index("endDate", inplace=True)
    df.index = pd.to_datetime(df.index, unit="s")
    df = df.transpose()
    df.index.name = "Breakdown"
    return df


def _raw_get_daily_info(site):
    session = HTMLSession()
    resp = session.get(site)
    tables = pd.read_html(resp.html.raw_html)
    df = tables[0].copy()
    df.columns = tables[0].columns
    del df["52 Week Range"]
    df["% Change"] = df["% Change"].map(lambda x: float(x.strip("%+").replace(",", "")))
    return df


def get_top_crypto():
    """
    --->>> Gets the top 100 Cryptocurrencies by Market Cap
    """
    session = HTMLSession()
    resp = session.get("https://finance.yahoo.com/cryptocurrencies?offset=0&count=100")
    tables = pd.read_html(resp.html.raw_html)
    screener_df = pd.DataFrame(tables[0].copy()).sort_values(
        "% Change", ascending=False
    )
    screener_df["% Change"] = screener_df["% Change"].map(
        lambda x: float(x.strip("%").strip("+").replace(",", ""))
    )
    del screener_df["52 Week Range"]
    del screener_df["1 Day Chart"]
    del screener_df["Volume in Currency (Since 0:00 UTC)"]
    del screener_df["Volume in Currency (24Hr)"]
    session.close()
    screener_df = screener_df.set_index(["Name", "Symbol"])
    screener_df.columns = [
        "Price",
        "Change_$",
        "Change_%",
        "Market_Cap",
        "Total_24hr_Volume",
        "Circulating_Supply",
    ]
    return screener_df.sort_values(by="Change_%", ascending=False)


def get_day_gainers():
    screener_df = pd.DataFrame(
        _raw_get_daily_info("https://finance.yahoo.com/gainers?offset=0&count=100")
    ).sort_values("% Change", ascending=False)
    rank = []
    [rank.append(x) for x in range(1, len(screener_df) + 1)]
    screener_df["ranking"] = rank
    screener_df = screener_df.set_index("ranking")
    screener_df = screener_df.drop("PE Ratio (TTM)", axis=1)
    return screener_df


def get_day_losers():
    screener_df = pd.DataFrame(
        _raw_get_daily_info("https://finance.yahoo.com/losers?offset=0&count=100")
    ).sort_values("% Change", ascending=False)
    rank = []
    [rank.append(x) for x in range(1, len(screener_df) + 1)]
    screener_df["ranking"] = rank
    screener_df = screener_df.set_index("ranking")
    screener_df = screener_df.drop("PE Ratio (TTM)", axis=1)
    return screener_df


def get_day_most_active():
    screener_df = pd.DataFrame(
        _raw_get_daily_info("https://finance.yahoo.com/most-active?offset=0&count=100")
    ).sort_values("% Change", ascending=False)
    rank = []
    [rank.append(x) for x in range(1, len(screener_df) + 1)]
    screener_df["ranking"] = rank
    screener_df = screener_df.set_index("ranking")
    screener_df = screener_df.drop("PE Ratio (TTM)", axis=1)
    return screener_df


def get_trending_tickers():
    screener_df = pd.DataFrame(
        _raw_get_daily_info(
            "https://finance.yahoo.com/trending-tickers?offset=0&count=100"
        )
    ).sort_values("% Change", ascending=False)
    rank = []
    [rank.append(x) for x in range(1, len(screener_df) + 1)]
    screener_df["ranking"] = rank
    screener_df = pd.DataFrame(screener_df).set_index("ranking")
    screener_df = screener_df.drop(["Intraday High/Low", "Day Chart"], axis=1)
    return screener_df


def get_most_shorted_stocks():
    screener_df = pd.DataFrame(
        _raw_get_daily_info(
            "https://finance.yahoo.com/screener/predefined/most_shorted_stocks?offset=0&count=100"
        )
    ).sort_values("% Change", ascending=False)
    rank = []
    [rank.append(x) for x in range(1, len(screener_df) + 1)]
    screener_df["ranking"] = rank
    screener_df = screener_df.set_index("ranking")
    screener_df = screener_df.drop("PE Ratio (TTM)", axis=1)
    return screener_df


def get_undervalued_large_caps():
    # Returns the undervalued large caps table from Yahoo Finance
    screener_df = pd.DataFrame(
        _raw_get_daily_info(
            "https://finance.yahoo.com/screener/predefined/undervalued_large_caps?offset=0&count=100"
        )
    ).sort_values("% Change", ascending=False)
    rank = []
    [rank.append(x) for x in range(1, len(screener_df) + 1)]
    screener_df["ranking"] = rank
    screener_df = screener_df.set_index("ranking")
    screener_df = screener_df.drop("PE Ratio (TTM)", axis=1)
    return screener_df


def get_undervalued_growth_stocks():
    screener_df = pd.DataFrame(
        _raw_get_daily_info(
            "https://finance.yahoo.com/screener/predefined/undervalued_growth_stocks?offset=0&count=100"
        )
    ).sort_values("% Change", ascending=False)
    rank = []
    [rank.append(x) for x in range(1, len(screener_df) + 1)]
    screener_df["ranking"] = rank
    screener_df = screener_df.set_index("ranking")
    screener_df = screener_df.drop("PE Ratio (TTM)", axis=1)
    return screener_df


def get_growth_technology_stocks():
    screener_df = pd.DataFrame(
        _raw_get_daily_info(
            "https://finance.yahoo.com/screener/predefined/growth_technology_stocks?offset=0&count=100"
        )
    ).sort_values("% Change", ascending=False)
    rank = []
    [rank.append(x) for x in range(1, len(screener_df) + 1)]
    screener_df["ranking"] = rank
    screener_df = screener_df.set_index("ranking")
    screener_df = screener_df.drop("PE Ratio (TTM)", axis=1)
    return screener_df


def get_aggressive_small_caps():
    screener_df = pd.DataFrame(
        _raw_get_daily_info(
            "https://finance.yahoo.com/screener/predefined/aggressive_small_caps?offset=0&count=100"
        )
    ).sort_values("% Change", ascending=False)
    rank = []
    [rank.append(x) for x in range(1, len(screener_df) + 1)]
    screener_df["ranking"] = rank
    screener_df = screener_df.set_index("ranking")
    screener_df = screener_df.drop("PE Ratio (TTM)", axis=1)
    return screener_df


def get_small_cap_gainers():
    screener_df = pd.DataFrame(
        _raw_get_daily_info(
            "https://finance.yahoo.com/screener/predefined/small_cap_gainers?offset=0&count=100"
        )
    ).sort_values("% Change", ascending=False)
    rank = []
    [rank.append(x) for x in range(1, len(screener_df) + 1)]
    screener_df["ranking"] = rank
    screener_df = screener_df.set_index("ranking")
    screener_df = screener_df.drop("PE Ratio (TTM)", axis=1)
    return screener_df


def get_top_mutual_funds():
    screener_df = pd.DataFrame(
        _raw_get_daily_info(
            "https://finance.yahoo.com/screener/predefined/top_mutual_funds?offset=0&count=100"
        )
    ).sort_values("% Change", ascending=False)
    rank = []
    [rank.append(x) for x in range(1, len(screener_df) + 1)]
    screener_df["ranking"] = rank
    screener_df = screener_df.set_index("ranking")
    return screener_df


def get_portfolio_anchors():
    screener_df = pd.DataFrame(
        _raw_get_daily_info(
            "https://finance.yahoo.com/screener/predefined/portfolio_anchors?offset=0&count=100"
        )
    ).sort_values("% Change", ascending=False)
    rank = []
    [rank.append(x) for x in range(1, len(screener_df) + 1)]
    screener_df["ranking"] = rank
    screener_df = screener_df.set_index("ranking")
    return screener_df


def get_solid_large_growth_funds():
    screener_df = pd.DataFrame(
        _raw_get_daily_info(
            "https://finance.yahoo.com/screener/predefined/solid_large_growth_funds?offset=0&count=100"
        )
    ).sort_values("% Change", ascending=False)
    rank = []
    [rank.append(x) for x in range(1, len(screener_df) + 1)]
    screener_df["ranking"] = rank
    screener_df = screener_df.set_index("ranking")
    return screener_df


def get_solid_midcap_growth_funds():
    screener_df = pd.DataFrame(
        _raw_get_daily_info(
            "https://finance.yahoo.com/screener/predefined/solid_midcap_growth_funds?offset=0&count=100"
        )
    ).sort_values("% Change", ascending=False)
    rank = []
    [rank.append(x) for x in range(1, len(screener_df) + 1)]
    screener_df["ranking"] = rank
    screener_df = screener_df.set_index("ranking")
    return screener_df


def get_conservative_foreign_funds():
    screener_df = pd.DataFrame(
        _raw_get_daily_info(
            "https://finance.yahoo.com/screener/predefined/conservative_foreign_funds?offset=0&count=100"
        )
    ).sort_values("% Change", ascending=False)
    rank = []
    [rank.append(x) for x in range(1, len(screener_df) + 1)]
    screener_df["ranking"] = rank
    screener_df = screener_df.set_index("ranking")
    return screener_df


def get_high_yield_bond():
    screener_df = pd.DataFrame(
        _raw_get_daily_info(
            "https://finance.yahoo.com/screener/predefined/high_yield_bond?offset=0&count=100"
        )
    ).sort_values("% Change", ascending=False)
    rank = []
    [rank.append(x) for x in range(1, len(screener_df) + 1)]
    screener_df["ranking"] = rank
    screener_df = screener_df.set_index("ranking")
    return screener_df


def tickers_dow(include_company_data=False):
    """Downloads list of currently traded tickers on the Dow"""
    site = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
    table = pd.read_html(site, attrs={"id": "constituents"})[0]
    if include_company_data:
        return table
    dow_tickers = sorted(table["Symbol"].tolist())
    dow_tickers = [x.replace(".", "-") for x in dow_tickers]
    return dow_tickers


def tickers_sp100(include_company_data=False):
    site = "https://en.wikipedia.org/wiki/S%26P_100"
    table = pd.read_html(site, attrs={"id": "constituents"})[0]
    if include_company_data:
        return table
    sp100_tickers = sorted(table["Symbol"].tolist())
    sp100_tickers = [x.replace(".", "-") for x in sp100_tickers]
    return sp100_tickers


def tickers_sp400(include_company_data=False):
    site = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
    table = pd.read_html(site, attrs={"id": "constituents"})[0]
    df = pd.DataFrame(table).sort_values("Ticker symbol")
    tickers_lst = list(df["Ticker symbol"])
    tickers_lst = [x.replace(".", "-") for x in tickers_lst]
    return tickers_lst  # , company_lst, table


def tickers_sp500(include_company_data=False):
    """Downloads list of tickers currently listed in the S&P 500"""
    sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    sp500["Symbol"] = sp500["Symbol"].str.replace(".", "-")
    if include_company_data:
        return sp500
    sp500_tickers = sp500.Symbol.tolist()
    sp500_tickers = sorted(sp500_tickers)
    return sp500_tickers


def tickers_sp600(include_company_data=False):
    site = "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"
    table = pd.read_html(site, attrs={"id": "constituents"})[0]
    df = pd.DataFrame(table).sort_values("Ticker symbol")
    tickers_lst = list(df["Ticker symbol"])
    tickers_lst = [x.replace(".", "-") for x in tickers_lst]
    return tickers_lst  # , company_lst, table


def clean(list):
    clean_lst = []
    for l in list:
        if l.isalpha() == True:
            clean_lst.append(l)
    return clean_lst


def tickers_nasdaq(include_company_data=False):
    """Downloads list of tickers currently listed in the NASDAQ"""
    ftp = ftplib.FTP("ftp.nasdaqtrader.com")
    ftp.login()
    ftp.cwd("SymbolDirectory")
    r = io.BytesIO()
    ftp.retrbinary("RETR nasdaqlisted.txt", r.write)
    if include_company_data:
        r.seek(0)
        data = pd.read_pickle(r, sep="|")
        return data
    info = r.getvalue().decode()
    splits = info.split("|")
    tickers = [x for x in splits if "\r\n" in x]
    tickers = [x.split("\r\n")[1] for x in tickers if "NASDAQ" not in x != "\r\n"]
    tickers = [ticker for ticker in tickers if "File" not in ticker]
    ftp.close()
    new_tickers = clean(tickers)
    return new_tickers


def tickers_other(include_company_data=False):
    """Downloads list of tickers currently listed in the "otherlisted.txt"
    file on "ftp.nasdaqtrader.com" """
    ftp = ftplib.FTP("ftp.nasdaqtrader.com")
    ftp.login()
    ftp.cwd("SymbolDirectory")
    r = io.BytesIO()
    ftp.retrbinary("RETR otherlisted.txt", r.write)
    if include_company_data:
        r.seek(0)
        data = pd.read_pickle(r, sep="|")
        return data
    info = r.getvalue().decode()
    splits = info.split("|")
    tickers = [x for x in splits if "\r\n" in x]
    tickers = [x.split("\r\n")[1] for x in tickers]
    tickers = [ticker for ticker in tickers if "File" not in ticker]
    ftp.close()
    new_tickers = clean(tickers)
    return new_tickers


foverview = Overview()


def tickers_finviz():
    signals = [
        "Major News",
        "Top Losers",
        "Wedge Up",
        "Multiple Top",
        "Downgrades",
        "Triangle Ascending",
        "Head & Shoulders",
        "Double Top",
        "Wedge",
        "Most Volatile",
        "Head & Shoulders Inverse",
        "Double Bottom",
        "Recent Insider Selling",
        "Overbought",
        "TL Resistance",
        "Multiple Bottom",
        "Upgrades",
        "Most Active",
        "Recent Insider Buying",
        "TL Support",
        "New Low",
        "Triangle Descending",
        "Channel",
        "Horizontal S/R",
        "Channel Down",
        "Wedge Down",
        "Channel Up",
        "New High",
        "Top Gainers",
        "Oversold",
        "Unusual Volume",
    ]
    signals_ticker_list = []
    for sig in signals:
        foverview.set_filter(signal=sig)
        df = foverview.ScreenerView()[:100]
        temp_lst = list(df["Ticker"])
        [signals_ticker_list.append(x) for x in temp_lst]
    signals_ticker_list = list(set(signals_ticker_list))
    new_tickers = clean(signals_ticker_list)
    return new_tickers


def tickers_fool():
    fool_composite = [
        "NET",
        "LTCN",
        "BILI",
        "MA",
        "SHOP",
        "LO",
        "PYPL",
        "NNOX",
        "MRNA",
        "W",
        "NEE",
        "LRCX",
        "SIVB",
        "SOHU",
        "ATVI",
        "BA",
        "DOCU",
        "TDOC",
        "PTON",
        "RDFN",
        "TTD",
        "LOGI",
        "NYT",
        "CHGG",
        "SAM",
        "BL",
        "FUBO",
        "Z",
        "CLLS",
        "U",
        "SNOW",
        "RIOT",
        "RMD",
        "ARKG",
        "ASML",
        "RGEN",
        "WING",
        "BLDP",
        "AXON",
        "ROKU",
        "PLTR",
        "AAPL",
        "ZM",
        "ZEN",
        "ABNB",
        "STAA",
        "SNBR",
        "LMND",
        "ZG",
        "OROCF",
        "TWLO",
        "GDRX",
        "FSLY",
        "AVAV",
        "PLNT",
        "TECK",
        "GWPH",
        "ETCG",
        "APHA",
        "TEAM",
        "COUP",
        "ARKK",
        "NTDOY",
        "PAYC",
        "MTCH",
        "SE",
        "ETSY",
        "UPST",
        "AMZN",
        "TSLA",
        "SQ",
        "PINS",
        "DIS",
        "SKLZ",
        "LULU",
        "CUE",
        "NVDA",
        "EQIX",
        "CRM",
        "CRLBF",
        "OKTA",
        "ECL",
        "FVRR",
        "DDOG",
        "GBTC",
        "HRVSF",
    ]
    return fool_composite


def tickers_oxford():
    oxford_composite = [
        "NET",
        "VWEHX, VSMAX",
        "CRNT",
        "SWKS",
        "EA",
        "MRNA",
        "MARA",
        "MGPHF",
        "FSLR",
        "GRUB",
        "FTNT",
        "SYY",
        "STOR",
        "GKOS",
        "EMF",
        "CNK",
        "YETI",
        "EXPE",
        "BABA",
        "MRK",
        "EXAS",
        "MCK",
        "BUD",
        "BYND",
        "GDOT",
        "BZUN",
        "TDF",
        "ENB",
        "VFSTX",
        "IRBT",
        "TTWO",
        "VGSLX",
        "CCI",
        "BE",
        "PFGC",
        "LUN-T",
        "INSG",
        "SYAAF",
        "VIPSX",
        "CROX",
        "BRKR",
        "GRMN",
        "ORCL",
        "SPG",
        "GS",
        "BZH",
        "POOL",
        "BIIB",
        "SAM",
        "LBTYA",
        "HA",
        "Z",
        "IEP",
        "LOPE",
        "CLLS",
        "VZ",
        "RIOT",
        "NGPHF",
        "KHRNF",
        "NVCR",
        "PCLOF",
        "NLOK",
        "TRIP",
        "MRVL",
        "F",
        "IIPR",
        "BLDP",
        "FAST",
        "EBAY",
        "DG",
        "FUN",
        "VEUSX",
        "EGHT",
        "CVS",
        "BYDDY",
        "MKL",
        "ZEN",
        "VEMAX",
        "ET",
        "NOK",
        "BRK-B",
        "ALB",
        "IBM",
        "EQR",
        "INCR-U",
        "WIX",
        "VTSAX",
        "BCO",
        "PFPT",
        "CX",
        "BRPHF",
        "PFE",
        "GDX",
        "COLM",
        "FCEL",
        "CMI",
        "LIT",
        "RUN",
        "IMPUY",
        "GRWG",
        "AMT",
        "SAIL",
        "AEIS",
        "DAL",
        "RDS-B",
        "DPZ",
        "CZZ",
        "CURLF",
        "SPWR",
        "AMC",
        "NUVA",
        "JLL",
        "VPADX",
        "DEM",
        "CAMP",
        "BLCM",
        "CSCO",
        "DLR",
        "LULU",
        "V",
        "VRTX",
        "AIG",
        "CUE",
        "NVDA",
        "LSCC",
        "ZEST",
        "NEO",
        "MU",
        "VRA",
        "LVGO",
        "CRLBF",
        "GTBIF",
        "WMT",
        "OKTA",
        "HAS",
        "SIX",
        "UBER",
        "HSY",
        "XLE",
        "NFE",
        "PSHZF",
        "DRNA",
        "GOLD",
        "AMKBY",
        "ERIC",
        "HUYA",
        "NFLX",
    ]
    return oxford_composite


def tickers_top_crypto():
    session = HTMLSession()
    resp = session.get("https://finance.yahoo.com/cryptocurrencies?offset=0&count=100")
    tables = pd.read_html(resp.html.raw_html)
    screener_df = pd.DataFrame(tables[0].copy())
    session.close()
    ticker_list = list(screener_df["Symbol"])
    ticker_list.sort()
    return ticker_list


def tickers_most_active():
    screener_df = pd.DataFrame(
        _raw_get_daily_info("https://finance.yahoo.com/most-active?offset=0&count=100")
    )
    ticker_list = list(screener_df["Symbol"])
    ticker_list.sort()
    return ticker_list


def tickers_day_gainers():
    screener_df = pd.DataFrame(
        _raw_get_daily_info("https://finance.yahoo.com/gainers?offset=0&count=100")
    )
    ticker_list = list(screener_df["Symbol"])
    ticker_list.sort()
    return ticker_list


def tickers_day_losers():
    screener_df = pd.DataFrame(
        _raw_get_daily_info("https://finance.yahoo.com/losers?offset=0&count=100")
    )
    ticker_list = list(screener_df["Symbol"])
    ticker_list.sort()
    return ticker_list


def tickers_trending_tickers():
    screener_df = pd.DataFrame(
        _raw_get_daily_info(
            "https://finance.yahoo.com/trending-tickers?offset=0&count=100"
        )
    )
    ticker_list = list(screener_df["Symbol"])
    ticker_list.sort()
    return ticker_list


def tickers_most_shorted_stocks():
    screener_df = pd.DataFrame(
        _raw_get_daily_info(
            "https://finance.yahoo.com/screener/predefined/most_shorted_stocks?offset=0&count=100"
        )
    )
    ticker_list = list(screener_df["Symbol"])
    ticker_list.sort()
    return ticker_list


def tickers_undervalued_large_caps():
    screener_df = pd.DataFrame(
        _raw_get_daily_info(
            "https://finance.yahoo.com/screener/predefined/undervalued_large_caps?offset=0&count=100"
        )
    )
    ticker_list = list(screener_df["Symbol"])
    ticker_list.sort()
    return ticker_list


def tickers_undervalued_growth_stocks():
    screener_df = pd.DataFrame(
        _raw_get_daily_info(
            "https://finance.yahoo.com/screener/predefined/undervalued_growth_stocks?offset=0&count=100"
        )
    )
    ticker_list = list(screener_df["Symbol"])
    ticker_list.sort()
    return ticker_list


def tickers_growth_technology_stocks():
    screener_df = pd.DataFrame(
        _raw_get_daily_info(
            "https://finance.yahoo.com/screener/predefined/growth_technology_stocks?offset=0&count=100"
        )
    )
    ticker_list = list(screener_df["Symbol"])
    ticker_list.sort()
    return ticker_list


def tickers_aggressive_small_caps():
    screener_df = pd.DataFrame(
        _raw_get_daily_info(
            "https://finance.yahoo.com/screener/predefined/aggressive_small_caps?offset=0&count=100"
        )
    )
    ticker_list = list(screener_df["Symbol"])
    ticker_list.sort()
    return ticker_list


def tickers_small_cap_gainers():
    screener_df = pd.DataFrame(
        _raw_get_daily_info(
            "https://finance.yahoo.com/screener/predefined/small_cap_gainers?offset=0&count=100"
        )
    )
    ticker_list = list(screener_df["Symbol"])
    ticker_list.sort()
    return ticker_list


def tickers_top_mutual_funds():
    screener_df = pd.DataFrame(
        _raw_get_daily_info(
            "https://finance.yahoo.com/screener/predefined/top_mutual_funds?offset=0&count=100"
        )
    )
    ticker_list = list(screener_df["Symbol"])
    ticker_list.sort()
    return ticker_list


def tickers_portfolio_anchors():
    screener_df = pd.DataFrame(
        _raw_get_daily_info(
            "https://finance.yahoo.com/screener/predefined/portfolio_anchors?offset=0&count=100"
        )
    )
    ticker_list = list(screener_df["Symbol"])
    ticker_list.sort()
    return ticker_list


def tickers_solid_large_growth_funds():
    screener_df = pd.DataFrame(
        _raw_get_daily_info(
            "https://finance.yahoo.com/screener/predefined/solid_large_growth_funds?offset=0&count=100"
        )
    )
    ticker_list = list(screener_df["Symbol"])
    ticker_list.sort()
    return ticker_list


def tickers_solid_midcap_growth_funds():
    screener_df = pd.DataFrame(
        _raw_get_daily_info(
            "https://finance.yahoo.com/screener/predefined/solid_midcap_growth_funds?offset=0&count=100"
        )
    )
    ticker_list = list(screener_df["Symbol"])
    ticker_list.sort()
    return ticker_list


def tickers_conservative_foreign_funds():
    screener_df = pd.DataFrame(
        _raw_get_daily_info(
            "https://finance.yahoo.com/screener/predefined/conservative_foreign_funds?offset=0&count=100"
        )
    )
    ticker_list = list(screener_df["Symbol"])
    ticker_list.sort()
    return ticker_list


def tickers_high_yield_bond():
    screener_df = pd.DataFrame(
        _raw_get_daily_info(
            "https://finance.yahoo.com/screener/predefined/high_yield_bond?offset=0&count=100"
        )
    )
    ticker_list = list(screener_df["Symbol"])
    ticker_list.sort()
    return ticker_list


def get_income_statement(ticker, yearly=True):
    """Scrape income statement from Yahoo Finance for a given ticker
    @param: ticker
    """
    income_site = (
        "https://finance.yahoo.com/quote/" + ticker + "/financials?p=" + ticker
    )
    json_info = _parse_json(income_site)
    if yearly:
        temp = json_info["incomeStatementHistory"]["incomeStatementHistory"]
    else:
        temp = json_info["incomeStatementHistoryQuarterly"]["incomeStatementHistory"]
    return _parse_table(temp)


def get_balance_sheet(ticker, yearly=True):
    """Scrapes balance sheet from Yahoo Finance for an input ticker
    @param: ticker
    """
    balance_sheet_site = (
        "https://finance.yahoo.com/quote/" + ticker + "/balance-sheet?p=" + ticker
    )
    json_info = _parse_json(balance_sheet_site)
    if yearly:
        temp = json_info["balanceSheetHistory"]["balanceSheetStatements"]
    else:
        temp = json_info["balanceSheetHistoryQuarterly"]["balanceSheetStatements"]
    return _parse_table(temp)


def get_cash_flow(ticker, yearly=True):
    """Scrapes the cash flow statement from Yahoo Finance for an input ticker
    @param: ticker
    """
    cash_flow_site = (
        "https://finance.yahoo.com/quote/" + ticker + "/cash-flow?p=" + ticker
    )
    json_info = _parse_json(cash_flow_site)
    if yearly:
        temp = json_info["cashflowStatementHistory"]["cashflowStatements"]
    else:
        temp = json_info["cashflowStatementHistoryQuarterly"]["cashflowStatements"]
    return _parse_table(temp)


def get_financials(ticker, yearly=True, quarterly=True):
    """Scrapes financials data from Yahoo Finance for an input ticker, including
    balance sheet, cash flow statement, and income statement.  Returns dictionary
    of results.
    @param: ticker
    @param: yearly = True
    @param: quarterly = True
    """
    if not yearly and not quarterly:
        raise AssertionError("yearly or quarterly must be True")
    financials_site = (
        "https://finance.yahoo.com/quote/" + ticker + "/financials?p=" + ticker
    )
    json_info = _parse_json(financials_site)
    result = {}
    if yearly:
        temp = json_info["incomeStatementHistory"]["incomeStatementHistory"]
        table = _parse_table(temp)
        result["yearly_income_statement"] = table
        temp = json_info["balanceSheetHistory"]["balanceSheetStatements"]
        table = _parse_table(temp)
        result["yearly_balance_sheet"] = table
        temp = json_info["cashflowStatementHistory"]["cashflowStatements"]
        table = _parse_table(temp)
        result["yearly_cash_flow"] = table
    if quarterly:
        temp = json_info["incomeStatementHistoryQuarterly"]["incomeStatementHistory"]
        table = _parse_table(temp)
        result["quarterly_income_statement"] = table
        temp = json_info["balanceSheetHistoryQuarterly"]["balanceSheetStatements"]
        table = _parse_table(temp)
        result["quarterly_balance_sheet"] = table
        temp = json_info["cashflowStatementHistoryQuarterly"]["cashflowStatements"]
        table = _parse_table(temp)
        result["quarterly_cash_flow"] = table
    return result


def get_holders(ticker):
    """Scrapes the Holders page from Yahoo Finance for an input ticker
    @param: ticker
    """
    holders_site = "https://finance.yahoo.com/quote/" + ticker + "/holders?p=" + ticker
    tables = pd.read_html(holders_site, header=0)
    table_names = [
        "Major Holders",
        "Direct Holders (Forms 3 and 4)",
        "Top Institutional Holders",
        "Top Mutual Fund Holders",
    ]
    table_mapper = {key: val for key, val in zip(table_names, tables)}
    return table_mapper


def get_analysts_info(ticker):
    """Scrapes the Analysts page from Yahoo Finance for an input ticker
    @param: ticker
    """
    analysts_site = (
        "https://finance.yahoo.com/quote/" + ticker + "/analysts?p=" + ticker
    )
    tables = pd.read_html(analysts_site, header=0)
    table_names = [table.columns[0] for table in tables]
    table_mapper = {key: val for key, val in zip(table_names, tables)}
    return table_mapper


def get_live_price(ticker):
    """Gets the live price of input ticker
    @param: ticker
    """
    df = get_data(ticker, end_date=pd.Timestamp.today() + pd.DateOffset(10))
    return df.close[-1]


def get_dividends(ticker, start_date=None, end_date=None, index_as_date=True):
    """Downloads historical dividend data into a pandas data frame.
    @param: ticker
    @param: start_date = None
    @param: end_date = None
    @param: index_as_date = True
    """
    # build and connect to URL
    site, params = build_url(ticker, start_date, end_date, "1d")
    resp = requests.get(site, params=params)
    if not resp.ok:
        raise AssertionError(resp.json())
    # get JSON response
    data = resp.json()
    # check if there is data available for dividends
    if "dividends" not in data["chart"]["result"][0]["events"]:
        raise AssertionError(
            "There is no data available on dividends, or none have been granted"
        )
    # get the dividend data
    frame = pd.DataFrame(data["chart"]["result"][0]["events"]["dividends"])
    frame = frame.transpose()
    frame.index = pd.to_datetime(frame.index, unit="s")
    frame.index = frame.index.map(lambda dt: dt.floor("d"))
    # sort in chronological order
    frame = frame.sort_index()
    frame["ticker"] = ticker.upper()
    # remove old date column
    frame = frame.drop(columns="date")
    frame = frame.rename({"amount": "dividend"}, axis="columns")
    if not index_as_date:
        frame = frame.reset_index()
        frame.rename(columns={"index": "date"}, inplace=True)
    return frame


def get_splits(ticker, start_date=None, end_date=None, index_as_date=True):
    """Downloads historical stock split data into a pandas data frame.
    @param: ticker
    @param: start_date = None
    @param: end_date = None
    @param: index_as_date = True
    """
    # build and connect to URL
    site, params = build_url(ticker, start_date, end_date, "1d")
    resp = requests.get(site, params=params)
    if not resp.ok:
        raise AssertionError(resp.json())
    # get JSON response
    data = resp.json()
    # check if there is data available for splits
    if "splits" not in data["chart"]["result"][0]["events"]:
        raise AssertionError(
            "There is no data available on stock splits, or none have occured"
        )
    # get the split data
    frame = pd.DataFrame(data["chart"]["result"][0]["events"]["splits"])
    frame = frame.transpose()
    frame.index = pd.to_datetime(frame.index, unit="s")
    frame.index = frame.index.map(lambda dt: dt.floor("d"))
    # sort in to chronological order
    frame = frame.sort_index()
    frame["ticker"] = ticker.upper()
    # remove unnecessary columns
    frame = frame.drop(columns=["date", "denominator", "numerator"])
    if not index_as_date:
        frame = frame.reset_index()
        frame.rename(columns={"index": "date"}, inplace=True)
    return frame


def get_earnings(ticker):
    """Scrapes earnings data from Yahoo Finance for an input ticker
    @param: ticker"""
    financials_site = (
        "https://finance.yahoo.com/quote/" + ticker + "/financials?p=" + ticker
    )
    json_info = _parse_json(financials_site)
    temp = json_info["earnings"]
    result = {}
    result["quarterly_results"] = pd.DataFrame.from_dict(
        temp["earningsChart"]["quarterly"]
    )
    result["yearly_revenue_earnings"] = pd.DataFrame.from_dict(
        temp["financialsChart"]["yearly"]
    )
    result["quarterly_revenue_earnings"] = pd.DataFrame.from_dict(
        temp["financialsChart"]["quarterly"]
    )
    return result


def _parse_earnings_json(url):
    resp = requests.get(url)
    content = resp.content.decode(encoding="utf-8", errors="strict")
    page_data = [
        row for row in content.split("\n") if row.startswith("root.App.main = ")
    ][0][:-1]
    page_data = page_data.split("root.App.main = ", 1)[1]
    return json.loads(page_data)


def get_next_earnings_date(ticker):
    base_earnings_url = "https://finance.yahoo.com/quote"
    new_url = base_earnings_url + "/" + ticker
    parsed_result = _parse_earnings_json(new_url)
    temp = parsed_result["context"]["dispatcher"]["stores"]["QuoteSummaryStore"][
        "calendarEvents"
    ]["earnings"]["earningsDate"][0]["raw"]
    return datetime.datetime.fromtimestamp(temp)


def get_earnings_history(ticker):
    """Inputs: @ticker
    Returns the earnings calendar history of the input ticker with
    EPS actual vs. expected data."""
    url = "https://finance.yahoo.com/calendar/earnings?symbol=" + ticker
    result = _parse_earnings_json(url)
    return result["context"]["dispatcher"]["stores"]["ScreenerResultsStore"]["results"][
        "rows"
    ]


def get_earnings_for_date(date, offset=0, count=1):
    """Inputs: @date
    Returns a dictionary of stock tickers with earnings expected on the
    input date.  The dictionary contains the expected EPS values for each
    stock if available."""
    base_earnings_url = "https://finance.yahoo.com/calendar/earnings"
    if offset >= count:
        return []
    temp = pd.Timestamp(date)
    date = temp.strftime("%Y-%m-%d")
    dated_url = "{0}?day={1}&offset={2}&size={3}".format(
        base_earnings_url, date, offset, 100
    )
    result = _parse_earnings_json(dated_url)
    stores = result["context"]["dispatcher"]["stores"]
    earnings_count = stores["ScreenerCriteriaStore"]["meta"]["total"]
    new_offset = offset + 100
    more_earnings = get_earnings_for_date(date, new_offset, earnings_count)
    current_earnings = stores["ScreenerResultsStore"]["results"]["rows"]
    total_earnings = current_earnings + more_earnings
    return total_earnings


def get_earnings_in_date_range(start_date, end_date):
    """Inputs: @start_date
            @end_date
    Returns the stock tickers with expected EPS data for all dates in the
    input range (inclusive of the start_date and end_date."""
    earnings_data = []
    days_diff = pd.Timestamp(end_date) - pd.Timestamp(start_date)
    days_diff = days_diff.days
    current_date = pd.Timestamp(start_date)
    dates = [current_date + datetime.timedelta(diff) for diff in range(days_diff + 1)]
    dates = [d.strftime("%Y-%m-%d") for d in dates]
    i = 0
    while i < len(dates):
        try:
            earnings_data += get_earnings_for_date(dates[i])
        except Exception:
            pass
        i += 1
    return earnings_data


def get_currencies():
    """Returns the currencies table from Yahoo Finance"""
    tables = pd.read_html("https://finance.yahoo.com/currencies")
    result = tables[0]
    return result


def get_futures():
    """Returns the futures table from Yahoo Finance"""
    tables = pd.read_html("https://finance.yahoo.com/commodities")
    result = tables[0]
    return result


def get_quote_data(ticker):
    """Inputs: @ticker
    Returns a dictionary containing over 70 elements corresponding to the
    input ticker, including company name, book value, moving average data,
    pre-market / post-market price (when applicable), and more."""
    site = "https://query1.finance.yahoo.com/v7/finance/quote?symbols=" + ticker
    resp = requests.get(site)
    if not resp.ok:
        raise AssertionError(
            """Invalid response from server.  Check if ticker is valid."""
        )
    json_result = resp.json()
    info = json_result["quoteResponse"]["result"]
    return info[0]


def get_market_status():
    """Returns the current state of the market - PRE, POST, OPEN, or CLOSED"""
    quote_data = get_quote_data("^dji")
    return quote_data["marketState"]


def get_premarket_price(ticker):
    """Inputs: @ticker
    Returns the current pre-market price of the input ticker
    (returns value if pre-market price is available."""
    quote_data = get_quote_data(ticker)
    if "preMarketPrice" in quote_data:
        return quote_data["preMarketPrice"]
    raise AssertionError("Premarket price not currently available.")


def get_postmarket_price(ticker):
    """Inputs: @ticker
    Returns the current post-market price of the input ticker
    (returns value if pre-market price is available."""
    quote_data = get_quote_data(ticker)
    if "postMarketPrice" in quote_data:
        return quote_data["postMarketPrice"]
    raise AssertionError("Postmarket price not currently available.")
