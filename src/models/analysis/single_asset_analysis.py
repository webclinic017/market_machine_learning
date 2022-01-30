import pandas as pd
import finviz
from finvizfinance.quote import finvizfinance
from finvizfinance.news import News
from finviz import get_analyst_price_targets, get_news, get_stock
from IPython.display import Image
import streamlit as st



class Single_Asset_Analysis(object):

    def __init__(self):
        self.ticker = st.sidebar.text_input(label='Enter Stock')


    def run(self):

        if st.sidebar.button('Review'):
            stock = finvizfinance(self.ticker)

            stock_description = stock.TickerDescription()
            outer_ratings_df = stock.TickerOuterRatings()
            news_df = stock.TickerNews()
            fnews = News()
            all_news = fnews.getNews()
            inside_trader_df = stock.TickerInsideTrader()
            signals = stock.TickerSignal

            st.title(self.ticker)
            st.write('__'*25)

            st.header('Description')
            st.markdown(stock_description)
            st.write('__'*25)

            st.header('Stock Information')
            st.dataframe(pd.DataFrame.from_dict(get_stock('AAPL'), orient='index'))
            st.write('__'*25)
            
            st.header('Stock Chart')
            st.image(stock.TickerCharts())
            st.write('__'*25)

            st.header('Analyst Ratings')
            st.dataframe(outer_ratings_df.head().set_index('Date'))
            st.write('__'*25)

            st.header('Stock News')
            x = pd.DataFrame(get_news(self.ticker))
            x.columns=['date','headline','link','source']
            st.dataframe(x.set_index('date')    )
            st.write('__'*25)

            st.header('Stock Insider Trading')
            st.dataframe(inside_trader_df.head())
            st.dataframe(pd.DataFrame.from_records(finviz.get_insider(self.ticker)))
            st.write('__'*25)

            st.header('Stock Signals')
            st.dataframe(pd.DataFrame(get_analyst_price_targets(self.ticker)).set_index('date'))
            st.write('__'*25)
