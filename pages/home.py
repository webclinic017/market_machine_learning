import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from datetime import datetime

from src.tools import functions as f0
from src.tools import lists as l0
from src.tools import scripts as s0
from src.tools import widgets as w0

plt.style.use("ggplot")
sm, med, lg = "20", "25", "30"
plt.rcParams["font.size"] = sm  # controls default text sizes
plt.rc("axes", titlesize=med)  # fontsize of the axes title
plt.rc("axes", labelsize=med)  # fontsize of the x & y labels
plt.rc("xtick", labelsize=sm)  # fontsize of the tick labels
plt.rc("ytick", labelsize=sm)  # fontsize of the tick labels
plt.rc("legend", fontsize=sm)  # legend fontsize
plt.rc("figure", titlesize=lg)  # fontsize of the figure title
plt.rc("axes", linewidth=2)  # linewidth of plot lines
plt.rcParams["figure.figsize"] = [17, 12]
plt.rcParams["figure.dpi"] = 134
plt.rcParams["axes.facecolor"] = "silver"


class Home(object):
    def __init__(self):
        # st.markdown("<div id='linkto_top'></div>", unsafe_allow_html=True)
        self.today_stamp = str(datetime.now())[:10]
        self.stage_lst = l0.general_pages
        self.major_indicies = l0.major_indicies
        self.major_index_names = l0.major_index_names

        st.title(" · Welcome To The Investment App 4m · ")
        st.write(f"{'_'*25} \n {'_'*25}")

    def run_home(self):
        st.header("** · How To Use The App· **")
        with st.expander("", expanded=False):
            st.write(s0.navGuide_a)
            st.write(s0.navGuide_b)

        st.header("** · Profile · **")
        with st.expander("", expanded=False):

            st.subheader("** > Synopsis **")
            st.write(s0.overview_home)

            stage_lst_df = pd.DataFrame(self.stage_lst, columns=["App Sections"])
            stage_lst_df["page_number"] = range(1, len(self.stage_lst) + 1)
            clicked = w0.home_widget_analyst_rec(
                "** > Application · Stages **", stage_lst_df.set_index("page_number")
            )

            st.subheader("** > Summary **")
            st.write(s0.instructions_home)

        st.header("** · About · **")
        with st.expander("", expanded=False):

            st.subheader("** > About The Author: **")
            st.write(" * Creator: Gordon D. Pisciotta ")

            st.subheader("** > Disclosures **")
            st.write(f" * {s0.financial_disclosure}")

        # st.markdown("<a href='#linkto_top'>Link to top</a>", unsafe_allow_html=True)
