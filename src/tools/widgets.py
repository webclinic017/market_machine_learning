import streamlit as st
import webbrowser

# - - - - - HOME PAGE - - - - -
def home_widget_analyst_rec(subhead, table1):
    st.subheader(subhead)
    st.table(table1)


def home_widget_instructions(header, key):
    st.write(header)
    st.write(key)


def home_widget_tools(subhead, category_1, key_1):
    st.subheader(f"** > {subhead} **")
    st.write(f"** {category_1} ** ")
    st.write(key_1)


# - - - - - SNAPSHOT PAGE - - - - -
def snapshot_widget(head, key):
    st.subheader(head)
    st.markdown(key)


def snapshot_widget_screener(key):
    st.dataframe(key.set_index("ranking"))


def snapshot_widget_index(keya, keyb):
    for r in range(len(keya)):
        st.write(f"{keya[r]} {keyb[r]}")
    st.write("__" * 25)


def widget_header(web1, subhead, key):
    st.subheader(f"**>Online Resources:**")
    st.write(f"{web1}")
    st.subheader(f"**{subhead}:**")
    st.write(f"{key}")


def my_widget_overview(header, key):
    st.subheader(header)
    st.write(key)


def widget_online_resource(key):
    st.subheader("**Online Resources:**")
    st.write(f" - {key}")


def widget_basic(head, subhead, key):
    st.header(f"**{head}**")
    st.subheader(f"**{subhead}:**")
    st.write(f" - {key}")


def my_widget_financial_disclosure(header, key):
    st.subheader(header)
    st.write(key)


def widget_prophet(key_1, key_2, web_address):
    st.subheader("**Details:**")
    st.write(f"{key_1}")
    st.write(f"{key_2}")
    st.subheader("**Online Resources:**")
    st.write(f" - {web_address}")
    if st.button("Open Prophet Model Web Page"):
        webbrowser.open_new_tab(web_address)


def widget_analysis(subhead, key):
    st.subheader(f"**{subhead}:**")
    st.write(f"{key}")


def widget_univariate(title, key):
    st.header(f"**{title}**")
    st.write(f"{key}")


# - - - - - XXXXX PAGE - - - - -
