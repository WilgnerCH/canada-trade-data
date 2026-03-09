import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Canada Trade Dashboard",
    layout="wide"
)

st.title("Canada International Trade Dashboard")

DATA_FILE = "data_processed/canada_trade_full.csv.gz"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    return df

df = load_data()

st.write("Dataset size:", len(df))

# Trade type filter
trade_type = st.selectbox(
    "Select trade type",
    df["trade_type"].unique()
)

filtered = df[df["trade_type"] == trade_type]

st.subheader("Monthly Trade Volume")

monthly = filtered.groupby("REF_DATE")["VALUE"].sum().reset_index()

st.line_chart(monthly.set_index("REF_DATE"))

st.subheader("Top Trading Partners")

top_countries = (
    filtered.groupby("COUNTRY")["VALUE"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

st.bar_chart(top_countries)

st.subheader("Raw Data")

st.dataframe(filtered.head(100))
