import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Canada Trade Dashboard",
    layout="wide"
)

st.title("🇨🇦 Canada Trade Dashboard")

DATA_URL = "https://huggingface.co/datasets/WilgnerCH/canada-trade-data/resolve/main/canada_trade_full.csv.gz"


@st.cache_data
def load_data():
    return pd.read_csv(
        DATA_URL,
        compression="gzip",
        low_memory=False
    )


df = load_data()

st.success("Dataset loaded successfully")


# Overview
st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Rows", f"{len(df):,}")
col2.metric("Columns", len(df.columns))
col3.metric("Trade Types", df["trade_type"].nunique())


# Preview
st.subheader("Dataset Preview")

st.dataframe(df.head(20), use_container_width=True)


# Charts
numeric_cols = df.select_dtypes(include=["number"]).columns

if len(numeric_cols) > 0:

    st.subheader("Trade values (sample)")

    st.bar_chart(df[numeric_cols[0]].head(100))


if "trade_type" in df.columns:

    st.subheader("Import vs Export")

    trade_counts = df["trade_type"].value_counts()

    st.bar_chart(trade_counts)
