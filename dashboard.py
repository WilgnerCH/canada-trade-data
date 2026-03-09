import streamlit as st
import pandas as pd

st.set_page_config(page_title="Canada Trade Dashboard", layout="wide")

st.title("🇨🇦 Canada Trade Dashboard")

DATA_URL = "https://raw.githubusercontent.com/WilgnerCH/canada-trade-data/main/data_processed/canada_trade_full.csv"

try:
    df = pd.read_csv(DATA_URL)

    st.success("Dataset loaded successfully!")

    st.subheader("Dataset preview")
    st.dataframe(df.head())

    st.subheader("Columns")
    st.write(df.columns.tolist())

    numeric_cols = df.select_dtypes(include=["number"]).columns

    if len(numeric_cols) > 0:
        st.subheader("Trade values chart")
        st.bar_chart(df[numeric_cols[0]].head(100))

except Exception as e:
    st.error("Failed to load dataset")
    st.write(e)
