import streamlit as st
import pandas as pd

st.set_page_config(page_title="Canada Trade Dashboard", layout="wide")

st.title("🇨🇦 Canada Trade Dashboard")

DATA_PATH = "data_processed/canada_trade_full.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, low_memory=False)

try:
    df = load_data()

    st.success("Dataset loaded successfully")

    st.subheader("Dataset preview")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=["number"]).columns

    if len(numeric_cols) > 0:
        st.subheader("Trade values")
        st.bar_chart(df[numeric_cols[0]].head(100))

except Exception as e:
    st.error("Dataset not available yet.")
    st.write(e)
