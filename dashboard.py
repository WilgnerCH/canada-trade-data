import streamlit as st
import pandas as pd
import os
import subprocess

st.set_page_config(page_title="Canada Trade Dashboard", layout="wide")

st.title("🇨🇦 Canada Trade Dashboard")

DATA_PATH = "data_processed/canada_trade_full.csv"

# Se dataset não existir, gerar automaticamente
if not os.path.exists(DATA_PATH):

    st.warning("Dataset not found. Generating dataset...")

    subprocess.run(["python", "src/download_data.py"])
    subprocess.run(["python", "src/process_data.py"])

# carregar dataset
if os.path.exists(DATA_PATH):

    df = pd.read_csv(DATA_PATH)

    st.success("Dataset loaded!")

    st.subheader("Dataset preview")
    st.dataframe(df.head())

    st.subheader("Columns")
    st.write(df.columns.tolist())

else:

    st.error("Dataset generation failed.")
