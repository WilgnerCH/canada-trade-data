import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Canada Trade Dashboard", layout="wide")

st.title("🇨🇦 Canada Trade Dashboard")

DATA_PATH = "data_processed/canada_trade_full.csv"

# Verificar se dataset existe
if not os.path.exists(DATA_PATH):
    st.error("Dataset not found. Run the pipeline first.")
    st.stop()

# Carregar dados
df = pd.read_csv(DATA_PATH)

st.success("Dataset loaded successfully!")

# Mostrar dados
st.subheader("Dataset preview")
st.dataframe(df.head())

# Verificar colunas básicas
st.subheader("Columns in dataset")
st.write(df.columns.tolist())

# Se tiver coluna de valor, criar gráfico
value_cols = [c for c in df.columns if "VALUE" in c.upper()]

if value_cols:
    value_col = value_cols[0]

    st.subheader("Trade values distribution")

    chart_data = df[value_col].dropna()

    st.bar_chart(chart_data.head(100))

else:
    st.warning("No trade value column detected.")
