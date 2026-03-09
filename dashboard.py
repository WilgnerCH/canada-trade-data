import streamlit as st
import pandas as pd
import zipfile
import requests
from io import BytesIO

st.set_page_config(page_title="Canada Trade Dashboard", layout="wide")

st.title("🇨🇦 Canada Trade Dashboard")

FILES = {
    "Import 2023": "https://www150.statcan.gc.ca/n1/pub/71-607-x/2021004/zip/CIMT-CICM_Imp_2023.zip",
    "Import 2024": "https://www150.statcan.gc.ca/n1/pub/71-607-x/2021004/zip/CIMT-CICM_Imp_2024.zip",
    "Import 2025": "https://www150.statcan.gc.ca/n1/pub/71-607-x/2021004/zip/CIMT-CICM_Imp_2025.zip",
    "Export 2023": "https://www150.statcan.gc.ca/n1/pub/71-607-x/2021004/zip/CIMT-CICM_Tot_Exp_2023.zip",
    "Export 2024": "https://www150.statcan.gc.ca/n1/pub/71-607-x/2021004/zip/CIMT-CICM_Tot_Exp_2024.zip",
    "Export 2025": "https://www150.statcan.gc.ca/n1/pub/71-607-x/2021004/zip/CIMT-CICM_Tot_Exp_2025.zip",
}


@st.cache_data
def load_data():

    dfs = []

    for name, url in FILES.items():

        st.write(f"Downloading {name}...")

        r = requests.get(url)

        z = zipfile.ZipFile(BytesIO(r.content))

        for file in z.namelist():

            if "ODPFN015" in file and file.endswith(".csv"):

                df = pd.read_csv(z.open(file))

                df["dataset"] = name

                dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


df = load_data()

st.success("Dataset loaded!")

st.subheader("Preview")
st.dataframe(df.head())

st.subheader("Columns")
st.write(df.columns.tolist())

numeric_cols = df.select_dtypes(include=["number"]).columns

if len(numeric_cols) > 0:
    st.subheader("Trade values")
    st.bar_chart(df[numeric_cols[0]].head(100))
