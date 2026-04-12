import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Canada Trade Intelligence",
    layout="wide"
)

st.markdown("""
<style>
.metric {
    font-size: 20px !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🇨🇦 Canada Trade Intelligence Dashboard")

DATA_URL = "https://huggingface.co/datasets/WilgnerCH/canada-trade-data/resolve/main/canada_trade_full.parquet"

@st.cache_data
def load_data():
    return pd.read_parquet(DATA_URL)

df = load_data()

# =========================
# SIDEBAR FILTERS
# =========================
st.sidebar.header("Filters")

trade_type = st.sidebar.selectbox(
    "Trade Type",
    ["All"] + list(df["trade_type"].unique())
)

countries = st.sidebar.multiselect(
    "Country",
    options=sorted(df["Country"].unique())
)

date_range = st.sidebar.multiselect(
    "Date",
    options=sorted(df["date"].unique()),
    default=sorted(df["date"].unique())[-6:]
)

# =========================
# APPLY FILTERS
# =========================
filtered_df = df.copy()

if trade_type != "All":
    filtered_df = filtered_df[filtered_df["trade_type"] == trade_type]

if countries:
    filtered_df = filtered_df[filtered_df["Country"].isin(countries)]

if date_range:
    filtered_df = filtered_df[filtered_df["date"].isin(date_range)]

# =========================
# KPIs
# =========================
st.subheader("Key Metrics")

total_imports = filtered_df[filtered_df["trade_type"] == "Import"]["Value"].sum()
total_exports = filtered_df[filtered_df["trade_type"] == "Export"]["Value"].sum()
trade_balance = total_exports - total_imports

col1, col2, col3 = st.columns(3)

col1.metric("Total Imports", f"${total_imports:,.0f}")
col2.metric("Total Exports", f"${total_exports:,.0f}")
col3.metric("Trade Balance", f"${trade_balance:,.0f}")

# =========================
# MONTHLY TREND
# =========================
st.subheader("Monthly Trade Trend")

monthly = (
    filtered_df.groupby(["date", "trade_type"])["Value"]
    .sum()
    .reset_index()
)

pivot = monthly.pivot(index="date", columns="trade_type", values="Value").fillna(0)

st.line_chart(pivot)

# =========================
# TOP COUNTRIES
# =========================
st.subheader("Top Trade Partners")

top_countries = (
    filtered_df.groupby("Country")["Value"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

st.bar_chart(top_countries)

# =========================
# TOP PRODUCTS
# =========================
st.subheader("Top Products (HS Codes)")

top_products = (
    filtered_df.groupby("HS")["Value"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

st.bar_chart(top_products)

# =========================
# TABLE
# =========================
st.subheader("Detailed Data")

st.dataframe(filtered_df.head(100), use_container_width=True)
