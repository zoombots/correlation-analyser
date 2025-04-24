# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations

st.set_page_config(page_title="Correlation Analyzer", layout="wide")

st.title("📈 Asset Correlation Analyzer")

# --- Sidebar Inputs ---
st.sidebar.header("Settings")

tickers = st.sidebar.text_area(
    "Enter tickers (comma separated)",
    value="AAPL, MSFT, GOOG, AMZN, META, TSLA, NVDA, NFLX, XOM, GLD, SLV, USO, BTC-USD, ETH-USD",
)

tickers = [t.strip().upper() for t in tickers.split(",") if t.strip() != ""]

period = st.sidebar.selectbox("Data Period", options=["1y", "2y", "5y", "max"], index=1)
corr_method = st.sidebar.selectbox("Correlation Method", options=["pearson", "spearman"], index=0)
top_n = st.sidebar.slider("Top N Pairs", 5, 100, 20)

# --- Data Loading ---
@st.cache_data
def load_data(tickers, period):
    df = yf.download(tickers, period=period, group_by='ticker', auto_adjust=True)

    if isinstance(df.columns, pd.MultiIndex):
        adj_close = pd.DataFrame({
            ticker: df[ticker]['Close']
            for ticker in df.columns.levels[0]
            if 'Close' in df[ticker]
        })
    else:
        adj_close = df[['Close']]
        adj_close.columns = [tickers[0]]

    # Drop rows (dates) where most symbols are missing, but not columns (symbols)
    return adj_close.dropna(thresh=len(adj_close.columns) - 2)


    # Drop rows (dates) where most symbols are missing, but not columns (symbols)
    return adj_close.dropna(thresh=len(adj_close.columns) - 2)
    
missing = set(tickers) - set(adj_close.columns)
if missing:
    st.warning(f"Could not load data for: {', '.join(missing)}")

    return adj_close.dropna(axis=1, how="any")

if len(tickers) < 2:
    st.warning("Please enter at least two tickers.")
    st.stop()

st.info("Fetching data...")
data = load_data(tickers, period)
st.success(f"Loaded data for {len(data.columns)} symbols.")

# --- Correlation Matrix ---
returns = data.pct_change().dropna()
corr = returns.corr(method=corr_method)

# --- Top Correlated Pairs ---
def top_corr_pairs(corr_matrix, n=10):
    pairs = []
    for i, j in combinations(corr_matrix.columns, 2):
        value = corr_matrix.loc[i, j]
        pairs.append((i, j, value))
    sorted_pairs = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)
    return sorted_pairs[:n]

top_pairs = top_corr_pairs(corr, top_n)
top_df = pd.DataFrame(top_pairs, columns=["Asset 1", "Asset 2", "Correlation"])
st.subheader(f"🔗 Top {top_n} Correlated Pairs")
st.dataframe(top_df, use_container_width=True)

# --- Heatmap ---
st.subheader("🔥 Correlation Heatmap")
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
st.pyplot(fig)
