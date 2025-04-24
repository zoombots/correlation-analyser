# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations



st.title("Top Correlated Stocks & Commodities")

# Input for ticker symbols
tickers = st.text_area("Enter ticker symbols (comma-separated)")
tickers = [t.strip().upper() for t in tickers.split(",") if t.strip() != ""]

# Sidebar options for timeframe, lag, correlation method, and top N pairs
timeframe = st.sidebar.selectbox("Timeframe", options=["1 Hour", "1 Day", "1 Month"], index=1)
lag = st.sidebar.selectbox("Lagging Indicator", options=["1 Minute", "1 Hour", "1 Day"], index=1)

if timeframe == "1 Hour":
    period = "7d"
    interval = "1h"
elif timeframe == "1 Day":
    period = "1mo"
    interval = "1d"
elif timeframe == "1 Month":
    period = "1y"
    interval = "1mo"

corr_method = st.sidebar.selectbox("Correlation Method", options=["pearson", "spearman"], index=0)
top_n = st.sidebar.slider("Top N Pairs", 5, 100, 20)

@st.cache_data

def load_data(tickers, period, interval):
    if not tickers:
        return pd.DataFrame()
    data = yf.download(tickers, period=period, interval=interval)
    return data["Adj Close"] if "Adj Close" in data.columns else data

data = load_data(tickers, period, interval)

# Apply lag based on selection
lag_map = {
    "1 Minute": 1,
    "1 Hour": 1,
    "1 Day": 1
}

if interval == "1h" and lag == "1 Minute":
    st.warning("1 Minute lag not applicable for 1 Hour interval data")
elif interval == "1d" and lag == "1 Minute":
    st.warning("1 Minute lag not applicable for 1 Day interval data")
elif interval == "1mo" and lag in ["1 Minute", "1 Hour"]:
    st.warning("Lag selection not applicable for 1 Month interval data")
else:
    lag_shift = lag_map.get(lag, 1)
    if not data.empty:
        shifted_data = data.shift(lag_shift)
        corr = data.corrwith(shifted_data, axis=0, method=corr_method)
        pairs = list(itertools.combinations(corr.index, 2))
        corr_pairs = [(a, b, data[a].corr(shifted_data[b], method=corr_method)) for a, b in pairs if a in shifted_data and b in shifted_data]
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        top_corrs = corr_pairs[:top_n]
        st.subheader(f"Top {top_n} Correlated Pairs with Lag ({lag})")
        for a, b, c in top_corrs:
            st.write(f"{a} - {b}: {c:.2f}")
    else:
        st.warning("Please enter valid tickers and ensure data is available.")

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
