import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

st.set_page_config(layout="wide")

st.title("Lagged Correlation Heatmap")

# Sidebar controls
tickers_input = st.sidebar.text_input("Enter comma-separated tickers", value="AAPL,MSFT,GOOG")
period = st.sidebar.selectbox("Data Period", options=["1d", "1mo", "1y"], index=0)
interval = st.sidebar.selectbox("Data Interval", options=["1m", "1h", "1d"], index=2)
lag_option = st.sidebar.selectbox("Lag Amount", options=["1min", "1h", "1d"], index=1)
corr_method = st.sidebar.selectbox("Correlation Method", options=["pearson", "spearman"], index=0)
top_n = st.sidebar.slider("Top N Correlation Pairs", 5, 100, 20)

# Parse tickers
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

@st.cache_data

def load_data(tickers, period, interval):
    df = yf.download(tickers, period=period, interval=interval, group_by="ticker", auto_adjust=True)
    if len(tickers) == 1:
        df = df[["Close"]].rename(columns={"Close": tickers[0]})
    else:
        df = pd.concat([df[ticker]["Close"].rename(ticker) for ticker in tickers], axis=1)
    return df.dropna()

# Load data
data = load_data(tickers, period, interval)

# Apply lag
lag_map = {"1min": 1, "1h": 2, "1d": 3}  # Customize as needed
lag_steps = lag_map[lag_option]

lagged_data = pd.DataFrame()
for i, ticker in enumerate(data.columns):
    lagged_data[ticker] = data[ticker].shift(i * lag_steps)

lagged_data = lagged_data.dropna()

# Calculate correlation matrix
corr = lagged_data.corr(method=corr_method)

# Display heatmap
st.subheader("Lagged Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Display top correlation pairs
st.subheader("Top Correlation Pairs")
pairs = list(itertools.combinations(corr.columns, 2))
pair_scores = [(i, j, corr.loc[i, j]) for i, j in pairs if not np.isnan(corr.loc[i, j])]
pair_scores.sort(key=lambda x: abs(x[2]), reverse=True)
top_pairs_df = pd.DataFrame(pair_scores[:top_n], columns=["Ticker 1", "Ticker 2", "Correlation"])
st.dataframe(top_pairs_df)
