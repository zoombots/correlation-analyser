import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

st.title("Stock and Commodity Correlation Analyzer")

tickers = st.sidebar.text_input("Enter tickers separated by commas", value="AAPL, MSFT, GOOGL, AMZN, TSLA")
tickers = [t.strip().upper() for t in tickers.split(",") if t.strip() != ""]

period_display = st.sidebar.selectbox("Display Period", options=["1 Hour", "1 Day", "1 Month"], index=1)
period_mapping = {"1 Hour": "7d", "1 Day": "1y", "1 Month": "5y"}
interval_mapping = {"1 Hour": "1h", "1 Day": "1d", "1 Month": "1mo"}
period = period_mapping[period_display]
interval = interval_mapping[period_display]

lag_display = st.sidebar.selectbox("Lagging Indicator", options=["None", "1 Minute", "1 Hour", "1 Day"], index=0)
lag_mapping = {"None": None, "1 Minute": 1, "1 Hour": 60, "1 Day": 1440}
lag_minutes = lag_mapping[lag_display]

corr_method = st.sidebar.selectbox("Correlation Method", options=["pearson", "spearman"], index=0)
top_n = st.sidebar.slider("Top N Pairs", 5, 100, 20)

@st.cache_data(ttl=3600)
def load_data(tickers, period, interval):
    df = yf.download(tickers, period=period, interval=interval, group_by='ticker')
    price_data = pd.DataFrame()
    for ticker in tickers:
        if (ticker, 'Adj Close') in df.columns:
            price_data[ticker] = df[(ticker, 'Adj Close')]
        elif 'Adj Close' in df.columns:
            price_data = df['Adj Close']
            break
    return price_data.dropna()

data = load_data(tickers, period, interval)

if lag_minutes is not None:
    lagged_data = data.shift(int(lag_minutes / 5))  # Assuming 5-min bars if interval is "1h"
    combined_data = pd.concat([data, lagged_data.add_suffix("_lag")], axis=1).dropna()
else:
    combined_data = data

corr = combined_data.corr(method=corr_method)

st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
plt.title("Correlation Heatmap")
st.pyplot(fig)

# Drop rows/columns with all NaNs
corr = corr.dropna(how='all').dropna(axis=1, how='all')

# Check if there's anything to plot
if not corr.empty and not corr.isnull().all().all():
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
else:
    st.warning("Correlation matrix is empty or contains only NaNs. Adjust tickers or lag settings.")

