import streamlit as st
import yfinance as yf
import pandas as pd
import itertools

# Sidebar user inputs
tickers = st.sidebar.text_input("Enter comma-separated tickers (e.g. AAPL, MSFT, GOOGL)")
tickers = [t.strip().upper() for t in tickers.split(",") if t.strip() != ""]

period = st.sidebar.selectbox("Data Period", options=["1h", "1d", "1mo"], index=1)
corr_method = st.sidebar.selectbox("Correlation Method", options=["pearson", "spearman"], index=0)
top_n = st.sidebar.slider("Top N Pairs", 5, 100, 20)

lag_option = st.sidebar.selectbox("Lagging Indicator", options=["None", "1 minute", "1 hour", "1 day"], index=0)
lag_mapping = {"None": 0, "1 minute": 1, "1 hour": 60, "1 day": 1440}  # minutes
lag_minutes = lag_mapping[lag_option]

@st.cache_data

def load_data(tickers, interval):
    data = yf.download(tickers, interval=interval, period="7d", group_by='ticker', auto_adjust=True)
    if len(tickers) == 1:
        data = data['Close'].to_frame(name=tickers[0])
    else:
        data = pd.concat([data[ticker]['Close'].rename(ticker) for ticker in tickers if ticker in data.columns.get_level_values(0)], axis=1)
    return data.dropna()

# Determine appropriate interval based on period
interval = {
    "1h": "1m",
    "1d": "5m",
    "1mo": "1h"
}[period]

data = load_data(tickers, interval)

# Ensure data is not empty
if data.empty:
    st.error("No data loaded. Please check ticker symbols and try again.")
else:
    # Apply lag shift
    if lag_minutes > 0:
        lag_steps = lag_minutes // {
            "1m": 1,
            "5m": 5,
            "1h": 60
        }[interval]
        shifted_data = data.shift(lag_steps)
        corr = data.corrwith(shifted_data, method=corr_method).to_frame(name="Correlation")
    else:
        corr = data.corr(method=corr_method)

    if isinstance(corr, pd.DataFrame):
        try:
            pairs = list(itertools.combinations(corr.index, 2))
            correlations = [
                (a, b, corr.loc[a, b]) for a, b in pairs if a in corr.columns and b in corr.columns
            ]
            top_corrs = sorted(correlations, key=lambda x: abs(x[2]), reverse=True)[:top_n]
            df_top = pd.DataFrame(top_corrs, columns=["Ticker 1", "Ticker 2", "Correlation"])
            st.dataframe(df_top)
        except Exception as e:
            st.error(f"Failed to compute correlations: {e}")
    else:
        st.error("Correlation matrix not generated properly. Please check your data and settings.")
