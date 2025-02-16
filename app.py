import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm
import datetime

# --------------------------
# Streamlit Page Setup
# --------------------------
st.set_page_config(page_title="Value-at-Risk Analysis", layout="wide")
st.title("Value-at-Risk (VaR) Analysis")

# --------------------------
# Sidebar Parameters
# --------------------------
start_date = st.sidebar.date_input("Start Date", datetime.date(2018, 1, 1))
end_date   = st.sidebar.date_input("End Date", datetime.date(2024, 1, 1))
confidence_level = st.sidebar.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01)
investment = st.sidebar.number_input("Investment Amount (₹)", value=1000000, step=100000)

# Define stress periods
stress_periods = {
    'COVID-19': ('2020-02-01', '2020-04-30'),
    'Adani_Crisis': ('2023-01-01', '2023-02-28')
}

# --------------------------
# User Input: Ticker Symbols
# --------------------------
st.subheader("Enter Ticker Symbols")
ticker_input = st.text_area(
    "Enter ticker symbols separated by commas",
    "NIFTYBEES.NS, AXISNIFTY.NS, SETFNIF50.NS, NPBET.NS, LICNETFN50.NS, SETFNN50.NS",
    height=150
)
tickers = [t.strip() for t in ticker_input.split(",") if t.strip()]

# --------------------------
# Helper Functions
# --------------------------
def calculate_var(returns_series, confidence=0.95):
    """Calculate Historical VaR on the given returns."""
    return np.percentile(returns_series, (1 - confidence) * 100)

def calculate_parametric_var(returns_series, confidence=0.95):
    """Calculate Parametric (Normal) VaR on the given returns."""
    mu = np.mean(returns_series)
    sigma = np.std(returns_series)
    return norm.ppf(1 - confidence, mu, sigma)

def stress_test(returns_series, stress_periods, var_func):
    """
    Slice returns_series by each stress period and compute:
      - VaR
      - VaR in rupees
      - Standard deviation of returns during the period
      - Worst day return
      - Cumulative return over the period
    """
    results = {}
    for event, (start, end) in stress_periods.items():
        period_data = returns_series.loc[start:end]
        if not period_data.empty:
            var_value = var_func(period_data)
            results[event] = {
                'VaR': var_value,
                'VaR ₹': var_value * investment,
                'Returns Std Dev': period_data.std(),
                'Worst Day': period_data.min(),
                'Stress Period Returns': period_data.sum()
            }
    return pd.DataFrame(results).T

def compute_avg_days(dates):
    """
    Compute the average difference in days between consecutive dates in the DatetimeIndex.
    Returns 1 if only one observation is available.
    """
    diffs = np.diff(dates.values).astype('timedelta64[D]').astype(int)
    if len(diffs) == 0:
        return 1
    return np.mean(diffs)

def fetch_full_name(ticker):
    """
    Fetch the ticker's full name using yfinance info.
    If longName isn't available, fallback to shortName or the ticker itself.
    """
    try:
        info = yf.Ticker(ticker).info
        return info.get('longName') or info.get('shortName') or ticker
    except Exception:
        return ticker

# --------------------------
# Main Analysis
# --------------------------
if st.button("Run Analysis"):
    with st.spinner("Fetching data..."):
        # Here we explicitly fetch 'Close' instead of 'Adj Close'.
        global_data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Adj Close']
    
    # Fetch full names for all tickers from yfinance
    full_names = {}
    with st.spinner("Fetching ticker names..."):
        for ticker in tickers:
            full_names[ticker] = fetch_full_name(ticker)
    
    # --------------------------
    # NAV Growth Plot & Per-Ticker Calculations
    # --------------------------
    st.subheader("NAV Growth Comparison")
    fig, ax = plt.subplots(figsize=(12, 6))
    full_period_results = {}
    stress_summary_covid = []
    stress_summary_adani = []
    
    for ticker in tickers:
        if ticker not in global_data.columns:
            continue
        # Use each ticker's available series (drop only its own missing values)
        ticker_series = global_data[ticker].dropna()
        if ticker_series.empty:
            continue
        
        # Plot normalized NAV for this ticker
        normalized = ticker_series / ticker_series.iloc[0]
        ax.plot(ticker_series.index, normalized, label=ticker)
        
        # Compute ticker returns
        ticker_returns = ticker_series.pct_change().dropna()
        if ticker_returns.empty:
            continue
        
        # Adjust annual volatility for irregular frequency
        avg_days = compute_avg_days(ticker_series.index)
        annual_vol = ticker_returns.std() * np.sqrt(252 / avg_days)
        
        # Full period VaR
        hist_var = calculate_var(ticker_returns, confidence_level)
        para_var = calculate_parametric_var(ticker_returns, confidence_level)
        
        full_period_results[ticker] = {
            'Historical VaR': hist_var,
            'Parametric VaR': para_var,
            'Annual Volatility': annual_vol,
            'Historical VaR ₹': hist_var * investment,
            'Parametric VaR ₹': para_var * investment
        }
        
        # Stress Test
        historical_stress = stress_test(ticker_returns, stress_periods, lambda x: calculate_var(x, confidence_level))
        parametric_stress = stress_test(ticker_returns, stress_periods, lambda x: calculate_parametric_var(x, confidence_level))
        
        # COVID-19
        hist_covid = historical_stress.loc['COVID-19'] if 'COVID-19' in historical_stress.index else None
        para_covid = parametric_stress.loc['COVID-19'] if 'COVID-19' in parametric_stress.index else None
        stress_summary_covid.append({
            "Ticker": ticker,
            "ETF Full Name": full_names.get(ticker, ticker),
            "Historical VaR (₹)": f"₹{hist_covid['VaR ₹']:,.2f}" if hist_covid is not None else "NA",
            "Historical Worst Day": f"{hist_covid['Worst Day']:.2%}" if hist_covid is not None else "NA",
            "Historical Period Return": f"{hist_covid['Stress Period Returns']:.2%}" if hist_covid is not None else "NA",
            "Historical Std Dev": f"{hist_covid['Returns Std Dev']:.2%}" if hist_covid is not None else "NA",
            "Parametric VaR (₹)": f"₹{para_covid['VaR ₹']:,.2f}" if para_covid is not None else "NA",
            "Parametric Worst Day": f"{para_covid['Worst Day']:.2%}" if para_covid is not None else "NA",
            "Parametric Period Return": f"{para_covid['Stress Period Returns']:.2%}" if para_covid is not None else "NA",
            "Parametric Std Dev": f"{para_covid['Returns Std Dev']:.2%}" if para_covid is not None else "NA"
        })
        
        # Adani_Crisis
        hist_adani = historical_stress.loc['Adani_Crisis'] if 'Adani_Crisis' in historical_stress.index else None
        para_adani = parametric_stress.loc['Adani_Crisis'] if 'Adani_Crisis' in parametric_stress.index else None
        stress_summary_adani.append({
            "Ticker": ticker,
            "ETF Full Name": full_names.get(ticker, ticker),
            "Historical VaR (₹)": f"₹{hist_adani['VaR ₹']:,.2f}" if hist_adani is not None else "NA",
            "Historical Worst Day": f"{hist_adani['Worst Day']:.2%}" if hist_adani is not None else "NA",
            "Historical Period Return": f"{hist_adani['Stress Period Returns']:.2%}" if hist_adani is not None else "NA",
            "Historical Std Dev": f"{hist_adani['Returns Std Dev']:.2%}" if hist_adani is not None else "NA",
            "Parametric VaR (₹)": f"₹{para_adani['VaR ₹']:,.2f}" if para_adani is not None else "NA",
            "Parametric Worst Day": f"{para_adani['Worst Day']:.2%}" if para_adani is not None else "NA",
            "Parametric Period Return": f"{para_adani['Stress Period Returns']:.2%}" if para_adani is not None else "NA",
            "Parametric Std Dev": f"{para_adani['Returns Std Dev']:.2%}" if para_adani is not None else "NA"
        })
    
    # Finalize NAV Growth Plot
    ax.set_title("NAV Growth Comparison (Using 'Close')")
    ax.set_ylabel("Normalized NAV")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True)
    st.pyplot(fig)
    
    # --------------------------
    # Full Period VaR Table
    # --------------------------
    st.subheader("Full Period VaR Analysis")
    full_period_var_df = pd.DataFrame(full_period_results).T.reset_index().rename(columns={"index": "Ticker"})
    full_period_var_df["ETF Full Name"] = full_period_var_df["Ticker"].map(full_names)
    # Reorder so that Ticker and ETF Full Name are first columns
    cols = full_period_var_df.columns.tolist()
    new_order = ["Ticker", "ETF Full Name"] + [c for c in cols if c not in ["Ticker", "ETF Full Name"]]
    full_period_var_df = full_period_var_df[new_order]
    
    if not full_period_var_df.empty:
        st.dataframe(
            full_period_var_df.style.format({
                'Historical VaR': "{:.4f}",
                'Parametric VaR': "{:.4f}",
                'Annual Volatility': "{:.2%}",
                'Historical VaR ₹': "₹{:,.2f}",
                'Parametric VaR ₹': "₹{:,.2f}"
            }),
            use_container_width=True
        )
    else:
        st.info("No valid data for VaR calculations.")
    
    # --------------------------
    # Stress Test Tables
    # --------------------------
    st.subheader("COVID-19 Stress Period Analysis")
    if stress_summary_covid:
        df_covid = pd.DataFrame(stress_summary_covid)
        cols = df_covid.columns.tolist()
        new_order = ["Ticker", "ETF Full Name"] + [c for c in cols if c not in ["Ticker", "ETF Full Name"]]
        df_covid = df_covid[new_order]
        st.dataframe(df_covid, use_container_width=True)
    else:
        st.write("No valid COVID-19 data.")
    
    st.subheader("Adani Crisis Stress Period Analysis")
    if stress_summary_adani:
        df_adani = pd.DataFrame(stress_summary_adani)
        cols = df_adani.columns.tolist()
        new_order = ["Ticker", "ETF Full Name"] + [c for c in cols if c not in ["Ticker", "ETF Full Name"]]
        df_adani = df_adani[new_order]
        st.dataframe(df_adani, use_container_width=True)
    else:
        st.write("No valid Adani Crisis data.")
