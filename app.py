# ==============================================================================
# V9 - PRACTICAL APP
# ==============================================================================
# This version provides a tabbed interface to separate the slow backtest from
# the fast, actionable live portfolio generation.
# VERSION 9.1: Added robust error handling to prevent crashes.
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import backend  # Import our final backend logic

# --- Page Configuration ---
st.set_page_config(
    page_title="Quantitative Investment Assistant",
    page_icon="�",
    layout="wide"
)

# --- App State Initialization ---
if 'live_portfolio' not in st.session_state:
    st.session_state['live_portfolio'] = None
if 'backtest_results' not in st.session_state:
    st.session_state['backtest_results'] = None

# --- Helper function for backtest analysis ---
def analyze_backtest_results(results_df, prices):
    st.header("Backtest Performance Analysis")
    portfolio_monthly_returns = []
    hist_ret_for_cov = prices.resample('M').last().pct_change()
    for date in results_df['PredictionDate'].unique():
        month_preds = results_df[results_df['PredictionDate'] == date].set_index('Ticker')
        watchlist = month_preds[month_preds['P50'] > 0.005].sort_values('P50', ascending=False).head(15)
        if len(watchlist) > 1:
            expected_returns = watchlist['P50']
            cov_end_date = date - pd.DateOffset(months=1)
            cov_start_date = cov_end_date - pd.DateOffset(months=12)
            historical_cov_data = hist_ret_for_cov.loc[cov_start_date:cov_end_date][watchlist.index]
            if historical_cov_data.shape[0] > 10 and not historical_cov_data.isnull().values.any():
                optimal_weights = backend.optimize_portfolio_weights(expected_returns, historical_cov_data)
                portfolio_return = np.dot(optimal_weights, watchlist['ActualReturn'])
            else:
                portfolio_return = 0
        else:
            portfolio_return = 0
        portfolio_monthly_returns.append(portfolio_return)
    portfolio_df = pd.DataFrame({'Date': pd.to_datetime(results_df['PredictionDate'].unique()), 'Model Portfolio': portfolio_monthly_returns}).set_index('Date')
    spy_monthly_returns = prices['SPY'].resample('M').last().pct_change()
    portfolio_df['SPY Benchmark'] = spy_monthly_returns.reindex(portfolio_df.index)
    portfolio_df.dropna(inplace=True)
    model_sharpe = (portfolio_df['Model Portfolio'].mean() * 12) / (portfolio_df['Model Portfolio'].std() * np.sqrt(12))
    spy_sharpe = (portfolio_df['SPY Benchmark'].mean() * 12) / (portfolio_df['SPY Benchmark'].std() * np.sqrt(12))
    col1, col2 = st.columns(2)
    col1.metric("Model Sharpe Ratio", f"{model_sharpe:.2f}")
    col2.metric("SPY Sharpe Ratio", f"{spy_sharpe:.2f}")
    st.subheader("Cumulative Returns (Log Scale)")
    st.line_chart((1 + portfolio_df).cumprod())
    st.subheader("Raw Backtest Data")
    st.dataframe(results_df)

# --- Main Page Content ---
st.sidebar.title("📈 Investment Assistant")
st.sidebar.markdown("---")
st.title("Quantitative Model Dashboard")

tab1, tab2 = st.tabs(["Live Portfolio", "Strategy Backtest"])

# --- Live Portfolio Tab ---
with tab1:
    st.header("Actionable Portfolio for Next Month")
    st.markdown("Click the button to generate a portfolio with exact weights. Use this to update your Trading 212 Pie for the upcoming month.")
    
    if st.button("Generate Live Portfolio", type="primary"):
        with st.spinner("Running live prediction pipeline..."):
            try:
                live_portfolio = backend.run_live_prediction_pipeline(st)
                st.session_state['live_portfolio'] = live_portfolio
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.session_state['live_portfolio'] = None

    if st.session_state['live_portfolio'] is not None:
        portfolio = st.session_state['live_portfolio']
        if not portfolio.empty:
            st.subheader("Target Portfolio Allocation")
            st.dataframe(
                portfolio.style.format({'Weight': '{:.2%}'}),
                use_container_width=True
            )
            st.success("✅ Your portfolio is ready. Rebalance your Pie to match these target weights.")
        else:
            st.warning("⚠️ The model recommends holding cash this month (no stocks met the criteria).")

# --- Strategy Backtest Tab ---
with tab2:
    st.header("Full Historical Strategy Backtest")
    st.warning("Note: This is a slow, computationally intensive process that validates the strategy over many years.")

    if st.button("Run Full Backtest"):
        with st.spinner("Running full backtest pipeline... This will take several minutes."):
            try:
                results, prices = backend.run_backtest_pipeline(st)
                st.session_state['backtest_results'] = (results, prices)
            except Exception as e:
                st.error(f"An error occurred during the backtest: {e}")
                st.session_state['backtest_results'] = None
    
    if st.session_state['backtest_results'] is not None:
        results, prices = st.session_state['backtest_results']
        if results is not None and not results.empty:
            analyze_backtest_results(results, prices)
        else:
            st.error("The backtest ran but did not produce any results.")
