# ==============================================================================
# V8 - FINAL INTEGRATED APP
# ==============================================================================
# This Streamlit app now runs the full backtest with the enhanced backend
# and displays the final performance metrics and charts.
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import backend  # Import our final backend logic
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(
    page_title="Quantitative Investment Assistant",
    page_icon="📈",
    layout="wide"
)

# --- App State Management ---
if 'backtest_results' not in st.session_state:
    st.session_state['backtest_results'] = None
if 'model_run_completed' not in st.session_state:
    st.session_state['model_run_completed'] = False

# --- Helper function for analysis ---
def analyze_backtest_results(results_df, prices):
    """
    Takes the raw backtest results and calculates portfolio returns
    and performance metrics.
    """
    st.header("Backtest Performance Analysis")
    
    # --- Portfolio Simulation with Optimizer ---
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
                realized_returns = watchlist['ActualReturn']
                portfolio_return = np.dot(optimal_weights, realized_returns)
            else:
                portfolio_return = 0
        else:
            portfolio_return = 0
        portfolio_monthly_returns.append(portfolio_return)

    portfolio_df = pd.DataFrame({
        'Date': pd.to_datetime(results_df['PredictionDate'].unique()),
        'Model Portfolio': portfolio_monthly_returns
    }).set_index('Date')

    spy_monthly_returns = prices['SPY'].resample('M').last().pct_change()
    portfolio_df['SPY Benchmark'] = spy_monthly_returns.reindex(portfolio_df.index)
    portfolio_df.dropna(inplace=True)

    # --- Metrics ---
    model_sharpe = (portfolio_df['Model Portfolio'].mean() * 12) / (portfolio_df['Model Portfolio'].std() * np.sqrt(12))
    spy_sharpe = (portfolio_df['SPY Benchmark'].mean() * 12) / (portfolio_df['SPY Benchmark'].std() * np.sqrt(12))

    col1, col2 = st.columns(2)
    col1.metric("Model Sharpe Ratio", f"{model_sharpe:.2f}")
    col2.metric("SPY Sharpe Ratio", f"{spy_sharpe:.2f}")

    # --- Chart ---
    st.subheader("Cumulative Returns (Log Scale)")
    st.line_chart((1 + portfolio_df).cumprod())
    
    return portfolio_df


# --- Sidebar ---
with st.sidebar:
    st.title("📈 Investment Assistant")
    st.markdown("---")
    st.header("Actions")
    if st.button("Run Full Backtest", type="primary", use_container_width=True):
        with st.status("Running full backtest pipeline...", expanded=True) as status:
            try:
                results, prices = backend.run_backtest_pipeline(status)
                st.session_state['backtest_results'] = (results, prices)
                st.session_state['model_run_completed'] = True
                status.update(label="Backtest executed successfully!", state="complete", expanded=False)
            except Exception as e:
                status.update(label=f"An error occurred: {e}", state="error", expanded=True)

    st.markdown("---")
    st.info("This dashboard runs a full walk-forward backtest of the quantitative strategy, including the enhanced Sector Momentum features.")

# --- Main Page Content ---
st.title("Quantitative Model Dashboard")

if not st.session_state['model_run_completed']:
    st.info("👈 Please click 'Run Full Backtest' in the sidebar to begin. This will take several minutes.")
else:
    results, prices = st.session_state['backtest_results']
    if results is not None and not results.empty:
        # Analyze and display the results
        portfolio_df = analyze_backtest_results(results, prices)
        
        st.subheader("Raw Backtest Data")
        st.dataframe(results)
    else:
        st.error("The backtest did not produce any results.")
